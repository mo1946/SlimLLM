"""HTTP protocol fixtures only, not vLLM model inference or performance results."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import threading
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
import httpx

from slimllm.serving.api import create_app
from slimllm.serving.inference import InputTooLong, InvalidModelOutput
from slimllm.serving.response_cache import RedisExactCache
from slimllm.runtime import ROOT, read_jsonl
from slimllm.schema import AfterSalesTicket
from slimllm.serving.vllm_backend import BackendRequestRejected, BackendTimeout, BackendUnavailable, VllmExtractor

EXAMPLE = read_jsonl(ROOT/'data/examples.jsonl')[0]


class TokenizerFixture:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['enable_thinking'] is False
        self.messages = messages
        self.template_options = kwargs
        return messages[-1]['content']

    def encode(self, text, **kwargs):
        self.encode_options = kwargs
        return list(range(len(text)))


class RemoteEngineFixture:
    fingerprint = 'remote-unit-fixture'
    structured_output_mode = 'json_schema'
    structured_schema_sha256 = 'remote-schema-unit-fixture'
    def __init__(self):
        self.ready = False
        self.calls = 0
        self.lock = threading.Lock()
        self.barrier = threading.Barrier(2)

    def load(self): self.ready = True
    def close(self): self.ready = False
    def check_health(self): return self.ready

    def extract_with_metrics(self, text):
        with self.lock: self.calls += 1
        self.barrier.wait(timeout=3)
        return AfterSalesTicket.model_validate(EXAMPLE['label']), {'generated_tokens':10}


class VllmTests(unittest.TestCase):
    def engine(self, handler, *, structured_outputs='json_schema'):
        engine = VllmExtractor(structured_outputs=structured_outputs)
        engine.tokenizer = TokenizerFixture()
        engine.client = httpx.Client(base_url='http://127.0.0.1:8001', transport=httpx.MockTransport(handler))
        engine.ready = True
        self.addCleanup(engine.close)
        return engine

    def loaded_engine(self, *, structured_outputs='json_schema', version='0.11.2', alias=True):
        """Exercise startup using protocol fixtures; never load model weights."""
        def handler(request):
            if request.url.path == '/v1/models':
                return httpx.Response(200, json={'data': [{'id': 'slimllm-lora'}] if alias else []})
            if request.url.path == '/version':
                return httpx.Response(200, json={'version': version})
            if request.url.path == '/health':
                return httpx.Response(200)
            self.fail(f'Unexpected startup request: {request.url.path}')

        engine = VllmExtractor(structured_outputs=structured_outputs)
        self.addCleanup(engine.close)
        client = httpx.Client(base_url='http://127.0.0.1:8001', transport=httpx.MockTransport(handler))
        with patch.dict('os.environ'), \
             patch('transformers.AutoTokenizer.from_pretrained', return_value=TokenizerFixture()), \
             patch('slimllm.serving.vllm_backend.httpx.Client', return_value=client):
            engine.load()
        return engine

    def test_protocol_uses_frozen_prompt_tokens_and_lora_alias(self):
        def handler(request):
            body = json.loads(request.content)
            self.assertEqual(request.method, 'POST')
            self.assertEqual(request.url.path, '/v1/completions')
            self.assertEqual(set(body), {'model', 'prompt', 'max_tokens', 'temperature',
                                         'seed', 'n', 'stream', 'structured_outputs'})
            self.assertEqual(body['model'], 'slimllm-lora')
            self.assertEqual(body['prompt'], list(range(len(EXAMPLE['text']))))
            self.assertEqual(body['temperature'], 0)
            self.assertEqual(body['max_tokens'], 384)
            self.assertEqual(body['seed'], 42)
            self.assertEqual(body['n'], 1)
            self.assertIs(body['stream'], False)
            schema = body['structured_outputs']['json']
            self.assertEqual(schema, AfterSalesTicket.model_json_schema())
            self.assertEqual(list(schema['properties']), list(AfterSalesTicket.model_fields))
            return httpx.Response(200, json={'choices':[{'text':json.dumps(EXAMPLE['label']), 'finish_reason':'stop'}], 'usage':{'completion_tokens':31}})
        engine = self.engine(handler)
        result, metrics = engine.extract_with_metrics(EXAMPLE['text'])
        self.assertEqual(result.model_dump(), EXAMPLE['label'])
        self.assertEqual(metrics['generated_tokens'], 31)
        self.assertNotIn('generation_seconds', metrics)
        self.assertNotIn('peak_allocated_bytes', metrics)
        self.assertEqual(engine.calls, 1)
        frozen_prompt = json.loads((ROOT/'reports/baseline/run.json').read_text(encoding='utf-8'))['prompt']
        self.assertEqual(engine.tokenizer.messages, [
            {'role': 'system', 'content': frozen_prompt}, {'role': 'user', 'content': EXAMPLE['text']}])
        self.assertEqual(engine.tokenizer.template_options,
                         {'tokenize': False, 'add_generation_prompt': True, 'enable_thinking': False})
        self.assertEqual(engine.tokenizer.encode_options, {'add_special_tokens': False})

    def test_off_mode_omits_constraints_and_keeps_output_validation(self):
        def handler(request):
            body = json.loads(request.content)
            self.assertEqual(set(body), {'model', 'prompt', 'max_tokens', 'temperature', 'seed', 'n', 'stream'})
            return httpx.Response(200, json={
                'choices': [{'text': json.dumps(EXAMPLE['label'] | {'requested_action': '安排寄修'}),
                             'finish_reason': 'stop'}], 'usage': {'completion_tokens': 31}})
        engine = self.engine(handler, structured_outputs='off')
        self.assertEqual(engine.structured_output_mode, 'off')
        self.assertIsNone(engine.structured_schema_sha256)
        with self.assertRaises(InvalidModelOutput):
            engine.extract_with_metrics(EXAMPLE['text'])
        self.assertEqual(engine.calls, 1)

    def test_mode_default_environment_override_and_invalid_values(self):
        with patch.dict('os.environ'):
            os.environ.pop('SLIMLLM_VLLM_STRUCTURED_OUTPUTS', None)
            self.assertEqual(VllmExtractor().structured_output_mode, 'json_schema')
            os.environ['SLIMLLM_VLLM_STRUCTURED_OUTPUTS'] = 'off'
            self.assertEqual(VllmExtractor().structured_output_mode, 'off')
            self.assertEqual(VllmExtractor(structured_outputs='json_schema').structured_output_mode, 'json_schema')
            for value in ('', 'auto', 'JSON_SCHEMA', '0'):
                os.environ['SLIMLLM_VLLM_STRUCTURED_OUTPUTS'] = value
                with self.subTest(value=value), self.assertRaises(ValueError):
                    VllmExtractor()
                self.assertEqual(VllmExtractor(structured_outputs='off').structured_output_mode, 'off')
        for value in ('', 'auto', 1, False):
            with self.subTest(constructor=value), self.assertRaises(ValueError):
                VllmExtractor(structured_outputs=value)

    def test_request_builder_has_no_http_count_or_shared_mutable_payload(self):
        def handler(request):
            self.fail('Building a request must not send HTTP')
        engine = self.engine(handler)
        first = engine.build_request(EXAMPLE['text'])
        expected_schema = AfterSalesTicket.model_json_schema()
        first['prompt'].append(999)
        first['structured_outputs']['json']['properties']['priority']['enum'].append('illegal')
        second = engine.build_request(EXAMPLE['text'])
        self.assertEqual(engine.calls, 0)
        self.assertEqual(second['prompt'], list(range(len(EXAMPLE['text']))))
        self.assertEqual(second['structured_outputs']['json'], expected_schema)
        self.assertEqual(AfterSalesTicket.model_json_schema(), expected_schema)
        canonical = json.dumps(expected_schema, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        self.assertEqual(engine.structured_schema_sha256, hashlib.sha256(canonical.encode('utf-8')).hexdigest())
        with self.assertRaises(AttributeError):
            engine.structured_output_mode = 'off'

    def test_concurrent_request_builders_do_not_mix_prompts_or_schema(self):
        engine = self.engine(lambda request: self.fail('Builder must not send HTTP'))
        texts = [EXAMPLE['text'], EXAMPLE['text'] + '请核实。']
        with ThreadPoolExecutor(max_workers=2) as pool:
            bodies = list(pool.map(engine.build_request, texts))
        for text, body in zip(texts, bodies):
            self.assertEqual(body['prompt'], list(range(len(text))))
            self.assertEqual(body['structured_outputs']['json'], AfterSalesTicket.model_json_schema())
        self.assertIsNot(bodies[0]['structured_outputs']['json'], bodies[1]['structured_outputs']['json'])
        self.assertEqual(engine.calls, 0)

    def test_loaded_fingerprints_are_stable_and_isolate_modes_and_old_cache(self):
        on = self.loaded_engine()
        same = self.loaded_engine()
        off = self.loaded_engine(structured_outputs='off')
        self.assertTrue(on.ready)
        self.assertEqual(on.calls, 0)
        self.assertEqual(on.version, '0.11.2')
        self.assertEqual(on.fingerprint, same.fingerprint)
        self.assertNotEqual(on.fingerprint, off.fingerprint)
        old_identity = {'base': on._base_fingerprint, 'backend': 'vllm',
                        'version': '0.11.2', 'adapter_alias': 'slimllm-lora'}
        old = hashlib.sha256(json.dumps(old_identity, sort_keys=True).encode()).hexdigest()
        values = {}
        client = Mock()
        client.get.side_effect = values.get
        client.setex.side_effect = lambda key, ttl, value: values.__setitem__(key, value)
        for namespace in (old, on._base_fingerprint):
            RedisExactCache(namespace, client=client).set(EXAMPLE['text'], 'old-cache-value')
        on_cache = RedisExactCache(on.fingerprint, client=client)
        off_cache = RedisExactCache(off.fingerprint, client=client)
        self.assertIsNone(on_cache.get(EXAMPLE['text']))
        self.assertIsNone(off_cache.get(EXAMPLE['text']))
        on_cache.set(EXAMPLE['text'], 'new-cache-value')
        self.assertEqual(on_cache.get(EXAMPLE['text']), 'new-cache-value')
        self.assertIsNone(off_cache.get(EXAMPLE['text']))

    def test_schema_content_change_changes_hash_and_loaded_fingerprint(self):
        original = self.loaded_engine()
        revised = AfterSalesTicket.model_json_schema()
        revised['description'] += ' Unit fixture revision.'
        read_text = Path.read_text

        def fixture_read(path, *args, **kwargs):
            if path == ROOT / 'data/ticket.schema.json':
                return json.dumps(revised, ensure_ascii=False)
            return read_text(path, *args, **kwargs)

        # Simulate matching contract versions in memory, never modify schema files.
        with patch.object(AfterSalesTicket, 'model_json_schema', return_value=revised), \
             patch.object(Path, 'read_text', new=fixture_read):
            changed = self.loaded_engine()
        self.assertNotEqual(original.structured_schema_sha256, changed.structured_schema_sha256)
        self.assertNotEqual(original.fingerprint, changed.fingerprint)

    def test_schema_drift_is_rejected_before_any_http(self):
        revised = AfterSalesTicket.model_json_schema()
        revised['description'] += ' Mismatched fixture.'
        with patch.object(AfterSalesTicket, 'model_json_schema', return_value=revised):
            with self.assertRaisesRegex(ValueError, 'schema differs'):
                VllmExtractor()

    def test_protocol_revision_and_grammar_backend_are_in_fingerprint(self):
        original = self.loaded_engine()
        with patch('slimllm.serving.vllm_backend.REQUEST_PROTOCOL_REVISION', 'unit-fixture-v2'):
            revised = self.loaded_engine()
        with patch('slimllm.serving.vllm_backend.STRUCTURED_OUTPUT_BACKEND', 'unit-fixture-backend'):
            other_backend = self.loaded_engine()
        self.assertNotEqual(original.fingerprint, revised.fingerprint)
        self.assertNotEqual(original.fingerprint, other_backend.fingerprint)

    def test_unvalidated_server_version_and_missing_lora_stop_startup(self):
        for version in ('0.12.0', None):
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, 'Unvalidated vLLM version'):
                self.loaded_engine(version=version)
        with self.assertRaisesRegex(ValueError, 'missing the required LoRA'):
            self.loaded_engine(alias=False)

    def test_invalid_truncated_or_ungrounded_output_rejected(self):
        for payload in (
            {'choices':[{'text':'{}','finish_reason':'length'}],'usage':{'completion_tokens':384}},
            {'choices':[{'text':'{}','finish_reason':'stop'}],'usage':{'completion_tokens':3}},
            {'choices':[{'text':'```json\n{}\n```','finish_reason':'stop'}],'usage':{'completion_tokens':10}},
            {'choices':[{'text':json.dumps(EXAMPLE['label'] | {'requested_action':'安排寄修'}), 'finish_reason':'stop'}],'usage':{'completion_tokens':31}},
            {'choices':[{'text':json.dumps(EXAMPLE['label'] | {'reason':'made up'}),'finish_reason':'stop'}],'usage':{'completion_tokens':10}},
            {'choices':[]},
        ):
            with self.subTest(payload=payload):
                engine = self.engine(lambda request: httpx.Response(200, json=payload))
                with self.assertRaises(InvalidModelOutput): engine.extract_with_metrics(EXAMPLE['text'])

    def test_input_limit_blocks_upstream_request(self):
        def handler(request): self.fail('Overlong input must never reach server')
        engine = self.engine(handler)
        with self.assertRaises(InputTooLong): engine.extract_with_metrics('x'*2049)
        self.assertEqual(engine.calls, 0)

    def test_transport_errors_and_unhealthy_endpoint(self):
        def timeout(request): raise httpx.ReadTimeout('fixture private', request=request)
        engine = self.engine(timeout)
        with self.assertRaises(BackendTimeout): engine.extract_with_metrics(EXAMPLE['text'])
        def unavailable(request): raise httpx.ConnectError('fixture private', request=request)
        engine = self.engine(unavailable)
        self.assertFalse(engine.check_health())
        with self.assertRaises(BackendUnavailable): engine.extract_with_metrics(EXAMPLE['text'])

    def test_upstream_rejection_never_retries_or_drops_constraints(self):
        for mode in ('json_schema', 'off'):
            for status in (400, 422):
                with self.subTest(mode=mode, status=status):
                    requests = []

                    def handler(request):
                        requests.append(json.loads(request.content))
                        return httpx.Response(status, json={'error': {'message': 'private upstream detail'}})

                    engine = self.engine(handler, structured_outputs=mode)
                    with self.assertRaises(BackendRequestRejected) as caught:
                        engine.extract_with_metrics(EXAMPLE['text'])
                    self.assertEqual(caught.exception.status_code, status)
                    self.assertNotIn('private upstream detail', str(caught.exception))
                    self.assertEqual(len(requests), 1)
                    self.assertEqual(engine.calls, 1)
                    self.assertTrue(engine.ready)
                    self.assertEqual('structured_outputs' in requests[0], mode == 'json_schema')

    def test_other_upstream_errors_keep_existing_mapping_without_retry(self):
        for status, expected in ((429, BackendUnavailable), (503, BackendUnavailable),
                                 (500, InvalidModelOutput), (404, InvalidModelOutput)):
            with self.subTest(status=status):
                requests = []

                def handler(request):
                    requests.append(request)
                    return httpx.Response(status, text='private upstream detail')

                engine = self.engine(handler)
                with self.assertRaises(expected):
                    engine.extract_with_metrics(EXAMPLE['text'])
                self.assertEqual(len(requests), 1)
                self.assertEqual(engine.calls, 1)

    def test_api_upstream_rejection_is_sanitized_not_cached_and_consumer_recovers(self):
        for status in (400, 422):
            with self.subTest(status=status):
                requests = []

                def handler(request):
                    requests.append(json.loads(request.content))
                    if len(requests) == 1:
                        return httpx.Response(status, json={'error': {
                            'message': 'private-schema-error', 'ticket': EXAMPLE['text']}})
                    return httpx.Response(200, json={
                        'choices': [{'text': json.dumps(EXAMPLE['label']), 'finish_reason': 'stop'}],
                        'usage': {'completion_tokens': 31}})

                engine = self.engine(handler)
                redis = Mock()
                redis.ping.return_value = True
                redis.get.return_value = None
                cache = RedisExactCache(engine.fingerprint, client=redis)
                with patch.object(engine, 'load'), patch.object(engine, 'check_health', return_value=True), \
                     TestClient(create_app(lambda: engine, lambda _: cache, backend='vllm')) as client:
                    with self.assertLogs('slimllm.api', level='WARNING') as logs:
                        response = client.post('/extract', json={'text': EXAMPLE['text']})
                    self.assertEqual(response.status_code, 502)
                    self.assertEqual(response.json(), {'detail': {'code': 'backend_request_rejected'}})
                    logged = '\n'.join(logs.output)
                    self.assertIn(f'status={status}', logged)
                    self.assertIn('type=BackendRequestRejected', logged)
                    for private in ('private-schema-error', EXAMPLE['text']):
                        self.assertNotIn(private, response.text)
                        self.assertNotIn(private, logged)
                    redis.setex.assert_not_called()
                    self.assertEqual(len(requests), 1)
                    self.assertFalse(client.get('/health').json()['busy'])
                    recovered = client.post('/extract', json={'text': EXAMPLE['text']})
                    self.assertEqual(recovered.status_code, 200)
                    self.assertEqual(recovered.json(), EXAMPLE['label'])
                    self.assertEqual(len(requests), 2)
                    self.assertTrue(all('structured_outputs' in body for body in requests))
                    self.assertEqual(engine.calls, 2)
                    redis.setex.assert_called_once()

    def test_health_reports_configured_mode_and_schema_hash(self):
        for mode in ('json_schema', 'off'):
            with self.subTest(mode=mode):
                engine = self.loaded_engine(structured_outputs=mode)
                with patch.object(engine, 'load'), TestClient(create_app(lambda: engine, backend='vllm')) as client:
                    response = client.get('/health')
                    self.assertEqual(response.status_code, 200)
                    health = response.json()
                    self.assertEqual(health['structured_output_mode'], mode)
                    self.assertEqual(health['structured_schema_sha256'], engine.structured_schema_sha256)
                    self.assertEqual(health['queue']['workers'], 2)
                    self.assertEqual(engine.calls, 0)

    def test_only_local_url_and_valid_backend_allowed(self):
        for url in ('https://example.com','http://127.0.0.1:8001/v1','http://user@localhost:8001'):
            with patch.dict('os.environ',{'SLIMLLM_VLLM_URL':url}):
                with self.assertRaises(ValueError): VllmExtractor()
        with self.assertRaises(ValueError): create_app(backend='not-a-backend')

    def test_remote_dispatch_is_concurrent_but_bounded(self):
        engine = RemoteEngineFixture()
        with patch.dict('os.environ',{'SLIMLLM_VLLM_MAX_INFLIGHT':'2'}):
            with TestClient(create_app(lambda:engine, backend='vllm')) as client, ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(client.post,'/extract',json={'text':EXAMPLE['text']}) for _ in range(2)]
                responses=[f.result(timeout=4) for f in futures]
                self.assertEqual([r.status_code for r in responses],[200,200])
                health=client.get('/health').json()
                self.assertEqual(health['backend'],'vllm')
                self.assertEqual(health['structured_output_mode'], 'json_schema')
                self.assertEqual(health['structured_schema_sha256'], engine.structured_schema_sha256)
                self.assertEqual(health['queue']['workers'],2)
                self.assertEqual(health['queue']['waiting'],0)
                self.assertEqual(engine.calls,2)

    def test_transformers_ignores_remote_inflight_setting(self):
        engine = RemoteEngineFixture()
        with patch.dict('os.environ',{'SLIMLLM_VLLM_MAX_INFLIGHT':'4', 'SLIMLLM_VLLM_STRUCTURED_OUTPUTS':'invalid'}):
            with TestClient(create_app(lambda:engine, backend='transformers')) as client:
                health = client.get('/health').json()
                self.assertEqual(health['backend'],'transformers')
                self.assertEqual(health['queue']['workers'],1)
                self.assertEqual(health['structured_output_mode'], 'none')
                self.assertIsNone(health['structured_schema_sha256'])


if __name__ == '__main__': unittest.main()
