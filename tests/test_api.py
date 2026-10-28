"""HTTP/validation unit fixtures only; real GPU HTTP calls are recorded separately."""

from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from slimllm.serving.api import create_app
from slimllm.serving.inference import InputTooLong, InvalidModelOutput, validate_output
from slimllm.serving.response_cache import RedisExactCache
from slimllm.runtime import ROOT, read_jsonl
from slimllm.schema import AfterSalesTicket
from slimllm.serving.vllm_backend import BackendRequestRejected, BackendTimeout, BackendUnavailable


EXAMPLE = read_jsonl(ROOT / 'data/examples.jsonl')[0]


class EngineFixture:
    fingerprint = 'unit-fixture'

    def __init__(self):
        self.ready = False
        self.calls = 0
        self.closed = False
        self.error = None
        self.started = None
        self.release = None

    def load(self):
        self.ready = True

    def close(self):
        self.ready = False
        self.closed = True

    def extract(self, text):
        self.calls += 1
        if self.started:
            self.started.set()
            if not self.release.wait(5):
                raise RuntimeError('Fixture wait timed out.')
        if self.error:
            raise self.error
        return AfterSalesTicket.model_validate(EXAMPLE['label'])


class RedisClientFixture:
    def __init__(self):
        self.values = {}
        self.fail = False
        self.last_ttl = None

    def ping(self):
        if self.fail:
            raise ConnectionError('fixture unavailable')
        return True

    def get(self, key):
        self.ping()
        return self.values.get(key)

    def setex(self, key, ttl, value):
        self.ping()
        self.last_ttl = ttl
        self.values[key] = value

    def close(self):
        pass


class ApiTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict('os.environ', {'SLIMLLM_BACKEND':'transformers'})
        environment.start()
        self.addCleanup(environment.stop)

    def test_health_valid_response_and_shutdown(self):
        engine = EngineFixture()
        with TestClient(create_app(lambda: engine)) as client:
            self.assertEqual(client.get('/health').status_code, 200)
            health = client.get('/health').json()
            self.assertEqual(health['structured_output_mode'], 'none')
            self.assertIsNone(health['structured_schema_sha256'])
            response = client.post('/extract', json={'text': EXAMPLE['text']})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), EXAMPLE['label'])
            self.assertEqual(response.headers['X-SlimLLM-Cache'], 'DISABLED')
        self.assertTrue(engine.closed)

    def test_invalid_requests_never_invoke_model(self):
        engine = EngineFixture()
        with TestClient(create_app(lambda: engine)) as client:
            for body in ({}, {'text': ''}, {'text': '   '}, {'text': 42}, {'text': None},
                         {'text': 'x' * 2001}, {'text': 'a', 'max_tokens': 99999}):
                with self.subTest(body=body):
                    self.assertEqual(client.post('/extract', json=body).status_code, 422)
            self.assertEqual(engine.calls, 0)

    def test_inference_metrics_are_headers_without_changing_ticket(self):
        engine = EngineFixture()
        engine.last_metrics = {'generated_tokens': 31, 'generation_seconds': 1.25,
                               'peak_allocated_bytes': 100, 'peak_reserved_bytes': 200}
        with TestClient(create_app(lambda: engine)) as client:
            response = client.post('/extract', json={'text': EXAMPLE['text']})
            self.assertEqual(response.json(), EXAMPLE['label'])
            self.assertEqual(response.headers['X-SlimLLM-generated-tokens'], '31')
            self.assertEqual(response.headers['X-SlimLLM-generation-seconds'], '1.25')
            self.assertEqual(response.headers['X-SlimLLM-peak-allocated-bytes'], '100')

    def test_unready_model_returns_503(self):
        engine = EngineFixture()
        with TestClient(create_app(lambda: engine)) as client:
            engine.ready = False
            self.assertEqual(client.get('/health').status_code, 503)
            self.assertEqual(client.post('/extract', json={'text': EXAMPLE['text']}).status_code, 503)

    def test_model_errors_are_sanitized_and_consumer_continues(self):
        engine, redis = EngineFixture(), RedisClientFixture()
        cache = RedisExactCache(engine.fingerprint, client=redis)
        with TestClient(create_app(lambda: engine, lambda _: cache)) as client:
            for error, expected, code in (
                (InputTooLong('secret text'), 422, 'input_token_limit'),
                (InvalidModelOutput('secret text'), 502, 'invalid_model_output'),
                (MemoryError('secret text'), 503, 'gpu_memory_unavailable'),
                (RuntimeError('secret text'), 500, 'inference_failed'),
                (BackendTimeout('secret text'), 504, 'backend_timeout'),
                (BackendUnavailable('secret text'), 503, 'backend_unavailable'),
                (BackendRequestRejected(400), 502, 'backend_request_rejected'),
                (BackendRequestRejected(422), 502, 'backend_request_rejected'),
            ):
                engine.error = error
                response = client.post('/extract', json={'text': EXAMPLE['text']})
                self.assertEqual(response.status_code, expected)
                self.assertEqual(response.json(), {'detail': {'code': code}})
                self.assertNotIn('secret text', response.text)
                self.assertFalse(client.get('/health').json()['busy'])
                self.assertEqual(redis.values, {})

    def test_openapi_declares_upstream_and_queue_errors(self):
        document = create_app().openapi()
        responses = document['paths']['/extract']['post']['responses']
        for status in ('502', '503', '504'):
            with self.subTest(status=status):
                self.assertTrue(responses[status]['description'])
        self.assertIn('configuration rejected', responses['502']['description'])
        self.assertIn('configured decoding policy', document['paths']['/health']['get']['description'])
        self.assertNotIn('raw_output', json.dumps(document))

    def test_queue_full_rejected_but_queued_request_succeeds(self):
        engine = EngineFixture()
        engine.started, engine.release = threading.Event(), threading.Event()
        with TestClient(create_app(lambda: engine, max_waiting=1)) as client, ThreadPoolExecutor(max_workers=2) as pool:
            pending = pool.submit(client.post, '/extract', json={'text': EXAMPLE['text']})
            try:
                self.assertTrue(engine.started.wait(3))
                self.assertTrue(client.get('/health').json()['busy'])
                queued = pool.submit(client.post, '/extract', json={'text': EXAMPLE['text']})
                deadline = time.monotonic() + 2
                while client.get('/health').json()['queue']['waiting'] != 1:
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(.005)
                response = client.post('/extract', json={'text': EXAMPLE['text']})
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.json()['detail']['code'], 'queue_full')
            finally:
                engine.release.set()
            self.assertEqual(pending.result(timeout=3).status_code, 200)
            queued_response = queued.result(timeout=3)
            self.assertEqual(queued_response.status_code, 200)
            self.assertGreater(float(queued_response.headers['X-SlimLLM-Queue-Wait-Seconds']), 0)
            self.assertEqual(engine.calls, 2)

    def test_queue_wait_timeout_does_not_start_another_generation(self):
        engine = EngineFixture()
        engine.started, engine.release = threading.Event(), threading.Event()
        with TestClient(create_app(lambda: engine, wait_timeout=.03)) as client, ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(client.post, '/extract', json={'text': EXAMPLE['text']})
            try:
                self.assertTrue(engine.started.wait(3))
                response = client.post('/extract', json={'text': EXAMPLE['text']})
                self.assertEqual(response.status_code, 504)
                self.assertEqual(response.json()['detail']['code'], 'queue_wait_timeout')
                self.assertEqual(client.get('/health').json()['queue']['waiting'], 0)
                self.assertEqual(engine.calls, 1)
            finally:
                engine.release.set()
            self.assertEqual(pending.result(timeout=3).status_code, 200)

    def test_exact_cache_hit_and_key_versioning(self):
        engine, redis = EngineFixture(), RedisClientFixture()
        cache = RedisExactCache(engine.fingerprint, client=redis, ttl=60)
        with TestClient(create_app(lambda: engine, lambda _: cache)) as client:
            first = client.post('/extract', json={'text': EXAMPLE['text']})
            second = client.post('/extract', json={'text': EXAMPLE['text']})
            self.assertEqual(first.headers['X-SlimLLM-Cache'], 'MISS')
            self.assertEqual(second.headers['X-SlimLLM-Cache'], 'HIT')
            self.assertEqual(first.json(), second.json())
            self.assertEqual(engine.calls, 1)
            self.assertEqual(redis.last_ttl, 60)
            self.assertNotEqual(cache.key(EXAMPLE['text']), cache.key(EXAMPLE['text'] + ' '))
            self.assertNotEqual(cache.key(EXAMPLE['text']), RedisExactCache('new-model').key(EXAMPLE['text']))
            self.assertNotIn(EXAMPLE['text'], cache.key(EXAMPLE['text']))

    def test_corrupt_cache_is_not_served(self):
        engine, redis = EngineFixture(), RedisClientFixture()
        cache = RedisExactCache(engine.fingerprint, client=redis)
        redis.values[cache.key(EXAMPLE['text'])] = '{}'
        with TestClient(create_app(lambda: engine, lambda _: cache)) as client:
            response = client.post('/extract', json={'text': EXAMPLE['text']})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers['X-SlimLLM-Cache'], 'MISS')
            self.assertEqual(engine.calls, 1)

    def test_cache_hit_bypasses_full_model_queue(self):
        engine, redis = EngineFixture(), RedisClientFixture()
        engine.started, engine.release = threading.Event(), threading.Event()
        cache = RedisExactCache(engine.fingerprint, client=redis)
        redis.values[cache.key(EXAMPLE['text'])] = json.dumps(EXAMPLE['label'])
        with TestClient(create_app(lambda: engine, lambda _: cache, max_waiting=1)) as client, ThreadPoolExecutor(max_workers=2) as pool:
            active = pool.submit(client.post, '/extract', json={'text': EXAMPLE['text'] + ' 1'})
            try:
                self.assertTrue(engine.started.wait(3))
                queued = pool.submit(client.post, '/extract', json={'text': EXAMPLE['text'] + ' 2'})
                deadline = time.monotonic() + 2
                while client.get('/health').json()['queue']['waiting'] != 1:
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(.005)
                hit = client.post('/extract', json={'text': EXAMPLE['text']})
                self.assertEqual(hit.status_code, 200)
                self.assertEqual(hit.headers['X-SlimLLM-Cache'], 'HIT')
                self.assertEqual(hit.headers['X-SlimLLM-Queue-Wait-Seconds'], '0')
                self.assertEqual(engine.calls, 1)
            finally:
                engine.release.set()
            self.assertEqual(active.result(timeout=3).status_code, 200)
            self.assertEqual(queued.result(timeout=3).status_code, 200)

    def test_cache_outage_keeps_inference_available(self):
        engine, redis = EngineFixture(), RedisClientFixture()
        cache = RedisExactCache(engine.fingerprint, client=redis)
        with TestClient(create_app(lambda: engine, lambda _: cache)) as client:
            redis.fail = True
            self.assertEqual(client.post('/extract', json={'text': EXAMPLE['text']}).status_code, 200)
            self.assertEqual(client.get('/health').json()['cache'], 'unavailable')

    def test_model_output_strictness_and_grounding(self):
        with self.assertRaises(InvalidModelOutput):
            validate_output('```json\n{}\n```', EXAMPLE['text'])
        with self.assertRaises(InvalidModelOutput):
            validate_output(json.dumps({**EXAMPLE['label'], 'order_id': 'null'}), EXAMPLE['text'])
        with self.assertRaises(InvalidModelOutput):
            validate_output(json.dumps({**EXAMPLE['label'], 'reason': 'made-up reason'}), EXAMPLE['text'])
        self.assertEqual(validate_output(json.dumps(EXAMPLE['label']), EXAMPLE['text']).model_dump(), EXAMPLE['label'])


if __name__ == '__main__':
    unittest.main()
