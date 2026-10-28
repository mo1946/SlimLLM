"""Scoring/protocol fixtures only: no real model requests or performance results."""

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from slimllm.evaluation.evaluate_structured import (REGRESSION_SHA256, assess_output, collect_completion,
                                 compare, completion_valid, load_evaluation_data, score_structured)
from slimllm.runtime import ROOT, file_hash, read_jsonl
from slimllm.schema import AfterSalesTicket


BASE = read_jsonl(ROOT / 'data/examples.jsonl')[0]
REGRESSION = read_jsonl(ROOT / 'data/structured_cases.jsonl')


def prediction(row, raw=..., **changes):
    return {'id': row['id'], 'attempted': True, 'status': 200,
            'raw_output': json.dumps(row['label'], ensure_ascii=False) if raw is ... else raw,
            'finish_reason': 'stop', 'usage': {'completion_tokens': 31}, 'max_tokens': 384,
            **changes}


def copies(count):
    return [dict(BASE, id=f'fixture-{i}', label=dict(BASE['label'])) for i in range(count)]


class StructuredScoringTests(unittest.TestCase):
    def test_fixed_arithmetic_from_plan(self):
        rows = copies(3)
        wrong = rows[1]['label'] | {'priority': 'high'}
        result = score_structured(rows, [prediction(rows[0]), prediction(rows[1], json.dumps(wrong)),
                                         prediction(rows[2], 'not JSON')])
        for key in ('json_valid_rate_all', 'enum_valid_rate_all', 'schema_valid_rate_all'):
            self.assertEqual(result[key], 2 / 3)
        self.assertEqual(result['classification_macro_accuracy_all'], 7 / 12)
        self.assertEqual(result['classification_macro_accuracy_enum_valid'], 7 / 8)
        self.assertEqual(result['enum_valid_subset_count'], 2)
        self.assertEqual(result['accepted_completion_count'], 2)
        self.assertIsNone(result['business_api_success_rate'])

    def test_illegal_enum_keeps_other_observable_classifications(self):
        raw = json.dumps(BASE['label'] | {'requested_action': '安排寄修'})
        result = score_structured([BASE], [prediction(BASE, raw)])
        self.assertEqual(result['json_valid_rate_all'], 1)
        self.assertEqual(result['enum_valid_rate_all'], 0)
        self.assertEqual(result['schema_valid_rate_all'], 0)
        self.assertEqual(result['enum_field_valid_rate_all']['requested_action'], 0)
        self.assertEqual(result['enum_field_valid_rate_all']['intent'], 1)
        self.assertEqual(result['classification_macro_accuracy_all'], .75)
        self.assertEqual(result['classification_accuracy_all']['intent'], 1)
        self.assertIsNone(result['classification_macro_accuracy_enum_valid'])
        self.assertEqual(result['legacy_seven_field_metrics']['field_macro_accuracy'], 0)

    def test_legal_but_wrong_priority_and_action_are_semantic_errors(self):
        row = REGRESSION[0]
        raw = json.dumps(row['label'] | {'priority': 'high', 'requested_action': 'refund'})
        result = score_structured([row], [prediction(row, raw)])
        self.assertEqual(result['enum_valid_rate_all'], 1)
        self.assertEqual(result['schema_valid_rate_all'], 1)
        self.assertEqual(result['classification_macro_accuracy_all'], .5)
        self.assertEqual(result['classification_accuracy_all']['requested_action'], 0)
        self.assertEqual(result['classification_accuracy_all']['priority'], 0)

    def test_json_syntax_is_independent_of_schema(self):
        raw = json.dumps(BASE['label'])
        invalid_json = [raw[:-1], '```json\n' + raw + '\n```',
                        '{"intent":"inquiry",' + raw[1:], raw.replace('"normal"', 'NaN')]
        for value in invalid_json:
            with self.subTest(raw=value):
                values = score_structured([BASE], [prediction(BASE, value)])
                self.assertEqual(values['json_valid_rate_all'], 0)
                self.assertEqual(values['schema_valid_rate_all'], 0)
        for value in ('{}', '[]', 'null'):
            with self.subTest(raw=value):
                values = score_structured([BASE], [prediction(BASE, value)])
                self.assertEqual(values['json_valid_rate_all'], 1)
                self.assertEqual(values['enum_valid_rate_all'], 0)
                self.assertEqual(values['schema_valid_rate_all'], 0)

    def test_schema_errors_do_not_erase_valid_classification_fields(self):
        values = [BASE['label'] | {'extra': 'bad'}, BASE['label'] | {'order_id': 123},
                  {key: value for key, value in BASE['label'].items() if key != 'order_id'},
                  BASE['label'] | {'product_name': ' '}, BASE['label'] | {'reason': '中' * 301}]
        for label in values:
            with self.subTest(label=label):
                result = score_structured([BASE], [prediction(BASE, json.dumps(label))])
                self.assertEqual(result['schema_valid_rate_all'], 0)
                self.assertEqual(result['enum_valid_rate_all'], 1)
                self.assertEqual(result['classification_macro_accuracy_all'], 1)
                self.assertEqual(result['legacy_seven_field_metrics']['field_macro_accuracy'], 0)

    def test_grounding_null_unicode_and_escaping_are_separate(self):
        invented = json.dumps(BASE['label'] | {'reason': 'made-up reason'})
        result = score_structured([BASE], [prediction(BASE, invented)])
        self.assertEqual(result['schema_valid_rate_all'], 1)
        self.assertEqual(result['classification_macro_accuracy_all'], 1)
        self.assertEqual(result['grounded_rate_all'], 0)
        self.assertEqual(result['accepted_completion_count'], 0)
        for row, raw in ((REGRESSION[4], json.dumps(REGRESSION[4]['label'])),
                         (BASE, json.dumps(BASE['label'], ensure_ascii=True))):
            with self.subTest(id=row['id']):
                checked = assess_output(row, prediction(row, raw))
                self.assertTrue(checked['schema_valid'])
                self.assertTrue(checked['grounded'])

    def test_parseable_truncation_never_counts_as_accepted_completion(self):
        for changes in ({'finish_reason': 'length'}, {'usage': {'completion_tokens': 384}},
                        {'usage': {'completion_tokens': True}}, {'usage': {'completion_tokens': 0}},
                        {'usage': None}):
            with self.subTest(changes=changes):
                item = prediction(BASE, **changes)
                result = score_structured([BASE], [item])
                self.assertEqual(result['json_valid_rate_all'], 1)
                self.assertEqual(result['schema_valid_rate_all'], 1)
                self.assertEqual(result['completed_count'], 0)
                self.assertEqual(result['accepted_completion_count'], 0)
                self.assertFalse(completion_valid(item))

    def test_failed_missing_and_error_json_stay_in_denominator(self):
        rows = copies(3)
        failed = prediction(rows[1], status=503, error='upstream_http_503')
        result = score_structured(rows, [prediction(rows[0]), failed])
        self.assertEqual(result['planned_count'], 3)
        self.assertEqual(result['attempted_count'], 2)
        self.assertEqual(result['not_executed_count'], 1)
        self.assertEqual(result['http_success_rate_all'], 1 / 3)
        self.assertEqual(result['json_valid_rate_all'], 1 / 3)
        self.assertEqual(result['classification_macro_accuracy_all'], 1 / 3)
        self.assertEqual(result['error_counts']['not_executed'], 1)

    def test_duplicate_unknown_and_invalid_gold_ids_rejected(self):
        for records, predictions in (([], []), ([BASE, BASE], []),
                                      ([BASE], [prediction(BASE), prediction(BASE)]),
                                      ([BASE], [prediction(BASE, id='unknown')])):
            with self.subTest(records=len(records), predictions=len(predictions)), self.assertRaises(ValueError):
                score_structured(records, predictions)

    def test_reviewed_regression_and_frozen_test_sets(self):
        datasets, sources = load_evaluation_data()
        self.assertEqual(len(datasets['frozen_test']), 50)
        self.assertEqual(len(datasets['regression']), 12)
        self.assertEqual(file_hash(ROOT / 'data/structured_cases.jsonl'), REGRESSION_SHA256)
        self.assertEqual(sources['regression']['sha256'], REGRESSION_SHA256)
        self.assertEqual(datasets['regression'][0]['label']['reason'], '充不进去电了')
        self.assertEqual(datasets['regression'][2]['label']['priority'], 'high')
        self.assertEqual(datasets['regression'][3]['label']['priority'], 'low')
        self.assertEqual(datasets['regression'][5]['label']['order_id'], '000012')

    def test_changed_regression_hash_is_not_silently_accepted(self):
        with patch('slimllm.evaluation.evaluate_structured.file_hash', return_value='changed'):
            with self.assertRaisesRegex(ValueError, 'Regression labels changed'):
                load_evaluation_data()


class StructuredProtocolTests(unittest.TestCase):
    def setUp(self):
        self.datasets = {'frozen_test': [dict(BASE, id=f'f{i}', text=BASE['text'] + str(i)) for i in range(2)],
                         'regression': [REGRESSION[0], REGRESSION[4]]}
        self.by_text = {row['text']: row for rows in self.datasets.values() for row in rows}
        self.instances = []
        self.calls = Counter()
        self.requests = []

    def factory(self, *, failure=None, changed=None):
        owner = self

        class EngineFixture:
            def __init__(self, structured_outputs):
                owner.assertTrue(all(engine.closed for engine in owner.instances))
                self.mode = structured_outputs
                self._base_fingerprint = 'unit-fixture-base'
                self.fingerprint = 'unit-fixture-' + self.mode
                self.url = 'http://127.0.0.1:8001'
                self.version = 'unit-test-fixture-not-a-server'
                self.model_name = 'slimllm-lora'
                self.adapter = {'config_sha256': 'unit-fixture', 'weights_sha256': 'unit-fixture'}
                self.prompt = 'unit fixture; not real inference'
                self.closed = False
                if self.mode == 'json_schema' and changed == 'identity':
                    self.version = 'changed-unit-fixture'
                self.client = httpx.Client(base_url=self.url, transport=httpx.MockTransport(self.handle))
                owner.instances.append(self)

            def load(self):
                pass

            def close(self):
                self.client.close()
                self.closed = True

            def build_request(self, text):
                payload = {'model': self.model_name, 'prompt': list(text.encode('utf-8')), 'max_tokens': 384,
                           'temperature': 0, 'seed': 42, 'n': 1, 'stream': False}
                if self.mode == 'json_schema':
                    payload['structured_outputs'] = {'json': AfterSalesTicket.model_json_schema()}
                    if changed == 'seed':
                        payload['seed'] = 43
                    elif changed == 'tokens':
                        payload['prompt'] = payload['prompt'] + [1]
                    elif changed == 'missing_constraint':
                        del payload['structured_outputs']
                return payload

            def handle(self, request):
                owner.assertEqual(request.url.path, '/v1/completions')
                payload = json.loads(request.content)
                owner.calls[self.mode] += 1
                owner.requests.append((self.mode, payload))
                if failure is not None:
                    failed = failure(self.mode, owner.calls[self.mode], request)
                    if failed is not None:
                        return failed
                text = bytes(payload['prompt']).decode('utf-8')
                label = owner.by_text[text]['label']
                if self.mode == 'off' and owner.calls[self.mode] == 3:
                    label = label | {'requested_action': '安排寄修'}
                return httpx.Response(200, json={
                    'choices': [{'text': json.dumps(label, ensure_ascii=False), 'finish_reason': 'stop'}],
                    'usage': {'completion_tokens': 31}})

        return EngineFixture

    def run_fixture(self, folder, factory):
        # Every generated report is explicitly marked unit_test_fixture and is
        # confined to an automatically cleaned temporary test directory.
        with patch('slimllm.evaluation.evaluate_structured.load_evaluation_data', return_value=(self.datasets, {'scope': 'unit_fixture'})):
            return compare(str(folder), engine_factory=factory)

    def test_collector_preserves_raw_before_any_output_validation(self):
        raw = json.dumps(BASE['label'] | {'requested_action': '安排寄修'})
        class ClientFixture:
            def post(self, path, **kwargs):
                return httpx.Response(200, json={'choices': [{'text': raw, 'finish_reason': 'stop'}],
                                                'usage': {'completion_tokens': 31}})
        class EngineFixture:
            client = ClientFixture()
        with patch.object(AfterSalesTicket, 'model_validate', side_effect=AssertionError('Must persist before validation')):
            result = collect_completion(EngineFixture(), BASE, {'prompt': [1, 2], 'max_tokens': 384})
        self.assertEqual(result['raw_output'], raw)
        self.assertIsNone(result['error'])
        self.assertEqual(result['usage'], {'completion_tokens': 31})
        self.assertFalse(assess_output(BASE, result)['enum_valid'])

    def test_malformed_upstream_envelope_aborts_and_parseable_truncation_is_retained(self):
        responses = [httpx.Response(200, text='not JSON'),
                     httpx.Response(200, json={'choices': []}),
                     httpx.Response(200, json={'choices': [{'text': '{}', 'finish_reason': 'stop'}],
                                               'usage': {'completion_tokens': True}})]
        for response in responses:
            with self.subTest(response=response.text):
                class EngineFixture:
                    client = type('ClientFixture', (), {'post': lambda *args, **kwargs: response})()
                item = collect_completion(EngineFixture(), BASE, {'prompt': [1], 'max_tokens': 384})
                self.assertEqual(item['abort_reason'], 'invalid_upstream_envelope')
                self.assertEqual(item['raw_response'], response.text)
        response = httpx.Response(200, json={
            'choices': [{'text': json.dumps(BASE['label']), 'finish_reason': 'length'}],
            'usage': {'completion_tokens': 384}})
        item = collect_completion(EngineFixture(), BASE, {'prompt': [1], 'max_tokens': 384})
        self.assertEqual(item['error'], 'truncated_completion')
        self.assertIsNone(item['abort_reason'])
        self.assertEqual(json.loads(item['raw_output']), BASE['label'])

    def test_sequential_compare_persists_raw_and_excludes_warmups(self):
        with tempfile.TemporaryDirectory(prefix='unit_structured_', dir=ROOT / 'reports') as temporary:
            folder = Path(temporary) / 'run'
            original_score = score_structured

            def score_after_write(rows, predictions):
                mode = self.instances[-1].mode
                dataset = next(name for name, values in self.datasets.items() if rows == values)
                persisted = read_jsonl(folder / mode / f'{dataset}_predictions.jsonl')
                self.assertEqual(len(persisted), len(rows))
                return original_score(rows, predictions)

            with patch('slimllm.evaluation.evaluate_structured.score_structured', side_effect=score_after_write):
                result = self.run_fixture(folder, self.factory())
            self.assertTrue(result['completed_all_requests'])
            self.assertEqual(result['evidence_kind'], 'unit_test_fixture')
            self.assertEqual(self.calls, {'off': 6, 'json_schema': 6})
            self.assertTrue(all(engine.closed for engine in self.instances))
            for mode in ('off', 'json_schema'):
                self.assertEqual(len(read_jsonl(folder / mode / 'warmup.jsonl')), 2)
                for metrics in result['groups'][mode]['metrics'].values():
                    self.assertEqual(metrics['planned_count'], 2)
                    self.assertEqual(metrics['attempted_count'], 2)
                self.assertTrue(result['groups'][mode]['conditions_verified'])
            raw = read_jsonl(folder / 'off/frozen_test_predictions.jsonl')[0]['raw_output']
            self.assertIn('安排寄修', raw)
            self.assertEqual(result['groups']['off']['metrics']['frozen_test']['enum_valid_count'], 1)
            self.assertEqual(result['groups']['json_schema']['metrics']['frozen_test']['enum_valid_count'], 2)
            self.assertIn('unit_test_fixture', (folder / 'report.md').read_text(encoding='utf-8'))

    def test_http_rejection_oom_and_timeout_abort_remaining_work_without_retry(self):
        for failure_kind in (400, 422, 500, 503, 'timeout'):
            with self.subTest(failure=failure_kind), tempfile.TemporaryDirectory(prefix='unit_structured_', dir=ROOT / 'reports') as temporary:
                self.instances, self.calls, self.requests = [], Counter(), []

                def fail(mode, number, request):
                    if number == 3:
                        if failure_kind == 'timeout':
                            raise httpx.ReadTimeout('unit fixture', request=request)
                        return httpx.Response(failure_kind, json={'error': {'message': 'unit fixture resource/config failure'}})

                folder = Path(temporary) / 'run'
                result = self.run_fixture(folder, self.factory(failure=fail))
                self.assertFalse(result['completed_all_requests'])
                self.assertTrue(result['aborted_reason'])
                self.assertEqual(self.calls['off'], 3)
                self.assertEqual(self.calls['json_schema'], 0)
                self.assertEqual(len(self.instances), 1)
                self.assertTrue(self.instances[0].closed)
                attempted = sum(values['attempted_count'] for group in result['groups'].values() for values in group['metrics'].values())
                self.assertEqual(attempted, 1)
                for mode in ('off', 'json_schema'):
                    for dataset, rows in self.datasets.items():
                        predictions = read_jsonl(folder / mode / f'{dataset}_predictions.jsonl')
                        self.assertEqual({item['id'] for item in predictions}, {row['id'] for row in rows})
                        self.assertEqual(len(predictions), len(rows))

    def test_warmup_failure_is_not_mixed_into_measured_denominators(self):
        with tempfile.TemporaryDirectory(prefix='unit_structured_', dir=ROOT / 'reports') as temporary:
            result = self.run_fixture(Path(temporary) / 'run', self.factory(
                failure=lambda mode, number, request: httpx.Response(503)))
            self.assertEqual(self.calls['off'], 1)
            for group in result['groups'].values():
                for metrics in group['metrics'].values():
                    self.assertEqual(metrics['attempted_count'], 0)
                    self.assertEqual(metrics['not_executed_count'], metrics['planned_count'])

    def test_changed_conditions_stop_on_group_before_generation(self):
        for changed in ('identity', 'seed', 'tokens', 'missing_constraint'):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory(prefix='unit_structured_', dir=ROOT / 'reports') as temporary:
                self.instances, self.calls, self.requests = [], Counter(), []
                result = self.run_fixture(Path(temporary) / 'run', self.factory(changed=changed))
                self.assertFalse(result['completed_all_requests'])
                self.assertEqual(self.calls['off'], 6)
                self.assertEqual(self.calls['json_schema'], 0)
                self.assertTrue(all(engine.closed for engine in self.instances))

    def test_artifact_drift_aborts_before_first_generation(self):
        with tempfile.TemporaryDirectory(prefix='unit_structured_', dir=ROOT / 'reports') as temporary:
            with patch('slimllm.evaluation.evaluate_structured._artifact_snapshot', side_effect=[{'fixture': 'before'}, {'fixture': 'changed'}]):
                result = self.run_fixture(Path(temporary) / 'run', self.factory())
            self.assertFalse(result['completed_all_requests'])
            self.assertIn('Frozen artifacts changed', result['aborted_reason'])
            self.assertEqual(sum(self.calls.values()), 0)
            self.assertTrue(all(engine.closed for engine in self.instances))

    def test_startup_failure_closes_client_and_keeps_planned_denominators(self):
        with tempfile.TemporaryDirectory(prefix='unit_structured_', dir=ROOT / 'reports') as temporary:
            factory = self.factory()
            with patch.object(factory, 'load', side_effect=RuntimeError('unit fixture startup failure')):
                result = self.run_fixture(Path(temporary) / 'run', factory)
            self.assertFalse(result['completed_all_requests'])
            self.assertEqual(sum(self.calls.values()), 0)
            self.assertEqual(len(self.instances), 1)
            self.assertTrue(self.instances[0].closed)
            for group in result['groups'].values():
                for metrics in group['metrics'].values():
                    self.assertEqual(metrics['attempted_count'], 0)
                    self.assertEqual(metrics['not_executed_count'], metrics['planned_count'])

    def test_output_guard_rejects_existing_and_outside_reports_before_loading(self):
        with tempfile.TemporaryDirectory(prefix='unit_structured_', dir=ROOT / 'reports') as temporary:
            with patch('slimllm.evaluation.evaluate_structured.load_evaluation_data', side_effect=AssertionError('No data or engine loading')):
                for folder in (temporary, str(ROOT / '.cache/invalid_comparison_output')):
                    with self.subTest(folder=folder), self.assertRaises(ValueError):
                        compare(folder)

    def test_default_client_rejects_nonlocal_url_without_loading_model(self):
        with tempfile.TemporaryDirectory(prefix='unit_structured_', dir=ROOT / 'reports') as temporary:
            with patch.dict('os.environ', {'SLIMLLM_VLLM_URL': 'https://example.invalid'}), \
                 patch('slimllm.evaluation.evaluate_structured.load_evaluation_data', return_value=(self.datasets, {'scope': 'unit_fixture'})), \
                 patch('slimllm.serving.vllm_backend.VllmExtractor.load', side_effect=AssertionError('No network or model loading')):
                result = compare(str(Path(temporary) / 'run'))
            self.assertFalse(result['completed_all_requests'])
            self.assertIn('local HTTP endpoint', result['aborted_reason'])
            self.assertEqual(sum(values['attempted_count'] for group in result['groups'].values()
                                 for values in group['metrics'].values()), 0)


if __name__ == '__main__':
    unittest.main()
