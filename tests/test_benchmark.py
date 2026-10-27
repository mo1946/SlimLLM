"""Metric/HTTP unit fixtures only; live timings are stored in reports/stage4_run."""

import asyncio
import json
import unittest

import httpx

from slimllm.evaluation.benchmark import fixed_records, latency_stats, request_one, summarize
from slimllm.runtime import ROOT, read_jsonl


class BenchmarkTests(unittest.TestCase):
    def test_interpolated_percentiles_and_empty_success_set(self):
        stats = latency_stats([1., 2., 3., 4.])
        self.assertEqual(stats['mean_ms'], 2500.)
        self.assertEqual(stats['p50_ms'], 2500.)
        self.assertAlmostEqual(stats['p95_ms'], 3850.)
        self.assertAlmostEqual(stats['p99_ms'], 3970.)
        self.assertIsNone(latency_stats([])['p99_ms'])

    def test_http_error_json_is_not_a_valid_ticket_or_correct_extraction(self):
        example = read_jsonl(ROOT / 'data/examples.jsonl')[0] | {'id': 'fixture'}

        async def fixture_calls():
            def handler(request):
                if request.headers.get('x-fixture-error'):
                    return httpx.Response(503, json={'detail': {'code': 'model_busy'}})
                return httpx.Response(200, json=example['label'], headers={
                    'X-SlimLLM-generated-tokens': '20', 'X-SlimLLM-generation-seconds': '2',
                    'X-SlimLLM-queue-wait-seconds': '0.1',
                    'X-SlimLLM-peak-allocated-bytes': '100', 'X-SlimLLM-peak-reserved-bytes': '200'})
            async with httpx.AsyncClient(base_url='http://unit-fixture', transport=httpx.MockTransport(handler)) as client:
                good = await request_one(client, example, 0)
                client.headers['x-fixture-error'] = '1'
                busy = await request_one(client, example, 1)
                return good, busy

        good, busy = asyncio.run(fixture_calls())
        self.assertTrue(good['schema_valid'])
        self.assertFalse(busy['json_valid'])
        self.assertEqual(busy['error'], 'model_busy')
        result = summarize([good, busy], 4.)
        self.assertEqual(result['http_success_rate'], .5)
        self.assertEqual(result['json_valid_rate_http_success'], 1.)
        self.assertEqual(result['schema_valid_rate_all'], .5)
        self.assertEqual(result['field_macro_accuracy_all'], .5)
        self.assertEqual(result['field_macro_accuracy_schema_valid'], 1.)
        self.assertEqual(result['attempt_rps'], .5)
        self.assertEqual(result['success_rps'], .25)
        self.assertEqual(result['model_generated_tokens_per_generation_second'], 10.)
        self.assertEqual(result['queue_wait']['p95_ms'], 100.)
        self.assertEqual(summarize([busy], 1.)['latency_http_success']['count'], 0)

    def test_fixed_workload_uses_frozen_test_order(self):
        rows, provenance = fixed_records(32)
        split = json.loads((ROOT / 'data/generated/stage1/splits.json').read_text(encoding='utf-8'))
        self.assertEqual([r['id'] for r in rows], split['ids']['test'][:32])
        self.assertEqual(provenance['ids'], [r['id'] for r in rows])
        self.assertFalse(set(provenance['ids']) & set(split['ids']['train']))

    def test_remote_token_count_does_not_fabricate_cuda_time_or_memory(self):
        fixture = {'http_success':True, 'schema_valid':True, 'json_valid':True,
            'correct_fields':[], 'exact_match':False, 'error':None, 'status':200,
            'elapsed_seconds':2., 'id':'fixture', 'generated_tokens':20,
            'backend_request_seconds':1.9, 'queue_wait_seconds':.1}
        result = summarize([fixture], 2.)
        self.assertEqual(result['generated_tokens_measured'], 20)
        self.assertEqual(result['generated_tokens_per_phase_second'], 10.)
        self.assertIsNone(result['model_generated_tokens_per_generation_second'])
        self.assertIsNone(result['peak_allocated_bytes'])
        self.assertEqual(result['generation_metrics_response_count'], 0)


if __name__ == '__main__':
    unittest.main()
