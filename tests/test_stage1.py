"""Offline unit tests with explicit fixtures; never reported as model metrics."""

import json
import subprocess
import sys
import unittest
from unittest.mock import patch

from baseline import read_range
from dataset import payload_for, split_records, validate_record
from evaluate import score, strict_json
from runtime import ROOT, project_path, read_jsonl


class StageOneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = read_jsonl(ROOT / 'data/examples.jsonl')

    def test_reject_hallucinated_entity(self):
        row = self.rows[0]
        with self.assertRaises(ValueError):
            validate_record({'text': row['text'], 'label': {**row['label'], 'order_id': 'NOT-IN-TEXT'}})

    def test_accept_manual_fixture(self):
        row = self.rows[0]
        self.assertEqual(validate_record({'text': row['text'], 'label': row['label']})['label'], row['label'])

    def test_output_cannot_escape_workspace(self):
        with self.assertRaises(ValueError):
            project_path('../outside')

    def test_request_json_mode_and_caps(self):
        body = payload_for(0, 10)
        self.assertEqual(body['response_format'], {'type': 'json_object'})
        self.assertEqual(body['thinking'], {'type': 'disabled'})
        self.assertEqual(body['max_tokens'], 4096)

    def test_strict_json_does_not_repair(self):
        for text in ('{"a":1,"a":2}', '{"a":NaN}', '```json\n{}\n```', '{} trailing'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                strict_json(text)

    def test_metric_denominators_include_invalid_outputs(self):
        records = self.rows[:4]
        wrong = {**records[3]['label'], 'priority': 'high'}
        predictions = [
            {'id': records[0]['id'], 'raw_output': json.dumps(records[0]['label'])},
            {'id': records[1]['id'], 'raw_output': 'not json'},
            {'id': records[2]['id'], 'raw_output': '{}'},
            {'id': records[3]['id'], 'raw_output': json.dumps(wrong)},
        ]
        result = score(records, predictions)
        self.assertEqual(result['json_valid_rate'], .75)
        self.assertEqual(result['schema_valid_rate'], .5)
        self.assertEqual(result['field_accuracy']['priority'], .25)
        self.assertEqual(result['field_accuracy']['intent'], .5)
        self.assertEqual(result['exact_match_rate'], .25)

    def test_missing_extra_and_duplicate_predictions_fail(self):
        prediction = {'id': self.rows[0]['id'], 'raw_output': json.dumps(self.rows[0]['label'])}
        for values in ([], [prediction, prediction], [{**prediction, 'id': 'other'}]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                score(self.rows[:1], values)

    def test_template_variants_stay_together_and_split_is_deterministic(self):
        records = []
        for row in self.rows:
            for number in range(3):
                records.append({**row, 'id': f"{row['id']}-{number}"})
        first = split_records(records)
        self.assertEqual(first, split_records(records))
        all_ids = [item for ids in first.values() for item in ids]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(set(all_ids), {r['id'] for r in records})
        for row in self.rows:
            memberships = {name for name, ids in first.items() for identifier in ids if identifier.startswith(row['id'])}
            self.assertEqual(len(memberships), 1)

    def test_single_template_cannot_be_split(self):
        rows = [{**self.rows[0], 'id': f'fixture-{i}'} for i in range(10)]
        with self.assertRaises(ValueError):
            split_records(rows)

    def test_review_dry_run_cannot_send_requests(self):
        result = subprocess.run([sys.executable, '-B', str(ROOT / 'dataset.py'), 'review', '--dry-run'],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('no request was sent', result.stderr)

    def test_range_download_rejects_wrong_offset(self):
        # HTTP transport fixture only; no simulated model inference or benchmark output.
        class Response:
            status = 206
            headers = {'Content-Range': 'bytes 0-3/8'}
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
        with patch('baseline.urllib.request.urlopen', return_value=Response()), patch('baseline.time.sleep'):
            with self.assertRaisesRegex(ValueError, 'wrong byte range'):
                read_range('https://example.invalid/weight', 4, 7, 8)


if __name__ == '__main__':
    unittest.main()
