"""离线契约测试；手写样例不是模型推理或评测结果。"""

import json
from pathlib import Path
import unittest

from pydantic import ValidationError

from schema import AfterSalesTicket


ROOT = Path(__file__).resolve().parents[1]
FIELDS = {
    "intent", "order_id", "product_name", "issue_type", "priority",
    "requested_action", "reason",
}


class SchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.examples = [
            json.loads(line)
            for line in (ROOT / "data/examples.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_handwritten_examples(self) -> None:
        self.assertEqual(len(self.examples), 5)
        self.assertEqual(len({item["id"] for item in self.examples}), 5)
        for item in self.examples:
            with self.subTest(example=item["id"]):
                model = AfterSalesTicket.model_validate_json(json.dumps(item["label"]))
                self.assertEqual(model.model_dump(), item["label"])
                for key in ("order_id", "product_name", "reason"):
                    if item["label"][key] is not None:
                        self.assertIn(item["label"][key], item["text"])

    def test_all_keys_required_even_when_nullable(self) -> None:
        for key in FIELDS:
            payload = self.examples[0]["label"].copy()
            del payload[key]
            with self.subTest(field=key), self.assertRaises(ValidationError):
                AfterSalesTicket.model_validate(payload)

    def test_extra_keys_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            AfterSalesTicket.model_validate({**self.examples[0]["label"], "phone": "123"})

    def test_invalid_categories_rejected(self) -> None:
        for key in ("intent", "issue_type", "priority", "requested_action"):
            for value in ("invalid", "", None, 1):
                with self.subTest(field=key, value=value), self.assertRaises(ValidationError):
                    AfterSalesTicket.model_validate({**self.examples[0]["label"], key: value})

    def test_no_implicit_string_conversion(self) -> None:
        for key in ("order_id", "product_name", "reason"):
            for value in (123, True, [], {}):
                with self.subTest(field=key, value=value), self.assertRaises(ValidationError):
                    AfterSalesTicket.model_validate({**self.examples[0]["label"], key: value})

    def test_unknown_entities_are_null(self) -> None:
        payload = {**self.examples[0]["label"], "order_id": None, "product_name": None, "reason": None}
        self.assertEqual(AfterSalesTicket.model_validate(payload).model_dump(), payload)

    def test_empty_and_oversized_strings_rejected(self) -> None:
        for key, limit in (("order_id", 64), ("product_name", 120), ("reason", 300)):
            for value in ("", "   ", "x" * (limit + 1)):
                with self.subTest(field=key, value=value), self.assertRaises(ValidationError):
                    AfterSalesTicket.model_validate({**self.examples[0]["label"], key: value})
            model = AfterSalesTicket.model_validate({**self.examples[0]["label"], key: "x" * limit})
            self.assertEqual(getattr(model, key), "x" * limit)

    def test_order_id_preserves_zeros_and_rejects_invalid_format(self) -> None:
        self.assertEqual(AfterSalesTicket.model_validate(self.examples[0]["label"]).order_id, "001234")
        for value in ("订单123", "12 34", "123\n"):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                AfterSalesTicket.model_validate({**self.examples[0]["label"], "order_id": value})

    def test_malformed_json_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            AfterSalesTicket.model_validate_json('{"intent":')

    def test_json_schema_contract(self) -> None:
        schema = AfterSalesTicket.model_json_schema()
        self.assertEqual(set(schema["properties"]), FIELDS)
        self.assertEqual(set(schema["required"]), FIELDS)
        self.assertFalse(schema["additionalProperties"])
        exported = json.loads((ROOT / "data/ticket.schema.json").read_text(encoding="utf-8"))
        self.assertEqual(exported, schema)

    def test_grammar_preflight_fixture_expectations(self) -> None:
        # Fixtures exercise the real Linux grammar separately; this only checks
        # that their expected contract agrees with current Windows Pydantic.
        from evaluate import strict_json
        from evaluate_structured import schema_check_cases

        cases = schema_check_cases()
        self.assertEqual(len(cases), len({case['id'] for case in cases}))
        for case in cases:
            with self.subTest(case=case['id']):
                try:
                    AfterSalesTicket.model_validate(strict_json(case['raw']))
                    accepted = True
                except (ValueError, TypeError):
                    accepted = False
                self.assertEqual(accepted, case['expected_valid'])

    def test_unicode_whitespace_and_escaped_text(self) -> None:
        base = self.examples[0]['label']
        for field in ('product_name', 'reason'):
            for text in ('\u00a0', '\u2003', '\u3000', '\u0085', '\t\n'):
                with self.subTest(field=field, text=repr(text)), self.assertRaises(ValidationError):
                    AfterSalesTicket.model_validate(base | {field: text})
            for text in (' 中 ', '\t中\n', '耳机"左耳"\\配件', '\u200b', '😀'):
                with self.subTest(field=field, text=repr(text)):
                    self.assertEqual(AfterSalesTicket.model_validate(base | {field: text}).model_dump()[field], text)


if __name__ == "__main__":
    unittest.main()
