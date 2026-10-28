"""Training-data and disk-policy tests; no GPU training or simulated model results."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from slimllm.training.baseline import adapter_provenance
from slimllm.evaluation.evaluate import comparison
from slimllm.runtime import ROOT
from slimllm.training.train import TrainConfig, encode_example, frozen_data, padded_batch, train, trainer_options


class TokenizerFixture:
    """Character IDs test masking mechanics only; real tokenizer is checked by prepare."""
    eos_token_id = 999

    def apply_chat_template(self, messages, **kwargs):
        return 'prompt:' + messages[-1]['content']

    def encode(self, text, **kwargs):
        return [ord(character) for character in text]


class TrainingTests(unittest.TestCase):
    def test_prompt_and_padding_do_not_contribute_to_loss(self):
        row = {'id': 'fixture', 'text': 'a', 'label': {'field': 'value'}}
        encoded = encode_example(row, 'system', TokenizerFixture(), 1024)
        boundary = len('prompt:a')
        self.assertEqual(encoded['labels'][:boundary], [-100] * boundary)
        self.assertEqual(encoded['labels'][boundary:], encoded['input_ids'][boundary:])
        self.assertEqual(encoded['labels'][-1], 999)
        short = {'input_ids': [4], 'attention_mask': [1], 'labels': [4]}
        batch = padded_batch([encoded, short], 0)
        self.assertEqual(batch['attention_mask'][1][1:], [0] * (len(encoded['input_ids']) - 1))
        self.assertEqual(batch['labels'][1][1:], [-100] * (len(encoded['input_ids']) - 1))

    def test_overlong_labels_are_rejected_not_truncated(self):
        row = {'id': 'too-long', 'text': 'a' * 1000, 'label': {'field': 'value'}}
        with self.assertRaisesRegex(ValueError, 'no label truncation'):
            encode_example(row, 'system', TokenizerFixture(), 768)

    def test_collator_rejects_misaligned_labels(self):
        with self.assertRaises(ValueError):
            padded_batch([{'input_ids': [1, 2], 'attention_mask': [1, 1], 'labels': [1]}], 0)

    def test_training_uses_only_frozen_train_and_validation_ids(self):
        rows, prompt, provenance = frozen_data(TrainConfig())
        split = json.loads((ROOT / 'data/generated/stage1/splits.json').read_text(encoding='utf-8'))
        self.assertEqual(len(rows['train']), 399)
        self.assertEqual(len(rows['validation']), 50)
        selected = {row['id'] for items in rows.values() for row in items}
        self.assertFalse(selected & set(split['ids']['test']))
        self.assertTrue(prompt)
        self.assertEqual(provenance['train_ids'], split['ids']['train'])

    def test_source_tampering_stops_preparation(self):
        with patch('slimllm.training.train.file_hash', return_value='changed'):
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                frozen_data(TrainConfig())

    def test_configuration_cannot_target_outside_workspace(self):
        with self.assertRaises(ValueError):
            replace(TrainConfig(), output_path='../outside').validate()
        with self.assertRaises(ValueError):
            replace(TrainConfig(), max_length=4096).validate()

    def test_no_intermediate_checkpoint_or_external_reporting(self):
        options = trainer_options(TrainConfig())
        self.assertEqual(options['save_strategy'], 'no')
        self.assertFalse(options['load_best_model_at_end'])
        self.assertFalse(options['push_to_hub'])
        self.assertEqual(options['report_to'], 'none')
        self.assertEqual(options['dataloader_num_workers'], 0)
        self.assertTrue(options['gradient_checkpointing'])

    def test_missing_dependencies_stop_before_loading_model(self):
        with patch('slimllm.training.train.dependency_status', return_value={'missing': ['peft']}):
            with self.assertRaisesRegex(ValueError, 'Missing training dependencies'):
                train(TrainConfig())

    def test_adapter_requires_valid_config_and_weights(self):
        directory = ROOT / '.cache/tmp'
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            path = Path(temporary)
            relative = str(path.relative_to(ROOT))
            with self.assertRaises(ValueError):
                adapter_provenance(relative)
            (path / 'adapter_config.json').write_text('{"peft_type":"PREFIX_TUNING"}', encoding='utf-8')
            (path / 'adapter_model.safetensors').write_bytes(b'file-validation-fixture-only')
            with self.assertRaisesRegex(ValueError, 'LoRA'):
                adapter_provenance(relative)

    def test_comparison_rejects_changed_prompt(self):
        folder = ROOT / 'reports/baseline'
        metrics = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
        manifest = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
        with self.assertRaisesRegex(ValueError, 'different data, prompt'):
            comparison(metrics, metrics, manifest, {**manifest, 'prompt': 'changed'})
        result = comparison(metrics, metrics, manifest, manifest)
        self.assertTrue(all(value == 0 for value in result['delta_percentage_points'].values()))


if __name__ == '__main__':
    unittest.main()
