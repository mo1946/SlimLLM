"""Stage two: offline preparation and explicitly invoked NF4 QLoRA training."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import time

from slimllm.runtime import ROOT, configure, file_hash, project_path, read_jsonl, save_json

REQUIRED = ('torch', 'transformers', 'safetensors', 'peft', 'trl', 'bitsandbytes', 'accelerate', 'datasets')
TARGETS = ('q_proj', 'k_proj', 'v_proj', 'o_proj')


@dataclass(frozen=True)
class TrainConfig:
    model_path: str = '.cache/huggingface/qwen3-1.7b'
    data_path: str = 'data/generated/stage1'
    baseline_path: str = 'reports/baseline'
    output_path: str = 'outputs/qlora'
    max_length: int = 1024
    epochs: int = 3
    learning_rate: float = 2e-4
    gradient_accumulation_steps: int = 8
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    seed: int = 42

    def validate(self) -> None:
        if self.max_length not in (768, 1024):
            raise ValueError('Initial max_length must be 768 or 1024.')
        if self.epochs < 1 or not 0 < self.learning_rate <= 1e-3:
            raise ValueError('Invalid epochs or learning rate.')
        if self.lora_r != 8 or self.gradient_accumulation_steps < 1:
            raise ValueError('This MVP requires rank 8 and positive accumulation steps.')
        for value in (self.model_path, self.data_path, self.baseline_path, self.output_path):
            project_path(value)


def dependency_status() -> dict:
    versions = {}
    for name in REQUIRED:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {'versions': versions, 'missing': [name for name, version in versions.items() if version is None]}


def frozen_data(config: TrainConfig) -> tuple[dict, str, dict]:
    folder = project_path(config.data_path)
    source = folder / 'records.jsonl'
    split_path = folder / 'splits.json'
    splits = json.loads(split_path.read_text(encoding='utf-8'))
    baseline = json.loads((project_path(config.baseline_path) / 'run.json').read_text(encoding='utf-8'))
    digest = file_hash(source)
    if digest != splits['source_sha256'] or digest != baseline['source_sha256']:
        raise ValueError('Frozen source hash mismatch.')
    if file_hash(split_path) != baseline['splits_sha256']:
        raise ValueError('Frozen split hash mismatch.')
    groups = splits['ids']
    all_ids = [identifier for name in ('train', 'validation', 'test') for identifier in groups[name]]
    rows = read_jsonl(source)
    if len(all_ids) != len(set(all_ids)) or len(rows) != len(all_ids):
        raise ValueError('Split IDs overlap or dataset contains duplicate/missing records.')
    by_id = {row['id']: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != set(all_ids):
        raise ValueError('Split IDs and source IDs do not match.')
    selected = {name: [by_id[identifier] for identifier in groups[name]] for name in ('train', 'validation')}
    return selected, baseline['prompt'], {
        'source_sha256': digest, 'splits_sha256': file_hash(split_path),
        'prompt_sha256': hashlib.sha256(baseline['prompt'].encode('utf-8')).hexdigest(),
        'train_ids': groups['train'], 'validation_ids': groups['validation'],
    }


def encode_example(row: dict, prompt: str, tokenizer, max_length: int) -> dict:
    messages = [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': row['text']}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
    answer = json.dumps(row['label'], ensure_ascii=False, separators=(',', ':'))
    answer_ids = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
    if not answer_ids or tokenizer.eos_token_id is None:
        raise ValueError('Tokenizer must define an end-of-turn token.')
    ids = prompt_ids + answer_ids
    if len(ids) > max_length:
        raise ValueError(f"{row['id']}: {len(ids)} tokens exceeds {max_length}; no label truncation allowed.")
    return {'input_ids': ids, 'attention_mask': [1] * len(ids),
            'labels': [-100] * len(prompt_ids) + answer_ids}


def padded_batch(features: list[dict], pad_token_id: int) -> dict:
    if not features:
        raise ValueError('Cannot collate an empty batch.')
    length = max(len(row['input_ids']) for row in features)
    result = {'input_ids': [], 'attention_mask': [], 'labels': []}
    for row in features:
        size = len(row['input_ids'])
        if not (size == len(row['labels']) == len(row['attention_mask'])):
            raise ValueError('Misaligned token labels or mask.')
        result['input_ids'].append(row['input_ids'] + [pad_token_id] * (length - size))
        result['attention_mask'].append(row['attention_mask'] + [0] * (length - size))
        result['labels'].append(row['labels'] + [-100] * (length - size))
    return result


class CompletionCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]):
        import torch
        return {name: torch.tensor(value, dtype=torch.long)
                for name, value in padded_batch(features, self.pad_token_id).items()}


def prepare(config: TrainConfig) -> tuple[dict, object, dict]:
    from transformers import AutoTokenizer
    config.validate()
    rows, prompt, provenance = frozen_data(config)
    tokenizer = AutoTokenizer.from_pretrained(str(project_path(config.model_path)), local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = {name: [encode_example(row, prompt, tokenizer, config.max_length) for row in items]
               for name, items in rows.items()}
    summary = {'configuration': asdict(config), 'provenance': provenance,
        'dependency_status': dependency_status(), 'training_executed': False,
        'sequence_lengths': {name: {'count': len(items), 'min': min(len(r['input_ids']) for r in items),
            'max': max(len(r['input_ids']) for r in items),
            'answer_tokens_min': min(sum(t != -100 for t in r['labels']) for r in items)} for name, items in encoded.items()},
        'test_data_used': False, 'loss': 'assistant JSON and EOS only; prompt/padding labels are -100',
        'quantization': {'bits': 4, 'type': 'nf4', 'double_quant': True, 'compute_dtype': 'bfloat16'},
        'lora': {'r': config.lora_r, 'alpha': config.lora_alpha, 'target_modules': list(TARGETS)},
        'batch_size': 1, 'gradient_checkpointing': True, 'intermediate_checkpoints': False}
    return encoded, tokenizer, summary


def trainer_options(config: TrainConfig) -> dict:
    return dict(output_dir=str(project_path(config.output_path)), max_length=config.max_length,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.epochs, learning_rate=config.learning_rate,
        optim='adamw_torch', lr_scheduler_type='cosine', warmup_steps=0.03,
        bf16=True, fp16=False, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False}, use_cache=False,
        save_strategy='no', load_best_model_at_end=False,
        eval_strategy='epoch', prediction_loss_only=True, eval_accumulation_steps=1,
        report_to='none', push_to_hub=False, logging_steps=10,
        dataloader_num_workers=0, dataloader_pin_memory=False,
        packing=False, padding_free=False, torch_compile=False,
        dataset_kwargs={'skip_prepare_dataset': True}, completion_only_loss=False,
        remove_unused_columns=False, seed=config.seed, data_seed=config.seed)


def load_training_model(config: TrainConfig):
    status = dependency_status()
    if status['missing']:
        raise ValueError('Missing training dependencies: ' + ', '.join(status['missing']) + '. Download approval required.')
    import torch
    from bitsandbytes.nn import Linear4bit
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig, GenerationConfig, Qwen3ForCausalLM, set_seed
    from slimllm.training.baseline import load_gpu_state

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError('This configuration requires a CUDA GPU supporting BF16; no silent fallback.')
    set_seed(config.seed)
    torch.cuda.reset_peak_memory_stats()
    model_config, state = load_gpu_state(project_path(config.model_path))
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = Qwen3ForCausalLM.from_pretrained(None, config=model_config, state_dict=state,
        device_map={'': 0}, quantization_config=quantization, dtype=torch.float16,
        generation_config=GenerationConfig.from_pretrained(str(project_path(config.model_path)), local_files_only=True),
        local_files_only=True, trust_remote_code=False, attn_implementation='sdpa')
    del state
    torch.cuda.empty_cache()
    quantized = [module for module in model.modules() if isinstance(module, Linear4bit)]
    if not quantized or any(m.weight.quant_state is None or m.weight.quant_type != 'nf4' for m in quantized):
        raise ValueError('NF4 quantization was not applied completely.')
    model.config.name_or_path = str(project_path(config.model_path))
    model.name_or_path = str(project_path(config.model_path))
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False})
    model = get_peft_model(model, LoraConfig(r=config.lora_r, lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout, target_modules=list(TARGETS), bias='none', task_type='CAUSAL_LM'))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not trainable or any(p.requires_grad and 'lora_' not in name for name, p in model.named_parameters()):
        raise ValueError('Only LoRA parameters may be trainable.')
    return model, {'trainable_parameters': trainable, 'quantized_linear_layers': len(quantized),
                   'loading_peak_allocated_bytes': torch.cuda.max_memory_allocated()}


def train(config: TrainConfig) -> None:
    status = dependency_status()
    if status['missing']:
        raise ValueError('Missing training dependencies: ' + ', '.join(status['missing']) + '. Download approval required.')
    destination = project_path(config.output_path)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Training output is not empty; use a fresh directory. No overwrites or automatic resume.')
    encoded, tokenizer, manifest = prepare(config)
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    destination.mkdir(parents=True, exist_ok=True)
    save_json(destination / 'run.json', manifest)
    start = time.perf_counter()
    model, loading = load_training_model(config)
    trainer = SFTTrainer(model=model, args=SFTConfig(**trainer_options(config)),
        train_dataset=Dataset.from_list(encoded['train']), eval_dataset=Dataset.from_list(encoded['validation']),
        processing_class=tokenizer, data_collator=CompletionCollator(tokenizer.pad_token_id))
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train()
    final_adapter = destination / 'final_adapter'
    model.save_pretrained(str(final_adapter), safe_serialization=True, save_embedding_layers=False)
    # Do not use trainer.save_model/save_state, merge_and_unload or save the tokenizer/base model.
    allowed = {'adapter_model.safetensors', 'adapter_config.json', 'README.md'}
    unexpected = {p.name for p in final_adapter.iterdir()} - allowed
    if unexpected or not (final_adapter / 'adapter_model.safetensors').exists():
        raise ValueError('Unexpected files in final adapter; inspect without automatic deletion.')
    save_json(destination / 'training_metrics.json', {'training_executed': True,
        'metrics': result.metrics, 'elapsed_seconds': time.perf_counter() - start,
        **loading,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'memory_scope': 'peak_* cover trainer.train(); loading peak is recorded separately',
        'adapter_bytes': sum(p.stat().st_size for p in final_adapter.iterdir() if p.is_file()),
        'versions': status['versions']})
    manifest.update(training_executed=True, status='completed')
    (destination / 'run.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Training complete; final adapter only: {final_adapter}')


def main() -> None:
    configure()
    for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE', 'HF_DEACTIVATE_ASYNC_LOAD'):
        os.environ[name] = '1'
    os.environ['USE_HUB_KERNELS'] = '0'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'prepare', 'check-model', 'train'])
    parser.add_argument('--max-length', type=int, choices=[768, 1024], default=1024)
    parser.add_argument('--output', default='outputs/qlora')
    parser.add_argument('--report', default='reports/stage2_preflight.json')
    args = parser.parse_args()
    config = TrainConfig(max_length=args.max_length, output_path=args.output)
    try:
        if args.command == 'check':
            print(json.dumps(dependency_status(), indent=2))
        elif args.command == 'prepare':
            _, _, summary = prepare(config)
            save_json(project_path(args.report), summary)
            print(json.dumps({key: summary[key] for key in ('sequence_lengths', 'dependency_status', 'training_executed')}, indent=2))
        elif args.command == 'check-model':
            import torch
            config.validate()
            model, loading = load_training_model(config)
            model.eval()
            summary = {**loading, 'model_loaded': True, 'training_executed': False,
                'model_memory_bytes': model.get_memory_footprint(),
                'loading_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'note': 'Load/quantization/LoRA checks only; no optimizer, forward pass or training step.'}
            save_json(project_path(args.report), summary)
            print(json.dumps(summary, indent=2))
        else:
            train(config)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f'{type(exc).__name__}: {exc}\n')


if __name__ == '__main__':
    main()
