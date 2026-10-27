"""Download only when explicitly requested; otherwise run Qwen entirely offline."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

from runtime import append_jsonl, configure, extraction_prompt, file_hash, project_path, read_jsonl, save_json

MODEL_ID = 'Qwen/Qwen3-1.7B'
REVISION = '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'


def read_range(url: str, start: int, end: int, size: int) -> bytes:
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
            with urllib.request.urlopen(request, timeout=90) as response:
                if response.status == 206:
                    if response.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                        raise ValueError('Server returned the wrong byte range.')
                elif not (response.status == 200 and start == 0 and end == size - 1):
                    raise ValueError('Server ignored range request.')
                block = response.read(end - start + 2)
                if len(block) != end - start + 1:
                    raise ValueError('Incomplete download range; refusing to append.')
                return block
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(2)
    raise RuntimeError('Range download failed.')


def download(args: argparse.Namespace) -> None:
    location = project_path(args.model_path)
    location.mkdir(parents=True, exist_ok=True)
    metadata_url = f'https://huggingface.co/api/models/{MODEL_ID}/revision/{REVISION}?blobs=true'
    with urllib.request.urlopen(metadata_url, timeout=30) as response:
        metadata = json.load(response)
    if metadata['sha'] != REVISION:
        raise ValueError('Model revision mismatch.')
    for item in metadata['siblings']:
        name = item['rfilename']
        if '/' in name or not (name.endswith(('.json', '.safetensors')) or name in ('merges.txt', 'vocab.json', 'LICENSE')):
            continue
        destination = location / name
        size = item['size']
        expected_hash = item.get('lfs', {}).get('sha256')

        def valid(path: Path) -> bool:
            if not path.exists() or path.stat().st_size != size:
                return False
            if expected_hash:
                with path.open('rb') as stream:
                    return hashlib.file_digest(stream, 'sha256').hexdigest() == expected_hash
            return True

        if valid(destination):
            print(f'Reuse {name}', flush=True)
            continue
        partial = destination.with_name(name + '.incomplete')
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > size:
            raise ValueError(f'Oversized partial file: {name}')
        url = f'https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}'
        with ThreadPoolExecutor(max_workers=4) as pool:
            while offset < size:
                chunk = 32 * 1024 * 1024
                ranges = [(start, min(start + chunk, size) - 1)
                          for start in range(offset, min(offset + 4 * chunk, size), chunk)]
                for block in pool.map(lambda span: read_range(url, span[0], span[1], size), ranges):
                    with partial.open('ab') as stream:
                        stream.write(block)
                        stream.flush()
                        os.fsync(stream.fileno())
                    offset += len(block)
                    print(f'{name}: {offset}/{size} bytes', flush=True)
        if not valid(partial):
            raise ValueError(f'SHA256 validation failed for {name}; partial preserved for inspection.')
        os.replace(partial, destination)
    if not (location / 'download_manifest.json').exists():
        save_json(location / 'download_manifest.json', {'model': MODEL_ID, 'revision': REVISION,
            'weights_sha256': {item['rfilename']: item['lfs']['sha256'] for item in metadata['siblings'] if item['rfilename'].endswith('.safetensors')}})
    print(f'Model verified and saved once to {location}')


def load_gpu_state(model_path: Path):
    """Read the existing Qwen checkpoint without mmap or duplicate tied tensors."""
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig

    model_config = AutoConfig.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
    index = json.loads((model_path / 'model.safetensors.index.json').read_text(encoding='utf-8'))
    state = {}
    for filename in sorted(set(index['weight_map'].values())):
        with safe_open(str(model_path / filename), framework='pt', device='cpu', backend='pread') as shard:
            for name in shard.keys():
                if model_config.tie_word_embeddings and name == 'lm_head.weight':
                    continue
                state[name] = shard.get_tensor(name).to(device='cuda', dtype=torch.float16)
        print(f'Read {filename}: {len(state)} tensors loaded', flush=True)
    if model_config.tie_word_embeddings:
        state['lm_head.weight'] = state['model.embed_tokens.weight']
    return model_config, state


def adapter_provenance(value: str) -> dict:
    folder = project_path(value)
    configuration = folder / 'adapter_config.json'
    weights = folder / 'adapter_model.safetensors'
    if not configuration.is_file() or not weights.is_file():
        raise ValueError('Adapter config or safetensors weights missing.')
    if json.loads(configuration.read_text(encoding='utf-8')).get('peft_type') != 'LORA':
        raise ValueError('Expected a LoRA adapter.')
    return {'path': str(folder), 'config_sha256': file_hash(configuration), 'weights_sha256': file_hash(weights)}


def load_fp16_model(model_path: Path, adapter_path: str | None = None):
    """Shared offline loader for evaluation and the single-process API."""
    import torch
    from transformers import AutoTokenizer, GenerationConfig, Qwen3ForCausalLM

    if not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; no implicit CPU fallback.')
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
    model_config, state = load_gpu_state(model_path)
    with torch.device('meta'):
        model = Qwen3ForCausalLM(model_config)
    model.load_state_dict(state, strict=True, assign=True)
    with torch.device('cuda'):
        model.model.rotary_emb = type(model.model.rotary_emb)(model_config)
    model.tie_weights()
    model.set_attn_implementation('sdpa')
    model.generation_config = GenerationConfig.from_pretrained(str(model_path), local_files_only=True)
    model.eval()
    if (set(state) != set(model.state_dict())
            or any(p.is_meta or p.device.type != 'cuda' for p in model.parameters())
            or any(b.is_meta for b in model.buffers())):
        raise ValueError('Model parameters were not completely loaded.')
    del state
    torch.cuda.empty_cache()
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False, local_files_only=True).eval()
    return model, tokenizer


def run(args: argparse.Namespace) -> None:
    # Must be set before Transformers/Hugging Face imports. Never auto-download.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DEACTIVATE_ASYNC_LOAD'] = '1'
    os.environ['USE_HUB_KERNELS'] = '0'
    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = project_path(args.model_path)
    if not (model_path / 'config.json').exists():
        raise ValueError('Local Qwen3 model is missing. Download requires user approval.')
    folder = project_path(args.data)
    source = folder / 'records.jsonl'
    splits = json.loads((folder / 'splits.json').read_text(encoding='utf-8'))
    if splits['source_sha256'] != file_hash(source):
        raise ValueError('Dataset changed after split freeze.')
    rows = {row['id']: row for row in read_jsonl(source)}
    ids = splits['ids'][args.split]
    output = project_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    prompt = extraction_prompt()
    config = {'model': MODEL_ID, 'revision': REVISION, 'model_path': str(model_path.resolve()),
              'source_sha256': file_hash(source), 'splits_sha256': file_hash(folder / 'splits.json'),
              'prompt': prompt, 'ids': ids, 'split': args.split,
              'max_new_tokens': 384, 'do_sample': False, 'enable_thinking': False,
              'dtype': 'float16', 'seed': 42}
    if args.adapter:
        if output == project_path('reports/baseline'):
            raise ValueError('Adapter evaluation requires a separate --output; preserve the original baseline.')
        config['adapter'] = adapter_provenance(args.adapter)
    manifest = output / 'run.json'
    if manifest.exists():
        if json.loads(manifest.read_text(encoding='utf-8')) != config:
            raise ValueError('Baseline settings changed; use a new report directory.')
    else:
        save_json(manifest, config)
    predictions = output / 'predictions.jsonl'
    previous = read_jsonl(predictions)
    done = {row['id'] for row in previous}
    if len(done) != len(previous) or not done.issubset(ids):
        raise ValueError('Duplicate or unexpected prediction IDs.')
    if len(done) == len(ids):
        print('All selected samples already have real model outputs.')
        return
    print('Loading existing PyTorch/Transformers runtime...', flush=True)
    import torch
    import transformers

    if not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable in this interpreter; no implicit CPU fallback.')
    print(f'CUDA ready: {torch.cuda.get_device_name(0)}', flush=True)
    model, tokenizer = load_fp16_model(model_path, config['adapter']['path'] if args.adapter else None)
    tied_embeddings = model.config.tie_word_embeddings
    if not (output / 'environment.json').exists():
        save_json(output / 'environment.json', {'torch': torch.__version__, 'transformers': transformers.__version__,
            'gpu': torch.cuda.get_device_name(0), 'loader': 'safetensors pread -> FP16 GPU tensors -> PyTorch meta/assign',
            'tied_embeddings': tied_embeddings, 'async_loading': False, 'hub_kernels': False})
    print('Model loaded; starting frozen evaluation.', flush=True)
    torch.cuda.reset_peak_memory_stats()
    for identifier in ids:
        if identifier in done:
            continue
        row = rows[identifier]
        messages = [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': row['text']}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(rendered, return_tensors='pt', add_special_tokens=False).to('cuda')
        if inputs.input_ids.shape[-1] > 3072:
            raise ValueError('Prompt exceeds 3072 tokens; refusing silent truncation.')
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=384, do_sample=False,
                pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        tokens = generated[0, inputs.input_ids.shape[-1]:]
        append_jsonl(predictions, {'id': identifier, 'raw_output': tokenizer.decode(tokens, skip_special_tokens=True),
            'generated_tokens': len(tokens), 'elapsed_seconds': time.perf_counter() - start,
            'hit_token_limit': len(tokens) >= 384,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'torch': torch.__version__, 'transformers': transformers.__version__})
        done.add(identifier)
        print(f'Baseline {len(done)}/{len(ids)} {identifier}', flush=True)


def main() -> None:
    configure()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['download', 'run'])
    parser.add_argument('--model-path', default='.cache/huggingface/qwen3-1.7b')
    parser.add_argument('--data', default='data/generated/stage1')
    parser.add_argument('--split', choices=['validation', 'test'], default='test')
    parser.add_argument('--output', default='reports/baseline')
    parser.add_argument('--adapter', help='Local LoRA adapter; attaches without merging or copying the base model.')
    args = parser.parse_args()
    try:
        (download if args.command == 'download' else run)(args)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f'{type(exc).__name__}: {exc}\n')


if __name__ == '__main__':
    main()
