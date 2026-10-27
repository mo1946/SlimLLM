"""Read-only local vLLM prerequisite check; never install, download or load a model."""

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys

import psutil

from runtime import ROOT, configure, file_hash, project_path, save_json


def command(args: list[str]) -> dict:
    try:
        process_env = os.environ.copy()
        # Prefer native default output; UTF-16 is detected explicitly below.
        process_env.pop('WSL_UTF8', None)
        result = subprocess.run(args, capture_output=True, timeout=15,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), env=process_env)
        def decode(raw):
            if b'\x00' in raw:
                return raw.decode('utf-16-le', errors='replace').strip()
            try:
                return raw.decode('utf-8').strip()
            except UnicodeDecodeError:
                return raw.decode('mbcs' if sys.platform == 'win32' else 'utf-8', errors='replace').strip()
        return {'command': args, 'exit_code': result.returncode,
                'stdout': decode(result.stdout), 'stderr': decode(result.stderr)}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'command': args, 'error': type(exc).__name__}


def collect() -> dict:
    model = ROOT / '.cache/huggingface/qwen3-1.7b'
    adapter = ROOT / 'outputs/qlora/final_adapter'
    config = json.loads((model / 'config.json').read_text(encoding='utf-8'))
    lora = json.loads((adapter / 'adapter_config.json').read_text(encoding='utf-8'))
    from safetensors import safe_open
    import torch
    with safe_open(str(adapter / 'adapter_model.safetensors'), framework='pt', device='cpu', backend='pread') as stream:
        names = list(stream.keys())
        standard_ab = all('.lora_A.weight' in name or '.lora_B.weight' in name for name in names)
        rank_shapes = all(stream.get_slice(name).get_shape()[0 if '.lora_A.' in name else 1] == 8 for name in names)
    # FP16 KV storage estimate only, excluding allocation overhead and all model/runtime memory.
    kv_per_token = 2 * config['num_hidden_layers'] * config['num_key_value_heads'] * config['head_dim'] * 2
    volumes = {}
    for path in (('B:\\', 'C:\\') if sys.platform == 'win32' else (str(ROOT),)):
        try:
            volumes[path] = dict(shutil.disk_usage(path)._asdict())
        except OSError as exc:
            volumes[path] = {'error': type(exc).__name__}
    wsl = {}
    if sys.platform == 'win32':
        for flag in (['--version'], ['--status'], ['--list', '--verbose']):
            wsl[' '.join(flag)] = command(['wsl.exe', *flag])
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Software\Microsoft\Windows\CurrentVersion\Lxss') as key:
                wsl['registered_distribution_count'] = winreg.QueryInfoKey(key)[0]
        except OSError as exc:
            wsl['registered_distribution_count'] = None
            wsl['registry_error'] = type(exc).__name__
    return {
        'platform': platform.platform(), 'windows_version': platform.win32_ver(),
        'python': sys.version, 'interpreter': sys.executable,
        'ram': dict(psutil.virtual_memory()._asdict()), 'volumes': volumes, 'wsl': wsl,
        'gpu_query': command(['nvidia-smi', '--query-gpu=name,driver_version,compute_cap,memory.total,memory.used,memory.free', '--format=csv,noheader,nounits']),
        'nvcc_on_path': shutil.which('nvcc'), 'torch': torch.__version__,
        'torch_built_cuda': torch.version.cuda, 'cuda_available_in_this_python': torch.cuda.is_available(),
        'vllm_installed_in_this_python': importlib.util.find_spec('vllm') is not None,
        'model': {'architectures':config['architectures'], 'model_type':config['model_type'],
            'file_bytes':sum(p.stat().st_size for p in model.rglob('*') if p.is_file()),
            'path':str(model), 'wsl_path':'/mnt/b/python/SlimLLM/.cache/huggingface/qwen3-1.7b'},
        'adapter': {'path':str(adapter), 'wsl_path':'/mnt/b/python/SlimLLM/outputs/qlora/final_adapter',
            'peft_type':lora['peft_type'], 'r':lora['r'], 'alpha':lora['lora_alpha'],
            'target_modules':sorted(lora['target_modules']), 'bias':lora['bias'],
            'use_dora':lora['use_dora'], 'modules_to_save':lora['modules_to_save'],
            'base_model_metadata':lora['base_model_name_or_path'], 'tensor_count':len(names),
            'standard_AB_keys':standard_ab, 'rank8_shapes':rank_shapes,
            'weight_bytes':(adapter/'adapter_model.safetensors').stat().st_size,
            'weights_sha256':file_hash(adapter/'adapter_model.safetensors')},
        'fp16_kv_estimate':{'bytes_per_token_per_sequence':kv_per_token,
            'max_model_len':2560, 'sequence_bytes':{str(n):kv_per_token*2560*n for n in (2,4,8)},
            'note':'Capacity estimate, not measured vLLM memory; runtime preallocates according to its KV budget.'},
        'preserved_code_sha256':{name:file_hash(ROOT/name) for name in ('api.py','request_queue.py','inference.py','benchmark.py','schema.py')},
        'vllm_runtime_validated':False,
        'new_installations_executed':False, 'model_inference_executed':False,
    }


def main():
    configure()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='reports/vllm_preflight.json')
    args = parser.parse_args()
    result = collect()
    save_json(project_path(args.output), result)
    print(f'Read-only preflight saved: {args.output}; no install/download/model inference.')
    print(json.dumps({'volumes':result['volumes'],'gpu':result['gpu_query'],
                      'wsl_exit_codes':{k:v.get('exit_code') for k,v in result['wsl'].items() if isinstance(v,dict)}},indent=2))


if __name__ == '__main__':
    main()
