"""Bounded, reproducible asyncio/httpx benchmark of the existing local /extract API."""

import argparse
import asyncio
from collections import Counter
from datetime import datetime
import importlib.metadata as metadata
import json
import math
import platform
import subprocess
import time
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import httpx
from pydantic import ValidationError

from evaluate import FIELDS, strict_json
from runtime import ROOT, append_jsonl, configure, file_hash, project_path, read_jsonl, save_json
from schema import AfterSalesTicket


def latency_stats(values: list[float]) -> dict:
    """Linear interpolation at (n - 1) * p; empty sets have no percentile."""
    if not values:
        return dict.fromkeys(('count', 'mean_ms', 'p50_ms', 'p95_ms', 'p99_ms'), None) | {'count': 0}
    ordered = sorted(values)

    def percentile(p):
        position = (len(ordered) - 1) * p
        low, high = math.floor(position), math.ceil(position)
        return (ordered[low] + (ordered[high] - ordered[low]) * (position - low)) * 1000

    return {'count': len(values), 'mean_ms': sum(values) / len(values) * 1000,
            **{f'p{int(p * 100)}_ms': percentile(p) for p in (.5, .95, .99)}}


def summarize(rows: list[dict], elapsed: float) -> dict:
    count = len(rows)
    successful = [r for r in rows if r['http_success']]
    valid = [r for r in rows if r['schema_valid']]
    generated = [r for r in successful if r.get('generated_tokens') is not None]
    timed = [r for r in generated if r.get('generation_seconds') is not None]
    fields = {name: sum(name in r['correct_fields'] for r in rows) for name in FIELDS}
    tokens = sum(r['generated_tokens'] for r in generated)
    generation_seconds = sum(r['generation_seconds'] for r in timed)
    ratio = lambda n, d: n / d if d else None
    return {
        'requests': count, 'elapsed_seconds': elapsed,
        'http_success_count': len(successful), 'http_success_rate': ratio(len(successful), count),
        'attempt_rps': ratio(count, elapsed), 'success_rps': ratio(len(successful), elapsed),
        'status_counts': dict(Counter(str(r['status']) for r in rows)),
        'error_counts': dict(Counter(r['error'] for r in rows if r['error'])),
        'latency_all': latency_stats([r['elapsed_seconds'] for r in rows]),
        'latency_http_success': latency_stats([r['elapsed_seconds'] for r in successful]),
        'latency_failed': latency_stats([r['elapsed_seconds'] for r in rows if not r['http_success']]),
        'queue_wait': latency_stats([r['queue_wait_seconds'] for r in successful if 'queue_wait_seconds' in r]),
        'json_valid_count': sum(r['json_valid'] for r in rows),
        'json_valid_rate_all': ratio(sum(r['json_valid'] for r in rows), count),
        'json_valid_rate_http_success': ratio(sum(r['json_valid'] for r in rows), len(successful)),
        'schema_valid_count': len(valid), 'schema_valid_rate_all': ratio(len(valid), count),
        'schema_valid_rate_http_success': ratio(len(valid), len(successful)),
        'field_accuracy_all': {name: ratio(n, count) for name, n in fields.items()},
        'field_macro_accuracy_all': ratio(sum(fields.values()), count * len(FIELDS)),
        'field_macro_accuracy_schema_valid': ratio(sum(fields.values()), len(valid) * len(FIELDS)),
        'exact_match_count': sum(r['exact_match'] for r in rows),
        'exact_match_rate_all': ratio(sum(r['exact_match'] for r in rows), count),
        'exact_match_rate_schema_valid': ratio(sum(r['exact_match'] for r in rows), len(valid)),
        'successful_sample_ids': [r['id'] for r in successful],
        'generated_tokens_measured': tokens, 'generation_seconds_measured': generation_seconds,
        'generation_metrics_response_count': len(timed),
        'model_generated_tokens_per_generation_second': ratio(tokens, generation_seconds) if len(timed) == len(generated) else None,
        'generated_tokens_per_phase_second': ratio(tokens, elapsed),
        'peak_allocated_bytes': max((r.get('peak_allocated_bytes', 0) for r in successful), default=0) or None,
        'peak_reserved_bytes': max((r.get('peak_reserved_bytes', 0) for r in successful), default=0) or None,
    }


async def request_one(client: httpx.AsyncClient, row: dict, index: int) -> dict:
    result = {'index': index, 'id': row['id'], 'status': None, 'http_success': False,
              'json_valid': False, 'schema_valid': False, 'correct_fields': [],
              'exact_match': False, 'error': None}
    start = time.perf_counter()
    try:
        response = await client.post('/extract', json={'text': row['text']})
        result['elapsed_seconds'] = time.perf_counter() - start
    except httpx.RequestError as exc:
        result.update(elapsed_seconds=time.perf_counter() - start, error=type(exc).__name__)
        return result
    result.update(status=response.status_code, http_success=200 <= response.status_code < 300,
                  raw_response=response.text, cache=response.headers.get('X-SlimLLM-Cache'))
    if not result['http_success']:
        try:
            detail = response.json().get('detail', {})
            result['error'] = detail.get('code', f'HTTP_{response.status_code}') if isinstance(detail, dict) else f'HTTP_{response.status_code}'
        except (ValueError, AttributeError):
            result['error'] = f'HTTP_{response.status_code}'
        return result
    try:
        parsed = strict_json(response.text)
        result['json_valid'] = True
        prediction = AfterSalesTicket.model_validate(parsed).model_dump()
        result['schema_valid'] = True
        result['correct_fields'] = [name for name in FIELDS if prediction[name] == row['label'][name]]
        result['exact_match'] = len(result['correct_fields']) == len(FIELDS)
    except (ValueError, TypeError, ValidationError):
        result['error'] = 'invalid_schema' if result['json_valid'] else 'invalid_json'
    for name in ('generated_tokens', 'generation_seconds', 'peak_allocated_bytes', 'peak_reserved_bytes', 'queue_wait_seconds', 'backend_request_seconds'):
        raw = response.headers.get('X-SlimLLM-' + name.replace('_', '-'))
        if raw is not None:
            result[name] = float(raw) if name in ('generation_seconds', 'queue_wait_seconds', 'backend_request_seconds') else int(raw)
    return result


class GPUSampler:
    """Sample whole-card memory using existing nvidia-smi, with no extra dependency."""
    def __init__(self):
        self.samples = []
        self.errors = []
        self.stop = asyncio.Event()
        self.unsafe = False

    async def sample(self):
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                'nvidia-smi', '--id=0', '--query-gpu=memory.used,memory.total',
                '--format=csv,noheader,nounits', stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            output, _ = await asyncio.wait_for(process.communicate(), timeout=5)
            if process.returncode:
                raise RuntimeError('nvidia-smi failed')
            used, total = (int(x.strip()) for x in output.decode().strip().split(','))
            self.samples.append({'monotonic_seconds': time.perf_counter(), 'used_mib': used, 'total_mib': total})
            self.unsafe |= total - used < 512 or used / total >= .95
        except Exception as exc:
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()
            self.errors.append(type(exc).__name__)
            self.unsafe = True

    async def run(self):
        while not self.stop.is_set():
            try:
                await asyncio.wait_for(self.stop.wait(), timeout=.5)
            except asyncio.TimeoutError:
                await self.sample()


def fixed_records(count: int) -> tuple[list[dict], dict]:
    folder = ROOT / 'data/generated/stage1'
    manifest = json.loads((ROOT / 'reports/baseline/run.json').read_text(encoding='utf-8'))
    assert file_hash(folder / 'records.jsonl') == manifest['source_sha256'], 'Frozen data changed'
    assert file_hash(folder / 'splits.json') == manifest['splits_sha256'], 'Frozen split changed'
    splits = json.loads((folder / 'splits.json').read_text(encoding='utf-8'))
    assert splits['ids']['test'] == manifest['ids'], 'Frozen test IDs changed'
    by_id = {r['id']: r for r in read_jsonl(folder / 'records.jsonl')}
    rows = [by_id[name] for name in manifest['ids'][:count]]
    assert len(rows) == count
    for row in rows:
        AfterSalesTicket.model_validate(row['label'])
    return rows, {'source_sha256': manifest['source_sha256'], 'splits_sha256': manifest['splits_sha256'],
                  'ids': [row['id'] for row in rows], 'source': 'DeepSeek synthetic, frozen stage-one test set; not real customer records'}


async def run(args):
    rows, data = fixed_records(args.requests)
    output = project_path(args.output)
    if output.exists():
        raise ValueError('Use a new output directory; prior benchmark evidence is never overwritten.')
    limits = httpx.Limits(max_connections=8, max_keepalive_connections=8, keepalive_expiry=60)
    async with httpx.AsyncClient(base_url=args.url, timeout=httpx.Timeout(60, connect=5),
                                limits=limits, trust_env=False, http2=False) as client:
        health = await client.get('/health')
        health.raise_for_status()
        before = health.json()
        if before['cache'] != 'disabled' or before['busy'] or not before['adapter_loaded']:
            raise ValueError('Benchmark requires ready, idle model and disabled Redis.')
        output.mkdir(parents=True, exist_ok=False)
        gpu = await asyncio.create_subprocess_exec('nvidia-smi', '--id=0',
            '--query-gpu=uuid,name,driver_version,memory.total', '--format=csv,noheader,nounits',
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        gpu_info, _ = await asyncio.wait_for(gpu.communicate(), timeout=5)
        save_json(output / 'run.json', {
            'started_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'data': data,
            'concurrency': args.concurrency, 'requests_per_phase': args.requests,
            'workload': 'Fixed ordered requests in barrier-separated waves of C simultaneous requests; no retries, no pacing, await each complete wave.',
            'warmup_requests': 2, 'request': {'method': 'POST', 'path': '/extract', 'body_keys': ['text']},
            'url': args.url, 'timeout_seconds': 60, 'max_connections': 8, 'http2': False,
            'cache': 'disabled', 'health_before': before,
            'gpu': gpu_info.decode().strip(), 'gpu_sample_interval_seconds': .5,
            'safety': 'Stop future waves/phases on OOM, transport timeout, <512MiB free, >=95% card memory or failed memory probe. Expected model_busy does not trigger retries.',
            'python': platform.python_version(), 'platform': platform.platform(),
            'packages': {name: metadata.version(name) for name in ('httpx', 'fastapi', 'torch', 'transformers', 'peft')},
            'code_sha256': {name: file_hash(ROOT / name) for name in ('benchmark.py', 'api.py', 'inference.py', 'schema.py', 'request_queue.py')},
            'ttft': None, 'ttft_note': 'Existing /extract is non-streaming; TTFT cannot be measured.',
            'latency': 'Client full HTTP response; all requests and HTTP successes reported separately; linear-interpolated percentiles.',
            'quality_denominators': 'HTTP 2xx, strict ticket JSON, schema and exact-label fields tracked separately. HTTP errors never count as valid ticket JSON/schema; failed requests count wrong in all-attempt accuracy.',
        })
        for i in range(2):
            warmup = await request_one(client, rows[i], i)
            append_jsonl(output / 'warmup.jsonl', warmup)
            if not warmup['schema_valid'] or 'generated_tokens' not in warmup:
                raise RuntimeError('Warmup failed or server lacks measured inference headers; evidence retained.')
        phases = []
        aborted = None
        for concurrency in args.concurrency:
            pre = (await client.get('/health')).json()
            assert pre['status'] == 'ready' and not pre['busy'] and pre['cache'] == 'disabled'
            assert pre['model_fingerprint'] == before['model_fingerprint']
            sampler = GPUSampler()
            await sampler.sample()
            monitoring = asyncio.create_task(sampler.run())
            measured = []
            started = time.perf_counter()
            try:
                for offset in range(0, len(rows), concurrency):
                    if sampler.unsafe:
                        aborted = 'gpu_memory_safety_or_probe_failure'
                        break
                    wave = await asyncio.gather(*(request_one(client, row, index)
                        for index, row in enumerate(rows[offset:offset + concurrency], start=offset)))
                    measured.extend(wave)
                    for result in wave:
                        append_jsonl(output / f'c{concurrency}_requests.jsonl', result)
                    if any(r['error'] in ('gpu_memory_unavailable', 'ReadTimeout', 'ConnectTimeout', 'PoolTimeout', 'WriteTimeout', 'ConnectError') or r['status'] == 500 for r in wave):
                        aborted = 'resource_or_transport_failure'
                        break
            finally:
                elapsed = time.perf_counter() - started
                sampler.stop.set()
                await monitoring
                await sampler.sample()
            if sampler.unsafe:
                aborted = aborted or 'gpu_memory_safety_or_probe_failure'
            post = (await client.get('/health')).json()
            phase = summarize(measured, elapsed)
            phase.update(concurrency=concurrency, requested_requests=args.requests,
                completed_workload=len(measured) == args.requests, aborted=aborted,
                health_before=pre, health_after=post,
                gpu_card_peak_used_mib=max((s['used_mib'] for s in sampler.samples), default=None),
                gpu_memory_samples=sampler.samples, gpu_probe_errors=sampler.errors)
            phase['model_call_delta'] = post['inference_calls'] - pre['inference_calls']
            save_json(output / f'c{concurrency}_summary.json', phase)
            phases.append(phase)
            print(f"C={concurrency} requests={phase['requests']} success={phase['http_success_count']} elapsed={elapsed:.3f}s success_RPS={phase['success_rps']} errors={phase['error_counts']}", flush=True)
            if aborted:
                break
        save_json(output / 'summary.json', {'phases': phases, 'aborted': aborted,
            'completed_all_requested_phases': len(phases) == len(args.concurrency) and not aborted,
            'health_after': (await client.get('/health')).json()})


def main():
    configure()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--requests', type=int, default=32)
    parser.add_argument('--concurrency', nargs='+', type=int, choices=(1, 2, 4, 8), default=[1, 2, 4, 8])
    parser.add_argument('--output', default='reports/stage4_run')
    args = parser.parse_args()
    parsed = urlparse(args.url)
    if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost'):
        parser.error('Only a local HTTP API may be benchmarked.')
    if args.requests < 8 or args.requests > 48 or args.requests % 8:
        parser.error('--requests must be a multiple of 8 between 8 and 48.')
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
