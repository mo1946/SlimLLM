"""CPU grammar checks and raw-output comparisons against an already running local vLLM."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import os
import sys
import time
import warnings

from slimllm.evaluation.evaluate import score, strict_json
from slimllm.runtime import ROOT, append_jsonl, configure, file_hash, project_path, read_jsonl, save_json
from slimllm.schema import AfterSalesTicket


REGRESSION_SHA256 = '8301b490afd7346b60f08471ad1baa4b809e696c9d75b246c5097092ef2367be'
CLASSIFICATION_FIELDS = tuple(name for name, spec in AfterSalesTicket.model_json_schema()['properties'].items()
                              if 'enum' in spec)


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def load_evaluation_data() -> tuple[dict, dict]:
    from slimllm.evaluation.benchmark import fixed_records

    frozen, provenance = fixed_records(50)
    path = ROOT / 'data/structured_cases.jsonl'
    if file_hash(path) != REGRESSION_SHA256:
        raise ValueError('Regression labels changed after review; do not relabel from generated outputs.')
    regression = read_jsonl(path)
    if [row['id'] for row in regression] != [f'g{i:02d}' for i in range(1, 13)]:
        raise ValueError('Expected the 12 frozen regression IDs in order.')
    for index, row in enumerate(regression):
        expected_origin = 'user_provided_test_sentence' if index < 2 else 'handwritten_regression'
        if row.get('origin') != expected_origin:
            raise ValueError('Regression data must be identified as test sentences, not customer gold.')
        label = AfterSalesTicket.model_validate(row['label']).model_dump()
        for field in ('order_id', 'product_name', 'reason'):
            if label[field] is not None and label[field] not in row['text']:
                raise ValueError(f'Regression gold has an ungrounded {field}: {row["id"]}')
    return {'frozen_test': frozen, 'regression': regression}, {
        'frozen_test': provenance,
        'regression': {'sha256': REGRESSION_SHA256, 'ids': [row['id'] for row in regression],
                       'source': 'Two user-provided test sentences and ten handwritten cases; not real customer data or independent business gold.',
                       'reason_policy': 'g01-g04 use the preselected shortest complete symptom: 充不进去电了.'}}


def assess_output(row: dict, prediction: dict) -> dict:
    """Separate content checks from transport/completion status; never repair output."""
    schema = AfterSalesTicket.model_json_schema()
    enums = {name: set(schema['properties'][name]['enum']) for name in CLASSIFICATION_FIELDS}
    result = {'json_valid': False, 'schema_valid': False, 'grounded': False,
              'enum_fields': dict.fromkeys(CLASSIFICATION_FIELDS, False),
              'correct_fields': dict.fromkeys(CLASSIFICATION_FIELDS, False)}
    if not prediction.get('attempted') or prediction.get('status') != 200:
        result['enum_valid'] = False
        return result
    try:
        parsed = strict_json(prediction.get('raw_output'))
        result['json_valid'] = True
    except (ValueError, TypeError):
        result['enum_valid'] = False
        return result
    if isinstance(parsed, dict):
        for field in CLASSIFICATION_FIELDS:
            value = parsed.get(field)
            valid = isinstance(value, str) and value in enums[field]
            result['enum_fields'][field] = valid
            result['correct_fields'][field] = valid and value == row['label'][field]
    result['enum_valid'] = all(result['enum_fields'].values())
    try:
        ticket = AfterSalesTicket.model_validate(parsed).model_dump()
        result['schema_valid'] = True
        result['grounded'] = all(ticket[field] is None or ticket[field] in row['text']
                                 for field in ('order_id', 'product_name', 'reason'))
    except (ValueError, TypeError):
        pass
    return result


def completion_valid(prediction: dict) -> bool:
    usage = prediction.get('usage')
    count = usage.get('completion_tokens') if isinstance(usage, dict) else None
    return (bool(prediction.get('attempted')) and prediction.get('status') == 200
            and prediction.get('finish_reason') == 'stop' and type(count) is int
            and 1 <= count < prediction.get('max_tokens', 384))


def score_structured(records: list[dict], predictions: list[dict]) -> dict:
    if not records or len({row['id'] for row in records}) != len(records):
        raise ValueError('Gold must be nonempty with unique IDs.')
    by_id = {item['id']: item for item in predictions}
    if len(by_id) != len(predictions) or not set(by_id).issubset({row['id'] for row in records}):
        raise ValueError('Duplicate or unknown prediction ID.')
    checks, normalized, errors = [], [], []
    for row in records:
        AfterSalesTicket.model_validate(row['label'])
        prediction = by_id.get(row['id'], {'id': row['id'], 'attempted': False,
                                          'raw_output': None, 'error': 'not_executed'})
        normalized.append(prediction)
        checked = assess_output(row, prediction)
        checks.append(checked)
        categories = []
        if not prediction.get('attempted'):
            categories.append('not_executed')
        elif prediction.get('error'):
            categories.append(prediction['error'])
        if prediction.get('attempted') and prediction.get('status') == 200:
            if not checked['json_valid']:
                categories.append('invalid_json')
            else:
                if not checked['enum_valid']:
                    categories.append('invalid_enum')
                if not checked['schema_valid']:
                    categories.append('invalid_schema')
                elif not checked['grounded']:
                    categories.append('ungrounded_text')
                if not all(checked['correct_fields'].values()):
                    categories.append('classification_mismatch')
            if not completion_valid(prediction):
                categories.append('incomplete_completion')
        if categories:
            errors.append({'id': row['id'], 'categories': list(dict.fromkeys(categories)),
                           'wrong_classifications': [key for key, valid in checked['correct_fields'].items() if not valid]})
    n = len(records)
    valid_enums = [item for item in checks if item['enum_valid']]
    enum_n = len(valid_enums)
    legacy = score(records, [{'id': row['id'], 'raw_output': prediction.get('raw_output')
                              if prediction.get('attempted') and prediction.get('status') == 200 else None}
                             for row, prediction in zip(records, normalized)])
    count = lambda key: sum(item[key] for item in checks)
    correct = {field: sum(item['correct_fields'][field] for item in checks) for field in CLASSIFICATION_FIELDS}
    conditional = {field: sum(item['correct_fields'][field] for item in valid_enums) for field in CLASSIFICATION_FIELDS}
    attempted = sum(bool(item.get('attempted')) for item in normalized)
    http = sum(bool(item.get('attempted')) and isinstance(item.get('status'), int)
               and 200 <= item['status'] < 300 for item in normalized)
    return {'planned_count': n, 'attempted_count': attempted, 'not_executed_count': n - attempted,
            'http_success_count': http, 'http_success_rate_all': http / n,
            **{f'{key}_count': count(key) for key in ('json_valid', 'enum_valid', 'schema_valid', 'grounded')},
            **{f'{key}_rate_all': count(key) / n for key in ('json_valid', 'enum_valid', 'schema_valid', 'grounded')},
            'enum_field_valid_rate_all': {field: sum(item['enum_fields'][field] for item in checks) / n
                                          for field in CLASSIFICATION_FIELDS},
            'classification_accuracy_all': {field: correct[field] / n for field in CLASSIFICATION_FIELDS},
            'classification_macro_accuracy_all': sum(correct.values()) / (n * len(CLASSIFICATION_FIELDS)),
            'enum_valid_subset_count': enum_n,
            'classification_accuracy_enum_valid': {field: conditional[field] / enum_n if enum_n else None
                                                    for field in CLASSIFICATION_FIELDS},
            'classification_macro_accuracy_enum_valid': sum(conditional.values()) / (enum_n * len(CLASSIFICATION_FIELDS)) if enum_n else None,
            'completed_count': sum(completion_valid(item) for item in normalized),
            'accepted_completion_count': sum(completion_valid(prediction) and checked['schema_valid'] and checked['grounded']
                                             for prediction, checked in zip(normalized, checks)),
            'finish_reason_counts': dict(Counter(str(item.get('finish_reason')) for item in normalized if item.get('attempted'))),
            'error_counts': dict(Counter(category for item in errors for category in item['categories'])),
            'errors': errors, 'legacy_seven_field_metrics': legacy,
            'business_api_success_rate': None,
            'policy': 'All planned samples remain in denominators. Raw /v1/completions only; no /extract success inferred. Format/semantic scores are not completion acceptance. Legacy seven-field scoring remains unchanged.'}


def collect_completion(engine, row: dict, payload: dict) -> dict:
    import httpx

    result = {'id': row['id'], 'attempted': True, 'status': None, 'raw_output': None,
              'raw_response': None, 'finish_reason': None, 'usage': None, 'error': None,
              'abort_reason': None, 'request': payload, 'max_tokens': payload['max_tokens'],
              'prompt_token_ids_sha256': _digest(payload['prompt'])}
    start = time.perf_counter()
    try:
        response = engine.client.post('/v1/completions', json=payload)
        result.update(status=response.status_code, raw_response=response.text)
        if response.status_code != 200:
            result.update(error=f'upstream_http_{response.status_code}', abort_reason=f'upstream_http_{response.status_code}')
        else:
            envelope = response.json()
            choice = envelope['choices'][0]
            result.update(raw_output=choice['text'], finish_reason=choice.get('finish_reason'), usage=envelope.get('usage'))
            if not isinstance(result['raw_output'], str):
                raise ValueError('Completion text is not a string.')
            usage = result['usage']
            if not isinstance(usage, dict) or type(usage.get('completion_tokens')) is not int or usage['completion_tokens'] < 1:
                raise ValueError('Completion usage is invalid.')
            if not completion_valid(result):
                result['error'] = 'truncated_completion' if result['finish_reason'] == 'length' or usage['completion_tokens'] >= payload['max_tokens'] else 'incomplete_completion'
    except (httpx.RequestError, KeyboardInterrupt) as exc:
        result.update(error=type(exc).__name__, abort_reason=type(exc).__name__)
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        result.update(error='invalid_upstream_envelope', abort_reason='invalid_upstream_envelope', error_type=type(exc).__name__)
    result['elapsed_seconds'] = time.perf_counter() - start
    return result


def _artifact_snapshot() -> dict:
    names = ('slimllm/evaluation/evaluate_structured.py', 'slimllm/serving/vllm_backend.py', 'slimllm/serving/inference.py', 'slimllm/training/baseline.py', 'slimllm/schema.py', 'slimllm/runtime.py',
             'slimllm/evaluation/evaluate.py', 'slimllm/evaluation/benchmark.py', 'scripts/wsl/run_vllm.sh', 'data/ticket.schema.json',
             'data/structured_cases.jsonl', 'data/generated/stage1/records.jsonl',
             'data/generated/stage1/splits.json', 'reports/baseline/run.json')
    return {name: file_hash(ROOT / name) for name in names}


def _write_comparison_report(output, summary):
    percent = lambda value: f'{value:.2%}' if value is not None else 'n/a'
    lines = ['# Structured-output raw comparison', '', f"Evidence: {summary['evidence_kind']}",
             f"All requests completed: {summary['completed_all_requests']}; aborted: {summary['aborted_reason']}",
             '', 'Frozen test: DeepSeek synthetic. Regression: test sentences/handwritten labels; neither is real customer gold.',
             'Raw outputs are preserved without repair. Warmups are excluded. Missing/failed samples remain in planned denominators.',
             'No business /extract success rate or performance improvement is inferred.', '',
             '| Mode | Dataset | Planned / attempted | HTTP | JSON | Enum | Schema | Grounded | Class macro (all) | Class macro (enum-valid E) | Accepted completion |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for mode, group in summary['groups'].items():
        for dataset, values in group['metrics'].items():
            lines.append(f"| {mode} | {dataset} | {values['planned_count']} / {values['attempted_count']} | "
                         + ' | '.join(percent(values[key]) for key in ('http_success_rate_all', 'json_valid_rate_all',
                             'enum_valid_rate_all', 'schema_valid_rate_all', 'grounded_rate_all', 'classification_macro_accuracy_all'))
                         + f" | {percent(values['classification_macro_accuracy_enum_valid'])} (E={values['enum_valid_subset_count']}) | {values['accepted_completion_count']} |")
    lines += ['', '| Mode | Dataset | Intent | Issue | Priority | Action | Legacy 7-field macro / exact match |',
              '| --- | --- | ---: | ---: | ---: | ---: | --- |']
    for mode, group in summary['groups'].items():
        for dataset, values in group['metrics'].items():
            legacy = values['legacy_seven_field_metrics']
            lines.append(f'| {mode} | {dataset} | ' + ' | '.join(percent(values['classification_accuracy_all'][key]) for key in CLASSIFICATION_FIELDS)
                         + f" | {percent(legacy['field_macro_accuracy'])} / {percent(legacy['exact_match_rate'])} |")
    lines += ['', '| Mode | First warmup HTTP seconds | Warmups attempted |', '| --- | ---: | ---: |']
    for mode, group in summary['groups'].items():
        elapsed = group['first_request_seconds']
        lines.append(f"| {mode} | {elapsed if elapsed is not None else 'n/a'} | {len(group['warmup'])} |")
    lines += ['', 'The first warmup may include grammar initialization; it is not isolated compilation time or TTFT.',
              '', 'Error categories (can overlap):']
    for mode, group in summary['groups'].items():
        for dataset, values in group['metrics'].items():
            lines.append(f"- {mode}/{dataset}: `{json.dumps(values['error_counts'], ensure_ascii=False)}`")
    lines += ['', 'Reason exact-match boundaries can have multiple reasonable answers; labels were frozen before requests.',
              'Semantic correctness and grounded text are not guaranteed by a valid Schema. Per-ID raw text, completion/rejection reasons and timing are in JSONL/metrics files.',
              'Server prefix cache must be disabled by run_vllm.sh; this script does not remotely attest that setting.',
              'No claim of full Unicode-escape coverage: the previously accepted llguidance boundary remains.']
    with (output / 'report.md').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines) + '\n')


def compare(output_path: str, *, engine_factory=None) -> dict:
    """Sequential A/B against one existing engine. Injected factories are labelled test fixtures."""
    from slimllm.serving.vllm_backend import VllmExtractor
    from slimllm.training.baseline import MODEL_ID, REVISION

    output = project_path(output_path)
    if not output.is_relative_to((ROOT / 'reports').resolve()) or output == ROOT / 'reports':
        raise ValueError('Comparison output must be a new directory under reports/.')
    if output.exists():
        raise ValueError('Use a new output directory; previous evidence is never overwritten.')
    datasets, data = load_evaluation_data()
    snapshot = _artifact_snapshot()
    output.mkdir(parents=True, exist_ok=False)
    summary = {'evidence_kind': 'unit_test_fixture' if engine_factory is not None else 'raw_model_requests',
               'started_at': datetime.now(timezone.utc).isoformat(), 'groups': {},
               'aborted_reason': None, 'completed_all_requests': False}
    save_json(output / 'run.json', {'started_at': summary['started_at'], 'evidence_kind': summary['evidence_kind'],
        'datasets': data, 'artifact_sha256': snapshot, 'mode_order': ['off', 'json_schema'],
        'conditions': {'concurrency': 1, 'retries': 0, 'warmups_per_mode': 2, 'redis': 'bypassed',
                       'prefix_cache': 'required disabled by run_vllm.sh; not remotely attested',
                       'unicode_escape_boundary': 'accepted limitation recorded in T1; not fixed'},
        'packages': {name: metadata.version(name) for name in ('httpx', 'pydantic', 'transformers')}})
    reference = None
    factory = engine_factory or VllmExtractor
    for mode in ('off', 'json_schema'):
        folder = output / mode
        folder.mkdir()
        predictions = {name: [] for name in datasets}
        group = {'identity': None, 'request_signatures': None, 'warmup': [], 'conditions_verified': False,
                 'first_request_seconds': None, 'metrics': {}}
        engine = None
        try:
            if summary['aborted_reason'] is None:
                engine = factory(structured_outputs=mode)
                engine.load()  # VllmExtractor loads a CPU tokenizer and HTTP client only.
                identity = {'base_fingerprint': engine._base_fingerprint, 'url': engine.url,
                            'version': engine.version, 'model_alias': engine.model_name,
                            'base_model': MODEL_ID, 'base_revision': REVISION,
                            'schema_sha256': _digest(AfterSalesTicket.model_json_schema()),
                            'adapter': engine.adapter, 'prompt': engine.prompt, 'prompt_sha256': _digest(engine.prompt)}
                payloads, signatures = {}, {}
                for name, rows in datasets.items():
                    payloads[name] = {row['id']: engine.build_request(row['text']) for row in rows}
                    signatures[name] = {}
                    for row in rows:
                        payload = payloads[name][row['id']]
                        if (mode == 'off' and 'structured_outputs' in payload) or (mode == 'json_schema' and
                                payload.get('structured_outputs') != {'json': AfterSalesTicket.model_json_schema()}):
                            raise ValueError('Requested constraint mode is missing or incorrect.')
                        signatures[name][row['id']] = _digest({key: value for key, value in payload.items() if key != 'structured_outputs'})
                conditions = {'identity': identity, 'requests': signatures}
                if reference is not None and conditions != reference:
                    raise ValueError('A/B input tokens, generation settings or model identity changed.')
                if _artifact_snapshot() != snapshot:
                    raise ValueError('Frozen artifacts changed before requests.')
                reference = conditions
                group.update(identity=identity, request_signatures=signatures, conditions_verified=True,
                             model_fingerprint=engine.fingerprint)
                # Persist model/version and per-ID request signatures before any generation.
                save_json(folder / 'run.json', {key: value for key, value in group.items() if key != 'metrics'})
                for index, row in enumerate(datasets['frozen_test'][:2]):
                    item = collect_completion(engine, row, payloads['frozen_test'][row['id']])
                    item.update(mode=mode, dataset='frozen_test', role='warmup')
                    append_jsonl(folder / 'warmup.jsonl', item)
                    group['warmup'].append(item)
                    if index == 0:
                        group['first_request_seconds'] = item['elapsed_seconds']
                    if item['abort_reason']:
                        summary['aborted_reason'] = item['abort_reason']
                        break
                for name, rows in datasets.items():
                    for row in rows:
                        if summary['aborted_reason'] is not None:
                            break
                        item = collect_completion(engine, row, payloads[name][row['id']])
                        item.update(mode=mode, dataset=name, role='measured')
                        # Write raw generation before any Pydantic or semantic scoring.
                        append_jsonl(folder / f'{name}_predictions.jsonl', item)
                        predictions[name].append(item)
                        if item['abort_reason']:
                            summary['aborted_reason'] = item['abort_reason']
                if _artifact_snapshot() != snapshot:
                    raise ValueError('Frozen artifacts changed during requests.')
        except (Exception, KeyboardInterrupt) as exc:
            summary['aborted_reason'] = f'{type(exc).__name__}: {exc}'
        finally:
            if engine is not None:
                try:
                    engine.close()
                except Exception as exc:
                    group['close_error_type'] = type(exc).__name__
                    summary['aborted_reason'] = summary['aborted_reason'] or f'close_failed: {type(exc).__name__}'
        for name, rows in datasets.items():
            observed = {item['id'] for item in predictions[name]}
            for row in rows:
                if row['id'] not in observed:
                    item = {'id': row['id'], 'mode': mode, 'dataset': name, 'role': 'measured',
                            'attempted': False, 'status': None, 'raw_output': None,
                            'error': 'not_executed', 'skip_reason': summary['aborted_reason']}
                    predictions[name].append(item)
                    append_jsonl(folder / f'{name}_predictions.jsonl', item)
            group['metrics'][name] = score_structured(rows, predictions[name])
        save_json(folder / 'metrics.json', group['metrics'])
        summary['groups'][mode] = group
    summary['completed_all_requests'] = summary['aborted_reason'] is None and all(
        values['attempted_count'] == values['planned_count'] for group in summary['groups'].values()
        for values in group['metrics'].values())
    summary['finished_at'] = datetime.now(timezone.utc).isoformat()
    save_json(output / 'summary.json', summary)
    _write_comparison_report(output, summary)
    return summary


def schema_check_cases() -> list[dict]:
    """Handwritten syntax fixtures, not generated predictions or semantic gold."""
    examples = read_jsonl(ROOT / 'data/examples.jsonl')
    base = examples[0]['label']
    cases = []

    def add(name, payload, expected, *, raw=None, ensure_ascii=False):
        cases.append({'id': name, 'expected_valid': expected,
                      'raw': raw if raw is not None else json.dumps(
                          payload, ensure_ascii=ensure_ascii, separators=(',', ':'))})

    for row in examples:
        add(row['id'], row['label'], True)
    add('escaped_unicode', base, True, ensure_ascii=True)
    add('escaped_quotes_backslash_newline', base | {
        'product_name': '蓝牙"耳机"\\配件', 'reason': '左耳\n没有声音'}, True)
    for field, limit in (('order_id', 64), ('product_name', 120), ('reason', 300)):
        char = 'A' if field == 'order_id' else '中'
        for length in (1, limit):
            add(f'{field}_length_{length}', base | {field: char * length}, True)
        add(f'{field}_over_limit', base | {field: char * (limit + 1)}, False)
        add(f'{field}_empty', base | {field: ''}, False)
        add(f'{field}_null', base | {field: None}, True)
        add(f'{field}_wrong_type', base | {field: 123}, False)
        for index, text in enumerate((' ', '\t\n\r', '\u00a0', '\u2003', '\u3000', '\u0085')):
            add(f'{field}_whitespace_{index}', base | {field: text}, False)
    for field in ('product_name', 'reason'):
        for index, text in enumerate((' 中 ', '\t中\n', '\u200b', '😀')):
            add(f'{field}_nonblank_{index}', base | {field: text}, True)
    for index, text in enumerate(('订单123', '12 34', '123\n', '-_ABC012')):
        add(f'order_id_pattern_{index}', base | {'order_id': text}, index == 3)
    for field in AfterSalesTicket.model_fields:
        add(f'missing_{field}', {key: value for key, value in base.items() if key != field}, False)
    for field in ('intent', 'issue_type', 'priority', 'requested_action'):
        add(f'{field}_illegal_enum', base | {field: '安排寄修'}, False)
        add(f'{field}_null_enum', base | {field: None}, False)
    add('extra_key', base | {'extra': 'value'}, False)
    raw = json.dumps(base, ensure_ascii=False, separators=(',', ':'))
    add('duplicate_key', None, False, raw='{"intent":"inquiry",' + raw[1:])
    add('truncated_json', None, False, raw=raw[:-1])
    add('trailing_comma', None, False, raw=raw[:-1] + ',}')
    add('non_json_constant', None, False, raw=raw.replace('"normal"', 'NaN'))
    add('code_fence', None, False, raw='```json\n' + raw + '\n```')
    return cases


def _pydantic_accepts(raw: str) -> bool:
    try:
        AfterSalesTicket.model_validate(strict_json(raw))
    except (ValueError, TypeError):
        return False
    return True


def _grammar_accepts(raw, tokenizer, ll_tokenizer, grammar, matcher_class):
    """Check text and EOS against a fresh real matcher, without sampling logits."""
    tokens = tokenizer.encode(raw, add_special_tokens=False)
    if tokenizer.decode(tokens, skip_special_tokens=False) != raw:
        raise ValueError('Tokenizer round-trip changed the candidate; check is inconclusive.')
    matcher = matcher_class(ll_tokenizer, grammar)
    if matcher.get_error():
        raise ValueError(matcher.get_error())
    prefix = matcher.validate_tokens(tokens)
    consumed = prefix == len(tokens) and matcher.consume_tokens(tokens)
    accepting = bool(consumed and matcher.is_accepting())
    # A complete fixed-key object can stop with NoExtension before EOS. Like
    # vLLM's GuidanceGrammar, use its EOS mask then; consuming another token
    # into that stopped matcher would itself create an artificial error.
    eos_allowed = bool(accepting and matcher.compute_logit_bias()[tokenizer.eos_token_id] == 200)
    already_stopped = matcher.is_stopped()
    completed = eos_allowed and (already_stopped or (
        matcher.validate_tokens([tokenizer.eos_token_id]) == 1
        and matcher.consume_tokens([tokenizer.eos_token_id])))
    return {
        'grammar_valid': bool(completed and matcher.is_stopped() and not matcher.get_error()),
        'tokens': len(tokens), 'allowed_prefix_tokens': prefix,
        'accepting_before_eos': accepting, 'eos_allowed': eos_allowed,
        'eos_handling': 'stopped_matcher_mask' if already_stopped else 'consume_if_allowed',
        'stop_reason': matcher.stop_reason(),
        'matcher_error': matcher.get_error(), 'grammar_warnings': matcher.get_grammar_warnings(),
    }


def check_schema() -> dict:
    report = {
        'started_at': datetime.now(timezone.utc).isoformat(), 'status': 'blocked',
        'scope': 'CPU grammar/tokenizer compatibility fixtures only; no model inference or semantic accuracy.',
        'platform': sys.platform, 'python': sys.version, 'cases': [], 'warnings': [],
        'model_weights_loaded': False,
    }
    schema_path = ROOT / 'data/ticket.schema.json'
    report['schema_sha256'] = file_hash(schema_path)
    schema = json.loads(schema_path.read_text(encoding='utf-8'))
    try:
        if schema != AfterSalesTicket.model_json_schema():
            raise ValueError('Pydantic schema differs from data/ticket.schema.json; do not weaken either schema.')
        if sys.platform != 'linux':
            raise RuntimeError('check-schema requires the existing WSL Linux environment; no dependency was installed.')
        # Import native Linux libraries only here, keeping Windows tests lightweight.
        import llguidance
        import llguidance.hf
        from transformers import AutoTokenizer

        report['versions'] = {name: metadata.version(name) for name in
                              ('vllm', 'llguidance', 'transformers', 'pydantic', 'tokenizers')}
        if report['versions']['vllm'] != '0.11.2' or report['versions']['llguidance'] != '1.3.0':
            raise ValueError('This compatibility check targets installed vLLM 0.11.2 / llguidance 1.3.0.')
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        model_path = ROOT / '.cache/huggingface/qwen3-1.7b'
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
        if tokenizer.eos_token_id is None:
            raise ValueError('Local tokenizer has no EOS token.')
        vocab_size = json.loads((model_path / 'config.json').read_text(encoding='utf-8'))['vocab_size']
        ll_tokenizer = llguidance.hf.from_tokenizer(tokenizer, n_vocab=vocab_size)
        report['tokenizer'] = {'path': str(model_path), 'eos_token_id': tokenizer.eos_token_id,
                               'vocab_size': vocab_size, 'sha256': file_hash(model_path / 'tokenizer.json')}
        # Same defaults as vLLM 0.11.2 serialize_guidance_grammar; lenient stays false.
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            grammar = llguidance.LLMatcher.grammar_from_json_schema(
                schema, defaults={'whitespace_flexible': True})
            error = llguidance.LLMatcher.validate_grammar(grammar, ll_tokenizer)
            has_error, messages = llguidance.LLMatcher.validate_grammar_with_warnings(grammar, ll_tokenizer)
            report['warnings'] = [str(item.message) for item in captured]
        report['grammar_validation'] = {'error': error, 'has_error': has_error, 'messages': messages}
        if error or has_error:
            raise ValueError(f'Full schema grammar validation failed: {error or messages}')
        report['warnings'].extend(messages)
        for case in schema_check_cases():
            item = dict(case, pydantic_valid=_pydantic_accepts(case['raw']))
            try:
                item.update(_grammar_accepts(case['raw'], tokenizer, ll_tokenizer,
                                             grammar, llguidance.LLMatcher))
                item['passed'] = item['pydantic_valid'] == item['grammar_valid'] == case['expected_valid']
            except Exception as exc:
                item.update(passed=False, error_type=type(exc).__name__, error=str(exc))
            report['cases'].append(item)
        report['passed_count'] = sum(item['passed'] for item in report['cases'])
        report['case_count'] = len(report['cases'])
        report['failed_ids'] = [item['id'] for item in report['cases'] if not item['passed']]
        report['status'] = 'passed' if not report['failed_ids'] else 'failed'
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc))
    finally:
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        torch = sys.modules.get('torch')
        report['cuda_initialized'] = bool(torch and torch.cuda.is_initialized())
    return report


def main() -> None:
    configure()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    check = commands.add_parser('check-schema', help='CPU-only full-schema grammar compatibility check in existing WSL.')
    check.add_argument('--output', help='New JSON report path inside the project; never overwrite evidence.')
    comparison = commands.add_parser('compare', help='Raw off/on comparison; requires an already running local vLLM engine.')
    comparison.add_argument('--output', required=True, help='New directory under reports/; never overwrite evidence.')
    args = parser.parse_args()
    if args.command == 'compare':
        try:
            result = compare(args.output)
        except (ValueError, OSError, AssertionError) as exc:
            parser.exit(2, f'{exc}\n')
        print(f"Comparison completed: {result['completed_all_requests']}; aborted: {result['aborted_reason']}")
        print(f'Report: {project_path(args.output) / "report.md"}')
        raise SystemExit(0 if result['completed_all_requests'] else 1)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = project_path(args.output or f'reports/structured_schema_{stamp}.json')
    if output.exists():
        parser.exit(2, 'Use a new output file; previous evidence is never overwritten.\n')
    report = check_schema()
    save_json(output, report)
    print(f"Schema check: {report['status']}; passed {report.get('passed_count', 0)}/{report.get('case_count', 0)}")
    print(f'Report: {output}')
    if report.get('error'):
        print(report['error'])
    for item in report['cases']:
        if not item['passed']:
            print(f"FAIL {item['id']}: Pydantic={item['pydantic_valid']}, grammar={item.get('grammar_valid')}; {item.get('error', '')}")
    raise SystemExit(0 if report['status'] == 'passed' else 1)


if __name__ == '__main__':
    main()
