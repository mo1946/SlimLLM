"""Score raw Qwen outputs against the frozen test set. No inference simulation."""

import argparse
import json

from pydantic import ValidationError

from slimllm.runtime import configure, file_hash, project_path, read_jsonl, save_json
from slimllm.schema import AfterSalesTicket

FIELDS = tuple(AfterSalesTicket.model_fields)


def comparison(before: dict, after: dict, before_run: dict, after_run: dict) -> dict:
    keys = ('model', 'revision', 'source_sha256', 'splits_sha256', 'prompt', 'ids',
            'split', 'max_new_tokens', 'do_sample', 'enable_thinking', 'dtype', 'seed')
    if any(key not in before_run or key not in after_run or before_run[key] != after_run[key] for key in keys):
        raise ValueError('Cannot compare runs with different data, prompt, model or decoding settings.')
    if (before.get('source_sha256') != before_run['source_sha256']
            or after.get('source_sha256') != after_run['source_sha256']
            or before.get('sample_count') != len(before_run['ids'])
            or after.get('sample_count') != len(after_run['ids'])):
        raise ValueError('Metric provenance does not match the evaluation manifest.')
    names = ('json_valid_rate', 'schema_valid_rate', 'field_macro_accuracy', 'exact_match_rate')
    return {'before': {key: before[key] for key in names}, 'after': {key: after[key] for key in names},
            'delta_percentage_points': {key: (after[key] - before[key]) * 100 for key in names},
            'field_delta_percentage_points': {key: (after['field_accuracy'][key] - before['field_accuracy'][key]) * 100 for key in FIELDS}}


def strict_json(raw: str):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key.')
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f'Non-JSON constant: {value}')

    return json.loads(raw, object_pairs_hook=object_pairs, parse_constant=invalid_constant)


def score(records: list[dict], predictions: list[dict]) -> dict:
    if not records or len({r['id'] for r in records}) != len(records):
        raise ValueError('Expected records must be nonempty with unique IDs.')
    by_id = {p['id']: p for p in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {r['id'] for r in records}:
        raise ValueError('Predictions must match every expected ID exactly once.')
    correct = dict.fromkeys(FIELDS, 0)
    nonnull_total = dict.fromkeys(FIELDS, 0)
    nonnull_correct = dict.fromkeys(FIELDS, 0)
    json_valid = schema_valid = exact = 0
    errors = []
    for row in records:
        gold = AfterSalesTicket.model_validate(row['label']).model_dump()
        for key in FIELDS:
            nonnull_total[key] += gold[key] is not None
        prediction = by_id[row['id']]
        try:
            parsed = strict_json(prediction['raw_output'])
            json_valid += 1
        except (ValueError, TypeError):
            errors.append({'id': row['id'], 'error': 'invalid_json'})
            continue
        try:
            actual = AfterSalesTicket.model_validate(parsed).model_dump()
            schema_valid += 1
        except ValidationError:
            errors.append({'id': row['id'], 'error': 'invalid_schema'})
            continue
        wrong = []
        for key in FIELDS:
            matches = actual[key] == gold[key]
            correct[key] += matches
            nonnull_correct[key] += matches and gold[key] is not None
            if not matches:
                wrong.append(key)
        exact += not wrong
        if wrong:
            errors.append({'id': row['id'], 'error': 'field_mismatch', 'fields': wrong})
    count = len(records)
    return {'sample_count': count, 'json_valid_count': json_valid, 'schema_valid_count': schema_valid,
        'json_valid_rate': json_valid / count, 'schema_valid_rate': schema_valid / count,
        'field_accuracy': {key: correct[key] / count for key in FIELDS},
        'field_macro_accuracy': sum(correct.values()) / (count * len(FIELDS)),
        'exact_match_rate': exact / count,
        'nonnull_accuracy': {key: nonnull_correct[key] / nonnull_total[key] if nonnull_total[key] else None for key in FIELDS},
        'nonnull_counts': nonnull_total, 'errors': errors,
        'policy': 'All expected samples in denominator; invalid JSON/schema counts wrong for all fields. No repair or fence stripping.'}


def main() -> None:
    configure()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/generated/stage1')
    parser.add_argument('--run', default='reports/baseline')
    parser.add_argument('--compare-to', help='Existing baseline report folder with identical frozen evaluation settings.')
    args = parser.parse_args()
    try:
        folder, output = project_path(args.data), project_path(args.run)
        source = folder / 'records.jsonl'
        split = json.loads((folder / 'splits.json').read_text(encoding='utf-8'))
        manifest = json.loads((output / 'run.json').read_text(encoding='utf-8'))
        if file_hash(source) != split['source_sha256'] or file_hash(source) != manifest['source_sha256']:
            raise ValueError('Dataset hash mismatch.')
        if file_hash(folder / 'splits.json') != manifest['splits_sha256']:
            raise ValueError('Split manifest hash mismatch.')
        ids = split['ids'][manifest['split']]
        if ids != manifest['ids']:
            raise ValueError('Evaluation IDs differ from frozen run.')
        rows = [row for row in read_jsonl(source) if row['id'] in ids]
        predictions = read_jsonl(output / 'predictions.jsonl')
        result = score(rows, predictions)
        result['source_sha256'] = file_hash(source)
        result['predictions_sha256'] = file_hash(output / 'predictions.jsonl')
        compared = None
        if args.compare_to:
            before_folder = project_path(args.compare_to)
            before_run = json.loads((before_folder / 'run.json').read_text(encoding='utf-8'))
            before = json.loads((before_folder / 'metrics.json').read_text(encoding='utf-8'))
            if file_hash(before_folder / 'predictions.jsonl') != before['predictions_sha256']:
                raise ValueError('Baseline prediction hash mismatch.')
            compared = comparison(before, result, before_run, manifest)
        save_json(output / 'metrics.json', result)
        lines = ['# Qwen3-1.7B Baseline', '', f"样本数：{result['sample_count']}；集合：{manifest['split']}。",
                 f"JSON 合法率：{result['json_valid_rate']:.2%}", f"Schema 合法率：{result['schema_valid_rate']:.2%}",
                 f"字段宏平均准确率：{result['field_macro_accuracy']:.2%}", f"整条完全匹配率：{result['exact_match_rate']:.2%}",
                 '', '| 字段 | 准确率 |', '| --- | --- |']
        lines += [f'| {key} | {value:.2%} |' for key, value in result['field_accuracy'].items()]
        if compared is not None:
            save_json(output / 'comparison.json', compared)
            lines += ['', '| 指标 | 微调前 | 微调后 | 变化（百分点） |', '| --- | ---: | ---: | ---: |']
            lines += [f"| {key} | {compared['before'][key]:.2%} | {compared['after'][key]:.2%} | {delta:+.2f} |"
                      for key, delta in compared['delta_percentage_points'].items()]
        lines += ['', '分母为全部样本，非法 JSON/Schema 的所有字段计错；无 JSON 修复。',
                  '标签来自 DeepSeek 合成数据；结果不代表真实生产工单的准确率。',
                  '原始输出、运行参数、逐项错误和文件哈希见同目录 JSON/JSONL。']
        with (output / 'report.md').open('x', encoding='utf-8') as stream:
            stream.write('\n'.join(lines) + '\n')
        print('\n'.join(lines))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
