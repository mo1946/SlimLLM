"""DeepSeek generation, validation and leakage-aware dataset split. No model imports."""

import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import json
import os
from pathlib import Path
import random
import re
import time
import unicodedata
import urllib.error
import urllib.request

from pydantic import ValidationError

from runtime import ROOT, append_jsonl, configure, extraction_prompt, file_hash, project_path, read_jsonl, save_json
from schema import AfterSalesTicket

MODEL = 'deepseek-flash'
INPUT_USD_PER_M = 0.30
OUTPUT_USD_PER_M = 1.20
ISSUES = ['quality', 'damaged', 'missing', 'wrong_item', 'logistics', 'not_as_described', 'no_reason', 'other', 'unknown']


def load_key() -> str:
    key = os.environ.get('DEEPSEEK_API_KEY', '').strip()
    if not key and (ROOT / '.env').exists():
        for line in (ROOT / '.env').read_text(encoding='utf-8-sig').splitlines():
            name, separator, value = line.partition('=')
            if separator and name.strip() == 'DEEPSEEK_API_KEY':
                key = value.strip().strip('\"\'')
    if not key:
        raise ValueError('Missing DEEPSEEK_API_KEY. Configure it locally in .env or the environment.')
    return key


def validate_record(raw: dict) -> dict:
    if not isinstance(raw, dict) or set(raw) != {'text', 'label'}:
        raise ValueError('Record must contain only text and label.')
    text = raw['text']
    if not isinstance(text, str) or not 8 <= len(text.strip()) <= 600:
        raise ValueError('Ticket text must contain 8 to 600 characters.')
    label = AfterSalesTicket.model_validate(raw['label']).model_dump()
    for field in ('order_id', 'product_name', 'reason'):
        if label[field] is not None and label[field] not in text:
            raise ValueError(f'{field} is not a verbatim substring.')
    if re.search(r'(?<!\d)1[3-9]\d{9}(?!\d)', text):
        raise ValueError('Possible telephone number; exclude from synthetic data.')
    return {'text': text.strip(), 'label': label}


def normalized(text: str) -> str:
    return re.sub(r'[\W_]+', '', unicodedata.normalize('NFKC', text).lower())


def template(record: dict) -> str:
    text = record['text']
    for field in ('order_id', 'product_name'):
        value = record['label'][field]
        if value:
            text = text.replace(value, f'<{field}>')
    return re.sub(r'\d+', '#', normalized(text))


def payload_for(batch: int, size: int, model: str = MODEL) -> dict:
    issue = ISSUES[batch % len(ISSUES)]
    return {
        'model': model, 'thinking': {'type': 'disabled'}, 'temperature': 1.0,
        'max_tokens': 4096, 'response_format': {'type': 'json_object'},
        'messages': [
            {'role': 'system', 'content': extraction_prompt() + '\n现在生成虚构训练样本，每条只有 text 和 label。'},
            {'role': 'user', 'content': (
                f'生成 {size} 条互不近似的中文单工单 JSON 数据，批次 {batch}，主要问题类型 {issue}。'
                '返回 {"records":[{"text":"工单原文","label":{七个字段}}]}。'
                '每条使用不同商品、句式和叙述顺序；混合简短口语、错别字、礼貌表达和抱怨。'
                '约三分之一省略订单号，约五分之一省略商品名；订单号用 TEST- 前缀。'
                '覆盖办理请求、纯投诉和咨询，正常/催急/不急优先级；涉及真实安全风险才 urgent。'
                '部分仅描述问题而不提出动作。unknown 类覆盖售后规则咨询与缺信息诉求。'
                '重要：纯描述故障并抱怨、没有要求办理时 intent=complaint；不要标成售后办理。'
                '已实际发生的缺件、故障即使是在咨询也应标明问题类型，不得用 unknown。'
                '未下单或假设情况的规则咨询使用 issue_type=unknown、reason=null。'
                '泛泛说帮我处理但没说具体动作，requested_action=unknown；催一下不等于补发。'
                '仅明确要求补发才用 reship，明确要求查物流才用 track_delivery。'
                '同一条只能有一个实际商品和主要问题；避免反问句同时像投诉又像咨询。'
                '不用真实个人信息、手机号和地址，不含多订单或互相冲突的诉求。'
                '原因字段必须是原文连续片段。严禁复制同一模板仅替换商品或订单号。'
            )},
        ],
    }


def generate(args: argparse.Namespace) -> None:
    folder = project_path(args.output)
    source = folder / 'records.jsonl'
    ledger = folder / 'requests.jsonl'
    manifest = folder / 'generation.json'
    if args.dry_run:
        print(json.dumps({'network_requests': 0, 'target': args.count, 'model': MODEL,
                          'budget_usd_upper_bound': args.budget_usd, 'output': str(folder),
                          'request_preview': payload_for(0, min(10, args.count))}, ensure_ascii=False, indent=2))
        return
    key = load_key()
    configuration = {'model': MODEL, 'prompt_sha256': __import__('hashlib').sha256(extraction_prompt().encode()).hexdigest(),
                     'prices_usd_per_m': {'input': INPUT_USD_PER_M, 'output': OUTPUT_USD_PER_M},
                     'max_requests': args.max_requests, 'budget_usd': args.budget_usd}
    if manifest.exists():
        if json.loads(manifest.read_text(encoding='utf-8')) != configuration:
            raise ValueError('Generation settings changed; use a new output directory.')
    else:
        save_json(manifest, configuration)
    rows = read_jsonl(source)
    if (folder / 'splits.json').exists() and len(rows) < args.count:
        raise ValueError('Dataset is frozen; use a new output directory for additional generation.')
    for row in rows:
        validate_record({'text': row['text'], 'label': row['label']})
    seen = {normalized(row['text']) for row in rows}
    examples = read_jsonl(ROOT / 'data/examples.jsonl')
    excluded = [template(row) for row in examples]
    reservations = [event for event in read_jsonl(ledger) if event['event'] == 'reserved']
    reserved = sum(event['reserved_usd'] for event in reservations)
    batch = len(reservations)
    while len(rows) < args.count:
        if batch >= args.max_requests:
            raise ValueError('Request cap reached; partial data preserved.')
        payload = payload_for(batch, min(10, args.count - len(rows)))
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        # UTF-8 bytes conservatively bound input tokens; peak prices bound off-peak costs.
        ceiling = ((len(body) + 512) * INPUT_USD_PER_M + payload['max_tokens'] * OUTPUT_USD_PER_M) / 1_000_000
        if reserved + ceiling > args.budget_usd:
            raise ValueError('Conservative API budget cap reached; partial data preserved.')
        prompt_hash = __import__('hashlib').sha256(body).hexdigest()
        append_jsonl(ledger, {'event': 'reserved', 'batch': batch, 'reserved_usd': ceiling, 'payload_sha256': prompt_hash})
        reserved += ceiling
        request = urllib.request.Request('https://api.deepseek.com/chat/completions', data=body,
            headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {key}'}, method='POST')
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                result = json.loads(response.read(2_000_000))
        except urllib.error.HTTPError as exc:
            append_jsonl(ledger, {'event': 'http_error', 'batch': batch, 'status': exc.code})
            if exc.code not in (429, 500, 502, 503, 504):
                raise ValueError(f'DeepSeek HTTP {exc.code}; response body omitted to protect credentials.') from None
            batch += 1
            time.sleep(3)
            continue
        except (urllib.error.URLError, TimeoutError) as exc:
            append_jsonl(ledger, {'event': 'network_error', 'batch': batch, 'type': type(exc).__name__})
            raise RuntimeError('DeepSeek network request failed; reservation retained. Resume after connectivity is restored.') from None
        usage = result.get('usage', {})
        append_jsonl(ledger, {'event': 'response', 'batch': batch, 'usage': usage, 'response_id': result.get('id')})
        accepted = 0
        rejected = Counter()
        try:
            choice = result['choices'][0]
            if choice.get('finish_reason') != 'stop':
                raise ValueError('Incomplete response.')
            records = json.loads(choice['message']['content'])['records']
            if not isinstance(records, list) or len(records) > 10:
                raise ValueError('Unexpected records array.')
            for raw in records:
                try:
                    clean = validate_record(raw)
                    signature = normalized(clean['text'])
                    shape = template(clean)
                    if signature in seen or any(SequenceMatcher(None, shape, item).ratio() >= .82 for item in excluded):
                        raise ValueError('Duplicate or too similar to a prompt example.')
                    row = {'id': f'ds-{len(rows) + 1:04d}', **clean, 'batch': batch, 'source': 'deepseek-api'}
                    append_jsonl(source, row)
                    rows.append(row)
                    seen.add(signature)
                    accepted += 1
                    if len(rows) == args.count:
                        break
                except (ValidationError, ValueError, TypeError) as exc:
                    rejected[type(exc).__name__] += 1
        except (KeyError, IndexError, ValueError, TypeError):
            rejected['invalid_response'] += 1
        append_jsonl(ledger, {'event': 'validated', 'batch': batch, 'accepted': accepted, 'rejected': dict(rejected)})
        print(f'batch={batch} accepted={accepted} total={len(rows)}/{args.count} reserved_usd={reserved:.4f}', flush=True)
        batch += 1


def split_records(rows: list[dict], seed: int = 42) -> dict[str, list[str]]:
    if len(rows) < 10 or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Need at least 10 records with unique IDs.')
    signatures = [template(row) for row in rows]
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    # A complete similarity graph keeps transitive template variants together.
    for i in range(len(rows)):
        for j in range(i):
            if SequenceMatcher(None, signatures[i], signatures[j], autojunk=False).ratio() >= .82:
                parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row['id'])
    if len(groups) < 3:
        raise ValueError('Fewer than three distinct template groups; cannot safely split.')
    shuffled = list(groups.values())
    random.Random(seed).shuffle(shuffled)
    shuffled.sort(key=len, reverse=True)
    result = {'train': [], 'validation': [], 'test': []}
    size = len(rows)
    targets = {'train': size - 2 * round(size * .1), 'validation': round(size * .1), 'test': round(size * .1)}
    for index, group in enumerate(shuffled):
        empty = [name for name, ids in result.items() if not ids]
        candidates = empty if len(shuffled) - index <= len(empty) else list(result)
        destination = max(candidates, key=lambda name: targets[name] - len(result[name]))
        result[destination].extend(group)
    if any(not ids for ids in result.values()):
        raise ValueError('Template groups are too imbalanced for a three-way split.')
    return result


def split(args: argparse.Namespace) -> None:
    folder = project_path(args.output)
    source = folder / 'records.jsonl'
    rows = read_jsonl(source)
    if not 400 <= len(rows) <= 600:
        raise ValueError('Final stage-one split requires 400 to 600 real generated records.')
    for row in rows:
        validate_record({'text': row['text'], 'label': row['label']})
    splits = split_records(rows)
    save_json(folder / 'splits.json', {'seed': 42, 'source_sha256': file_hash(source), 'ids': splits,
        'counts': {name: len(ids) for name, ids in splits.items()},
        'issue_distribution': {name: dict(Counter(r['label']['issue_type'] for r in rows if r['id'] in ids)) for name, ids in splits.items()},
        'method': 'near-template connected components, threshold .82; grouped approximately 80/10/10'})
    print(json.dumps({name: len(ids) for name, ids in splits.items()}))


def review(args: argparse.Namespace) -> None:
    """Blind second-pass labels; audit entries permit crash-safe replay, not human gold labels."""
    folder = project_path(args.output)
    source, audit, ledger = folder / 'records.jsonl', folder / 'review.jsonl', folder / 'requests.jsonl'
    if (folder / 'splits.json').exists():
        raise ValueError('Cannot relabel an already frozen dataset.')
    rows = read_jsonl(source)
    if not rows:
        raise ValueError('Generate data before review.')
    by_id = {r['id']: r for r in rows}
    done = set()
    for event in read_jsonl(audit):
        if event.get('event') == 'blind_review':
            for entry in event['records']:
                by_id[entry['id']]['label'] = entry['after']
                done.add(entry['id'])

    def persist():
        temporary = source.with_suffix('.tmp')
        with temporary.open('w', encoding='utf-8', newline='\n') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, source)

    persist()
    key = load_key()
    pending = [r for r in rows if r['id'] not in done]
    for start in range(0, len(pending), 20):
        selected = pending[start:start + 20]
        instruction = extraction_prompt() + (
            '\n独立复核以下工单，不存在预设问题类别，不要按照批次猜类别。'
            '只有实际要求办理才是 aftersales_request；纯描述故障、抱怨为 complaint。'
            '问能否换货/补发等具体办理请求为 aftersales_request；问规则、时效为 inquiry。'
            '实际缺件/故障即使是咨询也必须标明问题类型；未下单/假设的规则咨询为 unknown 且 reason=null。'
            '所有功能故障如漏水、无法开机、异响、卡住、开胶等均是 quality，不是 other 或 no_reason。'
            'no_reason 仅用于明确因买多、改变主意、不喜欢等非故障退货；other 仅用于无法落入已有类别的实际问题。'
            '仅泛泛催促或请求处理不代表具体解决动作，填 unknown。明确不急才 low，单纯未下单不代表 low。'
            '只输出 JSON {"records":[{"id":"原ID","label":{七字段}}]}，每个输入ID恰好一次。'
        )
        payload = {'model': MODEL, 'thinking': {'type': 'disabled'}, 'temperature': 0,
            'max_tokens': 8192, 'response_format': {'type': 'json_object'},
            'messages': [{'role': 'system', 'content': instruction},
                         {'role': 'user', 'content': json.dumps([{'id': r['id'], 'text': r['text']} for r in selected], ensure_ascii=False)}]}
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        reservations = [e for e in read_jsonl(ledger) if e['event'] == 'reserved']
        ceiling = ((len(body) + 512) * INPUT_USD_PER_M + 8192 * OUTPUT_USD_PER_M) / 1_000_000
        if len(reservations) >= args.max_requests or sum(e['reserved_usd'] for e in reservations) + ceiling > args.budget_usd:
            raise ValueError('Shared generation/review request or budget cap reached.')
        batch = len(reservations)
        append_jsonl(ledger, {'event': 'reserved', 'stage': 'review', 'batch': batch, 'reserved_usd': ceiling,
            'payload_sha256': __import__('hashlib').sha256(body).hexdigest()})
        request = urllib.request.Request('https://api.deepseek.com/chat/completions', data=body,
            headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {key}'}, method='POST')
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                result = json.load(response)
        except (urllib.error.URLError, TimeoutError) as exc:
            append_jsonl(ledger, {'event': 'review_network_error', 'batch': batch, 'type': type(exc).__name__})
            raise RuntimeError('Review request failed; credential-bearing response omitted.') from None
        append_jsonl(ledger, {'event': 'response', 'stage': 'review', 'batch': batch,
            'usage': result.get('usage', {}), 'response_id': result.get('id')})
        choice = result['choices'][0]
        if choice.get('finish_reason') != 'stop':
            raise ValueError('Truncated review response; stop before changing labels.')
        labels = json.loads(choice['message']['content'])['records']
        if len(labels) != len(selected) or {r['id'] for r in labels} != {r['id'] for r in selected}:
            raise ValueError('Review returned missing or duplicated IDs.')
        entries = []
        for item in labels:
            original = by_id[item['id']]
            fallback = None
            proposed_reason = item['label'].get('reason')
            if isinstance(proposed_reason, str) and proposed_reason not in original['text']:
                fallback = proposed_reason
                item['label']['reason'] = original['label']['reason']
            label = validate_record({'text': original['text'], 'label': item['label']})['label']
            entry = {'id': item['id'], 'before': original['label'], 'after': label}
            if fallback is not None:
                entry['reason_needs_manual_review'] = fallback
            entries.append(entry)
        append_jsonl(audit, {'event': 'blind_review', 'reviewer': MODEL, 'human_review': False,
                            'batch': batch, 'records': entries})
        for entry in entries:
            by_id[entry['id']]['label'] = entry['after']
            done.add(entry['id'])
        persist()
        print(f'Blind review {len(done)}/{len(rows)} changed={sum(e["before"] != e["after"] for e in entries)}', flush=True)


def main() -> None:
    configure()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['generate', 'review', 'split'])
    parser.add_argument('--output', default='data/generated/stage1')
    parser.add_argument('--count', type=int, default=500)
    parser.add_argument('--max-requests', type=int, default=80)
    parser.add_argument('--budget-usd', type=float, default=1.0)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.dry_run and args.command != 'generate':
        parser.error('--dry-run is supported only for generate; no request was sent.')
    if not 1 <= args.count <= 600 or not 1 <= args.max_requests <= 100 or not 0 < args.budget_usd <= 5:
        parser.error('count 1..600, max-requests 1..100, budget-usd (0,5] required.')
    try:
        {'generate': generate, 'review': review, 'split': split}[args.command](args)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
