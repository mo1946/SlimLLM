"""Small sequential functional checks against a live API; not a load benchmark."""

import argparse
import time

import httpx

from slimllm.runtime import ROOT, read_jsonl, project_path, save_json
from slimllm.schema import AfterSalesTicket


def check_response(response: httpx.Response, text: str) -> dict:
    assert response.status_code == 200, (response.status_code, response.text)
    ticket = AfterSalesTicket.model_validate(response.json())
    for name in ('order_id', 'product_name', 'reason'):
        value = getattr(ticket, name)
        assert value is None or value in text, name
    return {'status': response.status_code, 'cache': response.headers.get('X-SlimLLM-Cache'),
            'result': ticket.model_dump()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['core', 'redis', 'fallback'])
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    examples = read_jsonl(ROOT / 'data/examples.jsonl')
    first, second = examples[0]['text'], examples[3]['text']
    evidence = {'phase': args.phase, 'real_model_http_calls': True, 'load_test': False, 'checks': []}
    with httpx.Client(base_url=args.url, timeout=60, trust_env=False) as client:
        deadline = time.monotonic() + 90
        while True:
            try:
                health = client.get('/health')
                if health.status_code == 200:
                    break
            except httpx.ConnectError as exc:
                if '10013' in str(exc):
                    raise
            if time.monotonic() >= deadline:
                raise TimeoutError('API did not become healthy within 90 seconds.')
            time.sleep(.25)
        evidence['health_before'] = health.json()
        calls_before = health.json()['inference_calls']
        if args.phase == 'core':
            assert health.json()['cache'] == 'disabled'
            for text in (first, second):
                evidence['checks'].append(check_response(client.post('/extract', json={'text': text}), text))
            for body in ({}, {'text': ''}, {'text': '  '}, {'text': 7},
                         {'text': 'x' * 2001}, {'text': 'a', 'unexpected': True}):
                response = client.post('/extract', json=body)
                assert response.status_code == 422, response.text
            too_long = client.post('/extract', json={'text': '\u9f98' * 800})
            assert too_long.status_code == 422 and too_long.json()['detail']['code'] == 'input_token_limit'
            evidence['invalid_body_checks'] = 6
            evidence['token_limit_checked'] = True
            assert client.get('/health').json()['inference_calls'] == calls_before + 2
        elif args.phase == 'redis':
            assert health.json()['cache'] == 'available'
            response = client.post('/extract', json={'text': first})
            evidence['checks'].append(check_response(response, first))
            assert response.headers['X-SlimLLM-Cache'] == 'MISS'
            response = client.post('/extract', json={'text': first})
            evidence['checks'].append(check_response(response, first))
            assert response.headers['X-SlimLLM-Cache'] == 'HIT'
            assert client.get('/health').json()['inference_calls'] == calls_before + 1
            assert evidence['checks'][0]['result'] == evidence['checks'][1]['result']
        else:
            response = client.post('/extract', json={'text': second})
            evidence['checks'].append(check_response(response, second))
            assert response.headers['X-SlimLLM-Cache'] == 'MISS'
            after = client.get('/health').json()
            assert after['status'] == 'ready' and after['cache'] == 'unavailable'
            assert after['inference_calls'] == calls_before + 1
        evidence['health_after'] = client.get('/health').json()
    evidence['passed'] = True
    save_json(project_path(args.report), evidence)
    print(f'Live API {args.phase} checks: PASS. Evidence: {args.report}')


if __name__ == '__main__':
    main()
