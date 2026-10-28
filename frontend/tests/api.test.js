// HTTP contract fixtures only. These are not model inference or accuracy results.
import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { ApiError, extractTicket, getHealth, validateTicket } from '../src/api.js'

const fixture = JSON.parse(readFileSync(new URL('../../data/examples.jsonl', import.meta.url), 'utf8').split(/\r?\n/)[0])
function mockFetch(t, handler) {
  t.mock.method(globalThis, 'fetch', handler)
}
function json(data, status = 200, headers = {}) {
  return new Response(JSON.stringify(data), { status, headers: { 'Content-Type': 'application/json', ...headers } })
}

test('preserves Unicode input and order IDs, posts to the dev proxy and returns metadata', async (t) => {
  mockFetch(t, async (url, options) => {
    assert.equal(url, '/api/extract')
    assert.equal(options.method, 'POST')
    assert.deepEqual(JSON.parse(options.body), { text: fixture.text })
    assert.equal(options.headers['Content-Type'], 'application/json')
    return json(fixture.label, 200, { 'X-SlimLLM-Cache': 'MISS', 'X-SlimLLM-Queue-Wait-Seconds': '0.003' })
  })
  const result = await extractTicket(fixture.text)
  assert.deepEqual(result.ticket, fixture.label)
  assert.equal(result.cache, 'MISS')
  assert.equal(result.queueWait, '0.003')
})

test('rejects whitespace or oversized input before making a request', async (t) => {
  let calls = 0
  mockFetch(t, async () => { calls++; return json(fixture.label) })
  await assert.rejects(extractTicket(' \n\t '), ApiError)
  await assert.rejects(extractTicket('中'.repeat(2001)), ApiError)
  assert.equal(calls, 0)
})

test('does not coerce an invalid 200 response into a successful ticket', async (t) => {
  mockFetch(t, async () => json({ ...fixture.label, requested_action: '安排寄修' }))
  await assert.rejects(extractTicket(fixture.text), (err) => err.code === 'invalid_response' && err.status === 200)
  assert.throws(() => validateTicket({ ...fixture.label, extra: true }), ApiError)
  assert.throws(() => validateTicket({ ...fixture.label, order_id: 123 }), ApiError)
  assert.throws(() => validateTicket({ ...fixture.label, product_name: ' '.repeat(4) }), ApiError)
})

test('allows genuine nulls rather than inventing missing entities', () => {
  const sample = { ...fixture.label, order_id: null, product_name: null, reason: null }
  assert.deepEqual(validateTicket(sample), sample)
})

test('shows invalid_model_output without converting its error JSON into a result', async (t) => {
  mockFetch(t, async () => json({ detail: { code: 'invalid_model_output' } }, 502))
  await assert.rejects(extractTicket(fixture.text), (err) => err.status === 502 && err.code === 'invalid_model_output' && err.message.includes('未通过校验'))
})

test('maps queue saturation and validation failures without automatic retries', async (t) => {
  let calls = 0
  mockFetch(t, async () => { calls++; return json({ detail: { code: 'queue_full' } }, 503) })
  await assert.rejects(extractTicket(fixture.text), (err) => err.status === 503 && err.message.includes('队列已满'))
  assert.equal(calls, 1)
  mockFetch(t, async () => json({ detail: [{ type: 'string_too_long' }] }, 422))
  await assert.rejects(extractTicket(fixture.text), (err) => err.status === 422 && err.message.includes('校验'))
})

test('HTML proxy failures and network errors are actionable', async (t) => {
  mockFetch(t, async () => new Response('<h1>Bad Gateway</h1>', { status: 502 }))
  await assert.rejects(getHealth(), (err) => err.status === 502 && err.message.includes('代理'))
  mockFetch(t, async () => { throw new TypeError('fetch failed') })
  await assert.rejects(getHealth(), (err) => err.code === 'network_error')
})

test('health request times out rather than waiting indefinitely', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] })
  mockFetch(t, (_url, { signal }) => new Promise((_resolve, reject) => {
    signal.addEventListener('abort', () => reject(signal.reason), { once: true })
  }))
  const pending = getHealth()
  t.mock.timers.tick(5000)
  await assert.rejects(pending, (err) => err.code === 'client_timeout')
})

test('reports Transformers and vLLM health without choosing a backend for the user', async (t) => {
  for (const backend of ['transformers', 'vllm']) {
    const health = { status: 'ready', backend, structured_output_mode: backend === 'vllm' ? 'json_schema' : 'none' }
    mockFetch(t, async (url) => { assert.equal(url, '/api/health'); return json(health) })
    assert.deepEqual(await getHealth(), health)
  }
})

test('rejects unrelated JSON served from the wrong endpoint', async (t) => {
  mockFetch(t, async () => json({ hello: 'world' }))
  await assert.rejects(getHealth(), (err) => err.message.includes('健康检查'))
})
