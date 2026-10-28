import schema from '../../data/ticket.schema.json' with { type: 'json' }

export const fields = Object.keys(schema.properties)
export const fieldLabels = {
  intent: '用户意图', order_id: '订单编号', product_name: '商品名称',
  issue_type: '问题类型', priority: '处理优先级', requested_action: '期望操作', reason: '问题描述',
}

const errorMessages = {
  invalid_model_output: '模型输出未通过校验，请尝试重新表述工单。',
  backend_request_rejected: '推理后端拒绝了请求配置，请检查后端日志。',
  queue_full: '请求队列已满，请稍后再试。',
  queue_wait_timeout: '排队等待超时，请稍后再试。',
  backend_timeout: '推理后端响应超时，请稍后再试。',
  backend_unavailable: '推理后端暂不可用，请检查模型服务。',
  model_not_ready: '模型尚未就绪，请等待服务加载完成。',
  service_stopping: '服务正在停止，请稍后重试。',
  gpu_memory_unavailable: 'GPU 显存不足，请检查模型服务。',
  input_token_limit: '输入分词后超过模型上限，请缩短工单。',
  inference_failed: '模型推理失败，请查看后端日志。',
}

export class ApiError extends Error {
  constructor(message, status = null, code = null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
  }
}

async function request(path, { timeout = 120_000, ...options } = {}) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeout)
  try {
    const response = await fetch(`/api${path}`, { ...options, signal: controller.signal })
    let data
    try { data = await response.json() } catch {
      throw new ApiError('未收到 JSON 响应，请检查 API 服务与本地代理。', response.status)
    }
    if (!response.ok) {
      const code = data?.detail?.code
      const message = errorMessages[code] || (response.status === 422
        ? '输入未通过校验，请输入 1～2000 字的有效工单。'
        : '请求失败，请检查 API 服务并稍后重试。')
      throw new ApiError(message, response.status, code)
    }
    return { data, headers: response.headers }
  } catch (error) {
    if (error instanceof ApiError) throw error
    if (controller.signal.aborted) throw new ApiError('请求等待超时；后端可能仍在处理，请稍后再试。', null, 'client_timeout')
    throw new ApiError('无法连接服务，请确认 FastAPI 已启动并刷新服务状态。', null, 'network_error')
  } finally {
    clearTimeout(timer)
  }
}

// Check the current seven-field response contract, without transforming predictions.
export function validateTicket(data) {
  if (!data || typeof data !== 'object' || Array.isArray(data)
      || Object.keys(data).length !== fields.length || fields.some((key) => !Object.hasOwn(data, key))) {
    throw new ApiError('服务返回了不符合七字段契约的结果。', 200, 'invalid_response')
  }
  for (const key of fields) {
    const value = data[key]
    const rule = schema.properties[key]
    if (rule.enum) {
      if (!rule.enum.includes(value)) throw new ApiError('返回结果包含非法枚举值。', 200, 'invalid_response')
    } else if (value !== null) {
      const stringRule = rule.anyOf.find((branch) => branch.type === 'string')
      if (typeof value !== 'string' || [...value].length < stringRule.minLength
          || [...value].length > stringRule.maxLength || !new RegExp(stringRule.pattern).test(value)) {
        throw new ApiError('返回结果的文本字段不符合 Schema。', 200, 'invalid_response')
      }
    }
  }
  return Object.fromEntries(fields.map((key) => [key, data[key]]))
}

export async function extractTicket(text) {
  if (!text.trim() || [...text].length > 2000) throw new ApiError('请输入 1～2000 字的有效工单。')
  const { data, headers } = await request('/extract', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }),
  })
  return {
    ticket: validateTicket(data),
    cache: headers.get('X-SlimLLM-Cache'),
    queueWait: headers.get('X-SlimLLM-Queue-Wait-Seconds'),
  }
}

export async function getHealth() {
  const { data } = await request('/health', { timeout: 5000 })
  if (!data || !['ready', 'unavailable'].includes(data.status) || typeof data.backend !== 'string') {
    throw new ApiError('健康检查返回异常，请检查后端服务。')
  }
  return data
}
