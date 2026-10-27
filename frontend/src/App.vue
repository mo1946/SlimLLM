<script setup>
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { extractTicket, fields, fieldLabels, getHealth } from './api.js'

const examples = [
  { name: '退货退款', text: '订单001234的蓝牙耳机左耳没有声音，我想退货退款。' },
  { name: '维修申请', text: '订单 771234 的吸尘器充不进去电了，指示灯一直闪红灯，能不能安排寄修或者上门修一下？' },
  { name: '售后咨询', text: '我想了解一下你们的退换货规则。' },
]
const text = ref('')
const loading = ref(false)
const healthLoading = ref(false)
const health = ref(null)
const healthError = ref('')
const error = ref(null)
const result = ref(null)
const elapsed = ref(null)
const liveElapsed = ref(0)
const view = ref('json')
const copyState = ref('')
const completedInput = ref('')
let timer
let copyTimer
const count = computed(() => [...text.value].length)
const canSubmit = computed(() => !loading.value && !!text.value.trim() && count.value <= 2000)
const resultJson = computed(() => result.value ? JSON.stringify(result.value.ticket, null, 2) : '')
const isReady = computed(() => health.value?.status === 'ready')
const statusLabel = computed(() => healthLoading.value ? '检查中' : isReady.value ? '服务就绪' : '服务未连接')
const backendLabel = computed(() => health.value?.backend === 'vllm' ? 'vLLM' : health.value?.backend === 'transformers' ? 'Transformers' : '—')
const modeLabel = computed(() => health.value?.structured_output_mode === 'json_schema' ? 'JSON Schema 约束' : health.value ? '生成后校验' : '—')

async function refreshHealth() {
  if (healthLoading.value) return
  healthLoading.value = true
  healthError.value = ''
  try { health.value = await getHealth() } catch (err) {
    health.value = null
    healthError.value = err.message
  } finally { healthLoading.value = false }
}

async function submit() {
  if (!canSubmit.value) return
  loading.value = true
  error.value = null
  result.value = null
  copyState.value = ''
  elapsed.value = null
  liveElapsed.value = 0
  const started = performance.now()
  timer = setInterval(() => { liveElapsed.value = (performance.now() - started) / 1000 }, 100)
  try {
    const submittedText = text.value
    result.value = await extractTicket(submittedText)
    completedInput.value = submittedText
  } catch (err) {
    error.value = { message: err.message, status: err.status, code: err.code }
  } finally {
    elapsed.value = (performance.now() - started) / 1000
    clearInterval(timer)
    loading.value = false
  }
}

function useExample(example) {
  text.value = example.text
  result.value = null
  error.value = null
  elapsed.value = null
  copyState.value = ''
}

async function copyJson() {
  clearTimeout(copyTimer)
  try {
    await navigator.clipboard.writeText(resultJson.value)
    copyState.value = '已复制'
  } catch { copyState.value = '复制失败，请手动选择 JSON' }
  copyTimer = setTimeout(() => { copyState.value = '' }, 2500)
}

onMounted(refreshHealth)
onUnmounted(() => { clearInterval(timer); clearTimeout(copyTimer) })
</script>

<template>
  <div class="app-shell">
    <header class="topbar">
      <a class="brand" href="/" aria-label="SlimLLM 首页">
        <span class="brand-mark" aria-hidden="true">ϟ</span><span>Slim<span class="brand-light">LLM</span></span>
        <span class="brand-divider"></span><span class="workspace-label">演示工作台</span>
      </a>
      <div class="topbar-right">
        <span class="local-label"><span class="small-square"></span> 本地模型推理</span>
        <a href="http://127.0.0.1:8000/docs" target="_blank" rel="noopener noreferrer" class="docs-link">API 文档 <span aria-hidden="true">↗</span></a>
      </div>
    </header>

    <main>
      <section class="intro">
        <div><p class="eyebrow"><span></span> AFTER-SALES INTELLIGENCE</p>
          <h1>一句工单，<span>清晰拆解。</span></h1>
          <p class="intro-description">将自然语言中的售后诉求，转为可用的结构化数据。</p>
        </div>
        <div class="model-chip"><span class="model-symbol" aria-hidden="true">✦</span><div><strong>Qwen3-1.7B</strong><span>LoRA 微调 · 七字段抽取</span></div></div>
      </section>

      <section class="service-bar" aria-label="后端服务状态">
        <div class="status-group"><span class="status-pill" :class="{ online: isReady, checking: healthLoading }" role="status"><span class="status-dot"></span>{{ statusLabel }}</span><span class="service-separator"></span><span class="service-item">后端 <strong>{{ backendLabel }}</strong></span><span class="service-item mode-item">输出 <strong>{{ modeLabel }}</strong></span></div>
        <button class="refresh-button" :disabled="healthLoading" @click="refreshHealth" aria-label="刷新服务状态"><span :class="{ spinning: healthLoading }" aria-hidden="true">↻</span><span>刷新状态</span></button>
      </section>
      <p v-if="healthError" class="health-warning" role="status">{{ healthError }}</p>

      <div class="workspace-grid">
        <section class="panel input-panel">
          <div class="panel-heading"><div class="heading-group"><span class="step-number">01</span><h2>输入工单</h2></div><span class="panel-caption">自然语言</span></div>
          <form @submit.prevent="submit" class="input-form">
            <label for="ticket-text" class="input-label">用户说了什么？</label>
            <div class="textarea-wrap" :class="{ invalid: count > 2000 }">
              <textarea id="ticket-text" v-model="text" :disabled="loading" placeholder="例如：订单001234的蓝牙耳机左耳没有声音，我想退货退款。" @keydown.ctrl.enter.prevent="submit" @keydown.meta.enter.prevent="submit" aria-describedby="input-help character-count" spellcheck="false"></textarea>
              <div class="textarea-footer"><span id="input-help">保留原始描述，模型将识别售后诉求</span><span id="character-count" :class="{ over: count > 2000 }">{{ count }} <span>/ 2000</span></span></div>
            </div>
            <div class="examples"><span class="examples-label">试试示例</span><div class="example-buttons"><button v-for="example in examples" :key="example.name" type="button" :disabled="loading" @click="useExample(example)"><span aria-hidden="true">↗</span>{{ example.name }}</button></div></div>
            <div class="submit-row"><button type="submit" class="primary-button" :disabled="!canSubmit"><span v-if="loading" class="spinner" aria-hidden="true"></span><span v-else aria-hidden="true">✦</span>{{ loading ? '正在抽取…' : '抽取结构化 JSON' }}<span v-if="!loading" class="button-arrow" aria-hidden="true">→</span></button><span class="keyboard-hint">Ctrl + Enter</span></div>
          </form>
          <div class="input-note"><span aria-hidden="true">ⓘ</span><p>示例为手写演示工单。结果反映模型判断，格式合法不代表业务语义一定正确。</p></div>
        </section>

        <section class="panel output-panel" :aria-busy="loading" aria-label="结构化抽取结果">
          <div class="panel-heading"><div class="heading-group"><span class="step-number">02</span><h2>抽取结果</h2></div><span v-if="result" class="result-pill">响应已校验</span><span v-else class="panel-caption">JSON OUTPUT</span></div>
          <div v-if="loading" class="empty-state loading-state" role="status"><div class="empty-icon"><span class="spinner"></span></div><h3>正在理解这条工单</h3><p>本地模型正在生成结构化结果，请稍候。</p><span class="elapsed-live">{{ liveElapsed.toFixed(1) }} s</span></div>
          <div v-else-if="error" class="error-state" role="alert"><div class="error-icon">!</div><h3>本次抽取未完成</h3><p>{{ error.message }}</p><code v-if="error.status || error.code">{{ error.status ? `HTTP ${error.status}` : '' }}{{ error.status && error.code ? ' · ' : '' }}{{ error.code || '' }}</code><button class="secondary-button" :disabled="!canSubmit" @click="submit">重新尝试 <span aria-hidden="true">↻</span></button></div>
          <template v-else-if="result">
            <div class="result-toolbar"><div class="view-tabs" aria-label="结果显示方式"><button :class="{ selected: view === 'json' }" :aria-pressed="view === 'json'" @click="view = 'json'">JSON</button><button :class="{ selected: view === 'fields' }" :aria-pressed="view === 'fields'" @click="view = 'fields'">字段视图</button></div><button class="copy-button" @click="copyJson">{{ copyState || '复制 JSON' }} <span aria-hidden="true">⧉</span></button></div>
            <div v-if="view === 'json'" class="json-view" tabindex="0" aria-label="完整 JSON 结果"><pre><span class="json-brace">{</span><template v-for="(field, index) in fields" :key="field">
  <span class="json-key">{{ JSON.stringify(field) }}</span><span class="json-brace">: </span><span :class="result.ticket[field] === null ? 'json-null' : 'json-value'">{{ JSON.stringify(result.ticket[field]) }}</span><span v-if="index < fields.length - 1" class="json-brace">,</span></template>
<span class="json-brace">}</span></pre></div>
            <dl v-else class="field-list"><div v-for="field in fields" :key="field"><dt>{{ fieldLabels[field] }}<span>{{ field }}</span></dt><dd :class="{ 'null-value': result.ticket[field] === null }">{{ result.ticket[field] === null ? 'null' : result.ticket[field] }}</dd></div></dl>
            <p v-if="text !== completedInput" class="stale-notice">输入已更改，当前展示的是上一次请求结果。</p>
          </template>
          <div v-else class="empty-state"><div class="empty-icon" aria-hidden="true"><span>{ }</span><i>✦</i></div><h3>让非结构化的信息，井然有序</h3><p>在左侧输入一条工单，或选择示例。<br>七个字段，将在这里为你展开。</p><span class="empty-tag">等待你的第一条工单</span></div>
          <div class="result-footer"><span><span class="footer-dot" :class="{ complete: !!result }"></span>{{ result ? '七字段 Schema 检查通过' : loading ? '请求处理中' : error ? '结果未通过或请求失败' : '等待抽取' }}</span><span class="request-time">{{ elapsed !== null ? `请求耗时 ${elapsed.toFixed(3)} s` : '请求耗时 —' }}</span></div>
          <div v-if="result" class="request-meta"><span>缓存 {{ result.cache || '未提供' }}</span><span>排队 {{ result.queueWait !== null && Number.isFinite(Number(result.queueWait)) ? `${Number(result.queueWait).toFixed(3)} s` : '未提供' }}</span></div>
        </section>
      </div>

      <section class="schema-strip" aria-label="抽取字段说明"><span class="schema-label">一个工单，七个维度</span><div><span v-for="field in fields" :key="field">{{ fieldLabels[field] }}</span></div></section>
      <footer class="page-footer"><span>SlimLLM <span class="footer-slash">/</span> 轻量模型，清晰表达。</span><span>本地演示 · Qwen3 + LoRA · FastAPI</span></footer>
    </main>
  </div>
</template>
