import { api } from '../api.js'
import { chartCard, h, spinner, table, toast } from '../ui.js'

function resultBlock(result) {
  if (result.kind === 'llm') {
    return h('div', { class: 'panel' }, h('h3', {}, `Language model: ${result.provider} ${result.model}`), h('p', {}, `First token ${result.first_token_ms} ms | total ${result.total_ms} ms | ${result.tokens_per_sec ?? 'not reported'} tokens/s`), h('p', { class: 'muted' }, `${result.time}. ${result.note}`))
  }
  const ok = result.rows.filter((r) => r.status === 'ok')
  const rows = result.rows.map((r) => [r.target.toUpperCase(), r.status === 'ok' ? 'measured' : r.status, r.mean_ms ?? '', r.median_ms ?? '', r.p95_ms ?? '', r.load_ms ?? '', r.status === 'ok' ? r.providers.join(', ') : r.reason])
  return h(
    'div',
    { class: 'panel' },
    h('h3', {}, `${result.model} | ${result.iterations} runs`),
    table(['Target', 'Result', 'Mean ms', 'Median ms', 'p95 ms', 'Load ms', 'Providers / reason'], rows),
    ok.length ? chartCard({ title: 'Mean latency per inference (ms, lower is better)', kind: 'bar', labels: ok.map((r) => r.target.toUpperCase()), series: [{ label: 'ms', data: ok.map((r) => r.mean_ms) }] }) : '',
    h('p', { class: 'muted' }, `${result.time} | ${result.device}`),
  )
}

export async function render(root) {
  const [system, bench, ops] = await Promise.all([api.cached('/system'), api.get('/benchmarks'), api.get('/metrics')])
  const hw = system.hardware
  const select = h('select', {}, bench.models.length ? bench.models.map((m) => h('option', { value: m }, m)) : h('option', { value: '' }, 'No .onnx models found in models/'))
  const output = h('div', {}, bench.results.map(resultBlock))
  const run = async (button, body) => {
    button.disabled = true
    output.prepend(spinner('Benchmark running. The first NPU run can take a while.'))
    try {
      const result = await api.post('/benchmarks/run', body)
      output.replaceChildren(resultBlock(result), ...bench.results.map(resultBlock))
    } catch (e) {
      output.firstChild?.remove()
      toast(e.message, 'error')
    }
    button.disabled = false
  }
  const onnxButton = h('button', { class: 'btn', onclick: () => run(onnxButton, { kind: 'onnx', model: select.value, iterations: 30 }) }, 'Run CPU / GPU / NPU benchmark')
  const llmButton = h('button', { class: 'btn ghost', onclick: () => run(llmButton, { kind: 'llm' }) }, 'Benchmark language model')
  root.append(
    h('h1', {}, 'Hardware and performance'),
    h('p', { class: 'muted' }, 'Everything here is detected or measured on this PC. Nothing is estimated.'),
    h(
      'div',
      { class: 'grid2' },
      h(
        'div',
        { class: 'panel' },
        h('h2', {}, 'Detected hardware'),
        h('dl', { class: 'spec' }, [
          ['Device', hw.device], ['Processor', hw.cpu.name], ['Cores / threads', `${hw.cpu.cores} / ${hw.cpu.threads}`], ['Memory', `${hw.memory_gb} GB`],
          ['GPU', hw.gpus.map((g) => g.name).join(', ') || 'unknown'], ['NPU', hw.npus.map((n) => n.name).join(', ') || 'not detected'],
          ['Python', `${hw.python.version} (${hw.python.machine})`], ['ONNX providers', hw.onnxruntime.providers.join(', ') || 'none'],
          ['Embedding model', system.embedding.ready ? `running on ${system.embedding.accelerator.toUpperCase()}, loaded in ${system.embedding.load_ms} ms` : system.embedding.error || 'not loaded'],
        ].flatMap(([k, v]) => [h('dt', {}, k), h('dd', {}, v)])),
      ),
      h('div', { class: 'panel' }, h('h2', {}, 'Benchmark'), h('div', { class: 'field' }, h('label', {}, 'ONNX model'), select), h('div', { class: 'row' }, onnxButton, llmButton), h('p', { class: 'muted', style: 'margin-top:10px' }, 'Each target is tested strictly. If the NPU provider cannot start, the row says why instead of showing a number.')),
    ),
    h('h2', {}, 'Results'),
    bench.results.length ? '' : h('p', { class: 'muted' }, 'No benchmarks yet.'),
    output,
    h('h2', { style: 'margin-top:18px' }, 'Recent operations in this app'),
    ops.length ? table(['Time', 'Operation', 'Backend', 'ms', 'Detail'], ops.map((o) => [o.time, o.name, o.backend, o.ms, o.detail])) : h('p', { class: 'muted' }, 'Upload a file or ask a question to see timings.'),
  )
}
