import { api } from '../api.js'
import { h } from '../ui.js'

export async function render(root) {
  const [system, docs, sets] = await Promise.all([api.cached('/system'), api.get('/documents'), api.get('/datasets')])
  const hw = system.hardware
  const npu = hw.npus[0]?.name || 'No NPU detected'
  const spec = [
    ['Device', hw.device],
    ['Processor', hw.cpu.name],
    ['Memory', `${hw.memory_gb} GB`],
    ['NPU', npu],
    ['ONNX Runtime', hw.onnxruntime.installed ? `${hw.onnxruntime.package} ${hw.onnxruntime.version}` : 'not installed'],
    ['Language model', system.llm.ready ? `${system.llm.provider} ${system.llm.model}` : 'not ready'],
  ]
  root.append(
    h('section', { class: 'hero' }, h('h1', {}, 'Your documents and data, analysed on your own PC.'), h('p', { class: 'muted' }, 'SnapDesk AI reads files, answers questions and builds charts locally. Every result shows where it ran.')),
    h(
      'div',
      { class: 'grid2' },
      h('div', { class: 'panel' }, h('h2', {}, 'This PC'), h('dl', { class: 'spec' }, spec.map(([k, v]) => [h('dt', {}, k), h('dd', {}, v)]).flat())),
      h(
        'div',
        {},
        h(
          'div',
          { class: 'panel' },
          h('h2', {}, 'Workspace'),
          h('p', {}, `${docs.length} document(s) and ${sets.length} dataset(s) loaded.`),
          h('div', { class: 'row' }, h('a', { class: 'btn', href: '#/documents' }, 'Add documents'), h('a', { class: 'btn ghost', href: '#/data' }, 'Add data'), h('a', { class: 'btn ghost', href: '#/assistant' }, 'Ask the assistant')),
        ),
        h(
          'div',
          { class: 'panel' },
          h('h2', {}, 'Setup checklist'),
          system.checklist.map((c) =>
            h('div', { class: 'check' }, h('span', { class: c.ok ? 'ok' : 'no' }, c.ok ? 'OK' : '!'), h('div', {}, h('b', {}, c.label), h('div', { class: 'muted' }, c.ok ? c.detail : c.hint || c.detail))),
          ),
        ),
      ),
    ),
  )
}
