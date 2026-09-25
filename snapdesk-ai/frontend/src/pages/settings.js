import { api } from '../api.js'
import { h, toast } from '../ui.js'

const groups = [
  ['Language model', [
    ['llm_provider', 'Provider', ['auto', 'ollama', 'openai', 'genie', 'none']],
    ['ollama_url', 'Ollama URL'], ['ollama_model', 'Ollama model'], ['ollama_vision_model', 'Ollama vision model'],
    ['openai_url', 'OpenAI-compatible URL (llama-server, LM Studio)'], ['openai_model', 'OpenAI-compatible model name'], ['openai_api_key', 'API key (leave empty for local)', 'password'],
    ['genie_bin', 'Genie executable path (genie-t2t-run)'], ['genie_config', 'Genie config path'],
  ]],
  ['Acceleration and models', [
    ['accelerator', 'Preferred accelerator for ONNX models', ['auto', 'npu', 'gpu', 'cpu']],
    ['embedding_model', 'Embedding model path'], ['vision_classifier', 'Image classifier path'],
  ]],
  ['Documents and generation', [['chunk_size', 'Passage size (characters)', 'number'], ['top_k', 'Passages per answer', 'number'], ['temperature', 'Temperature', 'number'], ['max_tokens', 'Max answer tokens', 'number']]],
]

export async function render(root) {
  const values = await api.get('/settings')
  const inputs = {}
  const field = ([key, label, extra]) => {
    const control = Array.isArray(extra)
      ? h('select', {}, extra.map((o) => h('option', { value: o, selected: values[key] === o }, o)))
      : h('input', { type: extra || 'text', value: values[key], step: 'any' })
    inputs[key] = control
    return h('div', { class: 'field' }, h('label', {}, label), control)
  }
  const block = h('input', { type: 'checkbox', checked: values.block_external })
  const save = async () => {
    const patch = Object.fromEntries(Object.entries(inputs).map(([k, el]) => [k, el.value]))
    patch.block_external = block.checked
    try {
      await api.put('/settings', patch)
      toast('Settings saved')
    } catch (e) {
      toast(e.message, 'error')
    }
  }
  root.append(
    h('h1', {}, 'Settings'),
    h('div', { class: 'panel' }, h('h2', {}, 'Privacy'), h('label', { class: 'row' }, block, 'Block any model endpoint that is not on this PC (recommended)'), h('p', { class: 'muted' }, 'When this is on, only localhost endpoints are used. Turning it off allows remote servers and every answer will be marked as external.'), h('button', { class: 'btn ghost', onclick: async () => { if (confirm('Delete all uploaded documents and datasets from this PC?')) { await api.del('/data'); toast('Local data deleted') } } }, 'Delete all local data')),
    ...groups.map(([title, fields]) => h('div', { class: 'panel' }, h('h2', {}, title), fields.map(field))),
    h('button', { class: 'btn', onclick: save }, 'Save settings'),
  )
}
