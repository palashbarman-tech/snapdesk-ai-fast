import { api } from '../api.js'
import { dropzone, h, icon, md, provenance, spinner, tabs, toast } from '../ui.js'

let selected = null

function sources(list) {
  return list.map((s) => h('div', { class: 'source' }, h('b', {}, `[${s.n ?? ''}] ${s.doc}${s.page ? ` | page ${s.page}` : ''}`), h('div', {}, highlight(s.text, s.terms))))
}

function highlight(text, terms = []) {
  if (!terms.length) return text
  const box = document.createElement('span')
  const pattern = new RegExp(`(${terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`, 'gi')
  text.split(pattern).forEach((part, i) => box.append(i % 2 ? h('mark', {}, part) : part))
  return box
}

function overview(doc) {
  const box = h('div', {}, spinner('Summarizing on this PC...'))
  api
    .get(`/documents/${doc.id}/analysis`)
    .then((a) => {
      const entities = Object.entries(a.entities).map(([k, v]) => h('div', {}, h('b', {}, k + ': '), v.join(', ')))
      box.replaceChildren(
        md(a.markdown),
        a.note ? h('p', { class: 'muted' }, a.note) : '',
        provenance(a.privacy, a.stats),
        h('h3', { style: 'margin-top:16px' }, 'Important information'),
        entities.length ? entities : h('p', { class: 'muted' }, 'No dates, amounts or contacts found.'),
        h('h3', { style: 'margin-top:16px' }, 'Keywords'),
        h('p', {}, a.keywords.join(', ')),
        h('button', { class: 'btn ghost small', onclick: () => api.get(`/documents/${doc.id}/analysis?refresh=true`).then(() => detail(doc)) }, 'Summarize again'),
      )
    })
    .catch((e) => box.replaceChildren(h('p', {}, e.message)))
  return box
}

function ask(doc) {
  const input = h('input', { type: 'text', placeholder: 'Ask a question about your documents' })
  const out = h('div')
  const run = async () => {
    if (!input.value.trim()) return
    out.replaceChildren(spinner())
    try {
      const r = await api.post('/documents/ask', { question: input.value, doc_ids: doc ? [doc.id] : null })
      out.replaceChildren(md(r.answer), provenance(r.privacy, r.stats), r.error ? h('p', { class: 'muted' }, r.error) : '', h('h3', { style: 'margin-top:14px' }, 'Sources'), sources(r.sources))
    } catch (e) {
      out.replaceChildren(h('p', {}, e.message))
    }
  }
  input.addEventListener('keydown', (e) => e.key === 'Enter' && run())
  return h('div', {}, h('div', { class: 'row' }, h('div', { class: 'grow' }, input), h('button', { class: 'btn', onclick: run }, 'Ask')), out)
}

function search(doc) {
  const input = h('input', { type: 'text', placeholder: 'Search inside the document' })
  const out = h('div')
  const run = async () => {
    const hits = await api.get(`/documents-search?q=${encodeURIComponent(input.value)}&doc_ids=${doc.id}`)
    out.replaceChildren(h('p', { class: 'muted' }, `${hits.length} match(es)`), sources(hits.map((x, i) => ({ ...x, n: i + 1 }))))
  }
  input.addEventListener('keydown', (e) => e.key === 'Enter' && run())
  return h('div', {}, h('div', { class: 'row' }, h('div', { class: 'grow' }, input), h('button', { class: 'btn', onclick: run }, 'Search')), out)
}

let panel
function detail(doc) {
  panel.replaceChildren(
    h('div', { class: 'row', style: 'justify-content:space-between' }, h('h2', {}, doc.name), h('span', { class: 'muted' }, `${doc.words.toLocaleString()} words | ${doc.chunks} passages | ${doc.embedded ? 'semantic + keyword search' : 'keyword search'}`)),
    tabs([
      { label: 'Overview', render: () => overview(doc) },
      { label: 'Ask', render: () => ask(doc) },
      { label: 'Search', render: () => search(doc) },
    ]),
  )
}

export async function render(root) {
  const list = h('div', { class: 'list' })
  panel = h('div', { class: 'panel' }, h('p', { class: 'muted' }, 'Choose a document to see its summary, ask questions or search.'))
  const load = async () => {
    const docs = await api.get('/documents')
    list.replaceChildren(
      ...docs.map((d) =>
        h(
          'div',
          { class: `item ${selected === d.id ? 'active' : ''}`, onclick: () => ((selected = d.id), load(), detail(d)) },
          h('div', {}, d.name, h('small', {}, `${d.kind.toUpperCase()} | ${d.words.toLocaleString()} words`)),
          h('button', { class: 'btn ghost small', title: 'Delete', onclick: async (e) => (e.stopPropagation(), await api.del(`/documents/${d.id}`), selected === d.id && (selected = null), panel.replaceChildren(), load()) }, icon('Trash2', 14)),
        ),
      ),
    )
    if (!docs.length) list.replaceChildren(h('p', { class: 'muted' }, 'No documents yet.'))
    return docs
  }
  const upload = async (files) => {
    try {
      const r = await api.upload('/documents', files)
      r.errors.forEach((e) => toast(`${e.name}: ${e.error}`, 'error'))
      if (r.added.length) (selected = r.added[0].id), toast(`${r.added.length} document(s) added`)
      const docs = await load()
      const doc = docs.find((d) => d.id === selected)
      if (doc) detail(doc)
    } catch (e) {
      toast(e.message, 'error')
    }
  }
  root.append(h('h1', {}, 'Documents'), h('p', { class: 'muted' }, 'PDF, DOCX and TXT. Summaries, questions and search run on this PC.'), h('div', { class: 'grid2' }, h('div', {}, dropzone('Add documents', '.pdf,.txt,.md,.docx', upload), list), panel))
  const docs = await load()
  const first = docs.find((d) => d.id === selected)
  if (first) detail(first)
}
