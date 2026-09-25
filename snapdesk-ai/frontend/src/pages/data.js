import { api } from '../api.js'
import { charts, dropzone, fmt, h, icon, md, provenance, spinner, table, tabs, toast } from '../ui.js'

let selected = null

function overview(a, id) {
  const stats = [
    ['Rows', a.rows.toLocaleString()],
    ['Columns', a.columns.length],
    ['Missing cells', a.missing_total.toLocaleString()],
    ['Duplicate rows', a.duplicates.toLocaleString()],
    ['Outlier columns', a.outliers.length],
  ]
  const ai = h('div')
  const run = async (refresh) => {
    ai.replaceChildren(spinner('Writing insights with the local model...'))
    try {
      const r = await api.post(`/datasets/${id}/insights?refresh=${refresh}`)
      ai.replaceChildren(md(r.markdown), provenance(r.privacy, r.stats))
    } catch (e) {
      ai.replaceChildren(h('p', { class: 'muted' }, e.message))
    }
  }
  return h(
    'div',
    {},
    h('div', { class: 'stats' }, stats.map(([k, v]) => h('div', { class: 'stat' }, h('b', {}, v), h('span', {}, k)))),
    h('h2', {}, 'Automatic insights'),
    a.insights.map((i) => h('div', { class: `insight ${i.level}` }, i.text)),
    h('h2', { style: 'margin-top:18px' }, 'Language model insights'),
    h('button', { class: 'btn ghost', onclick: () => run(false) }, 'Generate with local AI'),
    ai,
    h('h2', { style: 'margin-top:18px' }, 'Preview'),
    table(Object.keys(a.preview[0] || {}), a.preview.map((r) => Object.values(r))),
  )
}

function columns(a) {
  const rows = a.columns.map((c) => [c.name, c.kind, c.unique, `${c.missing_pct}%`, c.mean ?? '', c.median ?? '', c.min ?? '', c.max ?? ''])
  const out = a.outliers.map((o) => [o.column, o.count, `${o.pct}%`, o.lower, o.upper, o.examples.slice(0, 3).join(', ')])
  const corr = a.correlation
  return h(
    'div',
    {},
    table(['Column', 'Type', 'Unique', 'Missing', 'Mean', 'Median', 'Min', 'Max'], rows),
    h('h2', { style: 'margin-top:18px' }, 'Outliers (IQR rule)'),
    out.length ? table(['Column', 'Count', 'Share', 'Lower bound', 'Upper bound', 'Examples'], out) : h('p', { class: 'muted' }, 'No outliers found.'),
    corr ? h('h2', { style: 'margin-top:18px' }, 'Correlations') : '',
    corr ? table(['', ...corr.columns], corr.columns.map((c, i) => [c, ...corr.values[i].map((v) => (v == null ? '-' : v.toFixed(2)))])) : '',
  )
}

function ask(id) {
  const input = h('input', { type: 'text', placeholder: 'e.g. total Sales by Region, average Price by Product, monthly Sales in 2024' })
  const out = h('div')
  const run = async () => {
    if (!input.value.trim()) return
    out.replaceChildren(spinner('Calculating...'))
    try {
      const r = await api.post(`/datasets/${id}/ask`, { question: input.value })
      out.replaceChildren(md(r.answer), r.table && !r.table.value ? table(r.table.columns, r.table.rows) : '', r.table?.chart ? charts([r.table.chart]) : '', r.plan ? h('p', { class: 'muted', style: 'margin-top:10px' }, 'Query plan: ' + JSON.stringify(r.plan)) : '')
    } catch (e) {
      out.replaceChildren(h('p', {}, e.message))
    }
  }
  input.addEventListener('keydown', (e) => e.key === 'Enter' && run())
  return h('div', {}, h('div', { class: 'row' }, h('div', { class: 'grow' }, input), h('button', { class: 'btn', onclick: run }, 'Ask')), h('p', { class: 'muted' }, 'Answers are calculated with pandas, so numbers are exact and never guessed.'), out)
}

async function detail(panel, dataset) {
  panel.replaceChildren(spinner('Analysing dataset...'))
  try {
    const a = await api.get(`/datasets/${dataset.id}/analysis`)
    panel.replaceChildren(
      h('h2', {}, dataset.name),
      tabs([
        { label: 'Overview', render: () => overview(a, dataset.id) },
        { label: 'Charts', render: () => charts(a.charts) },
        { label: 'Columns and quality', render: () => columns(a) },
        { label: 'Ask', render: () => ask(dataset.id) },
      ]),
    )
  } catch (e) {
    panel.replaceChildren(h('p', {}, e.message))
  }
}

export async function render(root) {
  const list = h('div', { class: 'list' })
  const panel = h('div', { class: 'panel' }, h('p', { class: 'muted' }, 'Choose a dataset to see statistics, charts and to ask questions.'))
  let datasets = []
  const load = async () => {
    datasets = await api.get('/datasets')
    list.replaceChildren(
      ...datasets.map((d) =>
        h(
          'div',
          { class: `item ${selected === d.id ? 'active' : ''}`, onclick: () => ((selected = d.id), load(), detail(panel, d)) },
          h('div', {}, d.name, h('small', {}, `${fmt(d.rows)} rows | ${d.columns} columns`)),
          h('button', { class: 'btn ghost small', onclick: async (e) => (e.stopPropagation(), await api.del(`/datasets/${d.id}`), selected === d.id && (selected = null), panel.replaceChildren(), load()) }, icon('Trash2', 14)),
        ),
      ),
    )
    if (!datasets.length) list.replaceChildren(h('p', { class: 'muted' }, 'No datasets yet.'))
  }
  const upload = async (files) => {
    try {
      const r = await api.upload('/datasets', files)
      r.errors.forEach((e) => toast(`${e.name}: ${e.error}`, 'error'))
      if (r.added.length) selected = r.added[0].id
      await load()
      const dataset = datasets.find((d) => d.id === selected)
      if (dataset) detail(panel, dataset)
    } catch (e) {
      toast(e.message, 'error')
    }
  }
  root.append(h('h1', {}, 'Data Analysis'), h('p', { class: 'muted' }, 'CSV and Excel. Statistics, outliers, trends and charts are created automatically.'), h('div', { class: 'grid2' }, h('div', {}, dropzone('Add CSV or Excel', '.csv,.tsv,.xlsx,.xlsm,.xls', upload), list), panel))
  await load()
  const current = datasets.find((d) => d.id === selected)
  if (current) detail(panel, current)
}
