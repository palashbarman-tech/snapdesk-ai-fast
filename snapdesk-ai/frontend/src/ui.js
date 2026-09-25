import { Chart, registerables } from 'chart.js'
import DOMPurify from 'dompurify'
import { marked } from 'marked'
import { createElement, House, FileText, Sheet, MessageSquare, Cpu, Settings, Upload, Trash2, Send, Image, Monitor, Shield } from 'lucide'

Chart.register(...registerables)

const icons = { House, FileText, Sheet, MessageSquare, Cpu, Settings, Upload, Trash2, Send, Image, Monitor, Shield }

export function icon(name, size = 18) {
  return createElement(icons[name], { width: size, height: size, 'stroke-width': 1.8 })
}

export function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag)
  for (const [key, value] of Object.entries(attrs || {})) {
    if (key.startsWith('on')) node.addEventListener(key.slice(2), value)
    else if (key === 'class') node.className = value
    else if (value !== false && value != null) node.setAttribute(key, value === true ? '' : value)
  }
  for (const child of children.flat()) {
    if (child == null || child === false) continue
    node.append(child instanceof Node ? child : document.createTextNode(String(child)))
  }
  return node
}

export function md(text) {
  const box = h('div', { class: 'md' })
  box.innerHTML = DOMPurify.sanitize(marked.parse(text || '', { breaks: true }))
  return box
}

export function toast(message, kind = 'info') {
  const node = h('div', { class: `toast ${kind}` }, message)
  document.body.append(node)
  setTimeout(() => node.remove(), 4200)
}

export const spinner = (text = 'Working locally...') => h('div', { class: 'spinner' }, h('span', { class: 'dot' }), text)

export function fmt(value, digits = 2) {
  if (value == null) return '-'
  if (typeof value !== 'number') return String(value)
  return Math.abs(value) >= 1000 ? Math.round(value).toLocaleString() : Number(value.toFixed(digits)).toString()
}

export function table(columns, rows) {
  return h(
    'div',
    { class: 'table-wrap' },
    h('table', {}, h('thead', {}, h('tr', {}, columns.map((c) => h('th', {}, c)))), h('tbody', {}, rows.map((r) => h('tr', {}, r.map((v) => h('td', {}, fmt(v))))))),
  )
}

export function tabs(items, initial = 0) {
  const bar = h('div', { class: 'tabs' })
  const body = h('div', { class: 'tab-body' })
  const show = (index) => {
    bar.querySelectorAll('button').forEach((b, i) => b.classList.toggle('active', i === index))
    body.replaceChildren(items[index].render())
  }
  items.forEach((item, index) => bar.append(h('button', { onclick: () => show(index) }, item.label)))
  show(initial)
  return h('div', {}, bar, body)
}

export function dropzone(label, accept, onFiles) {
  const input = h('input', { type: 'file', multiple: true, accept, hidden: true })
  const zone = h('div', { class: 'drop', tabindex: 0 }, icon('Upload', 22), h('strong', {}, label), h('span', {}, 'Drop files here or click to choose. Files stay on this PC.'), input)
  const pick = (files) => files.length && onFiles([...files])
  zone.addEventListener('click', () => input.click())
  zone.addEventListener('keydown', (e) => e.key === 'Enter' && input.click())
  input.addEventListener('change', () => pick(input.files))
  zone.addEventListener('dragover', (e) => (e.preventDefault(), zone.classList.add('over')))
  zone.addEventListener('dragleave', () => zone.classList.remove('over'))
  zone.addEventListener('drop', (e) => (e.preventDefault(), zone.classList.remove('over'), pick(e.dataTransfer.files)))
  return zone
}

export function provenance(privacy, stats = {}, totalMs) {
  const parts = [privacy?.label || 'Local']
  if (stats.ttft_ms != null) parts.push(`first token ${stats.ttft_ms} ms`)
  if (stats.tokens_per_sec) parts.push(`${stats.tokens_per_sec} tok/s`)
  if (totalMs != null) parts.push(`${(totalMs / 1000).toFixed(1)} s total`)
  return h('div', { class: `prov ${privacy?.external ? 'external' : 'local'}` }, h('span', { class: 'prov-dot' }), parts.join('  |  '))
}

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim()

export function chartCard(spec) {
  const canvas = h('canvas')
  const card = h('figure', { class: 'chart' }, h('figcaption', {}, spec.title), h('div', { class: 'chart-box' }, canvas))
  requestAnimationFrame(() => {
    const accent = css('--accent')
    const text = css('--muted')
    const grid = css('--line')
    const scatter = spec.kind === 'scatter'
    new Chart(canvas, {
      type: spec.kind,
      data: {
        labels: spec.labels,
        datasets: spec.series.map((s) => ({
          label: s.label,
          data: s.data,
          backgroundColor: scatter ? accent + 'aa' : accent + 'cc',
          borderColor: accent,
          borderWidth: spec.kind === 'line' ? 2 : 0,
          pointRadius: scatter ? 3 : 0,
          tension: 0.25,
          borderRadius: 3,
        })),
      },
      options: {
        indexAxis: spec.horizontal ? 'y' : 'x',
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: text, maxTicksLimit: 8 }, grid: { color: grid }, title: { display: !!spec.x_title, text: spec.x_title, color: text } },
          y: { ticks: { color: text }, grid: { color: grid }, title: { display: !!spec.y_title, text: spec.y_title, color: text } },
        },
      },
    })
  })
  return card
}

export const charts = (list) => h('div', { class: 'chart-grid' }, list.map(chartCard))
