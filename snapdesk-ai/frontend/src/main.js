import '@fontsource-variable/instrument-sans'
import './styles.css'
import { api } from './api.js'
import { h, icon } from './ui.js'
import * as home from './pages/home.js'
import * as documents from './pages/documents.js'
import * as data from './pages/data.js'
import * as assistant from './pages/assistant.js'
import * as hardware from './pages/hardware.js'
import * as settings from './pages/settings.js'

const pages = [
  { id: 'home', label: 'Home', icon: 'House', page: home },
  { id: 'documents', label: 'Documents', icon: 'FileText', page: documents },
  { id: 'data', label: 'Data Analysis', icon: 'Sheet', page: data },
  { id: 'assistant', label: 'AI Assistant', icon: 'MessageSquare', page: assistant },
  { id: 'hardware', label: 'Hardware', icon: 'Cpu', page: hardware },
  { id: 'settings', label: 'Settings', icon: 'Settings', page: settings },
]

document.documentElement.dataset.theme = localStorage.getItem('theme') || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')

const content = h('div', { class: 'main' })
const nav = h('nav', { class: 'nav' })
const themeButton = h('button', { class: 'btn ghost small', onclick: toggleTheme }, 'Switch theme')

function toggleTheme() {
  const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'
  document.documentElement.dataset.theme = next
  localStorage.setItem('theme', next)
  route()
}

function topline() {
  const line = h('div', { class: 'topline' })
  api
    .cached('/system')
    .then((s) => {
    const npuActive = s.embedding.accelerator === 'npu'
      line.append(
      h('span', { class: `chip ${s.privacy.external ? 'bad' : 'good'}` }, h('i'), s.privacy.external ? 'External service in use' : 'Processing stays on this PC'),
      h('span', { class: `chip ${s.llm.ready ? 'good' : 'warn'}` }, h('i'), s.llm.ready ? `LLM: ${s.llm.provider} ${s.llm.model}` : 'No language model'),
      h('span', { class: `chip ${npuActive ? 'good' : 'warn'}` }, h('i'), npuActive ? 'NPU active for embeddings' : 'NPU not in use'),
      )
    })
    .catch(() => line.append(h('span', { class: 'chip bad' }, h('i'), 'Backend not reachable on port 8000')))
  return line
}

async function route() {
  const id = location.hash.slice(2) || 'home'
  const current = pages.find((p) => p.id === id) || pages[0]
  nav.querySelectorAll('a').forEach((a) => a.classList.toggle('active', a.dataset.id === current.id))
  content.replaceChildren(topline())
  const body = h('div')
  content.append(body)
  await current.page.render(body)
}

document.querySelector('#app').append(
  h(
    'div',
    { class: 'shell' },
    h(
      'aside',
      { class: 'side' },
      h('div', { class: 'brand' }, 'Snap', h('span', {}, 'Desk'), ' AI'),
      nav,
      h('div', { class: 'side-foot' }, h('span', { class: 'muted' }, 'Local-first AI workspace'), themeButton),
    ),
    content,
  ),
)
pages.forEach((p) => nav.append(h('a', { href: `#/${p.id}`, 'data-id': p.id }, icon(p.icon), p.label)))
addEventListener('hashchange', route)
route()
