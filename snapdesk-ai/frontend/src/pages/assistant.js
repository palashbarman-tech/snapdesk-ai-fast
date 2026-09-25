import { chatStream } from '../api.js'
import { charts, h, icon, md, provenance, table, toast } from '../ui.js'

const history = []
let thread
let image = null

function sourceCards(list) {
  return list.map((s) => h('div', { class: 'source' }, h('b', {}, `[${s.n}] ${s.doc}${s.page ? ` | page ${s.page}` : ''}`), h('div', {}, s.text)))
}

async function send(text, attach) {
  const bubble = h('div', { class: 'msg user' }, text, attach ? h('div', { class: 'muted' }, `Image: ${attach.name || 'screenshot'}`) : '')
  const route = h('div', { class: 'route' }, 'Routing...')
  const body = h('div', { class: 'md' })
  const extras = h('div')
  const foot = h('div')
  const reply = h('div', { class: 'msg bot' }, route, body, extras, foot)
  thread.append(bubble, reply)
  thread.scrollTop = thread.scrollHeight
  let answer = ''
  try {
    for await (const event of chatStream(text, history, attach)) {
      if (event.type === 'route') route.textContent = event.label
      if (event.type === 'token') {
        answer += event.text
        body.innerHTML = md(answer).innerHTML
      }
      if (event.type === 'extras') {
        if (event.sources?.length) extras.append(h('h3', { style: 'margin-top:12px' }, 'Sources'), sourceCards(event.sources))
        if (event.table) extras.append(h('div', { style: 'margin-top:10px' }, table(event.table.columns, event.table.rows)))
        if (event.charts?.length) extras.append(h('div', { style: 'margin-top:10px' }, charts(event.charts)))
        if (event.plan) extras.append(h('p', { class: 'muted' }, 'Plan: ' + JSON.stringify(event.plan)))
        if (event.components) extras.append(h('p', { class: 'muted' }, event.components.map((c) => `${c.name}: ${c.backend}, ${c.ms} ms`).join(' | ')))
      }
      if (event.type === 'error') body.append(h('p', { class: 'muted' }, event.message))
      if (event.type === 'done') foot.append(provenance(event.privacy, event.stats, event.total_ms))
      thread.scrollTop = thread.scrollHeight
    }
  } catch (e) {
    body.textContent = e.message
  }
  history.push({ role: 'user', content: text }, { role: 'assistant', content: answer })
}

async function captureScreen() {
  const stream = await navigator.mediaDevices.getDisplayMedia({ video: true })
  const video = h('video')
  video.srcObject = stream
  await video.play()
  await new Promise((r) => setTimeout(r, 400))
  const canvas = h('canvas')
  canvas.width = video.videoWidth
  canvas.height = video.videoHeight
  canvas.getContext('2d').drawImage(video, 0, 0)
  stream.getTracks().forEach((t) => t.stop())
  return new Promise((resolve) => canvas.toBlob((blob) => resolve(new File([blob], 'screenshot.png', { type: 'image/png' })), 'image/png'))
}

export async function render(root) {
  thread = h('div', { class: 'thread' })
  const input = h('textarea', { placeholder: 'Ask about your documents, datasets or an image. Ctrl+V pastes a screenshot.' })
  const chip = h('span', { class: 'muted' })
  const file = h('input', { type: 'file', accept: 'image/*', hidden: true })
  const setImage = (f) => ((image = f), (chip.textContent = f ? `Attached: ${f.name || 'image'}` : ''))
  const submit = () => {
    const text = input.value.trim() || (image ? 'Describe this image' : '')
    if (!text) return
    input.value = ''
    const attach = image
    setImage(null)
    send(text, attach)
  }
  input.addEventListener('keydown', (e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), submit()))
  input.addEventListener('paste', (e) => {
    const item = [...e.clipboardData.items].find((i) => i.type.startsWith('image/'))
    if (item) setImage(item.getAsFile())
  })
  file.addEventListener('change', () => setImage(file.files[0]))
  const suggestions = ['Summarize my document', 'Give me an overview of my dataset', 'Total Sales by Region', 'Which NPU am I using?']
  root.append(
    h('h1', {}, 'AI Assistant'),
    h('p', { class: 'muted' }, 'One chat for documents, datasets and images. The label above each answer shows which feature handled it.'),
    h(
      'div',
      { class: 'chat' },
      thread,
      h('div', { class: 'suggest' }, suggestions.map((s) => h('button', { onclick: () => ((input.value = s), submit()) }, s))),
      h(
        'div',
        { class: 'composer' },
        h('div', { class: 'grow', style: 'flex:1' }, input, chip),
        file,
        h('button', { class: 'btn ghost', title: 'Attach image', onclick: () => file.click() }, icon('Image')),
        h('button', { class: 'btn ghost', title: 'Capture screen', onclick: () => captureScreen().then(setImage).catch(() => toast('Screen capture was cancelled', 'error')) }, icon('Monitor')),
        h('button', { class: 'btn', onclick: submit }, icon('Send'), 'Send'),
      ),
    ),
  )
}
