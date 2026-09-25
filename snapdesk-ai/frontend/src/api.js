async function parse(response) {
  if (!response.ok) {
    let message = response.statusText
    try {
      message = (await response.json()).detail || message
    } catch {}
    throw new Error(message)
  }
  return response.json()
}

const cache = new Map()

export const api = {
  get: (url) => fetch('/api' + url).then(parse),
  cached: (url, seconds = 8) => {
    const hit = cache.get(url)
    if (hit && Date.now() - hit.time < seconds * 1000) return hit.promise
    const promise = fetch('/api' + url).then(parse)
    cache.set(url, { time: Date.now(), promise })
    promise.catch(() => cache.delete(url))
    return promise
  },
  forget: () => cache.clear(),
  put: (url, body) => {
    cache.clear()
    return fetch('/api' + url, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }).then(parse)
  },
  post: (url, body = {}) =>
    fetch('/api' + url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }).then(parse),
  del: (url) => fetch('/api' + url, { method: 'DELETE' }).then(parse),
  upload: (url, files) => {
    const form = new FormData()
    files.forEach((file) => form.append('files', file))
    return fetch('/api' + url, { method: 'POST', body: form }).then(parse)
  },
}

export async function* chatStream(message, history, image) {
  const form = new FormData()
  form.append('message', message)
  form.append('history', JSON.stringify(history))
  if (image) form.append('image', image)
  const response = await fetch('/api/assistant/chat', { method: 'POST', body: form })
  if (!response.ok) throw new Error(response.statusText)
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop()
    for (const line of lines) if (line.trim()) yield JSON.parse(line)
  }
}
