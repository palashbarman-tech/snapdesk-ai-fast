import codecs
import json
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from . import config, metrics

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
PROBE_SECONDS = 5

_probe = {"time": 0.0, "provider": "none"}
_models_cache = {"time": 0.0, "url": "", "models": None}
_status_cache = {"time": 0.0, "value": None}
CACHE_SECONDS = 5


class LLMError(RuntimeError):
    pass


def is_local_url(url):
    return (urlparse(url).hostname or "") in LOCAL_HOSTS


def ollama_models(url=None):
    url = (url or config.get()["ollama_url"]).rstrip("/")
    now = time.time()
    if _models_cache["url"] == url and now - _models_cache["time"] < CACHE_SECONDS:
        return _models_cache["models"]
    try:
        response = requests.get(f"{url}/api/tags", timeout=1.0)
        response.raise_for_status()
        models = [m["name"] for m in response.json().get("models", [])]
    except (requests.RequestException, ValueError):
        models = None
    _models_cache.update(time=now, url=url, models=models)
    return models


def has_model(models, wanted):
    return any(name == wanted or name == f"{wanted}:latest" for name in models or [])


def _openai_reachable(settings):
    try:
        headers = {"Authorization": f"Bearer {settings['openai_api_key']}"} if settings["openai_api_key"] else {}
        return requests.get(f"{settings['openai_url'].rstrip('/')}/models", headers=headers, timeout=1.5).ok
    except requests.RequestException:
        return False


def _genie_ready(settings):
    binary = settings["genie_bin"] or shutil.which("genie-t2t-run")
    return bool(binary and Path(binary).exists() and settings["genie_config"] and Path(settings["genie_config"]).exists())


def _resolve(settings):
    provider = settings["llm_provider"]
    if provider != "auto":
        return provider
    now = time.time()
    if now - _probe["time"] < PROBE_SECONDS:
        return _probe["provider"]
    if _genie_ready(settings):
        found = "genie"
    elif ollama_models(settings["ollama_url"]) is not None:
        found = "ollama"
    elif is_local_url(settings["openai_url"]) and _openai_reachable(settings):
        found = "openai"
    else:
        found = "none"
    _probe.update(time=now, provider=found)
    return found


def _endpoint(settings, provider):
    return {"ollama": settings["ollama_url"], "openai": settings["openai_url"]}.get(provider, "")


def status():
    now = time.time()
    if _status_cache["value"] is not None and now - _status_cache["time"] < CACHE_SECONDS:
        return dict(_status_cache["value"])
    value = _status()
    _status_cache.update(time=now, value=value)
    return dict(value)


def _status():
    settings = config.get()
    provider = _resolve(settings)
    endpoint = _endpoint(settings, provider)
    info = {
        "provider": provider,
        "model": "",
        "endpoint": endpoint,
        "local": is_local_url(endpoint) if endpoint else True,
        "ready": False,
        "detail": "",
    }
    if provider == "none":
        info["detail"] = "No local language model found. Documents and data still work in rule-based mode."
        return info
    if endpoint and not info["local"] and settings["block_external"]:
        info["detail"] = "This endpoint is not on this PC. External calls are blocked in Settings > Privacy."
        return info
    if provider == "ollama":
        info["model"] = settings["ollama_model"]
        models = ollama_models(settings["ollama_url"])
        if models is None:
            info["detail"] = "Ollama is not running. Start Ollama or change the provider in Settings."
        elif not has_model(models, settings["ollama_model"]):
            info["detail"] = f"Model not pulled yet. Run: ollama pull {settings['ollama_model']}"
        else:
            info["ready"] = True
    elif provider == "openai":
        info["model"] = settings["openai_model"]
        info["ready"] = _openai_reachable(settings)
        info["detail"] = "" if info["ready"] else "The OpenAI-compatible server did not respond."
    elif provider == "genie":
        info["model"] = Path(settings["genie_config"]).parent.name if settings["genie_config"] else ""
        info["ready"] = _genie_ready(settings)
        info["detail"] = "" if info["ready"] else "Set the Genie binary and config paths in Settings."
    return info


def privacy():
    info = status()
    if not info["ready"]:
        return {"local": True, "external": False, "label": "Local | rule-based, no model"}
    place = "Local" if info["local"] else "External service"
    return {"local": info["local"], "external": not info["local"], "label": f"{place} | {info['provider']} {info['model']}"}


def _error_text(response):
    try:
        return response.json().get("error", response.text)
    except ValueError:
        return response.text[:300]


def _stream_ollama(settings, messages, opts, stats):
    body = {
        "model": settings["ollama_model"],
        "messages": messages,
        "stream": True,
        "keep_alive": "30m",
        "options": {"temperature": opts["temperature"], "num_predict": opts["max_tokens"]},
    }
    if opts["json_mode"]:
        body["format"] = "json"
    url = settings["ollama_url"].rstrip("/")
    with requests.post(f"{url}/api/chat", json=body, stream=True, timeout=(5, 300)) as response:
        if response.status_code != 200:
            raise LLMError(str(_error_text(response)))
        for line in response.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            if data.get("error"):
                raise LLMError(data["error"])
            piece = data.get("message", {}).get("content", "")
            if piece:
                yield piece
            if data.get("done"):
                count, nanos = data.get("eval_count"), data.get("eval_duration")
                if count and nanos:
                    stats.update(tokens=count, tokens_per_sec=round(count / (nanos / 1e9), 1))
                stats["prompt_tokens"] = data.get("prompt_eval_count")


def _stream_openai(settings, messages, opts, stats):
    url = settings["openai_url"].rstrip("/")
    headers = {"Authorization": f"Bearer {settings['openai_api_key']}"} if settings["openai_api_key"] else {}
    body = {
        "model": settings["openai_model"],
        "messages": messages,
        "stream": True,
        "temperature": opts["temperature"],
        "max_tokens": opts["max_tokens"],
    }
    with requests.post(f"{url}/chat/completions", json=body, headers=headers, stream=True, timeout=(5, 300)) as response:
        if response.status_code != 200:
            raise LLMError(str(_error_text(response)))
        response.encoding = "utf-8"
        for raw in response.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            payload = raw[5:].strip()
            if payload == "[DONE]":
                break
            choices = json.loads(payload).get("choices") or []
            piece = choices[0].get("delta", {}).get("content") if choices else None
            if piece:
                yield piece


def _stream_genie(settings, messages, opts, stats):
    binary = settings["genie_bin"] or shutil.which("genie-t2t-run")
    if not _genie_ready(settings):
        raise LLMError("Genie binary or config path is missing. Check Settings.")
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    user = "\n\n".join(m["content"] for m in messages if m["role"] != "system")
    prompt = settings["genie_prompt_template"].replace("{system}", system).replace("{user}", user)
    process = subprocess.Popen(
        [binary, "-c", settings["genie_config"], "-p", prompt], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    decoder = codecs.getincrementaldecoder("utf-8")("ignore")
    buffer, started, keep = "", False, len("[END]") - 1
    try:
        while True:
            chunk = process.stdout.read1(256)
            if not chunk:
                break
            buffer += decoder.decode(chunk)
            if not started:
                if "[BEGIN]:" not in buffer:
                    continue
                buffer, started = buffer.split("[BEGIN]:", 1)[1], True
            if "[END]" in buffer:
                text = buffer.split("[END]", 1)[0]
                if text:
                    yield text
                return
            if len(buffer) > keep:
                yield buffer[:-keep]
                buffer = buffer[-keep:]
    finally:
        if process.poll() is None:
            process.kill()


_STREAMS = {"ollama": _stream_ollama, "openai": _stream_openai, "genie": _stream_genie}


def stream(messages, stats=None, json_mode=False, max_tokens=None, temperature=None):
    settings = config.get()
    stats = stats if stats is not None else {}
    provider = _resolve(settings)
    if provider == "none":
        raise LLMError("No local language model is available. See Settings or the README.")
    endpoint = _endpoint(settings, provider)
    if endpoint and not is_local_url(endpoint) and settings["block_external"]:
        raise LLMError("This endpoint is not on this PC and external calls are blocked in Settings.")
    opts = {
        "json_mode": json_mode,
        "max_tokens": max_tokens or settings["max_tokens"],
        "temperature": settings["temperature"] if temperature is None else temperature,
    }
    model = {"ollama": settings["ollama_model"], "openai": settings["openai_model"]}.get(provider, "genie")
    start = time.perf_counter()
    first, chunks = None, 0
    try:
        for piece in _STREAMS[provider](settings, messages, opts, stats):
            if first is None:
                first = time.perf_counter()
            chunks += 1
            yield piece
    except requests.RequestException as exc:
        raise LLMError(f"Could not reach the {provider} service: {exc}") from exc
    end = time.perf_counter()
    stats.update(
        provider=provider,
        model=model,
        chunks=chunks,
        ttft_ms=round(((first or end) - start) * 1000),
        total_ms=round((end - start) * 1000),
    )
    metrics.record("LLM generation", (end - start) * 1000, f"{provider} {model}", f"first token {stats['ttft_ms']} ms")


def complete(messages, **options):
    stats = {}
    text = "".join(stream(messages, stats=stats, **options))
    return text.strip(), stats


def reset_probe():
    _probe["time"] = 0.0
    _models_cache["time"] = 0.0
    _status_cache["value"] = None
