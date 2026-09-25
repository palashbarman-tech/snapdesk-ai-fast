import json
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
FRONTEND_DIST = ROOT / "frontend" / "dist"
SETTINGS_FILE = DATA_DIR / "settings.json"

LLAMA3_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

DEFAULTS = {
    "llm_provider": "auto",
    "ollama_url": "http://127.0.0.1:11434",
    "ollama_model": "llama3.2:3b",
    "ollama_vision_model": "moondream",
    "openai_url": "http://127.0.0.1:8080/v1",
    "openai_model": "local-model",
    "openai_api_key": "",
    "genie_bin": "",
    "genie_config": "",
    "genie_prompt_template": LLAMA3_TEMPLATE,
    "accelerator": "auto",
    "embedding_model": "models/embedding/model.onnx",
    "vision_classifier": "models/vision/classifier.onnx",
    "block_external": True,
    "chunk_size": 700,
    "top_k": 5,
    "temperature": 0.2,
    "max_tokens": 512,
}

SECRET_KEYS = {"openai_api_key"}
MASK = "********"

_lock = threading.Lock()
_cache = None


def _coerce(key, value):
    default = DEFAULTS[key]
    if isinstance(default, bool):
        return value if isinstance(value, bool) else str(value).lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return str(value)


def get():
    global _cache
    with _lock:
        if _cache is None:
            saved = {}
            if SETTINGS_FILE.exists():
                try:
                    saved = json.loads(SETTINGS_FILE.read_text("utf-8"))
                except ValueError:
                    saved = {}
            _cache = {**DEFAULTS}
            for key, value in saved.items():
                if key in DEFAULTS:
                    _cache[key] = _coerce(key, value)
            for key in ("ollama_url", "openai_url"):
                _cache[key] = _cache[key].replace("//localhost", "//127.0.0.1")
        return dict(_cache)


def update(patch):
    global _cache
    current = get()
    for key, value in patch.items():
        if key not in DEFAULTS or (key in SECRET_KEYS and value == MASK):
            continue
        current[key] = _coerce(key, value)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(current, indent=2), "utf-8")
    with _lock:
        _cache = current
    return public()


def public():
    values = get()
    for key in SECRET_KEYS:
        if values.get(key):
            values[key] = MASK
    return values


def resolve(path_text):
    path = Path(path_text).expanduser()
    return path if path.is_absolute() else ROOT / path
