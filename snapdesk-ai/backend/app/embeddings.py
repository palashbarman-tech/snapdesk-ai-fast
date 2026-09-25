import threading
from pathlib import Path

import numpy as np

from . import accelerators, config

_state = {"accel": None, "tokenizer": None, "error": "", "loading": False, "seq_len": 256}
_lock = threading.Lock()


def model_path():
    return config.resolve(config.get()["embedding_model"])


def load():
    with _lock:
        if _state["accel"] is not None:
            return
        _state.update(loading=True, error="")
        try:
            path = model_path()
            tokenizer_file = path.parent / "tokenizer.json"
            if not path.exists():
                raise FileNotFoundError(f"Embedding model not found: {path}")
            if not tokenizer_file.exists():
                raise FileNotFoundError(f"tokenizer.json not found next to the model: {tokenizer_file}")
            from tokenizers import Tokenizer

            accel = accelerators.create_session(path, config.get()["accelerator"])
            shape = accel.session.get_inputs()[0].shape
            seq_len = shape[1] if len(shape) > 1 and isinstance(shape[1], int) else 256
            tokenizer = Tokenizer.from_file(str(tokenizer_file))
            tokenizer.enable_truncation(max_length=seq_len)
            tokenizer.enable_padding(length=seq_len)
            _state.update(accel=accel, tokenizer=tokenizer, seq_len=seq_len)
        except ImportError as exc:
            _state["error"] = f"Missing package: {exc.name}. Run: pip install -r requirements-ai.txt"
        except Exception as exc:
            _state["error"] = str(exc)
        finally:
            _state["loading"] = False


def ready():
    return _state["accel"] is not None


def backend_label():
    accel = _state["accel"]
    return accel.label if accel else "not loaded"


def status():
    accel = _state["accel"]
    return {
        "model_file": model_path().exists(),
        "loading": _state["loading"],
        "ready": accel is not None,
        "accelerator": accel.actual if accel else None,
        "providers": accel.providers if accel else [],
        "load_ms": round(accel.load_ms) if accel else None,
        "note": accel.note if accel else "",
        "error": _state["error"],
    }


def embed(texts):
    accel, tokenizer = _state["accel"], _state["tokenizer"]
    if accel is None:
        raise RuntimeError("Embedding model is not loaded")
    names = {i.name for i in accel.session.get_inputs()}
    vectors = []
    for text in texts:
        encoded = tokenizer.encode(text)
        feed = {
            "input_ids": np.array([encoded.ids], dtype=np.int64),
            "attention_mask": np.array([encoded.attention_mask], dtype=np.int64),
            "token_type_ids": np.array([encoded.type_ids], dtype=np.int64),
        }
        output = accel.session.run(None, {k: v for k, v in feed.items() if k in names})[0]
        if output.ndim == 3:
            mask = feed["attention_mask"][..., None].astype(np.float32)
            output = (output * mask).sum(axis=1) / max(mask.sum(), 1.0)
        vector = output[0]
        vectors.append(vector / (np.linalg.norm(vector) or 1.0))
    return np.array(vectors, dtype=np.float32)
