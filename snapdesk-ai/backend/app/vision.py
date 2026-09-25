import base64
import io
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from . import accelerators, config, llm, metrics

_ocr = {"engine": None, "tried": False, "error": ""}
_classifier = {"accel": None, "labels": [], "error": ""}

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _ocr_engine():
    if not _ocr["tried"]:
        _ocr["tried"] = True
        try:
            from rapidocr_onnxruntime import RapidOCR

            _ocr["engine"] = RapidOCR()
        except Exception as exc:
            _ocr["error"] = f"OCR not installed ({exc.__class__.__name__}). See README, optional OCR."
    return _ocr["engine"]


def _classifier_session():
    if _classifier["accel"] is None and not _classifier["error"]:
        settings = config.get()
        model = config.resolve(settings["vision_classifier"])
        labels_file = model.parent / "labels.txt"
        try:
            if not model.exists() or not labels_file.exists():
                raise FileNotFoundError("classifier.onnx and labels.txt not found in models/vision")
            _classifier["accel"] = accelerators.create_session(model, settings["accelerator"])
            _classifier["labels"] = labels_file.read_text("utf-8").splitlines()
        except Exception as exc:
            _classifier["error"] = str(exc)
    return _classifier["accel"]


def status():
    settings = config.get()
    models = llm.ollama_models(settings["ollama_url"])
    vlm_ready = models is not None and llm.has_model(models, settings["ollama_vision_model"])
    model_file = config.resolve(settings["vision_classifier"])
    accel = _classifier["accel"]
    return {
        "image_info": True,
        "ocr_installed": _ocr["engine"] is not None or _has_ocr_package(),
        "classifier_file": model_file.exists(),
        "classifier_accelerator": accel.actual if accel else None,
        "classifier_error": _classifier["error"],
        "vlm_model": settings["ollama_vision_model"],
        "vlm_ready": vlm_ready,
    }


def _has_ocr_package():
    import importlib.util

    return importlib.util.find_spec("rapidocr_onnxruntime") is not None


def _colors(image, count=5):
    small = image.convert("RGB").resize((96, 96))
    palette = small.quantize(colors=count, method=Image.Quantize.MEDIANCUT)
    colors = palette.getcolors() or []
    raw = palette.getpalette() or []
    total = sum(c for c, _ in colors) or 1
    return [
        {"hex": "#{:02x}{:02x}{:02x}".format(*raw[i * 3 : i * 3 + 3]), "share": round(c / total * 100)}
        for c, i in sorted(colors, reverse=True)
    ]


def _classify(image):
    accel = _classifier_session()
    if accel is None:
        return [], None
    session = accel.session
    input_info = session.get_inputs()[0]
    shape = input_info.shape
    height = shape[2] if len(shape) == 4 and isinstance(shape[2], int) else 224
    width = shape[3] if len(shape) == 4 and isinstance(shape[3], int) else 224
    pixels = np.asarray(image.convert("RGB").resize((width, height)), dtype=np.float32) / 255.0
    tensor = ((pixels - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    output = session.run(None, {input_info.name: tensor})[0][0]
    if not (0.99 <= float(output.sum()) <= 1.01 and output.min() >= 0):
        exp = np.exp(output - output.max())
        output = exp / exp.sum()
    labels = _classifier["labels"]
    top = np.argsort(-output)[:5]
    return [
        {"label": labels[i] if i < len(labels) else f"class {i}", "confidence": round(float(output[i]) * 100, 1)} for i in top
    ], accel.label


def _vlm(image, question):
    settings = config.get()
    url = settings["ollama_url"].rstrip("/")
    if not llm.is_local_url(url) and settings["block_external"]:
        raise llm.LLMError("Vision endpoint is not on this PC and external calls are blocked.")
    models = llm.ollama_models(url)
    if models is None or not llm.has_model(models, settings["ollama_vision_model"]):
        return None
    small = image.convert("RGB")
    small.thumbnail((1024, 1024))
    buffer = io.BytesIO()
    small.save(buffer, format="JPEG", quality=88)
    prompt = question or "Describe this image or screenshot. Mention visible text, apps, charts, tables and anything notable."
    body = {
        "model": settings["ollama_vision_model"],
        "stream": False,
        "keep_alive": "30m",
        "messages": [{"role": "user", "content": prompt, "images": [base64.b64encode(buffer.getvalue()).decode()]}],
    }
    try:
        response = requests.post(f"{url}/api/chat", json=body, timeout=(5, 300))
        response.raise_for_status()
    except requests.RequestException as exc:
        raise llm.LLMError(f"Vision model request failed: {exc}") from exc
    return response.json().get("message", {}).get("content", "").strip()


def analyze(content, question=""):
    components = []
    image = Image.open(io.BytesIO(content))
    image.load()

    start = time.perf_counter()
    grey = np.asarray(image.convert("L"), dtype=np.float32)
    info = {
        "width": image.width,
        "height": image.height,
        "format": image.format or "unknown",
        "brightness": round(float(grey.mean()) / 255 * 100),
        "colors": _colors(image),
    }
    components.append({"name": "Image details", "backend": "CPU (Pillow)", "ms": round((time.perf_counter() - start) * 1000, 1)})

    result = {"info": info, "ocr_text": "", "labels": [], "description": "", "notes": [], "components": components}

    engine = _ocr_engine()
    if engine is not None:
        start = time.perf_counter()
        found, _ = engine(np.asarray(image.convert("RGB")))
        result["ocr_text"] = "\n".join(line[1] for line in (found or []))
        ms = (time.perf_counter() - start) * 1000
        components.append({"name": "Text recognition (OCR)", "backend": "ONNX Runtime CPU", "ms": round(ms, 1)})
        metrics.record("OCR", ms, "ONNX Runtime CPU", f"{len(result['ocr_text'])} chars")
    else:
        result["notes"].append(_ocr["error"])

    start = time.perf_counter()
    labels, backend = _classify(image)
    if backend:
        ms = (time.perf_counter() - start) * 1000
        result["labels"] = labels
        components.append({"name": "Image classifier", "backend": backend, "ms": round(ms, 1)})
        metrics.record("Image classification", ms, backend)
    elif _classifier["error"]:
        result["notes"].append(f"Classifier unavailable: {_classifier['error']}")

    start = time.perf_counter()
    try:
        description = _vlm(image, question)
    except llm.LLMError as exc:
        description = None
        result["notes"].append(str(exc))
    if description:
        settings = config.get()
        ms = (time.perf_counter() - start) * 1000
        result["description"] = description
        components.append({"name": "Vision language model", "backend": f"ollama {settings['ollama_vision_model']}", "ms": round(ms, 1)})
        metrics.record("Vision model", ms, f"ollama {settings['ollama_vision_model']}")
    elif result["ocr_text"] and llm.status()["ready"]:
        prompt = (
            f"This text was extracted from a screenshot by OCR:\n\n{result['ocr_text'][:3000]}\n\n"
            f"{question or 'Explain what this screenshot shows and list anything important.'}"
        )
        try:
            result["description"], stats = llm.complete([{"role": "user", "content": prompt}], temperature=0.2)
            components.append({"name": "Language model on OCR text", "backend": f"{stats['provider']} {stats['model']}", "ms": stats["total_ms"]})
        except llm.LLMError as exc:
            result["notes"].append(str(exc))
    elif description is None and not any("Vision model" in n for n in result["notes"]):
        result["notes"].append(
            f"No vision model found. Run: ollama pull {config.get()['ollama_vision_model']} to enable image descriptions."
        )
    return result


def to_markdown(result):
    info = result["info"]
    parts = []
    if result["description"]:
        parts.append(result["description"])
    if result["ocr_text"]:
        parts.append("**Text found in the image**\n\n> " + result["ocr_text"][:1200].replace("\n", "\n> "))
    if result["labels"]:
        parts.append("**Likely content:** " + ", ".join(f"{l['label']} ({l['confidence']}%)" for l in result["labels"][:3]))
    palette = ", ".join(f"{c['hex']} ({c['share']}%)" for c in info["colors"][:3])
    parts.append(
        f"**Image details:** {info['width']} x {info['height']} {info['format']}, brightness {info['brightness']}%, main colors {palette}."
    )
    for note in result["notes"]:
        parts.append(f"_{note}_")
    return "\n\n".join(parts)
