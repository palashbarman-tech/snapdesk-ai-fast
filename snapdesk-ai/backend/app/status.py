from . import embeddings, hardware, llm, vision


def _item(key, label, ok, detail, required=False, hint=""):
    return {"key": key, "label": label, "ok": bool(ok), "detail": detail, "required": required, "hint": hint}


def build(refresh=False):
    info = hardware.detect(refresh)
    ort = info["onnxruntime"]
    qnn_ep = "QNNExecutionProvider" in ort["providers"]
    model = llm.status()
    embed = embeddings.status()
    vis = vision.status()
    npu_names = ", ".join(n["name"] for n in info["npus"]) or "none found"

    checklist = [
        _item("snapdragon", "Snapdragon processor", info["snapdragon"], info["cpu"]["name"]),
        _item(
            "arm64",
            "Native ARM64 Python",
            info["python"]["arm64"],
            f"Python {info['python']['version']} ({info['python']['machine']})",
            hint="Install ARM64 Python 3.11 or 3.12 to use the NPU."
            if info["python"]["emulated"]
            else "Only needed on Snapdragon PCs.",
        ),
        _item("npu", "NPU device in Windows", bool(info["npus"]), npu_names, hint="Update the Qualcomm NPU driver from HP or Windows Update."),
        _item(
            "ort",
            "ONNX Runtime",
            ort["installed"],
            f"{ort['package']} {ort['version']}" if ort["installed"] else "not installed",
            hint="pip install onnxruntime-qnn (Snapdragon) or onnxruntime",
        ),
        _item(
            "qnn_ep",
            "QNN execution provider",
            qnn_ep,
            ", ".join(ort["providers"]) or "none",
            hint="Install onnxruntime-qnn in an ARM64 Python.",
        ),
        _item(
            "qnn_lib",
            f"{info['qnn']['library']} runtime library",
            bool(info["qnn"]["found"]),
            info["qnn"]["found"][0] if info["qnn"]["found"] else "not found",
            hint="Bundled with onnxruntime-qnn, or set QNN_SDK_ROOT.",
        ),
        _item("llm", "Local language model", model["ready"], f"{model['provider']} {model['model']}".strip() or model["detail"], required=True, hint=model["detail"]),
        _item(
            "embedding",
            "Embedding model (semantic search)",
            embed["ready"],
            f"{embed['accelerator'].upper()}" if embed["ready"] else (embed["error"] or "not loaded"),
            hint="python scripts/download_models.py",
        ),
        _item(
            "vision",
            "Vision model",
            vis["vlm_ready"],
            vis["vlm_model"],
            hint=f"ollama pull {vis['vlm_model']}",
        ),
        _item("genie", "Qualcomm Genie (NPU LLM)", bool(info["genie"]["binary"] and info["genie"]["config"]), info["genie"]["binary"] or "not configured", hint="Optional. See README."),
        _item(
            "ai_hub",
            "Qualcomm AI Hub account",
            info["ai_hub"]["configured"],
            "client.ini found" if info["ai_hub"]["configured"] else "not configured",
            hint="Optional, used only to compile models.",
        ),
    ]
    return {
        "hardware": info,
        "llm": model,
        "embedding": embed,
        "vision": vis,
        "privacy": llm.privacy(),
        "checklist": checklist,
    }
