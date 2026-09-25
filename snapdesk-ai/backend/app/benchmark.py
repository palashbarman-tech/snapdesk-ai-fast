import json
import statistics
import time

import numpy as np

from . import accelerators, config, hardware, llm

RESULTS_FILE = config.DATA_DIR / "benchmarks.json"
BENCH_PROMPT = "Explain in about 80 words why running AI models on a laptop NPU can save battery compared with the CPU."
TARGETS = ("cpu", "gpu", "npu")


def list_models():
    if not config.MODELS_DIR.exists():
        return []
    return sorted(str(p.relative_to(config.ROOT)).replace("\\", "/") for p in config.MODELS_DIR.rglob("*.onnx"))


def load_results():
    if not RESULTS_FILE.exists():
        return []
    try:
        return json.loads(RESULTS_FILE.read_text("utf-8"))
    except ValueError:
        return []


def _save(entry):
    entry["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    entry["device"] = hardware.summary_line()
    results = [entry] + load_results()
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(results[:30], indent=2), "utf-8")
    return entry


def _dummy_inputs(session):
    feed = {}
    rng = np.random.default_rng(0)
    for item in session.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else (1 if i == 0 else 128) for i, d in enumerate(item.shape)]
        if "int" in item.type:
            dtype = np.int64 if "64" in item.type else np.int32
            feed[item.name] = np.ones(shape, dtype=dtype)
        else:
            dtype = np.float16 if "float16" in item.type else np.float32
            feed[item.name] = rng.standard_normal(shape).astype(dtype)
    return feed


def bench_onnx(model, iterations=30, warmup=5):
    path = config.resolve(model)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {model}")
    rows = []
    for target in TARGETS:
        row = {"target": target}
        try:
            accel = accelerators.create_session(path, target, strict=True)
            feed = _dummy_inputs(accel.session)
            for _ in range(warmup):
                accel.session.run(None, feed)
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                accel.session.run(None, feed)
                times.append((time.perf_counter() - start) * 1000)
            times.sort()
            row.update(
                status="ok",
                providers=accel.providers,
                load_ms=round(accel.load_ms, 1),
                mean_ms=round(statistics.mean(times), 2),
                median_ms=round(statistics.median(times), 2),
                p95_ms=round(times[int(len(times) * 0.95) - 1], 2),
                min_ms=round(times[0], 2),
            )
        except accelerators.AcceleratorError as exc:
            row.update(status="unavailable", reason=str(exc))
        except Exception as exc:
            row.update(status="failed", reason=str(exc))
        rows.append(row)
    return _save({"kind": "onnx", "model": model, "iterations": iterations, "warmup": warmup, "rows": rows})


def bench_llm(runs=2):
    info = llm.status()
    if not info["ready"]:
        raise llm.LLMError(info["detail"] or "No language model is ready.")
    llm.complete([{"role": "user", "content": "Say hi."}], max_tokens=4)
    measurements = []
    for _ in range(runs):
        _, stats = llm.complete([{"role": "user", "content": BENCH_PROMPT}], max_tokens=120, temperature=0)
        measurements.append(stats)
    entry = {
        "kind": "llm",
        "provider": info["provider"],
        "model": info["model"],
        "runs": runs,
        "first_token_ms": round(statistics.mean(m["ttft_ms"] for m in measurements)),
        "total_ms": round(statistics.mean(m["total_ms"] for m in measurements)),
        "tokens_per_sec": next(
            (round(statistics.mean(m["tokens_per_sec"] for m in measurements), 1) for m in measurements if "tokens_per_sec" in m), None
        ),
        "note": "Tokens per second is reported by the provider when available. The provider decides which chip runs the model.",
    }
    return _save(entry)
