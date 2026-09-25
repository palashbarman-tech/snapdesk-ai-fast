import platform
import time
from dataclasses import dataclass, field

QNN_LIBS = {
    "npu": "QnnHtp.dll" if platform.system() == "Windows" else "libQnnHtp.so",
    "gpu": "QnnGpu.dll" if platform.system() == "Windows" else "libQnnGpu.so",
}


class AcceleratorError(RuntimeError):
    pass


@dataclass
class Accelerated:
    session: object
    requested: str
    actual: str
    providers: list = field(default_factory=list)
    load_ms: float = 0.0
    note: str = ""

    @property
    def label(self):
        return {"npu": "NPU (Hexagon via QNN)", "gpu": "GPU", "cpu": "CPU"}[self.actual]


def _qnn_options(target):
    options = {"backend_path": QNN_LIBS[target]}
    if target == "npu":
        options.update({"htp_performance_mode": "burst", "enable_htp_fp16_precision": "1"})
    return options


def _plans(preference, available, strict):
    order = {"auto": ["npu", "gpu", "cpu"], "npu": ["npu", "cpu"], "gpu": ["gpu", "cpu"], "cpu": ["cpu"]}[preference]
    if strict:
        order = [preference]
    plans = []
    for target in order:
        if target in ("npu", "gpu") and "QNNExecutionProvider" in available and not (
            target == "gpu" and "DmlExecutionProvider" in available
        ):
            plans.append((target, [("QNNExecutionProvider", _qnn_options(target)), "CPUExecutionProvider"]))
        elif target == "gpu" and "DmlExecutionProvider" in available:
            plans.append((target, ["DmlExecutionProvider", "CPUExecutionProvider"]))
        elif target == "cpu":
            plans.append((target, ["CPUExecutionProvider"]))
    return plans


def _actual_target(requested, active):
    if "QNNExecutionProvider" in active:
        return requested if requested in ("npu", "gpu") else "cpu"
    if "DmlExecutionProvider" in active:
        return "gpu"
    return "cpu"


def create_session(model_path, preference="auto", strict=False):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise AcceleratorError("onnxruntime is not installed") from exc

    plans = _plans(preference, ort.get_available_providers(), strict)
    if not plans:
        raise AcceleratorError(f"No execution provider available for '{preference}'")

    errors = []
    for target, providers in plans:
        options = ort.SessionOptions()
        options.log_severity_level = 3
        start = time.perf_counter()
        try:
            session = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
        except Exception as exc:
            errors.append(f"{target}: {exc}")
            continue
        active = session.get_providers()
        actual = _actual_target(target, active)
        if strict and actual != preference:
            errors.append(f"{target}: provider did not activate (active: {', '.join(active)})")
            continue
        note = ""
        if preference != "cpu" and actual == "cpu" and errors:
            note = "Fell back to CPU. " + " | ".join(errors)
        return Accelerated(session, preference, actual, active, (time.perf_counter() - start) * 1000, note)
    raise AcceleratorError(" | ".join(errors) or "Session could not be created")
