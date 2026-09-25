import importlib.metadata as metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import threading
from pathlib import Path

import psutil

from . import config

ARM_MACHINES = {"arm64", "aarch64"}
QNN_FAMILY = ("snapdragon", "qualcomm", "oryon", "hexagon")
ORT_PACKAGES = ("onnxruntime-qnn", "onnxruntime-directml", "onnxruntime-gpu", "onnxruntime")

_cache = None
_lock = threading.Lock()


WINDOWS_SCRIPT = (
    "$r=@{cpu=@(Get-CimInstance Win32_Processor|Select-Object Name,Architecture);"
    "sys=@(Get-CimInstance Win32_ComputerSystem|Select-Object Manufacturer,Model);"
    "gpu=@(Get-CimInstance Win32_VideoController|Select-Object Name,DriverVersion);"
    "npu=@(Get-CimInstance Win32_PnPEntity|Where-Object{$_.Name -match 'NPU|Hexagon|Neural'}|Select-Object Name,Status)};"
    "$r"
)
_windows = {}


def _windows_info():
    if not _windows and platform.system() == "Windows":
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", f"{WINDOWS_SCRIPT} | ConvertTo-Json -Depth 4 -Compress"],
                capture_output=True, text=True, timeout=40,
            )
            _windows.update(json.loads(result.stdout))
        except (OSError, subprocess.SubprocessError, ValueError):
            _windows["failed"] = True
    return _windows


def _rows(key):
    value = _windows_info().get(key) or []
    return value if isinstance(value, list) else [value]


def _cpu():
    system = platform.system()
    name = platform.processor() or platform.machine()
    architecture = None
    if system == "Windows":
        rows = _rows("cpu")
        if rows:
            name = str(rows[0].get("Name") or name).strip()
            architecture = rows[0].get("Architecture")
    elif system == "Darwin":
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        name = out.stdout.strip() or name
    elif system == "Linux" and Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text(errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                name = line.split(":", 1)[1].strip()
                break
    return {
        "name": name,
        "cores": psutil.cpu_count(logical=False),
        "threads": psutil.cpu_count(logical=True),
        "device_arm64": architecture == 12 or platform.machine().lower() in ARM_MACHINES,
    }


def _device():
    rows = _rows("sys")
    if rows:
        return f"{rows[0].get('Manufacturer', '')} {rows[0].get('Model', '')}".strip()
    return platform.node()


def _gpus():
    rows = _rows("gpu")
    return [{"name": r.get("Name"), "driver": r.get("DriverVersion")} for r in rows]


def _npus():
    rows = _rows("npu")
    return [{"name": r.get("Name"), "status": r.get("Status")} for r in rows]


def _package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _onnxruntime():
    if importlib.util.find_spec("onnxruntime") is None:
        return {"installed": False, "version": None, "package": None, "providers": []}
    import onnxruntime as ort

    package = next((name for name in ORT_PACKAGES if _package_version(name)), "onnxruntime")
    return {
        "installed": True,
        "version": ort.__version__,
        "package": package,
        "providers": ort.get_available_providers(),
    }


def _qnn_runtime():
    library = "QnnHtp.dll" if platform.system() == "Windows" else "libQnnHtp.so"
    found = []
    spec = importlib.util.find_spec("onnxruntime")
    if spec and spec.origin:
        found += list(Path(spec.origin).parent.rglob(library))
    sdk_root = os.environ.get("QNN_SDK_ROOT", "")
    if sdk_root and (Path(sdk_root) / "lib").exists():
        found += list((Path(sdk_root) / "lib").rglob(library))[:3]
    return {"library": library, "found": [str(p) for p in found[:3]], "sdk_root": sdk_root}


def _genie():
    settings = config.get()
    binary = settings["genie_bin"] or shutil.which("genie-t2t-run") or ""
    model_config = settings["genie_config"]
    return {
        "binary": binary if binary and Path(binary).exists() else "",
        "config": model_config if model_config and Path(model_config).exists() else "",
    }


def _ai_hub():
    return {
        "client_installed": importlib.util.find_spec("qai_hub") is not None,
        "configured": (Path.home() / ".qai_hub" / "client.ini").exists(),
    }


def _collect():
    cpu = _cpu()
    npus = _npus()
    identity = " ".join([cpu["name"], platform.processor()] + [n["name"] or "" for n in npus]).lower()
    machine = platform.machine()
    return {
        "device": _device(),
        "os": f"{platform.system()} {platform.release()}",
        "cpu": cpu,
        "memory_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "gpus": _gpus(),
        "npus": npus,
        "snapdragon": any(key in identity for key in QNN_FAMILY),
        "python": {
            "version": platform.python_version(),
            "machine": machine,
            "arm64": machine.lower() in ARM_MACHINES,
            "emulated": cpu["device_arm64"] and machine.lower() not in ARM_MACHINES,
        },
        "onnxruntime": _onnxruntime(),
        "qnn": _qnn_runtime(),
        "genie": _genie(),
        "ai_hub": _ai_hub(),
    }


def detect(refresh=False):
    global _cache
    with _lock:
        if _cache is None or refresh:
            if refresh:
                _windows.clear()
            _cache = _collect()
        return _cache


def summary_line():
    info = detect()
    npu = info["npus"][0]["name"] if info["npus"] else "no NPU detected"
    return f"{info['device']} | {info['cpu']['name']} | {npu}"
