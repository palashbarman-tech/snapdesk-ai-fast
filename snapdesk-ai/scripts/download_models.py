import subprocess
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main"
FILES = {"onnx/model.onnx": "model.onnx", "tokenizer.json": "tokenizer.json"}
FIX_SHAPES = "--fix-shapes" in sys.argv


def download(remote, target):
    if target.exists():
        print(f"Already exists: {target}")
        return
    print(f"Downloading {remote} ...")
    with requests.get(f"{BASE}/{remote}", stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(target, "wb") as file:
            for chunk in response.iter_content(1 << 20):
                file.write(chunk)


def main():
    folder = ROOT / "models" / "embedding"
    folder.mkdir(parents=True, exist_ok=True)
    for remote, name in FILES.items():
        download(remote, folder / name)
    if FIX_SHAPES:
        fixed = folder / "model_fixed.onnx"
        command = [
            sys.executable, "-m", "onnxruntime.tools.make_dynamic_shape_fixed",
            "--dim_param", "batch_size", "--dim_value", "1",
            "--dim_param", "sequence_length", "--dim_value", "256",
            str(folder / "model.onnx"), str(fixed),
        ]
        if subprocess.run(command).returncode == 0:
            fixed.replace(folder / "model.onnx")
            print("Fixed input shapes (batch 1, length 256) for the NPU.")
        else:
            print("Shape fixing failed. Keep the dynamic model; it still works on CPU. See README troubleshooting.")
    print("Done. Restart the backend.")


if __name__ == "__main__":
    main()
