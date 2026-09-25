# SnapDesk AI

**English:** A local-first AI workspace for Snapdragon-powered HP PCs. It reads documents, analyses CSV/Excel data, answers questions, and looks at screenshots, all on your own PC. No paid API is needed.

**Hindi:** SnapDesk AI ek local-first AI workspace hai jo Snapdragon HP PC ke liye bana hai. Documents padhta hai, CSV/Excel ka analysis karta hai, sawaal ke jawab deta hai aur screenshot samajhta hai. Sab kuch aapke apne PC par chalta hai, koi paid API nahi chahiye.

---

## Honest status / Sachchi jaankari

- **Tested (Linux, CPU):** backend APIs, document search, data analysis, charts, query engine, assistant routing, frontend build.
- **NOT tested on real Snapdragon hardware.** The NPU code path (ONNX Runtime QNN, Genie) is real integration code, but you must verify it on your HP Snapdragon PC. The app never claims NPU use unless ONNX Runtime reports the QNN provider as active.
- Benchmarks show only numbers measured on your PC. If a target (NPU/GPU) cannot start, the row shows the reason instead of a number.
- The language model (Ollama) runs through its own runtime. SnapDesk AI does not claim that it uses the NPU. Only the **Genie** provider runs an LLM on the Hexagon NPU.

Yeh project real Snapdragon PC par test nahi hua hai. NPU wala hissa aapko apne PC par verify karna hoga. App jhooth nahi bolta: NPU tabhi dikhata hai jab ONNX Runtime QNN provider sach mein active ho.

---

## Quick start on Windows (double-click)

1. Install Python 3.12 and Node.js LTS (and Ollama for AI answers).
2. Double-click `setup.bat` (one time, takes a few minutes).
3. Double-click `start.bat`. The browser opens at http://127.0.0.1:8000.

The detailed manual steps are below.

Windows par sabse aasan tareeka: `setup.bat` ek baar chalao, phir `start.bat` chalao.

---

## 1. Install / Installation

Requirements:

- Python 3.11 or 3.12. On a Snapdragon PC install the **ARM64** build of Python from python.org (Windows ARM64 installer).
- Node.js 20+ (ARM64 or x64 both work).
- [Ollama](https://ollama.com) (free, local LLM runner).

## 2. Python setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

Now install **one** ONNX Runtime package (do not install both, they conflict):

```bash
pip install onnxruntime-qnn     # Snapdragon PC (ARM64 Python) - enables NPU
# pip install onnxruntime       # any other PC (CPU only)
```

Optional, for semantic search:

```bash
pip install -r requirements-ai.txt
```

## 3. NPM setup

```bash
cd frontend
npm install
npm run build        # demo mode: backend serves the built UI
```

## 4. Run / Kaise chalayein

**Demo mode (simple):**

```bash
cd backend
python run.py
```

Open http://127.0.0.1:8000

**Development mode (two terminals):**

```bash
cd backend && python run.py
cd frontend && npm run dev      # open http://localhost:5173
```

The server listens on `127.0.0.1` only, so nobody else on your network can reach it.

## 5. Model setup

**Language model (Ollama):**

```bash
ollama pull llama3.2:3b         # text model, about 2 GB
ollama pull moondream           # small vision model for screenshots (optional)
```

SnapDesk AI detects Ollama automatically. Change model names in **Settings**. Without any model the app still works in rule-based mode (keyword search, statistics, charts).

**Embedding model (semantic search, optional):**

```bash
python scripts/download_models.py
```

This downloads all-MiniLM-L6-v2 (ONNX) into `models/embedding/`. For the NPU add `--fix-shapes`, because the Qualcomm HTP backend needs fixed input shapes.

**Optional OCR for screenshots** (do not install plain `onnxruntime` next to `onnxruntime-qnn`):

```bash
pip install rapidocr-onnxruntime --no-deps
pip install pyclipper opencv-python-headless Shapely PyYAML six tqdm
```

If some wheel is not available for ARM64, skip OCR. The vision model still works.

## 6. Qualcomm AI Hub / Snapdragon setup

1. Create a free account at https://aihub.qualcomm.com and copy your API token.
2. `pip install -r backend/requirements-qai-hub.txt`
3. `qai-hub configure --api_token YOUR_TOKEN`
4. List devices: `python scripts/aihub_compile.py --list-devices`
5. Compile an ONNX model for Snapdragon X Elite:

```bash
python scripts/aihub_compile.py --model my_model.onnx --input-name input --shape 1,3,224,224 --output models/vision/classifier.onnx
```

You can also use ready-made models: `python -m qai_hub_models.models.mobilenet_v2.export --help` (options depend on your qai-hub-models version).

**Privacy note:** AI Hub compile and profile jobs run on Qualcomm's cloud. That is a one-time setup step and uploads only the model, never your documents. At runtime everything is local.

For the vision classifier also put a `labels.txt` (one class name per line) next to `classifier.onnx` in `models/vision/`.

## 7. NPU configuration

1. Install the Qualcomm NPU driver (Windows Update or HP support page). Device Manager should show "Snapdragon ... Hexagon NPU".
2. Use ARM64 Python and `pip install onnxruntime-qnn`.
3. Open **Hardware** in the app. The checklist must show: Native ARM64 Python, QNN execution provider, `QnnHtp.dll` found.
4. Set **Settings > Preferred accelerator** to `auto` or `npu`.
5. Put `model.onnx` in `models/embedding/`. The Hardware page shows `running on NPU` only if ONNX Runtime really activated `QNNExecutionProvider`.
6. Run **Benchmark**. It tests CPU, GPU and NPU one by one on the same model.

Note: ONNX Runtime may run unsupported operators on CPU even when the QNN provider is active. Compare benchmark numbers to see the real benefit.

**LLM on the NPU (advanced, Qualcomm Genie):** download a Genie-ready model bundle from Qualcomm AI Hub (for example Llama 3.2 3B for Snapdragon X), install the Qualcomm AI Runtime / Genie SDK, then in **Settings** set the `genie-t2t-run` path, the `genie_config.json` path, and provider `genie`. The Genie provider is experimental: adjust the prompt template in `backend/app/config.py` if your model uses a different chat format.

**Alternative:** run `llama-server` (llama.cpp) with Adreno OpenCL, then use provider `openai` with URL `http://localhost:8080/v1`.

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| "Backend not reachable" at top | Start `python run.py` in `backend/` |
| "No language model" | Start Ollama and run `ollama pull llama3.2:3b` |
| QNN provider missing | Uninstall `onnxruntime`, install `onnxruntime-qnn` in ARM64 Python |
| Python shows `AMD64` on Snapdragon | You installed x64 Python. Install the ARM64 build |
| NPU session fails on embedding model | Run `python scripts/download_models.py --fix-shapes`. The app falls back to CPU automatically and shows the reason on the Hardware page |
| A pip package has no ARM64 wheel | Use Python 3.12, upgrade pip, or skip optional packages (`tokenizers`, OCR) |
| Scanned PDF has no text | OCR for PDFs is not included. Use a text PDF |
| Slow answers | Use a smaller model, lower "Passages per answer" in Settings |
| Answers marked "External" | An endpoint outside this PC is configured. Set it back to localhost |

## 9. Hackathon demo (5 minutes)

1. **Home:** show the detected device, NPU and the setup checklist. Say: "Everything here is detected, not typed in."
2. **Documents:** upload `samples/company_policy.txt`. Show the summary and important information (dates, amounts, email). Ask: "What is the device allowance?" and "How many remote days are allowed?". Point at the source passages and the green "Local" strip.
3. **Data Analysis:** upload `samples/sales_data.csv`. Show missing values, duplicates, outliers, trend charts. Ask: "total Sales by Region" and "monthly Sales in 2024". Explain that pandas calculates numbers, so they are exact.
4. **AI Assistant:** type "Summarize my document", then "average Price by Product", then paste a screenshot. Show the routing label above each answer.
5. **Hardware:** run the benchmark on the embedding model. Show CPU vs NPU latency measured live. If NPU is unavailable, show the reason honestly.
6. **Settings:** show "Block external endpoints" and "Delete all local data" to close on privacy.

Hindi mein: demo ke dauran har jagah dikhao ki data PC se bahar nahi ja raha aur benchmark ke numbers live measure hue hain.

## Project layout

```
backend/    FastAPI app (app/*.py), run.py, requirements
frontend/   Vite + vanilla JS UI
scripts/    download_models.py, aihub_compile.py
samples/    demo document and dataset
models/     put ONNX / GGUF / Genie models here
data/       local uploads and benchmark results (created at runtime)
```
