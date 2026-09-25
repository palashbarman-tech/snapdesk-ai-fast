# SnapDesk AI

A local-first AI workspace for Snapdragon-powered HP PCs. It reads documents, analyses CSV/Excel data, answers questions, and understands screenshots — entirely on-device. No paid API required.

Built for the Snapdragon AI Lab Build & Present Challenge.

## Features

- **Document AI** — Upload PDF, DOCX, TXT. Get summaries, key points, entity extraction (dates, amounts, contacts), cited Q&A, and in-document search.
- **Data Analysis** — Upload CSV/Excel. Automatic statistics, missing values, duplicates, outlier detection, trend analysis, auto-generated charts, and natural-language queries answered with exact pandas calculations.
- **AI Assistant** — One chat interface that routes each question to the right feature (documents, data, or image) automatically.
- **Image/Screenshot Analysis** — OCR, ONNX-based image classification, and local vision-model description.
- **Hardware & Performance** — Detects the real CPU/NPU/ONNX Runtime providers on the PC and runs a live CPU vs GPU vs NPU benchmark.

## Snapdragon optimization

Embedding and vision models run through ONNX Runtime with the Qualcomm QNN execution provider, preferring the Hexagon NPU, then GPU, then CPU. Hardware detection is real — the app never reports NPU usage unless the QNN provider is actually active. The embedding model in this repo was compiled for a real Snapdragon X device using Qualcomm AI Hub.

## Privacy

All document/dataset processing and rule-based analysis run on-device. Every AI answer is labeled Local or External with the provider and timing shown. External endpoints are blocked by default in Settings.

## Setup

Requirements: Python 3.11/3.12, Node.js 20+, Ollama (optional, for AI answers).

### Windows quick start
1. Install Python 3.12 (check "Add python.exe to PATH") and Node.js LTS.
2. Double-click `setup.bat` (one-time, installs dependencies).
3. Double-click `start.bat`. Opens at http://127.0.0.1:8000

### Manual setup