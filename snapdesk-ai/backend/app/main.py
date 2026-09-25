import json
import logging
import threading
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import assistant, benchmark, config, data_analysis, documents, embeddings, hardware, llm, metrics, status, vision
from .utils import to_native

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("snapdesk")

app = FastAPI(title="SnapDesk AI", version="1.0.0")


@app.on_event("startup")
def startup():
    documents.load_all()
    data_analysis.load_all()
    threading.Thread(target=hardware.detect, daemon=True).start()
    threading.Thread(target=embeddings.load, daemon=True).start()


class AskBody(BaseModel):
    question: str
    doc_ids: Optional[List[str]] = None


class QuestionBody(BaseModel):
    question: str


class BenchBody(BaseModel):
    kind: str
    model: Optional[str] = None
    iterations: int = 30


def _guard(call):
    try:
        return call()
    except KeyError as exc:
        raise HTTPException(404, str(exc).strip("'\""))
    except (ValueError, llm.LLMError) as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/system")
def system(refresh: bool = False):
    return to_native(status.build(refresh))


@app.get("/api/settings")
def get_settings():
    return config.public()


@app.put("/api/settings")
def put_settings(patch: dict):
    try:
        saved = config.update(patch)
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, f"Invalid setting: {exc}")
    llm.reset_probe()
    if "embedding_model" in patch or "accelerator" in patch:
        embeddings._state["accel"] = None
        threading.Thread(target=embeddings.load, daemon=True).start()
    return saved


@app.delete("/api/data")
def delete_all_data():
    documents.clear()
    data_analysis.clear()
    return {"ok": True}


@app.get("/api/metrics")
def get_metrics():
    return metrics.recent()


@app.post("/api/documents")
def upload_documents(files: List[UploadFile] = File(...)):
    added, errors = [], []
    for file in files:
        try:
            added.append(documents.add(file.filename, file.file.read()))
        except Exception as exc:
            log.warning("Upload failed for %s: %s", file.filename, exc)
            errors.append({"name": file.filename, "error": str(exc)})
    return {"added": added, "errors": errors}


@app.get("/api/documents")
def list_documents():
    return documents.listing()


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    documents.remove(doc_id)
    return {"ok": True}


@app.post("/api/documents/reindex")
def reindex_documents():
    return {"indexed": documents.reindex()}


@app.get("/api/documents/{doc_id}/analysis")
def document_analysis(doc_id: str, refresh: bool = False):
    return _guard(lambda: to_native(documents.analyze(doc_id, refresh)))


@app.post("/api/documents/ask")
def ask_documents(body: AskBody):
    return _guard(lambda: to_native(documents.ask(body.question, body.doc_ids)))


@app.get("/api/documents-search")
def search_documents(q: str, doc_ids: Optional[str] = None):
    ids = doc_ids.split(",") if doc_ids else None
    return documents.search(q, ids, limit=10)


@app.post("/api/datasets")
def upload_datasets(files: List[UploadFile] = File(...)):
    added, errors = [], []
    for file in files:
        try:
            added += data_analysis.add(file.filename, file.file.read())
        except Exception as exc:
            log.warning("Upload failed for %s: %s", file.filename, exc)
            errors.append({"name": file.filename, "error": str(exc)})
    return {"added": added, "errors": errors}


@app.get("/api/datasets")
def list_datasets():
    return data_analysis.listing()


@app.delete("/api/datasets/{dataset_id}")
def delete_dataset(dataset_id: str):
    data_analysis.remove(dataset_id)
    return {"ok": True}


@app.get("/api/datasets/{dataset_id}/analysis")
def dataset_analysis(dataset_id: str):
    return _guard(lambda: data_analysis.analyze(data_analysis.get(dataset_id)))


@app.post("/api/datasets/{dataset_id}/insights")
def dataset_insights(dataset_id: str, refresh: bool = False):
    return _guard(lambda: to_native(data_analysis.ai_insights(data_analysis.get(dataset_id), refresh)))


@app.post("/api/datasets/{dataset_id}/ask")
def ask_dataset(dataset_id: str, body: QuestionBody):
    def run():
        dataset = data_analysis.get(dataset_id)
        plan, result = data_analysis.query(dataset, body.question)
        if plan is None:
            return {"answer": "I could not turn that into a calculation. Try 'average Sales by Region'.", "plan": None}
        answer = data_analysis.describe_result(result)
        return to_native({"answer": answer, "plan": plan, "table": result if "error" not in result else None})

    return _guard(run)


@app.post("/api/vision/analyze")
def analyze_image(image: UploadFile = File(...), question: str = Form("")):
    return _guard(lambda: to_native(vision.analyze(image.file.read(), question)))


@app.post("/api/assistant/chat")
def chat(message: str = Form(...), history: str = Form("[]"), image: Optional[UploadFile] = File(None)):
    content = image.file.read() if image else None
    try:
        turns = json.loads(history)
    except ValueError:
        turns = []

    def events():
        for event in assistant.chat(message, turns, content):
            yield json.dumps(to_native(event), ensure_ascii=False) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.get("/api/benchmarks")
def get_benchmarks():
    return {"models": benchmark.list_models(), "results": benchmark.load_results()}


@app.post("/api/benchmarks/run")
def run_benchmark(body: BenchBody):
    def run():
        if body.kind == "onnx":
            if not body.model:
                raise ValueError("Choose an ONNX model first.")
            return benchmark.bench_onnx(body.model, max(5, min(body.iterations, 200)))
        if body.kind == "llm":
            return benchmark.bench_llm()
        raise ValueError("Unknown benchmark type.")

    try:
        return _guard(run)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))


if config.FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=config.FRONTEND_DIST, html=True), name="frontend")
