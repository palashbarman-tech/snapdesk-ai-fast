import io
import json
import re
import threading
import time
import uuid
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np

from . import config, embeddings, llm, metrics
from .retrieval import BM25, split_sentences, tokenize

DOCS_DIR = config.DATA_DIR / "documents"
SUPPORTED = {".pdf", ".txt", ".md", ".docx"}
WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

_docs = {}
_vectors = {}
_version = 0
_index = {"key": None}
_lock = threading.RLock()

SUMMARY_SYSTEM = (
    "You are SnapDesk AI, a private assistant running on the user's PC. Use only the document text provided. "
    "Write in the same language as the document."
)
QA_SYSTEM = (
    "You are SnapDesk AI, a private assistant running on the user's PC. Answer using only the numbered context "
    "passages. Cite passages like [1] or [2]. If the answer is not in the context, say you could not find it. "
    "Reply in the language of the question. Be concise."
)

ENTITY_PATTERNS = {
    "Dates": [
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.? \d{1,2}(?:st|nd|rd|th)?,? \d{4}\b",
        r"\b\d{1,2}(?:st|nd|rd|th)? (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*,? \d{4}\b",
    ],
    "Amounts": [
        r"(?:₹|Rs\.?|INR|\$|USD|€|EUR|£)\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:lakh|crore|million|billion|thousand|k|M|B)\b)?"
    ],
    "Percentages": [r"\b\d+(?:\.\d+)?\s?%"],
    "Emails": [r"[\w.+-]+@[\w-]+\.[\w.-]+\w"],
    "Links": [r"https?://[^\s)>\]]+"],
    "Phone numbers": [r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)", r"\+\d{1,3}[\s-]?\d{2,4}[\s-]?\d{3,4}[\s-]?\d{3,4}"],
}


def _decode(content):
    for encoding in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return content.decode(encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return content.decode("utf-8", errors="ignore")


def _docx_text(content):
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    lines = ["".join(t.text or "" for t in p.iter(f"{WORD_NS}t")) for p in root.iter(f"{WORD_NS}p")]
    return "\n\n".join(line for line in lines if line.strip())


def extract_pages(name, content):
    extension = Path(name).suffix.lower()
    if extension == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("This PDF is password protected.")
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    if extension == ".docx":
        return [(None, _docx_text(content))]
    if extension in {".txt", ".md"}:
        return [(None, _decode(content))]
    raise ValueError(f"Unsupported file type '{extension}'. Use PDF, TXT, MD or DOCX.")


def _split_long(paragraph, size):
    pieces, buffer = [], ""
    for sentence in split_sentences(paragraph):
        while len(sentence) > size:
            cut = sentence.rfind(" ", 0, size)
            cut = cut if cut > 0 else size
            pieces.append(sentence[:cut])
            sentence = sentence[cut:].strip()
        if buffer and len(buffer) + len(sentence) + 1 > size:
            pieces.append(buffer)
            buffer = sentence
        else:
            buffer = f"{buffer} {sentence}".strip()
    if buffer:
        pieces.append(buffer)
    return pieces


def make_chunks(pages, size):
    chunks = []
    for page, text in pages:
        buffer = ""
        for paragraph in re.split(r"\n\s*\n", text):
            paragraph = " ".join(paragraph.split())
            if not paragraph:
                continue
            for piece in [paragraph] if len(paragraph) <= size else _split_long(paragraph, size):
                if buffer and len(buffer) + len(piece) + 1 > size:
                    chunks.append({"text": buffer, "page": page})
                    buffer = piece
                else:
                    buffer = f"{buffer} {piece}".strip()
        if buffer:
            chunks.append({"text": buffer, "page": page})
    return chunks


def _save(doc):
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / f"{doc['id']}.json").write_text(json.dumps(doc, ensure_ascii=False), "utf-8")
    vectors = _vectors.get(doc["id"])
    if vectors is not None:
        np.save(DOCS_DIR / f"{doc['id']}.npy", vectors)


def _embed(doc):
    if not embeddings.ready():
        return None
    with metrics.timed("Embed document chunks", embeddings.backend_label(), f"{len(doc['chunks'])} chunks"):
        return embeddings.embed([c["text"] for c in doc["chunks"]])


def add(name, content):
    global _version
    with metrics.timed("Parse document", "CPU", name):
        pages = extract_pages(name, content)
    text = "\n\n".join(t for _, t in pages).strip()
    if len(text) < 20:
        raise ValueError("No readable text found. Scanned PDFs need OCR, which is not part of the document pipeline.")
    doc = {
        "id": uuid.uuid4().hex[:10],
        "name": name,
        "kind": Path(name).suffix.lower().lstrip("."),
        "uploaded_at": time.strftime("%Y-%m-%d %H:%M"),
        "pages": len([1 for p, _ in pages if p]) or None,
        "words": len(text.split()),
        "text": text,
        "chunks": make_chunks(pages, config.get()["chunk_size"]),
        "analysis": None,
    }
    with _lock:
        vectors = _embed(doc)
        if vectors is not None:
            _vectors[doc["id"]] = vectors
        _docs[doc["id"]] = doc
        _version += 1
        _save(doc)
    return public(doc)


def public(doc):
    return {
        "id": doc["id"],
        "name": doc["name"],
        "kind": doc["kind"],
        "uploaded_at": doc["uploaded_at"],
        "pages": doc["pages"],
        "words": doc["words"],
        "chunks": len(doc["chunks"]),
        "embedded": doc["id"] in _vectors,
        "analyzed": bool(doc["analysis"]),
    }


def listing():
    with _lock:
        return [public(d) for d in sorted(_docs.values(), key=lambda d: d["uploaded_at"], reverse=True)]


def get(doc_id):
    doc = _docs.get(doc_id)
    if doc is None:
        raise KeyError("Document not found")
    return doc


def remove(doc_id):
    global _version
    with _lock:
        _docs.pop(doc_id, None)
        _vectors.pop(doc_id, None)
        for suffix in (".json", ".npy"):
            (DOCS_DIR / f"{doc_id}{suffix}").unlink(missing_ok=True)
        _version += 1


def clear():
    for doc_id in list(_docs):
        remove(doc_id)


def load_all():
    global _version
    if not DOCS_DIR.exists():
        return
    for file in DOCS_DIR.glob("*.json"):
        try:
            doc = json.loads(file.read_text("utf-8"))
            _docs[doc["id"]] = doc
            vector_file = file.with_suffix(".npy")
            if vector_file.exists():
                vectors = np.load(vector_file)
                if len(vectors) == len(doc["chunks"]):
                    _vectors[doc["id"]] = vectors
        except (ValueError, KeyError, OSError):
            continue
    _version += 1


def reindex():
    done = 0
    for doc in list(_docs.values()):
        if doc["id"] not in _vectors:
            vectors = _embed(doc)
            if vectors is not None:
                _vectors[doc["id"]] = vectors
                _save(doc)
                done += 1
    return done


def _index_for(doc_ids):
    key = (tuple(doc_ids), _version)
    if _index["key"] == key:
        return _index
    entries = [(d, i) for d in doc_ids for i in range(len(_docs[d]["chunks"]))]
    bm25 = BM25([tokenize(_docs[d]["chunks"][i]["text"]) for d, i in entries])
    matrix = None
    if doc_ids and all(d in _vectors for d in doc_ids):
        matrix = np.vstack([_vectors[d] for d in doc_ids])
    _index.update(key=key, entries=entries, bm25=bm25, matrix=matrix)
    return _index


def search(query, doc_ids=None, limit=None):
    limit = limit or config.get()["top_k"]
    with _lock:
        ids = [d for d in (doc_ids or list(_docs)) if d in _docs]
        if not ids:
            return []
        index = _index_for(ids)
        tokens = tokenize(query)
        with metrics.timed("Search documents", "CPU (BM25)" if index["matrix"] is None else "BM25 + embeddings"):
            scores = np.array(index["bm25"].scores(tokens), dtype=np.float32)
            if scores.max(initial=0) > 0:
                scores = scores / scores.max()
            if index["matrix"] is not None and embeddings.ready():
                query_vector = embeddings.embed([query])[0]
                scores = 0.5 * scores + 0.5 * np.clip(index["matrix"] @ query_vector, 0, None)
        hits = []
        for position in np.argsort(-scores)[:limit]:
            if scores[position] <= 0:
                continue
            doc_id, chunk_index = index["entries"][position]
            chunk = _docs[doc_id]["chunks"][chunk_index]
            hits.append(
                {
                    "doc_id": doc_id,
                    "doc": _docs[doc_id]["name"],
                    "page": chunk["page"],
                    "text": chunk["text"],
                    "score": round(float(scores[position]), 3),
                    "terms": tokens,
                }
            )
        return hits


def coverage(query):
    tokens = tokenize(query)
    if len(tokens) < 2 or not _docs:
        return 0.0
    with _lock:
        vocabulary = set(_index_for(list(_docs))["bm25"].idf)
    return sum(t in vocabulary for t in tokens) / len(tokens)


def find_named(text):
    lowered = text.lower()
    return [d["id"] for d in _docs.values() if Path(d["name"]).stem.lower() in lowered or d["name"].lower() in lowered]


def rank_sentences(sentences):
    frequency = Counter(tokenize(" ".join(sentences)))
    scored = []
    for position, sentence in enumerate(sentences):
        tokens = tokenize(sentence)
        if len(tokens) < 5 or len(sentence) > 400:
            scored.append(0.0)
            continue
        scored.append(sum(frequency[t] for t in set(tokens)) / len(set(tokens)) * (1 + 0.3 / (1 + position / 10)))
    return scored


def pick_salient(sentences, budget):
    scores = rank_sentences(sentences)
    chosen, used = [], 0
    for position in np.argsort(scores)[::-1]:
        if scores[position] <= 0 or used + len(sentences[position]) > budget:
            continue
        chosen.append(int(position))
        used += len(sentences[position])
    return [sentences[p] for p in sorted(chosen)]


def extract_entities(text):
    found = {}
    for label, patterns in ENTITY_PATTERNS.items():
        values = []
        for pattern in patterns:
            values += [m.group(0).strip() for m in re.finditer(pattern, text)]
        unique = list(dict.fromkeys(values))[:10]
        if unique:
            found[label] = unique
    return found


def top_keywords(text, count=12):
    words = [t for t in tokenize(text) if len(t) > 3 and not t.isdigit()]
    return [w for w, _ in Counter(words).most_common(count)]


def _extractive_markdown(sentences):
    scores = np.array(rank_sentences(sentences))
    order = [int(i) for i in np.argsort(-scores) if scores[i] > 0]
    summary = [sentences[i] for i in sorted(order[:4])]
    points = [sentences[i] for i in sorted(order[4:10])]
    text = "## Summary\n" + " ".join(summary or sentences[:3])
    if points:
        text += "\n\n## Key points\n" + "\n".join(f"- {p}" for p in points)
    return text


def analyze(doc_id, refresh=False):
    doc = get(doc_id)
    if doc["analysis"] and not refresh:
        return doc["analysis"]
    text = doc["text"]
    sentences = split_sentences(text)
    analysis = {
        "entities": extract_entities(text),
        "keywords": top_keywords(text),
        "mode": "extractive",
        "note": "",
        "stats": {},
        "privacy": llm.privacy(),
    }
    source = text if len(text) <= 6000 else "\n".join(pick_salient(sentences, 5000))
    if len(text) > 6000:
        analysis["note"] = "Long document: the most important passages were selected locally before summarizing."
    if llm.status()["ready"]:
        prompt = (
            "Summarize the document below. Reply in Markdown with exactly two sections: "
            "'## Summary' (3 to 5 sentences) and '## Key points' (5 to 8 short bullet points).\n\n"
            f"Document: {doc['name']}\n\n{source}"
        )
        try:
            markdown, stats = llm.complete(
                [{"role": "system", "content": SUMMARY_SYSTEM}, {"role": "user", "content": prompt}], temperature=0.1
            )
            analysis.update(markdown=markdown, mode="llm", stats=stats)
        except llm.LLMError as exc:
            analysis["note"] = f"Language model failed ({exc}). Showing rule-based summary."
    if "markdown" not in analysis:
        analysis["markdown"] = _extractive_markdown(sentences)
    doc["analysis"] = analysis
    _save(doc)
    return analysis


def entities_markdown(doc_id):
    doc = get(doc_id)
    entities = extract_entities(doc["text"])
    if not entities:
        return f"No dates, amounts, emails, links or phone numbers were found in **{doc['name']}**."
    lines = [f"Important information found in **{doc['name']}**:"]
    for label, values in entities.items():
        lines.append(f"- **{label}:** " + ", ".join(values))
    return "\n".join(lines)


def prepare_answer(question, doc_ids=None):
    hits = search(question, doc_ids)
    sources = [
        {"n": i + 1, "doc": h["doc"], "page": h["page"], "text": h["text"], "score": h["score"]} for i, h in enumerate(hits)
    ]
    if not hits:
        return None, sources
    context = "\n\n".join(f"[{s['n']}] ({s['doc']}{', page ' + str(s['page']) if s['page'] else ''})\n{s['text']}" for s in sources)
    messages = [
        {"role": "system", "content": QA_SYSTEM},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    return messages, sources


def extractive_answer(sources):
    if not sources:
        return "I could not find anything about that in your documents."
    best = sources[0]
    where = f" (page {best['page']})" if best["page"] else ""
    return f"No language model is running, so here is the best matching passage from **{best['doc']}**{where}:\n\n> {best['text']}"


def ask(question, doc_ids=None):
    messages, sources = prepare_answer(question, doc_ids)
    result = {"sources": sources, "privacy": llm.privacy(), "mode": "extractive", "stats": {}}
    if messages is None:
        result["answer"] = extractive_answer(sources)
        return result
    if llm.status()["ready"]:
        try:
            answer, stats = llm.complete(messages, temperature=0.1)
            result.update(answer=answer, mode="llm", stats=stats)
            return result
        except llm.LLMError as exc:
            result["error"] = str(exc)
    result["answer"] = extractive_answer(sources)
    return result
