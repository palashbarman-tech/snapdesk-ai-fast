import re
import time

from . import data_analysis, documents, hardware, llm, vision

SYSTEM = (
    "You are SnapDesk AI, a private assistant running locally on the user's Snapdragon PC. "
    "Be concise and accurate. Reply in the language the user writes in. If you do not know, say so."
)

DATA_WORDS = re.compile(
    r"\b(average|mean|median|sum|total|column|columns|row|rows|dataset|csv|excel|spreadsheet|sheet|correlat\w*|outliers?|"
    r"missing|duplicates?|chart|plot|graph|trend\w*|group|top \d+|highest|lowest|maximum|minimum|per month|by month|"
    r"statistics|stats|revenue|sales)\b",
    re.I,
)
DOC_WORDS = re.compile(
    r"\b(document|documents|pdf|docx|file|report|policy|contract|paper|article|according to|says?|mentions?|clause|"
    r"section|page|summar\w*|key points?|main points?|takeaways?|deadline|agreement)\b",
    re.I,
)
SYSTEM_WORDS = re.compile(r"\b(npu|snapdragon|hexagon|accelerator|hardware|which model are you|running locally|on-device|on device)\b", re.I)
SEARCH_TASK = re.compile(r"^\s*(search|find|look for|locate)\b|\bwhere (does|is|do)\b.*\bmention", re.I)
EXTRACT_TASK = re.compile(r"\b(extract|important (info|information|details)|key (info|information|details)|dates|amounts|emails?|phone)\b", re.I)
SUMMARY_TASK = re.compile(r"summar\w*|tl;?dr|key points?|main points?|highlights|takeaways|overview of (the )?(doc|document|pdf|file)|सारांश|संक्षेप", re.I)
OVERVIEW_TASK = re.compile(r"\b(analy[sz]e|overview|profile|insights?|explore|describe the (data|dataset)|summar\w*)\b", re.I)
CHART_TASK = re.compile(r"\b(chart|plot|graph|visuali[sz]e|histogram)\b", re.I)


def _llm_route(message):
    try:
        text, _ = llm.complete(
            [
                {
                    "role": "user",
                    "content": "Classify the request as one word: document (about text files), data (about tables, numbers, "
                    f"spreadsheets) or general.\nRequest: {message}\nAnswer:",
                }
            ],
            max_tokens=4,
            temperature=0,
        )
    except llm.LLMError:
        return None
    word = text.lower().strip()
    return next((w for w in ("document", "data", "general") if w in word), None)


def route(message, has_image):
    docs, datasets = documents.listing(), data_analysis.listing()
    base = {"route": "general", "doc_ids": [], "dataset_id": None, "reason": ""}
    if has_image:
        return {**base, "route": "image", "reason": "An image was attached."}

    named_docs, named_sets = documents.find_named(message), data_analysis.find_named(message)
    column_hits = data_analysis.mentioned_columns(message)
    best_set = max(column_hits, key=column_hits.get) if column_hits else None
    if named_sets:
        best_set = named_sets[0]
    data_score = len(DATA_WORDS.findall(message)) + 2 * (column_hits.get(best_set, 0) if best_set else 0) + 3 * bool(named_sets)
    doc_score = len(DOC_WORDS.findall(message)) + 3 * bool(named_docs) + (2 if documents.coverage(message) >= 0.6 else 0)

    if SYSTEM_WORDS.search(message) and data_score + doc_score <= 1:
        return {**base, "route": "system", "reason": "Question about this PC or the AI runtime."}

    if docs and datasets:
        if data_score == doc_score and data_score > 0:
            choice = _llm_route(message) or "document"
        elif data_score == doc_score:
            choice = "general"
        else:
            choice = "data" if data_score > doc_score else "document"
    elif docs:
        choice = "document" if doc_score + data_score > 0 else "general"
    elif datasets:
        choice = "data" if data_score > 0 else "general"
    else:
        choice = "general"

    if choice == "document":
        return {**base, "route": "document", "doc_ids": named_docs, "reason": "Matches your uploaded documents."}
    if choice == "data":
        dataset_id = best_set or datasets[0]["id"]
        return {**base, "route": "data", "dataset_id": dataset_id, "reason": "Matches your uploaded dataset."}
    return {**base, "reason": "General question."}


AREAS = {"document": "Documents", "data": "Data analysis", "image": "Image analysis", "system": "System", "general": "Chat"}


def _step(area, task):
    return {"type": "route", "route": area, "label": f"{AREAS[area]} | {task}"}


def _tokens(text):
    yield {"type": "token", "text": text}


def _stream_llm(messages, ctx):
    stats = {}
    for piece in llm.stream(messages, stats=stats):
        yield {"type": "token", "text": piece}
    ctx.update(stats=stats, used_llm=True)


def _documents(message, decision, ctx):
    ids = decision["doc_ids"]
    if SEARCH_TASK.search(message):
        query = re.sub(r"^\s*(search|find|look for|locate)( for)?\s*", "", message, flags=re.I)
        hits = documents.search(query or message, ids or None, limit=6)
        sources = [{"n": i + 1, "doc": h["doc"], "page": h["page"], "text": h["text"], "score": h["score"]} for i, h in enumerate(hits)]
        yield _step("document", "Search")
        yield from _tokens(f"Found {len(hits)} matching passage(s)." if hits else "No matching passages found.")
        yield {"type": "extras", "sources": sources}
        return
    if SUMMARY_TASK.search(message) or EXTRACT_TASK.search(message):
        target = ids[0] if ids else (documents.listing()[0]["id"] if documents.listing() else None)
        if target is None:
            yield from _tokens("Upload a document first.")
            return
        name = documents.get(target)["name"]
        if SUMMARY_TASK.search(message):
            yield _step("document", f"Summary of {name}")
            analysis = documents.analyze(target)
            ctx.update(stats=analysis["stats"], used_llm=analysis["mode"] == "llm")
            yield from _tokens(analysis["markdown"] + (f"\n\n_{analysis['note']}_" if analysis["note"] else ""))
        else:
            yield _step("document", f"Key information in {name}")
            yield from _tokens(documents.entities_markdown(target))
        return
    yield _step("document", "Question answering")
    messages, sources = documents.prepare_answer(message, ids or None)
    if messages is None:
        yield from _tokens("I could not find anything about that in your documents.")
    elif llm.status()["ready"]:
        yield from _stream_llm(messages, ctx)
    else:
        yield from _tokens(documents.extractive_answer(sources))
    yield {"type": "extras", "sources": sources}


def _data(message, decision, ctx):
    dataset = data_analysis.get(decision["dataset_id"])
    wants_chart = CHART_TASK.search(message)
    plan, result = data_analysis.query(dataset, message)
    if plan is None or result is None:
        if OVERVIEW_TASK.search(message) or wants_chart:
            analysis = data_analysis.analyze(dataset)
            yield _step("data", f"Automatic analysis of {dataset['name']}")
            yield from _tokens(data_analysis.overview_markdown(dataset))
            yield {"type": "extras", "charts": analysis["charts"][:4]}
            return
        yield _step("data", "Question about the dataset")
        if llm.status()["ready"]:
            messages = [
                {"role": "system", "content": SYSTEM + " Use only the dataset summary provided. Never invent numbers."},
                {"role": "user", "content": f"{data_analysis.profile_text(dataset)}\n\nQuestion: {message}"},
            ]
            yield from _stream_llm(messages, ctx)
        else:
            yield from _tokens(
                "I could not turn that into a calculation without a language model. Try: 'average Sales by Region', "
                "'top 5 rows by Revenue' or 'correlation between Price and Quantity'."
            )
        return
    yield _step("data", f"Calculated with pandas on {dataset['name']}")
    yield from _tokens(data_analysis.describe_result(result))
    extras = {"plan": plan}
    if "error" not in result:
        extras["table"] = {"columns": result["columns"], "rows": result["rows"][:20]}
        if result.get("chart"):
            extras["charts"] = [result["chart"]]
    yield {"type": "extras", **extras}
    ctx["used_llm"] = llm.status()["ready"]


def _system_answer():
    info = hardware.detect()
    model = llm.status()
    npu = info["npus"][0]["name"] if info["npus"] else "none detected"
    qnn = "QNNExecutionProvider" in info["onnxruntime"]["providers"]
    lines = [
        f"- **Device:** {info['device']} ({info['cpu']['name']}), {info['memory_gb']} GB RAM",
        f"- **Snapdragon detected:** {'yes' if info['snapdragon'] else 'no'}",
        f"- **NPU device:** {npu}",
        f"- **ONNX Runtime QNN provider:** {'available' if qnn else 'not available'}",
        f"- **Language model:** {model['provider']} {model['model']} ({'ready' if model['ready'] else 'not ready'}, "
        f"{'local' if model['local'] else 'external'})",
        "- The language model runs through its own runtime. SnapDesk AI does not claim NPU use for it unless the runtime is Genie.",
    ]
    return "Here is what SnapDesk AI detected on this PC:\n\n" + "\n".join(lines)


def _image(message, content, ctx):
    yield _step("image", "Analyzing image")
    result = vision.analyze(content, message if len(message.split()) > 2 else "")
    yield from _tokens(vision.to_markdown(result))
    yield {"type": "extras", "components": result["components"]}
    ctx["used_llm"] = any("model" in c["name"].lower() for c in result["components"])


def chat(message, history=None, image=None):
    started = time.perf_counter()
    ctx = {"stats": {}, "used_llm": False}
    decision = route(message, image is not None)
    yield {**_step(decision["route"], "routing"), "reason": decision["reason"]}
    try:
        if decision["route"] == "image":
            yield from _image(message, image, ctx)
        elif decision["route"] == "document":
            yield from _documents(message, decision, ctx)
        elif decision["route"] == "data":
            yield from _data(message, decision, ctx)
        elif decision["route"] == "system":
            yield from _tokens(_system_answer())
        elif llm.status()["ready"]:
            turns = [{"role": m["role"], "content": m["content"]} for m in (history or [])[-6:] if m.get("role") in {"user", "assistant"}]
            yield from _stream_llm([{"role": "system", "content": SYSTEM}, *turns, {"role": "user", "content": message}], ctx)
        else:
            state = llm.status()
            yield from _tokens(
                "No local language model is running, so I can only help with rule-based features right now.\n\n"
                f"{state['detail']}\n\nUpload a document or dataset and ask about it, or see the README to enable a model."
            )
    except llm.LLMError as exc:
        yield {"type": "error", "message": str(exc)}
    except Exception as exc:
        yield {"type": "error", "message": f"Something went wrong: {exc}"}
    privacy = llm.privacy() if ctx["used_llm"] else {"local": True, "external": False, "label": "Local | no model used"}
    yield {"type": "done", "privacy": privacy, "stats": ctx["stats"], "total_ms": round((time.perf_counter() - started) * 1000)}
