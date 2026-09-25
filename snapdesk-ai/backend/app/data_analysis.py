import difflib
import io
import json
import re
import time
import uuid
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, llm, metrics
from .utils import to_native

DATA_DIR = config.DATA_DIR / "datasets"
SUPPORTED = {".csv", ".tsv", ".xlsx", ".xlsm", ".xls"}
AGGREGATES = {"sum", "mean", "median", "min", "max", "count", "nunique"}
SUM_HINT = re.compile(r"sales|revenue|amount|quantity|units|profit|cost|total|orders|price|spend", re.I)
DATE_LIKE = re.compile(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}|[A-Za-z]{3,9}\.? \d{1,2},? \d{4}")
NUMBER_LIKE = re.compile(r"^[\s₹$€£+-]*\d[\d,]*\.?\d*\s*%?$")

_datasets = {}


def _read_csv(content, separator):
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            frame = pd.read_csv(io.BytesIO(content), sep=separator, encoding=encoding)
            if frame.shape[1] == 1 and separator == ",":
                frame = pd.read_csv(io.BytesIO(content), sep=None, engine="python", encoding=encoding)
            return frame
        except UnicodeDecodeError:
            continue
        except pd.errors.ParserError:
            return pd.read_csv(io.BytesIO(content), sep=None, engine="python", encoding=encoding)
    raise ValueError("Could not decode this file.")


def read_file(filename, content):
    extension = Path(filename).suffix.lower()
    if extension in {".csv", ".tsv"}:
        return [(None, _read_csv(content, "\t" if extension == ".tsv" else ","))]
    if extension in {".xlsx", ".xlsm", ".xls"}:
        sheets = pd.read_excel(io.BytesIO(content), sheet_name=None)
        return [(name, frame) for name, frame in sheets.items() if not frame.dropna(how="all").empty]
    raise ValueError(f"Unsupported file type '{extension}'. Use CSV or Excel.")


def prepare(frame):
    frame = frame.copy()
    frame.columns = [str(c).strip() for c in frame.columns]
    frame = frame.dropna(how="all").dropna(axis=1, how="all")
    frame = frame.loc[:, ~frame.columns.str.match(r"^Unnamed: \d+$")]
    for column in [c for c in frame.columns if _is_text(frame[c])]:
        values = frame[column].dropna().astype(str)
        if values.empty:
            continue
        if values.map(lambda v: bool(NUMBER_LIKE.match(v))).mean() >= 0.9 and not values.str.match(r"^0\d").any():
            cleaned = frame[column].astype(str).str.replace(r"[,\s₹$€£%]", "", regex=True)
            frame[column] = pd.to_numeric(cleaned, errors="coerce")
        elif values.head(20).map(lambda v: bool(DATE_LIKE.search(v))).mean() >= 0.8:
            frame[column] = _parse_dates(frame[column])
    return frame


def _is_text(series):
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


def _parse_dates(series):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = pd.to_datetime(series, errors="coerce")
        second = pd.to_datetime(series, errors="coerce", dayfirst=True)
    best = second if second.notna().sum() > first.notna().sum() else first
    return best if best.notna().mean() >= 0.8 else series


def _dataset_files(dataset_id):
    return DATA_DIR / f"{dataset_id}.csv", DATA_DIR / f"{dataset_id}.json"


def add(filename, content):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with metrics.timed("Load dataset", "CPU (pandas)", filename):
        sheets = read_file(filename, content)
    if not sheets:
        raise ValueError("The file has no data.")
    added = []
    for sheet, frame in sheets:
        frame = prepare(frame)
        if frame.empty:
            continue
        dataset = {
            "id": uuid.uuid4().hex[:10],
            "name": filename if sheet is None else f"{filename} - {sheet}",
            "filename": filename,
            "sheet": sheet,
            "uploaded_at": time.strftime("%Y-%m-%d %H:%M"),
            "df": frame,
            "analysis": None,
            "ai_insights": None,
        }
        _datasets[dataset["id"]] = dataset
        csv_file, meta_file = _dataset_files(dataset["id"])
        frame.to_csv(csv_file, index=False)
        meta_file.write_text(json.dumps({k: v for k, v in dataset.items() if k not in {"df", "analysis"}}), "utf-8")
        added.append(public(dataset))
    return added


def public(dataset):
    frame = dataset["df"]
    return {
        "id": dataset["id"],
        "name": dataset["name"],
        "uploaded_at": dataset["uploaded_at"],
        "rows": len(frame),
        "columns": len(frame.columns),
    }


def listing():
    return [public(d) for d in sorted(_datasets.values(), key=lambda d: d["uploaded_at"], reverse=True)]


def get(dataset_id):
    dataset = _datasets.get(dataset_id)
    if dataset is None:
        raise KeyError("Dataset not found")
    return dataset


def remove(dataset_id):
    _datasets.pop(dataset_id, None)
    for file in _dataset_files(dataset_id):
        file.unlink(missing_ok=True)


def clear():
    for dataset_id in list(_datasets):
        remove(dataset_id)


def load_all():
    if not DATA_DIR.exists():
        return
    for meta_file in DATA_DIR.glob("*.json"):
        try:
            meta = json.loads(meta_file.read_text("utf-8"))
            frame = prepare(pd.read_csv(meta_file.with_suffix(".csv")))
            _datasets[meta["id"]] = {**meta, "df": frame, "analysis": None, "ai_insights": None}
        except (ValueError, KeyError, OSError):
            continue


def find_named(text):
    lowered = text.lower()
    return [d["id"] for d in _datasets.values() if Path(d["filename"]).stem.lower() in lowered]


def mentioned_columns(text):
    lowered = text.lower().replace("_", " ")
    counts = {}
    for dataset_id, dataset in _datasets.items():
        counts[dataset_id] = sum(
            1 for c in dataset["df"].columns if re.search(rf"\b{re.escape(c.lower().replace('_', ' '))}s?\b", lowered)
        )
    return counts


def _is_identifier(series, rows):
    if rows < 20 or series.nunique() != series.notna().sum():
        return False
    if pd.api.types.is_integer_dtype(series):
        return series.is_monotonic_increasing or "id" in str(series.name).lower()
    return _is_text(series)


def _kind(series, rows):
    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if _is_identifier(series, rows):
        return "identifier"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if series.nunique() > 50 and series.nunique() / max(rows, 1) > 0.5:
        return "text"
    return "categorical"


def _profile(series, kind, rows):
    missing = int(series.isna().sum())
    base = {
        "name": series.name,
        "kind": kind,
        "dtype": str(series.dtype),
        "missing": missing,
        "missing_pct": round(missing / max(rows, 1) * 100, 2),
        "unique": int(series.nunique()),
    }
    if kind == "numeric" and series.notna().any():
        base.update(
            mean=series.mean(),
            median=series.median(),
            std=series.std(),
            min=series.min(),
            max=series.max(),
            q1=series.quantile(0.25),
            q3=series.quantile(0.75),
            skew=series.skew(),
        )
    elif kind == "datetime" and series.notna().any():
        base.update(min=series.min(), max=series.max())
    elif kind in {"categorical", "text"}:
        base["top"] = [{"value": str(k), "count": int(v)} for k, v in series.value_counts().head(5).items()]
    return base


def _outliers(series):
    clean = series.dropna()
    if len(clean) < 8 or clean.std() == 0:
        return None
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    spread = q3 - q1
    low, high = q1 - 1.5 * spread, q3 + 1.5 * spread
    flagged = clean[(clean < low) | (clean > high)]
    if flagged.empty:
        return None
    return {
        "column": series.name,
        "count": int(len(flagged)),
        "pct": round(len(flagged) / len(clean) * 100, 2),
        "lower": low,
        "upper": high,
        "examples": flagged.sort_values(key=lambda s: (s - clean.median()).abs(), ascending=False).head(5).tolist(),
    }


def _period_for(dates):
    span = (dates.max() - dates.min()).days
    return "M" if span > 120 else "D"


def _trends(frame, date_column, numeric_columns):
    dates = frame[date_column].dropna()
    if len(dates) < 8:
        return []
    period = _period_for(dates)
    keys = frame[date_column].dt.to_period(period)
    trends = []
    for column in numeric_columns[:5]:
        how = "sum" if SUM_HINT.search(column) else "mean"
        series = frame.groupby(keys)[column].agg(how).dropna()
        if len(series) < 4:
            continue
        x = np.arange(len(series))
        y = series.to_numpy(dtype=float)
        slope = np.polyfit(x, y, 1)[0]
        correlation = np.corrcoef(x, y)[0, 1] if y.std() > 0 else 0.0
        drift = slope * (len(y) - 1) / abs(y.mean()) * 100 if y.mean() else 0.0
        direction = "increasing" if drift > 10 and correlation > 0.5 else "decreasing" if drift < -10 and correlation < -0.5 else "stable"
        trends.append(
            {
                "column": column,
                "aggregate": how,
                "period": "month" if period == "M" else "day",
                "direction": direction,
                "change_pct": round(drift, 1),
                "labels": [str(p) for p in series.index],
                "values": y.tolist(),
            }
        )
    return trends


def _histogram_chart(series):
    counts, edges = np.histogram(series.dropna(), bins=12)
    labels = [f"{edges[i]:.4g} to {edges[i + 1]:.4g}" for i in range(len(counts))]
    return {
        "id": f"hist-{series.name}",
        "title": f"Distribution of {series.name}",
        "kind": "bar",
        "labels": labels,
        "series": [{"label": "Rows", "data": counts.tolist()}],
    }


def _bar_chart(series, title, horizontal=True):
    return {
        "id": f"bar-{title}",
        "title": title,
        "kind": "bar",
        "horizontal": horizontal,
        "labels": [str(i) for i in series.index],
        "series": [{"label": series.name or "Value", "data": series.tolist()}],
    }


def _insights(frame, columns, duplicates, outliers, pairs, trends):
    rows = len(frame)
    items = []
    kinds = [c["kind"] for c in columns]
    items.append(
        {
            "level": "info",
            "text": f"{rows:,} rows and {len(columns)} columns: {kinds.count('numeric')} numeric, "
            f"{kinds.count('categorical')} categorical, {kinds.count('datetime')} date.",
        }
    )
    missing = sorted([c for c in columns if c["missing"]], key=lambda c: -c["missing_pct"])
    if missing:
        worst = ", ".join(f"{c['name']} ({c['missing_pct']}%)" for c in missing[:3])
        items.append({"level": "warn", "text": f"Missing values in {len(missing)} column(s). Highest: {worst}."})
    else:
        items.append({"level": "good", "text": "No missing values."})
    if duplicates:
        items.append({"level": "warn", "text": f"{duplicates:,} duplicate rows ({duplicates / rows * 100:.1f}%). Consider removing them."})
    else:
        items.append({"level": "good", "text": "No duplicate rows."})
    for item in sorted(outliers, key=lambda o: -o["pct"])[:3]:
        items.append(
            {"level": "warn", "text": f"{item['column']} has {item['count']} outlier(s) ({item['pct']}%) outside {item['lower']:.4g} to {item['upper']:.4g}."}
        )
    for a, b, r in pairs[:2]:
        items.append({"level": "info", "text": f"{a} and {b} are {'positively' if r > 0 else 'negatively'} correlated (r = {r:.2f})."})
    for trend in trends[:3]:
        if trend["direction"] != "stable":
            items.append(
                {"level": "info", "text": f"{trend['column']} is {trend['direction']} over time ({trend['change_pct']:+.0f}% across the {trend['period']}ly {trend['aggregate']})."}
            )
    for column in columns:
        if column["kind"] == "categorical" and column.get("top") and rows:
            share = column["top"][0]["count"] / rows * 100
            if share > 60 and column["unique"] > 1:
                items.append({"level": "info", "text": f"{column['name']} is dominated by '{column['top'][0]['value']}' ({share:.0f}% of rows)."})
        if column["unique"] <= 1 and rows > 1:
            items.append({"level": "warn", "text": f"{column['name']} has a single value and adds no information."})
        if column["kind"] == "identifier":
            items.append({"level": "info", "text": f"{column['name']} looks like an identifier and was left out of statistics."})
    return items[:14]


def analyze(dataset, refresh=False):
    if dataset["analysis"] and not refresh:
        return dataset["analysis"]
    started = time.perf_counter()
    frame = dataset["df"]
    rows = len(frame)
    columns = [_profile(frame[c], _kind(frame[c], rows), rows) for c in frame.columns]
    by_kind = lambda kind: [c["name"] for c in columns if c["kind"] == kind]
    numeric, categorical, dates = by_kind("numeric"), by_kind("categorical"), by_kind("datetime")

    duplicates = int(frame.duplicated().sum())
    outliers = [o for o in (_outliers(frame[c]) for c in numeric) if o]

    matrix, pairs = None, []
    if len(numeric) >= 2:
        corr = frame[numeric[:10]].corr()
        matrix = {"columns": corr.columns.tolist(), "values": corr.values.tolist()}
        names = corr.columns.tolist()
        pairs = sorted(
            [(a, b, float(corr.loc[a, b])) for i, a in enumerate(names) for b in names[i + 1 :] if not np.isnan(corr.loc[a, b])],
            key=lambda p: -abs(p[2]),
        )
        pairs = [p for p in pairs if abs(p[2]) >= 0.5]

    trends = _trends(frame, dates[0], numeric) if dates and numeric else []

    charts = []
    missing_series = pd.Series({c["name"]: c["missing_pct"] for c in columns if c["missing"]}, dtype=float, name="Missing %")
    if not missing_series.empty:
        charts.append(_bar_chart(missing_series.sort_values(ascending=False).head(10), "Missing values by column (%)"))
    for trend in trends[:2]:
        charts.append(
            {
                "id": f"trend-{trend['column']}",
                "title": f"{trend['column']} over time ({trend['period']}ly {trend['aggregate']})",
                "kind": "line",
                "labels": trend["labels"],
                "series": [{"label": trend["column"], "data": trend["values"]}],
            }
        )
    for name in categorical:
        if 2 <= frame[name].nunique() <= 30 and len([c for c in charts if c["id"].startswith("bar-Top")]) < 3:
            charts.append(_bar_chart(frame[name].value_counts().head(10).rename("Rows"), f"Top {name} values"))
    for name in numeric[:4]:
        charts.append(_histogram_chart(frame[name]))
    if pairs:
        a, b, r = pairs[0]
        sample = frame[[a, b]].dropna().sample(min(400, len(frame[[a, b]].dropna())), random_state=0)
        charts.append(
            {
                "id": f"scatter-{a}-{b}",
                "title": f"{a} vs {b} (r = {r:.2f})",
                "kind": "scatter",
                "x_title": a,
                "y_title": b,
                "series": [{"label": f"{a} vs {b}", "data": [{"x": x, "y": y} for x, y in zip(sample[a], sample[b])]}],
            }
        )

    result = to_native(
        {
            "id": dataset["id"],
            "name": dataset["name"],
            "rows": rows,
            "columns": columns,
            "duplicates": duplicates,
            "missing_total": int(frame.isna().sum().sum()),
            "outliers": outliers,
            "correlation": matrix,
            "top_pairs": [{"a": a, "b": b, "r": r} for a, b, r in pairs[:5]],
            "trends": [{k: v for k, v in t.items() if k not in {"labels", "values"}} for t in trends],
            "insights": _insights(frame, columns, duplicates, outliers, pairs, trends),
            "charts": charts,
            "preview": frame.head(10).astype(object).where(frame.head(10).notna(), None).to_dict("records"),
        }
    )
    metrics.record("Analyze dataset", (time.perf_counter() - started) * 1000, "CPU (pandas)", dataset["name"])
    dataset["analysis"] = result
    return result


def overview_markdown(dataset):
    analysis = analyze(dataset)
    icons = {"good": "OK", "warn": "Check", "info": "Note"}
    lines = [f"**{analysis['name']}** at a glance:"]
    lines += [f"- {icons[i['level']]}: {i['text']}" for i in analysis["insights"]]
    return "\n".join(lines)


def profile_text(dataset, limit=3500):
    analysis = analyze(dataset)
    lines = [f"Dataset: {analysis['name']}, {analysis['rows']} rows, {analysis['duplicates']} duplicate rows."]
    for c in analysis["columns"]:
        if c["kind"] == "numeric":
            lines.append(f"- {c['name']} (numeric): mean {c.get('mean')}, min {c.get('min')}, max {c.get('max')}, missing {c['missing_pct']}%")
        elif c["kind"] in {"categorical", "text"}:
            top = ", ".join(f"{t['value']} ({t['count']})" for t in c.get("top", [])[:3])
            lines.append(f"- {c['name']} (categorical, {c['unique']} unique): {top}")
        else:
            lines.append(f"- {c['name']} ({c['kind']})")
    for pair in analysis["top_pairs"]:
        lines.append(f"Correlation {pair['a']} vs {pair['b']}: {pair['r']:.2f}")
    for trend in analysis["trends"]:
        lines.append(f"Trend {trend['column']}: {trend['direction']} ({trend['change_pct']}%)")
    for item in analysis["outliers"][:3]:
        lines.append(f"Outliers in {item['column']}: {item['count']}")
    return "\n".join(lines)[:limit]


def ai_insights(dataset, refresh=False):
    if dataset["ai_insights"] and not refresh:
        return dataset["ai_insights"]
    messages = [
        {"role": "system", "content": "You are a careful data analyst. Use only the statistics provided. Do not invent numbers."},
        {
            "role": "user",
            "content": "Write 5 short business insights and 3 suggested next analyses as Markdown bullet lists under the headings "
            f"'## Insights' and '## Next steps'.\n\n{profile_text(dataset)}",
        },
    ]
    text, stats = llm.complete(messages, temperature=0.2)
    dataset["ai_insights"] = {"markdown": text, "stats": stats, "privacy": llm.privacy()}
    return dataset["ai_insights"]


def _column(frame, name):
    if name in frame.columns:
        return name
    lowered = {c.lower().replace("_", " "): c for c in frame.columns}
    key = str(name).lower().replace("_", " ")
    if key in lowered:
        return lowered[key]
    close = difflib.get_close_matches(key, lowered, n=1, cutoff=0.8)
    if close:
        return lowered[close[0]]
    raise ValueError(f"Unknown column: {name}")


def _apply_filters(frame, filters):
    mask = pd.Series(True, index=frame.index)
    for rule in filters:
        column = _column(frame, rule["column"])
        operator, value = rule.get("op", "=="), rule.get("value")
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            value = float(value)
        elif pd.api.types.is_datetime64_any_dtype(series):
            text = str(value)
            if operator == "==" and re.fullmatch(r"\d{4}", text):
                mask &= series.dt.year == int(text)
                continue
            if operator == "==" and re.fullmatch(r"\d{4}-\d{2}", text):
                mask &= series.dt.to_period("M") == pd.Period(text)
                continue
            value = pd.to_datetime(value)
        else:
            series, value = series.astype(str).str.lower(), str(value).lower()
        checks = {
            "==": lambda: series == value,
            "!=": lambda: series != value,
            ">": lambda: series > value,
            ">=": lambda: series >= value,
            "<": lambda: series < value,
            "<=": lambda: series <= value,
            "contains": lambda: series.str.contains(value, regex=False),
        }
        if operator not in checks:
            raise ValueError(f"Unsupported filter operator: {operator}")
        mask &= checks[operator]()
    return frame[mask]


def run_plan(frame, plan):
    operation = plan.get("op")
    work = _apply_filters(frame, plan.get("filters") or [])
    filters_text = "; ".join(f"{f['column']} {f.get('op', '==')} {f.get('value')}" for f in plan.get("filters") or [])
    suffix = f" (where {filters_text})" if filters_text else ""
    top = min(int(plan.get("top") or 10), 50)

    if operation == "aggregate":
        agg = plan.get("agg", "sum")
        if agg not in AGGREGATES:
            raise ValueError(f"Unsupported aggregate: {agg}")
        measure = _column(frame, plan["column"]) if plan.get("column") else None
        group = _column(frame, plan["group_by"]) if plan.get("group_by") else None
        if measure is None and agg != "count":
            raise ValueError("This question needs a column to calculate on.")
        if measure and agg not in {"count", "nunique"} and not pd.api.types.is_numeric_dtype(frame[measure]):
            raise ValueError(f"{measure} is not numeric, so '{agg}' cannot be calculated.")
        label = f"{agg} of {measure}" if measure else "row count"
        if group is None:
            value = len(work) if measure is None else getattr(work[measure], agg)()
            return {"title": f"{label}{suffix}", "columns": [label], "rows": [[value]], "value": value, "chart": None}
        keys, period = work[group], plan.get("period")
        if period and pd.api.types.is_datetime64_any_dtype(keys):
            keys = keys.dt.to_period(period).astype(str)
        grouped = work.groupby(keys)
        series = grouped.size() if measure is None else grouped[measure].agg(agg)
        if period:
            series = series.sort_index().tail(60)
        else:
            series = series.sort_values(ascending=plan.get("sort") == "asc").head(top)
        rows = [[str(k), v] for k, v in series.items()]
        chart = {
            "id": "answer-chart",
            "title": f"{label} by {group}{suffix}",
            "kind": "line" if period else "bar",
            "horizontal": not period,
            "labels": [r[0] for r in rows],
            "series": [{"label": label, "data": [r[1] for r in rows]}],
        }
        return {"title": f"{label} by {group}{suffix}", "columns": [group, label], "rows": rows, "value": None, "chart": chart}

    if operation == "top_rows":
        column = _column(frame, plan["sort_by"])
        ordered = work.sort_values(column, ascending=bool(plan.get("ascending"))).head(min(top, 20))
        shown = ordered.iloc[:, :10]
        rows = shown.astype(object).where(shown.notna(), None).values.tolist()
        return {"title": f"Rows with {'lowest' if plan.get('ascending') else 'highest'} {column}{suffix}", "columns": shown.columns.tolist(), "rows": rows, "value": None, "chart": None}

    if operation == "value_counts":
        column = _column(frame, plan["column"])
        counts = work[column].value_counts().head(top)
        rows = [[str(k), int(v)] for k, v in counts.items()]
        chart = {"id": "answer-chart", "title": f"Most common {column}{suffix}", "kind": "bar", "horizontal": True, "labels": [r[0] for r in rows], "series": [{"label": "Rows", "data": [r[1] for r in rows]}]}
        return {"title": f"Most common {column}{suffix}", "columns": [column, "Rows"], "rows": rows, "value": None, "chart": chart}

    if operation == "correlation":
        names = [_column(frame, c) for c in plan.get("columns", [])[:2]]
        if len(names) != 2:
            raise ValueError("Correlation needs two numeric columns.")
        value = work[names[0]].corr(work[names[1]])
        return {"title": f"Correlation of {names[0]} and {names[1]}{suffix}", "columns": ["correlation"], "rows": [[value]], "value": value, "chart": None}

    if operation == "describe":
        column = _column(frame, plan["column"])
        described = work[column].describe()
        rows = [[str(k), v if not isinstance(v, pd.Timestamp) else str(v)] for k, v in described.items()]
        return {"title": f"Summary of {column}{suffix}", "columns": ["statistic", column], "rows": rows, "value": None, "chart": None}

    raise ValueError("This question could not be turned into a calculation.")


def _llm_plan(frame, question):
    lines = []
    for column in frame.columns[:40]:
        series = frame[column]
        kind = _kind(series, len(frame))
        if kind == "numeric":
            detail = f"numeric, min {series.min():.4g}, max {series.max():.4g}"
        elif kind == "datetime":
            detail = f"date, {series.min().date()} to {series.max().date()}"
        else:
            detail = f"{kind}, e.g. " + ", ".join(map(str, series.dropna().unique()[:5]))
        lines.append(f"- {column}: {detail}")
    prompt = (
        "Turn the question into a JSON query plan for a table. Reply with JSON only.\n"
        "Columns:\n" + "\n".join(lines) + "\n\nPlan formats:\n"
        '{"op":"aggregate","agg":"sum|mean|median|min|max|count|nunique","column":"<column or null for row count>",'
        '"group_by":"<column or null>","period":"D|W|M|Y or null (only when grouping by a date column)",'
        '"filters":[{"column":"<column>","op":"==|!=|>|>=|<|<=|contains","value":"<value>"}],"sort":"desc|asc","top":10}\n'
        '{"op":"top_rows","sort_by":"<column>","ascending":false,"top":5,"filters":[]}\n'
        '{"op":"value_counts","column":"<column>","top":10}\n'
        '{"op":"correlation","columns":["<a>","<b>"]}\n'
        '{"op":"describe","column":"<column>"}\n'
        '{"op":"none"} when the table cannot answer the question.\n\n'
        f"Question: {question}"
    )
    text, _ = llm.complete([{"role": "user", "content": prompt}], json_mode=True, max_tokens=250, temperature=0)
    match = re.search(r"\{.*\}", text, re.S)
    plan = json.loads(match.group(0)) if match else {}
    return None if plan.get("op") in (None, "none") else plan


def _rule_plan(frame, question):
    text = re.sub(r"[?,!]", " ", question.lower().replace("_", " "))
    names = {c: c.lower().replace("_", " ") for c in frame.columns}
    found = []
    for column, name in names.items():
        match = re.search(rf"\b{re.escape(name)}s?\b", text)
        if match:
            found.append((match.start(), column))
    mentioned = [c for _, c in sorted(found)]
    numeric = [c for c in mentioned if pd.api.types.is_numeric_dtype(frame[c])]
    dates = [c for c in frame.columns if pd.api.types.is_datetime64_any_dtype(frame[c])]
    others = [c for c in mentioned if c not in numeric]

    if re.search(r"correlat|relationship", text) and len(numeric) >= 2:
        return {"op": "correlation", "columns": numeric[:2]}

    agg = None
    for word, name in [
        (r"average|mean|avg", "mean"),
        (r"median", "median"),
        (r"total|sum", "sum"),
        (r"minimum|lowest|least|smallest|min\b", "min"),
        (r"maximum|highest|largest|biggest|most|max\b", "max"),
        (r"unique|distinct", "nunique"),
        (r"how many|count|number of", "count"),
    ]:
        if re.search(word, text):
            agg = name
            break
    top_match = re.search(r"top (\d+)", text)
    top = int(top_match.group(1)) if top_match else 10
    ascending = bool(re.search(r"lowest|least|smallest|bottom|minimum", text))

    filters = []
    for column in frame.columns:
        if column in numeric or column in dates or frame[column].nunique() > 50:
            continue
        for value in frame[column].dropna().astype(str).unique():
            if len(value) > 1 and re.search(rf"\b{re.escape(value.lower())}\b", text):
                filters.append({"column": column, "op": "==", "value": value})
                break
    year = re.search(r"\b(19|20)\d{2}\b", text)
    if year and dates:
        filters.append({"column": dates[0], "op": "==", "value": year.group(0)})

    period = None
    if dates and re.search(r"monthly|per month|by month|each month|month wise", text):
        period = "M"
    elif dates and re.search(r"yearly|per year|by year|annual", text):
        period = "Y"
    elif dates and re.search(r"daily|per day|by day", text):
        period = "D"

    group = dates[0] if period else next((c for c in others if c not in dates and not any(f["column"] == c for f in filters)), None)
    if group is None and others and not period:
        group = next((c for c in others if not any(f["column"] == c for f in filters)), None)
    measure = numeric[0] if numeric else None

    if re.search(r"how many (rows|records|entries)|number of (rows|records)", text):
        return {"op": "aggregate", "agg": "count", "column": None, "group_by": None, "filters": filters}
    if measure and re.search(r"top \d+|show .* rows|which rows|list .* rows", text) and not group:
        return {"op": "top_rows", "sort_by": measure, "ascending": ascending, "top": top, "filters": filters}
    if measure and (group or agg):
        return {
            "op": "aggregate",
            "agg": agg or "sum",
            "column": measure,
            "group_by": group,
            "period": period,
            "filters": filters,
            "sort": "asc" if ascending else "desc",
            "top": top,
        }
    if group and agg == "count" or (group and re.search(r"distribution|breakdown|most common|frequency", text)):
        return {"op": "value_counts", "column": group, "top": top, "filters": filters}
    if measure and re.search(r"describe|summary|statistics|stats|distribution", text):
        return {"op": "describe", "column": measure, "filters": filters}
    return None


def query(dataset, question):
    frame = dataset["df"]
    plan = None
    if llm.status()["ready"]:
        try:
            plan = _llm_plan(frame, question)
            if plan:
                run_plan(frame, plan)
        except (llm.LLMError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            plan = None
    if plan is None:
        plan = _rule_plan(frame, question)
    if plan is None:
        return None, None
    try:
        return plan, to_native(run_plan(frame, plan))
    except (ValueError, KeyError, TypeError) as exc:
        return plan, {"error": str(exc)}


def format_value(value):
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return f"{value:,.2f}".rstrip("0").rstrip(".") if abs(value) < 1e15 else str(value)
    return str(value)


def describe_result(result):
    if "error" in result:
        return f"I could not calculate that: {result['error']}"
    if result.get("value") is not None:
        return f"**{result['title']}: {format_value(result['value'])}**"
    rows = result["rows"]
    if not rows:
        return f"{result['title']}: no matching rows."
    text = f"**{result['title']}**"
    if len(result["columns"]) == 2 and len(rows) > 1:
        text += f"\n\nFirst: {rows[0][0]} ({format_value(rows[0][1])}). Last: {rows[-1][0]} ({format_value(rows[-1][1])})."
    return text
