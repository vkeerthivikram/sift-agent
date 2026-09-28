"""Dataset Q&A: deterministic context retrieval plus intent answers, optional LLM.

``build_context`` picks the profile facts relevant to a question (matched
columns + keyword-routed sections); ``answer_question`` answers either with a
deterministic regex intent engine (offline) or with one grounded LLM call.
Standalone module — no dependency on ``graph`` or ``investigate``.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

import pandas as pd

from .analysis import column_summary

logger = logging.getLogger("sift_agent.qa")

MAX_INTENT_COLUMNS = 3
MAX_UNIQUE_VALUES = 8
TRUNCATION_MARKER = "[context truncated]"
OFFLINE_PREFACE = (
    "Offline mode: this question is not covered by the offline intent engine; "
    "here are the facts retrieved from the dataset profile."
)

QA_SYSTEM_PROMPT = (
    "You are a grounded data analyst answering questions about a dataset. "
    "Ground every statement in the context provided; "
    "never fabricate values or column names."
)

QA_PROMPT = """Answer the user's question about the dataset using only the context below.

<context>
{context}
</context>

<question>
{question}
</question>

Rules:
- Use only numbers that appear in the context; never estimate or invent values.
- Cite the column names you relied on (in backticks).
- If the context does not contain the answer, say so plainly.
"""

_MISSING_RE = re.compile(r"\b(missing|miss|null|nulls|n/?a|blank|empty)\b")
_HOW_MANY_RE = re.compile(r"\b(how many|how much|number of|count|total)\b")
_ROW_RE = re.compile(r"\b(rows?|records?|entries|observations)\b")
_MAX_RE = re.compile(r"\b(max|maximum|largest|highest|biggest)\b")
_MIN_RE = re.compile(r"\b(min|minimum|smallest|lowest)\b")
_AVG_RE = re.compile(r"\b(avg|average|mean)\b")
_UNIQUE_RE = re.compile(r"\b(unique|distinct|categories)\b")
_CORR_RE = re.compile(r"\b(correlat\w*|relationship|related|associat\w*)\b")
_TYPE_RE = re.compile(r"\b(data\s*types?|dtypes?|types?)\b")
_DUP_RE = re.compile(r"\bduplicat\w*\b")
_OUTLIER_SECTION_RE = re.compile(r"\b(outliers?|odd|extremes?|anomal\w*)\b")
_QUALITY_SECTION_RE = re.compile(r"\b(quality|dirty|issues?|messy)\b")

# Question words that must never partially match a multi-word column name.
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "am",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "can",
        "could",
        "should",
        "would",
        "you",
        "your",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "there",
        "their",
        "what",
        "whats",
        "which",
        "who",
        "how",
        "why",
        "when",
        "where",
        "many",
        "much",
        "tell",
        "show",
        "give",
        "list",
        "find",
        "get",
        "me",
        "please",
        "any",
        "all",
        "some",
        "none",
        "than",
        "then",
        "so",
        "if",
        "but",
        "about",
        "into",
        "per",
        "between",
        "over",
        "under",
        "data",
        "dataset",
        "table",
        "column",
        "columns",
        "row",
        "rows",
        "record",
        "records",
        "value",
        "values",
    }
)


def _words(text: str) -> list[str]:
    """Lowercased alphanumeric words of a text (punctuation dropped)."""
    return [w for w in re.split(r"[^a-z0-9]+", text.lower()) if w]


def _canon_word(w: str) -> str:
    """Crude singular form so 'days'/'day' match (both sides canonicalized)."""
    if len(w) > 3 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 2 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def _match_columns(question: str, columns: list[str]) -> list[str]:
    """Columns mentioned in the question, in order of first appearance.

    A column matches when its full name appears in the question (case and
    separator insensitive, plural tolerant), or — for multi-word column names —
    when one distinctive word of the name appears.
    """
    q_text = f" {' '.join(_canon_word(w) for w in _words(question))} "
    scored: list[tuple[int, int, str]] = []
    for col in columns:
        tokens = [_canon_word(w) for w in _words(str(col))]
        if not tokens:
            continue
        phrase = " ".join(tokens)
        idx = q_text.find(f" {phrase} ")
        if idx >= 0:
            scored.append((q_text[:idx].count(" "), -len(phrase), str(col)))
            continue
        if len(tokens) < 2:
            continue
        best: int | None = None
        for tok in tokens:
            if len(tok) < 3 or tok in _STOPWORDS:
                continue
            i = q_text.find(f" {tok} ")
            if i >= 0:
                pos = q_text[:i].count(" ")
                best = pos if best is None else min(best, pos)
        if best is not None:
            scored.append((best, -len(phrase), str(col)))
    scored.sort()
    return [col for _, _, col in scored]


def _column_universe(df: pd.DataFrame, profile: dict) -> list[str]:
    return list(profile.get("columns") or {}) or list(df.columns)


def _fmt_num(v) -> str:
    """Deterministic, human-friendly number rendering."""
    f = float(v)
    if not math.isfinite(f):
        return "n/a"
    if f == int(f) and abs(f) < 1e15:
        return f"{int(f):,}"
    return f"{f:,.4f}".rstrip("0").rstrip(".")


def _fmt_pct(v) -> str:
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


def _fmt_count(v) -> str:
    return f"{v:,}" if isinstance(v, int) else str(v)


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _overview_line(df: pd.DataFrame, profile: dict) -> str:
    return (
        f"Dataset: {_fmt_count(profile.get('n_rows', len(df)))} rows x "
        f"{_fmt_count(profile.get('n_columns', df.shape[1]))} columns "
        f"({_fmt_count(profile.get('n_numeric_columns', 0))} numeric, "
        f"{_fmt_count(profile.get('n_categorical_columns', 0))} categorical, "
        f"{_fmt_count(profile.get('duplicate_rows', 0))} duplicate rows)."
    )


def _column_lines(col: str, info: dict) -> list[str]:
    return [
        f"Column `{col}` — {info.get('type', 'unknown')} "
        f"(dtype {info.get('dtype', 'unknown')})",
        f"  missing: {_fmt_count(info.get('n_missing', 0))} "
        f"({_fmt_pct(info.get('pct_missing', 0))}%), "
        f"unique values: {_fmt_count(info.get('n_unique', 'n/a'))}",
        f"  {column_summary(info)}",
    ]


def _section_blocks(question: str, profile: dict) -> list[tuple[str, list[str]]]:
    """Whole-profile sections routed by question keywords."""
    q = question.lower()
    blocks: list[tuple[str, list[str]]] = []

    if _MISSING_RE.search(q):
        rows = [
            f"- {m['column']}: {_fmt_count(m['n_missing'])} missing "
            f"({_fmt_pct(m['pct_missing'])}%)"
            for m in (profile.get("top_missing") or [])
        ]
        blocks.append(
            (
                "section: missing",
                ["Columns with missing values:"] + (rows or ["- none"]),
            )
        )

    if _CORR_RE.search(q):
        pearson = profile.get("top_correlated_pairs") or []
        spearman = profile.get("spearman_pairs") or []
        lines = ["Top Pearson correlations:"] + (
            [
                f"- {p['col_a']} vs {p['col_b']}: r={p['pearson_r']:+.3f}"
                for p in pearson
            ]
            or ["- none above threshold"]
        )
        lines.append("Top Spearman correlations:")
        lines += [
            f"- {p['col_a']} vs {p['col_b']}: spearman={p['spearman_r']:+.3f}"
            for p in spearman
        ] or ["- none above threshold"]
        blocks.append(("section: correlations", lines))

    if _OUTLIER_SECTION_RE.search(q):
        outs = profile.get("outlier_columns") or []
        rows = [
            f"- {o['column']}: {_fmt_count(o['count'])} outliers "
            f"({_fmt_pct(o['pct'])}%), "
            f"e.g. {', '.join(str(e) for e in o.get('examples') or []) or 'n/a'}"
            for o in outs
        ]
        blocks.append(
            (
                "section: outliers",
                ["Columns with IQR outliers:"] + (rows or ["- none"]),
            )
        )

    if _DUP_RE.search(q):
        blocks.append(
            (
                "section: duplicates",
                [f"Duplicate rows: {_fmt_count(profile.get('duplicate_rows', 0))}"],
            )
        )

    if _QUALITY_SECTION_RE.search(q):
        issues = profile.get("data_quality") or []
        rows = [f"- {i['column']} ({i['check']}): {i['detail']}" for i in issues]
        blocks.append(
            (
                "section: data_quality",
                ["Data quality issues:"] + (rows or ["- none detected"]),
            )
        )

    return blocks


def build_context(
    question: str,
    df: pd.DataFrame,
    profile: dict,
    max_chars: int = 6000,
) -> tuple[str, list[str]]:
    """Assemble the profile context relevant to ``question``.

    Returns ``(context_text, sources)`` where sources are labels like
    ``"column: amount_usd"`` / ``"section: correlations"``. The dataset shape
    line always comes first; the context is hard-capped at ``max_chars``
    characters and truncated only on block boundaries, never mid-line.
    """
    max_chars = max(max_chars, len(TRUNCATION_MARKER) + 2)
    cols = profile.get("columns") or {}
    matched = _match_columns(question, _column_universe(df, profile))

    blocks: list[tuple[str, list[str]]] = [
        ("section: overview", [_overview_line(df, profile)])
    ]
    for col in matched:
        blocks.append((f"column: {col}", _column_lines(col, cols.get(col) or {})))
    blocks += _section_blocks(question, profile)

    chunks: list[str] = []
    sources: list[str] = []
    used = 0
    truncated = False
    for source, lines in blocks:
        chunk = "\n".join(lines)
        add = len(chunk) + (1 if chunks else 0)
        if used + add + len(TRUNCATION_MARKER) > max_chars:
            truncated = True
            break
        chunks.append(chunk)
        sources.append(source)
        used += add

    if not chunks:  # degenerate cap: keep at least a slice of the shape line
        room = max_chars - len(TRUNCATION_MARKER) - 1
        chunks = [blocks[0][1][0][:room]]
        sources = [blocks[0][0]]
        truncated = True
    if truncated:
        chunks.append(TRUNCATION_MARKER)
    return "\n".join(chunks), sources


def _ans_missing(
    q: str, df: pd.DataFrame, profile: dict, matched: list[str]
) -> tuple[str, list[str]] | None:
    if not (_MISSING_RE.search(q) and matched):
        return None
    n_rows = len(df)
    used = matched[:MAX_INTENT_COLUMNS]
    parts = []
    for col in used:
        n = int(df[col].isna().sum())
        pct = round(n / n_rows * 100, 2) if n_rows else 0.0
        parts.append(f"`{col}`: {n:,} of {n_rows:,} values missing ({_fmt_pct(pct)}%).")
    return " ".join(parts), [f"column: {c}" for c in used]


def _ans_rows(
    q: str, df: pd.DataFrame, profile: dict, matched: list[str]
) -> tuple[str, list[str]] | None:
    if not (_ROW_RE.search(q) and _HOW_MANY_RE.search(q) and not matched):
        return None
    text = f"The dataset has {len(df):,} rows x {df.shape[1]} columns."
    sources = ["section: overview"]
    if _DUP_RE.search(q):
        dup = int(df.duplicated().sum())
        text += f" Duplicates: {dup:,} of {len(df):,} rows."
        sources.append("section: duplicates")
    if _MISSING_RE.search(q):
        k = int(df.isna().any(axis=1).sum())
        text += f" {k:,} rows contain at least one missing value."
    return text, sources


def _ans_stat(
    q: str, df: pd.DataFrame, profile: dict, matched: list[str]
) -> tuple[str, list[str]] | None:
    wanted: list[tuple[str, str]] = []
    if _MAX_RE.search(q):
        wanted.append(("maximum", "max"))
    if _MIN_RE.search(q):
        wanted.append(("minimum", "min"))
    if _AVG_RE.search(q):
        wanted.append(("average", "mean"))
    if not wanted:
        return None
    numeric = [c for c in matched if _is_numeric(df[c])][:MAX_INTENT_COLUMNS]
    lines = []
    for col in numeric:
        s = df[col].dropna()
        if s.empty:
            continue
        values = {"max": s.max(), "min": s.min(), "mean": s.mean()}
        stats = ", ".join(f"{label} {_fmt_num(values[key])}" for label, key in wanted)
        lines.append(f"`{col}`: {stats} ({len(s):,} non-null values).")
    if not lines:
        return None
    return "\n".join(lines), [f"column: {c}" for c in numeric]


def _ans_unique(
    q: str, df: pd.DataFrame, profile: dict, matched: list[str]
) -> tuple[str, list[str]] | None:
    if not (_UNIQUE_RE.search(q) and matched):
        return None
    col = matched[0]
    s = df[col].dropna()
    n_unique = int(s.nunique())
    counts = sorted(
        ((str(k), int(v)) for k, v in s.value_counts().items()),
        key=lambda kv: (-kv[1], kv[0]),
    )
    top = [f"{k} ({v:,})" for k, v in counts[:MAX_UNIQUE_VALUES]]
    if not top:
        return f"`{col}` has no non-null values.", [f"column: {col}"]
    head = f"`{col}` has {n_unique:,} unique values"
    suffix = " (top 8 shown)" if n_unique > MAX_UNIQUE_VALUES else ""
    return f"{head}; most frequent: {', '.join(top)}{suffix}.", [f"column: {col}"]


def _profile_pair(profile: dict, a: str, b: str) -> tuple[str, float] | None:
    """Look up a column pair in the profile's precomputed pair lists."""
    want = frozenset((a, b))
    sections = (
        ("pearson_r", profile.get("top_correlated_pairs") or []),
        ("spearman_r", profile.get("spearman_pairs") or []),
        ("cramers_v", profile.get("cramers_v_pairs") or []),
    )
    for key, pairs in sections:
        for p in pairs:
            if frozenset((p["col_a"], p["col_b"])) == want:
                return key, float(p[key])
    return None


def _ans_correlation(
    q: str, df: pd.DataFrame, profile: dict, matched: list[str]
) -> tuple[str, list[str]] | None:
    if not (_CORR_RE.search(q) and len(matched) >= 2):
        return None
    a, b = matched[0], matched[1]
    sources = [f"column: {a}", f"column: {b}"]
    if _is_numeric(df[a]) and _is_numeric(df[b]):
        n = int((df[a].notna() & df[b].notna()).sum())
        if n < 2:
            return (
                f"Correlation between `{a}` and `{b}` is undefined "
                f"(only {n} complete pair(s)).",
                sources,
            )
        r = df[a].corr(df[b])
        if r is None or pd.isna(r):
            return (
                f"Pearson r between `{a}` and `{b}` is undefined "
                "(a column has zero variance).",
                sources,
            )
        return (
            f"Pearson r between `{a}` and `{b}` = {r:+.3f} ({n:,} complete pairs).",
            sources,
        )
    hit = _profile_pair(profile, a, b)
    if hit is None:
        return None
    key, value = hit
    return (
        f"`{a}` vs `{b}`: {key} = {value:+.3f} (from the dataset profile).",
        sources,
    )


def _ans_dtype(
    q: str, df: pd.DataFrame, profile: dict, matched: list[str]
) -> tuple[str, list[str]] | None:
    if not (_TYPE_RE.search(q) and matched):
        return None
    col = matched[0]
    ptype = (profile.get("columns") or {}).get(col, {}).get("type", "unknown")
    return (
        f"`{col}` has dtype {df[col].dtype} (profiled as {ptype}).",
        [f"column: {col}"],
    )


def _ans_duplicates(
    q: str, df: pd.DataFrame, profile: dict, matched: list[str]
) -> tuple[str, list[str]] | None:
    if not _DUP_RE.search(q):
        return None
    dup = int(df.duplicated().sum())
    return (
        f"Duplicate rows: {dup:,} of {len(df):,} rows.",
        ["section: duplicates"],
    )


def _intent_answer(
    question: str, df: pd.DataFrame, profile: dict
) -> tuple[str, list[str]] | None:
    """Deterministic intent engine; returns None when nothing matches."""
    q = question.lower()
    matched = _match_columns(question, _column_universe(df, profile))
    handlers = (
        _ans_missing,
        _ans_rows,
        _ans_stat,
        _ans_unique,
        _ans_correlation,
        _ans_dtype,
        _ans_duplicates,
    )
    for handler in handlers:
        result = handler(q, df, profile, matched)
        if result is not None:
            return result
    return None


def _response_text(response: Any) -> str:
    """Extract non-empty text from a chat response or raise for unusable output."""
    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    text = str(content).strip()
    if not text:
        raise ValueError("LLM returned an empty response")
    return text


def _offline_result(
    question: str,
    df: pd.DataFrame,
    profile: dict,
    context: str,
    context_sources: list[str],
) -> dict:
    hit = _intent_answer(question, df, profile)
    if hit is not None:
        answer, sources = hit
        return {"answer": answer, "offline": True, "sources": sources}
    return {
        "answer": f"{OFFLINE_PREFACE}\n{context}",
        "offline": True,
        "sources": context_sources,
    }


def answer_question(
    question: str,
    df: pd.DataFrame,
    profile: dict,
    llm: Any | None = None,
) -> dict:
    """Answer a natural-language question about a profiled DataFrame.

    Returns ``{"answer": str, "offline": bool, "sources": list[str]}``. With
    ``llm=None`` (or when the LLM call fails) a deterministic regex intent
    engine answers from the data; unsupported questions get a best-effort
    answer listing the retrieved context facts, never fabricated numbers.
    A failed LLM call additionally sets a ``"warning"`` key naming the error.
    """
    context, sources = build_context(question, df, profile)
    if llm is None:
        return _offline_result(question, df, profile, context, sources)
    try:
        resp = llm.invoke(
            [
                ("system", QA_SYSTEM_PROMPT),
                ("human", QA_PROMPT.format(context=context, question=question)),
            ]
        )
        return {"answer": _response_text(resp), "offline": False, "sources": sources}
    except Exception as exc:
        logger.warning("question LLM call failed; answering offline instead: %s", exc)
        result = _offline_result(question, df, profile, context, sources)
        result["offline"] = True
        result["warning"] = (
            f"question LLM call failed ({exc.__class__.__name__}); "
            "answered with the offline engine instead"
        )
        return result
