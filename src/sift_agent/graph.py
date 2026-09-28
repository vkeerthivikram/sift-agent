"""LangGraph state machine wiring the auto-EDA pipeline together.

Flow: load_data -> statistical_analysis -> generate_visualizations
      -> extract_insights -> recommendations -> agentic_investigation
      -> anomaly_drilldown -> build_report
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langgraph.graph import END, START, StateGraph

from .analysis import DATA_QUALITY_ADVICE, profile_dataframe
from .html_report import write_html_report
from .investigate import TOOL_SPECS, run_tool
from .loader import LoadError, load_table
from .report import write_report
from .scoring import health_score
from .state import EDAState
from .visualize import generate_charts

logger = logging.getLogger("sift_agent.graph")

MAX_INPUT_BYTES = 512 * 1024 * 1024
MAX_ROWS = 1_000_000
LLM_MAX_ATTEMPTS = 3
LLM_RETRY_BACKOFF_S = 2.0
MAX_AGENT_ROUNDS = 4  # tool-loop rounds before the agent must write findings
MAX_TOOL_CALLS_PER_ROUND = 2
MAX_TOOL_CALLS_TOTAL = 6
TOOL_RESULT_CHARS = 2000  # ToolMessage content cap fed back to the model
DIGEST_CHARS = 3500  # compact profile digest cap for the agent seed message
ANOMALY_PAYLOAD_CHARS = 4000  # drill-down JSON cap for the narrative prompt

SYSTEM_PROMPT = (
    "You are a senior data scientist performing exploratory data analysis (EDA). "
    "Ground every statement in the numbers provided; never fabricate values or column names."
)

INSIGHT_PROMPT = """Below is the statistical profile of a dataset, plus the charts that were generated for it.

<profile>
{profile_json}
</profile>

<charts>
{charts}
</charts>

Produce the "Key Insights" body of an EDA report as GitHub markdown:
- Open with one sentence describing the dataset (rows, columns, mix of column types).
- Then 5-10 concise bullet points. Each bullet must cite concrete column names and numbers, covering data quality (missing values, duplicates), distributions and skew, outliers, correlations between numeric columns, and categorical cardinality or imbalance where relevant.
- Do NOT add a section heading of your own; output bullets only.
"""

RECOMMENDATION_PROMPT = """You are continuing an EDA report. Based on the profile and insights below, produce the "Recommendations" body as GitHub markdown.

<insights>
{insights}
</insights>

<profile>
{profile_json}
</profile>

Write prioritized, concrete next steps in three short groups:
1. **Data cleaning** — exact columns and strategies (imputation, dedup, dropping).
2. **Feature engineering** — transformations (e.g. log for skewed columns), encodings, new features.
3. **Analysis / modeling** — what to try next and what to validate.

Cite column names. Do NOT add a section heading of your own.
"""

AGENT_SYSTEM_PROMPT = (
    "You are a data investigator with tools. Call the provided tools to examine "
    "the dataset, ground every finding in tool output, and cite column names. "
    "If a tool returns an error, adjust the arguments and try again. "
    "After investigating, write your findings as markdown bullets."
)

INVESTIGATION_PROMPT = """A statistical profile digest of the dataset follows.

<digest>
{digest}
</digest>

Investigate this dataset with the available tools (a handful of calls is enough).
Prioritize the outliers, relationships, and group differences the digest hints at.
Every finding must cite the column names and numbers returned by the tools.
Finish with your findings as GitHub markdown bullets (no heading)."""

ANOMALY_PROMPT = """Below is the tool output from an anomaly drill-down in an EDA report.

<drilldown>
{payload}
</drilldown>

Write 2-3 sentences of prose explaining what drives the outliers of the column,
grounded strictly in the JSON above; cite column and category names, and never
invent numbers. No heading, no bullet list."""


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(content)


def _response_text(response: Any) -> str:
    """Extract non-empty text from a chat response or raise for unusable output."""
    text = _content_to_text(response.content).strip()
    if not text:
        raise ValueError("LLM returned an empty response")
    return text


def _profile_json(profile: dict, limit: int = 30_000) -> str:
    text = json.dumps(profile, indent=2, default=str)
    if len(text) > limit:
        text = text[:limit] + "\n... (profile truncated for length)"
    return text


def _charts_text(charts: list[dict]) -> str:
    if not charts:
        return "No charts were generated."
    return "\n".join(f"- {c['title']}: {c['caption']}" for c in charts)


def _infer_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert object columns whose values look like ISO dates/timestamps to datetime."""
    for col in df.columns:
        if not (df[col].dtype == object or pd.api.types.is_string_dtype(df[col])):
            continue
        sample = df[col].dropna().head(20).astype(str)
        if sample.empty:
            continue
        try:
            pd.to_datetime(sample, errors="raise", format="ISO8601")
        except (ValueError, TypeError):
            continue
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def heuristic_insights(profile: dict) -> str:
    """Deterministic insights used when no LLM is configured (provider 'none')."""
    cols = profile["columns"]
    n_rows = max(profile["n_rows"], 1)
    bullets: list[str] = []

    bullets.append(
        f"Dataset has **{profile['n_rows']:,} rows x {profile['n_columns']} columns** "
        f"({profile['n_numeric_columns']} numeric, {profile['n_categorical_columns']} categorical), "
        f"{profile['duplicate_rows']:,} duplicate rows, {profile['memory_mb']} MB in memory."
    )

    missing = profile.get("top_missing") or []
    if missing:
        worst = ", ".join(f"`{m['column']}` ({m['pct_missing']}%)" for m in missing[:5])
        bullets.append(f"Missing values in {len(missing)} column(s); worst: {worst}.")
    else:
        bullets.append("No missing values detected.")

    if profile.get("constant_columns"):
        bullets.append(
            "Constant (zero-variance) columns: "
            + ", ".join(f"`{c}`" for c in profile["constant_columns"])
            + "."
        )

    skewed = sorted(
        (
            (name, c["skew"])
            for name, c in cols.items()
            if c.get("type") == "numeric"
            and c.get("skew") is not None
            and abs(c["skew"]) >= 1
        ),
        key=lambda t: abs(t[1]),
        reverse=True,
    )
    if skewed:
        top = ", ".join(f"`{n}` (skew {v:+.2f})" for n, v in skewed[:5])
        bullets.append(f"Strongly skewed numeric columns: {top}.")

    outliers = sorted(
        (
            (name, c["n_outliers_iqr"])
            for name, c in cols.items()
            if c.get("type") == "numeric" and c.get("n_outliers_iqr", 0) > 0
        ),
        key=lambda t: t[1],
        reverse=True,
    )
    if outliers:
        top = ", ".join(
            f"`{n}` ({cnt} rows, {cnt / n_rows:.1%})" for n, cnt in outliers[:5]
        )
        bullets.append(f"Columns with the most IQR outliers: {top}.")

    strong = [
        p
        for p in (profile.get("top_correlated_pairs") or [])
        if abs(p["pearson_r"]) >= 0.5
    ]
    if strong:
        top = "; ".join(
            f"`{p['col_a']}` vs `{p['col_b']}` (r={p['pearson_r']:+.2f})"
            for p in strong[:5]
        )
        bullets.append(f"Notable numeric correlations: {top}.")

    high_card = [
        (n, c["n_unique"])
        for n, c in cols.items()
        if c.get("type") == "categorical" and c["n_unique"] > 50
    ]
    if high_card:
        top = ", ".join(f"`{n}` ({u} unique)" for n, u in high_card[:5])
        bullets.append(
            f"High-cardinality categorical columns (likely IDs/free text): {top}."
        )

    imbalanced = []
    for name, c in cols.items():
        if (
            c.get("type") in ("categorical", "boolean")
            and c.get("value_counts")
            and c["n_unique"] > 1
        ):
            top = c["value_counts"][0]
            if top["count"] / n_rows > 0.8:
                imbalanced.append((name, top))
    if imbalanced:
        top = ", ".join(
            f"`{n}` ('{t['value']}' at {t['count'] / n_rows:.0%})"
            for n, t in imbalanced[:5]
        )
        bullets.append(f"Heavily imbalanced categories: {top}.")

    id_like = profile.get("id_like_columns") or []
    if id_like:
        top = ", ".join(f"`{i['column']}` ({i['reason']})" for i in id_like[:5])
        bullets.append(f"Likely ID columns, not signal: {top}.")

    quality = profile.get("data_quality") or []
    if quality:
        top = ", ".join(
            f"`{q['column']}` ({q['check'].replace('_', ' ')})" for q in quality[:5]
        )
        bullets.append(f"Data-quality issues: {top}.")

    pii = profile.get("pii_columns") or []
    if pii:
        top = ", ".join(f"`{p['column']}` ({p['label']})" for p in pii[:5])
        bullets.append(f"Possible PII detected — redact before sharing: {top}.")

    if (profile.get("co_missing") or {}).get("fully_empty_rows"):
        bullets.append(
            f"{profile['co_missing']['fully_empty_rows']:,} rows are entirely empty."
        )

    datetime_spans = [
        (name, c)
        for name, c in cols.items()
        if c.get("type") == "datetime" and c.get("span_days")
    ]
    if datetime_spans:

        def _short(v: str | None) -> str:
            return (v or "").replace(" 00:00:00", "")

        top = ", ".join(
            f"`{n}` spans {_short(c['min'])} → {_short(c['max'])} ({c['span_days']:g} days)"
            for n, c in datetime_spans[:3]
        )
        bullets.append(f"Time coverage: {top}.")

    pearson_pairs = profile.get("top_correlated_pairs") or []
    pearson_lookup = {
        frozenset((p["col_a"], p["col_b"])): abs(p["pearson_r"]) for p in pearson_pairs
    }
    non_linear = [
        p
        for p in (profile.get("spearman_pairs") or [])
        if abs(p["spearman_r"]) >= 0.5
        and pearson_lookup.get(frozenset((p["col_a"], p["col_b"])), 0.0) < 0.3
    ]
    if non_linear:
        top = "; ".join(
            f"`{p['col_a']}` vs `{p['col_b']}` (ρ={p['spearman_r']:+.2f})"
            for p in non_linear[:5]
        )
        bullets.append(
            f"Monotonic but non-linear relationships (Spearman ≫ Pearson): {top}."
        )

    strong_assoc = [
        p for p in (profile.get("cramers_v_pairs") or []) if p["cramers_v"] >= 0.5
    ]
    if strong_assoc:
        top = "; ".join(
            f"`{p['col_a']}` ~ `{p['col_b']}` (V={p['cramers_v']:.2f})"
            for p in strong_assoc[:5]
        )
        bullets.append(f"Strong categorical associations: {top}.")

    group_effects = [
        e for e in (profile.get("category_effects") or []) if e["spread"] >= 0.5
    ]
    if group_effects:
        parts = []
        for e in group_effects[:3]:
            means = list(e["means"].values())
            lo, hi = min(means), max(means)
            parts.append(
                f"`{e['numeric']}` varies most by `{e['category']}` ({lo:g} → {hi:g})"
            )
        bullets.append("; ".join(parts) + ".")

    return "\n".join(f"- {b}" for b in bullets)


def heuristic_recommendations(profile: dict) -> str:
    """Deterministic recommendations used when no LLM is configured (provider 'none')."""
    cols = profile["columns"]
    recs: list[str] = []

    if profile["duplicate_rows"]:
        recs.append(
            f"Remove the {profile['duplicate_rows']:,} duplicate rows (`df.drop_duplicates()`)."
        )

    pii = profile.get("pii_columns") or []
    if pii:
        top = ", ".join(f"`{p['column']}` ({p['label']})" for p in pii[:5])
        recs.append(
            f"Redact or hash likely PII columns before sharing this data or report: {top}."
        )

    heavy_missing = [
        m["column"] for m in (profile.get("top_missing") or []) if m["pct_missing"] > 30
    ]
    if heavy_missing:
        recs.append(
            "Consider dropping columns with >30% missing: "
            + ", ".join(f"`{c}`" for c in heavy_missing)
            + "."
        )

    skewed = [
        n
        for n, c in cols.items()
        if c.get("type") == "numeric"
        and c.get("skew") is not None
        and abs(c["skew"]) >= 1
    ]
    if skewed:
        recs.append(
            "Apply log/sqrt transforms to skewed numeric columns: "
            + ", ".join(f"`{c}`" for c in skewed[:6])
            + "."
        )

    collinear = [
        p
        for p in (profile.get("top_correlated_pairs") or [])
        if abs(p["pearson_r"]) >= 0.8
    ]
    if collinear:
        detail = "; ".join(
            f"`{p['col_a']}` ~ `{p['col_b']}` (r={p['pearson_r']:+.2f})"
            for p in collinear[:4]
        )
        recs.append(
            f"Watch multicollinearity ({detail}) — drop or combine one column of each pair."
        )

    if profile.get("constant_columns"):
        recs.append(
            "Drop constant columns before modeling: "
            + ", ".join(f"`{c}`" for c in profile["constant_columns"])
            + "."
        )

    co = profile.get("co_missing") or {}
    if co.get("fully_empty_rows"):
        recs.append(
            f"Drop the {co['fully_empty_rows']:,} entirely empty rows "
            "(`df.dropna(how='all')`)."
        )

    seen_checks: set[str] = set()
    for issue in profile.get("data_quality") or []:
        check = issue["check"]
        if check in seen_checks:
            continue
        seen_checks.add(check)
        advice = DATA_QUALITY_ADVICE.get(check)
        if advice:
            recs.append(f"{advice} (`{issue['column']}`: {issue['detail']}).")

    id_like = [i["column"] for i in (profile.get("id_like_columns") or [])]
    if id_like:
        recs.append(
            "Exclude ID-like columns from features: "
            + ", ".join(f"`{c}`" for c in id_like[:4])
            + "."
        )

    high_card = [
        n
        for n, c in cols.items()
        if c.get("type") == "categorical" and c["n_unique"] > 50
    ]
    if high_card:
        recs.append(
            "Avoid one-hot encoding high-cardinality columns ("
            + ", ".join(f"`{c}`" for c in high_card[:4])
            + "); use frequency/target encoding or drop ID-like keys."
        )

    redundant = [
        p for p in (profile.get("cramers_v_pairs") or []) if p["cramers_v"] >= 0.8
    ]
    if redundant:
        detail = "; ".join(
            f"`{p['col_a']}` ~ `{p['col_b']}` (V={p['cramers_v']:.2f})"
            for p in redundant[:4]
        )
        recs.append(
            f"Categorical pairs carry near-identical information ({detail}) — keep one."
        )

    group_effects = [
        e for e in (profile.get("category_effects") or []) if e["spread"] >= 1.5
    ]
    if group_effects:
        top = ", ".join(
            f"`{e['numeric']}` across `{e['category']}`" for e in group_effects[:3]
        )
        recs.append(
            f"Analyze at group level — group means differ widely for {top}; "
            "aggregate comparisons or interaction terms will matter."
        )

    return (
        "\n".join(f"- {r}" for r in recs)
        if recs
        else "- Dataset looks clean — proceed to hypothesis-driven analysis."
    )


def _profile_digest(profile: dict, limit: int = DIGEST_CHARS) -> str:
    """Compact one-screen digest of the profile used to seed the agent loop."""
    lines: list[str] = [
        f"shape: {profile.get('n_rows')} rows x {profile.get('n_columns')} columns "
        f"({profile.get('n_numeric_columns')} numeric, "
        f"{profile.get('n_categorical_columns')} categorical), "
        f"{profile.get('duplicate_rows')} duplicate rows"
    ]
    health = profile.get("health") or {}
    if health.get("verdict"):
        lines.append(
            f"health: {health.get('score')}/100 (grade {health.get('grade')}) — "
            f"{health['verdict']}"
        )
    missing = (profile.get("top_missing") or [])[:5]
    if missing:
        lines.append(
            "most-missing columns: "
            + "; ".join(f"`{m['column']}` {m['pct_missing']}%" for m in missing)
        )
    outliers = (profile.get("outlier_columns") or [])[:5]
    if outliers:
        lines.append(
            "outlier columns (IQR): "
            + "; ".join(f"`{o['column']}` ({o['count']} rows)" for o in outliers)
        )
    corr = [
        f"`{p['col_a']}` ~ `{p['col_b']}` r={p['pearson_r']:+.2f}"
        for p in (profile.get("top_correlated_pairs") or [])[:5]
    ] + [
        f"`{p['col_a']}` ~ `{p['col_b']}` ρ={p['spearman_r']:+.2f}"
        for p in (profile.get("spearman_pairs") or [])[:3]
    ]
    if corr:
        lines.append("correlated numeric pairs: " + "; ".join(corr))
    cramers = (profile.get("cramers_v_pairs") or [])[:3]
    if cramers:
        lines.append(
            "categorical associations: "
            + "; ".join(
                f"`{p['col_a']}` ~ `{p['col_b']}` V={p['cramers_v']:.2f}"
                for p in cramers
            )
        )
    effects = (profile.get("category_effects") or [])[:3]
    if effects:
        lines.append(
            "group differences: "
            + "; ".join(
                f"`{e['numeric']}` by `{e['category']}` (spread {e['spread']})"
                for e in effects
            )
        )
    quality = (profile.get("data_quality") or [])[:5]
    if quality:
        lines.append(
            "quality checks flagged: "
            + "; ".join(f"`{q['column']}` ({q['check']})" for q in quality)
        )
    text = "\n".join(lines)
    if len(text) > limit:
        text = text[:limit] + "\n…"
    return text


def _worst_outlier_column(profile: dict) -> str | None:
    outliers = profile.get("outlier_columns") or []
    return outliers[0]["column"] if outliers else None


def _top_correlated_pair(profile: dict) -> tuple[str, str] | None:
    pairs = profile.get("top_correlated_pairs") or profile.get("spearman_pairs") or []
    if not pairs:
        return None
    return str(pairs[0]["col_a"]), str(pairs[0]["col_b"])


def _default_group_pair(profile: dict) -> tuple[str, str] | None:
    """Numeric x categorical pair for the deterministic group comparison."""
    effects = profile.get("category_effects") or []
    if effects:
        return str(effects[0]["numeric"]), str(effects[0]["category"])
    columns = profile.get("columns") or {}
    numeric = next(
        (name for name, c in columns.items() if c.get("type") == "numeric"), None
    )
    cats = _drilldown_categoricals(profile)
    if numeric is not None and cats:
        return numeric, cats[0]
    return None


def _drilldown_categoricals(profile: dict) -> list[str]:
    """Categorical/boolean columns with 2-20 groups, most groups first."""
    columns = profile.get("columns") or {}
    cats = [
        (name, info.get("n_unique") or 0)
        for name, info in columns.items()
        if info.get("type") in ("categorical", "boolean")
        and 1 < (info.get("n_unique") or 0) <= 20
    ]
    cats.sort(key=lambda t: (-t[1], t[0]))
    return [name for name, _ in cats]


def _deterministic_plan(profile: dict) -> list[tuple[str, dict[str, str]]]:
    """Three profile-informed tool calls for the offline investigation."""
    plan: list[tuple[str, dict[str, str]]] = []
    outlier_column = _worst_outlier_column(profile)
    if outlier_column:
        plan.append(("outlier_inspect", {"column": outlier_column}))
    pair = _top_correlated_pair(profile)
    if pair:
        plan.append(("correlation_check", {"column_a": pair[0], "column_b": pair[1]}))
    group_pair = _default_group_pair(profile)
    if group_pair:
        plan.append(
            (
                "group_compare",
                {"numeric_column": group_pair[0], "categorical_column": group_pair[1]},
            )
        )
    return plan


def _json_safe(value: Any) -> Any:
    """Round-trip a value through json so trace entries stay serializable."""
    return json.loads(json.dumps(value, default=str))


def _trace_tool_call(
    df: pd.DataFrame, name: str, args: dict, round_no: int, trace: list[dict]
) -> dict:
    result = run_tool(df, name, args)
    trace.append(
        {
            "round": round_no,
            "tool": name,
            "args": _json_safe(args),
            "summary": str(result.get("summary") or ""),
        }
    )
    return result


def _findings_from_trace(trace: list[dict]) -> str:
    bullets = [f"- {entry['summary']}" for entry in trace if entry.get("summary")]
    return (
        "\n".join(bullets)
        or "- no investigation tools could be applied to this dataset"
    )


def _deterministic_investigation(
    df: pd.DataFrame, profile: dict
) -> tuple[list[dict], list[dict]]:
    """Offline fallback: run the profile-informed plan and bullet its summaries."""
    trace: list[dict] = []
    for name, args in _deterministic_plan(profile):
        _trace_tool_call(df, name, args, 1, trace)
    investigations = [
        {
            "question": "agent-led investigation",
            "tool": "multi",
            "args": {},
            "summary": _findings_from_trace(trace),
        }
    ]
    return trace, investigations


def _run_tool_loop(bound, df: pd.DataFrame, messages: list, trace: list[dict]) -> str:
    """Drive the bound LLM through tool rounds; return the final findings text.

    Caps: MAX_AGENT_ROUNDS rounds, <=MAX_TOOL_CALLS_PER_ROUND executed calls per
    round, <=MAX_TOOL_CALLS_TOTAL overall. Calls beyond a cap still receive a
    ToolMessage (skipped) so the message history stays protocol-valid. Returns
    "" when no round produced plain text; callers then fall back to trace
    bullets.
    """
    total_calls = 0
    last_text = ""
    for round_no in range(1, MAX_AGENT_ROUNDS + 1):
        resp = _invoke_llm(bound, messages)
        messages.append(resp)
        content = getattr(resp, "content", "")
        last_text = _content_to_text("" if content is None else content).strip()
        tool_calls = list(getattr(resp, "tool_calls", None) or [])
        if not tool_calls:
            return last_text
        executed = 0
        for index, call in enumerate(tool_calls):
            if not isinstance(call, dict):
                call = {"name": str(call), "args": {}, "id": None}
            name = str(call.get("name") or "")
            args = call.get("args")
            call_id = str(call.get("id") or f"call_{round_no}_{index}")
            if (
                executed >= MAX_TOOL_CALLS_PER_ROUND
                or total_calls >= MAX_TOOL_CALLS_TOTAL
            ):
                result: dict = {
                    "error": "tool call skipped",
                    "summary": "skipped: tool-call cap reached",
                }
            else:
                result = run_tool(df, name, args if isinstance(args, dict) else {})
                trace.append(
                    {
                        "round": round_no,
                        "tool": name,
                        "args": _json_safe(args if isinstance(args, dict) else {}),
                        "summary": str(result.get("summary") or ""),
                    }
                )
                executed += 1
                total_calls += 1
            messages.append(
                ToolMessage(
                    content=json.dumps(result, default=str)[:TOOL_RESULT_CHARS],
                    tool_call_id=call_id,
                )
            )
        if total_calls >= MAX_TOOL_CALLS_TOTAL:
            return last_text
    return last_text


def _num(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.4g}"
    return str(value)


def _drilldown_comparison(out_res: dict, group_results: list[dict]) -> str:
    """Deterministic stats sentence (means/gaps) from the drill-down tools."""
    parts: list[str] = []
    if "error" not in out_res and out_res.get("summary"):
        parts.append(str(out_res["summary"]))
    for g in group_results:
        gap = g.get("biggest_gap") if isinstance(g, dict) else None
        if not gap:
            continue
        means = {r["category"]: r["mean"] for r in g.get("groups") or []}
        parts.append(
            f"by `{g.get('categorical_column')}`: mean `{gap['high_group']}` "
            f"{_num(means.get(gap['high_group']))} vs `{gap['low_group']}` "
            f"{_num(means.get(gap['low_group']))} (gap {gap['gap']})"
        )
    return " ".join(parts) if parts else "no comparison statistics available"


def _drilldown_template(
    column: str, n_outliers: int, out_res: dict, group_results: list[dict]
) -> str:
    pct = out_res.get("pct_outliers") if "error" not in out_res else None
    pct_txt = f" ({pct}% of non-null values)" if isinstance(pct, (int, float)) else ""
    gaps = [
        f"`{g['categorical_column']}`: '{g['biggest_gap']['high_group']}' vs "
        f"'{g['biggest_gap']['low_group']}' (gap {g['biggest_gap']['gap']})"
        for g in group_results
        if isinstance(g, dict) and g.get("biggest_gap")
    ]
    gap_txt = (
        " Group means diverge most across " + "; ".join(gaps) + "." if gaps else ""
    )
    return (
        f"`{column}` has {n_outliers} IQR outlier value(s){pct_txt}.{gap_txt} "
        "Review the extreme values before modeling: they are either data-entry "
        "errors worth correcting or a genuine heavy tail worth keeping."
    )


def _drilldown_narrative(
    llm: BaseChatModel | None,
    column: str,
    n_outliers: int,
    out_res: dict,
    group_results: list[dict],
) -> tuple[str, str | None]:
    """LLM narrative over the drill-down JSON, or the deterministic template."""
    if llm is not None:
        payload = json.dumps(
            {
                "column": column,
                "outlier_inspect": out_res,
                "group_comparisons": [g for g in group_results if "error" not in g],
            },
            indent=2,
            default=str,
        )[:ANOMALY_PAYLOAD_CHARS]
        try:
            resp = _invoke_llm(
                llm,
                [
                    ("system", SYSTEM_PROMPT),
                    ("human", ANOMALY_PROMPT.format(payload=payload)),
                ],
            )
            return _response_text(resp), None
        except Exception as exc:
            logger.warning("anomaly narrative LLM call failed: %s", exc)
            return (
                _drilldown_template(column, n_outliers, out_res, group_results),
                f"anomaly narrative LLM call failed ({exc.__class__.__name__}); "
                "using the deterministic template instead",
            )
    return _drilldown_template(column, n_outliers, out_res, group_results), None


def _invoke_llm(llm: BaseChatModel, messages: list) -> Any:
    """Invoke the LLM with bounded retries and exponential backoff."""
    last_exc: Exception | None = None
    for attempt in range(1, LLM_MAX_ATTEMPTS + 1):
        try:
            return llm.invoke(messages)
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "LLM call attempt %d/%d failed: %s", attempt, LLM_MAX_ATTEMPTS, exc
            )
            if attempt < LLM_MAX_ATTEMPTS:
                time.sleep(LLM_RETRY_BACKOFF_S * attempt)
    assert last_exc is not None
    raise last_exc


def run_pipeline(
    initial: dict,
    llm: BaseChatModel | None,
    on_node: Callable[[str], None] | None = None,
) -> dict:
    """Run the compiled EDA graph and accumulate node updates into one state.

    ``warnings`` lists accumulate across nodes (mirroring the EDAState
    reducer); every other key takes the latest node's value.
    """
    graph = build_graph(llm)
    final: dict = dict(initial)
    for update in graph.stream(initial, stream_mode="updates"):
        for node, delta in update.items():
            if isinstance(delta, dict):
                for key, value in delta.items():
                    if key == "warnings" and value:
                        final["warnings"] = [*final.get("warnings", []), *value]
                    else:
                        final[key] = value
            if on_node is not None:
                on_node(node)
    return final


def build_graph(llm: BaseChatModel | None):
    """Compile the EDA pipeline graph. LLM nodes fall back to heuristics when llm is None."""

    def load_data(state: EDAState) -> dict:
        input_path = state["input_path"]
        try:
            size = Path(input_path).stat().st_size
        except OSError as exc:
            return {"error": f"could not stat input '{input_path}': {exc}"}
        if size > MAX_INPUT_BYTES:
            return {
                "error": (
                    f"input is {size / 1e6:.1f} MB; maximum supported size is "
                    f"{MAX_INPUT_BYTES / 1e6:.0f} MB"
                )
            }
        try:
            df, load_warnings = load_table(
                Path(input_path), sheet=state.get("sheet") or None
            )
        except LoadError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            return {"error": f"could not load '{input_path}': {exc}"}
        if df.shape[1] == 0:
            return {"error": "input contains no columns"}
        if df.shape[0] == 0:
            return {"error": "input contains no data rows"}
        if df.shape[0] > MAX_ROWS:
            return {
                "error": (
                    f"input has {df.shape[0]:,} rows; maximum supported is {MAX_ROWS:,}"
                )
            }
        result: dict = {"df": _infer_datetimes(df)}
        if load_warnings:
            result["warnings"] = load_warnings
        return result

    def statistical_analysis(state: EDAState) -> dict:
        profile = profile_dataframe(state["df"])
        profile["health"] = health_score(profile)
        return {"profile": profile}

    def generate_visualizations(state: EDAState) -> dict:
        chart_failures: list[str] = []
        charts = generate_charts(
            state["df"],
            Path(state["output_dir"]) / "charts",
            failures=chart_failures,
            corr_pairs=(state.get("profile") or {}).get("top_correlated_pairs"),
        )
        return {"charts": charts, "warnings": chart_failures}

    def extract_insights(state: EDAState) -> dict:
        profile = state["profile"]
        if llm is None:
            return {"insights": heuristic_insights(profile)}
        prompt = INSIGHT_PROMPT.format(
            profile_json=_profile_json(profile),
            charts=_charts_text(state.get("charts") or []),
        )
        try:
            resp = _invoke_llm(llm, [("system", SYSTEM_PROMPT), ("human", prompt)])
            return {"insights": _response_text(resp)}
        except Exception as exc:
            logger.warning(
                "LLM insight extraction failed after %d attempts; "
                "falling back to heuristics: %s",
                LLM_MAX_ATTEMPTS,
                exc,
            )
            return {
                "insights": heuristic_insights(profile),
                "warnings": [
                    f"insight LLM call failed ({exc.__class__.__name__}); "
                    "showing deterministic heuristic insights instead"
                ],
            }

    def make_recommendations(state: EDAState) -> dict:
        profile = state["profile"]
        if llm is None:
            return {"recommendations": heuristic_recommendations(profile)}
        prompt = RECOMMENDATION_PROMPT.format(
            insights=state.get("insights", ""),
            profile_json=_profile_json(profile),
        )
        try:
            resp = _invoke_llm(llm, [("system", SYSTEM_PROMPT), ("human", prompt)])
            return {"recommendations": _response_text(resp)}
        except Exception as exc:
            logger.warning(
                "LLM recommendations failed after %d attempts; "
                "falling back to heuristics: %s",
                LLM_MAX_ATTEMPTS,
                exc,
            )
            return {
                "recommendations": heuristic_recommendations(profile),
                "warnings": [
                    f"recommendations LLM call failed ({exc.__class__.__name__}); "
                    "showing deterministic heuristic recommendations instead"
                ],
            }

    def agentic_investigation(state: EDAState) -> dict:
        df = state["df"]
        profile = state["profile"]
        if llm is None:
            trace, investigations = _deterministic_investigation(df, profile)
            return {"agent_trace": trace, "investigations": investigations}
        try:
            bound = llm.bind_tools(TOOL_SPECS)
        except Exception as exc:
            logger.warning("tool binding failed for the investigation loop: %s", exc)
            trace, investigations = _deterministic_investigation(df, profile)
            return {
                "agent_trace": trace,
                "investigations": investigations,
                "warnings": [
                    f"investigation tool binding failed ({exc.__class__.__name__}); "
                    "ran the deterministic investigation instead"
                ],
            }
        messages: list = [
            ("system", AGENT_SYSTEM_PROMPT),
            ("human", INVESTIGATION_PROMPT.format(digest=_profile_digest(profile))),
        ]
        trace: list[dict] = []
        try:
            findings = _run_tool_loop(bound, df, messages, trace)
        except Exception as exc:
            logger.warning(
                "investigation LLM loop failed; falling back to tools: %s", exc
            )
            trace, investigations = _deterministic_investigation(df, profile)
            return {
                "agent_trace": trace,
                "investigations": investigations,
                "warnings": [
                    f"investigation LLM call failed ({exc.__class__.__name__}); "
                    "ran the deterministic investigation instead"
                ],
            }
        if not findings:
            findings = _findings_from_trace(trace)
        return {
            "agent_trace": trace,
            "investigations": [
                {
                    "question": "agent-led investigation",
                    "tool": "multi",
                    "args": {},
                    "summary": findings,
                }
            ],
        }

    def anomaly_drilldown(state: EDAState) -> dict:
        df = state["df"]
        profile = state["profile"]
        outlier_columns = [
            str(o["column"]) for o in (profile.get("outlier_columns") or [])
        ]
        if not outlier_columns:
            return {"anomaly_reports": []}
        reports: list[dict] = []
        warnings: list[str] = []
        for column in outlier_columns[:2]:
            out_res = run_tool(df, "outlier_inspect", {"column": column})
            group_results = [
                run_tool(
                    df,
                    "group_compare",
                    {"numeric_column": column, "categorical_column": cat},
                )
                for cat in _drilldown_categoricals(profile)[:2]
            ]
            n_outliers = out_res.get("n_outliers")
            if not isinstance(n_outliers, int):
                n_outliers = next(
                    (
                        o["count"]
                        for o in profile.get("outlier_columns") or []
                        if o["column"] == column
                    ),
                    0,
                )
            narrative, warning = _drilldown_narrative(
                llm, column, int(n_outliers), out_res, group_results
            )
            if warning:
                warnings.append(warning)
            reports.append(
                {
                    "column": column,
                    "n_outliers": int(n_outliers),
                    "comparison": _drilldown_comparison(out_res, group_results),
                    "narrative": narrative,
                }
            )
        result: dict = {"anomaly_reports": reports}
        if warnings:
            result["warnings"] = warnings
        return result

    def build_report(state: EDAState) -> dict:
        report_path = write_report(state)
        html_path = write_html_report(state, Path(state["output_dir"]))
        return {"report_path": report_path, "html_report_path": html_path}

    def _route_after_load(state: EDAState) -> str:
        return "statistical_analysis" if not state.get("error") else "end"

    graph = StateGraph(EDAState)
    graph.add_node("load_data", load_data)
    graph.add_node("statistical_analysis", statistical_analysis)
    graph.add_node("generate_visualizations", generate_visualizations)
    graph.add_node("extract_insights", extract_insights)
    graph.add_node("recommendations", make_recommendations)
    graph.add_node("agentic_investigation", agentic_investigation)
    graph.add_node("anomaly_drilldown", anomaly_drilldown)
    graph.add_node("build_report", build_report)

    graph.add_edge(START, "load_data")
    graph.add_conditional_edges(
        "load_data",
        _route_after_load,
        {"statistical_analysis": "statistical_analysis", "end": END},
    )
    graph.add_edge("statistical_analysis", "generate_visualizations")
    graph.add_edge("generate_visualizations", "extract_insights")
    graph.add_edge("extract_insights", "recommendations")
    graph.add_edge("recommendations", "agentic_investigation")
    graph.add_edge("agentic_investigation", "anomaly_drilldown")
    graph.add_edge("anomaly_drilldown", "build_report")
    graph.add_edge("build_report", END)
    return graph.compile()
