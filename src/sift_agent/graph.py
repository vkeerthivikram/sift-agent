"""LangGraph state machine wiring the auto-EDA pipeline together.

Flow: load_data -> statistical_analysis -> generate_visualizations
      -> extract_insights -> recommendations -> build_report
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
from langgraph.graph import END, START, StateGraph

from .analysis import profile_dataframe
from .loader import LoadError, load_table
from .report import write_report
from .state import EDAState
from .visualize import generate_charts

logger = logging.getLogger("sift_agent.graph")

MAX_INPUT_BYTES = 512 * 1024 * 1024
MAX_ROWS = 1_000_000
LLM_MAX_ATTEMPTS = 3
LLM_RETRY_BACKOFF_S = 2.0

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

    return "\n".join(f"- {b}" for b in bullets)


def heuristic_recommendations(profile: dict) -> str:
    """Deterministic recommendations used when no LLM is configured (provider 'none')."""
    cols = profile["columns"]
    recs: list[str] = []

    if profile["duplicate_rows"]:
        recs.append(
            f"Remove the {profile['duplicate_rows']:,} duplicate rows (`df.drop_duplicates()`)."
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

    if any(c.get("type") == "numeric" for c in cols.values()):
        recs.append(
            "Scale numeric features and start with a simple baseline (e.g. logistic regression or gradient boosting) using a proper train/test split."
        )

    return (
        "\n".join(f"- {r}" for r in recs)
        if recs
        else "- Dataset looks clean — proceed to hypothesis-driven analysis."
    )


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
        return {"profile": profile_dataframe(state["df"])}

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
            return {"insights": _content_to_text(resp.content).strip()}
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
            return {"recommendations": _content_to_text(resp.content).strip()}
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

    def build_report(state: EDAState) -> dict:
        return {"report_path": write_report(state)}

    def _route_after_load(state: EDAState) -> str:
        return "statistical_analysis" if not state.get("error") else "end"

    graph = StateGraph(EDAState)
    graph.add_node("load_data", load_data)
    graph.add_node("statistical_analysis", statistical_analysis)
    graph.add_node("generate_visualizations", generate_visualizations)
    graph.add_node("extract_insights", extract_insights)
    graph.add_node("recommendations", make_recommendations)
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
    graph.add_edge("recommendations", "build_report")
    graph.add_edge("build_report", END)
    return graph.compile()
