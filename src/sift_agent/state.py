from __future__ import annotations

from operator import add
from typing import Annotated, Any, TypedDict


class ChartRef(TypedDict):
    title: str
    path: str  # relative to the report directory, e.g. "charts/02_distributions.png"
    caption: str


class EDAState(TypedDict, total=False):
    # inputs
    csv_path: str
    output_dir: str
    provider: str
    model: str
    # intermediate results
    df: Any  # pandas DataFrame
    profile: dict[str, Any]
    charts: list[ChartRef]
    # outputs
    insights: str
    recommendations: str
    report_path: str
    error: str
    # non-fatal issues surfaced in the report (LLM fallbacks, skipped charts, ...)
    warnings: Annotated[list[str], add]
