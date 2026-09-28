from __future__ import annotations

from operator import add
from typing import Annotated, Any, TypedDict


class ChartRef(TypedDict):
    title: str
    path: str  # relative to the report directory, e.g. "charts/02_distributions.png"
    caption: str


class InvestigationFinding(TypedDict):
    question: str
    tool: str  # tool name, or "multi" for the loop-level findings entry
    args: dict[str, Any]
    summary: str


class AgentTraceEntry(TypedDict):
    round: int  # loop round (deterministic path always uses 1)
    tool: str
    args: dict[str, Any]
    summary: str


class AnomalyReport(TypedDict):
    column: str
    n_outliers: int
    comparison: str  # deterministic stats sentence built from tool output
    narrative: str  # LLM narrative or deterministic template


class EDAState(TypedDict, total=False):
    # inputs
    input_path: str
    sheet: str  # Excel/ODS sheet name or 0-based index; "" = auto (combine compatible sheets)
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
    investigations: list[InvestigationFinding]  # findings from the agentic loop
    agent_trace: list[AgentTraceEntry]  # one entry per executed tool call
    anomaly_reports: list[AnomalyReport]  # drill-down per top outlier column
    report_path: str
    html_report_path: str  # standalone shareable report.html (charts embedded)
    error: str
    # non-fatal issues surfaced in the report (LLM fallbacks, skipped charts, ...)
    warnings: Annotated[list[str], add]
