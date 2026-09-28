"""Markdown report assembly."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from .analysis import column_summary

logger = logging.getLogger("sift_agent.report")


def _md_cell(text: str) -> str:
    """Make text safe for a markdown table cell (no pipes, no newlines)."""
    return text.replace("|", "\\|").replace("\n", " ")


def write_report(state: dict) -> str:
    """Write report.md + profile.json into the output dir; return the report path."""
    out_dir = Path(state["output_dir"])
    profile = state["profile"]
    charts = state.get("charts") or []
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "profile.json").write_text(
        json.dumps(profile, indent=2, default=str), encoding="utf-8"
    )

    provider = state.get("provider") or "none"
    model = state.get("model") or ""
    llm_label = model or (
        "heuristic (offline)" if provider == "none" else "provider default"
    )

    lines: list[str] = []
    lines.append(f"# Auto-EDA report — {Path(state['input_path']).name}")
    lines.append("")
    lines.append(
        f"*Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} "
        f"· provider `{provider}` · model `{llm_label}`*"
    )
    lines.append("")

    health = profile.get("health") or {}
    if health:
        n_issues = len(profile.get("data_quality") or [])
        n_missing = len(profile.get("top_missing") or [])
        shape = (
            f"The dataset holds **{profile['n_rows']:,} rows across "
            f"{profile['n_columns']} columns** "
            f"({profile['n_numeric_columns']} numeric, "
            f"{profile['n_categorical_columns']} categorical)."
        )
        if n_issues or n_missing:
            findings = (
                f" The pipeline flagged {n_issues} data-quality issue(s) and "
                f"missing values in {n_missing} column(s)."
            )
        else:
            findings = (
                " No missing values, duplicates or formatting issues were detected."
            )
        lines.append("## Executive summary")
        lines.append("")
        lines.append(health.get("verdict") or "")
        lines.append("")
        lines.append(
            f"{shape}{findings} Overall data health is "
            f"**{health.get('score')}/100 (grade {health.get('grade')})**. "
            "Key findings and recommended next steps are below."
        )
        lines.append("")

        lines.append("### Data health")
        lines.append("")
        lines.append("| component | score | detail |")
        lines.append("|---|---|---|")
        for comp in health.get("components") or []:
            lines.append(
                f"| {comp.get('label') or comp.get('key')} | "
                f"{comp.get('score')}/100 | {comp.get('detail', '')} |"
            )
        lines.append("")

    lines.append("## Overview")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| rows | {profile['n_rows']:,} |")
    lines.append(f"| columns | {profile['n_columns']} |")
    lines.append(f"| numeric columns | {profile['n_numeric_columns']} |")
    lines.append(f"| categorical columns | {profile['n_categorical_columns']} |")
    lines.append(f"| duplicate rows | {profile['duplicate_rows']:,} |")
    lines.append(f"| memory | {profile['memory_mb']} MB |")
    lines.append("")

    sample = profile.get("sample_rows") or []
    if sample:
        lines.append("## Sample rows")
        lines.append("")
        cols = list(sample[0].keys())
        lines.append("| " + " | ".join(f"`{c}`" for c in cols) + " |")
        lines.append("|" + "---|" * len(cols))
        for row in sample:
            cells = [
                str(row.get(c) if row.get(c) is not None else "—").replace("|", "\\|")
                for c in cols
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    missing = profile.get("top_missing") or []
    lines.append("## Missing values")
    lines.append("")
    if missing:
        lines.append("| column | missing | % |")
        lines.append("|---|---|---|")
        for m in missing:
            lines.append(
                f"| `{m['column']}` | {m['n_missing']:,} | {m['pct_missing']}% |"
            )
    else:
        lines.append("No missing values detected.")
    lines.append("")

    co = profile.get("co_missing") or {}
    if co.get("fully_empty_rows"):
        n = co["fully_empty_rows"]
        lines.append(
            f"- {n:,} {'row is' if n == 1 else 'rows are'} entirely empty "
            "(`df.dropna(how='all')` removes them)."
        )
    for p in (co.get("top_pairs") or [])[:3]:
        n = p["count"]
        lines.append(
            f"- `{p['col_a']}` and `{p['col_b']}` go missing together in "
            f"{n:,} {'row' if n == 1 else 'rows'} ({p['pct_of_rows']}%)."
        )
    if co.get("fully_empty_rows") or co.get("top_pairs"):
        lines.append("")

    lines.append("## Column profiles")
    lines.append("")
    lines.append("| column | type | dtype | missing % | unique | summary |")
    lines.append("|---|---|---|---|---|---|")
    n_columns = len(profile["columns"])
    for name, c in list(profile["columns"].items())[:60]:
        summary = column_summary(c)
        lines.append(
            f"| `{name}` | {c.get('type', '?')} | {c['dtype']} | {c['pct_missing']}% | {c['n_unique']:,} | {summary} |"
        )
    lines.append("")
    if n_columns > 60:
        lines.append(f"*Showing the first 60 of {n_columns} columns.*")
        lines.append("")

    warnings = state.get("warnings") or []
    if warnings:
        lines.append("## Warnings")
        lines.append("")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    quality = profile.get("data_quality") or []
    lines.append("## Data quality")
    lines.append("")
    if quality:
        lines.append("| check | column | detail |")
        lines.append("|---|---|---|")
        for q in quality:
            lines.append(
                f"| {q['check'].replace('_', ' ')} | `{q['column']}` | {q['detail']} |"
            )
    else:
        lines.append("No data-quality issues detected.")
    lines.append("")

    pii = profile.get("pii_columns") or []
    if pii:
        lines.append("## Potential PII")
        lines.append("")
        lines.append("| column | looks like | matches | % |")
        lines.append("|---|---|---|---|")
        for p in pii:
            lines.append(
                f"| `{p['column']}` | {p['label']} | {p['count']:,} | {p['pct']}% |"
            )
        lines.append("")
        lines.append(
            "*Redact or hash these columns before sharing this report or dataset externally.*"
        )
        lines.append("")

    outliers = profile.get("outlier_columns") or []
    if outliers:
        lines.append("## Outliers (IQR)")
        lines.append("")
        lines.append("| column | count | % of rows | examples |")
        lines.append("|---|---|---|---|")
        for o in outliers:
            examples = (
                ", ".join(f"{v:,.6g}" for v in o["examples"]) if o["examples"] else "—"
            )
            lines.append(
                f"| `{o['column']}` | {o['count']:,} | {o['pct']}% | {examples} |"
            )
        lines.append("")

    corr = profile.get("top_correlated_pairs") or []
    if corr:
        lines.append("## Strongest correlations")
        lines.append("")
        lines.append("| col a | col b | pearson r |")
        lines.append("|---|---|---|")
        for p in corr:
            lines.append(f"| `{p['col_a']}` | `{p['col_b']}` | {p['pearson_r']:+.2f} |")
        lines.append("")

    spearman = profile.get("spearman_pairs") or []
    if spearman:
        lines.append("## Strongest rank correlations (Spearman)")
        lines.append("")
        lines.append("| col a | col b | spearman ρ |")
        lines.append("|---|---|---|")
        for p in spearman:
            lines.append(
                f"| `{p['col_a']}` | `{p['col_b']}` | {p['spearman_r']:+.2f} |"
            )
        lines.append("")

    cramers = profile.get("cramers_v_pairs") or []
    if cramers:
        lines.append("## Categorical associations (Cramér's V)")
        lines.append("")
        lines.append("| col a | col b | Cramér's V |")
        lines.append("|---|---|---|")
        for p in cramers:
            lines.append(f"| `{p['col_a']}` | `{p['col_b']}` | {p['cramers_v']:.2f} |")
        lines.append("")

    effects = profile.get("category_effects") or []
    if effects:
        lines.append("## Group differences")
        lines.append("")
        lines.append("| numeric | category | mean | n rows |")
        lines.append("|---|---|---|---|")
        for e in effects[:5]:
            for cat, mean in e["means"].items():
                lines.append(
                    f"| `{e['numeric']}` | `{e['category']}` = {cat} | "
                    f"{mean:,.4g} | {e['counts'].get(cat, 0):,} |"
                )
        lines.append("")

    if charts:
        lines.append("## Visualizations")
        lines.append("")
        for ch in charts:
            lines.append(f"### {ch['title']}")
            lines.append("")
            lines.append(f"![{ch['title']}]({ch['path']})")
            lines.append("")
            lines.append(f"*{ch['caption']}*")
            lines.append("")

    lines.append("## Key insights")
    lines.append("")
    lines.append(state.get("insights") or "_no insights generated_")
    lines.append("")
    lines.append("## Recommendations")
    lines.append("")
    lines.append(state.get("recommendations") or "_no recommendations generated_")
    lines.append("")

    trace = [t for t in (state.get("agent_trace") or []) if isinstance(t, dict)]
    investigations = [
        i for i in (state.get("investigations") or []) if isinstance(i, dict)
    ]
    if trace or investigations:
        lines.append("## Agent investigations")
        lines.append("")
        lines.append(
            "Agent-led drill-through over the dataset; every finding traces to a "
            "recorded tool call."
        )
        lines.append("")
        if trace:
            lines.append("| round | tool | args | summary |")
            lines.append("|---|---|---|---|")
            for t in trace:
                args_txt = json.dumps(
                    t.get("args") or {}, default=str, separators=(",", ":")
                )
                lines.append(
                    f"| {t.get('round', '')} | `{_md_cell(str(t.get('tool') or ''))}` | "
                    f"`{_md_cell(args_txt)}` | {_md_cell(str(t.get('summary') or ''))} |"
                )
            lines.append("")
        for inv in investigations:
            summary = str(inv.get("summary") or "").strip()
            if summary:
                lines.append(summary)
                lines.append("")

    anomaly_reports = [
        r for r in (state.get("anomaly_reports") or []) if isinstance(r, dict)
    ]
    if anomaly_reports:
        lines.append("## Anomaly drill-down")
        lines.append("")
        for r in anomaly_reports:
            lines.append(
                f"### `{_md_cell(str(r.get('column') or ''))}` — "
                f"{r.get('n_outliers', 0)} IQR outliers"
            )
            lines.append("")
            lines.append(f"- Comparison: {_md_cell(str(r.get('comparison') or ''))}")
            lines.append("")
            lines.append(str(r.get("narrative") or ""))
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*Generated by sift-agent — auto-EDA with LangGraph.*")

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("report written: %s", report_path)
    return str(report_path)
