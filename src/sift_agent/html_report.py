"""Standalone dark-theme HTML report for a sift-agent run.

Renders the run state (profile, charts, insights, recommendations, warnings)
into a single self-contained ``report.html``: inline CSS only, charts embedded
as base64 data URIs, and every dynamic value HTML-escaped, so the file opens
offline in any browser with zero external requests. The palette mirrors the
Streamlit theme in ``.streamlit/config.toml``.
"""

from __future__ import annotations

import base64
import html
import json
import logging
import math
import re
from datetime import UTC, datetime
from pathlib import Path

from .analysis import column_summary

logger = logging.getLogger("sift_agent.html_report")

MAX_COLUMNS_SHOWN = 60

_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
_CODE_SPAN_RE = re.compile(r"`([^`]+)`")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_BULLET_RE = re.compile(r"^[-*]\s+(.+)$")

_CSS = """
html{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:#1b1b23;color:#f5f5f7;
  font:15px/1.65 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:0 28px 56px}
.band{background:linear-gradient(120deg,#372d6b,#251f47 55%,#1b1b23);border-bottom:1px solid #35354a}
.band-inner{padding-top:38px;padding-bottom:30px}
h1{margin:0 0 8px;font-size:27px;font-weight:700;letter-spacing:-.01em}
.meta{display:flex;flex-wrap:wrap;gap:10px;align-items:center;color:#9d9dac;font-size:13px}
.pill{display:inline-block;padding:2px 12px;border-radius:999px;border:1px solid rgba(143,120,230,.55);
  background:rgba(100,74,201,.22);color:#cfc5f7;font-size:12.5px}
section{margin-top:40px}
h2{display:flex;align-items:center;gap:10px;font-size:19px;font-weight:600;margin:0 0 14px;
  padding-bottom:10px;border-bottom:1px solid #35354a}
h2::before{content:"";width:9px;height:9px;border-radius:2.5px;flex:none;
  background:linear-gradient(135deg,#8f78e6,#644ac9)}
h3.sub{font-size:14.5px;font-weight:600;color:#cfc9e8;margin:22px 0 8px}
h3.sub:first-child{margin-top:0}
.panel{background:#24242e;border:1px solid #35354a;border-radius:14px;padding:22px 26px}
.health{display:flex;flex-wrap:wrap;gap:26px 34px;align-items:center}
.score-side{display:flex;align-items:center;gap:20px}
.score-num{font-size:62px;font-weight:700;line-height:1;font-variant-numeric:tabular-nums}
.score-label{font-size:11px;text-transform:uppercase;letter-spacing:.12em;color:#9d9dac;margin-top:6px}
.grade{width:58px;height:58px;border-radius:50%;border:3px solid;display:flex;align-items:center;
  justify-content:center;font-size:25px;font-weight:700;flex:none}
.verdict{flex:1 1 260px;min-width:240px;color:#d9d9e3;font-size:14.5px;
  border-left:2px solid #35354a;padding-left:18px}
.comps{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px 30px;flex:1 1 480px}
.comp-head{display:flex;justify-content:space-between;font-size:13px;margin-bottom:5px}
.comp-head .lbl{color:#cfc9e8;font-weight:600}
.comp-head .val{color:#9d9dac;font-variant-numeric:tabular-nums}
.bar-track{height:8px;border-radius:999px;background:#33333f;overflow:hidden}
.bar-fill{height:100%;border-radius:999px}
.good{color:#3fb950}
.warn{color:#d29922}
.bad{color:#f85149}
.grade.good,.score-num.good{color:#3fb950}
.grade.warn,.score-num.warn{color:#d29922}
.grade.bad,.score-num.bad{color:#f85149}
.grade.good{border-color:#3fb950}
.grade.warn{border-color:#d29922}
.grade.bad{border-color:#f85149}
.bar-fill.good{background:linear-gradient(90deg,#2ea043,#3fb950)}
.bar-fill.warn{background:linear-gradient(90deg,#b8860b,#d29922)}
.bar-fill.bad{background:linear-gradient(90deg,#c62828,#f85149)}
.comp-detail{font-size:12px;color:#9d9dac;margin-top:5px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px}
.card{background:#24242e;border:1px solid #35354a;border-radius:12px;padding:14px 18px}
.card-v{font-size:26px;font-weight:700;font-variant-numeric:tabular-nums}
.card-v small{font-size:14px;font-weight:500;color:#9d9dac}
.card-l{font-size:10.5px;text-transform:uppercase;letter-spacing:.11em;color:#9d9dac;margin-top:3px}
.card-s{font-size:12px;color:#9d9dac;margin-top:2px}
.tbl-wrap{overflow-x:auto;border:1px solid #35354a;border-radius:10px;background:#24242e}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th{text-align:left;padding:9px 13px;font-size:10.5px;text-transform:uppercase;letter-spacing:.09em;
  color:#9d9dac;background:#2b2b37;border-bottom:1px solid #35354a;white-space:nowrap}
td{padding:8px 13px;border-bottom:1px solid rgba(53,53,74,.55);vertical-align:middle}
tbody tr:nth-child(even){background:rgba(255,255,255,.025)}
tbody tr:last-child td{border-bottom:none}
code{background:#2f2f3c;color:#c9befa;padding:1px 6px;border-radius:5px;font-size:.92em;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.cell-meter{display:flex;align-items:center;gap:8px}
.meter{width:90px;max-width:100%;height:6px;border-radius:999px;background:#33333f;overflow:hidden;flex:none}
.meter .m{height:100%;border-radius:999px}
.meter .m.good{background:#3fb950}
.meter .m.warn{background:#d29922}
.meter .m.bad{background:#f85149}
figure.chart{margin:0 0 18px;background:#24242e;border:1px solid #35354a;border-radius:12px;
  padding:14px 14px 10px}
figure.chart:last-child{margin-bottom:0}
figure.chart h3{margin:2px 4px 10px;font-size:14.5px;font-weight:600}
figure.chart img{display:block;max-width:100%;height:auto;margin:0 auto;border-radius:8px;background:#fff}
figcaption{font-size:12.5px;color:#9d9dac;padding:9px 4px 4px;margin-top:10px;
  border-top:1px solid rgba(53,53,74,.5)}
.prose ul{margin:8px 0;padding-left:22px}
.prose li{margin:5px 0}
.prose p{margin:8px 0}
.prose h3,.prose h4{margin:14px 0 6px;color:#cfc9e8}
.anomaly-card{background:#24242e;border:1px solid #35354a;border-radius:12px;
  padding:18px 22px;margin-bottom:14px}
.anomaly-card:last-child{margin-bottom:0}
.warnbox{background:rgba(210,153,34,.07);border:1px solid rgba(210,153,34,.4);
  border-left:3px solid #d29922;border-radius:10px;padding:16px 20px}
.warnbox ul{margin:6px 0;padding-left:20px}
.warnbox li{margin:4px 0}
.note{font-size:12.5px;color:#9d9dac;margin-top:10px;font-style:italic}
footer{margin-top:52px;padding-top:18px;border-top:1px solid #35354a;font-size:12.5px;color:#9d9dac}
@media (max-width:760px){
  .health{flex-direction:column;align-items:flex-start}
  .score-num{font-size:48px}
  h1{font-size:22px}
}
"""


def _esc(value) -> str:
    return html.escape("—" if value is None else str(value), quote=True)


def _to_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _to_pct(value) -> int:
    number = _to_float(value)
    if number is None:
        return 0
    return max(0, min(100, round(number)))


def _band_class(score: int) -> str:
    if score >= 90:
        return "good"
    return "warn" if score >= 70 else "bad"


def _fmt_int(value) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return _esc(value)


def _fmt_float(value, plus: bool = False, digits: int = 2) -> str:
    number = _to_float(value)
    if number is None:
        return "—"
    return f"{number:+.{digits}f}" if plus else f"{number:.{digits}f}"


def _pct(value, digits: int = 1) -> str:
    number = _to_float(value)
    return "—" if number is None else f"{number:.{digits}f}%"


def _fmt_example(value) -> str:
    number = _to_float(value)
    return f"{number:,.6g}" if number is not None else _esc(value)


def _compact_args(args) -> str:
    """Compact one-line JSON for the agent-trace args column."""
    try:
        return json.dumps(args or {}, default=str, separators=(",", ":"))
    except (TypeError, ValueError):
        return _esc(args)


def _md_inline(escaped: str) -> str:
    """Bold/code inline markdown over already-escaped text (never raw HTML)."""
    stashed: list[str] = []

    def _stash(match: re.Match[str]) -> str:
        stashed.append(match.group(1))
        return f"\x00{len(stashed) - 1}\x00"

    text = _CODE_SPAN_RE.sub(_stash, escaped)
    text = _BOLD_RE.sub(r"<strong>\1</strong>", text)
    for index, code_span in enumerate(stashed):
        text = text.replace(f"\x00{index}\x00", f"<code>{code_span}</code>")
    return text


def _markdown_to_html(escaped_markdown: str) -> str:
    """Tiny GitHub-flavored subset: headings, '- ' bullets, **bold**, `code`."""
    out: list[str] = []
    in_list = False

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    for raw_line in escaped_markdown.splitlines():
        line = raw_line.strip()
        if not line:
            close_list()
            continue
        heading = _HEADING_RE.match(line)
        if heading:
            close_list()
            level = 3 if len(heading.group(1)) <= 2 else 4
            out.append(f"<h{level}>{_md_inline(heading.group(2))}</h{level}>")
            continue
        bullet = _BULLET_RE.match(line)
        if bullet:
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{_md_inline(bullet.group(1))}</li>")
            continue
        close_list()
        out.append(f"<p>{_md_inline(line)}</p>")
    close_list()
    return "\n".join(out)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body_rows = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    return (
        '<div class="tbl-wrap"><table><thead><tr>'
        f"{head}</tr></thead><tbody>{body_rows}</tbody></table></div>"
    )


def _meter_cell(value) -> str:
    number = _to_float(value)
    if number is None:
        return "—"
    width = max(0.0, min(100.0, number))
    cls = "bad" if width >= 50 else ("warn" if width >= 20 else "good")
    return (
        f'<div class="cell-meter"><div class="meter">'
        f'<div class="m {cls}" style="width:{width:.0f}%"></div></div>'
        f"<span>{number:.1f}%</span></div>"
    )


def _card(value_html: str, label: str, sub: str = "") -> str:
    sub_html = f'<div class="card-s">{_esc(sub)}</div>' if sub else ""
    return (
        '<div class="card">'
        f'<div class="card-v">{value_html}</div>'
        f'<div class="card-l">{_esc(label)}</div>'
        f"{sub_html}</div>"
    )


def _dataset_name(state: dict) -> str:
    return Path(str(state.get("input_path") or "dataset")).name or "dataset"


def _llm_label(state: dict) -> str:
    provider = state.get("provider") or "none"
    model = state.get("model") or ""
    label = model or (
        "heuristic (offline)" if provider == "none" else "provider default"
    )
    return f"{provider} · {label}"


def _header_band(state: dict) -> str:
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    return (
        '<header class="band"><div class="wrap band-inner">'
        f"<h1>{_esc(_dataset_name(state))}</h1>"
        '<div class="meta">'
        f"<span>{_esc(stamp)}</span>"
        f'<span class="pill">{_esc(_llm_label(state))}</span>'
        "</div></div></header>"
    )


def _health_section(health) -> str:
    if not health:
        return ""
    score = _to_pct(health.get("score"))
    grade = str(health.get("grade") or "").strip()[:1].upper() or "—"
    cls = _band_class(score)
    comps: list[str] = []
    for comp in health.get("components") or []:
        comp_score = _to_pct(comp.get("score"))
        comps.append(
            '<div class="comp">'
            '<div class="comp-head">'
            f'<span class="lbl">{_esc(comp.get("label") or comp.get("key") or "component")}</span>'
            f'<span class="val">{comp_score}/100</span></div>'
            f'<div class="bar-track"><div class="bar-fill {_band_class(comp_score)}" '
            f'style="width:{comp_score}%"></div></div>'
            f'<div class="comp-detail">{_esc(comp.get("detail") or "")}</div>'
            "</div>"
        )
    comps_html = f'<div class="comps">{"".join(comps)}</div>' if comps else ""
    return (
        "<section><h2>Data health</h2>"
        '<div class="panel health">'
        '<div class="score-side">'
        f'<div><div class="score-num {cls}">{score}</div>'
        '<div class="score-label">health score / 100</div></div>'
        f'<div class="grade {cls}">{_esc(grade)}</div>'
        "</div>"
        f'<div class="verdict">{_esc(health.get("verdict") or "")}</div>'
        f"{comps_html}"
        "</div></section>"
    )


def _overview_section(profile: dict) -> str:
    cards: list[str] = []
    if "n_rows" in profile:
        cards.append(_card(_fmt_int(profile["n_rows"]), "Rows"))
    if "n_columns" in profile:
        sub = ""
        if "n_numeric_columns" in profile or "n_categorical_columns" in profile:
            sub = (
                f"{_fmt_int(profile.get('n_numeric_columns', 0))} numeric · "
                f"{_fmt_int(profile.get('n_categorical_columns', 0))} categorical"
            )
        cards.append(_card(_fmt_int(profile["n_columns"]), "Columns", sub))
    if "duplicate_rows" in profile:
        cards.append(_card(_fmt_int(profile["duplicate_rows"]), "Duplicate rows"))
    if "memory_mb" in profile:
        memory = f"{_fmt_float(profile['memory_mb'], digits=2)} <small>MB</small>"
        cards.append(_card(memory, "Memory"))
    if not cards:
        return ""
    return (
        f'<section><h2>Overview</h2><div class="cards">{"".join(cards)}</div></section>'
    )


def _corr_rows(pairs: list[dict], key: str, plus: bool) -> list[list[str]]:
    return [
        [
            f"<code>{_esc(p.get('col_a'))}</code>",
            f"<code>{_esc(p.get('col_b'))}</code>",
            _fmt_float(p.get(key), plus=plus),
        ]
        for p in pairs
    ]


def _embed_chart(out_dir: Path, chart: dict) -> str:
    rel = chart.get("path")
    if not rel:
        return ""
    try:
        data = (out_dir / str(rel)).read_bytes()
    except OSError as exc:
        logger.warning("chart file unreadable, skipping embed: %s (%s)", rel, exc)
        return ""
    encoded = base64.b64encode(data).decode("ascii")
    title = chart.get("title") or ""
    caption = chart.get("caption") or ""
    title_html = f"<h3>{_esc(title)}</h3>" if title else ""
    caption_html = f"<figcaption>{_esc(caption)}</figcaption>" if caption else ""
    alt = _esc(title or caption)
    return (
        '<figure class="chart">'
        f"{title_html}"
        f'<img src="data:image/png;base64,{encoded}" alt="{alt}">'
        f"{caption_html}"
        "</figure>"
    )


def _data_sections(state: dict, profile: dict, out_dir: Path) -> list[str]:
    sections: list[str] = []

    def add(title: str, inner: str) -> None:
        if inner.strip():
            sections.append(f"<section><h2>{title}</h2>{inner}</section>")

    missing = profile.get("top_missing") or []
    if missing:
        rows = [
            [
                f"<code>{_esc(m.get('column'))}</code>",
                _fmt_int(m.get("n_missing")),
                _meter_cell(m.get("pct_missing")),
            ]
            for m in missing
        ]
        add("Missing values", _table(["column", "missing", "% missing"], rows))

    quality = profile.get("data_quality") or []
    if quality:
        rows = [
            [
                _esc(str(q.get("check") or "").replace("_", " ")),
                f"<code>{_esc(q.get('column'))}</code>",
                _esc(q.get("detail") or ""),
            ]
            for q in quality
        ]
        add("Data quality", _table(["check", "column", "detail"], rows))

    pii = profile.get("pii_columns") or []
    if pii:
        rows = [
            [
                f"<code>{_esc(p.get('column'))}</code>",
                _esc(p.get("label") or ""),
                _fmt_int(p.get("count")),
                f"{_fmt_float(p.get('pct'), digits=1)}%",
            ]
            for p in pii
        ]
        add(
            "Potential PII — redact before sharing externally",
            _table(["column", "looks like", "matches", "%"], rows),
        )

    outliers = profile.get("outlier_columns") or []
    if outliers:
        rows = []
        for o in outliers:
            examples = o.get("examples") or []
            example_text = (
                ", ".join(_fmt_example(v) for v in examples) if examples else "—"
            )
            pct_text = _fmt_float(o.get("pct"), digits=1)
            rows.append(
                [
                    f"<code>{_esc(o.get('column'))}</code>",
                    _fmt_int(o.get("count")),
                    f"{pct_text}%" if pct_text != "—" else "—",
                    example_text,
                ]
            )
        add(
            "Outliers (IQR)",
            _table(["column", "count", "% of rows", "examples"], rows),
        )

    corr_parts: list[str] = []
    pearson = profile.get("top_correlated_pairs") or []
    if pearson:
        corr_parts.append('<h3 class="sub">Pearson r</h3>')
        corr_parts.append(
            _table(
                ["col a", "col b", "pearson r"],
                _corr_rows(pearson, "pearson_r", plus=True),
            )
        )
    spearman = profile.get("spearman_pairs") or []
    if spearman:
        corr_parts.append('<h3 class="sub">Spearman ρ</h3>')
        corr_parts.append(
            _table(
                ["col a", "col b", "spearman ρ"],
                _corr_rows(spearman, "spearman_r", plus=True),
            )
        )
    cramers = profile.get("cramers_v_pairs") or []
    if cramers:
        corr_parts.append('<h3 class="sub">Cramér\'s V</h3>')
        corr_parts.append(
            _table(
                ["col a", "col b", "Cramér's V"],
                _corr_rows(cramers, "cramers_v", plus=False),
            )
        )
    if corr_parts:
        add("Correlations", "".join(corr_parts))

    columns = profile.get("columns") or {}
    if columns:
        rows = []
        for name, c in list(columns.items())[:MAX_COLUMNS_SHOWN]:
            c = c or {}
            rows.append(
                [
                    f"<code>{_esc(name)}</code>",
                    _esc(c.get("type") or "?"),
                    _esc(c.get("dtype") or "?"),
                    _pct(c.get("pct_missing")),
                    _fmt_int(c.get("n_unique")),
                    _esc(column_summary(c)),
                ]
            )
        inner = _table(
            ["column", "type", "dtype", "missing", "unique", "summary"], rows
        )
        if len(columns) > MAX_COLUMNS_SHOWN:
            inner += (
                f'<p class="note">Showing the first {MAX_COLUMNS_SHOWN} of '
                f"{len(columns)} columns.</p>"
            )
        add("Column profiles", inner)

    figures: list[str] = []
    for chart in state.get("charts") or []:
        figure = _embed_chart(out_dir, chart)
        if figure:
            figures.append(figure)
    if figures:
        add("Visualizations", "".join(figures))

    insights = state.get("insights")
    if isinstance(insights, str) and insights.strip():
        prose = _markdown_to_html(_esc(insights))
        add("Key insights", f'<div class="panel prose">{prose}</div>')

    recommendations = state.get("recommendations")
    if isinstance(recommendations, str) and recommendations.strip():
        prose = _markdown_to_html(_esc(recommendations))
        add("Recommendations", f'<div class="panel prose">{prose}</div>')

    trace = [t for t in (state.get("agent_trace") or []) if isinstance(t, dict)]
    investigations = [
        i for i in (state.get("investigations") or []) if isinstance(i, dict)
    ]
    if trace or investigations:
        parts: list[str] = []
        if trace:
            rows = [
                [
                    _esc(t.get("round")),
                    f"<code>{_esc(t.get('tool'))}</code>",
                    f"<code>{_esc(_compact_args(t.get('args')))}</code>",
                    _esc(t.get("summary") or ""),
                ]
                for t in trace
            ]
            parts.append(_table(["round", "tool", "args", "summary"], rows))
        findings = "\n\n".join(
            str(i.get("summary") or "").strip()
            for i in investigations
            if str(i.get("summary") or "").strip()
        )
        if findings:
            prose = _markdown_to_html(_esc(findings))
            parts.append(f'<div class="panel prose">{prose}</div>')
        add("Agent investigations", "".join(parts))

    anomaly_reports = [
        r for r in (state.get("anomaly_reports") or []) if isinstance(r, dict)
    ]
    if anomaly_reports:
        cards: list[str] = []
        for r in anomaly_reports:
            comparison = str(r.get("comparison") or "").strip()
            comparison_html = (
                f"<p><strong>Comparison:</strong> {_md_inline(_esc(comparison))}</p>"
                if comparison
                else ""
            )
            narrative = _markdown_to_html(_esc(str(r.get("narrative") or "")))
            cards.append(
                '<div class="anomaly-card">'
                f'<h3 class="sub"><code>{_esc(r.get("column"))}</code> — '
                f"{_fmt_int(r.get('n_outliers'))} IQR outliers</h3>"
                f"{comparison_html}"
                f'<div class="prose">{narrative}</div>'
                "</div>"
            )
        add("Anomaly drill-down", "".join(cards))

    warnings = [w for w in (state.get("warnings") or []) if w]
    if warnings:
        items = "".join(f"<li>{_md_inline(_esc(w))}</li>" for w in warnings)
        add("Warnings", f'<div class="warnbox"><ul>{items}</ul></div>')

    return sections


def write_html_report(state: dict, out_dir: Path) -> str:
    """Render ``state`` into a standalone ``out_dir/report.html``; return its path."""
    out_dir = Path(out_dir)
    profile = state.get("profile") or {}
    out_dir.mkdir(parents=True, exist_ok=True)

    body = [
        _header_band(state),
        '<main class="wrap">',
        _health_section(profile.get("health")),
        _overview_section(profile),
        *_data_sections(state, profile, out_dir),
        "<footer>Generated by sift-agent — auto-EDA with LangGraph.</footer>",
        "</main>",
    ]
    page = (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>sift-agent report — {_esc(_dataset_name(state))}</title>\n"
        f"<style>{_CSS}</style>\n"
        "</head>\n<body>\n" + "\n".join(body) + "\n</body>\n</html>\n"
    )
    report_path = out_dir / "report.html"
    report_path.write_text(page, encoding="utf-8")
    logger.info("html report written: %s", report_path)
    return str(report_path)
