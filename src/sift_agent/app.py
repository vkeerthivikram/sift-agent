"""Streamlit UI for sift — the auto-EDA agent.

Run with:  uv run sift-ui
      or:  uv run streamlit run src/sift_agent/app.py
"""

from __future__ import annotations

import io
import os
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from sift_agent.analysis import column_summary
from sift_agent.config import DEFAULT_MODELS, PROVIDERS, REQUIRED_ENV, get_llm
from sift_agent.graph import run_pipeline
from sift_agent.loader import WORKBOOK_SUFFIXES, LoadError, sheet_names
from sift_agent.paths import unique_dir
from sift_agent.qa import answer_question

st.set_page_config(page_title="sift — auto-EDA", page_icon=":mag:", layout="wide")


def _health_banner(health: dict) -> None:
    """Colored one-line health banner plus a component breakdown expander."""
    score = health.get("score", 0)
    grade = health.get("grade", "?")
    color = "#3fb950" if score >= 90 else ("#d29922" if score >= 70 else "#f85149")
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:18px;padding:14px 22px;'
        f'border:1px solid #35354a;border-radius:12px;background:#24242e;">'
        f'<span style="font-size:44px;font-weight:700;color:{color};'
        f'line-height:1;">{score}</span>'
        f'<span style="font-size:22px;font-weight:700;color:{color};">{grade}</span>'
        f'<span style="color:#d9d9e3;">{health.get("verdict", "")}</span></div>',
        unsafe_allow_html=True,
    )
    components = health.get("components") or []
    if components:
        with st.expander("Health score components"):
            for comp in components:
                c_score = comp.get("score", 0)
                st.markdown(
                    f"**{comp.get('label', comp.get('key', ''))}** — "
                    f":{'green' if c_score >= 90 else ('orange' if c_score >= 70 else 'red')}:"
                    f"{c_score}/100 · {comp.get('detail', '')}"
                )


@st.cache_data
def _cached_sheet_names(data: bytes, suffix: str) -> list[str]:
    return sheet_names(io.BytesIO(data), suffix=suffix)


def _human_size(n_bytes: int) -> str:
    if n_bytes < 1_000_000:
        return f"{n_bytes / 1e3:.1f} KB"
    return f"{n_bytes / 1e6:.1f} MB"


def _credentials_ui(provider: str) -> dict[str, str]:
    """Optional credential inputs; returned as per-run overrides for get_llm()."""
    creds: dict[str, str] = {}
    if provider == "none":
        st.caption("Offline mode — deterministic heuristic insights, no LLM calls.")
        return creds
    st.caption(f"Env fallback: {REQUIRED_ENV.get(provider, '')}")

    if provider == "openai":
        creds["OPENAI_API_KEY"] = st.text_input("OPENAI_API_KEY", type="password")
    elif provider == "anthropic":
        creds["ANTHROPIC_API_KEY"] = st.text_input("ANTHROPIC_API_KEY", type="password")
    elif provider == "azure":
        creds["AZURE_OPENAI_ENDPOINT"] = st.text_input(
            "Endpoint", placeholder="https://<resource>.openai.azure.com"
        )
        creds["AZURE_OPENAI_API_VERSION"] = st.text_input(
            "API version", value="2024-10-21"
        )
        creds["AZURE_OPENAI_API_KEY"] = st.text_input("API key", type="password")
        creds["AZURE_OPENAI_DEPLOYMENT"] = st.text_input("Deployment name")
    elif provider == "bedrock":
        creds["AWS_ACCESS_KEY_ID"] = st.text_input("AWS access key ID")
        creds["AWS_SECRET_ACCESS_KEY"] = st.text_input(
            "AWS secret access key", type="password"
        )
        creds["AWS_REGION"] = st.text_input(
            "AWS region", value=os.environ.get("AWS_REGION", "")
        )
    elif provider == "openai-compatible":
        creds["OPENAI_BASE_URL"] = st.text_input(
            "Base URL", placeholder="http://localhost:11434/v1"
        )
        creds["OPENAI_API_KEY"] = st.text_input(
            "API key", type="password", placeholder="EMPTY for local servers"
        )
    return {k: v.strip() for k, v in creds.items() if v and v.strip()}


with st.sidebar:
    st.header("sift — auto-EDA")
    st.caption(
        "CSV/Excel/ODS in, statistical profile, charts, insights and recommendations out. Powered by LangGraph."
    )
    provider = st.selectbox("LLM provider", PROVIDERS, index=PROVIDERS.index("none"))
    model = st.text_input(
        "Model / deployment",
        placeholder=DEFAULT_MODELS.get(provider) or "provider-specific model id",
        help="Optional — leave empty to use the provider default. Required for 'openai-compatible'.",
    )
    temperature = st.slider(
        "Temperature", 0.0, 2.0, 0.2, 0.05, disabled=provider == "none"
    )
    st.divider()
    creds = _credentials_ui(provider)
    st.divider()
    st.caption(
        "Keys entered here are used for this run only — they never touch process environment."
    )
    st.divider()
    if st.button("Start over"):
        for key in (
            "result",
            "out_dir",
            "run_settings",
            "preview_df",
            "last_df",
            "last_profile",
            "run_llm_settings",
        ):
            st.session_state.pop(key, None)
        st.rerun()


def _run_analysis(
    uploaded,
    sheet: str,
    provider: str,
    model: str,
    temperature: float,
    env: dict[str, str],
) -> None:
    stem = Path(uploaded.name).stem
    base = Path("output") / f"ui_{stem}_{datetime.now(UTC):%Y%m%d-%H%M%S}"
    out_dir = unique_dir(base)
    suffix = Path(uploaded.name).suffix.lower() or ".csv"
    input_path = out_dir / f"input{suffix}"
    with open(input_path, "wb") as f:
        shutil.copyfileobj(uploaded, f)

    llm = None
    if provider != "none":
        try:
            llm = get_llm(
                provider, model=model or None, temperature=temperature, env=env
            )
        except Exception as exc:
            st.error(f"Could not initialise provider '{provider}': {exc}")
            return

    state = {
        "input_path": str(input_path),
        "sheet": sheet or "",
        "output_dir": str(out_dir),
        "provider": provider,
        "model": model or "",
    }

    with st.status("Running EDA graph…", expanded=True) as status:
        final = run_pipeline(
            state, llm, on_node=lambda node: st.write(f"done: {node.replace('_', ' ')}")
        )
        if final.get("error"):
            status.update(label="Analysis failed", state="error", expanded=True)
            st.error(final["error"])
            shutil.rmtree(out_dir, ignore_errors=True)
            return
        for w in final.get("warnings") or []:
            st.warning(w)
        status.update(label="Analysis complete", state="complete", expanded=False)

    df = final.pop("df", None)
    if df is not None:
        st.session_state["preview_df"] = df.head(100)
        st.session_state["last_df"] = df
    st.session_state["last_profile"] = final.get("profile") or {}
    st.session_state["run_llm_settings"] = {
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "env": env,
    }
    st.session_state["result"] = final
    st.session_state["out_dir"] = str(out_dir)
    for key in [k for k in st.session_state if str(k).startswith("chat_")]:
        del st.session_state[key]


def _render_chat_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("note"):
            st.caption(msg["note"])
        if msg.get("offline"):
            st.caption("offline — deterministic answer")
        if msg.get("sources"):
            st.caption("sources: " + ", ".join(msg["sources"]))


def _answer_chat(question: str) -> dict:
    """Answer one question against the last run's data; LLM failures go offline."""
    settings = st.session_state.get("run_llm_settings") or {}
    llm = None
    note = ""
    if settings.get("provider", "none") != "none":
        try:
            llm = get_llm(
                settings["provider"],
                model=settings.get("model") or None,
                temperature=settings.get("temperature", 0.2),
                env=settings.get("env") or {},
            )
        except Exception as exc:
            note = (
                f"LLM unavailable ({exc.__class__.__name__}); "
                "answered with the offline engine instead."
            )
    result = answer_question(
        question,
        st.session_state.get("last_df"),
        st.session_state.get("last_profile") or {},
        llm=llm,
    )
    if not note and result.get("warning"):
        note = result["warning"]
    if note:
        result["note"] = note
    return result


def _render_ask_tab(out_dir: str) -> None:
    df = st.session_state.get("last_df")
    profile = st.session_state.get("last_profile")
    if df is None or profile is None:
        st.info("Run an analysis first, then ask questions about the data.")
        return
    history: list[dict] = st.session_state.setdefault(f"chat_{out_dir}", [])
    for msg in history:
        _render_chat_message(msg)
    question = st.chat_input(
        "Ask a question about this dataset…", key=f"chat_input_{out_dir}"
    )
    if question:
        history.append({"role": "user", "content": question})
        _render_chat_message(history[-1])
        result = _answer_chat(question)
        entry = {
            "role": "assistant",
            "content": result["answer"],
            "sources": result.get("sources") or [],
            "offline": bool(result.get("offline")),
        }
        if result.get("note"):
            entry["note"] = result["note"]
        history.append(entry)
        _render_chat_message(entry)


def _render_agent_trace(res: dict) -> None:
    trace = res.get("agent_trace") or []
    if trace:
        with st.expander(f"Agent trace ({len(trace)} tool call(s))"):
            rows = [
                {
                    "round": entry.get("round", ""),
                    "tool": entry.get("tool", ""),
                    "args": ", ".join(
                        f"{k}={v!r}" for k, v in (entry.get("args") or {}).items()
                    )
                    or "—",
                    "summary": entry.get("summary", ""),
                }
                for entry in trace
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    anomalies = res.get("anomaly_reports") or []
    if anomalies:
        with st.expander(f"Anomaly drill-down ({len(anomalies)} report(s))"):
            for i, rep in enumerate(anomalies):
                st.markdown(
                    f"**`{rep.get('column', '?')}`** — "
                    f"{rep.get('n_outliers', '?')} outlier(s)"
                )
                if rep.get("comparison"):
                    st.caption(rep["comparison"])
                if rep.get("narrative"):
                    st.markdown(rep["narrative"])
                if i < len(anomalies) - 1:
                    st.divider()


def _render_results(res: dict, out_dir: str) -> None:
    profile = res["profile"]
    out = Path(out_dir)

    source = Path(res.get("input_path", ""))
    sheet = (res.get("sheet") or "").strip()
    source_label = f"`{source.name}`" + (f" · sheet `{sheet}`" if sheet else "")
    st.caption(f"Source: {source_label}")

    health = profile.get("health") or {}
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Rows", f"{profile['n_rows']:,}")
    m2.metric("Columns", profile["n_columns"])
    m3.metric("Duplicate rows", f"{profile['duplicate_rows']:,}")
    m4.metric("Columns w/ missing", len(profile.get("top_missing") or []))
    m5.metric("Memory", f"{profile['memory_mb']} MB")
    if health:
        m6.metric(
            "Data health",
            f"{health.get('score')}/100",
            f"grade {health.get('grade')}",
            delta_color="off",
        )

    if health:
        _health_banner(health)

    tab_ins, tab_rec, tab_charts, tab_cols, tab_prev, tab_ask, tab_dl = st.tabs(
        [
            "Insights",
            "Recommendations",
            "Charts",
            "Columns",
            "Preview",
            "Ask the data",
            "Downloads",
        ]
    )

    with tab_ins:
        st.markdown(res.get("insights") or "_no insights generated_")
        _render_agent_trace(res)

    with tab_rec:
        st.markdown(res.get("recommendations") or "_no recommendations generated_")

    with tab_charts:
        charts = res.get("charts") or []
        if not charts:
            st.info("No charts were applicable for this dataset.")
        for ch in charts:
            st.subheader(ch["title"])
            st.image(str(out / ch["path"]))
            st.caption(ch["caption"])

    with tab_cols:
        rows = []
        for name, c in profile["columns"].items():
            rows.append(
                {
                    "column": name,
                    "type": c["type"],
                    "dtype": c["dtype"],
                    "missing %": c["pct_missing"],
                    "unique": c["n_unique"],
                    "summary": column_summary(c),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        corr = profile.get("top_correlated_pairs") or []
        if corr:
            st.markdown("**Strongest correlations**")
            st.table(
                pd.DataFrame(corr).rename(
                    columns={
                        "col_a": "column a",
                        "col_b": "column b",
                        "pearson_r": "pearson r",
                    }
                )
            )

    with tab_prev:
        preview = st.session_state.get("preview_df")
        if preview is None:
            st.info("Run an analysis to see a preview of the loaded data.")
        else:
            st.dataframe(preview, use_container_width=True)
            st.caption(f"First {len(preview):,} row(s) of the loaded data.")

    with tab_ask:
        _render_ask_tab(out_dir)

    with tab_dl:
        html_report = Path(res.get("html_report_path") or "")
        if html_report.exists():
            st.download_button(
                "Download report.html (shareable)",
                html_report.read_bytes(),
                file_name="report.html",
                mime="text/html",
            )
        report = Path(res.get("report_path", ""))
        if report.exists():
            st.download_button(
                "Download report.md",
                report.read_bytes(),
                file_name="report.md",
                mime="text/markdown",
            )
        prof_json = out / "profile.json"
        if prof_json.exists():
            st.download_button(
                "Download profile.json",
                prof_json.read_bytes(),
                file_name="profile.json",
                mime="application/json",
            )
        st.caption(f"All artifacts are also saved on disk under `{out}`")

        charts_dir = out / "charts"
        pngs = sorted(charts_dir.glob("*.png")) if charts_dir.exists() else []
        if pngs:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                for png in pngs:
                    zf.write(png, png.name)
            st.download_button(
                f"Download all {len(pngs)} charts (.zip)",
                buf.getvalue(),
                file_name="charts.zip",
                mime="application/zip",
            )


st.title("sift — auto-EDA agent")
st.caption(
    "Upload a CSV, Excel or ODS file to get a statistical profile, visualizations, insights and recommendations."
)

uploaded = st.file_uploader(
    "Upload a data file", type=["csv", "xlsx", "xlsm", "xls", "ods"]
)

AUTO_SHEET = "(auto — combine matching sheets)"

if uploaded is not None:
    data = uploaded.getvalue()
    st.caption(f"`{uploaded.name}` · {_human_size(len(data))}")

    sheet = ""
    suffix = Path(uploaded.name).suffix.lower()
    if suffix in WORKBOOK_SUFFIXES:
        try:
            sheets = _cached_sheet_names(data, suffix)
        except LoadError as exc:
            st.caption(
                f"Could not preview sheets ({exc}); sheets are combined at run time."
            )
        else:
            if len(sheets) > 1:
                choice = st.selectbox("Sheet", [AUTO_SHEET, *sheets])
                sheet = "" if choice == AUTO_SHEET else choice
            else:
                st.caption(f"Workbook has one sheet: `{sheets[0]}`")

    settings_key = f"{uploaded.name}|{sheet}|{provider}|{model}|{temperature}"
    if st.session_state.get("run_settings") != settings_key:
        st.session_state["run_settings"] = settings_key
        st.session_state.pop("result", None)
        st.session_state.pop("out_dir", None)

    st.divider()
    if st.button("Run analysis", type="primary", use_container_width=True):
        _run_analysis(uploaded, sheet, provider, model, temperature, creds)

res = st.session_state.get("result")
if res:
    _render_results(res, st.session_state["out_dir"])
