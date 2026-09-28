# Architecture

## Overview

How the sift-agent pipeline fits together. Read this before changing pipeline flow, providers, or outputs.

## Pipeline

The pipeline is a LangGraph `StateGraph` over a typed state. Nodes run in order:

```
load_data → statistical_analysis → generate_visualizations → extract_insights → recommendations → agentic_investigation → anomaly_drilldown → build_report
     └─ error → END
```

`sift ask` and the UI "Ask the data" chat tab bypass the graph: they load + profile the file (or reuse the run's profile) and answer through `qa.answer_question` directly.

## File Map

| File | Role |
|---|---|
| `src/sift_agent/state.py` | `EDAState` typed graph state (DataFrame, profile, chart refs, insights, recommendations) — the contract between all nodes |
| `src/sift_agent/graph.py` | LangGraph pipeline: node functions, LLM prompts, offline heuristic fallbacks, agent investigation loop + anomaly drill-down, graph wiring |
| `src/sift_agent/loader.py` | Input loading: CSV, Excel (`.xlsx`/`.xlsm`/`.xls`) and ODS; multi-sheet combine or single-sheet selection |
| `src/sift_agent/analysis.py` | Statistical profiling (pure pandas/numpy) |
| `src/sift_agent/investigate.py` | Investigation toolkit for the agent loop: six pandas tools (`correlation_check`, `group_compare`, `missingness_analysis`, `value_scan`, `outlier_inspect`, `time_slice`) plus `TOOL_SPECS` and the `run_tool()` dispatch used by `agentic_investigation` / `anomaly_drilldown` |
| `src/sift_agent/scoring.py` | Data health score (0-100 + A-F grade + component breakdown) computed from the profile; stored as `profile["health"]` |
| `src/sift_agent/visualize.py` | Chart generation; builders return a `ChartRef` whose caption feeds the insight LLM |
| `src/sift_agent/report.py` | Assembles `report.md` (executive summary, health section) and `profile.json` |
| `src/sift_agent/html_report.py` | Standalone shareable `report.html` — inline CSS, base64-embedded charts, health panel |
| `src/sift_agent/config.py` | Multi-provider LLM factory `get_llm()`; `PROVIDERS`, `DEFAULT_MODELS`, `REQUIRED_ENV` |
| `src/sift_agent/qa.py` | Dataset Q&A: `build_context()` retrieval of profile facts relevant to a question + `answer_question()` (deterministic intent engine offline, one grounded LLM call online) — powers `sift ask` and the UI chat tab |
| `src/sift_agent/paths.py` | `unique_dir()` — atomically reserves a run output directory under concurrency |
| `src/sift_agent/cli.py` | `sift` CLI (typer + rich): `run` streams node-by-node progress, `ask` answers questions about a file, `providers` lists configuration |
| `src/sift_agent/app.py` / `ui.py` | Streamlit web UI (results tabs, "Ask the data" chat, agent-trace expander) and `sift-ui` launcher |

## Run Outputs

Each run writes artifacts to `output/<dataset>_<timestamp>/`: `report.md`, `report.html` (self-contained, charts embedded), `profile.json` (includes the `health` block), and `charts/*.png`. Charts that don't apply to the data are skipped automatically. When the agentic nodes ran, `report.md`/`report.html` additionally carry "Agent investigations" (trace + findings) and "Anomaly drill-down" sections.

## LLM Providers

- Providers: `openai`, `anthropic`, `azure`, `bedrock`, `openai-compatible`, and `none` (offline heuristics, also the default).
- Credentials come from environment variables (see `.env.example`) and are per-run: `get_llm(env={...})` overrides env vars for that call only.

## Extending

- New pipeline step → node function in `graph.py`, `graph.add_node(...)`, wire an edge; pass data through keys on `EDAState` in `state.py`.
- New investigation tool → builder in `investigate.py` returning a JSON-safe dict with a `summary`, plus a matching `TOOL_SPECS` entry so the agent loop can call it through `run_tool()`.
- New question intent → handler in `qa.py`; `answer_question()` tries intents offline before falling back to the retrieved-context answer.
- New input format → extend `loader.py` (`SUPPORTED_SUFFIXES`, engine selection, `load_table()`); `load_data` handles the rest.
- New provider → branch in `get_llm()` (`config.py`) returning a `langchain_core` chat model; update `PROVIDERS`, `DEFAULT_MODELS`, `REQUIRED_ENV`, and `.env.example`.
- New chart → builder in `visualize.py` returning a `ChartRef`; its caption is fed to the insight LLM automatically.
- Health-score tuning → `scoring.py` (`COMPONENT_WEIGHTS`, check-name sets); the HTML/CLI/UI all read `profile["health"]`, so no other file changes.
