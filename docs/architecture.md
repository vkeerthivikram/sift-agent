# Architecture

## Overview

How the sift-agent pipeline fits together. Read this before changing pipeline flow, providers, or outputs.

## Pipeline

The pipeline is a LangGraph `StateGraph` over a typed state. Nodes run in order:

```
load_data → statistical_analysis → generate_visualizations → extract_insights → recommendations → build_report
     └─ error → END
```

## File Map

| File | Role |
|---|---|
| `src/sift_agent/state.py` | `EDAState` typed graph state (DataFrame, profile, chart refs, insights, recommendations) — the contract between all nodes |
| `src/sift_agent/graph.py` | LangGraph pipeline: node functions, LLM prompts, offline heuristic fallbacks, graph wiring |
| `src/sift_agent/loader.py` | Input loading: CSV, Excel (`.xlsx`/`.xlsm`/`.xls`) and ODS; multi-sheet combine or single-sheet selection |
| `src/sift_agent/analysis.py` | Statistical profiling (pure pandas/numpy) |
| `src/sift_agent/visualize.py` | Chart generation; builders return a `ChartRef` whose caption feeds the insight LLM |
| `src/sift_agent/report.py` | Assembles `report.md` and `profile.json` |
| `src/sift_agent/config.py` | Multi-provider LLM factory `get_llm()`; `PROVIDERS`, `DEFAULT_MODELS`, `REQUIRED_ENV` |
| `src/sift_agent/paths.py` | `unique_dir()` — atomically reserves a run output directory under concurrency |
| `src/sift_agent/cli.py` | `sift` CLI (typer + rich), streams node-by-node progress |
| `src/sift_agent/app.py` / `ui.py` | Streamlit web UI and `sift-ui` launcher |

## Run Outputs

Each run writes artifacts to `output/<dataset>_<timestamp>/` (`report.md`, `profile.json`, `charts/*.png`). Charts that don't apply to the data are skipped automatically.

## LLM Providers

- Providers: `openai`, `anthropic`, `azure`, `bedrock`, `openai-compatible`, and `none` (offline heuristics, also the default).
- Credentials come from environment variables (see `.env.example`) and are per-run: `get_llm(env={...})` overrides env vars for that call only.

## Extending

- New pipeline step → node function in `graph.py`, `graph.add_node(...)`, wire an edge; pass data through keys on `EDAState` in `state.py`.
- New input format → extend `loader.py` (`SUPPORTED_SUFFIXES`, engine selection, `load_table()`); `load_data` handles the rest.
- New provider → branch in `get_llm()` (`config.py`) returning a `langchain_core` chat model; update `PROVIDERS`, `DEFAULT_MODELS`, `REQUIRED_ENV`, and `.env.example`.
- New chart → builder in `visualize.py` returning a `ChartRef`; its caption is fed to the insight LLM automatically.
