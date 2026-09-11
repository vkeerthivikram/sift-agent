# sift-agent

Auto-EDA agent: CSV in → statistical profile → charts → LLM (or offline heuristic) insights → recommendations → markdown report. Built on LangGraph, pandas, matplotlib/seaborn, typer/rich CLI, and an optional Streamlit UI.

## Quick Reference

- **Package manager:** [uv](https://docs.astral.sh/uv/) — everything runs through `uv run`
- **Setup:** `uv sync` (core) · `uv sync --extra ui` (Streamlit UI extra)
- **Test:** `uv run pytest` — tests live in `tests/`
- **Lint:** `uvx ruff check .` before finishing · `uvx ruff format .` when making style-invasive changes
- **Run CLI (offline, no API keys):** `uv run sift run examples/customers.csv -p none`
- **Run UI:** `uv run sift-ui`
- **Helper scripts** (source `.env` automatically; mirrored in justfile):
  - `./scripts/start-backend.sh` (`<csv> -p <provider>` optional)
  - `./scripts/start-ui.sh`
  - `./scripts/start-all.sh`

## Critical Rules

- Keep the CLI and the Streamlit UI functional at all times — both are first-class frontends over the same graph.
- Concurrent runs must stay safe (multiple CLI processes, or multiple UI sessions as threads in one process).

## Detailed Instructions

For specific guidelines, see:

- [Architecture](docs/architecture.md) — pipeline, file map, run outputs, LLM providers, extending recipes
- [Conventions](docs/conventions.md) — layering rules, concurrency safety, robustness
- [Testing](docs/testing.md) — test suite and patchable seams
