# Conventions

## Overview

Code-level rules that apply whenever editing source under `src/sift_agent/`.

## Language & Layout

- Python ≥ 3.11. Source lives in `src/sift_agent/` (src layout, hatchling build).

## Layering Rules

- Statistical code (`analysis.py`) stays pure pandas/numpy — no LLM calls there.
- LLM nodes must ground every claim in the computed profile and cite column names; offline heuristic fallbacks live in `graph.py` and must stay deterministic.

## Concurrency Safety

Multiple CLI processes and multiple UI sessions (threads in one process) must coexist:

- Default output dirs go through `unique_dir()` (`paths.py`) — atomic `mkdir`, never `mkdir(exist_ok=True)` on a timestamped name.
- Chart code uses the matplotlib object API (`Figure` + `FigureCanvasAgg`), never `matplotlib.pyplot` — the pyplot figure manager is not thread-safe.
- Credentials are per-run: `get_llm(env={...})` overrides env vars for that call only; never mutate `os.environ` in session/UI code.

## Robustness

- LLM nodes go through `graph._invoke_llm` (3 attempts, backoff) and fall back to deterministic heuristics on failure, recording a warning in state.
- Non-fatal issues (LLM fallback, skipped charts) accumulate in `EDAState["warnings"]` (reducer: append) and are rendered in the report's Warnings section.
- `load_data` enforces `MAX_INPUT_BYTES` / `MAX_ROWS` input guards with clear error messages; input parsing lives in `loader.py` and raises `LoadError` for every recoverable condition.
- Logging via `logging.getLogger("sift_agent.<module>")`; never silent except-swallow — chart builders log and report failures via the warnings channel.
