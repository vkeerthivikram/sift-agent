# Testing

## Overview

How the test suite works and which seams it relies on.

## Running

- Tests live in `tests/` (pytest; dev deps via `uv sync --dev`), run with `uv run pytest`.

## Patchable Seams

Chart/graph tests force failures by monkeypatching builders — keep these patchable module globals:

- `generate_charts(..., failures=...)`
- `graph.MAX_INPUT_BYTES` / `MAX_ROWS`
