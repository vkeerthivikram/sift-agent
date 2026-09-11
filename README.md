# sift-agent

Auto-EDA agent: upload a CSV, Excel workbook or ODS spreadsheet and get a statistical profile, visualizations, LLM-generated insights and recommendations. Powered by [LangGraph](https://langchain-ai.github.io/langgraph/).

```
CSV / Excel / ODS ──> load ──> statistical analysis ──> visualizations ──> insight extraction ──> recommendations ──> report
               │                                     └──────────── LLM (or offline heuristics) ────────┘
               └─ error ──────────────────────────────────────────────────────────────────────> stop
```

Supported input formats: `.csv`, `.xlsx`, `.xlsm`, `.xls`, `.ods`. Excel/ODS workbooks with multiple sheets are handled too — see [Multi-sheet workbooks](#multi-sheet-workbooks).

## Quickstart

```bash
uv sync                              # create .venv and install core deps
uv run sift run examples/customers.csv -p none   # full run, offline mode, no API keys
```

Want the web UI instead?

```bash
uv sync --extra ui
uv run sift-ui                       # opens http://localhost:8501
```

## What it does

- Input formats: CSV, Excel (`.xlsx`/`.xlsm`/`.xls`) and ODF spreadsheets (`.ods`), including multi-sheet workbooks (combined automatically when sheets share columns, or load one sheet by name/index).
- Statistical analysis of every column: means, quantiles, skew, IQR outliers, missing values, duplicates, constant columns, cardinality, top Pearson correlations, automatic ISO-datetime detection.
- Charts: missing-value bars, histograms with KDE, box plots, categorical counts, correlation heatmap, scatter plots of the strongest relationships. Each chart is a PNG on disk.
- Insight extraction: an LLM reads the computed profile plus chart captions and writes insights with real numbers. Offline mode uses deterministic heuristics instead.
- Recommendations: prioritized data cleaning, feature engineering and modeling next steps, specific to your columns.
- Multiple LLM backends: OpenAI, Anthropic, Azure OpenAI, AWS Bedrock, any OpenAI-compatible endpoint (Kilo Gateway, vLLM, Ollama, LM Studio, ...), plus an offline mode that needs no keys.
- Two frontends over the same pipeline: the `sift` CLI and the `sift-ui` web app.

## Requirements

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) for package and venv management

## Setup

```bash
uv sync            # core dependencies (CLI, pipeline)
uv sync --extra ui # adds Streamlit for the web UI
uv sync --dev      # adds pytest for the test suite
```

`uv run` uses the `.venv` automatically, so you never activate anything.

## Configuration

The app reads provider credentials from real environment variables (`os.environ`). It never loads a `.env` file by itself, and `.env.example` is only a template showing what to set. `.env` is gitignored, so secrets stay out of git.

To get variables into the process, pick one of these (no `export` needed):

```bash
# 1. Helper scripts source .env for you (see next section)
./scripts/start-backend.sh examples/customers.csv -p openai

# 2. uv loads the file into the process environment
uv run --env-file .env sift run data.csv -p openai

# 3. Load it in your current shell once per session
set -a; . ./.env; set +a
```

To start from the template:

```bash
cp .env.example .env   # then uncomment the block for your provider and fill in values
```

### Helper scripts

All scripts source `.env` through `scripts/env.sh` before running:

| script | what it does |
|---|---|
| `./scripts/start-backend.sh [csv] [-p provider] ...` | runs the CLI pipeline, defaults to `examples/customers.csv -p none` |
| `./scripts/start-ui.sh` | starts the Streamlit web UI |
| `./scripts/start-all.sh [csv] [-p provider]` | runs the pipeline once, then starts the UI |

## Running

### CLI

```bash
uv run sift run examples/customers.csv [options]

Options:
  -p, --provider TEXT     openai | anthropic | azure | bedrock | openai-compatible | none  [default: none]
  -m, --model TEXT        model / deployment name (provider-specific)
  -o, --output PATH       output directory (default: output/<name>_<timestamp>)
  -t, --temperature FLOAT LLM sampling temperature  [default: 0.2]
  -s, --sheet TEXT        Excel/ODS sheet: name or 0-based index  [default: combine matching sheets]

uv run sift providers     # list providers, required config and default models
```

#### Multi-sheet workbooks

By default, `load_data` combines every non-empty sheet whose columns match, adding a `sheet` provenance column that records each row's origin; empty sheets are skipped with a warning. Sheets with different columns are rejected with an error listing them — pick one instead:

```bash
uv run sift run sales.xlsx -p none            # combine 'Jan' + 'Feb' (same columns)
uv run sift run sales.xlsx -p none -s Feb     # only the 'Feb' sheet
uv run sift run sales.xlsx -p none -s 0       # first sheet (0-based index)
uv run sift run report.ods  -p none           # ODF spreadsheets work the same way
```

The CLI streams node-by-node progress, prints insights and recommendations, and reports where artifacts were written.

#### Examples

```bash
# Offline heuristics, no API keys (also the default)
uv run sift run examples/customers.csv -p none

# OpenAI
uv run --env-file .env sift run data.csv -p openai

# Anthropic
uv run --env-file .env sift run data.csv -p anthropic -m claude-sonnet-4-5

# Kilo Gateway (OpenAI-compatible), any model they host
uv run --env-file .env sift run data.csv -p openai-compatible -m zai-coding/glm-5.3-flash
# .env needs:
#   OPENAI_BASE_URL=https://api.kilo.ai/api/gateway
#   OPENAI_API_KEY=<your kilo key>

# Local server (Ollama, vLLM, LM Studio)
uv run --env-file .env sift run data.csv -p openai-compatible -m qwen2.5:14b
# OPENAI_BASE_URL=http://localhost:11434/v1 and OPENAI_API_KEY=EMPTY
```

### Web UI

```bash
uv run sift-ui          # or ./scripts/start-ui.sh to pick up .env
```

1. Pick an LLM provider in the sidebar (`none` for offline heuristics).
2. Paste credentials in the sidebar if you want. They apply to that run only and never touch the process environment. Leaving them empty falls back to environment variables.
3. Set the model name and temperature. For `openai-compatible` the model is required, e.g. `zai-coding/glm-5.3-flash`.
4. Upload a CSV, Excel or ODS file. For multi-sheet workbooks a **Sheet** dropdown appears, populated from the file itself — pick one sheet or keep *auto — combine matching sheets*.
5. Press **Run analysis**, then browse the tabs: Insights, Recommendations, Charts, Columns, **Preview** (first 100 rows of the loaded data) and Downloads (`report.md`, `profile.json`, all charts as a `.zip`).
6. Use **Start over** in the sidebar to clear results and run another file.

The UI look comes from the native Streamlit theme in `.streamlit/config.toml` (dark base, indigo primary).

## LLM providers

| provider | required configuration | default model |
|---|---|---|
| `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-4-5` |
| `azure` | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, optional `AZURE_OPENAI_API_VERSION`; `--model` is the deployment name (or set `AZURE_OPENAI_DEPLOYMENT`) | `gpt-4o` |
| `bedrock` | standard boto3 credential chain (`AWS_PROFILE`, or `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`), optional `AWS_REGION` | `anthropic.claude-3-5-sonnet-20241022-v2:0` |
| `openai-compatible` | `OPENAI_BASE_URL`, `OPENAI_API_KEY` (`EMPTY` for local servers); `--model` required | none, you must pass `-m` |
| `none` | no LLM, deterministic heuristic insights | not applicable |

## Output artifacts

Each run writes to `output/<dataset>_<timestamp>/`:

```
output/customers_20260908-171830/
├── report.md        # full markdown report: overview, missing values, column profiles,
│                    # correlations, embedded charts, insights, recommendations
├── profile.json     # machine-readable statistical profile
└── charts/          # PNG charts
    ├── 01_missing_values.png
    ├── 02_distributions.png     # histograms + KDE (numeric)
    ├── 03_boxplots.png          # box plots (numeric)
    ├── 04_categorical_counts.png
    ├── 05_correlation_heatmap.png
    └── 06_relationships.png     # scatters of strongest correlations
```

Charts that do not apply to the data (no missing values, too few numeric columns) are skipped automatically.

## How it works

The pipeline is a LangGraph `StateGraph` over a typed state (`EDAState`) carrying the DataFrame, statistical profile, chart references, insights and recommendations:

| node | what it does |
|---|---|
| `load_data` | reads the CSV/Excel/ODS input, resolves sheets, infers ISO datetime columns, enforces input size guards, short-circuits to END on error |
| `statistical_analysis` | computes the full profile, pure pandas/numpy |
| `generate_visualizations` | renders applicable charts with matplotlib/seaborn |
| `extract_insights` | LLM prompt over profile JSON + chart captions, heuristic fallback when offline |
| `recommendations` | LLM prompt over profile + insights, heuristic fallback when offline |
| `build_report` | assembles `report.md` and `profile.json` |

LLM prompts instruct the model to ground every claim in the provided numbers and cite column names, so nothing gets fabricated.

### Robustness notes

- LLM calls retry with backoff and fall back to deterministic heuristics on failure. The fallback is recorded as a warning and shown in the report.
- Output directories go through an atomic `unique_dir()` reservation, and chart rendering uses the matplotlib object API, so concurrent CLI runs and multiple UI sessions stay safe.
- Credentials can be passed per run without mutating `os.environ`, which matters when two Streamlit sessions use different providers.

## Project structure

```
sift-agent/
├── pyproject.toml          # deps, extras (ui, dev), console scripts: sift, sift-ui
├── .streamlit/config.toml  # native Streamlit theme for the web UI
├── .env.example            # template for provider credentials (copy to .env)
├── AGENTS.md               # agent/contributor entry point
├── docs/                   # architecture, conventions, testing notes
├── scripts/
│   ├── env.sh              # sources .env (used by all scripts below)
│   ├── start-backend.sh    # run the CLI pipeline
│   ├── start-ui.sh         # run the web UI
│   └── start-all.sh        # pipeline, then web UI
├── examples/customers.csv      # synthetic sample dataset
├── tests/                  # pytest suite
└── src/sift_agent/
    ├── cli.py              # `sift` CLI (typer + rich)
    ├── app.py              # Streamlit web UI
    ├── ui.py               # `sift-ui` launcher
    ├── config.py           # multi-provider LLM factory
    ├── graph.py            # LangGraph pipeline + prompts + heuristics
    ├── loader.py           # input loading: CSV, Excel/ODS sheets
    ├── analysis.py         # statistical profiling
    ├── visualize.py        # chart generation
    ├── report.py           # markdown report assembly
    └── state.py            # typed graph state
```

## Development

```bash
uv sync --dev              # install test deps
uv run pytest              # run the test suite
uvx ruff check .           # lint
uvx ruff format .          # format
```

Chart and graph tests force failures by monkeypatching builders, so keep `generate_charts(..., failures=...)` and the `MAX_INPUT_BYTES` / `MAX_ROWS` module globals patchable when you refactor.

## Extending

- New pipeline step: add a node function in `graph.py`, register it with `graph.add_node(...)` and wire an edge. State keys flow through `EDAState` in `state.py`.
- New provider: add a branch in `get_llm()` (`config.py`) returning any `langchain_core` chat model, then update `PROVIDERS`, `DEFAULT_MODELS` and `REQUIRED_ENV`.
- New chart: add a builder in `visualize.py` returning a `ChartRef`. Its caption is fed to the insight LLM automatically.

## Sample data

`examples/customers.csv` is a synthetic 615-row customer dataset (numeric, categorical, boolean, datetime and ID columns, injected missing values, duplicates, skew and correlated features) for trying the pipeline:

```bash
uv run sift run examples/customers.csv -p none
```
