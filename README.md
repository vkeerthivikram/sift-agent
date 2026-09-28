# sift-agent

Auto-EDA agent: upload a CSV, Excel workbook or ODS spreadsheet and get a statistical profile, visualizations, LLM-generated insights and recommendations. Powered by [LangGraph](https://langchain-ai.github.io/langgraph/).

```
CSV / Excel / ODS ──> load ──> statistical analysis ──> visualizations ──> insight extraction ──> recommendations ──> agentic investigation ──> anomaly drill-down ──> report
               │                                     └──────────────────── LLM (or offline heuristics) ────────────────────────────────┘                │
               └─ error ───────────────────────────────────────────────────────────────────────────────────────────> stop                                       │
                                                                 └──> sift ask / "Ask the data" chat (Q&A over the profile, no pipeline run needed)
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
- Statistical analysis of every column: means, quantiles (p05/p25/median/p75/p95), skew, kurtosis, mode, zeros/negatives, IQR outliers with examples, missing values, duplicates, constant columns, cardinality, top Pearson and Spearman correlations, bias-corrected Cramér's V between categoricals, numeric-by-category group means, datetime spans, co-missingness patterns, ID-like column detection, automatic ISO-datetime detection.
- Data-quality checks: numbers or dates stored as text, sentinel placeholder values (`n/a`, -999, …), stray/blank-only whitespace, inconsistent casing.
- **PII detection**: flags string columns that look like emails, phone numbers, US Social Security Numbers, credit-card numbers or IP addresses, so you know what to redact before sharing a report or dataset.
- **Data health score**: a 0-100 score with an A-F grade and a four-component breakdown (completeness, uniqueness, consistency, validity), shown in the CLI, the web UI and both reports.
- Charts: missing-value bars, histograms with KDE, box plots, categorical counts, correlation heatmap, scatter plots of the strongest relationships, numeric-by-category box plots, time-trend lines. Each chart is a PNG on disk.
- Insight extraction: an LLM reads the computed profile plus chart captions and writes insights with real numbers. Offline mode uses deterministic heuristics instead.
- Recommendations: prioritized data cleaning, feature engineering and modeling next steps, specific to your columns.
- **Shareable HTML report**: every run also writes a self-contained `report.html` — dark themed, charts embedded as base64, health panel up top — that opens offline in any browser and can be emailed to a client as-is.
- **Agentic mode**: after the fixed pipeline, an agent investigation loop probes the dataset with analysis tools (LLM tool-calling when a provider is configured, deterministic offline otherwise), an anomaly drill-down dissects the worst outlier column, and `sift ask` / the UI chat tab answer natural-language questions — see [Agentic mode](#agentic-mode).
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
uv run sift ask data.csv "how many rows are missing col_x?"   # Q&A, offline by default
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
5. Press **Run analysis**, then browse the tabs: Insights, Recommendations, Charts, Columns, **Preview** (first 100 rows of the loaded data) and Downloads (`report.md`, the shareable `report.html`, `profile.json`, all charts as a `.zip`). A **data-health banner** with score, grade and component breakdown sits above the tabs.
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

## Demoing it

The fastest client demo needs no API keys and shows every detection feature. `examples/messy_orders.csv` is a deliberately dirty dataset (duplicate rows, -999 sentinels, dates stored as text, mixed-case categories, whitespace notes, 13% missing delivery times, a 0.99-correlated revenue/cost pair):

```bash
uv sync --extra ui
uv run sift run examples/messy_orders.csv -p none   # watch the detectors fire
uv run sift-ui                                       # then do it live in the browser
```

The CLI prints the health score and grade up front; the UI shows a health banner, the flagged issues and the recommendations; `output/<run>/report.html` is the artifact to hand over. Verify the whole flow in one command with `uv run python scripts/verify_poc.py`.

## Agentic mode

Beyond the fixed pipeline, sift investigates the data like an analyst and answers your questions directly.

### Agent investigations

After recommendations, the `agentic_investigation` node runs a tool-calling loop over the dataset:

- **With an LLM configured** (`-p openai`, `-p anthropic`, ...): the model picks its own tools — correlation checks, group comparisons, missingness analysis, value scans, outlier inspection, time slices — up to four rounds, and writes up what it found.
- **Offline (`-p none`, the default)**: the same toolkit runs deterministically against the worst outlier column, the strongest correlated pair and the biggest numeric-by-category split.

Every call (round, tool, arguments, summary) lands in an **agent trace** rendered in `report.md` under "Agent investigations", in `report.html`, and in the UI's **Agent trace** expander.

### Anomaly drill-down

The `anomaly_drilldown` node takes the column with the most IQR outliers and dissects it: bounds, example values, how outliers shift the mean versus the rest, and which categories move most — 0-2 drill-down reports per run, shown in the "Anomaly drill-down" report section (skipped gracefully when there is nothing to drill into).

### Ask the data — CLI

`sift ask` answers natural-language questions about a file, no pipeline run needed:

```bash
uv run sift ask examples/messy_orders.csv "how many rows are missing delivery_days?"
# delivery_days: 75 of 550 values missing (13.64%).

uv run sift ask examples/messy_orders.csv "what is the max amount_usd?"
# amount_usd: maximum 23,501.25 (550 non-null values).

uv run sift ask examples/messy_orders.csv "is revenue correlated with cost?"
# Pearson r between revenue and cost = +0.997 (550 complete pairs).
```

Offline (the default) a deterministic intent engine answers questions about missingness, row counts, max/min/average, unique values, correlations and dtypes — every number computed from the data, never fabricated, with the touched columns listed under `sources:`. With a provider configured, questions go to the LLM grounded in the relevant profile facts, so free-form questions work too:

```bash
uv run --env-file .env sift ask examples/messy_orders.csv "why might delivery_days be missing?" -p openai
```

It takes the same options as `run` (`-p/--provider`, `-m/--model`, `-s/--sheet`, `-t/--temperature`).

### Ask the data — web UI

After a run, the **Ask the data** tab becomes a chat over the analyzed dataset: type a question, get a grounded answer with its sources; offline answers carry an `offline` badge, and if the LLM is unavailable the question is answered offline with a note. Chat history is kept per run, so re-running an analysis starts a fresh conversation.

## Output artifacts

Each run writes to `output/<dataset>_<timestamp>/`:

```
output/customers_20260908-171830/
├── report.md        # full markdown report: executive summary, data health,
│                    # overview, sample rows, missing values, column profiles,
│                    # data quality, potential PII, outliers, correlations (Pearson +
│                    # Spearman), categorical associations, group differences,
│                    # charts, insights, recommendations, agent investigations
│                    # (trace + findings), anomaly drill-down (when data exists)
├── report.html      # standalone shareable version — inline CSS, charts
│                    # embedded as base64, health panel; opens offline
├── profile.json     # machine-readable statistical profile (incl. health)
└── charts/          # PNG charts
    ├── 01_missing_values.png
    ├── 02_distributions.png     # histograms + KDE (numeric)
    ├── 03_boxplots.png          # box plots (numeric)
    ├── 04_categorical_counts.png
    ├── 05_correlation_heatmap.png
    ├── 06_relationships.png     # scatters of strongest correlations
    ├── 07_grouped_boxplots.png  # numeric split by low-cardinality category
    └── 08_time_trend.png        # numeric means per period over a datetime column
```

Charts that do not apply to the data (no missing values, too few numeric columns) are skipped automatically.

## How it works

The pipeline is a LangGraph `StateGraph` over a typed state (`EDAState`) carrying the DataFrame, statistical profile, chart references, insights, recommendations and the agent trace:

| node | what it does |
|---|---|
| `load_data` | reads the CSV/Excel/ODS input, resolves sheets, infers ISO datetime columns, enforces input size guards, short-circuits to END on error |
| `statistical_analysis` | computes the full profile plus the data-health score, pure pandas/numpy |
| `generate_visualizations` | renders applicable charts with matplotlib/seaborn |
| `extract_insights` | LLM prompt over profile JSON + chart captions, heuristic fallback when offline |
| `recommendations` | LLM prompt over profile + insights, heuristic fallback when offline |
| `agentic_investigation` | agent loop over the investigation toolkit: LLM tool-calling when a provider is configured, deterministic tool picks offline; every call recorded in the agent trace |
| `anomaly_drilldown` | dissects the top outlier column (IQR bounds, examples, mean shift vs the rest, category shifts); 0-2 reports per run |
| `build_report` | assembles `report.md`, the shareable `report.html` and `profile.json` |

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
    ├── investigate.py      # investigation toolkit for the agent loop (tools + specs)
    ├── qa.py               # dataset Q&A: context retrieval + offline/LLM answers
    ├── scoring.py          # data health score (0-100, A-F, components)
    ├── visualize.py        # chart generation
    ├── report.py           # markdown report assembly
    ├── html_report.py      # standalone shareable HTML report
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

Two synthetic datasets ship with the repo:

- `examples/customers.csv` — 615 rows (numeric, categorical, boolean, datetime and ID columns, injected missing values, duplicates, skew and correlated features) for a normal run.
- `examples/messy_orders.csv` — 550 deliberately dirty order records for demoing the detection features: 10 duplicate rows, 8 `-999` sentinels, non-ISO text dates, mixed-case regions, whitespace-only notes, 13.6% missing delivery times and a 0.99-correlated revenue/cost pair.

```bash
uv run sift run examples/customers.csv -p none
uv run sift run examples/messy_orders.csv -p none
```
