"""Guarded pandas analysis tools for the agentic investigation loop.

Each tool inspects one or two columns of a DataFrame and returns a JSON-safe
dict that always carries a deterministic ``summary`` sentence containing the
real numbers. Tools never raise on bad input: an unknown column, an
unexpected dtype, or malformed arguments produce ``{"error": ..., "summary": ...}``.
Dispatch happens through :func:`run_tool`, which validates arguments against
the tool signatures and converts unexpected crashes into error dicts.
"""

from __future__ import annotations

import inspect
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("sift_agent.investigate")

MAX_GROUPS = 12  # group_compare: categories shown, ranked by row count
MAX_TOP_VALUES = 8  # value_scan: most frequent values listed
MAX_OUTLIER_EXAMPLES = 5  # outlier_inspect: example outlier values
MAX_OUTLIER_COMPARISONS = 3  # outlier_inspect: other numeric columns compared
MAX_MISSINGNESS_SPLITS = 3  # missingness_analysis: categorical columns ranked
MAX_SPLIT_CARDINALITY = 20  # missingness_analysis: candidate column cardinality
MAX_TIME_PERIODS = 10  # time_slice: periods listed
MIN_CORRELATION_ROWS = 30  # below this, correlation_check adds a caution note
TIME_FREQS = frozenset({"D", "W", "M", "Q", "Y"})

# Same concept as analysis.SENTINEL_STRINGS, defined locally so this module
# never imports private helpers from analysis.
SENTINEL_STRINGS = frozenset(
    {"", "n/a", "na", "-", "?", "null", "none", "nan", "missing", "unknown"}
)


def _py(v):
    """Convert numpy/pandas scalars into JSON-safe Python values."""
    if v is None:
        return None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, np.bool_):
        return bool(v)
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _err(message: str) -> dict:
    """Uniform error payload; the summary repeats the error message."""
    return {"error": message, "summary": message}


def _fmt(v) -> str:
    """Compact rendering of a numeric detail in summary sentences."""
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:,.4g}"
    return str(v)


def _check_df(df) -> dict | None:
    """Guard shared by every tool: a non-empty pandas DataFrame."""
    if not isinstance(df, pd.DataFrame):
        return _err(f"expected a pandas DataFrame, got {type(df).__name__}")
    if df.empty:
        return _err("the DataFrame is empty")
    return None


def _column_error(df: pd.DataFrame, column) -> dict | None:
    if not isinstance(column, str):
        return _err(f"column must be a string, got {type(column).__name__}")
    if column not in df.columns:
        return _err(f"column {column!r} not found in the DataFrame")
    return None


def _is_numeric(s: pd.Series) -> bool:
    """Numeric and not boolean (matches the profiling rules in analysis.py)."""
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _numeric_error(df: pd.DataFrame, column) -> dict | None:
    err = _column_error(df, column)
    if err:
        return err
    if not _is_numeric(df[column]):
        return _err(f"column {column!r} must be numeric, got dtype {df[column].dtype}")
    return None


def _categorical_error(df: pd.DataFrame, column) -> dict | None:
    err = _column_error(df, column)
    if err:
        return err
    if _is_numeric(df[column]):
        return _err(
            f"column {column!r} must be categorical (non-numeric), "
            f"got dtype {df[column].dtype}"
        )
    return None


def correlation_check(df: pd.DataFrame, column_a: str, column_b: str) -> dict:
    """Pearson and Spearman correlations of two numeric columns on complete rows."""
    err = _check_df(df)
    if err:
        return err
    for column in (column_a, column_b):
        err = _numeric_error(df, column)
        if err:
            return err
    a, b = df[column_a], df[column_b]
    mask = a.notna() & b.notna()
    n = int(mask.sum())
    if n < 2:
        return _err(
            f"need at least 2 rows where both {column_a!r} and {column_b!r} "
            f"are non-null, got {n}"
        )
    sa, sb = a[mask], b[mask]
    if sa.nunique() < 2 or sb.nunique() < 2:
        return _err(
            f"correlation is undefined: each column needs at least 2 distinct "
            f"values across the {n} complete rows"
        )
    pearson = _py(round(float(sa.corr(sb)), 4))
    # Spearman = Pearson on average ranks (scipy-free, tie-safe).
    spearman = _py(round(float(sa.rank().corr(sb.rank())), 4))
    caution = " (low n, treat with caution)" if n < MIN_CORRELATION_ROWS else ""
    summary = (
        f"Pearson r={pearson:.3f} and Spearman r={spearman:.3f} between "
        f"'{column_a}' and '{column_b}' over {n} complete rows{caution}"
    )
    return {
        "summary": summary,
        "column_a": column_a,
        "column_b": column_b,
        "pearson_r": pearson,
        "spearman_r": spearman,
        "n": n,
    }


def group_compare(
    df: pd.DataFrame, numeric_column: str, categorical_column: str
) -> dict:
    """Per-group n/mean/median/std of a numeric column across a categorical one."""
    err = _check_df(df)
    if err:
        return err
    err = _numeric_error(df, numeric_column)
    if err:
        return err
    err = _categorical_error(df, categorical_column)
    if err:
        return err
    sub = df[[numeric_column, categorical_column]].dropna()
    if sub.empty:
        return _err(
            f"no rows where both {numeric_column!r} and "
            f"{categorical_column!r} are non-null"
        )
    agg = sub.groupby(categorical_column, observed=True)[numeric_column].agg(
        ["count", "mean", "median", "std"]
    )
    records = [
        {
            "category": str(category),
            "n": int(count),
            "mean": _py(round(float(mean), 4)),
            "median": _py(round(float(median), 4)),
            "std": _py(round(float(std), 4)) if pd.notna(std) else None,
        }
        for category, count, mean, median, std in agg.itertuples(name=None)
    ]
    records.sort(key=lambda r: (-r["n"], r["category"]))
    shown = records[:MAX_GROUPS]

    s = df[numeric_column].dropna()
    overall = {
        "n": len(s),
        "mean": _py(round(float(s.mean()), 4)),
        "median": _py(round(float(s.median()), 4)),
        "std": _py(round(float(s.std()), 4)) if len(s) >= 2 else None,
    }

    biggest_gap = None
    with_mean = [r for r in shown if r["mean"] is not None]
    if len(with_mean) >= 2:
        high = max(with_mean, key=lambda r: r["mean"])
        low = min(with_mean, key=lambda r: r["mean"])
        biggest_gap = {
            "high_group": high["category"],
            "low_group": low["category"],
            "gap": _py(round(high["mean"] - low["mean"], 4)),
        }

    n_total = len(records)
    if biggest_gap:
        summary = (
            f"'{numeric_column}' mean differs by {_fmt(biggest_gap['gap'])} between "
            f"'{biggest_gap['high_group']}' ({_fmt(high['mean'])}) and "
            f"'{biggest_gap['low_group']}' ({_fmt(low['mean'])}) across "
            f"{len(shown)} of {n_total} '{categorical_column}' groups "
            f"({overall['n']} non-null values)"
        )
    else:
        only = shown[0]
        summary = (
            f"'{numeric_column}' has a single '{categorical_column}' group "
            f"('{only['category']}', n={only['n']}, mean {_fmt(only['mean'])}); "
            f"no between-group gap to report"
        )
    return {
        "summary": summary,
        "numeric_column": numeric_column,
        "categorical_column": categorical_column,
        "n_groups_total": n_total,
        "n_groups_shown": len(shown),
        "groups": shown,
        "overall": overall,
        "biggest_gap": biggest_gap,
    }


def missingness_analysis(df: pd.DataFrame, column: str) -> dict:
    """Where a column's missing values concentrate, plus sentinel placeholders."""
    err = _check_df(df)
    if err:
        return err
    err = _column_error(df, column)
    if err:
        return err
    s = df[column]
    n_rows = len(df)
    mask = s.isna()
    n_missing = int(mask.sum())
    overall_raw = float(mask.mean() * 100)
    pct = round(overall_raw, 2)
    if _is_numeric(s):
        n_sentinel = 0
    else:
        vals = s.dropna().astype(str).str.strip().str.lower()
        n_sentinel = int(vals.isin(SENTINEL_STRINGS).sum())

    splits: list[dict] = []
    if n_missing:
        candidates = [
            c
            for c in df.columns
            if c != column
            and not _is_numeric(df[c])
            and 0 < df[c].nunique(dropna=True) <= MAX_SPLIT_CARDINALITY
        ]
        for cand in candidates:
            frame = pd.DataFrame({"m": mask.to_numpy(), "c": df[cand].to_numpy()})
            agg = frame.groupby("c", observed=True)["m"].agg(["size", "mean"])
            best_category, best_n, best_rate, best_gap = None, 0, 0.0, -1.0
            for category, size, rate in sorted(
                agg.itertuples(name=None), key=lambda t: str(t[0])
            ):
                rate_pct = float(rate) * 100
                gap = abs(rate_pct - overall_raw)
                if gap > best_gap:
                    best_category, best_n = str(category), int(size)
                    best_rate, best_gap = rate_pct, gap
            if best_gap > 0:
                splits.append(
                    {
                        "column": cand,
                        "category": best_category,
                        "n": best_n,
                        "rate": round(best_rate, 2),
                        "overall_rate": pct,
                        "gap": round(best_gap, 2),
                    }
                )
        splits.sort(key=lambda sp: (-sp["gap"], sp["column"], sp["category"]))
        splits = splits[:MAX_MISSINGNESS_SPLITS]
    biggest_divergence = splits[0] if splits else None

    parts = [f"'{column}' is missing in {n_missing}/{n_rows} rows ({pct}%)"]
    parts.append(
        f"{n_sentinel} sentinel-looking values"
        if n_sentinel
        else "no sentinel-looking values"
    )
    if not n_missing:
        parts.append("no missing values to explain")
    elif biggest_divergence:
        d = biggest_divergence
        parts.append(
            f"missingness diverges most by '{d['column']}': {d['rate']}% when "
            f"'{d['category']}' vs {d['overall_rate']}% overall"
        )
    return {
        "summary": "; ".join(parts),
        "column": column,
        "n_rows": n_rows,
        "n_missing": n_missing,
        "pct_missing": pct,
        "n_sentinel": n_sentinel,
        "splits": splits,
        "biggest_divergence": biggest_divergence,
    }


def value_scan(df: pd.DataFrame, column: str) -> dict:
    """Value inventory of one column: dtype, top values, and shape stats."""
    err = _check_df(df)
    if err:
        return err
    err = _column_error(df, column)
    if err:
        return err
    s = df[column]
    dtype = str(s.dtype)
    n_unique = int(s.nunique(dropna=True))
    top_values = [
        {"value": str(k), "count": int(v)}
        for k, v in s.value_counts(dropna=True).head(MAX_TOP_VALUES).items()
    ]
    result: dict = {
        "column": column,
        "dtype": dtype,
        "n_unique": n_unique,
        "top_values": top_values,
    }
    valid = s.dropna()
    numeric = _is_numeric(s)
    string_like = (
        not numeric
        and not pd.api.types.is_datetime64_any_dtype(s)
        and not pd.api.types.is_bool_dtype(s)
    )
    if numeric:
        result.update(
            {
                "min": _py(round(float(valid.min()), 4)) if len(valid) else None,
                "max": _py(round(float(valid.max()), 4)) if len(valid) else None,
                "mean": _py(round(float(valid.mean()), 4)) if len(valid) else None,
            }
        )
    elif string_like:
        lengths = valid.astype(str).str.len()
        result.update(
            {
                "min_length": int(lengths.min()) if len(lengths) else None,
                "max_length": int(lengths.max()) if len(lengths) else None,
            }
        )

    top_txt = (
        f"top '{top_values[0]['value']}' ({top_values[0]['count']} rows)"
        if top_values
        else "no non-null values"
    )
    if numeric:
        summary = (
            f"'{column}' ({dtype}): {n_unique} unique values, {top_txt}, "
            f"min {_fmt(result['min'])}, max {_fmt(result['max'])}, "
            f"mean {_fmt(result['mean'])}"
        )
    elif string_like:
        summary = (
            f"'{column}' ({dtype}): {n_unique} unique values, {top_txt}, "
            f"length {_fmt(result['min_length'])}-{_fmt(result['max_length'])} chars"
        )
    else:
        summary = f"'{column}' ({dtype}): {n_unique} unique values, {top_txt}"
    result["summary"] = summary
    return result


def outlier_inspect(df: pd.DataFrame, column: str) -> dict:
    """IQR outlier audit of a numeric column and what else changes with it."""
    err = _check_df(df)
    if err:
        return err
    err = _numeric_error(df, column)
    if err:
        return err
    s = df[column].dropna()
    n_valid = len(s)
    if n_valid == 0:
        return _err(f"column {column!r} has no non-null values")
    q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
    iqr = q3 - q1
    result: dict = {
        "column": column,
        "n_valid": n_valid,
        "n_outliers": 0,
        "pct_outliers": 0.0,
        "examples": [],
        "comparisons": [],
    }
    if iqr == 0:
        result.update({"lower_bound": None, "upper_bound": None})
        result["summary"] = (
            f"no IQR outliers in '{column}' (0 of {n_valid} non-null values); "
            f"the IQR is zero so the 1.5x fences are undefined"
        )
        return result

    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    full = df[column]
    out_mask = full.notna() & ((full < lower) | (full > upper))
    rest_mask = full.notna() & ~out_mask
    n_outliers = int(out_mask.sum())
    out_vals = s[(s < lower) | (s > upper)]
    median = float(s.median())
    ranked = out_vals.reindex(
        out_vals.sub(median).abs().sort_values(ascending=False).index
    )
    examples = [_py(round(float(v), 6)) for v in ranked.head(MAX_OUTLIER_EXAMPLES)]

    comparisons: list[dict] = []
    if n_outliers:
        others = [c for c in df.columns if c != column and _is_numeric(df[c])][
            :MAX_OUTLIER_COMPARISONS
        ]
        for other in others:
            mo = _py(df.loc[out_mask, other].mean())
            mr = _py(df.loc[rest_mask, other].mean())
            shift = None
            if mo is not None and mr is not None and mr != 0:
                shift = _py(round((mo - mr) / abs(mr) * 100, 2))
            comparisons.append(
                {
                    "column": other,
                    "mean_outliers": _py(round(mo, 4)) if mo is not None else None,
                    "mean_rest": _py(round(mr, 4)) if mr is not None else None,
                    "shift_pct": shift,
                }
            )

    pct = round(n_outliers / n_valid * 100, 2)
    parts = [
        f"{n_outliers} IQR outlier(s) in '{column}' ({pct}% of {n_valid} non-null "
        f"values), bounds [{_fmt(round(lower, 4))}, {_fmt(round(upper, 4))}]"
    ]
    if examples:
        parts.append("most extreme: " + ", ".join(_fmt(v) for v in examples))
    shift_txt = [
        f"{c['column']} {c['shift_pct']:+.2f}%"
        for c in comparisons
        if c["shift_pct"] is not None
    ]
    if shift_txt:
        parts.append("outlier rows shift means: " + "; ".join(shift_txt))
    result.update(
        {
            "lower_bound": _py(round(lower, 4)),
            "upper_bound": _py(round(upper, 4)),
            "n_outliers": n_outliers,
            "pct_outliers": pct,
            "examples": examples,
            "comparisons": comparisons,
            "summary": "; ".join(parts),
        }
    )
    return result


def time_slice(df: pd.DataFrame, date_column: str, freq: str = "M") -> dict:
    """Row counts and first-numeric means per calendar period of a date column."""
    err = _check_df(df)
    if err:
        return err
    if not isinstance(freq, str) or freq not in TIME_FREQS:
        return _err(f"freq must be one of {sorted(TIME_FREQS)}, got {freq!r}")
    err = _column_error(df, date_column)
    if err:
        return err
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        return _err(
            f"column {date_column!r} must have datetime dtype, "
            f"got {df[date_column].dtype}"
        )
    s = df[date_column]
    if getattr(s.dtype, "tz", None) is not None:
        s = s.dt.tz_localize(None)
    mask = s.notna().to_numpy()
    if not mask.any():
        return _err(f"column {date_column!r} has no datetime values")
    periods = s.dt.to_period(freq).to_numpy()[mask]
    frame = pd.DataFrame({"p": periods})
    num_col = next((c for c in df.columns if _is_numeric(df[c])), None)
    means = None
    if num_col is not None:
        frame["v"] = df.loc[mask, num_col].to_numpy()
        means = frame.groupby("p", observed=True)["v"].mean()
    counts = frame.groupby("p", observed=True).size()

    rows = []
    for period, n in counts.items():
        mean = None
        if means is not None:
            m = means.get(period)
            if m is not None and pd.notna(m):
                mean = _py(round(float(m), 4))
        rows.append({"period": period, "n": int(n), "mean": mean})
    rows.sort(key=lambda r: (-r["n"], r["period"]))
    shown = [
        {"period": str(r["period"]), "n": r["n"], "mean": r["mean"]}
        for r in rows[:MAX_TIME_PERIODS]
    ]

    top = shown[0]
    if num_col is not None and top["mean"] is not None:
        detail = f", mean '{num_col}' {_fmt(top['mean'])}"
    elif num_col is not None:
        detail = f" (mean of '{num_col}' undefined there)"
    else:
        detail = " (no numeric column to average)"
    summary = (
        f"'{date_column}' spans {len(rows)} {freq} period(s); "
        f"busiest {top['period']} with {top['n']} rows{detail}"
    )
    return {
        "summary": summary,
        "date_column": date_column,
        "freq": freq,
        "numeric_column": num_col,
        "n_periods": len(rows),
        "periods": shown,
    }


TOOL_SPECS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "correlation_check",
            "description": (
                "Measure how two numeric columns move together: Pearson r "
                "(linear) and Spearman r (monotonic rank) computed on "
                "pairwise-complete rows, with the row count used and a caution "
                "note when n < 30. Use for questions like 'are X and Y "
                "related?'. Both columns must be numeric (not boolean)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column_a": {
                        "type": "string",
                        "description": "First numeric column name.",
                    },
                    "column_b": {
                        "type": "string",
                        "description": "Second numeric column name.",
                    },
                },
                "required": ["column_a", "column_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "group_compare",
            "description": (
                "Compare a numeric column across the categories of a "
                "categorical column: per-group n/mean/median/std for the top "
                "12 categories by row count, overall stats, and the biggest "
                "mean gap between any two shown groups. Use for 'does X "
                "differ by group Z?'. numeric_column must be numeric; "
                "categorical_column must be non-numeric."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "numeric_column": {
                        "type": "string",
                        "description": "Numeric column to aggregate.",
                    },
                    "categorical_column": {
                        "type": "string",
                        "description": "Categorical column defining the groups.",
                    },
                },
                "required": ["numeric_column", "categorical_column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "missingness_analysis",
            "description": (
                "Find where a column's missing values concentrate: n_missing, "
                "pct_missing, the count of sentinel-looking placeholder "
                "strings ('n/a', '', 'null', ...), and the top 3 categorical "
                "columns whose category values diverge most from the overall "
                "missingness rate. Use for 'why or where is X missing?'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Column to analyze for missingness.",
                    },
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "value_scan",
            "description": (
                "Inspect one column's value distribution: dtype, n_unique, "
                "the top 8 most frequent values with counts, plus numeric "
                "min/max/mean or string min/max length. Use for 'what values "
                "does column X contain?'. Works for any dtype."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Column to scan.",
                    },
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "outlier_inspect",
            "description": (
                "Examine the IQR outliers of a numeric column: the 1.5xIQR "
                "bounds, outlier count and percentage, 5 example outlier "
                "values (most extreme first), and the mean shift of outlier "
                "rows vs the rest for up to 3 other numeric columns. Use for "
                "'what is going on with the extreme values of X and what "
                "else changes with them?'. Column must be numeric."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Numeric column to audit for outliers.",
                    },
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "time_slice",
            "description": (
                "Aggregate rows into calendar periods by a datetime column: "
                "row counts and the mean of the first numeric column per "
                "period, listing the top 10 periods by row count. Use for "
                "'how does activity trend over time?'. date_column must have "
                "datetime dtype; freq must be one of D (day), W (week), "
                "M (month), Q (quarter), Y (year), default M."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date_column": {
                        "type": "string",
                        "description": "Datetime column to slice by.",
                    },
                    "freq": {
                        "type": "string",
                        "enum": sorted(TIME_FREQS),
                        "description": "Period frequency (default: M).",
                    },
                },
                "required": ["date_column"],
            },
        },
    },
]

_TOOL_FUNCS: dict[str, tuple] = {
    "correlation_check": (correlation_check, ("column_a", "column_b")),
    "group_compare": (group_compare, ("numeric_column", "categorical_column")),
    "missingness_analysis": (missingness_analysis, ("column",)),
    "value_scan": (value_scan, ("column",)),
    "outlier_inspect": (outlier_inspect, ("column",)),
    "time_slice": (time_slice, ("date_column",)),
}


def run_tool(df: pd.DataFrame, name: str, args: dict) -> dict:
    """Validated dispatch of a named investigate tool; never raises."""
    if not isinstance(name, str) or name not in _TOOL_FUNCS:
        return _err(f"unknown tool {name!r}; expected one of {sorted(_TOOL_FUNCS)}")
    fn, required = _TOOL_FUNCS[name]
    if not isinstance(args, dict):
        return _err(f"args for tool {name!r} must be a dict, got {type(args).__name__}")
    allowed = set(inspect.signature(fn).parameters) - {"df"}
    for key in args:
        if key not in allowed:
            return _err(f"unexpected argument {key!r} for tool {name!r}")
    for key in required:
        if key not in args:
            return _err(f"missing required argument {key!r} for tool {name!r}")
        if not isinstance(args[key], str):
            return _err(
                f"argument {key!r} for tool {name!r} must be a string, "
                f"got {type(args[key]).__name__}"
            )
    try:
        return fn(df, **dict(args))
    except Exception as exc:
        logger.warning("investigate tool %s crashed: %s", name, exc)
        return _err(f"tool {name!r} failed unexpectedly: {exc}")
