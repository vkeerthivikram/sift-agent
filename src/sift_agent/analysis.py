"""Statistical profiling of a pandas DataFrame (pure computation, no LLM)."""

from __future__ import annotations

import numpy as np
import pandas as pd

MAX_PROFILE_COLUMNS = 100


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


def _iqr_fences(s: pd.Series) -> tuple[float, float] | None:
    """IQR outlier fences of non-null values, or None when undefined."""
    if s.empty:
        return None
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return None
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def iqr_outlier_pct(s: pd.Series) -> float:
    """Percentage of non-null values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]."""
    s = s.dropna()
    fences = _iqr_fences(s)
    if fences is None:
        return 0.0
    lo, hi = fences
    return round(float(((s < lo) | (s > hi)).mean() * 100), 2)


def iqr_outlier_count(s: pd.Series) -> int:
    """Number of non-null values outside the IQR fences."""
    s = s.dropna()
    fences = _iqr_fences(s)
    if fences is None:
        return 0
    lo, hi = fences
    return int(((s < lo) | (s > hi)).sum())


def top_pairs_from_corr(
    corr: pd.DataFrame, threshold: float = 0.3, limit: int = 10
) -> list[dict]:
    """Strongest absolute Pearson correlations from a precomputed corr matrix."""
    cols = list(corr.columns)
    pairs: list[dict] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.notna(r) and abs(r) >= threshold:
                pairs.append(
                    {
                        "col_a": cols[i],
                        "col_b": cols[j],
                        "pearson_r": round(float(r), 3),
                    }
                )
    pairs.sort(key=lambda p: abs(p["pearson_r"]), reverse=True)
    return pairs[:limit]


def top_correlations(
    df: pd.DataFrame, threshold: float = 0.3, limit: int = 10
) -> list[dict]:
    """Strongest absolute Pearson correlations between numeric column pairs."""
    num = df.select_dtypes(include=[np.number])
    num = num.loc[:, [c for c in num.columns if num[c].nunique(dropna=True) > 1]]
    if num.shape[1] < 2:
        return []
    return top_pairs_from_corr(num.corr(numeric_only=True), threshold, limit)


def _top_values(s: pd.Series, limit: int = 5) -> list[dict]:
    vc = s.value_counts(dropna=True).head(limit)
    return [{"value": str(k), "count": int(v)} for k, v in vc.items()]


def _fmt(v) -> str:
    """Human-friendly rendering of a profile value (None -> em dash)."""
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:,.3g}"
    return str(v)


def column_summary(c: dict) -> str:
    """One-line summary of a profiled column, shared by the report and the UI."""
    if c.get("type") == "numeric":
        return (
            f"mean {_fmt(c.get('mean'))} · median {_fmt(c.get('median'))} · "
            f"std {_fmt(c.get('std'))} · skew {_fmt(c.get('skew'))}"
        )
    if c.get("type") == "datetime":
        return f"{c.get('min')} → {c.get('max')}"
    vc = c.get("value_counts") or []
    return f"top: {vc[0]['value']} ({vc[0]['count']:,})" if vc else "—"


def profile_dataframe(df: pd.DataFrame, max_columns: int = MAX_PROFILE_COLUMNS) -> dict:
    """Build a JSON-serializable statistical profile of the DataFrame."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]
    n_unique = {col: int(df[col].nunique(dropna=True)) for col in df.columns}

    columns: dict[str, dict] = {}
    for col in list(df.columns)[:max_columns]:
        s = df[col]
        info: dict = {
            "dtype": str(s.dtype),
            "n_missing": int(s.isna().sum()),
            "pct_missing": round(float(s.isna().mean() * 100), 2),
            "n_unique": n_unique[col],
        }
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            desc = s.describe()
            info.update(
                {
                    "type": "numeric",
                    "mean": _py(desc.get("mean")),
                    "std": _py(desc.get("std")),
                    "min": _py(desc.get("min")),
                    "p25": _py(desc.get("25%")),
                    "median": _py(desc.get("50%")),
                    "p75": _py(desc.get("75%")),
                    "max": _py(desc.get("max")),
                    "skew": _py(s.skew()),
                    "n_outliers_iqr": iqr_outlier_count(s),
                }
            )
        elif pd.api.types.is_datetime64_any_dtype(s):
            valid = s.dropna()
            info.update(
                {
                    "type": "datetime",
                    "min": str(valid.min()) if len(valid) else None,
                    "max": str(valid.max()) if len(valid) else None,
                }
            )
        elif pd.api.types.is_bool_dtype(s):
            info["type"] = "boolean"
            info["value_counts"] = _top_values(s)
        else:
            info["type"] = "categorical"
            info["value_counts"] = _top_values(s)
        columns[col] = info

    missing = [
        {
            "column": c,
            "n_missing": columns[c]["n_missing"],
            "pct_missing": columns[c]["pct_missing"],
        }
        for c in columns
        if columns[c]["n_missing"] > 0
    ]
    missing.sort(key=lambda m: m["pct_missing"], reverse=True)

    profile = {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "n_numeric_columns": len(num_cols),
        "n_categorical_columns": len(cat_cols),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": round(float(df.memory_usage(deep=True).sum() / 1e6), 3),
        "columns": columns,
        "constant_columns": [c for c in df.columns if n_unique[c] <= 1],
        "top_missing": missing[:10],
        "top_correlated_pairs": top_correlations(df),
    }
    if df.shape[1] > max_columns:
        profile["note"] = (
            f"profile truncated to the first {max_columns} of {df.shape[1]} columns"
        )
    return profile
