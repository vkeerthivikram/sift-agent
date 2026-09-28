"""Statistical profiling of a pandas DataFrame (pure computation, no LLM)."""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

MAX_PROFILE_COLUMNS = 100
MAX_QUALITY_COLUMNS = 200
MAX_CRAMERS_COLUMNS = 12
MAX_CRAMERS_CARDINALITY = 20
MAX_CATEGORY_CARDINALITY = 20
MAX_GROUP_SAMPLE_ROWS = 200_000
MIN_ID_LIKE_ROWS = 10

SENTINEL_STRINGS = frozenset(
    {"", "n/a", "na", "-", "?", "null", "none", "nan", "missing", "unknown"}
)
NUMERIC_SENTINELS = frozenset({-999, -9999})
MIN_PII_ROWS = 3
PII_MATCH_RATIO = 0.6

# check order matters: first pattern to clear PII_MATCH_RATIO wins the column
PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"^[\w.+-]+@[\w-]+\.[\w.-]+$"),
    "ssn": re.compile(r"^\d{3}-\d{2}-\d{4}$"),
    "credit_card": re.compile(r"^\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{3,4}$"),
    "ip_address": re.compile(
        r"^(25[0-5]|2[0-4]\d|1?\d?\d)(\.(25[0-5]|2[0-4]\d|1?\d?\d)){3}$"
    ),
    "phone": re.compile(r"^\+?1?[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}$"),
}
PII_LABELS = {
    "email": "email address",
    "ssn": "US Social Security Number",
    "credit_card": "credit card number",
    "ip_address": "IP address",
    "phone": "phone number",
}

# label -> advice for the recommendations layer (graph.py)
DATA_QUALITY_ADVICE = {
    "whitespace": "Strip leading/trailing whitespace in string columns",
    "numeric_as_text": "Convert text columns that hold numbers with `pd.to_numeric`",
    "dates_as_text": "Parse text columns that hold dates with `pd.to_datetime`",
    "sentinels": "Replace placeholder values (e.g. 'n/a') with real NaN",
    "mixed_case": "Normalize inconsistent casing (e.g. 'N' vs 'n') in categorical columns",
    "numeric_sentinel": "Review sentinel values (-999/-9999) — replace with NaN or document them",
}


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


def outlier_examples(s: pd.Series, limit: int = 3) -> list[float]:
    """Most extreme IQR outlier values, farthest from the median first."""
    s = s.dropna()
    fences = _iqr_fences(s)
    if fences is None or s.empty:
        return []
    lo, hi = fences
    out = s[(s < lo) | (s > hi)]
    if out.empty:
        return []
    median = s.median()
    ranked = out.reindex(out.sub(median).abs().sort_values(ascending=False).index).head(
        limit
    )
    return [round(float(v), 6) for v in ranked]


def _numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric, non-boolean columns with more than one distinct value."""
    num = df.select_dtypes(include=[np.number])
    keep = [
        c
        for c in num.columns
        if not pd.api.types.is_bool_dtype(num[c]) and num[c].nunique(dropna=True) > 1
    ]
    return num.loc[:, keep]


def top_pairs_from_corr(
    corr: pd.DataFrame, threshold: float = 0.3, limit: int = 10, key: str = "pearson_r"
) -> list[dict]:
    """Strongest absolute correlations from a precomputed corr matrix."""
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
                        key: round(float(r), 3),
                    }
                )
    pairs.sort(key=lambda p: abs(p[key]), reverse=True)
    return pairs[:limit]


def top_correlations(
    df: pd.DataFrame, threshold: float = 0.3, limit: int = 10
) -> list[dict]:
    """Strongest absolute Pearson correlations between numeric column pairs."""
    num = _numeric_frame(df)
    if num.shape[1] < 2:
        return []
    return top_pairs_from_corr(num.corr(numeric_only=True), threshold, limit)


def top_spearman_pairs(
    df: pd.DataFrame, threshold: float = 0.5, limit: int = 10
) -> list[dict]:
    """Strongest absolute Spearman (rank) correlations between numeric pairs.

    Catches monotonic relationships that Pearson's linear measure misses.
    """
    num = _numeric_frame(df)
    if num.shape[1] < 2:
        return []
    return top_pairs_from_corr(
        num.corr(method="spearman", numeric_only=True), threshold, limit, "spearman_r"
    )


def _chi2_statistic(table: np.ndarray) -> float:
    """Pearson chi-square statistic for a contingency table (no scipy)."""
    n = float(table.sum())
    if n <= 0:
        return 0.0
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(expected > 0, (table - expected) ** 2 / expected, 0.0)
    return float(terms.sum())


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bias-corrected Cramér's V association between two categorical series.

    0 = independent, 1 = perfectly associated. Returns 0.0 when undefined
    (degenerate tables).
    """
    table = pd.crosstab(x, y).to_numpy(dtype=float)
    n = table.sum()
    if n < 2 or table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    phi2 = _chi2_statistic(table) / n
    r, k = table.shape
    phi2_corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    if denom <= 0:
        return 0.0
    return math.sqrt(phi2_corr / denom)


def _low_cardinality_categoricals(
    df: pd.DataFrame, cap: int, max_card: int
) -> list[str]:
    cols = [
        c
        for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])
    ]
    return [c for c in cols if 1 < df[c].nunique(dropna=True) <= max_card][:cap]


def top_cramers_v_pairs(
    df: pd.DataFrame, threshold: float = 0.3, limit: int = 10
) -> list[dict]:
    """Strongest categorical-categorical associations by Cramér's V."""
    cats = _low_cardinality_categoricals(
        df, MAX_CRAMERS_COLUMNS, MAX_CRAMERS_CARDINALITY
    )
    pairs: list[dict] = []
    for i in range(len(cats)):
        for j in range(i + 1, len(cats)):
            a, b = df[cats[i]], df[cats[j]]
            mask = a.notna() & b.notna()
            if mask.sum() < 2:
                continue
            v = cramers_v(a[mask], b[mask])
            if v >= threshold:
                pairs.append(
                    {"col_a": cats[i], "col_b": cats[j], "cramers_v": round(v, 3)}
                )
    pairs.sort(key=lambda p: p["cramers_v"], reverse=True)
    return pairs[:limit]


def co_missingness(df: pd.DataFrame, limit: int = 5) -> dict:
    """Fully-empty row count and the column pairs that go missing together."""
    miss = df.isna()
    fully = int(miss.all(axis=1).sum())
    cols = [c for c in df.columns if miss[c].any()][:30]
    pairs: list[dict] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            both = int((miss[cols[i]] & miss[cols[j]]).sum())
            if both:
                pairs.append(
                    {
                        "col_a": cols[i],
                        "col_b": cols[j],
                        "count": both,
                        "pct_of_rows": round(both / len(df) * 100, 2),
                    }
                )
    pairs.sort(key=lambda p: p["count"], reverse=True)
    return {"fully_empty_rows": fully, "top_pairs": pairs[:limit]}


def top_category_pairs(df: pd.DataFrame, limit: int = 8) -> list[dict]:
    """Numeric-by-category group means, ranked by spread relative to column std."""
    cats = _low_cardinality_categoricals(
        df, MAX_CRAMERS_COLUMNS, MAX_CATEGORY_CARDINALITY
    )
    nums = list(_numeric_frame(df).columns)
    if not cats or not nums:
        return []
    sample = df
    if len(df) > MAX_GROUP_SAMPLE_ROWS:
        sample = df.sample(MAX_GROUP_SAMPLE_ROWS, random_state=0)
    results: list[dict] = []
    for cat in cats:
        for num in nums:
            grouped = sample[[cat, num]].dropna().groupby(cat, observed=True)[num]
            counts = grouped.size()
            if grouped.ngroups < 2 or counts.min() < 3:
                continue  # tiny groups make mean comparisons meaningless
            std = float(sample[num].std())
            means = grouped.mean()
            if not std or not np.isfinite(std):
                continue
            spread = float(means.max() - means.min()) / std
            if spread <= 0:
                continue
            top_cats = means.abs().sort_values(ascending=False).head(8).index
            results.append(
                {
                    "category": cat,
                    "numeric": num,
                    "spread": round(spread, 3),
                    "means": {str(k): round(float(means[k]), 4) for k in top_cats},
                    "counts": {str(k): int(counts[k]) for k in top_cats},
                }
            )
    results.sort(key=lambda r: r["spread"], reverse=True)
    return results[:limit]


def data_quality_checks(df: pd.DataFrame) -> list[dict]:
    """Scan columns for common dirty-data patterns (capped at MAX_QUALITY_COLUMNS)."""
    issues: list[dict] = []

    def add(check: str, column: str, detail: str) -> None:
        issues.append({"check": check, "column": column, "detail": detail})

    for col in list(df.columns)[:MAX_QUALITY_COLUMNS]:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        if pd.api.types.is_numeric_dtype(s):
            sent = int(s.dropna().isin(list(NUMERIC_SENTINELS)).sum())
            if sent:
                add("numeric_sentinel", col, f"{sent} rows equal -999/-9999")
            continue
        vals = s.dropna().astype(str).str.strip()
        if vals.empty:
            continue

        blank = vals == ""
        padded = (vals != vals.str.strip()) | blank
        n_ws = int(padded.sum())
        if n_ws:
            add(
                "whitespace",
                col,
                f"{n_ws}/{len(vals)} values have stray or blank-only whitespace",
            )

        nonblank = vals[~blank]
        if len(nonblank) >= 2:
            parsed = pd.to_numeric(
                nonblank.str.replace(",", "", regex=False), errors="coerce"
            )
            ratio = float(parsed.notna().mean())
            if ratio >= 0.6 and int(parsed.notna().sum()) >= 2:
                add(
                    "numeric_as_text",
                    col,
                    f"{int(parsed.notna().sum())}/{len(nonblank)} non-blank values parse as numbers",
                )

        probe = nonblank.head(1000)
        try:
            as_dates = pd.to_datetime(probe, errors="coerce", format="mixed")
        except (TypeError, ValueError):
            as_dates = pd.Series(pd.NaT, index=probe.index)
        n_dates = int(pd.notna(as_dates).sum())
        if len(probe) >= 3 and n_dates / len(probe) >= 0.6:
            add(
                "dates_as_text",
                col,
                f"{n_dates}/{len(probe)} non-blank values parse as dates",
            )

        lowered = vals.str.lower()
        sent_mask = lowered.isin(SENTINEL_STRINGS)
        n_sent = int(sent_mask.sum())
        if n_sent:
            example = next((v for v in vals[sent_mask].unique() if v != ""), "")
            detail = (
                f"values look like null placeholders (e.g. '{example}')"
                if example
                else "values are blank-only null placeholders"
            )
            add("sentinels", col, f"{n_sent}/{len(vals)} {detail}")

        n_raw = int(vals.nunique())
        n_lower = int(lowered.nunique())
        if 0 < n_lower < n_raw:
            add(
                "mixed_case",
                col,
                f"{n_raw} distinct values collapse to {n_lower} when lowercased",
            )
    return issues


def pii_scan(df: pd.DataFrame, min_ratio: float = PII_MATCH_RATIO) -> list[dict]:
    """Flag string columns whose values mostly look like emails, phone numbers,
    SSNs, credit-card numbers or IP addresses — a heads-up before sharing data.
    """
    findings: list[dict] = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
            continue
        vals = s.dropna().astype(str).str.strip()
        vals = vals[vals != ""]
        if len(vals) < MIN_PII_ROWS:
            continue
        for kind, pattern in PII_PATTERNS.items():
            matches = vals.str.match(pattern)
            ratio = float(matches.mean())
            if ratio >= min_ratio:
                findings.append(
                    {
                        "column": col,
                        "kind": kind,
                        "label": PII_LABELS[kind],
                        "count": int(matches.sum()),
                        "pct": round(ratio * 100, 1),
                    }
                )
                break  # first matching pattern wins the column
    findings.sort(key=lambda f: f["count"], reverse=True)
    return findings


def id_like_columns(df: pd.DataFrame, min_rows: int = MIN_ID_LIKE_ROWS) -> list[dict]:
    """Columns that look like row identifiers rather than signal."""
    n = len(df)
    if n < min_rows:
        return []
    ids: list[dict] = []
    for col in df.columns:
        s = df[col]
        if s.nunique(dropna=True) != n:
            continue
        if pd.api.types.is_integer_dtype(s):
            if s.dropna().is_monotonic_increasing:
                ids.append({"column": col, "reason": "sequential integer"})
            else:
                ids.append({"column": col, "reason": "unique per row"})
        elif not pd.api.types.is_numeric_dtype(s):
            ids.append({"column": col, "reason": "unique per row"})
    return ids


def sample_records(df: pd.DataFrame, n: int = 5, width: int = 40) -> list[dict]:
    """First n rows as stringified dicts (JSON-safe preview of the data)."""
    records: list[dict] = []
    for _, row in df.head(n).iterrows():
        rec: dict = {}
        for col, v in row.items():
            if pd.isna(v):
                rec[col] = None
            else:
                rec[col] = str(v)[:width]
        records.append(rec)
    return records


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
        base = f"{c.get('min')} → {c.get('max')}"
        span = c.get("span_days")
        return f"{base} ({span:g} days)" if span else base
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
            valid = s.dropna()
            mode = valid.mode(dropna=True)
            info.update(
                {
                    "type": "numeric",
                    "mean": _py(desc.get("mean")),
                    "std": _py(desc.get("std")),
                    "min": _py(desc.get("min")),
                    "p05": _py(valid.quantile(0.05)),
                    "p25": _py(desc.get("25%")),
                    "median": _py(desc.get("50%")),
                    "p75": _py(desc.get("75%")),
                    "p95": _py(valid.quantile(0.95)),
                    "max": _py(desc.get("max")),
                    "skew": _py(s.skew()),
                    "kurtosis": _py(s.kurtosis()),
                    "mode": _py(mode.iloc[0]) if len(mode) else None,
                    "n_zeros": int((valid == 0).sum()),
                    "n_negative": int((valid < 0).sum()),
                    "n_outliers_iqr": iqr_outlier_count(s),
                    "outlier_examples": outlier_examples(s),
                }
            )
        elif pd.api.types.is_datetime64_any_dtype(s):
            valid = s.dropna()
            span_days = None
            if len(valid) >= 2:
                span_days = round(
                    float((valid.max() - valid.min()).total_seconds() / 86400), 2
                )
            info.update(
                {
                    "type": "datetime",
                    "min": str(valid.min()) if len(valid) else None,
                    "max": str(valid.max()) if len(valid) else None,
                    "n_valid": len(valid),
                    "span_days": span_days,
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

    outlier_columns = sorted(
        (
            {
                "column": c,
                "count": columns[c]["n_outliers_iqr"],
                "pct": iqr_outlier_pct(df[c]),
                "examples": columns[c]["outlier_examples"],
            }
            for c in columns
            if columns[c].get("type") == "numeric"
            and columns[c].get("n_outliers_iqr", 0) > 0
        ),
        key=lambda o: o["count"],
        reverse=True,
    )

    profile = {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "n_numeric_columns": len(num_cols),
        "n_categorical_columns": len(cat_cols),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": round(float(df.memory_usage(deep=True).sum() / 1e6), 3),
        "columns": columns,
        "sample_rows": sample_records(df),
        "constant_columns": [c for c in df.columns if n_unique[c] <= 1],
        "top_missing": missing[:10],
        "top_correlated_pairs": top_correlations(df),
        "spearman_pairs": top_spearman_pairs(df),
        "cramers_v_pairs": top_cramers_v_pairs(df),
        "co_missing": co_missingness(df),
        "category_effects": top_category_pairs(df),
        "id_like_columns": id_like_columns(df),
        "data_quality": data_quality_checks(df),
        "outlier_columns": outlier_columns,
        "pii_columns": pii_scan(df),
    }
    if df.shape[1] > max_columns:
        profile["note"] = (
            f"profile truncated to the first {max_columns} of {df.shape[1]} columns"
        )
    return profile
