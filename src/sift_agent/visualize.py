"""Chart generation with matplotlib/seaborn (Agg backend, object API).

Uses ``Figure`` + ``FigureCanvasAgg`` directly instead of ``matplotlib.pyplot``
so concurrent runs (separate threads) never touch the global pyplot figure
manager, which is not thread-safe.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .analysis import (
    iqr_outlier_pct,
    top_category_pairs,
    top_correlations,
    top_pairs_from_corr,
)
from .state import ChartRef

sns.set_theme(style="whitegrid", palette="deep")

logger = logging.getLogger("sift_agent.visualize")

_MAX_NUMERIC_PLOTS = 12
_MAX_CATEGORICAL_PLOTS = 6
_CATEGORICAL_CARDINALITY = 20
_MAX_HEATMAP_COLS = 40
_MAX_SCATTER_POINTS = 2000
_MAX_KDE_POINTS = 50_000
_MAX_GROUP_PLOTS = 3


def _numeric_columns(df: pd.DataFrame, n_unique: dict[str, int]) -> list[str]:
    return [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if not pd.api.types.is_bool_dtype(df[c]) and n_unique[c] > 1
    ]


def _categorical_columns(df: pd.DataFrame, n_unique: dict[str, int]) -> list[str]:
    cols = [
        c
        for c in df.select_dtypes(exclude=[np.number, "datetime"]).columns
        if 1 < n_unique[c] <= _CATEGORICAL_CARDINALITY
    ]
    bools = [
        c for c in df.columns if pd.api.types.is_bool_dtype(df[c]) and n_unique[c] > 1
    ]
    return cols + bools


def _figure(figsize: tuple[float, float]) -> Figure:
    fig = Figure(figsize=figsize)
    FigureCanvasAgg(fig)
    return fig


def _grid(n: int, ncols: int, w: float = 4.2, h: float = 3.2):
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)
    fig = _figure((w * ncols, h * nrows))
    return fig, fig.subplots(nrows, ncols, squeeze=False)


def _save(fig: Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def _hide_extra(axes, used: int) -> None:
    for ax in axes.flat[used:]:
        ax.set_visible(False)


def _missing_chart(df: pd.DataFrame, charts_dir: Path) -> ChartRef | None:
    counts = df.isna().sum()
    counts = counts[counts > 0].sort_values()
    if counts.empty:
        return None
    path = charts_dir / "01_missing_values.png"
    fig = _figure((8, max(3, 0.45 * len(counts))))
    ax = fig.subplots()
    counts.plot.barh(ax=ax, color="#d95f02")
    ax.set_title("Missing values by column")
    ax.set_xlabel("missing count")
    _save(fig, path)
    worst = counts.sort_values(ascending=False).head(3)
    pct = (worst / len(df) * 100).round(1)
    caption = (
        "Missing-value counts; worst: "
        + ", ".join(f"{c} ({p}%) " for c, p in zip(worst.index, pct)).rstrip()
        + "."
    )
    return ChartRef(title="Missing Values", path=f"charts/{path.name}", caption=caption)


def _distribution_grid(
    df: pd.DataFrame, charts_dir: Path, numeric: list[str]
) -> ChartRef | None:
    cols = numeric[:_MAX_NUMERIC_PLOTS]
    if not cols:
        return None
    fig, axes = _grid(len(cols), 4)
    for ax, col in zip(axes.flat, cols):
        data = df[col].dropna()
        if len(data) > _MAX_KDE_POINTS:
            data = data.sample(_MAX_KDE_POINTS, random_state=0)
        sns.histplot(data, kde=True, bins=30, ax=ax)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
    _hide_extra(axes, len(cols))
    path = charts_dir / "02_distributions.png"
    _save(fig, path)
    skews = sorted(
        ((c, df[c].skew()) for c in cols if pd.notna(df[c].skew())),
        key=lambda t: abs(t[1]),
        reverse=True,
    )
    top = ", ".join(f"{c} (skew {s:+.2f})" for c, s in skews[:3]) if skews else "none"
    caption = f"Histograms with KDE; most skewed: {top}."
    return ChartRef(
        title="Numeric Distributions", path=f"charts/{path.name}", caption=caption
    )


def _boxplot_grid(
    df: pd.DataFrame, charts_dir: Path, numeric: list[str]
) -> ChartRef | None:
    cols = numeric[:_MAX_NUMERIC_PLOTS]
    if not cols:
        return None
    fig, axes = _grid(len(cols), 4)
    for ax, col in zip(axes.flat, cols):
        sns.boxplot(x=df[col].dropna(), ax=ax)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
    _hide_extra(axes, len(cols))
    path = charts_dir / "03_boxplots.png"
    _save(fig, path)
    outliers = sorted(
        ((c, iqr_outlier_pct(df[c])) for c in cols), key=lambda t: t[1], reverse=True
    )
    top = ", ".join(f"{c} ({p:.1f}%)" for c, p in outliers[:3] if p > 0) or "none"
    caption = f"Box plots per numeric column; most IQR outliers: {top}."
    return ChartRef(
        title="Numeric Box Plots", path=f"charts/{path.name}", caption=caption
    )


def _count_grid(
    df: pd.DataFrame, charts_dir: Path, n_unique: dict[str, int]
) -> ChartRef | None:
    cols = _categorical_columns(df, n_unique)[:_MAX_CATEGORICAL_PLOTS]
    if not cols:
        return None
    n = len(df)
    fig, axes = _grid(len(cols), 3, w=5.2)
    tops: list[tuple[str, str, float]] = []
    for ax, col in zip(axes.flat, cols):
        vc = df[col].value_counts().head(12)
        sns.barplot(x=vc.values, y=[str(i) for i in vc.index], color="#4c72b0", ax=ax)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
        tops.append((col, str(vc.index[0]), float(vc.iloc[0]) / n))
    _hide_extra(axes, len(cols))
    path = charts_dir / "04_categorical_counts.png"
    _save(fig, path)
    top = ", ".join(f"{c}='{v}' ({p:.0%})" for c, v, p in tops[:3])
    caption = f"Category frequency distributions; most common values: {top}."
    return ChartRef(
        title="Categorical Counts", path=f"charts/{path.name}", caption=caption
    )


def _correlation_heatmap(
    df: pd.DataFrame, charts_dir: Path, numeric: list[str]
) -> ChartRef | None:
    cols = numeric[:_MAX_HEATMAP_COLS]
    if len(cols) < 2:
        return None
    corr = df[cols].corr(numeric_only=True)
    size = max(6, 0.55 * len(cols) + 2)
    fig = _figure((size, size))
    ax = fig.subplots()
    sns.heatmap(
        corr,
        ax=ax,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        annot=len(cols) <= 12,
        fmt=".2f",
        cbar_kws={"shrink": 0.8},
        xticklabels=[c[:12] for c in cols],
        yticklabels=[c[:12] for c in cols],
    )
    ax.set_title("Correlation matrix (numeric columns)")
    _save(fig, path := charts_dir / "05_correlation_heatmap.png")
    pairs = top_pairs_from_corr(corr, threshold=0.3, limit=3)
    detail = (
        "; ".join(
            f"{p['col_a']} vs {p['col_b']} r={p['pearson_r']:+.2f}" for p in pairs
        )
        or "no pairwise |r| >= 0.3"
    )
    caption = f"Pearson correlations between numeric columns. Strongest: {detail}."
    return ChartRef(
        title="Correlation Heatmap", path=f"charts/{path.name}", caption=caption
    )


def _scatter_grid(
    df: pd.DataFrame,
    charts_dir: Path,
    numeric: list[str],
    pairs: list[dict] | None = None,
) -> ChartRef | None:
    if pairs is None:
        pairs = top_correlations(df, threshold=0.4, limit=3)
    else:
        pairs = [p for p in pairs if abs(p["pearson_r"]) >= 0.4][:3]
    pairs = [p for p in pairs if p["col_a"] in numeric and p["col_b"] in numeric]
    if not pairs:
        return None
    fig, axes = _grid(len(pairs), min(len(pairs), 3), w=5.0, h=4.0)
    for ax, p in zip(axes.flat, pairs):
        data = df[[p["col_a"], p["col_b"]]].dropna()
        if len(data) > _MAX_SCATTER_POINTS:
            data = data.sample(_MAX_SCATTER_POINTS, random_state=0)
        ax.scatter(data[p["col_a"]], data[p["col_b"]], s=10, alpha=0.5)
        ax.set_title(
            f"{p['col_a']} vs {p['col_b']} (r={p['pearson_r']:+.2f})", fontsize=10
        )
    _hide_extra(axes, len(pairs))
    path = charts_dir / "06_relationships.png"
    _save(fig, path)
    caption = "Scatter plots of the strongest linear relationships (downsampled for readability)."
    return ChartRef(
        title="Top Relationships", path=f"charts/{path.name}", caption=caption
    )


def _grouped_boxplots(df: pd.DataFrame, charts_dir: Path) -> ChartRef | None:
    """Box plots of numeric columns split by a low-cardinality category."""
    pairs = [p for p in top_category_pairs(df, limit=_MAX_GROUP_PLOTS)]
    if not pairs:
        return None
    fig, axes = _grid(len(pairs), min(len(pairs), 3), w=5.0, h=4.0)
    for ax, p in zip(axes.flat, pairs):
        data = df[[p["category"], p["numeric"]]].dropna()
        sns.boxplot(data=data, x=p["category"], y=p["numeric"], ax=ax)
        ax.set_title(f"{p['numeric']} by {p['category']}", fontsize=10)
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelsize=8)
    _hide_extra(axes, len(pairs))
    path = charts_dir / "07_grouped_boxplots.png"
    _save(fig, path)
    detail = "; ".join(
        f"{p['numeric']} by {p['category']} (spread {p['spread']:.1f}σ)" for p in pairs
    )
    caption = f"Numeric distributions grouped by category, strongest group differences first: {detail}."
    return ChartRef(
        title="Numeric by Category", path=f"charts/{path.name}", caption=caption
    )


def _time_trend(df: pd.DataFrame, charts_dir: Path) -> ChartRef | None:
    """Mean of numeric columns per period over the first datetime column."""
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if not dt_cols:
        return None
    dcol = dt_cols[0]
    nums = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if not pd.api.types.is_bool_dtype(df[c]) and df[c].nunique(dropna=True) > 1
    ][:3]
    if not nums:
        return None
    data = df[[dcol, *nums]].dropna(subset=[dcol]).sort_values(dcol)
    if len(data) < 3:
        return None
    span = data[dcol].max() - data[dcol].min()
    if span <= pd.Timedelta(days=90):
        freq, unit = "D", "day"
    elif span <= pd.Timedelta(days=3 * 365):
        freq, unit = "W", "week"
    else:
        freq, unit = "MS", "month"
    ts = data.set_index(dcol)[nums].resample(freq).mean().dropna(how="all")
    if len(ts) < 2:
        return None
    fig = _figure((10, 4.5))
    ax = fig.subplots()
    for col in ts.columns:
        ax.plot(ts.index, ts[col], label=col, linewidth=1.5)
    ax.set_title(f"Mean of {', '.join(map(str, ts.columns))} per {unit} over {dcol}")
    ax.set_ylabel("mean")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    path = charts_dir / "08_time_trend.png"
    _save(fig, path)
    caption = f"Time trend of {', '.join(map(str, ts.columns))}: mean per {unit} across `{dcol}`."
    return ChartRef(title="Time Trend", path=f"charts/{path.name}", caption=caption)


def generate_charts(
    df: pd.DataFrame,
    charts_dir: Path,
    failures: list[str] | None = None,
    corr_pairs: list[dict] | None = None,
) -> list[ChartRef]:
    """Generate all applicable charts into charts_dir; each may be skipped if not applicable.

    Failed builders are logged; when ``failures`` is provided, a one-line reason
    per failed chart is appended so it can be surfaced in the report.
    ``corr_pairs`` reuses precomputed correlation pairs (e.g. from the profile)
    instead of recomputing the correlation matrix.
    """
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    n_unique = {c: int(df[c].nunique(dropna=True)) for c in df.columns}
    numeric = _numeric_columns(df, n_unique)

    builders: list[tuple[str, callable]] = [
        ("missing_values", lambda: _missing_chart(df, charts_dir)),
        ("distributions", lambda: _distribution_grid(df, charts_dir, numeric)),
        ("boxplots", lambda: _boxplot_grid(df, charts_dir, numeric)),
        ("categorical_counts", lambda: _count_grid(df, charts_dir, n_unique)),
        (
            "correlation_heatmap",
            lambda: _correlation_heatmap(df, charts_dir, numeric),
        ),
        (
            "relationships",
            lambda: _scatter_grid(df, charts_dir, numeric, pairs=corr_pairs),
        ),
        ("grouped_boxplots", lambda: _grouped_boxplots(df, charts_dir)),
        ("time_trend", lambda: _time_trend(df, charts_dir)),
    ]

    def _safe(name: str, build):
        try:
            return build()
        except Exception as exc:
            logger.warning("chart '%s' failed: %s", name, exc, exc_info=True)
            if failures is not None:
                failures.append(
                    f"chart '{name}' skipped: {exc.__class__.__name__}: {exc}"
                )
            return None

    return [ref for ref in (_safe(n, b) for n, b in builders) if ref is not None]
