"""Data health scoring from a statistical profile (pure stdlib, no LLM)."""

from __future__ import annotations

CONSISTENCY_CHECKS = frozenset({"whitespace", "mixed_case"})
VALIDITY_CHECKS = frozenset(
    {"numeric_as_text", "dates_as_text", "sentinels", "numeric_sentinel"}
)

# Points deducted per flagged column. A fixed deduction keeps a finding
# meaningful regardless of how many columns dilute a share-based penalty
# (a dataset with one broken column of three and one of fifty both carry
# a real finding).
FLAG_DEDUCTION = 15.0

COMPONENT_WEIGHTS: dict[str, tuple[str, int]] = {
    "completeness": ("Completeness", 40),
    "uniqueness": ("Uniqueness", 20),
    "consistency": ("Consistency", 20),
    "validity": ("Validity", 20),
}

GRADE_BANDS = ((90, "A"), (80, "B"), (70, "C"), (60, "D"))

_CONSISTENCY_LABELS = (("whitespace", "whitespace"), ("mixed_case", "mixed case"))
_VALIDITY_LABELS = (
    ("numeric_as_text", "numeric-as-text"),
    ("dates_as_text", "dates-as-text"),
    ("sentinels", "sentinel strings"),
    ("numeric_sentinel", "numeric sentinels"),
)


def _clamp(value: float) -> float:
    """Clamp a component score into the 0..100 range."""
    return max(0.0, min(100.0, value))


def _fmt_pct(value: float) -> str:
    """Render a percentage compactly (18.0 -> '18', 8.33 -> '8.33')."""
    return f"{value:g}"


def _grade(score: int) -> str:
    """Letter grade for a 0..100 score."""
    for threshold, grade in GRADE_BANDS:
        if score >= threshold:
            return grade
    return "F"


def _columns_by_check(data_quality: list[dict]) -> dict[str, set[str]]:
    """Distinct column names per data-quality check name."""
    by_check: dict[str, set[str]] = {}
    for item in data_quality:
        check = item.get("check")
        column = item.get("column")
        if check is None or column is None:
            continue
        by_check.setdefault(check, set()).add(column)
    return by_check


def _flagged_columns(by_check: dict[str, set[str]], checks: frozenset[str]) -> set[str]:
    """Union of distinct columns flagged by any of the given checks."""
    flagged: set[str] = set()
    for check in checks:
        flagged |= by_check.get(check, set())
    return flagged


def _worst_missing(columns: dict) -> tuple[str | None, float]:
    """Highest-missingness column and its percentage (ties keep profile order)."""
    worst_col = None
    worst_pct = 0.0
    for name, col in columns.items():
        pct = float((col or {}).get("pct_missing") or 0)
        if pct > worst_pct:
            worst_col, worst_pct = name, pct
    return worst_col, worst_pct


def _completeness(columns: dict) -> tuple[float, str]:
    """100 minus the blend of mean and worst per-column missingness.

    Blending keeps a single heavily-missing column from being diluted by
    wide datasets (mean alone) while not over-reacting to one bad column
    (worst alone).
    """
    if not columns:
        return 100.0, "No column profiles to assess."
    pcts = [float((col or {}).get("pct_missing") or 0) for col in columns.values()]
    mean_missing = sum(pcts) / len(pcts)
    worst_col, worst_pct = _worst_missing(columns)
    penalty = (mean_missing + worst_pct) / 2
    if worst_col is None:
        detail = f"No missing values across {len(columns)} columns."
    else:
        detail = (
            f"Average {_fmt_pct(round(mean_missing, 2))}% missing across "
            f"{len(columns)} columns; worst is `{worst_col}` at "
            f"{_fmt_pct(worst_pct)}%."
        )
    return _clamp(100.0 - penalty), detail


def _uniqueness(n_rows: int, duplicate_rows: int) -> tuple[float, str]:
    """100 minus the duplicate-row percentage of all rows."""
    if duplicate_rows <= 0:
        return 100.0, "No duplicate rows."
    dup_pct = 100.0 * duplicate_rows / max(n_rows, 1)
    detail = (
        f"{duplicate_rows:,} of {n_rows:,} rows are duplicates "
        f"({_fmt_pct(round(dup_pct, 2))}%)."
    )
    return _clamp(100.0 - dup_pct), detail


def _consistency(
    n_columns: int, by_check: dict[str, set[str]], n_constant: int
) -> tuple[float, str]:
    """100 minus a fixed deduction per formatting-flagged or constant column."""
    flagged = _flagged_columns(by_check, CONSISTENCY_CHECKS)
    penalty = len(flagged) + n_constant
    score = _clamp(100.0 - FLAG_DEDUCTION * penalty)
    if not penalty:
        return score, "No whitespace, casing, or constant-column issues."
    parts = [
        f"{label}: {len(by_check[check])}"
        for check, label in _CONSISTENCY_LABELS
        if by_check.get(check)
    ]
    if n_constant:
        parts.append(f"constant: {n_constant}")
    detail = (
        f"{penalty} of {n_columns} columns have formatting issues ({', '.join(parts)})."
    )
    return score, detail


def _validity(n_columns: int, by_check: dict[str, set[str]]) -> tuple[float, str]:
    """100 minus a fixed deduction per type- or sentinel-flagged column."""
    flagged = _flagged_columns(by_check, VALIDITY_CHECKS)
    score = _clamp(100.0 - FLAG_DEDUCTION * len(flagged))
    if not flagged:
        return score, "No type, date-format, or sentinel-value issues."
    parts = [
        f"{label}: {len(by_check[check])}"
        for check, label in _VALIDITY_LABELS
        if check in by_check
    ]
    detail = (
        f"{len(flagged)} of {n_columns} columns have validity issues "
        f"({', '.join(parts)})."
    )
    return score, detail


def _verdict(
    score: int,
    grade: str,
    n_columns: int,
    duplicate_rows: int,
    issue_columns: set[str],
    worst: tuple[str | None, float],
) -> str:
    """One-sentence summary citing only counts that are non-zero in the profile."""
    clauses: list[str] = []
    if duplicate_rows > 0:
        plural = "" if duplicate_rows == 1 else "s"
        clauses.append(f"{duplicate_rows:,} duplicate row{plural}")
    if issue_columns:
        clauses.append(
            f"{len(issue_columns)} of {n_columns} columns with quality issues"
        )
    worst_col, worst_pct = worst
    if worst_col is not None:
        clauses.append(f"worst missingness `{worst_col}` at {_fmt_pct(worst_pct)}%")
    if not clauses:
        clauses.append("no duplicates, missing values, or quality issues detected")
    return f"Grade {grade} ({score}/100): {', '.join(clauses)}."


def health_score(profile: dict) -> dict:
    """Compute the 0-100 data health score for a statistical profile.

    Input is the profile dict produced by ``analysis.profile_dataframe``;
    missing or empty keys count as zero/empty, so this never raises. Returns
    a JSON-safe ``{"score", "grade", "verdict", "components"}`` where
    ``components`` holds ``{"key", "label", "score", "detail"}`` dicts in
    ``COMPONENT_WEIGHTS`` order.
    """
    profile = profile or {}
    columns = profile.get("columns") or {}
    n_rows = int(profile.get("n_rows") or 0)
    duplicate_rows = int(profile.get("duplicate_rows") or 0)
    n_columns = int(profile.get("n_columns") or 0) or len(columns)
    by_check = _columns_by_check(profile.get("data_quality") or [])
    constant_columns = profile.get("constant_columns") or []

    parts = {
        "completeness": _completeness(columns),
        "uniqueness": _uniqueness(n_rows, duplicate_rows),
        "consistency": _consistency(n_columns, by_check, len(constant_columns)),
        "validity": _validity(n_columns, by_check),
    }
    weight_sum = sum(weight for _, weight in COMPONENT_WEIGHTS.values())
    components: list[dict] = []
    total = 0.0
    for key, (label, weight) in COMPONENT_WEIGHTS.items():
        value, detail = parts[key]
        total += value * weight
        components.append(
            {"key": key, "label": label, "score": round(value), "detail": detail}
        )
    score = round(total / weight_sum)
    grade = _grade(score)
    issue_columns = set().union(*by_check.values())
    verdict = _verdict(
        score,
        grade,
        n_columns,
        duplicate_rows,
        issue_columns,
        _worst_missing(columns),
    )
    return {
        "score": score,
        "grade": grade,
        "verdict": verdict,
        "components": components,
    }
