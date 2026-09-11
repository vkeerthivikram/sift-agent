import numpy as np
import pandas as pd

from sift_agent.analysis import iqr_outlier_pct, profile_dataframe, top_correlations


def test_iqr_outlier_pct_clean_series():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert iqr_outlier_pct(s) == 0.0


def test_iqr_outlier_pct_with_outliers():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
    assert iqr_outlier_pct(s) > 0


def test_iqr_outlier_pct_empty_and_constant():
    assert iqr_outlier_pct(pd.Series(dtype=float)) == 0.0
    assert iqr_outlier_pct(pd.Series([7.0, 7.0, 7.0])) == 0.0


def test_top_correlations_finds_perfect_pair():
    df = pd.DataFrame(
        {"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10], "c": [5, 3, 1, 3, 5]}
    )
    pairs = top_correlations(df, threshold=0.3)
    assert pairs
    assert pairs[0]["col_a"] == "a" and pairs[0]["col_b"] == "b"
    assert abs(pairs[0]["pearson_r"] - 1.0) < 1e-6


def test_top_correlations_needs_two_numeric_columns():
    df = pd.DataFrame({"a": [1, 2, 3], "t": ["x", "y", "z"]})
    assert top_correlations(df) == []


def test_profile_dataframe_counts_and_types():
    df = pd.DataFrame(
        {
            "num": [1.0, 2.0, np.nan, 4.0],
            "cat": ["a", "b", "a", None],
            "const": [5, 5, 5, 5],
            "flag": [True, False, True, True],
        }
    )
    p = profile_dataframe(df)
    assert p["n_rows"] == 4
    assert p["n_columns"] == 4
    assert p["n_numeric_columns"] >= 1
    assert p["columns"]["num"]["type"] == "numeric"
    assert p["columns"]["cat"]["type"] == "categorical"
    assert p["columns"]["flag"]["type"] == "boolean"
    assert p["columns"]["num"]["n_missing"] == 1
    assert p["columns"]["num"]["pct_missing"] == 25.0
    assert p["constant_columns"] == ["const"]
    assert p["top_missing"][0]["column"] in {"num", "cat"}


def test_profile_dataframe_duplicate_rows():
    df = pd.DataFrame({"a": [1, 2, 2], "b": ["x", "y", "y"]})
    assert profile_dataframe(df)["duplicate_rows"] == 1


def test_profile_dataframe_truncates_wide_frames():
    df = pd.DataFrame(np.random.default_rng(0).normal(size=(5, 150)))
    p = profile_dataframe(df)
    assert len(p["columns"]) == 100
    assert "truncated" in p["note"]


def test_profile_is_json_serializable():
    import json

    df = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0],
            "cat": ["a", "b", "c"],
            "flag": [True, False, True],
        }
    )
    json.dumps(profile_dataframe(df))
