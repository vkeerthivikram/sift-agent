import numpy as np
import pandas as pd

from sift_agent.analysis import (
    co_missingness,
    cramers_v,
    data_quality_checks,
    id_like_columns,
    iqr_outlier_pct,
    outlier_examples,
    pii_scan,
    profile_dataframe,
    sample_records,
    top_cramers_v_pairs,
    top_correlations,
    top_category_pairs,
    top_spearman_pairs,
)


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


def test_profile_numeric_extras():
    df = pd.DataFrame({"x": [0.0, -1.0, 2.0, 3.0, 2.0, np.nan]})
    c = profile_dataframe(df)["columns"]["x"]
    assert c["n_zeros"] == 1
    assert c["n_negative"] == 1
    assert c["p05"] <= c["p95"]
    assert c["mode"] == 2.0
    assert "kurtosis" in c
    assert c["outlier_examples"] == []


def test_profile_numeric_outlier_examples():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
    assert outlier_examples(s) == [100.0]


def test_profile_datetime_span():
    df = pd.DataFrame({"d": pd.to_datetime(["2024-01-01", "2024-01-31", "2024-03-01"])})
    c = profile_dataframe(df)["columns"]["d"]
    assert c["n_valid"] == 3
    assert c["span_days"] == 60


def test_profile_includes_new_sections():
    rng = np.random.default_rng(1)
    grp = rng.choice(["a", "b"], 20)
    df = pd.DataFrame(
        {
            "seq": np.arange(20),
            "grp": grp,
            "val": rng.normal(0, 1, 20) + np.where(grp == "b", 50.0, 0.0),
        }
    )
    p = profile_dataframe(df)
    assert p["sample_rows"] and len(p["sample_rows"]) <= 5
    assert {i["column"] for i in p["id_like_columns"]} == {"seq"}
    assert isinstance(p["spearman_pairs"], list)
    assert p["cramers_v_pairs"] == []  # only one categorical
    assert p["category_effects"][0]["category"] == "grp"
    assert p["category_effects"][0]["numeric"] == "val"
    assert p["category_effects"][0]["spread"] > 1


def test_top_spearman_pairs_detects_monotonic_nonlinear():
    df = pd.DataFrame(
        {"x": np.arange(20, dtype=float), "y": np.exp(np.arange(20) / 3.0)}
    )
    pearson = top_correlations(df, threshold=0.0)[0]["pearson_r"]
    pairs = top_spearman_pairs(df, threshold=0.5)
    assert pairs
    assert pairs[0]["col_a"] == "x" and pairs[0]["col_b"] == "y"
    assert pairs[0]["spearman_r"] > 0.99
    assert pairs[0]["spearman_r"] > pearson


def test_top_spearman_pairs_needs_two_numeric_columns():
    df = pd.DataFrame({"a": [1, 2, 3], "t": ["x", "y", "z"]})
    assert top_spearman_pairs(df) == []


def test_cramers_v_bounds():
    perfect_x = pd.Series(["a", "a", "b", "b", "c", "c"])
    perfect_y = pd.Series(["p", "p", "q", "q", "r", "r"])
    assert cramers_v(perfect_x, perfect_y) > 0.99

    rng = np.random.default_rng(3)
    indep_x = pd.Series(rng.choice(["a", "b", "c"], 500))
    indep_y = pd.Series(rng.choice(["p", "q", "r"], 500))
    assert cramers_v(indep_x, indep_y) < 0.2

    assert cramers_v(pd.Series(["a", "a"]), pd.Series(["x", "x"])) == 0.0


def test_top_cramers_v_pairs():
    rng = np.random.default_rng(5)
    a = pd.Series(rng.choice(["lo", "hi"], 200))
    b = a.map({"lo": "small", "hi": "big"})  # perfect association
    other = pd.Series(rng.choice(["u", "v"], 200))
    df = pd.DataFrame({"a": a, "b": b, "other": other})
    pairs = top_cramers_v_pairs(df, threshold=0.3)
    assert pairs[0]["cramers_v"] > 0.99
    assert {pairs[0]["col_a"], pairs[0]["col_b"]} == {"a", "b"}


def test_data_quality_checks():
    df = pd.DataFrame(
        {
            "txt_num": ["1", "2", "3", "n/a"],
            "ws": [" a", "b ", "  ", "c"],
            "case": ["N", "n", "N", "n"],
            "date_txt": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "sentinel_num": [1.0, 2.0, -999.0, 4.0],
            "clean_num": [1.0, 2.0, 3.0, 4.0],
            "clean_txt": ["alpha", "beta", "gamma", "delta"],
        }
    )
    issues = data_quality_checks(df)
    found = {(i["check"], i["column"]) for i in issues}
    assert ("numeric_as_text", "txt_num") in found
    assert ("whitespace", "ws") in found
    assert ("mixed_case", "case") in found
    assert ("dates_as_text", "date_txt") in found
    assert ("sentinels", "txt_num") in found
    assert ("numeric_sentinel", "sentinel_num") in found
    assert not any(col in {"clean_num", "clean_txt"} for _, col in found)


def test_data_quality_checks_empty_columns_ignored():
    df = pd.DataFrame({"all_nan": [None, None], "ok": [1.0, 2.0]})
    assert data_quality_checks(df) == []


def test_data_quality_checks_skips_real_datetimes():
    df = pd.DataFrame({"d": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])})
    assert data_quality_checks(df) == []


def test_id_like_columns():
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {
            "seq": np.arange(50),
            "uid": [f"u{i}" for i in range(50)],
            "val": rng.normal(size=50),
            "rep": rng.integers(0, 5, 50),
        }
    )
    names = {i["column"]: i["reason"] for i in id_like_columns(df)}
    assert "sequential" in names.get("seq", "")
    assert "unique per row" in names.get("uid", "")
    assert "val" not in names and "rep" not in names


def test_id_like_columns_skips_small_frames():
    df = pd.DataFrame({"uid": ["a", "b", "c"]})
    assert id_like_columns(df) == []


def test_pii_scan_detects_emails_and_ips():
    df = pd.DataFrame(
        {
            "email": ["a@x.com", "b@y.com", "c@z.com", "d@w.com"],
            "ip": ["192.168.1.1", "10.0.0.5", "172.16.0.9", "8.8.8.8"],
            "name": ["Alice", "Bob", "Carla", "Deshawn"],
            "num": [1, 2, 3, 4],
        }
    )
    findings = {f["column"]: f["kind"] for f in pii_scan(df)}
    assert findings["email"] == "email"
    assert findings["ip"] == "ip_address"
    assert "name" not in findings
    assert "num" not in findings


def test_pii_scan_requires_majority_match():
    df = pd.DataFrame({"mixed": ["a@x.com", "not an email", "also not", "nope"]})
    assert pii_scan(df) == []


def test_pii_scan_skips_short_columns():
    df = pd.DataFrame({"email": ["a@x.com", "b@y.com"]})
    assert pii_scan(df) == []


def test_profile_dataframe_includes_pii_columns():
    df = pd.DataFrame({"email": ["a@x.com", "b@y.com", "c@z.com"]})
    p = profile_dataframe(df)
    assert p["pii_columns"][0]["column"] == "email"


def test_co_missingness():
    df = pd.DataFrame(
        {
            "a": [1.0, np.nan, np.nan, np.nan],
            "b": [2.0, np.nan, np.nan, 3.0],
            "c": [1.0, 2.0, 3.0, 4.0],
        }
    )
    result = co_missingness(df)
    assert result["fully_empty_rows"] == 0
    assert result["top_pairs"][0]["col_a"] == "a"
    assert result["top_pairs"][0]["col_b"] == "b"
    assert result["top_pairs"][0]["count"] == 2

    all_missing = pd.DataFrame({"a": [None, None], "b": [None, None]})
    assert co_missingness(all_missing)["fully_empty_rows"] == 2


def test_top_category_pairs():
    rng = np.random.default_rng(4)
    df = pd.DataFrame(
        {
            "grp": rng.choice(["x", "y"], 100),
            "sales": np.where(rng.random(100) < 0.5, 100.0, 110.0),
            "noise": rng.normal(0, 1, 100),
        }
    )
    df["sales"] += np.where(df["grp"] == "x", 100.0, 0.0)
    pairs = top_category_pairs(df, limit=5)
    assert pairs[0]["category"] == "grp"
    assert pairs[0]["numeric"] == "sales"
    assert set(pairs[0]["means"]) == {"x", "y"}


def test_sample_records_stringifies_and_caps():
    df = pd.DataFrame({"a": [1, 2, None], "b": ["x|y", "long" * 30, None]})
    rows = sample_records(df, n=2)
    assert len(rows) == 2
    assert rows[0]["a"] == "1.0"  # None forces object dtype -> floats
    assert rows[0]["b"] == "x|y"
    assert rows[1]["a"] == "2.0"
    assert len(rows[1]["b"]) == 40
