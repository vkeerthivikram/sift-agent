import json

import numpy as np
import pandas as pd

from sift_agent.investigate import (
    TOOL_SPECS,
    correlation_check,
    group_compare,
    missingness_analysis,
    outlier_inspect,
    run_tool,
    time_slice,
    value_scan,
)


def _assert_error(result):
    assert "error" in result and "summary" in result
    assert result["error"] == result["summary"]


def test_correlation_check_perfect_pair():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [2.0, 4.0, 6.0, 8.0, 10.0]})
    result = correlation_check(df, "a", "b")
    assert result["pearson_r"] == 1.0
    assert result["spearman_r"] == 1.0
    assert result["n"] == 5
    assert "caution" in result["summary"]  # n < 30
    json.dumps(result)


def test_correlation_check_negative_pairwise_complete():
    df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": [-1.0, -2.0, -3.0, -4.0]})
    result = correlation_check(df, "a", "b")
    assert result["n"] == 3  # row 2 dropped: a is NaN there
    assert result["pearson_r"] == -1.0
    assert result["spearman_r"] == -1.0
    json.dumps(result)


def test_correlation_check_guards():
    df = pd.DataFrame({"a": [1.0, 2.0], "t": ["x", "y"], "flag": [True, False]})
    _assert_error(correlation_check(df, "a", "nope"))
    _assert_error(correlation_check(df, "a", "t"))  # string dtype
    _assert_error(correlation_check(df, "a", "flag"))  # boolean dtype
    _assert_error(correlation_check(df, "a", 5))  # non-string column


def test_group_compare_known_means():
    df = pd.DataFrame(
        {"plan": ["vip", "vip", "free", "free"], "spend": [100.0, 100.0, 0.0, 0.0]}
    )
    result = group_compare(df, "spend", "plan")
    groups = {g["category"]: g for g in result["groups"]}
    assert groups["vip"]["mean"] == 100.0
    assert groups["vip"]["median"] == 100.0
    assert groups["vip"]["std"] == 0.0
    assert groups["vip"]["n"] == 2
    assert result["overall"]["n"] == 4
    assert result["overall"]["mean"] == 50.0
    assert result["biggest_gap"]["gap"] == 100.0
    assert result["biggest_gap"]["high_group"] == "vip"
    assert result["biggest_gap"]["low_group"] == "free"
    json.dumps(result)


def test_group_compare_caps_groups():
    df = pd.DataFrame(
        {"cat": [f"c{i:02d}" for i in range(20)], "val": list(range(1, 21))}
    )
    result = group_compare(df, "val", "cat")
    assert result["n_groups_total"] == 20
    assert len(result["groups"]) == 12
    assert result["n_groups_shown"] == 12
    # ties on n=1 break by category name -> shown groups are c00..c11, means 1..12
    assert result["biggest_gap"]["high_group"] == "c11"
    assert result["biggest_gap"]["gap"] == 11.0
    json.dumps(result)


def test_group_compare_guards():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "t": ["x", "y"]})
    _assert_error(group_compare(df, "nope", "t"))
    _assert_error(group_compare(df, "a", "b"))  # categorical column is numeric
    _assert_error(group_compare(df, "t", "t"))  # numeric column is a string


def test_missingness_analysis_divergence():
    df = pd.DataFrame(
        {
            "x": [np.nan, np.nan, 3.0, 4.0],
            "region": ["a", "a", "b", "b"],
        }
    )
    result = missingness_analysis(df, "x")
    assert result["n_missing"] == 2
    assert result["pct_missing"] == 50.0
    assert result["n_sentinel"] == 0
    d = result["biggest_divergence"]
    assert d["column"] == "region"
    assert d["category"] == "a"
    assert d["rate"] == 100.0
    assert d["overall_rate"] == 50.0
    assert d["gap"] == 50.0
    json.dumps(result)


def test_missingness_analysis_sentinels_and_no_missing():
    df = pd.DataFrame({"s": ["n/a", "", "ok", "ok"], "g": ["u", "u", "v", "v"]})
    result = missingness_analysis(df, "s")
    assert result["n_missing"] == 0
    assert result["pct_missing"] == 0.0
    assert result["n_sentinel"] == 2  # 'n/a' and ''
    assert result["splits"] == []
    assert result["biggest_divergence"] is None
    json.dumps(result)


def test_missingness_analysis_caps_splits():
    n = 20
    df = pd.DataFrame(
        {
            "x": [np.nan] * (n // 2) + [1.0] * (n // 2),
            **{f"c{i}": ["m"] * (n // 2) + ["v"] * (n // 2) for i in range(5)},
        }
    )
    result = missingness_analysis(df, "x")
    assert len(result["splits"]) == 3
    assert {s["column"] for s in result["splits"]} == {"c0", "c1", "c2"}
    assert all(s["gap"] == 50.0 for s in result["splits"])


def test_missingness_analysis_missing_column():
    df = pd.DataFrame({"x": [1.0, 2.0]})
    _assert_error(missingness_analysis(df, "nope"))


def test_value_scan_numeric():
    df = pd.DataFrame({"x": [1, 1, 2, 3]})
    result = value_scan(df, "x")
    assert result["dtype"] == "int64"
    assert result["n_unique"] == 3
    assert result["top_values"][0] == {"value": "1", "count": 2}
    assert result["min"] == 1
    assert result["max"] == 3
    assert result["mean"] == 1.75
    json.dumps(result)


def test_value_scan_string_lengths():
    df = pd.DataFrame({"s": ["ab", "cde", "", "f"]})
    result = value_scan(df, "s")
    assert result["n_unique"] == 4
    assert result["min_length"] == 0
    assert result["max_length"] == 3
    assert {"value": "ab", "count": 1} in result["top_values"]
    json.dumps(result)


def test_value_scan_caps_top_values():
    df = pd.DataFrame({"s": [f"v{i:02d}" for i in range(20)]})
    result = value_scan(df, "s")
    assert len(result["top_values"]) == 8
    json.dumps(result)


def test_value_scan_missing_column():
    df = pd.DataFrame({"x": [1.0]})
    _assert_error(value_scan(df, "nope"))


def test_outlier_inspect_known_values():
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0],
            "y": [10.0, 10.0, 10.0, 10.0, 10.0, 50.0],
        }
    )
    result = outlier_inspect(df, "x")
    assert result["lower_bound"] == -1.5
    assert result["upper_bound"] == 8.5
    assert result["n_outliers"] == 1
    assert result["pct_outliers"] == 16.67
    assert result["examples"] == [100.0]
    cmp = {c["column"]: c for c in result["comparisons"]}["y"]
    assert cmp["mean_outliers"] == 50.0
    assert cmp["mean_rest"] == 10.0
    assert cmp["shift_pct"] == 400.0
    json.dumps(result)


def test_outlier_inspect_caps_examples():
    s = list(range(26)) + [1000.0 + i for i in range(6)]
    result = outlier_inspect(pd.DataFrame({"x": s}), "x")
    assert result["n_outliers"] == 6
    assert len(result["examples"]) == 5
    assert result["examples"][0] == 1005.0  # farthest from the median first
    assert result["comparisons"] == []  # no other numeric columns
    json.dumps(result)


def test_outlier_inspect_no_spread():
    result = outlier_inspect(pd.DataFrame({"x": [5.0, 5.0, 5.0]}), "x")
    assert result["n_outliers"] == 0
    assert result["lower_bound"] is None and result["upper_bound"] is None
    json.dumps(result)


def test_outlier_inspect_wrong_dtype():
    df = pd.DataFrame({"x": [1.0, 2.0], "t": ["a", "b"]})
    _assert_error(outlier_inspect(df, "t"))
    _assert_error(outlier_inspect(df, "nope"))


def test_time_slice_monthly():
    df = pd.DataFrame(
        {
            "d": pd.to_datetime(
                ["2024-01-05", "2024-01-20", "2024-02-10", "2024-03-15"]
            ),
            "label": ["a", "b", "c", "d"],
            "v": [10.0, 20.0, 30.0, 40.0],
        }
    )
    result = time_slice(df, "d")  # default freq M
    assert result["freq"] == "M"
    assert result["numeric_column"] == "v"  # first numeric column, skips label
    assert result["n_periods"] == 3
    assert result["periods"][0] == {"period": "2024-01", "n": 2, "mean": 15.0}
    by_period = {p["period"]: p for p in result["periods"]}
    assert by_period["2024-02"]["n"] == 1
    assert by_period["2024-02"]["mean"] == 30.0
    json.dumps(result)


def test_time_slice_weekly():
    df = pd.DataFrame(
        {"d": pd.to_datetime(["2024-01-05", "2024-01-20"]), "v": [1.0, 2.0]}
    )
    result = time_slice(df, "d", freq="W")
    assert result["n_periods"] == 2
    assert result["periods"][0]["period"] == "2024-01-01/2024-01-07"
    json.dumps(result)


def test_time_slice_guards():
    df = pd.DataFrame(
        {
            "d_str": ["2024-01-01", "2024-01-02"],
            "d": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "v": [1.0, 2.0],
        }
    )
    _assert_error(time_slice(df, "d_str"))  # string dates are not datetime dtype
    _assert_error(time_slice(df, "d", freq="X"))
    _assert_error(time_slice(df, "d", freq="monthly"))
    _assert_error(time_slice(df, "nope"))


def test_run_tool_dispatch_matches_direct_call():
    df = pd.DataFrame(
        {"x": [1.0, 2.0], "d": pd.to_datetime(["2024-01-01", "2024-02-01"])}
    )
    assert run_tool(df, "value_scan", {"column": "x"}) == value_scan(df, "x")
    assert run_tool(df, "time_slice", {"date_column": "d", "freq": "W"}) == time_slice(
        df, "d", freq="W"
    )
    assert "error" not in run_tool(df, "time_slice", {"date_column": "d"})


def test_run_tool_guards():
    df = pd.DataFrame({"x": [1.0, 2.0]})
    _assert_error(run_tool(df, "magic", {"column": "x"}))  # unknown tool
    _assert_error(run_tool(df, 7, {}))  # non-string tool name
    _assert_error(run_tool(df, "value_scan", None))  # non-dict args
    _assert_error(run_tool(df, "value_scan", ["x"]))  # non-dict args
    _assert_error(run_tool(df, "value_scan", {}))  # missing required argument
    _assert_error(run_tool(df, "value_scan", {"column": 3}))  # non-string value
    _assert_error(run_tool(df, "value_scan", {"column": "x", "junk": 1}))


def test_tools_reject_empty_and_non_frames():
    for df in (pd.DataFrame(), None, "nope"):
        _assert_error(value_scan(df, "x"))
        _assert_error(correlation_check(df, "a", "b"))
        _assert_error(group_compare(df, "a", "b"))
        _assert_error(missingness_analysis(df, "x"))
        _assert_error(outlier_inspect(df, "x"))
        _assert_error(time_slice(df, "d"))


def test_run_tool_converts_unexpected_crash(monkeypatch):
    import sift_agent.investigate as inv

    def boom(df, column):
        raise RuntimeError("boom")

    monkeypatch.setitem(inv._TOOL_FUNCS, "value_scan", (boom, ("column",)))
    result = run_tool(pd.DataFrame({"x": [1.0]}), "value_scan", {"column": "x"})
    _assert_error(result)
    assert "boom" in result["error"]


def test_run_tool_all_six_json_safe():
    df = pd.DataFrame(
        {
            "d": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-02-01", "2024-02-03", "2024-03-01"]
            ),
            "v": [1.0, 2.0, 3.0, 4.0, 100.0],
            "w": [2.0, 4.0, 6.0, 8.0, 200.0],
            "g": ["a", "a", "b", "b", "c"],
        }
    )
    calls = [
        ("correlation_check", {"column_a": "v", "column_b": "w"}),
        ("group_compare", {"numeric_column": "v", "categorical_column": "g"}),
        ("missingness_analysis", {"column": "v"}),
        ("value_scan", {"column": "g"}),
        ("outlier_inspect", {"column": "v"}),
        ("time_slice", {"date_column": "d", "freq": "M"}),
    ]
    for name, args in calls:
        result = run_tool(df, name, args)
        assert "summary" in result and "error" not in result
        json.dumps(result)


def test_tool_specs_cover_all_six_tools():
    assert len(TOOL_SPECS) == 6
    names = {spec["function"]["name"] for spec in TOOL_SPECS}
    assert names == {
        "correlation_check",
        "group_compare",
        "missingness_analysis",
        "value_scan",
        "outlier_inspect",
        "time_slice",
    }
    for spec in TOOL_SPECS:
        assert spec["type"] == "function"
        fn = spec["function"]
        assert isinstance(fn["description"], str) and len(fn["description"]) > 20
        params = fn["parameters"]
        assert params["type"] == "object"
        assert isinstance(params["properties"], dict) and params["properties"]
        assert isinstance(params["required"], list)
        for prop in params["properties"].values():
            assert prop["type"] == "string"


def test_tool_specs_required_args_and_freq_enum():
    by_name = {s["function"]["name"]: s["function"] for s in TOOL_SPECS}
    assert by_name["correlation_check"]["parameters"]["required"] == [
        "column_a",
        "column_b",
    ]
    assert by_name["group_compare"]["parameters"]["required"] == [
        "numeric_column",
        "categorical_column",
    ]
    ts = by_name["time_slice"]["parameters"]
    assert ts["required"] == ["date_column"]
    assert ts["properties"]["freq"]["enum"] == ["D", "M", "Q", "W", "Y"]
