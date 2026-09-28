import json

import numpy as np
import pandas as pd
import pytest

from sift_agent.analysis import profile_dataframe
from sift_agent.scoring import health_score


def _clean_profile():
    return {
        "n_rows": 100,
        "n_columns": 3,
        "duplicate_rows": 0,
        "columns": {
            "id": {"pct_missing": 0.0},
            "amount": {"pct_missing": 0.0},
            "region": {"pct_missing": 0.0},
        },
        "constant_columns": [],
        "data_quality": [],
        "top_missing": [],
    }


def _dirty_profile():
    columns = {f"c{i}": {"pct_missing": 0.0} for i in range(11)}
    columns["age"] = {"pct_missing": 18.0}
    return {
        "n_rows": 1000,
        "n_columns": 12,
        "duplicate_rows": 45,
        "columns": columns,
        "constant_columns": [],
        "data_quality": [
            {"check": "whitespace", "column": "c0"},
            {"check": "whitespace", "column": "c1"},
            {"check": "mixed_case", "column": "c2"},
            {"check": "numeric_as_text", "column": "c3"},
            {"check": "dates_as_text", "column": "c4"},
            {"check": "sentinels", "column": "c5"},
            {"check": "numeric_sentinel", "column": "c6"},
        ],
        "top_missing": [{"column": "age", "n_missing": 180, "pct_missing": 18.0}],
    }


def _boundary_profile(
    duplicate_rows,
    validity_columns=0,
    consistency_columns=0,
    constant_columns=0,
    missing_pcts=(),
):
    n_columns = 10
    columns = {
        f"c{i}": {
            "pct_missing": float(missing_pcts[i]) if i < len(missing_pcts) else 0.0
        }
        for i in range(n_columns)
    }
    return {
        "n_rows": 100,
        "n_columns": n_columns,
        "duplicate_rows": duplicate_rows,
        "columns": columns,
        "constant_columns": [f"k{i}" for i in range(constant_columns)],
        "data_quality": [
            *(
                {"check": "numeric_as_text", "column": f"c{i}"}
                for i in range(validity_columns)
            ),
            *(
                {"check": "whitespace", "column": f"w{i}"}
                for i in range(consistency_columns)
            ),
        ],
    }


def _component(result, key):
    return next(c for c in result["components"] if c["key"] == key)


def test_clean_profile_is_grade_a():
    result = health_score(_clean_profile())
    assert result["score"] >= 95
    assert result["grade"] == "A"
    assert all(c["score"] == 100 for c in result["components"])


def test_completeness_blends_mean_and_worst():
    profile = _clean_profile()
    profile["n_columns"] = 4
    profile["columns"] = {
        "a": {"pct_missing": 0.0},
        "b": {"pct_missing": 0.0},
        "c": {"pct_missing": 30.0},
        "d": {"pct_missing": 30.0},
    }
    comp = _component(health_score(profile), "completeness")
    assert comp["score"] == 78  # 100 - (mean 15 + worst 30)/2
    assert "Average 15% missing across 4 columns" in comp["detail"]
    assert "worst is `c` at 30%" in comp["detail"]


def test_uniqueness_penalizes_duplicates():
    profile = _clean_profile()
    profile["n_rows"] = 200
    profile["duplicate_rows"] = 30
    comp = _component(health_score(profile), "uniqueness")
    assert comp["score"] == 85  # 100 - 100*30/200
    assert "30 of 200 rows are duplicates (15%)" in comp["detail"]


def test_consistency_counts_distinct_formatting_columns():
    profile = _clean_profile()
    profile["n_columns"] = 4
    profile["columns"] = {k: {"pct_missing": 0.0} for k in "abcd"}
    profile["constant_columns"] = ["c"]
    profile["data_quality"] = [
        {"check": "whitespace", "column": "a"},
        {"check": "mixed_case", "column": "a"},  # same column: counted once
        {"check": "mixed_case", "column": "b"},
        {"check": "numeric_as_text", "column": "d"},  # validity, not consistency
    ]
    comp = _component(health_score(profile), "consistency")
    assert comp["score"] == 55  # 100 - 15 * (2 distinct + 1 constant)
    assert "3 of 4 columns have formatting issues" in comp["detail"]
    assert "whitespace: 1" in comp["detail"]
    assert "mixed case: 2" in comp["detail"]
    assert "constant: 1" in comp["detail"]


def test_validity_counts_distinct_type_and_sentinel_columns():
    profile = _clean_profile()
    profile["n_columns"] = 5
    profile["columns"] = {k: {"pct_missing": 0.0} for k in "abcde"}
    profile["data_quality"] = [
        {"check": "numeric_as_text", "column": "a"},
        {"check": "dates_as_text", "column": "a"},  # same column: counted once
        {"check": "sentinels", "column": "b"},
        {"check": "numeric_sentinel", "column": "c"},
        {"check": "whitespace", "column": "d"},  # consistency, not validity
        {"check": "unknown_check", "column": "e"},  # not a scoring check
    ]
    comp = _component(health_score(profile), "validity")
    assert comp["score"] == 55  # 100 - 15 * 3 distinct columns
    assert "3 of 5 columns have validity issues" in comp["detail"]


def test_weighted_sum_is_exact():
    profile = _clean_profile()
    profile["n_rows"] = 100
    profile["n_columns"] = 4
    profile["duplicate_rows"] = 50  # uniqueness 50
    profile["columns"] = {k: {"pct_missing": 0.0} for k in "abcd"}
    profile["data_quality"] = [
        {"check": "whitespace", "column": "a"},
        {"check": "whitespace", "column": "b"},  # consistency 100 - 2*15 = 70
        {"check": "sentinels", "column": "c"},  # validity 100 - 15 = 85
    ]
    result = health_score(profile)
    assert {c["key"]: c["score"] for c in result["components"]} == {
        "completeness": 100,
        "uniqueness": 50,
        "consistency": 70,
        "validity": 85,
    }
    assert result["score"] == 81  # 0.4*100 + 0.2*50 + 0.2*70 + 0.2*85


def test_pathological_profile_clamps_to_zero():
    profile = {
        "n_rows": 10,
        "n_columns": 7,
        "duplicate_rows": 15,  # more duplicates than rows
        "columns": {k: {"pct_missing": 100.0} for k in "abcdefg"},
        "constant_columns": list("abcdefg"),
        "data_quality": [
            {"check": check, "column": col}
            for col in "abcdefg"
            for check in ("whitespace", "mixed_case", "numeric_as_text", "sentinels")
        ],
    }
    result = health_score(profile)
    assert all(c["score"] == 0 for c in result["components"])
    assert result["score"] == 0
    assert result["grade"] == "F"


def test_missing_optional_keys_never_raise():
    minimal = {
        "n_rows": 5,
        "n_columns": 2,
        "duplicate_rows": 1,
        "columns": {"x": {}, "y": {"pct_missing": 20.0}},  # x lacks pct_missing
    }
    result = health_score(minimal)
    # completeness 100 - (10+20)/2 = 85, uniqueness 100 - 20 = 80
    assert result["score"] == 90  # 0.4*85 + 0.2*80 + 0.2*100 + 0.2*100
    assert result["grade"] == "A"

    empty_columns = dict(minimal, columns={}, n_columns=0)
    result = health_score(empty_columns)
    assert _component(result, "completeness")["score"] == 100

    none_optionals = dict(
        minimal, data_quality=None, constant_columns=None, top_missing=None
    )
    assert health_score(none_optionals)["score"] == 90

    bare = health_score({})
    assert bare["score"] == 100
    assert bare["grade"] == "A"
    assert len(bare["components"]) == 4


def test_n_columns_falls_back_to_columns_dict():
    profile = _clean_profile()
    del profile["n_columns"]
    profile["columns"] = {"a": {"pct_missing": 0.0}, "b": {"pct_missing": 0.0}}
    profile["data_quality"] = [{"check": "whitespace", "column": "a"}]
    comp = _component(health_score(profile), "consistency")
    assert comp["score"] == 85  # one flagged column: 100 - 15
    assert "1 of 2 columns have formatting issues" in comp["detail"]


@pytest.mark.parametrize(
    ("profile", "score", "grade"),
    [
        (_boundary_profile(50), 90, "A"),
        (_boundary_profile(55), 89, "B"),
        (_boundary_profile(100), 80, "B"),
        (_boundary_profile(60, validity_columns=3, consistency_columns=3), 70, "C"),
        (_boundary_profile(85, validity_columns=7, consistency_columns=1), 60, "D"),
        (
            _boundary_profile(
                85, validity_columns=7, consistency_columns=1, missing_pcts=[4.55]
            ),
            59,
            "F",
        ),
    ],
    ids=["90-A", "89-B", "80-B", "70-C", "60-D", "59-F"],
)
def test_grade_boundaries(profile, score, grade):
    result = health_score(profile)
    assert result["score"] == score
    assert result["grade"] == grade


def test_verdict_cites_real_counts():
    result = health_score(_dirty_profile())
    assert result["verdict"] == (
        "Grade C (74/100): 45 duplicate rows, 7 of 12 columns with quality issues, "
        "worst missingness `age` at 18%."
    )


def test_verdict_clean_profile():
    result = health_score(_clean_profile())
    assert result["verdict"] == (
        "Grade A (100/100): no duplicates, missing values, or quality issues detected."
    )


def test_verdict_skips_zero_clauses():
    profile = _clean_profile()
    profile["n_columns"] = 4
    profile["columns"]["m"] = {"pct_missing": 50.0}
    result = health_score(profile)
    assert result["verdict"] == "Grade B (88/100): worst missingness `m` at 50%."
    assert "duplicate" not in result["verdict"]
    assert "quality issues" not in result["verdict"]


def test_components_shape_and_order():
    result = health_score(_clean_profile())
    assert [c["key"] for c in result["components"]] == [
        "completeness",
        "uniqueness",
        "consistency",
        "validity",
    ]
    for comp in result["components"]:
        assert set(comp) == {"key", "label", "score", "detail"}
        assert isinstance(comp["score"], int) and 0 <= comp["score"] <= 100
        assert isinstance(comp["detail"], str) and comp["detail"]


def test_result_is_json_serializable():
    json.dumps(health_score(_dirty_profile()))


def test_deterministic():
    profile = _dirty_profile()
    assert health_score(profile) == health_score(profile)


def test_health_score_on_real_profile():
    rng = np.random.default_rng(7)
    clean = pd.DataFrame(
        {
            "id": [f"r{i:03d}" for i in range(50)],
            "value": rng.normal(0, 1, 50),
            "group": rng.choice(["north", "south"], 50),
        }
    )
    result = health_score(profile_dataframe(clean))
    assert result["grade"] == "A"
    assert result["score"] >= 95

    dirty = pd.DataFrame(
        {
            "id": ["a", "a", "b", None, None],
            "num": ["1", "2", "n/a", " 4", "n/a"],
            "cat": [" X", "X", "x", "Y", None],
        }
    )
    dirty_result = health_score(profile_dataframe(dirty))
    assert dirty_result["score"] < result["score"]
    assert dirty_result["grade"] != "A"
