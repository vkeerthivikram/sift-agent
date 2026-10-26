"""Tests for sift_agent.query: plan validation, execution, dates, grammar."""

from datetime import datetime, UTC

import pandas as pd
import pytest

from sift_agent.query import (
    execute_plan,
    parse_question,
    resolve_token,
    validate_plan,
)

NOW = datetime(2026, 3, 30, 12, 0, tzinfo=UTC)  # a Monday


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region": ["EU", "EU", "US", "US", "APAC", "APAC"],
            "amount": [100.0, 200.0, 400.0, 500.0, 50.0, None],
            "day": pd.to_datetime(
                [
                    "2026-02-15",
                    "2026-03-05",
                    "2026-03-29",
                    "2026-03-30",
                    "2025-12-01",
                    "2026-04-02",
                ]
            ),
        }
    )


# --- token resolution -------------------------------------------------------


def test_resolve_tokens_utc():
    assert resolve_token("@now", now=NOW) == pd.Timestamp(
        "2026-03-30 12:00:00", tz="UTC"
    )
    assert resolve_token("@day_start", now=NOW) == pd.Timestamp(
        "2026-03-30"
    ).tz_localize("UTC")
    assert resolve_token("@week_start", now=NOW) == pd.Timestamp(
        "2026-03-30"
    ).tz_localize("UTC")
    assert resolve_token("@last_week_start", now=NOW) == pd.Timestamp(
        "2026-03-23"
    ).tz_localize("UTC")
    assert resolve_token("@month_start", now=NOW) == pd.Timestamp(
        "2026-03-01"
    ).tz_localize("UTC")
    assert resolve_token("@last_month_start", now=NOW) == pd.Timestamp(
        "2026-02-01"
    ).tz_localize("UTC")
    assert resolve_token("@quarter_start", now=NOW) == pd.Timestamp(
        "2026-01-01"
    ).tz_localize("UTC")
    assert resolve_token("@last_quarter_start", now=NOW) == pd.Timestamp(
        "2025-10-01"
    ).tz_localize("UTC")
    assert resolve_token("@year_start", now=NOW) == pd.Timestamp(
        "2026-01-01"
    ).tz_localize("UTC")
    assert resolve_token("@last_year_start", now=NOW) == pd.Timestamp(
        "2025-01-01"
    ).tz_localize("UTC")
    assert resolve_token("@last_7_days_start", now=NOW) == pd.Timestamp(
        "2026-03-24"
    ).tz_localize("UTC")


def test_resolve_tokens_new_york_across_dst():
    tz = "America/New_York"
    assert resolve_token("@last_month_start", tz=tz, now=NOW) == pd.Timestamp(
        "2026-02-01 00:00:00", tz="America/New_York"
    )
    assert resolve_token("@month_start", tz=tz, now=NOW) == pd.Timestamp(
        "2026-03-01 00:00:00", tz="America/New_York"
    )
    assert resolve_token("@day_start", tz=tz, now=NOW) == pd.Timestamp(
        "2026-03-30 00:00:00", tz="America/New_York"
    )


def test_resolve_token_rejects_unknown():
    with pytest.raises(ValueError, match="unknown date token"):
        resolve_token("@someday", now=NOW)


# --- validation -------------------------------------------------------------


def test_validate_rejects_unknown_metric_column():
    errors = validate_plan({"aggregate": "mean", "column": "nope"}, _df())
    assert any("not found" in e for e in errors)


def test_validate_rejects_non_numeric_mean():
    errors = validate_plan({"aggregate": "mean", "column": "region"}, _df())
    assert any("numeric" in e for e in errors)


def test_validate_rejects_string_value_on_numeric_column():
    plan = {
        "aggregate": "mean",
        "column": "amount",
        "filters": {"op": ">", "column": "amount", "value": "big"},
    }
    errors = validate_plan(plan, _df())
    assert any("must be a number" in e for e in errors)


def test_validate_rejects_ordered_compare_on_categorical():
    plan = {
        "aggregate": "mean",
        "column": "amount",
        "filters": {"op": ">", "column": "region", "value": 1},
    }
    errors = validate_plan(plan, _df())
    assert any("numeric or datetime" in e for e in errors)


def test_validate_rejects_count_with_column_and_bad_group():
    df = _df()
    assert any(
        "'count' counts matching rows" in e
        for e in validate_plan({"aggregate": "count", "column": "amount"}, df)
    )
    assert any(
        "not found" in e
        for e in validate_plan({"aggregate": "count", "group_by": ["missing"]}, df)
    )


def test_validate_rejects_date_value_that_is_not_iso():
    plan = {
        "aggregate": "count",
        "filters": {"op": ">=", "column": "day", "value": "yesterday-ish"},
    }
    errors = validate_plan(plan, _df())
    assert any("ISO 8601" in e for e in errors)


def test_validate_bounds_filter_tree_and_group_columns():
    df = _df()
    leaf = {"op": "is null", "column": "region"}
    deep = leaf
    for _ in range(45):
        deep = {"op": "or", "children": [deep, leaf]}
    errors = validate_plan({"aggregate": "count", "filters": deep}, df)
    assert any("exceeds" in e for e in errors)
    assert any(
        "at most 3" in e
        for e in validate_plan(
            {"aggregate": "count", "group_by": ["region", "amount", "day", "day"]}, df
        )
    )


# --- execution --------------------------------------------------------------


def test_execute_equality_filter_mean():
    plan = {
        "aggregate": "mean",
        "column": "amount",
        "filters": {"op": "=", "column": "region", "value": "EU"},
    }
    result = execute_plan(_df(), plan, now=NOW)
    assert result["ok"]
    assert result["value"] == 150.0
    assert result["n_matched"] == 2
    assert result["n_used"] == 2
    assert "`region` = 'EU'" in result["interpretation"]


def test_execute_boolean_tree_with_in():
    plan = {
        "aggregate": "sum",
        "column": "amount",
        "filters": {
            "op": "and",
            "children": [
                {"op": "in", "column": "region", "values": ["EU", "US"]},
                {"op": ">=", "column": "amount", "value": 200},
            ],
        },
    }
    result = execute_plan(_df(), plan, now=NOW)
    assert result["ok"]
    assert result["value"] == 1100.0


def test_execute_not_between_and_null():
    df = _df()
    plan = {
        "aggregate": "count",
        "filters": {
            "op": "and",
            "children": [
                {
                    "op": "not",
                    "child": {
                        "op": "between",
                        "column": "day",
                        "low": "2026-03-01",
                        "high": "2026-03-31",
                    },
                },
                {"op": "is null", "column": "amount"},
            ],
        },
    }
    result = execute_plan(df, plan, now=NOW)
    assert result["ok"]
    assert result["value"] == 1  # the APAC row on 2025-12-01 with amount NaN


def test_execute_grouped_sum():
    plan = {
        "aggregate": "sum",
        "column": "amount",
        "group_by": ["region"],
    }
    result = execute_plan(_df(), plan, now=NOW)
    assert result["ok"]
    by_region = {r["group"]["region"]: r["value"] for r in result["groups"]}
    assert by_region == {"APAC": 50.0, "EU": 300.0, "US": 900.0}
    ns = {r["group"]["region"]: r["n"] for r in result["groups"]}
    assert ns == {"APAC": 1, "EU": 2, "US": 2}


def test_execute_relative_date_filter():
    plan = {
        "aggregate": "count",
        "filters": {"op": ">=", "column": "day", "value": "@month_start"},
    }
    result = execute_plan(_df(), plan, tz="UTC", now=NOW)
    assert result["ok"]
    assert result["value"] == 4  # 03-05, 03-29, 03-30 and 04-02 are all >= Mar 1
    assert "@month_start=2026-03-01" in result["interpretation"]


def test_execute_returns_error_not_raise_for_bad_plan():
    result = execute_plan(_df(), {"aggregate": "mean", "column": "nope"})
    assert not result["ok"]
    assert result["error"]


# --- offline grammar --------------------------------------------------------


def test_parse_question_for_equality():
    plan = parse_question(
        "What is the average amount for region EU?", _df(), ["amount", "region"]
    )
    assert plan == {
        "aggregate": "mean",
        "column": "amount",
        "filters": {"op": "=", "column": "region", "value": "EU"},
    }
    result = execute_plan(_df(), plan, now=NOW)
    assert result["value"] == 150.0


def test_parse_question_where_in_and_group_by():
    plan = parse_question(
        "sum amount where region in (EU, US) by region", _df(), ["amount"]
    )
    assert plan is not None
    assert plan["group_by"] == ["region"]
    result = execute_plan(_df(), plan, now=NOW)
    by_region = {r["group"]["region"]: r["value"] for r in result["groups"]}
    assert by_region == {"EU": 300.0, "US": 900.0}


def test_parse_question_where_boolean_with_parens():
    plan = parse_question(
        "count where (region = US or amount > 150) and amount is not null",
        _df(),
        [],
    )
    assert plan is not None
    result = execute_plan(_df(), plan, now=NOW)
    assert result["value"] == 3  # rows 1 (EU,200), 2 (US,400), 3 (US,500)


def test_parse_question_last_month_phrase():
    plan = parse_question("how many rows in last month", _df(), [])
    assert plan is not None
    result = execute_plan(_df(), plan, tz="UTC", now=NOW)
    assert result["value"] == 1  # 2026-02-15


def test_parse_question_between_iso_dates():
    plan = parse_question("count between 2026-03-01 and 2026-03-31", _df(), [])
    assert plan is not None
    result = execute_plan(_df(), plan, tz="UTC", now=NOW)
    assert result["value"] == 3


def test_parse_question_returns_none_for_plain_aggregate():
    assert parse_question("What is the average amount?", _df(), ["amount"]) is None


def test_parse_question_returns_none_for_unresolvable_group():
    assert parse_question("sum amount by nope", _df(), ["amount"]) is None


def test_parse_question_returns_none_without_aggregate():
    assert parse_question("region for EU", _df(), []) is None
