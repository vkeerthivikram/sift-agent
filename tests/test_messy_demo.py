from pathlib import Path

import pytest

from sift_agent.analysis import id_like_columns, profile_dataframe
from sift_agent.loader import load_table

MESSY_CSV = Path(__file__).resolve().parents[1] / "examples" / "messy_orders.csv"


@pytest.fixture(scope="module")
def messy():
    df, _ = load_table(MESSY_CSV)
    return df, profile_dataframe(df)


def _issues(profile):
    return {(i["check"], i["column"]) for i in profile["data_quality"]}


def test_messy_csv_exists_with_enough_rows():
    assert MESSY_CSV.exists()
    df, _ = load_table(MESSY_CSV)
    assert len(df) >= 500


def test_exactly_ten_duplicate_rows(messy):
    _, profile = messy
    assert profile["duplicate_rows"] == 10


def test_dates_as_text_on_order_date(messy):
    _, profile = messy
    assert ("dates_as_text", "order_date") in _issues(profile)


def test_numeric_sentinel_on_discount_pct(messy):
    _, profile = messy
    assert ("numeric_sentinel", "discount_pct") in _issues(profile)


def test_mixed_case_on_customer_region(messy):
    _, profile = messy
    assert ("mixed_case", "customer_region") in _issues(profile)


def test_whitespace_on_notes(messy):
    _, profile = messy
    assert ("whitespace", "notes") in _issues(profile)


def test_channel_is_clean(messy):
    _, profile = messy
    assert not any(column == "channel" for _, column in _issues(profile))


def test_order_id_is_id_like(messy):
    # id_like_columns demands nunique == n_rows, which no column can satisfy
    # while verbatim duplicate rows exist (every column repeats in a dup pair),
    # so the identifier property is asserted on the deduplicated frame.
    df, profile = messy
    dedup = df.drop_duplicates(keep="first")
    assert {"column": "order_id", "reason": "unique per row"} in id_like_columns(dedup)
    assert profile["duplicate_rows"] == len(df) - len(dedup)


def test_revenue_cost_strongly_correlated(messy):
    _, profile = messy
    pair = next(
        (
            p
            for p in profile["top_correlated_pairs"]
            if {p["col_a"], p["col_b"]} == {"revenue", "cost"}
        ),
        None,
    )
    assert pair is not None
    assert abs(pair["pearson_r"]) >= 0.9


def test_delivery_days_in_top_missing(messy):
    _, profile = messy
    assert any(m["column"] == "delivery_days" for m in profile["top_missing"])


def test_amount_usd_skewed_with_outliers(messy):
    _, profile = messy
    amt = profile["columns"]["amount_usd"]
    assert amt["skew"] > 1
    assert amt["n_outliers_iqr"] > 0
