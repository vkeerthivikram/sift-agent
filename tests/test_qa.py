"""Tests for sift_agent.qa: context retrieval, offline intents, LLM paths."""

import pandas as pd
import pytest

from sift_agent.analysis import profile_dataframe
from sift_agent.qa import answer_question, build_context


class _CannedLLM:
    def __init__(self, text):
        self.text = text

    def invoke(self, messages):
        return type("Response", (), {"content": self.text})()


class _EmptyLLM:
    def invoke(self, messages):
        return type("Response", (), {"content": "   "})()


class _BoomLLM:
    def invoke(self, messages):
        raise RuntimeError("provider down")


def _make_df() -> pd.DataFrame:
    # 30 fixed rows: amount_usd has 3 missing (10%) and min/mean/max of
    # 100/230/360 over its 27 valid values; delivery_days correlates
    # perfectly with it; region has 3 unique values (EU 12, US 8, APAC 8).
    return pd.DataFrame(
        {
            "customer_id": list(range(1, 31)),
            "amount_usd": [100.0 + 10 * i for i in range(27)] + [None, None, None],
            "delivery_days": list(range(1, 31)),
            "region": ["EU"] * 12 + ["US"] * 8 + ["APAC"] * 8 + [None, None],
        }
    )


@pytest.fixture()
def df():
    return _make_df()


@pytest.fixture()
def profile(df):
    return profile_dataframe(df)


# --- build_context ---------------------------------------------------------


def test_build_context_column_match_with_space_tolerance(df, profile):
    context, sources = build_context("tell me about delivery days", df, profile)
    assert "delivery_days" in context
    assert "column: delivery_days" in sources


def test_build_context_always_includes_shape_and_sources(df, profile):
    context, sources = build_context("hello there", df, profile)
    assert context.startswith("Dataset:")
    assert "30 rows x 4 columns" in context
    assert sources


def test_build_context_respects_char_cap(df, profile):
    question = (
        "quality issues, outliers, duplicates, correlations, and missing values "
        "for amount_usd delivery_days region customer_id"
    )
    context, sources = build_context(question, df, profile, max_chars=300)
    assert len(context) <= 300
    assert "[context truncated]" in context
    assert sources[0] == "section: overview"
    assert "section: correlations" not in sources


def test_build_context_routes_correlation_section(df, profile):
    context, sources = build_context("are there any correlations?", df, profile)
    assert "section: correlations" in sources
    assert "Top Pearson correlations" in context
    assert "r=+1.000" in context


def test_build_context_routes_missing_section(df, profile):
    context, sources = build_context("which columns have missing values?", df, profile)
    assert "section: missing" in sources
    assert "amount_usd: 3 missing (10%)" in context


def test_build_context_routes_outlier_section(df, profile):
    context, sources = build_context("any extreme or odd values?", df, profile)
    assert "section: outliers" in sources
    assert "Columns with IQR outliers:" in context


def test_build_context_routes_quality_and_duplicate_sections(df, profile):
    _, sources = build_context("is the data dirty?", df, profile)
    assert "section: data_quality" in sources
    _, sources = build_context("do we have duplicate rows?", df, profile)
    assert "section: duplicates" in sources


# --- offline intents -------------------------------------------------------


def test_offline_missing_intent(df, profile):
    result = answer_question("How many rows are missing amount_usd?", df, profile)
    assert result["offline"] is True
    assert "`amount_usd`" in result["answer"]
    assert "3 of 30 values missing (10%)" in result["answer"]
    assert result["sources"] == ["column: amount_usd"]


def test_offline_missing_intent_space_tolerance(df, profile):
    result = answer_question("how many rows are missing delivery days?", df, profile)
    assert result["offline"] is True
    assert "`delivery_days`: 0 of 30 values missing (0%)." in result["answer"]


def test_offline_rows_intent(df, profile):
    result = answer_question("How many rows are in the dataset?", df, profile)
    assert result["offline"] is True
    assert "The dataset has 30 rows x 4 columns." in result["answer"]


def test_offline_stat_intents(df, profile):
    avg = answer_question("What is the average amount_usd?", df, profile)
    assert "`amount_usd`: average 230 (27 non-null values)" in avg["answer"]

    mx = answer_question("What's the maximum amount?", df, profile)
    assert "`amount_usd`: maximum 360" in mx["answer"]

    mn = answer_question("WHAT IS THE MINIMUM AMOUNT_USD?", df, profile)
    assert "minimum 100" in mn["answer"]


def test_offline_unique_intent(df, profile):
    result = answer_question("What are the unique values in region?", df, profile)
    assert "`region` has 3 unique values" in result["answer"]
    assert "EU (12)" in result["answer"]
    assert result["sources"] == ["column: region"]


def test_offline_correlation_intent(df, profile):
    q = "What is the correlation between amount_usd and delivery_days?"
    result = answer_question(q, df, profile)
    assert "+1.000" in result["answer"]
    assert "amount_usd" in result["answer"]
    assert "delivery_days" in result["answer"]
    assert result["sources"] == ["column: amount_usd", "column: delivery_days"]


def test_offline_dtype_intent(df, profile):
    result = answer_question("What is the dtype of delivery_days?", df, profile)
    assert "int64" in result["answer"]
    assert "column: delivery_days" in result["sources"]


def test_offline_duplicates_intent(df, profile):
    result = answer_question("Are there any duplicates?", df, profile)
    assert "Duplicate rows: 0 of 30 rows." in result["answer"]

    df_dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    dup_profile = profile_dataframe(df_dup)
    result2 = answer_question("Are there any duplicates?", df_dup, dup_profile)
    assert "Duplicate rows: 1 of 31 rows." in result2["answer"]


def test_offline_unmatched_question_best_effort(df, profile):
    result = answer_question("What is the meaning of life?", df, profile)
    assert result["offline"] is True
    assert result["answer"].startswith("Offline mode")
    assert result["answer"].strip()
    assert result["sources"]


def test_offline_non_numeric_pair_never_fabricates(df, profile):
    q = "Is there a correlation between region and delivery_days?"
    result = answer_question(q, df, profile)
    assert result["offline"] is True
    assert result["answer"].startswith("Offline mode")
    assert "column: region" in result["sources"]


def test_offline_determinism(df, profile):
    q = "What is the average amount_usd?"
    assert answer_question(q, df, profile) == answer_question(q, df, profile)


# --- online (LLM) paths ----------------------------------------------------


def test_online_canned_llm(df, profile):
    canned = "The `amount_usd` column has 3 missing values (10%)."
    result = answer_question("what is missing?", df, profile, llm=_CannedLLM(canned))
    assert result["answer"] == canned
    assert result["offline"] is False
    assert result["sources"]
    assert "warning" not in result


def test_online_llm_failure_falls_back_offline(df, profile):
    q = "How many rows are missing amount_usd?"
    result = answer_question(q, df, profile, llm=_BoomLLM())
    assert result["offline"] is True
    assert "3 of 30 values missing" in result["answer"]
    assert "RuntimeError" in result["warning"]


def test_online_empty_llm_falls_back_offline(df, profile):
    q = "How many rows are in the dataset?"
    result = answer_question(q, df, profile, llm=_EmptyLLM())
    assert result["offline"] is True
    assert "30 rows" in result["answer"]
    assert "ValueError" in result["warning"]
