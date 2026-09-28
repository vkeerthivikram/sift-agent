import numpy as np
import pandas as pd

from sift_agent.analysis import profile_dataframe
from sift_agent.graph import heuristic_insights, heuristic_recommendations


def _profile() -> dict:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "id": [f"c{i:04d}" for i in range(200)],
            "score": np.append(rng.normal(50, 5, 195), [1000, 1050, 1100, 1150, 1200]),
            "visits": rng.poisson(3, 200).astype(float),
            "grade": ["A"] * 195 + [None] * 5,
        }
    )
    df.loc[199, "score"] = df.loc[198, "score"]
    df.loc[199, "visits"] = df.loc[198, "visits"]
    df.loc[199, "grade"] = df.loc[198, "grade"]
    df.loc[199, "id"] = df.loc[198, "id"]
    return profile_dataframe(df)


def test_heuristic_insights_mention_dataset_shape():
    p = _profile()
    text = heuristic_insights(p)
    assert "200 rows x 4 columns" in text
    assert "`score`" in text


def test_heuristic_insights_cover_quality_and_skew():
    p = _profile()
    text = heuristic_insights(p)
    assert "duplicate rows" in text
    assert "Missing values" in text
    assert "skew" in text


def test_heuristic_insights_deterministic():
    p = _profile()
    assert heuristic_insights(p) == heuristic_insights(p)


def test_heuristic_recommendations_deterministic_and_specific():
    p = _profile()
    text = heuristic_recommendations(p)
    assert text == heuristic_recommendations(p)
    assert "drop_duplicates" in text
    assert "`grade`" in text


def test_heuristics_on_clean_profile():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]})
    p = profile_dataframe(df)
    assert "No missing values detected" in heuristic_insights(p)
    recs = heuristic_recommendations(p)
    assert "Scale numeric features" not in recs  # boilerplate removed
    assert "hypothesis-driven" in recs

    no_num = pd.DataFrame({"b": ["x", "y", "z"]})
    assert "hypothesis-driven" in heuristic_recommendations(profile_dataframe(no_num))


def test_heuristic_recommendations_data_aware():
    df = pd.DataFrame(
        {
            "txt_num": ["1", "2", "3", "n/a"] * 3,
            "uid": [f"u{i}" for i in range(12)],
        }
    )
    p = profile_dataframe(df)
    recs = heuristic_recommendations(p)
    assert "pd.to_numeric" in recs
    assert "ID-like" in recs or "ID" in recs


def test_heuristics_flag_pii_columns():
    df = pd.DataFrame(
        {
            "email": ["a@x.com", "b@y.com", "c@z.com", "d@w.com"],
            "value": [1, 2, 3, 4],
        }
    )
    p = profile_dataframe(df)
    assert "PII" in heuristic_insights(p)
    assert "PII" in heuristic_recommendations(p)


def test_heuristic_insights_cover_new_sections():
    rng = np.random.default_rng(9)
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    region = rng.choice(["N", "S"], 60)
    df = pd.DataFrame(
        {
            "day": dates,
            "sales": rng.normal(100, 5, 60) + np.where(region == "N", 100.0, 0.0),
            "region": region,
        }
    )
    p = profile_dataframe(df)
    insights = heuristic_insights(p)
    assert "`day` spans" in insights
    assert "`sales` varies most by `region`" in insights
