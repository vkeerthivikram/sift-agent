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
    assert "Scale numeric features" in heuristic_recommendations(p)

    no_num = pd.DataFrame({"b": ["x", "y", "z"]})
    assert "hypothesis-driven" in heuristic_recommendations(profile_dataframe(no_num))
