import numpy as np
import pandas as pd

from sift_agent.visualize import generate_charts


def _df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "a": np.append(rng.normal(0, 1, 98), [10.0, -10.0]),
            "b": rng.normal(5, 2, 100),
            "cat": rng.choice(["x", "y", "z"], 100),
            "with_nan": np.where(rng.random(100) < 0.1, np.nan, 1.0),
        }
    )


def test_generate_charts_writes_pngs_and_refs(tmp_path):
    refs = generate_charts(_df(), tmp_path / "charts")
    assert refs, "expected at least one chart"
    pngs = list((tmp_path / "charts").glob("*.png"))
    assert len(pngs) == len(refs)
    for ref in refs:
        assert ref["title"] and ref["caption"]
        assert (tmp_path / ref["path"]).stat().st_size > 1000


def test_generate_charts_skips_inapplicable(tmp_path):
    df = pd.DataFrame({"t": ["a", "b", "c", "a"]})
    refs = generate_charts(df, tmp_path / "charts")
    titles = [r["title"] for r in refs]
    assert "Numeric Distributions" not in titles
    assert "Correlation Heatmap" not in titles


def test_generate_charts_records_failures(tmp_path, monkeypatch):
    import sift_agent.visualize as viz

    def boom(df, charts_dir, numeric):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(viz, "_correlation_heatmap", boom)
    failures: list[str] = []
    refs = generate_charts(_df(), tmp_path / "charts", failures=failures)
    assert any("correlation_heatmap" in f and "kaboom" in f for f in failures)
    assert refs, "other charts should still render"
