import base64
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from sift_agent.html_report import write_html_report


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = Figure(figsize=(1.0, 1.0), dpi=30)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(10))
    fig.savefig(path, format="png")


def _state(charts: list[dict]) -> dict:
    return {
        "input_path": "orders/messy_orders.csv",
        "provider": "none",
        "model": "",
        "insights": "- bullet with **bold** and `code`",
        "recommendations": "## Next steps\n- clean `price` first",
        "warnings": ["chart skipped: <weird> & co"],
        "charts": charts,
        "profile": {
            "n_rows": 120,
            "n_columns": 4,
            "n_numeric_columns": 2,
            "n_categorical_columns": 2,
            "duplicate_rows": 3,
            "memory_mb": 0.42,
            "top_missing": [
                {
                    "column": "<script>alert(1)</script> & co",
                    "n_missing": 12,
                    "pct_missing": 10.0,
                }
            ],
            "data_quality": [
                {
                    "check": "mixed_case",
                    "column": "region",
                    "detail": "5 distinct values collapse to 3 when lowercased",
                }
            ],
            "outlier_columns": [
                {
                    "column": "amount",
                    "count": 7,
                    "pct": 5.8,
                    "examples": [1500.0, 2200.0],
                }
            ],
            "top_correlated_pairs": [
                {"col_a": "price", "col_b": "amount", "pearson_r": 0.93}
            ],
            "spearman_pairs": [
                {"col_a": "price", "col_b": "amount", "spearman_r": 0.88}
            ],
            "cramers_v_pairs": [
                {"col_a": "region", "col_b": "channel", "cramers_v": 0.71}
            ],
            "columns": {
                "price": {
                    "type": "numeric",
                    "dtype": "float64",
                    "pct_missing": 0.0,
                    "n_unique": 100,
                    "mean": 10.5,
                    "median": 10.0,
                    "std": 2.0,
                    "skew": 0.1,
                },
                "<script>alert(1)</script> & co": {
                    "type": "categorical",
                    "dtype": "object",
                    "pct_missing": 10.0,
                    "n_unique": 3,
                    "value_counts": [{"value": "n/a", "count": 12}],
                },
            },
            "health": {
                "score": 87,
                "grade": "B",
                "verdict": "Solid dataset with a few fixable issues.",
                "components": [
                    {
                        "key": "completeness",
                        "label": "Completeness",
                        "score": 90,
                        "detail": "mean missingness 5%",
                    },
                    {
                        "key": "validity",
                        "label": "Validity",
                        "score": 64,
                        "detail": "1 column with sentinel placeholders",
                    },
                ],
            },
        },
    }


def test_full_state_renders_all_sections(tmp_path):
    _write_png(tmp_path / "charts" / "distributions.png")
    _write_png(tmp_path / "charts" / "heatmap.png")
    state = _state(
        [
            {
                "title": "Numeric Distributions",
                "path": "charts/distributions.png",
                "caption": "histos & boxes",
            },
            {
                "title": "Correlation Heatmap",
                "path": "charts/heatmap.png",
                "caption": "pearson matrix",
            },
            {
                "title": "Ghost Chart",
                "path": "charts/ghost.png",
                "caption": "missing on disk",
            },
        ]
    )

    result = write_html_report(state, tmp_path)

    assert isinstance(result, str)
    report = Path(result)
    assert report == tmp_path / "report.html"
    assert report.is_file()
    page = report.read_text(encoding="utf-8")

    expected = base64.b64encode(
        (tmp_path / "charts" / "distributions.png").read_bytes()
    ).decode("ascii")
    assert f"data:image/png;base64,{expected}" in page
    assert page.count("data:image/png;base64") == 2
    assert 'src="charts/' not in page
    assert "Ghost Chart" not in page

    assert ">87<" in page
    assert ">B<" in page
    assert "Solid dataset with a few fixable issues." in page
    assert 'style="width:90%"' in page
    assert 'style="width:64%"' in page

    assert "&lt;script&gt;" in page
    assert "<script" not in page
    assert "<strong>bold</strong>" in page
    assert "<code>code</code>" in page
    assert "<h3>Next steps</h3>" in page
    assert "&lt;weird&gt;" in page

    assert "Spearman ρ" in page
    assert "Cramér's V" in page
    assert "top: n/a (12)" in page
    assert "mean 10.5" in page
    assert "1,500" in page
    assert "0.42" in page
    assert "Generated by sift-agent" in page


def test_minimal_state_still_renders_valid_html(tmp_path):
    state = {"input_path": "mini.csv", "profile": {"n_rows": 5, "n_columns": 2}}

    result = write_html_report(state, tmp_path)

    page = Path(result).read_text(encoding="utf-8")
    assert page.startswith("<!DOCTYPE html>")
    assert page.rstrip().endswith("</html>")
    assert "NaN" not in page
    assert "Overview" in page
    for absent in (
        "Data health",
        "Missing values",
        "Data quality",
        "Correlations",
        "Column profiles",
        "Visualizations",
        "Key insights",
        "Recommendations",
        "Warnings",
    ):
        assert absent not in page


def test_missing_chart_files_are_skipped(tmp_path):
    state = _state(
        [{"title": "Ghost", "path": "charts/ghost.png", "caption": "not on disk"}]
    )

    result = write_html_report(state, tmp_path)

    page = Path(result).read_text(encoding="utf-8")
    assert Path(result).is_file()
    assert "data:image/png;base64" not in page
    assert "Visualizations" not in page
    assert "<figure" not in page


def test_empty_state_does_not_raise(tmp_path):
    result = write_html_report({}, tmp_path)

    assert Path(result).is_file()
    page = Path(result).read_text(encoding="utf-8")
    assert "<html" in page
    assert "NaN" not in page
