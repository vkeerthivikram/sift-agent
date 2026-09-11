import json

import pandas as pd
import pytest

from sift_agent import graph as graph_mod
from sift_agent.graph import build_graph


class _BoomLLM:
    def invoke(self, messages):
        raise RuntimeError("provider down")


@pytest.fixture()
def csv_path(tmp_path):
    p = tmp_path / "tiny.csv"
    p.write_text(
        "id,score,grade,joined\n"
        "1,10,A,2024-01-01\n"
        "2,20,B,2024-01-02\n"
        "3,30,A,2024-01-03\n"
        "4,40,,2024-01-04\n"
        "4,40,,2024-01-04\n",
        encoding="utf-8",
    )
    return p


def _run(graph, input_path, out_dir):
    return graph.invoke(
        {
            "input_path": str(input_path),
            "output_dir": str(out_dir),
            "provider": "none",
            "model": "",
        }
    )


def test_offline_end_to_end(tmp_path, csv_path):
    final = _run(build_graph(None), csv_path, tmp_path / "out")
    assert not final.get("error")
    report = tmp_path / "out" / "report.md"
    assert report.exists()
    profile = json.loads((tmp_path / "out" / "profile.json").read_text())
    assert profile["n_rows"] == 5
    assert profile["duplicate_rows"] == 1
    assert profile["columns"]["joined"]["type"] == "datetime"
    assert (tmp_path / "out" / "charts").exists()
    assert final["insights"]
    assert final["recommendations"]


def test_bad_csv_routes_to_error(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("not,a,csv\n", encoding="utf-8")  # header only, no rows
    final = _run(build_graph(None), bad, tmp_path / "out")
    assert final["error"]
    assert "report_path" not in final


def test_missing_csv_file(tmp_path):
    final = _run(build_graph(None), tmp_path / "nope.csv", tmp_path / "out")
    assert "error" in final


def test_size_guard(tmp_path, csv_path, monkeypatch):
    monkeypatch.setattr(graph_mod, "MAX_INPUT_BYTES", 10)
    final = _run(build_graph(None), csv_path, tmp_path / "out")
    assert "maximum supported size" in final["error"]


def test_row_guard(tmp_path, csv_path, monkeypatch):
    monkeypatch.setattr(graph_mod, "MAX_ROWS", 2)
    final = _run(build_graph(None), csv_path, tmp_path / "out")
    assert "maximum supported is" in final["error"]


def test_llm_failure_falls_back_to_heuristics(tmp_path, csv_path, monkeypatch):
    monkeypatch.setattr(graph_mod, "LLM_RETRY_BACKOFF_S", 0.0)
    final = _run(build_graph(_BoomLLM()), csv_path, tmp_path / "out")
    assert not final.get("error")
    assert final["insights"]
    assert final["recommendations"]
    warnings = final.get("warnings") or []
    assert any("insight LLM call failed" in w for w in warnings)
    assert any("recommendations LLM call failed" in w for w in warnings)
    assert "provider down" in "".join(warnings) or "RuntimeError" in "".join(warnings)


def test_chart_failure_recorded_as_warning(tmp_path, csv_path, monkeypatch):
    import sift_agent.visualize as viz

    def boom(df, charts_dir, numeric):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(viz, "_correlation_heatmap", boom)
    final = _run(build_graph(None), csv_path, tmp_path / "out")
    assert not final.get("error")
    assert any("correlation_heatmap" in w for w in final.get("warnings") or [])
    report = (tmp_path / "out" / "report.md").read_text()
    assert "## Warnings" in report
    assert "correlation_heatmap" in report


def test_xlsx_end_to_end_multi_sheet(tmp_path):
    p = tmp_path / "book.xlsx"
    frame = pd.DataFrame({"id": [1, 2], "score": [10.0, 20.0]})
    with pd.ExcelWriter(p, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="Jan", index=False)
        frame.assign(id=[3, 4], score=[30.0, 40.0]).to_excel(
            writer, sheet_name="Feb", index=False
        )
    final = _run(build_graph(None), p, tmp_path / "out")
    assert not final.get("error")
    profile = json.loads((tmp_path / "out" / "profile.json").read_text())
    assert profile["n_rows"] == 4
    assert "sheet" in profile["columns"]
    assert any("combined 2 sheets" in w for w in final.get("warnings") or [])
    assert (tmp_path / "out" / "charts").exists()
