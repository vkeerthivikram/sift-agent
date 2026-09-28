import json
from pathlib import Path

import pandas as pd
import pytest

from sift_agent import graph as graph_mod
from sift_agent.graph import build_graph


def _resp(content, tool_calls=None):
    return type("Response", (), {"content": content, "tool_calls": tool_calls})()


class _ToolLoopLLM:
    """Answers insight/recommendation prompts generically; when the
    investigation prompt arrives it first requests one outlier_inspect call,
    then returns final findings text."""

    def __init__(self, column: str):
        self.column = column
        self.investigation_calls = 0

    def bind_tools(self, specs):
        return self

    def invoke(self, messages):
        text = " ".join(str(m) for m in messages)
        if "Investigate this dataset" in text:
            self.investigation_calls += 1
            if self.investigation_calls == 1:
                return _resp(
                    "",
                    tool_calls=[
                        {
                            "name": "outlier_inspect",
                            "args": {"column": self.column},
                            "id": "call_1",
                        }
                    ],
                )
        return _resp("- agent finding bullet")


class _NoBindLLM:
    """LLM without bind_tools: the investigation loop must fall back."""

    def invoke(self, messages):
        return _resp("- generic llm text")


class _NarrativeBoomLLM:
    """Cooperates everywhere except the anomaly narrative prompt, which raises."""

    def bind_tools(self, specs):
        return self

    def invoke(self, messages):
        text = " ".join(str(m) for m in messages)
        if "anomaly drill-down" in text:
            raise RuntimeError("narrative down")
        return _resp("- generic llm text")


@pytest.fixture()
def csv_path(tmp_path):
    # amount has 3 extreme IQR outliers; x/y are strongly correlated;
    # group is a low-cardinality categorical with mean differences.
    rows = []
    for i in range(40):
        x = 10 + (i % 10)
        amount = 100 + (i % 7) * 5
        if i in (5, 15, 25):
            amount = 5000 + i
        rows.append(
            {
                "amount": amount,
                "x": x,
                "y": 2 * x + (i % 3),
                "group": "abc"[i % 3],
            }
        )
    p = tmp_path / "agentic.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


@pytest.fixture()
def clean_path(tmp_path):
    # Correlated numeric pair + categorical, but no IQR outliers anywhere.
    rows = []
    for i in range(40):
        value1 = 10 + (i % 8)
        rows.append(
            {
                "value1": value1,
                "value2": 2 * value1 + (i % 3),
                "group": "ab"[i % 2],
            }
        )
    p = tmp_path / "clean.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
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


def test_offline_run_produces_trace_and_investigations(tmp_path, csv_path):
    final = _run(build_graph(None), csv_path, tmp_path / "out")
    assert not final.get("error")
    trace = final["agent_trace"]
    assert len(trace) >= 2
    assert {e["tool"] for e in trace} >= {"outlier_inspect", "correlation_check"}
    for entry in trace:
        assert set(entry) == {"round", "tool", "args", "summary"}
        assert entry["summary"]
        json.dumps(entry)  # trace entries stay JSON-safe
    investigations = final["investigations"]
    assert investigations
    finding = investigations[0]
    assert finding["question"] == "agent-led investigation"
    assert finding["tool"] == "multi"
    assert finding["args"] == {}
    assert finding["summary"].startswith("- ")

    report = (tmp_path / "out" / "report.md").read_text()
    assert "## Agent investigations" in report
    assert "## Anomaly drill-down" in report
    page = Path(final["html_report_path"]).read_text()
    assert "Agent investigations" in page


def test_offline_anomaly_reports_present(tmp_path, csv_path):
    final = _run(build_graph(None), csv_path, tmp_path / "out")
    assert not final.get("error")
    reports = final["anomaly_reports"]
    assert reports
    report = reports[0]
    assert report["column"] == "amount"
    assert report["n_outliers"] == 3
    assert isinstance(report["comparison"], str) and report["comparison"]
    assert isinstance(report["narrative"], str) and report["narrative"]
    assert len(reports) <= 2
    markdown = (tmp_path / "out" / "report.md").read_text()
    assert "`amount`" in markdown
    page = Path(final["html_report_path"]).read_text()
    assert "Anomaly drill-down" in page


def test_tool_loop_llm_records_trace_and_findings(tmp_path, csv_path):
    final = _run(build_graph(_ToolLoopLLM("amount")), csv_path, tmp_path / "out")
    assert not final.get("error")
    trace = final["agent_trace"]
    assert any(
        e["tool"] == "outlier_inspect" and e["args"] == {"column": "amount"}
        for e in trace
    )
    outlier_entry = next(e for e in trace if e["tool"] == "outlier_inspect")
    assert "amount" in outlier_entry["summary"]
    assert final["investigations"][0]["summary"] == "- agent finding bullet"
    markdown = (tmp_path / "out" / "report.md").read_text()
    assert "agent finding bullet" in markdown


def test_bind_failure_falls_back_deterministic(tmp_path, csv_path):
    final = _run(build_graph(_NoBindLLM()), csv_path, tmp_path / "out")
    assert not final.get("error")
    warnings = " ".join(final.get("warnings") or [])
    assert "investigation tool binding failed" in warnings
    assert "AttributeError" in warnings
    assert final["agent_trace"]
    assert final["investigations"][0]["summary"].startswith("- ")


def test_drilldown_narrative_failure_uses_template(tmp_path, csv_path, monkeypatch):
    monkeypatch.setattr(graph_mod, "LLM_RETRY_BACKOFF_S", 0.0)
    final = _run(build_graph(_NarrativeBoomLLM()), csv_path, tmp_path / "out")
    assert not final.get("error")
    warnings = " ".join(final.get("warnings") or [])
    assert "anomaly narrative LLM call failed" in warnings
    reports = final["anomaly_reports"]
    assert reports
    for report in reports:
        assert report["narrative"]
        assert report["column"] in report["narrative"]
        assert "IQR outlier" in report["narrative"]  # deterministic template


def test_no_outliers_skips_drilldown_but_investigates(tmp_path, clean_path):
    final = _run(build_graph(None), clean_path, tmp_path / "out")
    assert not final.get("error")
    assert final["anomaly_reports"] == []
    markdown = (tmp_path / "out" / "report.md").read_text()
    assert "## Anomaly drill-down" not in markdown
    page = Path(final["html_report_path"]).read_text()
    assert "Anomaly drill-down" not in page
    assert final["agent_trace"]
    assert final["investigations"]
    assert "## Agent investigations" in markdown
