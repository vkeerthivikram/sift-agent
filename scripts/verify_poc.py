"""End-to-end PoC verification: offline pipeline run on the messy demo dataset.

Asserts the artifacts a client demo depends on: report.md, a self-contained
report.html with embedded charts and a data-health panel, and profile.json
carrying a 0-100 health score with grade — plus the agentic extensions: a
populated agent trace, matching report sections, and an offline dataset Q&A
answer. Prints POC-VERIFY-OK, then POC-AGENTIC-OK, on success.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from sift_agent.analysis import profile_dataframe
from sift_agent.graph import run_pipeline
from sift_agent.loader import load_table
from sift_agent.qa import answer_question

REPO = Path(__file__).resolve().parent.parent
DATASET = REPO / "examples" / "messy_orders.csv"
QA_QUESTION = "how many rows are missing delivery_days?"


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="sift_poc_") as tmp:
        out_dir = Path(tmp) / "out"
        state = {
            "input_path": str(DATASET),
            "sheet": "",
            "output_dir": str(out_dir),
            "provider": "none",
            "model": "",
        }
        final = run_pipeline(state, llm=None)
        assert not final.get("error"), f"pipeline error: {final.get('error')}"

        report = Path(final["report_path"])
        html = Path(final["html_report_path"])
        profile_json = out_dir / "profile.json"
        assert report.exists(), "report.md missing"
        assert html.exists(), "report.html missing"
        assert profile_json.exists(), "profile.json missing"
        assert (out_dir / "charts").exists(), "charts/ missing"

        profile = json.loads(profile_json.read_text())
        health = profile.get("health") or {}
        assert isinstance(health.get("score"), int), "health.score missing"
        assert 0 <= health["score"] <= 100, f"score out of range: {health['score']}"
        assert health.get("grade") in {"A", "B", "C", "D", "F"}, "bad grade"
        assert len(health.get("components") or []) == 4, "expected 4 components"
        assert health.get("verdict"), "verdict missing"

        md = report.read_text()
        assert "## Executive summary" in md, "exec summary missing from report.md"
        assert "### Data health" in md, "health section missing from report.md"
        assert health["grade"] in md, "grade missing from report.md"

        page = html.read_text()
        assert "data:image/png;base64" in page, "charts not embedded in report.html"
        assert 'src="charts/' not in page, "report.html references external charts"
        assert "Data health" in page, "health panel missing from report.html"
        assert health["grade"] in page, "grade missing from report.html"
        assert "Key insights" in page and "Recommendations" in page

        n_charts = len(final.get("charts") or [])
        assert n_charts > 0, "no charts generated"
        print(
            f"artifacts OK: report.md, report.html ({n_charts} charts embedded), profile.json"
        )
        print(f"health: {health['score']}/100 grade {health['grade']}")

        trace = final.get("agent_trace") or []
        assert isinstance(trace, list) and trace, "agent_trace missing or empty"
        for entry in trace:
            assert {"round", "tool", "summary"} <= set(entry), (
                f"malformed agent_trace entry: {entry!r}"
            )
        assert "## Agent investigations" in md, (
            "agent investigations section missing from report.md"
        )
        assert "Agent investigations" in page, (
            "agent investigations missing from report.html"
        )
        anomalies = final.get("anomaly_reports") or []
        if anomalies:
            assert "## Anomaly drill-down" in md, (
                "anomaly reports exist but section missing from report.md"
            )
        print(
            f"agentic OK: {len(trace)} trace entries, "
            f"{len(anomalies)} anomaly report(s), report sections present"
        )

        qa_df = final.get("df")
        qa_profile = final.get("profile") or {}
        if qa_df is None:
            qa_df, _ = load_table(DATASET)
            qa_profile = profile_dataframe(qa_df)
        qa = answer_question(QA_QUESTION, qa_df, qa_profile)
        assert qa.get("answer"), "offline qa answer is empty"
        assert qa.get("offline") is True, "qa should answer offline without an LLM"
        haystack = f"{qa['answer']} {' '.join(qa.get('sources') or [])}"
        assert "delivery_days" in haystack, (
            "qa answer/sources do not mention delivery_days"
        )
        print(f"qa offline OK: {qa['answer'].strip()[:80]}")
    print("POC-VERIFY-OK")
    print("POC-AGENTIC-OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
