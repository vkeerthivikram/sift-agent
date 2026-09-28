"""Tests for the `sift ask` CLI command (offline deterministic engine)."""

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from sift_agent.cli import app

REPO = Path(__file__).resolve().parent.parent
MESSY_CSV = REPO / "examples" / "messy_orders.csv"

runner = CliRunner()


def _write_tiny_csv(path: Path) -> Path:
    pd.DataFrame(
        {
            "item": ["a", "b", "c", "d", "e", "f"],
            "amount": [10.0, 50.0, 42.0, 7.0, None, 21.0],
            "region": ["EU", "US", "EU", "APAC", "US", None],
        }
    ).to_csv(path, index=False)
    return path


def test_ask_missing_count_answer_with_sources(tmp_path):
    csv = _write_tiny_csv(tmp_path / "tiny.csv")
    result = runner.invoke(app, ["ask", str(csv), "how many rows are missing amount?"])
    assert result.exit_code == 0
    assert "1 of 6 values missing" in result.output
    assert "sources:" in result.output
    assert "column: amount" in result.output


def test_ask_offline_default_needs_no_env(tmp_path, monkeypatch):
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AZURE_OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    csv = _write_tiny_csv(tmp_path / "tiny.csv")
    result = runner.invoke(app, ["ask", str(csv), "how many rows are missing amount?"])
    assert result.exit_code == 0
    assert "Offline mode" in result.output
    assert "mode: offline" in result.output


def test_ask_nonexistent_path_fails_cleanly(tmp_path):
    result = runner.invoke(app, ["ask", str(tmp_path / "nope.csv"), "how many rows?"])
    assert result.exit_code == 2
    assert "does not exist" in result.output


def test_ask_unsupported_suffix_fails_cleanly(tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("not a table", encoding="utf-8")
    result = runner.invoke(app, ["ask", str(bad), "how many rows?"])
    assert result.exit_code == 1
    assert "unsupported file type" in result.output


def test_ask_max_of_numeric_column(tmp_path):
    csv = _write_tiny_csv(tmp_path / "tiny.csv")
    result = runner.invoke(app, ["ask", str(csv), "what is the max amount?"])
    assert result.exit_code == 0
    assert "maximum 50" in result.output


def test_ask_messy_orders_correlation_offline():
    result = runner.invoke(
        app, ["ask", str(MESSY_CSV), "is revenue correlated with cost?"]
    )
    assert result.exit_code == 0
    low = result.output.lower()
    assert "revenue" in low and "cost" in low
    assert "mode: offline" in result.output
