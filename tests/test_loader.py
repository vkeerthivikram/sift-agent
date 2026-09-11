import pandas as pd
import pytest

from sift_agent.loader import SUPPORTED_SUFFIXES, LoadError, load_table, sheet_names

TINY = {"id": [1, 2], "score": [10.0, 20.0]}


def _write_workbook(path, sheets: dict[str, pd.DataFrame]) -> None:
    engine = "odf" if path.suffix == ".ods" else "openpyxl"
    with pd.ExcelWriter(path, engine=engine) as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)


@pytest.fixture(params=["xlsx", "ods"])
def workbook_format(request):
    return request.param


def test_csv_passthrough(tmp_path):
    p = tmp_path / "tiny.csv"
    p.write_text("id,score\n1,10\n2,20\n", encoding="utf-8")
    df, warnings = load_table(p)
    assert df["id"].tolist() == [1, 2]
    assert warnings == []


def test_unsupported_suffix(tmp_path):
    with pytest.raises(LoadError, match="unsupported file type"):
        load_table(tmp_path / "tiny.parquet")


def test_single_sheet_loaded(tmp_path, workbook_format):
    p = tmp_path / f"tiny.{workbook_format}"
    _write_workbook(p, {"Only": pd.DataFrame(TINY)})
    df, warnings = load_table(p)
    assert df["score"].tolist() == [10.0, 20.0]
    assert "sheet" not in df.columns
    assert warnings == []


def test_multi_sheet_combined_with_provenance(tmp_path, workbook_format):
    p = tmp_path / f"multi.{workbook_format}"
    _write_workbook(
        p,
        {
            "Jan": pd.DataFrame(TINY),
            "Feb": pd.DataFrame({"id": [3, 4], "score": [30.0, 40.0]}),
        },
    )
    df, warnings = load_table(p)
    assert len(df) == 4
    assert df["id"].tolist() == [1, 2, 3, 4]
    assert df["sheet"].tolist() == ["Jan", "Jan", "Feb", "Feb"]
    assert any("combined 2 sheets: Jan, Feb" in w for w in warnings)


def test_select_sheet_by_name(tmp_path, workbook_format):
    p = tmp_path / f"multi.{workbook_format}"
    _write_workbook(
        p,
        {
            "Jan": pd.DataFrame(TINY),
            "Feb": pd.DataFrame({"id": [3], "score": [30.0]}),
        },
    )
    df, warnings = load_table(p, sheet="Feb")
    assert df["id"].tolist() == [3]
    assert "sheet" not in df.columns
    assert warnings == []


def test_select_sheet_by_index(tmp_path, workbook_format):
    p = tmp_path / f"multi.{workbook_format}"
    _write_workbook(
        p,
        {
            "Jan": pd.DataFrame(TINY),
            "Feb": pd.DataFrame({"id": [3], "score": [30.0]}),
        },
    )
    df, _ = load_table(p, sheet="1")
    assert df["id"].tolist() == [3]


def test_sheet_name_wins_over_numeric_index(tmp_path):
    p = tmp_path / "numeric_names.xlsx"
    _write_workbook(
        p,
        {
            "0": pd.DataFrame({"id": [1], "score": [10.0]}),
            "1": pd.DataFrame({"id": [2], "score": [20.0]}),
        },
    )
    df, _ = load_table(p, sheet="0")
    assert df["id"].tolist() == [1]


def test_empty_sheet_skipped_with_warning(tmp_path, workbook_format):
    p = tmp_path / f"with_empty.{workbook_format}"
    _write_workbook(
        p,
        {
            "Data": pd.DataFrame(TINY),
            "Blank": pd.DataFrame({"id": [], "score": []}),
        },
    )
    df, warnings = load_table(p)
    assert len(df) == 2
    assert any("skipped empty sheet 'Blank'" in w for w in warnings)


def test_all_sheets_empty(tmp_path, workbook_format):
    p = tmp_path / f"blank.{workbook_format}"
    _write_workbook(p, {"A": pd.DataFrame({"id": []})})
    with pytest.raises(LoadError, match="no data in any sheet"):
        load_table(p)


def test_differing_columns_error(tmp_path, workbook_format):
    p = tmp_path / f"mismatch.{workbook_format}"
    _write_workbook(
        p,
        {
            "A": pd.DataFrame(TINY),
            "B": pd.DataFrame({"x": [1]}),
        },
    )
    with pytest.raises(LoadError, match="different columns"):
        load_table(p)


def test_sheet_not_found(tmp_path):
    p = tmp_path / "tiny.xlsx"
    _write_workbook(p, {"Jan": pd.DataFrame(TINY)})
    with pytest.raises(LoadError, match=r"sheet 'Nope' not found.*Jan"):
        load_table(p, sheet="Nope")


def test_sheet_index_out_of_range(tmp_path):
    p = tmp_path / "tiny.xlsx"
    _write_workbook(p, {"Jan": pd.DataFrame(TINY)})
    with pytest.raises(LoadError, match="out of range"):
        load_table(p, sheet="5")


def test_selected_sheet_is_empty(tmp_path):
    p = tmp_path / "empty_pick.xlsx"
    _write_workbook(
        p,
        {
            "Data": pd.DataFrame(TINY),
            "Blank": pd.DataFrame({"id": [], "score": []}),
        },
    )
    with pytest.raises(LoadError, match="sheet 'Blank' is empty"):
        load_table(p, sheet="Blank")


def test_supported_suffixes_documented():
    assert SUPPORTED_SUFFIXES == (".csv", ".xlsx", ".xlsm", ".xls", ".ods")


def test_sheet_names_from_path(tmp_path):
    p = tmp_path / "multi.xlsx"
    _write_workbook(p, {"Jan": pd.DataFrame(TINY), "Feb": pd.DataFrame(TINY)})
    assert sheet_names(p) == ["Jan", "Feb"]


def test_sheet_names_from_stream(tmp_path):
    import io

    p = tmp_path / "multi.xlsx"
    _write_workbook(p, {"Jan": pd.DataFrame(TINY), "Feb": pd.DataFrame(TINY)})
    assert sheet_names(io.BytesIO(p.read_bytes()), suffix=".xlsx") == ["Jan", "Feb"]


def test_sheet_names_rejects_non_workbook():
    with pytest.raises(LoadError, match="not a workbook format"):
        sheet_names("/tmp/kilo/whatever.csv")
