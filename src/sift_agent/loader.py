"""Input loading: CSV, Excel (.xlsx/.xlsm/.xls, multi-sheet) and ODS."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import pandas as pd

SUPPORTED_SUFFIXES = (".csv", ".xlsx", ".xlsm", ".xls", ".ods")
WORKBOOK_SUFFIXES = (".xlsx", ".xlsm", ".xls", ".ods")

PROVENANCE_COLUMN = "sheet"


class LoadError(Exception):
    """Raised when an input file cannot be turned into a DataFrame."""


def _excel_engine(suffix: str) -> str:
    if suffix == ".ods":
        return "odf"
    if suffix in (".xlsx", ".xlsm"):
        return "openpyxl"
    return "xlrd"


def _resolve_sheet_name(sheet_names: list[str], sheet: str | int) -> str:
    """Resolve a sheet selector (name or 0-based index) to a sheet name."""
    text = str(sheet).strip()
    if text in sheet_names:  # exact names win over numeric-looking indices
        return text
    try:
        index = int(text)
    except ValueError:
        raise LoadError(
            f"sheet '{text}' not found; available sheets: {', '.join(sheet_names)}"
        ) from None
    if -len(sheet_names) <= index < len(sheet_names):
        return sheet_names[index]
    raise LoadError(
        f"sheet index {index} out of range; workbook has {len(sheet_names)} sheet(s)"
    )


def sheet_names(path: Path | BinaryIO, suffix: str | None = None) -> list[str]:
    """List the sheet names of an Excel/ODS workbook without loading its data.

    ``path`` may be a file path or an open binary stream (e.g. BytesIO).
    ``suffix`` is derived from the path when omitted; streams must pass it.
    """
    if suffix is None:
        suffix = path.suffix.lower() if isinstance(path, Path) else ""
    if suffix not in WORKBOOK_SUFFIXES:
        raise LoadError(f"'{suffix or '(unknown)'}' is not a workbook format")
    try:
        book = pd.ExcelFile(path, engine=_excel_engine(suffix))
    except Exception as exc:
        raise LoadError(f"could not open workbook: {exc}") from exc
    return list(book.sheet_names)


def load_table(
    path: Path, sheet: str | int | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Load ``path`` into a DataFrame.

    CSV files load as-is. Excel/ODS workbooks load the selected ``sheet``
    (name or 0-based index); by default every non-empty sheet is combined
    when the sheets share the same columns, and a ``sheet`` provenance
    column records each row's origin.

    Returns the DataFrame plus non-fatal warnings; raises LoadError otherwise.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise LoadError(
            f"unsupported file type '{suffix or '(none)'}'; supported: "
            f"{', '.join(SUPPORTED_SUFFIXES)}"
        )
    if suffix == ".csv":
        try:
            return pd.read_csv(path), []
        except Exception as exc:
            raise LoadError(f"could not load '{path.name}': {exc}") from exc

    try:
        book = pd.ExcelFile(path, engine=_excel_engine(suffix))
    except Exception as exc:
        raise LoadError(f"could not open '{path.name}' as spreadsheet: {exc}") from exc

    warnings: list[str] = []
    frames: dict[str, pd.DataFrame] = {}
    for name in book.sheet_names:
        frame = book.parse(name)
        if frame.shape[0] and frame.shape[1]:
            frames[name] = frame
        else:
            warnings.append(f"skipped empty sheet '{name}'")

    if not frames:
        raise LoadError(f"'{path.name}' has no data in any sheet")

    if sheet is not None:
        name = _resolve_sheet_name(book.sheet_names, sheet)
        if name not in frames:
            raise LoadError(f"sheet '{name}' is empty")
        return frames[name], warnings

    if len(frames) == 1:
        return next(iter(frames.values())), warnings

    column_sets = {tuple(map(str, frame.columns)) for frame in frames.values()}
    if len(column_sets) > 1:
        detail = "; ".join(
            f"'{name}' ({', '.join(map(str, frame.columns))})"
            for name, frame in frames.items()
        )
        raise LoadError(
            f"'{path.name}' has sheets with different columns — select one "
            f"sheet instead. Sheets: {detail}"
        )

    names = list(frames)
    combined = pd.concat(frames.values(), ignore_index=True)
    if PROVENANCE_COLUMN not in combined.columns:
        combined.insert(
            0,
            PROVENANCE_COLUMN,
            [name for name in names for _ in range(len(frames[name]))],
        )
    warnings.append(f"combined {len(names)} sheets: {', '.join(names)}")
    return combined, warnings
