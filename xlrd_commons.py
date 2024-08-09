from datetime import datetime
from typing import Any, Iterable, NamedTuple, Optional, Set, Sequence, Union

import xlrd

from sourcing.utils.date_utils import NaiveDateTime

_XLRD_CELL_TYPES: Set[int] = {
    xlrd.XL_CELL_EMPTY,
    xlrd.XL_CELL_TEXT,
    xlrd.XL_CELL_NUMBER,
    xlrd.XL_CELL_DATE,
    xlrd.XL_CELL_BOOLEAN,
    xlrd.XL_CELL_ERROR,
    xlrd.XL_CELL_BLANK,
}


class UnsupportedCellType(Exception):
    pass


def xls_parse(
    cell: xlrd.sheet.Cell,
    expected_type: int,
    allow_empty: bool = False,
    empty_text_as_null: bool = False,
) -> Any:

    assert expected_type in _XLRD_CELL_TYPES

    if cell.ctype == xlrd.XL_CELL_EMPTY:
        assert allow_empty, f"Got empty when not allowed for cell {str(cell)}"
        return None

    if expected_type != cell.ctype:
        assert (
            empty_text_as_null
            and cell.ctype == xlrd.XL_CELL_TEXT
            and cell.value.strip() == ""
        ), f"Expected type {expected_type} for cell {str(cell)}"
        return None

    if cell.ctype == xlrd.XL_CELL_TEXT:
        value = cell.value.strip()
        if value == "":
            assert allow_empty, "Got empty string when allow_empty is False"
            return None
        return value

    elif cell.ctype == xlrd.XL_CELL_NUMBER:
        return cell.value

    else:
        raise UnsupportedCellType(f"Cannot parse cell type: {cell.ctype}")


def is_none(cell: xlrd.sheet.Cell) -> bool:
    if cell.ctype == xlrd.XL_CELL_EMPTY:
        return True
    elif cell.ctype == xlrd.XL_CELL_TEXT:
        value = xls_parse(
            cell, xlrd.XL_CELL_TEXT, allow_empty=True, empty_text_as_null=True
        )
        return value is None
    else:
        return False


def is_number(cell: xlrd.sheet.Cell) -> bool:
    return cell.ctype == xlrd.XL_CELL_NUMBER


def ensure_int(cell: xlrd.sheet.Cell) -> int:
    value: float = xls_parse(cell, xlrd.XL_CELL_NUMBER)

    return _float_to_int(value)


def _float_to_int(value: float) -> int:
    assert int(value) == value, "Value provided has data after the decimal point."
    return int(value)


def ensure_int_or_none(cell: xlrd.sheet.Cell) -> Optional[int]:
    value: Optional[float] = xls_parse(
        cell, xlrd.XL_CELL_NUMBER, allow_empty=True, empty_text_as_null=True
    )

    if value is None:
        return None

    return _float_to_int(value)


def non_empty_string(cell: xlrd.sheet.Cell, float_to_str: bool = False) -> str:
    if float_to_str and cell.ctype == xlrd.XL_CELL_NUMBER:
        value = xls_parse(cell, xlrd.XL_CELL_NUMBER)

        return str(value)

    value = xls_parse(cell, xlrd.XL_CELL_TEXT)

    return value


def str_parse_or_none(cell: xlrd.sheet.Cell) -> Optional[str]:
    value = xls_parse(
        cell, xlrd.XL_CELL_TEXT, allow_empty=True, empty_text_as_null=True
    )

    return value


def float_parse(cell: xlrd.sheet.Cell) -> float:
    value = xls_parse(cell, xlrd.XL_CELL_NUMBER)

    return value


def float_parse_or_none(cell: xlrd.sheet.Cell) -> Optional[float]:
    value = xls_parse(
        cell, xlrd.XL_CELL_NUMBER, allow_empty=True, empty_text_as_null=True
    )

    return value


def parse_date(cell: xlrd.sheet.Cell, workbook: xlrd.Book) -> NaiveDateTime:
    return _parse_date_impl(cell, workbook.datemode)


def _parse_date_impl(cell: xlrd.sheet.Cell, datemode: int) -> NaiveDateTime:
    assert cell.ctype == xlrd.XL_CELL_DATE
    return NaiveDateTime(datetime(*xlrd.xldate_as_tuple(cell.value, datemode)))


class WrappedCell(NamedTuple):
    cell: xlrd.sheet.Cell
    datemode: int

    def value(self) -> Any:
        return self.cell.value

    def is_none(self) -> bool:
        return is_none(self.cell)

    def is_number(self) -> bool:
        return is_number(self.cell)

    def ensure_int(self) -> int:
        return ensure_int(self.cell)

    def ensure_int_or_none(self) -> Union[int, None]:
        return ensure_int_or_none(self.cell)

    def non_empty_string(self, float_to_str: bool = False) -> str:
        return non_empty_string(self.cell, float_to_str)

    def str_parse_or_none(self) -> Union[str, None]:
        return str_parse_or_none(self.cell)

    def float_parse(self) -> float:
        return float_parse(self.cell)

    def float_parse_or_none(self) -> Union[float, None]:
        return float_parse_or_none(self.cell)

    def parse_date(self) -> NaiveDateTime:
        return _parse_date_impl(self.cell, self.datemode)


class WrappedWorksheet(NamedTuple):
    sheet: xlrd.sheet.Sheet
    datemode: int

    def nrows(self) -> int:
        return self.sheet.nrows

    def ncols(self) -> int:
        return self.sheet.ncols

    def row(self, row_zero_indexed: int) -> Sequence[WrappedCell]:
        return [
            WrappedCell(cell=cell, datemode=self.datemode)
            for cell in self.sheet.row(row_zero_indexed)
        ]

    def iter_rows(
        self, min_row_zero_indexed: int, max_row_zero_indexed: int
    ) -> Iterable[Sequence[WrappedCell]]:

        for row_idx in range(min_row_zero_indexed, max_row_zero_indexed):
            yield self.row(row_idx)


class WrappedWorkbook(NamedTuple):
    workbook: xlrd.Book

    def sheet_names(self) -> Sequence[str]:
        return self.workbook.sheet_names()

    def sheet_by_name(self, name: str) -> WrappedWorksheet:
        return WrappedWorksheet(
            sheet=self.workbook.sheet_by_name(name), datemode=self.workbook.datemode
        )
