"""Validated query plans: filters, grouping and aggregates over a DataFrame.

A plan is a small JSON-serializable dict. Plans come from the offline grammar
parser (:func:`parse_question`) or from an LLM translation step in ``qa``, but
every plan is validated against the actual DataFrame and executed with pandas
here — model output never runs as code. Relative dates are ``@``-tokens that
resolve deterministically against an injectable clock and timezone.

Plan shape::

    {
        "aggregate": "count" | "count_nonnull" | "sum" | "mean" | "min" | "max",
        "column": str | None,            # None only for "count"
        "filters": <node> | None,
        "group_by": [str, ...] | None,   # at most MAX_GROUP_COLUMNS
    }

Filter nodes::

    {"op": "and"|"or", "children": [<node>, ...]}
    {"op": "not", "child": <node>}
    {"op": "="|"!="|"<"|"<="|">"|">=", "column": str, "value": scalar}
    {"op": "in"|"not in", "column": str, "values": [scalar, ...]}
    {"op": "between", "column": str, "low": scalar, "high": scalar}
    {"op": "is null"|"not null", "column": str}

Date tokens (resolve in the query timezone)::

    @now, @day_start, @week_start, @month_start, @quarter_start, @year_start,
    @last_day_start, @last_week_start, @last_month_start,
    @last_quarter_start, @last_year_start, @last_<N>_days_start

Weeks start Monday; quarters are calendar quarters. ``between`` is inclusive.
Naive datetime columns are compared as wall-clock time in the query timezone.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone as dt_timezone, UTC
from zoneinfo import ZoneInfo

import pandas as pd

AGGREGATES = ("count", "count_nonnull", "sum", "mean", "min", "max")
VALUE_OPS = ("=", "!=", "<", "<=", ">", ">=")
ALL_OPS = frozenset(
    {"and", "or", "not", "in", "not in", "between", "is null", "not null", *VALUE_OPS}
)

MAX_FILTER_NODES = 40
MAX_GROUP_COLUMNS = 3
MAX_IN_VALUES = 50
MAX_RESULT_GROUPS = 20

_TOKEN_RE = re.compile(
    r"^@(now|(?:last_)?(?:day|week|month|quarter|year)_start|last_\d+_days_start)$"
)

_AGG_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(avg|average|mean)\b"), "mean"),
    (re.compile(r"\b(sum|total)\b"), "sum"),
    (re.compile(r"\b(max|maximum|largest|highest|biggest)\b"), "max"),
    (re.compile(r"\b(min|minimum|smallest|lowest)\b"), "min"),
    (re.compile(r"\b(count|how many|number of)\b"), "count"),
)


def _zone(tz: str) -> ZoneInfo | dt_timezone:
    try:
        return ZoneInfo(tz)
    except Exception:
        raise ValueError(f"unknown timezone '{tz}'") from None


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _month_index(year: int, month: int) -> int:
    return year * 12 + (month - 1)


def _from_month_index(index: int) -> tuple[int, int]:
    year, month = divmod(index, 12)
    return year, month + 1


def _at_month_start(index: int, local: datetime) -> datetime:
    year, month = _from_month_index(index)
    return local.replace(
        year=year, month=month, day=1, hour=0, minute=0, second=0, microsecond=0
    )


def resolve_token(
    token: str, tz: str = "UTC", now: datetime | None = None
) -> pd.Timestamp:
    """Resolve an ``@``-token to a timezone-aware Timestamp (Monday weeks)."""
    zone = _zone(tz)
    if now is None:
        now = datetime.now(UTC)
    local = now.astimezone(zone)
    if token == "@now":
        return pd.Timestamp(local)
    today = local.replace(hour=0, minute=0, second=0, microsecond=0)
    m = re.fullmatch(r"@last_(\d+)_days_start", token)
    if m:
        return pd.Timestamp(today - timedelta(days=int(m.group(1)) - 1))
    m = re.fullmatch(r"@?(last_)?(day|week|month|quarter|year)_start", token)
    if not m:
        raise ValueError(f"unknown date token '{token}'")
    offset = -1 if m.group(1) else 0
    unit = m.group(2)
    if unit == "day":
        return pd.Timestamp(today + timedelta(days=offset))
    if unit == "week":
        monday = today - timedelta(days=local.weekday())
        return pd.Timestamp(monday + timedelta(weeks=offset))
    if unit == "month":
        return pd.Timestamp(
            _at_month_start(_month_index(local.year, local.month) + offset, local)
        )
    if unit == "quarter":
        q_month = (local.month - 1) // 3 * 3 + 1
        return pd.Timestamp(
            _at_month_start(_month_index(local.year, q_month) + offset * 3, local)
        )
    return pd.Timestamp(
        _at_month_start(_month_index(local.year, 1) + offset * 12, local)
    )


def _is_token(value) -> bool:
    return isinstance(value, str) and bool(_TOKEN_RE.match(value))


def _datetime_value(series: pd.Series, value, tz: str, now: datetime | None):
    ts = (
        resolve_token(value, tz, now)
        if _is_token(value)
        else pd.Timestamp(pd.to_datetime(value, format="ISO8601"))
    )
    col_zone = getattr(series.dtype, "tz", None)
    if col_zone is None:
        return pd.Timestamp(ts).tz_localize(None)
    return pd.Timestamp(ts).tz_convert(col_zone)


def _coerce(series: pd.Series, value, tz: str, now: datetime | None):
    if pd.api.types.is_datetime64_any_dtype(series):
        return _datetime_value(series, value, tz, now)
    if pd.api.types.is_bool_dtype(series):
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in ("true", "1", "yes"):
            return True
        if text in ("false", "0", "no"):
            return False
        raise ValueError(f"value {value!r} is not boolean")
    if pd.api.types.is_numeric_dtype(series):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"column is numeric but value {value!r} is not a number")
        return value
    return value if isinstance(value, str) else str(value)


def _count_nodes(node) -> int:
    if node is None:
        return 0
    if node.get("op") == "not":
        return 1 + _count_nodes(node.get("child"))
    children = node.get("children")
    if children is not None:
        return 1 + sum(_count_nodes(c) for c in children)
    return 1


def _validate_node(node, df: pd.DataFrame, errors: list[str], tz: str) -> None:
    if not isinstance(node, dict) or "op" not in node:
        errors.append("filter node must be an object with an 'op'")
        return
    op = node["op"]
    if op not in ALL_OPS:
        errors.append(f"unknown filter op {op!r}")
        return
    if op in ("and", "or"):
        children = node.get("children")
        if not isinstance(children, list) or len(children) < 2:
            errors.append(f"'{op}' needs at least two children")
            return
        for child in children:
            _validate_node(child, df, errors, tz)
        return
    if op == "not":
        if "child" not in node:
            errors.append("'not' needs a 'child'")
            return
        _validate_node(node["child"], df, errors, tz)
        return
    column = node.get("column")
    if not isinstance(column, str) or column not in df.columns:
        errors.append(f"filter column {column!r} not found")
        return
    series = df[column]
    if op in ("is null", "not null"):
        return
    if op in ("in", "not in"):
        values = node.get("values")
        if not isinstance(values, list) or not values:
            errors.append(f"'{op}' needs a non-empty 'values' list")
            return
        if len(values) > MAX_IN_VALUES:
            errors.append(f"'{op}' accepts at most {MAX_IN_VALUES} values")
            return
    elif op == "between":
        for key in ("low", "high"):
            if key not in node:
                errors.append("'between' needs 'low' and 'high'")
                return
        values = [node["low"], node["high"]]
    else:
        if "value" not in node:
            errors.append(f"'{op}' needs a 'value'")
            return
        values = [node["value"]]
    ordered = op in ("<", "<=", ">", ">=", "between")
    for value in values:
        if pd.api.types.is_datetime64_any_dtype(series):
            if not (isinstance(value, str) and (_is_token(value) or _iso_ok(value))):
                errors.append(
                    f"column '{column}' is datetime; value {value!r} must be an "
                    "@-token or ISO 8601 string"
                )
        elif pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(
            series
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(
                    f"column '{column}' is numeric; value {value!r} must be a number"
                )
        elif ordered and not pd.api.types.is_datetime64_any_dtype(series):
            if not pd.api.types.is_numeric_dtype(series):
                errors.append(
                    f"ordered comparison '{op}' needs a numeric or datetime column, "
                    f"'{column}' is neither"
                )


def _iso_ok(value: str) -> bool:
    try:
        pd.to_datetime(value, format="ISO8601")
    except (ValueError, TypeError):
        return False
    return True


def validate_plan(plan, df: pd.DataFrame, tz: str = "UTC") -> list[str]:
    """Structural and type validation of a plan against ``df``; empty = valid."""
    errors: list[str] = []
    if not isinstance(plan, dict):
        return ["plan must be a JSON object"]
    aggregate = plan.get("aggregate")
    if aggregate not in AGGREGATES:
        errors.append(f"aggregate must be one of {', '.join(AGGREGATES)}")
        return errors
    column = plan.get("column")
    if aggregate == "count":
        if column is not None:
            errors.append("'count' counts matching rows; omit 'column'")
    elif not isinstance(column, str) or column not in df.columns:
        errors.append(f"metric column {column!r} not found")
        return errors
    elif aggregate in ("sum", "mean"):
        if not _numeric(df[column]):
            errors.append(f"aggregate '{aggregate}' needs a numeric column")
    if plan.get("filters") is not None:
        _validate_node(plan["filters"], df, errors, tz)
        if _count_nodes(plan["filters"]) > MAX_FILTER_NODES:
            errors.append(f"filter tree exceeds {MAX_FILTER_NODES} nodes")
    group_by = plan.get("group_by")
    if group_by is not None:
        if not isinstance(group_by, list) or not group_by:
            errors.append("'group_by' must be a non-empty list or null")
        else:
            if len(group_by) > MAX_GROUP_COLUMNS:
                errors.append(f"'group_by' accepts at most {MAX_GROUP_COLUMNS} columns")
            for col in group_by:
                if not isinstance(col, str) or col not in df.columns:
                    errors.append(f"group-by column {col!r} not found")
    return errors


def _numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(
        series
    )


def _mask(df: pd.DataFrame, node, tz: str, now: datetime | None) -> pd.Series:
    op = node["op"]
    if op == "and":
        out = None
        for child in node["children"]:
            m = _mask(df, child, tz, now)
            out = m if out is None else out & m
        return out
    if op == "or":
        out = None
        for child in node["children"]:
            m = _mask(df, child, tz, now)
            out = m if out is None else out | m
        return out
    if op == "not":
        return ~_mask(df, node["child"], tz, now)
    series = df[node["column"]]
    if op == "is null":
        return series.isna()
    if op == "not null":
        return series.notna()
    if op in ("in", "not in"):
        values = [_coerce(series, v, tz, now) for v in node["values"]]
        hit = series.isin(values)
        return hit if op == "in" else ~hit
    if op == "between":
        low = _coerce(series, node["low"], tz, now)
        high = _coerce(series, node["high"], tz, now)
        return (series >= low) & (series <= high)
    value = _coerce(series, node["value"], tz, now)
    if op == "=":
        return series == value
    if op == "!=":
        return series != value
    if op == "<":
        return series < value
    if op == "<=":
        return series <= value
    if op == ">":
        return series > value
    return series >= value


def _py_value(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if hasattr(v, "item"):
        return v.item()
    return v


def _agg_value(series: pd.Series, aggregate: str):
    if aggregate == "count_nonnull":
        return int(series.count())
    if aggregate == "sum":
        return _py_value(series.sum())
    if aggregate == "mean":
        return _py_value(series.mean())
    if aggregate == "min":
        return _py_value(series.min())
    return _py_value(series.max())


def _resolve_bound_tokens(node, bounds: dict, tz: str, now: datetime | None) -> None:
    if node is None:
        return
    for key, child in (
        ("child", node.get("child")),
        *[(None, c) for c in node.get("children") or []],
    ):
        if key == "child" and child is not None:
            _resolve_bound_tokens(child, bounds, tz, now)
        elif child is not None:
            _resolve_bound_tokens(child, bounds, tz, now)
    for value_key in ("value", "low", "high"):
        value = node.get(value_key)
        if _is_token(value) and value not in bounds:
            bounds[value] = resolve_token(value, tz, now)
    for value in node.get("values") or []:
        if _is_token(value) and value not in bounds:
            bounds[value] = resolve_token(value, tz, now)


def describe_plan(plan, tz: str = "UTC", now: datetime | None = None) -> str:
    """Human-readable interpretation of a plan, with resolved date bounds."""
    bounds: dict[str, pd.Timestamp] = {}
    _resolve_bound_tokens(plan.get("filters"), bounds, tz, now)
    aggregate = plan.get("aggregate")
    column = plan.get("column")
    metric = (
        "count of matching rows"
        if aggregate == "count"
        else f"{aggregate} of `{column}`"
    )
    parts = [metric]
    filters = _describe_node(plan.get("filters"), bounds)
    if filters:
        parts.append(f"where {filters}")
    group_by = plan.get("group_by")
    if group_by:
        parts.append("grouped by " + ", ".join(f"`{c}`" for c in group_by))
    if bounds:
        resolved = ", ".join(
            f"{token}={ts.isoformat()}" for token, ts in sorted(bounds.items())
        )
        parts.append(f"[{tz}; {resolved}]")
    return " | ".join(parts)


def _fmt_bound(value, bounds: dict) -> str:
    if _is_token(value):
        ts = bounds.get(value)
        return str(ts) if ts is not None else value
    return repr(value)


def _describe_node(node, bounds: dict) -> str:
    if node is None:
        return ""
    op = node["op"]
    if op in ("and", "or"):
        joiner = " and " if op == "and" else " or "
        return (
            "(" + joiner.join(_describe_node(c, bounds) for c in node["children"]) + ")"
        )
    if op == "not":
        return "not (" + _describe_node(node["child"], bounds) + ")"
    column = node.get("column")
    if op in ("is null", "not null"):
        return f"`{column}` {op}"
    if op in ("in", "not in"):
        values = ", ".join(_fmt_bound(v, bounds) for v in node["values"])
        return f"`{column}` {op} ({values})"
    if op == "between":
        return (
            f"`{column}` between "
            f"{_fmt_bound(node['low'], bounds)} and {_fmt_bound(node['high'], bounds)}"
        )
    return f"`{column}` {op} {_fmt_bound(node['value'], bounds)}"


def execute_plan(
    df: pd.DataFrame, plan, tz: str = "UTC", now: datetime | None = None
) -> dict:
    """Validate and run a plan. Returns an ok/error dict, never raises."""
    try:
        errors = validate_plan(plan=plan, df=df, tz=tz)
        if errors:
            return {"ok": False, "error": "; ".join(errors)}
        _zone(tz)
        mask = _mask(df, plan.get("filters"), tz, now) if plan.get("filters") else None
        sub = df.loc[mask] if mask is not None else df
        aggregate = plan["aggregate"]
        column = plan.get("column")
        group_by = plan.get("group_by")
        interpretation = describe_plan(plan, tz, now)
        base = {
            "ok": True,
            "interpretation": interpretation,
            "n_matched": len(sub),
            "tz": tz,
        }
        if group_by:
            grouped = sub.groupby(group_by, observed=True)
            if aggregate == "count":
                series = grouped.size()
                rows = [
                    {"group": _group_key(key, group_by), "value": int(v), "n": int(v)}
                    for key, v in series.items()
                ]
            else:
                stats = grouped[column].agg(["size", "count", aggregate])
                rows = [
                    {
                        "group": _group_key(key, group_by),
                        "value": _py_value(row[aggregate]),
                        "n": int(row["count"]),
                        "n_rows": int(row["size"]),
                    }
                    for key, row in stats.iterrows()
                ]
            rows.sort(key=lambda r: tuple(str(v) for v in r["group"].values()))
            truncated = len(rows) > MAX_RESULT_GROUPS
            base.update({"groups": rows[:MAX_RESULT_GROUPS], "truncated": truncated})
            return base
        if aggregate == "count":
            base.update({"value": len(sub), "n_used": len(sub)})
            return base
        series = sub[column]
        base.update(
            {
                "value": _agg_value(series, aggregate),
                "n_used": int(series.count()),
            }
        )
        return base
    except Exception as exc:
        return {"ok": False, "error": f"{exc.__class__.__name__}: {exc}"}


def _group_key(key, group_by: list[str]) -> dict:
    values = key if isinstance(key, tuple) else (key,)
    return {col: str(v) for col, v in zip(group_by, values)}


class _ExprParser:
    """Recursive-descent parser for the offline ``where`` grammar."""

    TOKEN_RE = re.compile(
        r"""
        '(?:[^'\\]|\\.)*'       # single-quoted string
        |"(?:[^"\\]|\\.)*"      # double-quoted string
        |@\w+                   # date token
        |>=|<=|!=|<>|==|=|<|>   # comparison operators
        |\(|\)|\[|\]|,          # grouping punctuation
        |[^\s()\[,\]]+          # bare word / number
        """,
        re.VERBOSE,
    )

    def __init__(self, text: str, df: pd.DataFrame, tz: str, now: datetime | None):
        self.tokens = self.TOKEN_RE.findall(text)
        self.pos = 0
        self.df = df
        self.columns = {c: _norm(c) for c in df.columns}
        self.tz = tz
        self.now = now

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def next(self):
        token = self.peek()
        self.pos += 1
        return token

    def _column_of(self, token) -> str | None:
        wanted = _norm(token)
        for col, norm in self.columns.items():
            if norm == wanted:
                return col
        return None

    def parse(self):
        node = self._parse_or()
        if self.peek() is not None:
            return None
        return node

    def _parse_or(self):
        children = [self._parse_and()]
        while self.peek() and self.peek().lower() == "or":
            self.next()
            children.append(self._parse_and())
        return children[0] if len(children) == 1 else {"op": "or", "children": children}

    def _parse_and(self):
        children = [self._parse_not()]
        while self.peek() and self.peek().lower() == "and":
            self.next()
            children.append(self._parse_not())
        return (
            children[0] if len(children) == 1 else {"op": "and", "children": children}
        )

    def _parse_not(self):
        if self.peek() and self.peek().lower() == "not":
            self.next()
            child = self._parse_not()
            return {"op": "not", "child": child} if child else None
        return self._parse_comparison()

    def _parse_comparison(self):
        if self.peek() == "(":
            self.next()
            node = self._parse_or()
            if self.next() != ")":
                return None
            return node
        column_token = self.next()
        if column_token is None:
            return None
        column = self._column_of(column_token)
        if column is None:
            return None
        op_token = self.next()
        if op_token is None:
            return None
        op_low = op_token.lower()
        op_map = {
            "=": "=",
            "==": "=",
            "!=": "!=",
            "<>": "!=",
            "<": "<",
            "<=": "<=",
            ">": ">",
            ">=": ">=",
        }
        if op_low == "is":
            value = self.next()
            if value and value.lower() == "null":
                return {"op": "is null", "column": column}
            if value and value.lower() == "not":
                nxt = self.next()
                if nxt and nxt.lower() == "null":
                    return {"op": "not null", "column": column}
            return None
        if op_low == "between":
            low = self._parse_value()
            and_tok = self.next()
            high = self._parse_value()
            if low is None or high is None or (and_tok or "").lower() != "and":
                return None
            return {"op": "between", "column": column, "low": low, "high": high}
        if op_low in ("in",):
            values = self._parse_list()
            return {"op": "in", "column": column, "values": values} if values else None
        if op_low == "not":
            nxt = self.next()
            if (nxt or "").lower() != "in":
                return None
            values = self._parse_list()
            return (
                {"op": "not in", "column": column, "values": values} if values else None
            )
        if op_low not in op_map:
            return None
        value = self._parse_value()
        if value is None:
            return None
        return {"op": op_map[op_low], "column": column, "value": value}

    def _parse_list(self) -> list | None:
        opener = self.next()
        if opener not in ("(", "["):
            return None
        values = []
        while True:
            token = self.peek()
            if token is None:
                return None
            if token in (")", "]"):
                self.next()
                return values
            value = self._parse_value()
            if value is None:
                return None
            values.append(value)
            token = self.peek()
            if token == ",":
                self.next()

    def _parse_value(self):
        token = self.next()
        if token is None:
            return None
        if token and len(token) >= 2 and token[0] == token[-1] and token[0] in "'\"":
            return token[1:-1]
        if _is_token(token):
            return token
        try:
            return float(token) if "." in token or "e" in token.lower() else int(token)
        except ValueError:
            return token


_DATE_PHRASES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bin (?:the )?last (\d+) days?\b"), "last_n_days"),
    (
        re.compile(r"\b(?:in|during) (?:the )?last (day|week|month|quarter|year)\b"),
        "last_period",
    ),
    (
        re.compile(
            r"\b(?:in|during) (?:the )?(this|current) (day|week|month|quarter|year)\b"
        ),
        "this_period",
    ),
    (re.compile(r"\bsince (last (day|week|month|quarter|year))\b"), "since_last"),
    (
        re.compile(r"\bsince (this|current) (day|week|month|quarter|year)\b"),
        "since_this",
    ),
    (re.compile(r"\bsince (\d{4}-\d{2}-\d{2}(?:[ T][0-9:.+-]+)?)\b"), "since_iso"),
    (
        re.compile(
            r"\b(?:between|from) (\d{4}-\d{2}-\d{2}(?:[ T][0-9:.+-]+)?) (?:and|to) (\d{4}-\d{2}-\d{2}(?:[0-9:. T+-]*)?)\b"
        ),
        "range_iso",
    ),
    (
        re.compile(r"\b(after|before) (\d{4}-\d{2}-\d{2}(?:[ T][0-9:.+-]+)?)\b"),
        "offset_iso",
    ),
)


def _first_datetime_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    return None


def _date_filters(question: str, df: pd.DataFrame) -> list[dict] | None:
    """Filters for natural date phrases, attached to the first datetime column."""
    column = _first_datetime_column(df)
    if column is None:
        return []
    q = question.lower()
    filters: list[dict] = []
    for pattern, kind in _DATE_PHRASES:
        m = pattern.search(q)
        if not m:
            continue
        if kind == "last_n_days":
            filters.append(
                {
                    "op": ">=",
                    "column": column,
                    "value": f"@last_{m.group(1)}_days_start",
                }
            )
        elif kind == "last_period":
            unit = m.group(1)
            filters.append(
                {"op": ">=", "column": column, "value": f"@last_{unit}_start"}
            )
            filters.append({"op": "<", "column": column, "value": f"@{unit}_start"})
        elif kind == "this_period":
            filters.append(
                {"op": ">=", "column": column, "value": f"@{m.group(2)}_start"}
            )
        elif kind == "since_last":
            filters.append(
                {
                    "op": ">=",
                    "column": column,
                    "value": f"@{m.group(1).replace(' ', '_')}_start",
                }
            )
        elif kind == "since_this":
            filters.append(
                {"op": ">=", "column": column, "value": f"@{m.group(2)}_start"}
            )
        elif kind == "since_iso":
            filters.append({"op": ">=", "column": column, "value": m.group(1)})
        elif kind == "range_iso":
            filters.append(
                {
                    "op": "between",
                    "column": column,
                    "low": m.group(1),
                    "high": m.group(2),
                }
            )
        elif kind == "offset_iso":
            filters.append(
                {
                    "op": ">" if m.group(1) == "after" else "<",
                    "column": column,
                    "value": m.group(2),
                }
            )
        break
    return filters


_FOR_RE = re.compile(
    r"\bfor\s+([a-z0-9_ ]+?)\s+(?:=\s*|is\s+|equals\s+)?"
    r"('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|[a-z0-9_.-]+)",
    re.IGNORECASE,
)
_WHERE_RE = re.compile(r"\bwhere\b", re.IGNORECASE)
_GROUP_RE = re.compile(r"\s+(?:group\s+)?by\s+([a-z0-9_,\s]+)$", re.IGNORECASE)


def _resolve_columns(text: str, df: pd.DataFrame) -> list[str] | None:
    """Resolve a comma/space separated name list to real columns (all or None)."""
    tokens = [t for t in re.split(r"[,\s]+", text.strip()) if t]
    if not tokens:
        return None
    resolved: list[str] = []
    for token in tokens:
        wanted = _norm(token)
        exact = next((c for c in df.columns if _norm(c) == wanted), None)
        if exact is None:
            return None
        resolved.append(exact)
    return resolved


def parse_question(
    question: str,
    df: pd.DataFrame,
    metric_candidates: list[str] | None = None,
    tz: str = "UTC",
    now: datetime | None = None,
) -> dict | None:
    """Parse a filtered/grouped aggregate question into a plan, or None.

    Returns a plan only when it carries a filter or a grouping; plain
    aggregates stay with the existing intent handlers in ``qa``.
    """
    q_lower = question.lower()
    aggregate = next(
        (name for pattern, name in _AGG_PATTERNS if pattern.search(q_lower)), None
    )
    if aggregate is None:
        return None

    where_match = _WHERE_RE.search(q_lower)
    remainder = question[where_match.end() :] if where_match else question
    group_match = _GROUP_RE.search(remainder)
    group_by = None
    where_expr = None
    if group_match:
        group_by = _resolve_columns(group_match.group(1), df)
        if group_by is None:
            return None
        where_expr = remainder[: group_match.start()].strip() if where_match else None
    elif where_match:
        where_expr = remainder.strip()

    filters: list[dict] = []
    if where_expr:
        parser = _ExprParser(where_expr, df, tz, now)
        tree = parser.parse()
        if tree is None:
            return None
        filters.append(tree)

    if not where_match:
        for_match = _FOR_RE.search(question)
        if for_match:
            column = _resolve_columns(for_match.group(1), df)
            value = for_match.group(2)
            if (
                value
                and len(value) >= 2
                and value[0] == value[-1]
                and value[0] in "'\""
            ):
                value = value[1:-1]
            if not column or len(column) != 1:
                return None
            filters.append({"op": "=", "column": column[0], "value": value})

    date_filters = _date_filters(question, df) or []
    filters.extend(date_filters)

    if not filters and not group_by:
        return None

    metric_candidates = [c for c in (metric_candidates or []) if c in df.columns]
    column = None
    if aggregate != "count":
        column = next((c for c in metric_candidates if _numeric(df[c])), None)
        if column is None:
            return None

    plan: dict = {"aggregate": aggregate, "column": column, "filters": None}
    if filters:
        plan["filters"] = (
            filters[0] if len(filters) == 1 else {"op": "and", "children": filters}
        )
    if group_by:
        plan["group_by"] = group_by
    return plan
