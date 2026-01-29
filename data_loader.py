"""Data loading and schema extraction for CSV/JSON/SQLite sources."""
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".json", ".jsonl", ".sqlite", ".sqlite3", ".db"}


@dataclass
class ColumnInfo:
    """Column metadata for a table."""

    name: str
    dtype: str
    nullable: bool = True
    sample_values: List[str] = field(default_factory=list)


@dataclass
class TableSchema:
    """Schema and metadata for a data table."""

    name: str
    source_type: str
    path: str
    columns: List[ColumnInfo]
    row_count: Optional[int] = None
    description: Optional[str] = None
    primary_key: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataTable:
    """Loaded table with schema and optional in-memory data."""

    schema: TableSchema
    dataframe: Optional[pd.DataFrame] = None
    sqlite_path: Optional[str] = None


class DataCatalog:
    """Collection of tables and their schemas."""

    def __init__(self) -> None:
        self.tables: Dict[str, DataTable] = {}

    def add_table(self, table: DataTable) -> None:
        if table.schema.name in self.tables:
            raise ValueError(f"Duplicate table name: {table.schema.name}")
        self.tables[table.schema.name] = table

    def list_tables(self) -> List[str]:
        return sorted(self.tables.keys())

    def get_table(self, name: str) -> DataTable:
        if name not in self.tables:
            raise KeyError(f"Unknown table: {name}")
        return self.tables[name]

    def describe(self) -> str:
        """Human-readable schema summary."""
        lines: List[str] = []
        for table_name in self.list_tables():
            table = self.tables[table_name]
            lines.append(f"Table: {table.schema.name} ({table.schema.source_type})")
            if table.schema.row_count is not None:
                lines.append(f"  Rows: {table.schema.row_count}")
            for col in table.schema.columns:
                sample = ", ".join(col.sample_values) if col.sample_values else "-"
                lines.append(f"  - {col.name} [{col.dtype}] nullable={col.nullable} sample={sample}")
        return "\n".join(lines)


def discover_sources(paths: Iterable[str]) -> List[str]:
    """Discover supported data files from a list of files or directories."""
    discovered: List[str] = []
    for path in paths:
        if not path:
            continue
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for filename in files:
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in SUPPORTED_EXTENSIONS:
                        discovered.append(os.path.join(root, filename))
        elif os.path.isfile(path):
            ext = os.path.splitext(path)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                discovered.append(path)
    return sorted(set(discovered))


def load_catalog(paths: Iterable[str]) -> DataCatalog:
    """Load a data catalog from supported data sources."""
    catalog = DataCatalog()
    for path in discover_sources(paths):
        ext = os.path.splitext(path)[1].lower()
        if ext in {".csv", ".json", ".jsonl"}:
            _load_flat_file(path, catalog)
        else:
            _load_sqlite(path, catalog)
    return catalog


def _load_flat_file(path: str, catalog: DataCatalog) -> None:
    name = _table_name_from_path(path)
    dataframe = _read_flat_file(path)
    schema = TableSchema(
        name=name,
        source_type=_source_type_from_ext(path),
        path=path,
        columns=_infer_columns_from_dataframe(dataframe),
        row_count=len(dataframe),
        meta={"file": os.path.basename(path)},
    )
    catalog.add_table(DataTable(schema=schema, dataframe=dataframe))


def _load_sqlite(path: str, catalog: DataCatalog) -> None:
    conn = sqlite3.connect(path)
    try:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        for (table_name,) in cursor.fetchall():
            columns = _infer_columns_from_sqlite(conn, table_name)
            row_count = _count_rows(conn, table_name)
            schema = TableSchema(
                name=table_name,
                source_type="sqlite",
                path=path,
                columns=columns,
                row_count=row_count,
                meta={"file": os.path.basename(path)},
            )
            catalog.add_table(DataTable(schema=schema, sqlite_path=path))
    finally:
        conn.close()


def _read_flat_file(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        return pd.read_json(path)


def _infer_columns_from_dataframe(df: pd.DataFrame) -> List[ColumnInfo]:
    columns: List[ColumnInfo] = []
    for column in df.columns:
        series = df[column]
        sample_values = [
            _safe_sample_value(value) for value in series.dropna().head(3).tolist()
        ]
        columns.append(
            ColumnInfo(
                name=str(column),
                dtype=str(series.dtype),
                nullable=series.isna().any(),
                sample_values=sample_values,
            )
        )
    return columns


def _infer_columns_from_sqlite(conn: sqlite3.Connection, table_name: str) -> List[ColumnInfo]:
    columns: List[ColumnInfo] = []
    for row in conn.execute(f"PRAGMA table_info('{table_name}')"):
        _, name, dtype, notnull, _, _ = row
        sample_values = _sample_sqlite_values(conn, table_name, name)
        columns.append(
            ColumnInfo(
                name=name,
                dtype=dtype or "unknown",
                nullable=not bool(notnull),
                sample_values=sample_values,
            )
        )
    return columns


def _sample_sqlite_values(
    conn: sqlite3.Connection, table_name: str, column: str
) -> List[str]:
    try:
        cursor = conn.execute(
            f"SELECT {column} FROM '{table_name}' WHERE {column} IS NOT NULL LIMIT 3"
        )
        return [_safe_sample_value(row[0]) for row in cursor.fetchall()]
    except sqlite3.Error:
        return []


def _count_rows(conn: sqlite3.Connection, table_name: str) -> Optional[int]:
    try:
        cursor = conn.execute(f"SELECT COUNT(1) FROM '{table_name}'")
        return int(cursor.fetchone()[0])
    except sqlite3.Error:
        return None


def _table_name_from_path(path: str) -> str:
    base = os.path.basename(path)
    return os.path.splitext(base)[0]


def _source_type_from_ext(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".csv"}:
        return "csv"
    if ext in {".json", ".jsonl"}:
        return "json"
    return "sqlite"


def _safe_sample_value(value: Any) -> str:
    try:
        if isinstance(value, (dict, list)):
            return json.dumps(value)[:80]
        return str(value)[:80]
    except Exception:
        return "<unrepr>"
