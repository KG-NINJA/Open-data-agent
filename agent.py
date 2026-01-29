"""Core data agent logic: planning, querying, and summarizing."""
from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from .data_loader import DataCatalog, DataTable
from .log_store import LogStore


@dataclass
class FilterCondition:
    """Filter condition for a query plan."""

    column: str
    op: str
    value: Any


@dataclass
class MetricSpec:
    """Aggregation metric specification."""

    name: str
    op: str
    column: Optional[str] = None


@dataclass
class OrderSpec:
    """Ordering specification."""

    column: str
    direction: str = "desc"


@dataclass
class QueryPlan:
    """Structured query plan derived from a natural language question."""

    tables: List[str] = field(default_factory=list)
    filters: List[FilterCondition] = field(default_factory=list)
    group_by: List[str] = field(default_factory=list)
    metrics: List[MetricSpec] = field(default_factory=list)
    order_by: List[OrderSpec] = field(default_factory=list)
    limit: int = 20
    notes: str = ""


@dataclass
class AgentResponse:
    """Response payload for the data agent."""

    summary: str
    preview: str
    plan: QueryPlan
    used_tables: List[str]
    warnings: List[str] = field(default_factory=list)


class LLMClient:
    """Minimal OpenAI client wrapper."""

    def __init__(self, model: str, api_key: Optional[str] = None) -> None:
        self.model = model
        self.api_key = api_key

    def complete(self, prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content or ""


class DataAgent:
    """Orchestrates question analysis, data querying, and summarization."""

    def __init__(
        self,
        catalog: DataCatalog,
        log_store: Optional[LogStore] = None,
        model: Optional[str] = None,
        prompts_dir: Optional[str] = None,
    ) -> None:
        self.catalog = catalog
        self.log_store = log_store
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.prompts_dir = prompts_dir or os.path.join(
            os.path.dirname(__file__), "prompts"
        )
        api_key = os.getenv("OPENAI_API_KEY")
        self.llm = LLMClient(self.model, api_key=api_key) if api_key else None

    def answer(self, question: str) -> AgentResponse:
        """Answer a natural language question using the loaded data."""
        plan = self._plan(question)
        df, used_tables, warnings = self._execute(plan)
        summary, preview = self._summarize(question, plan, df)

        if self.log_store:
            self.log_store.record_interaction(
                question=question,
                used_tables=used_tables,
                plan=plan,
                summary=summary,
                preview=preview,
            )

        return AgentResponse(
            summary=summary,
            preview=preview,
            plan=plan,
            used_tables=used_tables,
            warnings=warnings,
        )

    def _plan(self, question: str) -> QueryPlan:
        schema_context = self._render_schema_context()
        prompt = self._render_prompt(
            "analyze_base.txt",
            question=question,
            schema_context=schema_context,
        )
        if self.llm:
            raw = self.llm.complete(prompt)
            return self._parse_plan(raw)
        return self._heuristic_plan(question)

    def _execute(self, plan: QueryPlan) -> tuple[pd.DataFrame, List[str], List[str]]:
        warnings: List[str] = []
        if not plan.tables:
            raise ValueError("No tables selected in plan")
        table_name = plan.tables[0]
        if table_name not in self.catalog.tables:
            fallback = self.catalog.list_tables()[0]
            warnings.append(f"Unknown table '{table_name}', using '{fallback}'.")
            table_name = fallback
        table = self.catalog.get_table(table_name)
        if table.schema.source_type == "sqlite":
            df = self._execute_sqlite(table, plan, warnings)
        else:
            df = self._execute_dataframe(table, plan, warnings)
        return df, [table_name], warnings

    def _execute_dataframe(
        self, table: DataTable, plan: QueryPlan, warnings: List[str]
    ) -> pd.DataFrame:
        if table.dataframe is None:
            raise ValueError("No dataframe loaded for table")
        df = table.dataframe.copy()
        df = self._apply_filters_dataframe(df, plan.filters, warnings)
        df = self._apply_group_metrics_dataframe(df, plan, warnings)
        df = self._apply_order_limit_dataframe(df, plan, warnings)
        return df

    def _execute_sqlite(
        self, table: DataTable, plan: QueryPlan, warnings: List[str]
    ) -> pd.DataFrame:
        if not table.sqlite_path:
            raise ValueError("Missing sqlite path")
        query, params = self._build_sql(plan, table.schema.name, warnings)
        with sqlite3.connect(table.sqlite_path) as conn:
            return pd.read_sql_query(query, conn, params=params)

    def _apply_filters_dataframe(
        self, df: pd.DataFrame, filters: List[FilterCondition], warnings: List[str]
    ) -> pd.DataFrame:
        for condition in filters:
            if condition.column not in df.columns:
                warnings.append(f"Unknown column '{condition.column}' in filter.")
                continue
            series = df[condition.column]
            op = condition.op
            value = condition.value
            if op in {"==", "!=", ">", ">=", "<", "<="}:
                df = df[self._compare(series, op, value)]
            elif op == "contains":
                df = df[series.astype(str).str.contains(str(value), case=False, na=False)]
            elif op == "in":
                values = value if isinstance(value, list) else [value]
                df = df[series.isin(values)]
            else:
                warnings.append(f"Unsupported filter op '{op}'.")
        return df

    def _apply_group_metrics_dataframe(
        self, df: pd.DataFrame, plan: QueryPlan, warnings: List[str]
    ) -> pd.DataFrame:
        if not plan.metrics:
            return df
        df_work = df.copy()
        agg_kwargs: Dict[str, Any] = {}
        for metric in plan.metrics:
            column = metric.column
            if metric.op == "count" and (column is None or column == "*"):
                df_work["_row_count"] = 1
                agg_kwargs[metric.name] = ("_row_count", "sum")
            else:
                if not column or column not in df_work.columns:
                    warnings.append(f"Unknown column for metric '{metric.name}'.")
                    continue
                agg_kwargs[metric.name] = (column, self._normalize_metric_op(metric.op))
        if plan.group_by:
            missing = [col for col in plan.group_by if col not in df_work.columns]
            if missing:
                warnings.append(f"Unknown group_by columns: {', '.join(missing)}")
                return df_work
            grouped = df_work.groupby(plan.group_by, dropna=False)
            result = grouped.agg(**agg_kwargs).reset_index()
        else:
            row: Dict[str, Any] = {}
            for name, (col, op) in agg_kwargs.items():
                if op == "sum":
                    row[name] = df_work[col].sum()
                elif op in {"avg", "mean"}:
                    row[name] = df_work[col].mean()
                elif op == "min":
                    row[name] = df_work[col].min()
                elif op == "max":
                    row[name] = df_work[col].max()
                elif op == "count":
                    row[name] = df_work[col].count()
                else:
                    warnings.append(f"Unsupported metric op '{op}'.")
                    row[name] = None
            result = pd.DataFrame([row])
        return result

    def _apply_order_limit_dataframe(
        self, df: pd.DataFrame, plan: QueryPlan, warnings: List[str]
    ) -> pd.DataFrame:
        if plan.order_by:
            sort_cols = []
            ascending = []
            for order in plan.order_by:
                if order.column not in df.columns:
                    warnings.append(f"Unknown order_by column '{order.column}'.")
                    continue
                sort_cols.append(order.column)
                ascending.append(order.direction.lower() != "desc")
            if sort_cols:
                df = df.sort_values(by=sort_cols, ascending=ascending)
        if plan.limit:
            df = df.head(plan.limit)
        return df

    def _normalize_metric_op(self, op: str) -> str:
        if op == "avg":
            return "mean"
        return op

    def _compare(self, series: pd.Series, op: str, value: Any) -> pd.Series:
        numeric_value = self._coerce_numeric(value)
        if numeric_value is not None:
            value = numeric_value
        if op == "==":
            return series == value
        if op == "!=":
            return series != value
        if op == ">":
            return series > value
        if op == ">=":
            return series >= value
        if op == "<":
            return series < value
        if op == "<=":
            return series <= value
        return series == value

    def _coerce_numeric(self, value: Any) -> Optional[float]:
        try:
            if isinstance(value, bool):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _build_sql(
        self, plan: QueryPlan, table_name: str, warnings: List[str]
    ) -> tuple[str, List[Any]]:
        params: List[Any] = []
        table_ident = self._quote_identifier(table_name)

        select_parts: List[str] = []
        group_by_parts: List[str] = []
        if plan.group_by:
            for col in plan.group_by:
                if not self._safe_identifier(col):
                    warnings.append(f"Unsafe group_by column '{col}'.")
                    continue
                group_by_parts.append(self._quote_identifier(col))
                select_parts.append(self._quote_identifier(col))
        if plan.metrics:
            for metric in plan.metrics:
                if metric.op == "count" and (metric.column is None or metric.column == "*"):
                    select_parts.append(f"COUNT(*) AS {self._quote_identifier(metric.name)}")
                    continue
                if not metric.column or not self._safe_identifier(metric.column):
                    warnings.append(f"Unsafe metric column '{metric.column}'.")
                    continue
                col_ident = self._quote_identifier(metric.column)
                op = metric.op.upper()
                select_parts.append(
                    f"{op}({col_ident}) AS {self._quote_identifier(metric.name)}"
                )
        if not select_parts:
            select_parts.append("*")
        query = f"SELECT {', '.join(select_parts)} FROM {table_ident}"

        where_clauses: List[str] = []
        for condition in plan.filters:
            if not self._safe_identifier(condition.column):
                warnings.append(f"Unsafe filter column '{condition.column}'.")
                continue
            col_ident = self._quote_identifier(condition.column)
            op = condition.op
            if op == "contains":
                where_clauses.append(f"{col_ident} LIKE ?")
                params.append(f"%{condition.value}%")
            elif op == "in":
                values = condition.value if isinstance(condition.value, list) else [condition.value]
                placeholders = ", ".join(["?"] * len(values))
                where_clauses.append(f"{col_ident} IN ({placeholders})")
                params.extend(values)
            elif op in {"==", "!=", ">", ">=", "<", "<="}:
                sql_op = "=" if op == "==" else op
                where_clauses.append(f"{col_ident} {sql_op} ?")
                params.append(condition.value)
            else:
                warnings.append(f"Unsupported filter op '{op}'.")
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
        if group_by_parts:
            query += " GROUP BY " + ", ".join(group_by_parts)
        if plan.order_by:
            order_parts = []
            for order in plan.order_by:
                if not self._safe_identifier(order.column):
                    warnings.append(f"Unsafe order_by column '{order.column}'.")
                    continue
                direction = "DESC" if order.direction.lower() == "desc" else "ASC"
                order_parts.append(f"{self._quote_identifier(order.column)} {direction}")
            if order_parts:
                query += " ORDER BY " + ", ".join(order_parts)
        if plan.limit:
            query += " LIMIT ?"
            params.append(plan.limit)
        return query, params

    def _safe_identifier(self, name: str) -> bool:
        return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name or ""))

    def _quote_identifier(self, name: str) -> str:
        if not self._safe_identifier(name):
            raise ValueError(f"Unsafe identifier: {name}")
        return f'"{name}"'

    def _summarize(self, question: str, plan: QueryPlan, df: pd.DataFrame) -> tuple[str, str]:
        preview_rows = min(len(df), 10)
        preview = df.head(preview_rows).to_string(index=False)
        if df.empty:
            summary = "No rows matched the query."
        elif plan.metrics:
            summary = f"Computed {len(df)} aggregated row(s)."
        else:
            summary = f"Returned {len(df)} row(s) from the selected table."
        return summary, preview

    def _render_schema_context(self) -> str:
        table_blocks: List[str] = []
        for table_name in self.catalog.list_tables():
            table = self.catalog.get_table(table_name)
            cols = []
            for col in table.schema.columns:
                sample = ", ".join(col.sample_values) if col.sample_values else "-"
                cols.append(f"{col.name} ({col.dtype}, nullable={col.nullable}, sample={sample})")
            block = [
                f"Name: {table.schema.name}",
                f"Source: {table.schema.source_type}",
                f"Rows: {table.schema.row_count}",
                "Columns:",
                "  - " + "\n  - ".join(cols) if cols else "  - (none)",
            ]
            table_blocks.append("\n".join(block))
        tables_text = "\n\n".join(table_blocks)
        template = self._load_prompt("context_schema_template.txt")
        return template.replace("{{tables}}", tables_text)

    def _render_prompt(self, name: str, **kwargs: str) -> str:
        template = self._load_prompt(name)
        for key, value in kwargs.items():
            template = template.replace(f"{{{{{key}}}}}", value)
        return template

    def _load_prompt(self, name: str) -> str:
        path = os.path.join(self.prompts_dir, name)
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()

    def _parse_plan(self, raw: str) -> QueryPlan:
        try:
            plan_dict = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                return self._empty_plan()
            plan_dict = json.loads(match.group(0))
        return self._plan_from_dict(plan_dict)

    def _plan_from_dict(self, data: Dict[str, Any]) -> QueryPlan:
        return QueryPlan(
            tables=[str(x) for x in data.get("tables", [])],
            filters=[
                FilterCondition(
                    column=str(item.get("column")),
                    op=str(item.get("op")),
                    value=item.get("value"),
                )
                for item in data.get("filters", [])
                if item
            ],
            group_by=[str(x) for x in data.get("group_by", [])],
            metrics=[
                MetricSpec(
                    name=str(item.get("name")),
                    op=str(item.get("op")),
                    column=item.get("column"),
                )
                for item in data.get("metrics", [])
                if item
            ],
            order_by=[
                OrderSpec(
                    column=str(item.get("column")),
                    direction=str(item.get("direction", "desc")),
                )
                for item in data.get("order_by", [])
                if item
            ],
            limit=int(data.get("limit", 20)),
            notes=str(data.get("notes", "")),
        )

    def _empty_plan(self) -> QueryPlan:
        return QueryPlan(tables=self.catalog.list_tables()[:1])

    def _heuristic_plan(self, question: str) -> QueryPlan:
        tokens = {token for token in re.split(r"\W+", question.lower()) if token}
        table_scores: Dict[str, int] = {}
        for table_name in self.catalog.list_tables():
            score = 0
            table = self.catalog.get_table(table_name)
            for col in table.schema.columns:
                if col.name.lower() in tokens:
                    score += 3
                for sample in col.sample_values:
                    if sample and sample.lower() in tokens:
                        score += 1
            table_scores[table_name] = score
        best_table = max(table_scores, key=table_scores.get, default=None)
        plan = QueryPlan(tables=[best_table] if best_table else [])
        if any(word in tokens for word in {"count", "number", "how", "many"}):
            plan.metrics = [MetricSpec(name="count", op="count", column="*")]
        if any(word in tokens for word in {"average", "avg", "mean"}):
            numeric_col = self._first_numeric_column(best_table)
            if numeric_col:
                plan.metrics = [MetricSpec(name="avg", op="avg", column=numeric_col)]
        return plan

    def _first_numeric_column(self, table_name: Optional[str]) -> Optional[str]:
        if not table_name:
            return None
        table = self.catalog.get_table(table_name)
        for col in table.schema.columns:
            if any(token in col.dtype for token in ("int", "float", "double")):
                return col.name
        return None
