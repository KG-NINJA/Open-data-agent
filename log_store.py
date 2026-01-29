"""Log storage for agent interactions in JSON Lines format."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class LogEntry:
    """Serialized interaction record."""

    timestamp: str
    question: str
    used_tables: List[str]
    plan: Dict[str, Any]
    summary: str
    preview: str


class LogStore:
    """Append-only JSON Lines log storage with basic search helpers."""

    def __init__(self, path: str) -> None:
        self.path = path
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

    def record_interaction(
        self,
        question: str,
        used_tables: List[str],
        plan: Any,
        summary: str,
        preview: str,
    ) -> None:
        entry = LogEntry(
            timestamp=datetime.utcnow().isoformat() + "Z",
            question=question,
            used_tables=used_tables,
            plan=self._serialize_plan(plan),
            summary=summary,
            preview=preview,
        )
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(entry), ensure_ascii=True) + "\n")

    def load_logs(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load logs from disk."""
        if not os.path.exists(self.path):
            return []
        logs: List[Dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                logs.append(json.loads(line))
                if limit and len(logs) >= limit:
                    break
        return logs

    def search_logs(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Simple keyword search over question/summary fields."""
        query_lower = query.lower()
        matches = []
        for entry in self.load_logs():
            text = f"{entry.get('question', '')} {entry.get('summary', '')}".lower()
            if query_lower in text:
                matches.append(entry)
                if len(matches) >= limit:
                    break
        return matches

    def search_logs_with_embeddings(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Placeholder for future embedding-based log search."""
        raise NotImplementedError("Embedding search not implemented yet.")

    def _serialize_plan(self, plan: Any) -> Dict[str, Any]:
        if hasattr(plan, "__dict__"):
            return {k: self._serialize_value(v) for k, v in plan.__dict__.items()}
        if isinstance(plan, dict):
            return plan
        return {"raw": str(plan)}

    def _serialize_value(self, value: Any) -> Any:
        if isinstance(value, list):
            return [self._serialize_value(item) for item in value]
        if hasattr(value, "__dict__"):
            return {k: self._serialize_value(v) for k, v in value.__dict__.items()}
        return value
