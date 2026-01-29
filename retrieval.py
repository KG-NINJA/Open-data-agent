"""Future-facing retrieval interfaces for self-improvement and RAG."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Protocol


@dataclass
class RetrievalItem:
    """Unit of information to index or retrieve."""

    key: str
    text: str
    metadata: Dict[str, Any]


class RetrievalBackend(Protocol):
    """Protocol for pluggable retrieval backends (e.g., embeddings)."""

    def add(self, items: Iterable[RetrievalItem]) -> None:
        """Index new items for retrieval."""

    def query(self, text: str, top_k: int = 5) -> List[RetrievalItem]:
        """Retrieve similar items for a query."""

    def save(self, path: str) -> None:
        """Persist the retrieval index to disk."""

    def load(self, path: str) -> None:
        """Load a retrieval index from disk."""


def build_retrieval_context(
    backend: RetrievalBackend, query: str, top_k: int = 5
) -> List[RetrievalItem]:
    """Fetch retrieval items to enrich the agent context."""
    return backend.query(query, top_k=top_k)
