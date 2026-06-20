"""Shared filtering rules for 封神演义 graph retrieval."""

from __future__ import annotations

from typing import Iterable, List

from langchain_core.documents import Document


GOLD_ALLOWED_NODE_TYPES = {
    "Person", "Faction", "Location", "Artifact", "Beast", "Formation", "Event", "DeityPosition",
    "TextChunk", "Chapter", "Community"
}
GOLD_ALLOWED_ENTITY_NODE_TYPES = {
    "Person", "Faction", "Location", "Artifact", "Beast", "Formation", "Event", "DeityPosition",
    "TextChunk", "Chapter"
}
GOLD_BLOCKED_NODE_TYPES = set()


def is_gold_node_type(node_type: str | None) -> bool:
    return str(node_type or "") in GOLD_ALLOWED_NODE_TYPES


def build_gold_entity_predicate(alias: str) -> str:
    allowed = ", ".join(f"'{item}'" for item in sorted(GOLD_ALLOWED_ENTITY_NODE_TYPES))
    return f"any(label IN labels({alias}) WHERE label IN [{allowed}])"


def filter_gold_edges(edges: Iterable[dict] | None) -> List[dict]:
    return [edge for edge in (edges or []) if str(edge.get("source_type") or "") not in GOLD_BLOCKED_NODE_TYPES and str(edge.get("target_type") or "") not in GOLD_BLOCKED_NODE_TYPES]


def is_gold_document(doc: Document) -> bool:
    metadata = doc.metadata or {}
    node_type = str(metadata.get("node_type") or "")
    if node_type in GOLD_BLOCKED_NODE_TYPES:
        return False
    if node_type and node_type in GOLD_ALLOWED_NODE_TYPES:
        return True
    labels = [str(label) for label in (metadata.get("labels") or [])]
    if any(label in GOLD_ALLOWED_NODE_TYPES for label in labels):
        return True
    return True


def filter_gold_documents(documents: Iterable[Document]) -> List[Document]:
    filtered: List[Document] = []
    for doc in documents:
        if not is_gold_document(doc):
            continue
        if doc.metadata and doc.metadata.get("subgraph_edges"):
            doc.metadata["subgraph_edges"] = filter_gold_edges(doc.metadata["subgraph_edges"])
        filtered.append(doc)
    return filtered
