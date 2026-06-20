"""Shared retrieval utilities used by hybrid_retrieval and intelligent_query_router."""

from __future__ import annotations

from langchain_core.documents import Document


def evidence_priority(doc: Document) -> float:
    """Score a document by how directly it carries original source evidence.

    TextChunk nodes and documents with explicit source_text / source_chunk_id
    rank highest; pure graph-path documents rank lower, ensuring the LLM
    sees actual text from the novel before inferred graph relationships.
    """
    metadata = doc.metadata or {}
    priority = 0.0
    if metadata.get("source_text"):
        priority += 2.5
    if metadata.get("source_chunk_id") or metadata.get("chunk_id"):
        priority += 1.5
    if metadata.get("node_type") == "TextChunk":
        priority += 1.6
    if metadata.get("search_type") == "vector_enhanced":
        priority += 1.0
    if metadata.get("search_type") in {"graph_path", "knowledge_subgraph"}:
        priority += 0.4
    if metadata.get("search_type") == "exact_entity_lookup":
        priority += 0.2
    return priority
