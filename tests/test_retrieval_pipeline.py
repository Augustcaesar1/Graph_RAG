"""Tests for the core retrieval and query processing pipeline."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_modules.query_heuristics import (
    extract_candidate_entities,
    infer_search_strategy,
    is_history_question,
)
from rag_modules.intelligent_query_router import IntelligentQueryRouter


def test_extract_candidate_entities_finds_names():
    result = extract_candidate_entities("哪吒的师父是谁？他属于哪个教派？")
    assert len(result) >= 1


def test_graph_strategy_for_relationship_query():
    strategy = infer_search_strategy("姜子牙和元始天尊是什么关系？", ["姜子牙", "元始天尊"])
    assert strategy == "graph_rag"


def test_traditional_strategy_for_lookup_query():
    strategy = infer_search_strategy("哪吒是谁？", ["哪吒"])
    assert strategy == "hybrid_traditional"


def test_is_history_question_positive():
    assert is_history_question("哪吒的师父是谁？")


def test_is_history_question_negative():
    assert not is_history_question("hello")
    assert not is_history_question("123")


def test_short_query_expansion_adds_retrieval_intent():
    router = IntelligentQueryRouter.__new__(IntelligentQueryRouter)

    expanded = router._expand_short_query("哪吒")

    assert "哪吒" in expanded
    assert "身份简介" in expanded
    assert "原文证据" in expanded


def test_question_like_query_is_not_expanded():
    router = IntelligentQueryRouter.__new__(IntelligentQueryRouter)
    query = "哪吒是谁？"

    assert router._expand_short_query(query) == query
