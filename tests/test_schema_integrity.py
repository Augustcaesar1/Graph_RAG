"""Validate schema definitions are complete and consistent."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_modules.gold_schema import (
    ALLOWED_RELATION_TYPES,
    RELATION_LABELS_ZH,
    ENTITY_TYPES,
    ENTITY_ID_FIELDS,
    ENTITY_NAME_FIELDS,
)
from rag_modules.retrieval_shared import evidence_priority
from rag_modules.source_evidence import (
    build_evidence_bundles,
    extract_cited_snippets,
    snippets_from_evidence_bundles,
    _extract_evidence_anchors,
    _trim_to_useful_excerpt,
)
from langchain_core.documents import Document


def test_all_relations_have_chinese_labels():
    for rel in sorted(ALLOWED_RELATION_TYPES):
        assert rel in RELATION_LABELS_ZH, f"Missing Chinese label for relation: {rel}"


def test_entity_types_have_id_fields():
    for etype in ENTITY_TYPES:
        assert etype in ENTITY_ID_FIELDS, f"Missing ID field for entity type: {etype}"
        assert etype in ENTITY_NAME_FIELDS, f"Missing name field for entity type: {etype}"


def test_evidence_priority_textchunk_scores_higher():
    chunk_doc = Document(page_content="test", metadata={
        "node_type": "TextChunk",
        "source_text": "original text"
    })
    graph_doc = Document(page_content="test", metadata={
        "node_type": "Person",
        "search_type": "graph_path"
    })
    assert evidence_priority(chunk_doc) > evidence_priority(graph_doc)


def test_evidence_priority_graph_path_positive():
    doc = Document(page_content="test", metadata={
        "search_type": "graph_path",
        "source_text": "some evidence"
    })
    assert evidence_priority(doc) > 0


def test_evidence_priority_empty_doc():
    doc = Document(page_content="test", metadata={})
    assert evidence_priority(doc) == 0.0


def test_cited_snippets_fallback_to_content_preview():
    snippets = extract_cited_snippets(
        "答案引用了图谱证据[1]。",
        [{"index": 1, "entity_name": "哪吒", "content_preview": "哪吒相关图谱支持内容"}],
        snippet_fetcher=lambda _key, _limit: [],
    )

    assert snippets[1] == ["哪吒相关图谱支持内容"]


def test_trim_to_useful_excerpt_keeps_keyword_context():
    text = "开头无关内容。" * 80 + "哪吒来到海边洗澡，惊动龙宫，随后与龙王发生冲突。" + "结尾无关内容。" * 80

    excerpt = _trim_to_useful_excerpt(text, ["哪吒", "龙王"], max_chars=120)

    assert "哪吒" in excerpt
    assert "龙王" in excerpt
    assert len(excerpt) <= 120


def test_cited_snippets_show_multiple_edge_evidences():
    source_text = (
        "古风一首：" + "无关诗文。" * 40
        + "今陛下作诗亵渎圣明，毫无虔敬之诚，是获罪于神圣。"
        + "中间无关。" * 10
        + "你三妖可隐其妖形，托身宫院，惑乱君心；俟武王伐纣，以助成功。"
        + "结尾。" * 20
    )
    subgraph_edges = [
        {"source": "纣王", "relation": "OPPOSES", "target": "女娲",
         "evidence": "今陛下作诗亵渎圣明，毫无虔敬之诚"},
        {"source": "女娲", "relation": "INITIATES", "target": "武王伐纣",
         "evidence": "你三妖可隐其妖形，托身宫院，惑乱君心"},
    ]
    content_preview = "娲 --[OPPOSES]--> 女娲；证据：今陛下作诗亵渎圣明...\\n女娲 --[INITIATES]--> 武王伐纣；证据：你三妖可隐其妖形..."

    snippets = extract_cited_snippets(
        "女娲生气是因为纣王题诗亵渎，随后派三妖惑乱纣王[1]。",
        [{
            "index": 1,
            "entity_name": "女娲",
            "source_text": source_text,
            "content_preview": content_preview,
            "subgraph_edges": subgraph_edges,
        }],
        snippet_fetcher=lambda _key, _limit: [],
    )

    all_snippets = "".join(snippets[1])
    assert "作诗亵渎" in all_snippets
    assert "三妖" in all_snippets
    assert len(snippets[1]) >= 2


def test_extract_evidence_anchors_from_content_preview():
    anchors = _extract_evidence_anchors("关系：女娲 --[INITIATES]--> 武王伐纣；证据：你三妖可隐其妖形，托身宫院，惑乱君心")

    assert any("你三妖可隐其妖形" in anchor for anchor in anchors)




def test_evidence_bundles_prefer_edge_source_text_and_deduplicate():
    doc = Document(
        page_content="图谱摘要：纣王与女娲。",
        metadata={
            "entity_name": "女娲",
            "source_text": "节点级原文背景。" * 20,
            "subgraph_edges": [
                {
                    "source": "纣王",
                    "relation": "OPPOSES",
                    "target": "女娲",
                    "evidence": "作诗亵渎圣明",
                    "source_text": "今陛下作诗亵渎圣明，毫无虔敬之诚，是获罪于神圣。",
                },
                {
                    "source": "纣王",
                    "relation": "OPPOSES",
                    "target": "女娲",
                    "evidence": "作诗亵渎圣明",
                    "source_text": "今陛下作诗亵渎圣明，毫无虔敬之诚，是获罪于神圣。",
                },
            ],
        },
    )

    bundle = build_evidence_bundles([doc])[0]
    texts = [item["text"] for item in bundle["original_excerpts"]]

    assert bundle["has_original_text"] is True
    assert "作诗亵渎圣明" in texts[0]
    assert sum("作诗亵渎圣明" in text for text in texts) == 1


def test_snippets_from_evidence_bundles_returns_cited_originals():
    bundles = [{
        "number": 1,
        "original_excerpts": [
            {"text": "商容说纣王作诗亵渎圣明。"},
            {"text": "女娲命三妖惑乱君心。"},
        ],
    }]

    snippets = snippets_from_evidence_bundles(bundles, [1], max_snippets_per_citation=2)

    assert snippets[1] == ["商容说纣王作诗亵渎圣明。", "女娲命三妖惑乱君心。"]
