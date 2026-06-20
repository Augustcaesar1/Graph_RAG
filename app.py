"""
封神演义知识图谱 - GraphRAG 问答系统
Streamlit Web界面 | 神魔文学主题
"""

import os
import sys
import json
import logging
import re
import html
import streamlit as st
from typing import List, Dict, Any

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()
from config import DEFAULT_CONFIG, GraphRAGConfig
from rag_modules.bootstrap import build_rag_system
from rag_modules.source_evidence import (
    build_evidence_bundles,
    extract_cited_indices,
    extract_cited_snippets,
    snippets_from_evidence_bundles,
    _extract_evidence_anchors,
    _extract_keywords,
    _trim_to_useful_excerpt,
)
from rag_modules.generation_integration import evidence_bundle_config_for_question
from rag_modules.ui_presenters import build_answer_badges, build_hero_metrics, build_source_summary
from ui.styles import FENGSHEN_CSS

logger = logging.getLogger(__name__)

# ── 页面基础配置 ─────────────────────────────────────────────
st.set_page_config(
    page_title="封神演义 · 知识图谱",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── 浅色学术分格 CSS ───────────────────────────────────────────
st.markdown(FENGSHEN_CSS, unsafe_allow_html=True)
# ── 初始化 RAG 系统 ──────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ 正在初始化封神演义知识图谱系统...")
def load_rag_system(neo4j_uri: str, neo4j_user: str, neo4j_password: str):
    try:
        config = GraphRAGConfig.from_dict({
            **DEFAULT_CONFIG.to_dict(),
            "neo4j_uri": neo4j_uri,
            "neo4j_user": neo4j_user,
            "neo4j_password": neo4j_password,
        })
        return build_rag_system(config=config)
    except Exception as e:
        logger.warning(f"系统初始化失败: {e}")
        raise


SOURCE_FILE_HINTS = (".pdf", ".txt", ".doc", ".docx", "z-library", "1lib", "z-lib")
SOURCE_TITLE_HINTS = ("封神演义", "封神榜", "许仲琳")
VISIBLE_GRAPH_NODE_TYPES = {"Person", "Faction", "Location", "Artifact", "Beast", "Formation", "Event", "DeityPosition", "Chapter"}


def _normalize_display_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _looks_like_source_filename(value: Any) -> bool:
    text = _normalize_display_text(value)
    lower = text.lower()
    if not text:
        return False
    if any(hint in lower for hint in SOURCE_FILE_HINTS):
        return True
    if "\\" in text or "/" in text:
        return True
    if re.search(r"_chunk_\d+$", lower):
        return True
    if re.search(r"_\d+_\d+_\d+$", text):
        return True
    if re.search(r"\.(pdf|txt|doc|docx)_\d+_\d+_\d+$", lower):
        return True
    return False


def _looks_like_internal_identifier(value: Any) -> bool:
    text = _normalize_display_text(value)
    lower = text.lower()
    if not text:
        return False
    if lower == "system_init":
        return True
    if lower.startswith("community_"):
        return True
    if re.fullmatch(r"community_l\d+_[\w-]+", lower):
        return True
    if re.fullmatch(r"l\d+_\d+", lower):
        return True
    return False


def _looks_like_source_title(value: Any) -> bool:
    text = _normalize_display_text(value)
    if not text:
        return False
    return any(hint in text for hint in SOURCE_TITLE_HINTS)


def _is_hidden_graph_node(name: Any, node_type: Any = "") -> bool:
    text = _normalize_display_text(name)
    ntype = _normalize_display_text(node_type)
    if not text:
        return True
    if ntype and ntype not in VISIBLE_GRAPH_NODE_TYPES:
        return True
    if _looks_like_source_filename(text) or _looks_like_source_title(text) or _looks_like_internal_identifier(text):
        return True
    return False


def _clean_graph_node_name(name: Any) -> str:
    return _normalize_display_text(name)


def _display_source_title(source: Dict[str, Any]) -> str:
    raw_name = _normalize_display_text(source.get("raw_entity_name") or source.get("entity_name") or "")
    search_type = _normalize_display_text(source.get("search_type") or "")
    lower = raw_name.lower()

    if not raw_name:
        return {
            "graph_path": "图谱路径",
            "knowledge_subgraph": "知识子图",
        }.get(search_type, "相关条目")

    if raw_name == "system_init":
        return "系统初始化"
    if raw_name.startswith("community_") or lower.startswith("community_l") or re.fullmatch(r"l\d+_\d+", lower):
        return "相关社团"
    if _looks_like_source_filename(raw_name) or _looks_like_source_title(raw_name):
        return "原文片段"
    if _looks_like_internal_identifier(raw_name):
        return {
            "graph_path": "图谱路径",
            "knowledge_subgraph": "知识子图",
        }.get(search_type, "相关条目")
    return raw_name


def _sanitize_source_preview(source: Dict[str, Any]) -> str:
    preview = _normalize_display_text(source.get("content_preview") or "")
    source_text = _normalize_display_text(source.get("source_text") or "")
    if not preview and not source_text:
        return ""

    anchors = _extract_evidence_anchors(preview)
    keywords = _extract_keywords(preview, source.get("entity_name"), source.get("raw_entity_name"))
    if source_text:
        return _trim_to_useful_excerpt(source_text, keywords, anchors=anchors, max_chars=360)

    evidence_items = []
    for anchor in anchors:
        if anchor and anchor not in evidence_items:
            evidence_items.append(anchor)
    if evidence_items:
        return "；".join(evidence_items[:3])

    preview = re.sub(r"\bcommunity_l\d+_[\w-]+\b", "相关社团", preview, flags=re.IGNORECASE)
    preview = re.sub(r"\bcommunity_[\w-]+\b", "相关社团", preview, flags=re.IGNORECASE)
    preview = re.sub(r"\bl\d+_\d+\b", "相关社团", preview, flags=re.IGNORECASE)
    preview = re.sub(r"\bsystem_init\b", "系统初始化", preview, flags=re.IGNORECASE)
    preview = re.sub(r"\b[\w.-]+_chunk_\d+\b", "原文片段", preview, flags=re.IGNORECASE)
    preview = re.sub(r"\b[^\s，,。；;：:]*?\.(?:pdf|txt|doc|docx)_\d+_\d+_\d+\b", "原文片段", preview, flags=re.IGNORECASE)
    preview = re.sub(r"\b[\w.-]+_\d+_\d+_\d+\b", "原文片段", preview)
    preview = re.sub(r"[^\s，,。；;：:]*?(?:\.pdf|\.txt|\.docx?|z-library|1lib|z-lib)[^\s，,。；;：:]*", "原始资料", preview, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", preview).strip()


def _sanitize_triples_for_display(triples: List[Any]) -> List[Dict[str, Any]]:
    cleaned = []
    seen = set()

    for edge in triples:
        if isinstance(edge, dict):
            source = edge.get("source", "")
            relation = edge.get("relation", "")
            target = edge.get("target", "")
            source_type = edge.get("source_type", "")
            target_type = edge.get("target_type", "")
        elif isinstance(edge, (list, tuple)) and len(edge) >= 3:
            source, relation, target = edge[0], edge[1], edge[2]
            source_type = ""
            target_type = ""
        else:
            continue

        if _is_hidden_graph_node(source, source_type) or _is_hidden_graph_node(target, target_type):
            continue

        source_name = _clean_graph_node_name(source)
        target_name = _clean_graph_node_name(target)
        relation_name = _normalize_display_text(relation)
        if not source_name or not target_name or not relation_name:
            continue

        key = (source_name, relation_name, target_name, source_type, target_type)
        if key in seen:
            continue
        seen.add(key)

        cleaned.append({
            "source": source_name,
            "relation": relation_name,
            "target": target_name,
            "source_type": source_type,
            "target_type": target_type,
            "source_desc": edge.get("source_desc", "") if isinstance(edge, dict) else "",
            "target_desc": edge.get("target_desc", "") if isinstance(edge, dict) else "",
        })

    return cleaned


def render_hero_header(rag_loaded: bool, stats: Dict[str, Any] | None = None, route_stats: Dict[str, Any] | None = None):
    metrics = build_hero_metrics(rag_loaded=rag_loaded, stats=stats or {}, route_stats=route_stats or {})
    metric_html = "".join(
        (
            f'<div class="metric-card">'
            f'<div class="metric-label">{html.escape(item["label"])}</div>'
            f'<div class="metric-value">{html.escape(item["value"])}</div>'
            f'</div>'
        )
        for item in metrics
    )
    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="hero-kicker">Evidence-first Workspace</div>
            <h1 class="hero-title">封神演义 · 神魔关系图谱问答</h1>
            <div class="hero-subtitle">
                探索封神人物、教派阵营、法宝阵法与封神结局。左侧提问，右侧溯源。
            </div>
            <div class="hero-pill-row">
                <span class="hero-pill">原文优先</span>
                <span class="hero-pill">图谱辅助</span>
                <span class="hero-pill">并排核验</span>
            </div>
            <div class="metric-grid">{metric_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_answer_header(strategy: str, docs_count: int, answer_text: str):
    badges = build_answer_badges(strategy=strategy, docs_count=docs_count, answer_text=answer_text)
    badge_html = "".join(
        f'<span class="meta-badge">{html.escape(item["label"])}：{html.escape(item["value"])}</span>'
        for item in badges
    )
    st.markdown(f'<div class="answer-shell"><div class="badge-row">{badge_html}</div></div>', unsafe_allow_html=True)


# ── 知识图谱可视化 ────────────────────────────────────────────
def build_pyvis_graph(triples: List[Dict[str, Any]], highlight_nodes: List[str] = None) -> str:
    try:
        from pyvis.network import Network
    except ImportError:
        return "<p style='color:#d4af37'>请安装 pyvis：pip install pyvis</p>"

    highlight_nodes = set(highlight_nodes or [])

    net = Network(
        height="420px", width="100%",
        bgcolor="#f8fbfd", font_color="#294050",
        directed=True
    )
    net.set_options(json.dumps({
        "nodes": {
            "shape": "dot",
            "size": 20,
            "font": {"size": 13, "color": "#1e293b", "face": "Noto Sans SC", "strokeWidth": 2, "strokeColor": "#ffffff"},
            "borderWidth": 2.5,
            "shadow": {"enabled": True, "size": 6}
        },
        "edges": {
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.8}},
            "color": {"color": "#94a3b8", "opacity": 0.6, "highlight": "#3b6e71"},
            "font": {"size": 10, "color": "#64748b", "align": "middle", "strokeWidth": 2, "strokeColor": "#ffffff"},
            "smooth": {"type": "continuous", "roundness": 0.15},
            "width": 1.2,
            "selectionWidth": 2.5
        },
        "physics": {
            "barnesHut": {
                "gravitationalConstant": -2000,
                "centralGravity": 0.3,
                "springLength": 140,
                "springConstant": 0.04,
                "damping": 0.3
            },
            "solver": "barnesHut",
            "stabilization": {"iterations": 200, "updateInterval": 25}
        },
        "interaction": {
            "hover": True,
            "tooltipDelay": 150,
            "navigationButtons": True,
            "keyboard": True
        }
    }))

    # 节点颜色 - 封神演义主题
    COLOR_MAP = {
        "Person": "#e74c3c",
        "Faction": "#e67e22",
        "Location": "#2ecc71",
        "Artifact": "#9b59b6",
        "Beast": "#f39c12",
        "Formation": "#3498db",
        "Event": "#e91e63",
        "DeityPosition": "#1abc9c",
        "TextChunk": "#95a5a6",
        "Chapter": "#7f8c8d",
    }

    # 关系中文翻译
    ZH_REL_MAP = {
        "MASTER_OF": "师父",
        "APPRENTICE_OF": "徒弟",
        "BELONGS_TO_SECT": "属于教派",
        "FIGHTS_FOR": "效力于",
        "FATHER_OF": "父亲",
        "CHILD_OF": "子女",
        "BROTHER_OF": "兄弟",
        "MARRIED_TO": "婚配",
        "KILLS": "击杀",
        "DEFEATS": "击败",
        "CAPTURES": "擒获",
        "OPPOSES": "对抗",
        "OWNS": "拥有",
        "BESTOWS": "赐予",
        "LOSES": "失去",
        "STEALS": "盗取",
        "CREATES": "创建",
        "DEPLOYS": "布阵",
        "BREAKS": "破阵",
        "PARTICIPATES_IN": "参与",
        "OCCURS_IN": "发生于",
        "LEADS": "率领",
        "INITIATES": "发起",
        "LISTED_ON": "上榜封神",
        "BECOMES": "成为",
        "ALLIES_WITH": "同盟",
        "BETRAYS": "背叛",
        "RELATED_TO": "相关",
        "MENTIONS": "提及",
        "BELONGS_TO_CHAPTER": "所属章节",
    }

    # 边统一使用全局配置颜色（避免用线颜色表达类型/关系）

    def guess_node_type(name: str) -> str:
        faction_names = {"商", "周", "阐教", "截教", "西方教", "龙族", "轩辕坟"}
        artifact_keywords = {"鞭", "圈", "绫", "轮", "枪", "印", "镜", "剑", "珠", "图", "幡", "宝"}
        beast_keywords = {"犬", "麒麟", "神牛", "青鸾", "四不像", "坐骑"}
        formation_keywords = {"阵"}
        event_keywords = {"大战", "伐", "进香", "闹海", "破", "征", "封神"}
        deity_keywords = {"大帝", "天尊", "星", "真君", "大神", "神位"}
        if name in faction_names or name.endswith("教") or name.endswith("阵营"):
            return "Faction"
        if any(keyword in name for keyword in formation_keywords):
            return "Formation"
        if any(keyword in name for keyword in beast_keywords):
            return "Beast"
        if any(keyword in name for keyword in artifact_keywords):
            return "Artifact"
        if any(keyword in name for keyword in deity_keywords):
            return "DeityPosition"
        if any(keyword in name for keyword in event_keywords):
            return "Event"
        return "Person"

    added_nodes = set()

    def add_node(name: str, ntype: str = None, description: str = ""):
        name = _clean_graph_node_name(name)
        if _is_hidden_graph_node(name, ntype):
            return
        if name in added_nodes:
            return name
        added_nodes.add(name)

        # 有些检索会把类型写成 "Concept"，这会导致所有点一个颜色。
        # 这里遇到 Concept/空值/未知类型时，回退到规则猜测，确保颜色能区分。
        if not ntype or ntype == "Concept" or ntype not in COLOR_MAP:
            ntype = guess_node_type(name)

        color  = COLOR_MAP.get(ntype, "#aaaaaa")
        border = "#ffffff" if name in highlight_nodes else color
        size   = 28 if name in highlight_nodes else 18
        title  = f"[{ntype}] {name}"
        if description:
            title += f"\n---\n{description}"
        net.add_node(
            name,
            label=name[:14] + ("…" if len(name) > 14 else ""),
            title=title,
            color={"background": color, "border": border},
            size=size,
        )

        return name

    for edge in _sanitize_triples_for_display(triples):
        head = edge.get("source")
        rel  = edge.get("relation")
        tail = edge.get("target")
        if not head or not tail:
            continue
        head_id = add_node(head, edge.get("source_type"), edge.get("source_desc", ""))
        tail_id = add_node(tail, edge.get("target_type"), edge.get("target_desc", ""))
        if not head_id or not tail_id:
            continue
        label = ZH_REL_MAP.get(rel, rel)
        net.add_edge(head_id, tail_id, label=label)

    return net.generate_html()


# ── 来源面板渲染 ──────────────────────────────────────────────
def _fetch_original_text_snippets(entity_name: str, limit: int = 2) -> List[str]:
    rag = st.session_state.get("rag")
    if not rag:
        return []
    snippet_service = rag.get("snippet_service")
    if not snippet_service:
        return []
    return snippet_service.fetch_original_text_snippets(entity_name, limit)


def _backfill_original_text(docs: List[Any]) -> List[Any]:
    enriched_docs = []
    for doc in docs:
        metadata = dict(doc.metadata or {})
        if metadata.get("source_text"):
            snippets = [str(item).strip() for item in metadata.get("source_snippets", []) if str(item or "").strip()]
            if metadata.get("source_text") not in snippets:
                snippets.insert(0, metadata.get("source_text"))
            if snippets:
                metadata["source_snippets"] = snippets[:4]
                if metadata != (doc.metadata or {}):
                    doc = doc.__class__(page_content=doc.page_content, metadata=metadata)
            enriched_docs.append(doc)
            continue

        lookup_candidates = [
            metadata.get("source_chunk_id"),
            metadata.get("chunk_id"),
            metadata.get("node_id"),
            metadata.get("entity_name"),
            doc.page_content,
        ]
        snippets = []
        for candidate in lookup_candidates:
            candidate = str(candidate or "").strip()
            if not candidate:
                continue
            snippets = _fetch_original_text_snippets(candidate, limit=2)
            if snippets:
                if candidate.startswith("ch") or "-p" in candidate:
                    metadata.setdefault("source_chunk_id", candidate)
                clean_snippets = [str(item).strip() for item in snippets if str(item or "").strip()]
                metadata["source_text"] = clean_snippets[0]
                metadata["source_snippets"] = clean_snippets[:4]
                break

        if metadata != (doc.metadata or {}):
            doc = doc.__class__(page_content=doc.page_content, metadata=metadata)
        enriched_docs.append(doc)

    enriched_docs.sort(
        key=lambda item: (
            1 if (item.metadata or {}).get("source_text") else 0,
            1 if ((item.metadata or {}).get("source_chunk_id") or (item.metadata or {}).get("chunk_id")) else 0,
            1 if (item.metadata or {}).get("node_type") == "TextChunk" else 0,
            float((item.metadata or {}).get("score", (item.metadata or {}).get("final_score", (item.metadata or {}).get("relevance_score", 0.0)))),
        ),
        reverse=True,
    )
    return enriched_docs


def render_inline_source_cards(source_list: List[Dict], answer_text: str = "", max_cards: int = 3, evidence_bundles: List[Dict] | None = None, question: str = ""):
    """在聊天回答下方直接展示简版原文溯源卡片。"""
    if not source_list:
        return

    cited_indices = extract_cited_indices(answer_text)
    cited_snippets = snippets_from_evidence_bundles(
        evidence_bundles,
        cited_indices,
        max_snippets_per_citation=2,
    )
    if not cited_snippets:
        cited_snippets = extract_cited_snippets(
            answer_text,
            source_list,
            snippet_fetcher=_fetch_original_text_snippets,
        )
    cards: List[tuple[str, str]] = []

    if cited_snippets:
        for idx in sorted(cited_snippets):
            src = next((s for s in source_list if s.get("index") == idx), {})
            title = f"[{idx}] {_display_source_title(src)}"
            for snippet in cited_snippets.get(idx, [])[:2]:
                if snippet:
                    cards.append((title, _compact_source_snippet(snippet, question)))
            if len(cards) >= max_cards:
                break

    if not cards:
        seen = set()
        for src in source_list:
            idx = src.get("index", "?")
            title = f"[{idx}] {_display_source_title(src)}"
            candidates = []
            candidates.extend([str(item) for item in src.get("original_excerpts", []) if str(item or "").strip()])
            if src.get("source_text"):
                candidates.append(str(src.get("source_text")))
            entity_name = str(src.get("entity_name") or "").strip()
            if entity_name:
                candidates.extend(_fetch_original_text_snippets(entity_name, limit=1))
            if src.get("content_preview"):
                candidates.append(str(src.get("content_preview")))
            snippet = next((c.strip() for c in candidates if c and c.strip()), "")
            if not snippet or snippet in seen:
                continue
            seen.add(snippet)
            cards.append((title, _compact_source_snippet(snippet, question)))
            if len(cards) >= max_cards:
                break

    if not cards:
        return

    st.markdown('<div class="block-caption" style="margin-top:12px;">引用原文</div>', unsafe_allow_html=True)
    for title, snippet in cards:
        st.markdown(
            f'<div class="original-source-card">'
            f'<div class="original-source-head"><span class="original-source-title">{html.escape(str(title))}</span></div>'
            f'<div class="original-source-text">{html.escape(str(snippet)[:900])}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _compact_source_snippet(snippet: str, question: str = "") -> str:
    text = str(snippet or "").strip()
    if not text:
        return ""
    cfg = evidence_bundle_config_for_question(question, for_display=True)
    max_chars = int(cfg.get("max_excerpt_chars", 220))
    if len(text) <= max_chars:
        return text
    keywords = _extract_keywords(question, text)
    anchors = _extract_evidence_anchors(question, text)
    return _trim_to_useful_excerpt(text, keywords, max_chars=max_chars, anchors=anchors)



def _extract_cited_snippets(answer: str, sources: List[Dict], window_chars: int = 180) -> Dict[int, List[str]]:
    """从 sources 里为每个引用编号抽取“原文片段”。

    优先从 Neo4j 里取 TextChunk 原文（避免把“知识子图描述”当成引用原文）。
    若 Neo4j 未命中，再回退到 source 的 content_preview。
    """
    if not answer or not sources:
        return {}

    index_to_source = {int(s.get("index")): s for s in sources if isinstance(s.get("index"), int) or str(s.get("index", "")).isdigit()}
    cited = sorted({int(x) for x in re.findall(r"\[(\d+)\]", answer)})
    if not cited:
        return {}

    # 以“。！？；\n”为句边界做一个粗分句，避免只展示 1 行太短
    split_pat = re.compile(r"(?<=[。！？；])|\n")

    out: Dict[int, List[str]] = {}
    for idx in cited:
        src = index_to_source.get(idx)
        if not src:
            continue

        entity_name = (src.get("entity_name") or "").strip()
        if entity_name:
            neo_snips = _fetch_original_text_snippets(entity_name, limit=2)
            if neo_snips:
                out[idx] = neo_snips
                continue

        content = (src.get("content_preview") or "").strip()
        if not content:
            continue

        pieces = [p.strip() for p in split_pat.split(content) if p.strip()]
        # 默认给一个“最相关片段”
        best = pieces[0] if pieces else content[:window_chars]

        # 尝试根据答案中靠近 [idx] 的中文片段进行匹配
        anchor = None
        m = re.search(rf"(.{{0,40}})\[{idx}\]", answer)
        if m:
            anchor = (m.group(1) or "").strip()
            anchor = re.sub(r"[\[\]\(\)\s，,。；;：:]", "", anchor)
            if anchor and len(anchor) < 4:
                anchor = None

        if anchor and pieces:
            for p in pieces:
                normalized = re.sub(r"\s+", "", p)
                if anchor in normalized:
                    best = p
                    break

        out[idx] = [best]

    return out


def render_source_panel(source_list: List[Dict], triples: List[Any], answer_text: str = "", evidence_bundles: List[Dict] | None = None):
    display_triples = _sanitize_triples_for_display(triples)
    summary = build_source_summary(source_list=source_list, triples=display_triples)

    if not source_list and not display_triples:
        st.info("暂无知识溯源信息")
        return

    st.markdown(
        f"""
        <div class="source-summary-grid">
            <div class="source-summary-card"><div class="metric-label">检索支持条目</div><b>{summary['source_count']}</b><div class="soft-caption">原文已在回答下方展示，右侧仅保留检索与图谱核验信息</div></div>
            <div class="source-summary-card"><div class="metric-label">图谱关系</div><b>{summary['triple_count']}</b><div class="soft-caption">用于补充人物、教派、法宝、阵法之间的关系线索</div></div>
        </div>
        <div class="evidence-lead">
            <div class="evidence-lead-title">图谱与检索核验区</div>
            <div class="evidence-lead-body">为避免与回答下方的“引用原文”重复，右侧不再单独展示长原文，只保留知识图谱和检索元信息。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(["🕸️ 知识图谱", "🧭 检索元信息"])

    with tab1:
        if display_triples:
            st.markdown('<div class="block-caption">检索到的知识三元组</div>', unsafe_allow_html=True)
            ZH_REL_MAP = {
                "MASTER_OF": "师父", "APPRENTICE_OF": "徒弟", "BELONGS_TO_SECT": "属于教派",
                "FIGHTS_FOR": "效力于", "FATHER_OF": "父亲", "CHILD_OF": "子女", "BROTHER_OF": "兄弟",
                "MARRIED_TO": "婚配", "KILLS": "击杀", "DEFEATS": "击败", "CAPTURES": "擒获",
                "OPPOSES": "对抗", "OWNS": "拥有", "BESTOWS": "赐予", "CREATES": "创建",
                "DEPLOYS": "布阵", "BREAKS": "破阵", "PARTICIPATES_IN": "参与", "OCCURS_IN": "发生于",
                "LEADS": "率领", "INITIATES": "发起", "LISTED_ON": "上榜封神", "BECOMES": "成为",
                "ALLIES_WITH": "同盟", "BETRAYS": "背叛", "RELATED_TO": "相关", "MENTIONS": "提及",
            }
            tags = []
            for edge in display_triples[:20]:
                h = edge.get("source", "")
                r = edge.get("relation", "")
                t = edge.get("target", "")
                r_zh = ZH_REL_MAP.get(r, r)
                tags.append(f'<span class="triple-tag">({h}, {r_zh}, {t})</span>')

            st.markdown(f'<div class="triple-cloud">{" ".join(tags)}</div>', unsafe_allow_html=True)

            st.markdown('<div class="block-caption">知识图谱子图（可拖动节点）</div>', unsafe_allow_html=True)
            graph_html = build_pyvis_graph(display_triples)
            import streamlit.components.v1 as components
            st.markdown('<div class="graph-shell">', unsafe_allow_html=True)
            components.html(graph_html, height=440, scrolling=False)
            st.markdown('</div>', unsafe_allow_html=True)

            # 图例
            col1, col2, col3, col4 = st.columns(4)
            col1.markdown('<span style="color:#c97d7d">● 人物</span>', unsafe_allow_html=True)
            col2.markdown('<span style="color:#c9a87d">● 教派/势力</span>', unsafe_allow_html=True)
            col3.markdown('<span style="color:#b49ad6">● 法宝</span>', unsafe_allow_html=True)
            col4.markdown('<span style="color:#8f9fcb">● 阵法</span>', unsafe_allow_html=True)
        else:
            st.info(
                "当前回答没有触发图谱子图展示。\n\n"
                "可以尝试提问关系型问题，例如“姜子牙和元始天尊是什么关系？”或“哪吒有哪些法宝？”以触发图谱路径。"
            )

    with tab2:
        if source_list:
            for src in source_list[:6]:
                title = _display_source_title(src)
                search_type = src.get("search_type", "unknown")
                score_val = src.get("score", 0)
                score_pct = f"{score_val:.1%}" if isinstance(score_val, (int, float)) and score_val <= 1 else f"{score_val:.3f}"
                st.markdown(
                    f'<div class="meta-card">'
                    f'<div class="meta-label">条目</div><div class="meta-value"><b>{html.escape(str(title))}</b></div>'
                    f'<div class="meta-label" style="margin-top:8px;">检索方式</div><div class="meta-value">{html.escape(str(search_type))}</div>'
                    f'<div class="meta-label" style="margin-top:8px;">相关度</div><div class="meta-value">{html.escape(str(score_pct))}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("暂无检索元信息")


# ── 数据查看与 CSV 导出 ───────────────────────────────────────
def render_data_export_panel(data_module):
    st.markdown('<div class="panel-title">数据查看与 CSV 导出</div><div class="panel-subtitle">保留研究与导出能力，但不再占据主问答入口。</div>', unsafe_allow_html=True)

    dataset_options = {
        "人物 Persons": "persons",
        "教派/势力 Factions": "factions",
        "地点 Locations": "locations",
        "法宝 Artifacts": "artifacts",
        "坐骑/灵兽 Beasts": "beasts",
        "阵法 Formations": "formations",
        "事件 Events": "events",
        "神位/封号 DeityPositions": "deity_positions",
        "文本片段 TextChunks": "text_chunks",
        "关系 Relations": "relations",
    }
    selected_label = st.selectbox("选择数据集", list(dataset_options.keys()), key="data_export_dataset")
    dataset_name = dataset_options[selected_label]
    relation_limit = 5000
    if dataset_name == "relations":
        relation_limit = st.slider("关系导出上限", 100, 20000, 5000, 100, key="relation_export_limit")

    try:
        rows = data_module.export_dataset_rows(dataset_name, relation_limit=relation_limit)
    except ValueError as exc:
        st.warning(str(exc))
        return
    st.caption(f"共 {len(rows)} 行")

    if not rows:
        st.info("当前数据集暂无内容")
        return

    preview_limit = min(200, len(rows))
    st.dataframe(rows[:preview_limit], use_container_width=True)
    if len(rows) > preview_limit:
        st.caption(f"当前预览前 {preview_limit} 行，下载可获取完整 CSV")

    csv_content = data_module.rows_to_csv(rows)
    st.download_button(
        label=f"下载 {dataset_name}.csv",
        data=csv_content.encode("utf-8-sig"),
        file_name=f"{dataset_name}.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ── 多对话状态管理 ───────────────────────────────────────────
def _ensure_conversations():
    """Initialize independent chat sessions while preserving old single-chat state."""
    if "conversations" not in st.session_state:
        existing_messages = st.session_state.get("messages", [])
        existing_source_history = st.session_state.get("source_history", {})
        existing_active_idx = st.session_state.get("active_source_idx", None)
        st.session_state.conversations = {
            "conv_1": {
                "title": "对话 1",
                "messages": existing_messages,
                "source_history": existing_source_history,
                "active_source_idx": existing_active_idx,
            }
        }
        st.session_state.active_conversation_id = "conv_1"
        st.session_state.next_conversation_id = 2

    if "active_conversation_id" not in st.session_state or st.session_state.active_conversation_id not in st.session_state.conversations:
        st.session_state.active_conversation_id = next(iter(st.session_state.conversations))
    if "next_conversation_id" not in st.session_state:
        st.session_state.next_conversation_id = len(st.session_state.conversations) + 1


def _active_conversation() -> Dict[str, Any]:
    _ensure_conversations()
    return st.session_state.conversations[st.session_state.active_conversation_id]


def _sync_active_conversation_state():
    conv = _active_conversation()
    conv.setdefault("messages", [])
    conv.setdefault("source_history", {})
    conv.setdefault("active_source_idx", None)
    st.session_state.messages = conv["messages"]
    st.session_state.source_history = conv["source_history"]
    st.session_state.active_source_idx = conv.get("active_source_idx")


def _persist_active_conversation_state():
    conv = _active_conversation()
    conv["messages"] = st.session_state.get("messages", [])
    conv["source_history"] = st.session_state.get("source_history", {})
    conv["active_source_idx"] = st.session_state.get("active_source_idx")


def _conversation_label(conv_id: str) -> str:
    conv = st.session_state.conversations[conv_id]
    messages = conv.get("messages", [])
    turns = sum(1 for msg in messages if msg.get("role") == "user")
    return f"{conv.get('title', conv_id)}（{turns}问）"


# ── 主界面 ────────────────────────────────────────────────────

def _clean_answer_display(raw_answer: str) -> str:
    """Strip internal graph notation and code artifacts from displayed answers."""
    import re
    text = str(raw_answer or "")
    # Remove --[RELATION]--> patterns
    text = re.sub(r'\s*--\s*\[([A-Z_]+)\]\s*-->\s*', ' ', text)
    # Remove <RELATION> angle-bracket notation
    text = re.sub(r'\s*<([A-Z_]+)>\s*', ' ', text)
    # Remove bare ALL_CAPS words (common graph identifiers)
    text = re.sub(r'\b(?:MASTER_OF|APPRENTICE_OF|BELONGS_TO_SECT|FIGHTS_FOR|OWNS|BESTOWS|LOSES|KILLS|DEFEATS|CAPTURES|OPPOSES|PARTICIPATES_IN|OCCURS_IN|LEADS|INITIATES|CREATES|LISTED_ON|BECOMES|ALLIES_WITH|BETRAYS|RELATED_TO|MENTIONS|FATHER_OF|CHILD_OF|BROTHER_OF|MARRIED_TO|MEMBER_OF|DEPLOYS|BREAKS|STEALS|IN_COMMUNITY|COMPOSED_OF|BELONGS_TO_CHAPTER|DESCRIBES_EVENT|MENTIONS_PERSON|MENTIONS_ORG|MENTIONS_EVENT)\b', '', text)
    # Clean up double spaces and triple newlines
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Fix lines that became just punctuation
    text = re.sub(r'\n\s*[\-\u2014\u2013]+\s*\n', '\n', text)
    return text.strip()



def main():
    _ensure_conversations()
    _sync_active_conversation_state()

    # ── 侧边栏 ───────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<div class="sidebar-section"><div class="sidebar-kicker">Knowledge Graph RAG</div><h3 style="margin:0 0 4px 0; font-family: \"Noto Serif SC\", serif;">封神演义</h3><div class="panel-subtitle">人物、教派、法宝与阵法关系溯源。</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section"><div class="sidebar-kicker">Workspace Controls</div><div class="panel-title" style="font-size:0.95rem; margin-bottom:10px;">系统配置</div>', unsafe_allow_html=True)

        # Neo4j connection fields (overrides .env)
        if "neo4j_uri" not in st.session_state:
            st.session_state.neo4j_uri = "bolt://localhost:7687"
        if "neo4j_user" not in st.session_state:
            st.session_state.neo4j_user = "neo4j"
        if "neo4j_password" not in st.session_state:
            st.session_state.neo4j_password = ""

        st.session_state.neo4j_uri = st.text_input("Neo4j URI", value=st.session_state.neo4j_uri, key="neo4j_uri_input")
        st.session_state.neo4j_user = st.text_input("Neo4j User", value=st.session_state.neo4j_user, key="neo4j_user_input")
        st.session_state.neo4j_password = st.text_input("Neo4j Password", value=st.session_state.neo4j_password, type="password", key="neo4j_pw_input")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section"><div class="sidebar-kicker">Query Controls</div><div class="panel-title" style="font-size:0.95rem; margin-bottom:10px;">系统配置</div>', unsafe_allow_html=True)

        explain_routing  = st.toggle("显示路由决策",      value=False, help="显示系统选择检索策略的原因")
        force_graph_rag  = st.toggle("强制使用知识图谱",  value=False, help="强制使用图谱检索，关闭智能路由")
        show_triples     = st.toggle("展示图谱三元组",    value=True)
        top_k            = st.slider("检索结果数 (Top-K)", 3, 10, 5)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section"><div class="sidebar-kicker">Runtime</div><div class="panel-title" style="font-size:0.95rem; margin-bottom:10px;">系统状态</div>', unsafe_allow_html=True)

        if "rag_loaded" not in st.session_state:
            st.session_state.rag_loaded = False
            st.session_state.rag_error  = None

        if st.button("🚀 初始化系统", type="primary", use_container_width=True):
            try:
                with st.spinner("正在连接 Neo4j 并加载《封神演义》图谱数据..."):
                    st.session_state.rag        = load_rag_system(
                        neo4j_uri=st.session_state.neo4j_uri,
                        neo4j_user=st.session_state.neo4j_user,
                        neo4j_password=st.session_state.neo4j_password,
                    )
                    st.session_state.rag_loaded = True
                    st.session_state.rag_error  = None
                st.success("✅ 《封神演义》知识图谱系统就绪！")
            except Exception as e:
                st.session_state.rag_loaded = False
                st.session_state.rag_error  = str(e)
                st.error(f"❌ 初始化失败：{e}")

        if st.session_state.get("rag_loaded"):
            rag = st.session_state.rag
            try:
                stats = rag["data_module"].get_statistics()
                st.metric("封神人物", stats.get("total_persons", 0))
                st.metric("教派/势力", stats.get("total_factions", 0))
                st.metric("法宝/阵法", stats.get("total_artifacts", 0) + stats.get("total_formations", 0))
                route_stats = rag["router"].get_route_statistics()
                st.metric("总查询次数", route_stats.get("total_queries", 0))
            except Exception:
                pass
        elif st.session_state.get("rag_error"):
            st.error("系统未就绪")
        else:
            st.info("点击上方按钮启动系统")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section"><div class="sidebar-kicker">Conversations</div><div class="panel-title" style="font-size:0.95rem; margin-bottom:10px;">对话管理</div>', unsafe_allow_html=True)
        conv_ids = list(st.session_state.conversations.keys())
        active_id = st.session_state.active_conversation_id
        selected_id = st.selectbox(
            "当前对话",
            conv_ids,
            index=conv_ids.index(active_id) if active_id in conv_ids else 0,
            format_func=_conversation_label,
            key="conversation_selector",
        )
        if selected_id != active_id:
            _persist_active_conversation_state()
            st.session_state.active_conversation_id = selected_id
            _sync_active_conversation_state()
            st.rerun()

        col_new, col_clear = st.columns(2)
        with col_new:
            if st.button("➕ 新建", use_container_width=True):
                _persist_active_conversation_state()
                next_id = st.session_state.next_conversation_id
                conv_id = f"conv_{next_id}"
                st.session_state.conversations[conv_id] = {
                    "title": f"对话 {next_id}",
                    "messages": [],
                    "source_history": {},
                    "active_source_idx": None,
                }
                st.session_state.next_conversation_id = next_id + 1
                st.session_state.active_conversation_id = conv_id
                _sync_active_conversation_state()
                st.rerun()
        with col_clear:
            if st.button("🗑️ 清空", use_container_width=True):
                conv = _active_conversation()
                conv["messages"] = []
                conv["source_history"] = {}
                conv["active_source_idx"] = None
                _sync_active_conversation_state()
                st.rerun()

        if len(st.session_state.conversations) > 1 and st.button("删除当前对话", use_container_width=True):
            current_id = st.session_state.active_conversation_id
            del st.session_state.conversations[current_id]
            st.session_state.active_conversation_id = next(iter(st.session_state.conversations))
            _sync_active_conversation_state()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section"><div class="sidebar-kicker">Prompt Ideas</div>', unsafe_allow_html=True)
        st.caption("💡 推荐提问")
        demo_questions = [
            "纣王是谁？",
            "哪吒的师父是谁？他属于哪个教派？有哪些法宝？",
            "姜子牙和元始天尊是什么关系？",
            "孙悟空在《封神演义》中有什么法宝？",
        ]

        for q in demo_questions:
            if st.button(q, use_container_width=True, key=f"demo_{st.session_state.active_conversation_id}_{q}"):
                st.session_state.pending_question = q
        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.get("rag_loaded"):
            with st.expander("数据查看与 CSV 导出", expanded=False):
                render_data_export_panel(st.session_state.rag["data_module"])

    if "messages"       not in st.session_state:
        st.session_state.messages       = []
    if "source_history" not in st.session_state:
        st.session_state.source_history = {}

    stats = {}
    route_stats = {}
    if st.session_state.get("rag_loaded"):
        try:
            stats = st.session_state.rag["data_module"].get_statistics()
            route_stats = st.session_state.rag["router"].get_route_statistics()
        except Exception:
            stats = {}
            route_stats = {}

    render_hero_header(
        rag_loaded=bool(st.session_state.get("rag_loaded")),
        stats=stats,
        route_stats=route_stats,
    )

    chat_col, source_col = st.columns([3, 2])

    with chat_col:
        st.markdown(
            '<div class="panel-shell"><div class="panel-title">提问与回答</div><div class="panel-subtitle">主舞台保留对话流，但把策略、证据数和生成模式压缩进回答头部，尽量减少视觉噪音。</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="query-note">推荐提问：人物身份、师承关系、教派阵营、法宝阵法、封神结局。回答生成后，下方会显示原文，右侧可核对图谱子图。</div>',
            unsafe_allow_html=True,
        )
        # 显示历史消息
        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant":
                    hist = st.session_state.source_history.get(i, {})
                    sources = hist.get("sources", [])
                    render_answer_header(
                        strategy="archived",
                        docs_count=len(sources),
                        answer_text=msg["content"],
                    )
                label = "Assistant" if msg["role"] == "assistant" else "Question"
                body_class = "assistant" if msg["role"] == "assistant" else "user"
                st.markdown(
                    f'<div class="chat-message-note">{html.escape(label)}</div>'
                    f'<div class="chat-message-body {body_class}">{html.escape(msg["content"])}</div>',
                    unsafe_allow_html=True,
                )
                if msg["role"] == "assistant" and i in st.session_state.source_history:
                    hist = st.session_state.source_history.get(i, {})
                    render_inline_source_cards(
                        hist.get("sources", []),
                        answer_text=hist.get("answer", msg["content"]),
                        max_cards=5,
                        evidence_bundles=hist.get("evidence_bundles", []),
                        question=hist.get("question", ""),
                    )
                    if st.button(f"查看证据视图", key=f"src_btn_{st.session_state.active_conversation_id}_{i}"):
                        st.session_state.active_source_idx = i
                        _persist_active_conversation_state()

        # 用户输入
        pre_fill   = st.session_state.pop("pending_question", None)
        user_input = st.chat_input("请输入您的封神演义问题，例如：哪吒的师父是谁？他有哪些法宝？")
        if pre_fill and not user_input:
            user_input = pre_fill

        if user_input:
            if not st.session_state.get("rag_loaded"):
                with st.chat_message("assistant"):
                    st.error("❌ 系统未初始化，请点击侧边栏的「🚀 初始化系统」按钮")
            else:
                rag = st.session_state.rag
                st.session_state.messages.append({"role": "user", "content": user_input})
                conv = _active_conversation()
                if conv.get("title", "").startswith("对话 ") and len(st.session_state.messages) == 1:
                    conv["title"] = user_input[:16] + ("…" if len(user_input) > 16 else "")
                _persist_active_conversation_state()
                with st.chat_message("user"):
                    st.markdown(
                        f'<div class="chat-message-note">Question</div>'
                        f'<div class="chat-message-body user">{html.escape(user_input)}</div>',
                        unsafe_allow_html=True,
                    )

                with st.chat_message("assistant"):
                    with st.spinner("🔍 正在查阅《封神演义》原文与知识图谱..."):
                        try:
                            router     = rag["router"]
                            gen_module = rag["gen_module"]
                            data_module= rag["data_module"]
                            cfg        = rag["config"]

                            if force_graph_rag:
                                docs = router.graph_rag_retrieval.graph_rag_search(user_input, top_k)
                                st.caption("🕸️ **强制启用知识图谱检索 (Graph RAG)**")
                            else:
                                if explain_routing:
                                    explanation = router.explain_routing_decision(user_input)
                                    with st.expander("🗺️ 路由决策详情"):
                                        st.text(explanation)
                                docs, analysis = router.route_query(user_input, top_k)
                                strategy_icons = {
                                    "hybrid_traditional": "🔍",
                                    "graph_rag":          "🕸️",
                                    "combined":           "🔄",
                                }
                                strategy = analysis.recommended_strategy.value
                                icon     = strategy_icons.get(strategy, "❓")
                                st.caption(f"{icon} 策略: **{strategy}** · 复杂度: {analysis.query_complexity:.2f} · 关系密集度: {analysis.relationship_intensity:.2f}")

                            if not docs:
                                answer      = "抱歉，知识图谱中没有找到与您问题相关的信息。\n\n请尝试：\n- 使用具体的人物名称或关系查询\n- 确认 Neo4j 数据库已导入《封神演义》数据"
                                sources     = []
                                triples_list= []
                                render_answer_header(
                                    strategy="no_result",
                                    docs_count=0,
                                    answer_text=answer,
                                )
                            else:
                                docs = _backfill_original_text(docs)
                                evidence_cfg = evidence_bundle_config_for_question(user_input, for_display=True)
                                evidence_bundles = build_evidence_bundles(docs, **evidence_cfg)
                                answer  = gen_module.generate_adaptive_answer(user_input, docs)

                                bundle_by_index = {int(b.get("number")): b for b in evidence_bundles if str(b.get("number", "")).isdigit()}
                                sources = []
                                for i, doc in enumerate(docs):
                                    metadata = doc.metadata or {}
                                    entity_name = metadata.get("entity_name", "未知")
                                    bundle = bundle_by_index.get(i + 1, {})
                                    sources.append({
                                        "index": i + 1,
                                        "entity_name": entity_name,
                                        "raw_entity_name": entity_name,
                                        "search_type": metadata.get("search_type", "未知"),
                                        "score": metadata.get("score", metadata.get("final_score", metadata.get("relevance_score", 0))),
                                        "content_preview": doc.page_content.strip(),
                                        "node_id": metadata.get("node_id", ""),
                                        "chunk_id": metadata.get("chunk_id", ""),
                                        "source_chunk_id": metadata.get("source_chunk_id", metadata.get("chunk_id", "")),
                                        "source_text": metadata.get("source_text", ""),
                                        "source_snippets": metadata.get("source_snippets", []),
                                        "original_excerpts": [item.get("text", "") for item in bundle.get("original_excerpts", []) if isinstance(item, dict)],
                                        "has_original_text": bool(bundle.get("has_original_text")),
                                        "node_type": metadata.get("node_type", ""),
                                        "subgraph_edges": metadata.get("subgraph_edges", []),
                                    })

                                # 提取图谱三元组
                                raw_triples  = []
                                entity_names = []
                                for doc in docs:
                                    nm = doc.metadata.get("entity_name", "")
                                    if nm:
                                        entity_names.append(nm)
                                    for e in doc.metadata.get("subgraph_edges", []):
                                        raw_triples.append(e)

                                if not raw_triples and entity_names and show_triples:
                                    try:
                                        db_triples = data_module.export_triples(
                                            entity_names=entity_names[:5], limit=40
                                        )
                                        for s, r, t in db_triples:
                                            raw_triples.append({"source": s, "relation": r, "target": t})
                                    except Exception:
                                        pass

                                triples_list = raw_triples

                                strategy_for_badge = "graph_rag" if force_graph_rag else strategy
                                render_answer_header(
                                    strategy=strategy_for_badge,
                                    docs_count=len(docs),
                                    answer_text=answer,
                                )

                            st.markdown(
                                f'<div class="chat-message-note">Assistant</div>'
                                f'<div class="chat-message-body assistant">{html.escape(answer)}</div>',
                                unsafe_allow_html=True,
                            )
                            render_inline_source_cards(
                                sources,
                                answer_text=answer,
                                max_cards=5,
                                evidence_bundles=evidence_bundles,
                                question=user_input,
                            )

                            msg_idx = len(st.session_state.messages)
                            st.session_state.source_history[msg_idx] = {
                                "sources": sources,
                                "triples": triples_list,
                                "answer": answer,
                                "evidence_bundles": evidence_bundles,
                                "question": user_input,
                            }
                            st.session_state.active_source_idx = msg_idx
                            st.session_state.messages.append({
                                "role":    "assistant",
                                "content": _clean_answer_display(answer)
                            })
                            _persist_active_conversation_state()

                        except Exception as e:
                            err_msg = f"抱歉，处理问题时出现错误：{str(e)}"
                            st.error(err_msg)
                            st.session_state.messages.append({
                                "role":    "assistant",
                                "content": err_msg
                            })
                            _persist_active_conversation_state()

    # ── 右侧知识溯源面板 ─────────────────────────────────────────
    with source_col:
        st.markdown(
            '<div class="panel-shell"><div class="panel-title">证据与图谱</div><div class="panel-subtitle">这里汇总回答所依赖的原文片段、知识三元组和检索元信息。</div></div>',
            unsafe_allow_html=True,
        )
        active_idx = st.session_state.get("active_source_idx")
        if active_idx is not None and active_idx in st.session_state.source_history:
            hist = st.session_state.source_history[active_idx]
            render_source_panel(
                hist.get("sources", []),
                hist.get("triples", []),
                answer_text=hist.get("answer", ""),
                evidence_bundles=hist.get("evidence_bundles", []),
            )
        else:
            if st.session_state.get("rag_loaded"):
                st.markdown("""
                <div class="evidence-lead">
                    <div class="evidence-lead-title">&#x1F50D; 等待提问</div>
                    <div class="evidence-lead-body">
                        在左侧输入框中提出《封神演义》相关问题，回答生成后，右侧面板将展示：<br>
                        <strong>原文片段</strong> — 答案引用的原始文本证据<br>
                        <strong>知识图谱</strong> — 人物、教派、法宝之间的关联关系<br>
                        <strong>可视化子图</strong> — 可交互拖拽的图谱节点与连线
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="evidence-lead">
                    <div class="evidence-lead-title">&#x1F4DA; 封神演义 · 知识图谱问答</div>
                    <div class="evidence-lead-body">
                        点击侧边栏 <strong>初始化系统</strong> 按钮，连接 Neo4j 知识图谱后开始提问。<br>
                        系统将结合原文检索与图谱推理，给出带证据溯源的精准答案。
                    </div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("""
        <div style="color:#7e8fa3; font-size:0.75em; text-align:center; padding-top:4px;">
        数据来源：《封神演义》TXT | 知识图谱由LLM自动抽取
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
