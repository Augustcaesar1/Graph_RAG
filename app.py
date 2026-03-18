"""
东周列国历史知识图谱 - GraphRAG 问答系统
Streamlit Web界面 | 古风黑金主题
"""

import os
import sys
import json
import logging
import streamlit as st
from typing import List, Dict, Any

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# ── 页面基础配置 ─────────────────────────────────────────────
st.set_page_config(
    page_title="东周列国 · 历史知识图谱",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── 古风黑金 CSS ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;700&display=swap');

/* 整体背景 */
.stApp {
    background: linear-gradient(135deg, #0d0d0d 0%, #1a1000 50%, #0d0800 100%);
    font-family: 'Noto Serif SC', serif;
}

/* 侧边栏 */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0800 0%, #1a1200 100%);
    border-right: 1px solid #8b6914;
}
section[data-testid="stSidebar"] * { color: #d4af37 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #ffd700 !important; }

/* 主内容区文字颜色 */
.stMarkdown p, .stMarkdown li, .stMarkdown td, .stMarkdown th { color: #e8d5a3 !important; }
h1, h2, h3 { color: #ffd700 !important; }

/* 聊天消息 */
.stChatMessage { border-radius: 8px; margin-bottom: 8px; }
[data-testid="stChatMessageContent"] { color: #e8d5a3 !important; }

/* 知识溯源卡片 */
.source-card {
    background: linear-gradient(135deg, rgba(20, 14, 0, 0.9) 0%, rgba(30, 20, 0, 0.9) 100%);
    border-left: 4px solid #d4af37;
    border-radius: 6px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.85em;
    box-shadow: 0 2px 8px rgba(212, 175, 55, 0.15);
    color: #e8d5a3;
}
/* 三元组标签 */
.triple-tag {
    display: inline-block;
    background: rgba(212, 175, 55, 0.1);
    border: 1px solid #8b6914;
    border-radius: 4px;
    padding: 2px 8px;
    margin: 2px;
    font-size: 0.78em;
    color: #ffd700;
    font-family: monospace;
}
/* 分割线 */
hr { border-color: #8b6914 !important; }

/* 按钮样式 */
.stButton > button {
    background: linear-gradient(135deg, #2a1a00 0%, #3d2500 100%) !important;
    border: 1px solid #8b6914 !important;
    color: #ffd700 !important;
    border-radius: 4px !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #3d2500 0%, #5c3800 100%) !important;
    border-color: #ffd700 !important;
}
/* 标题装饰 */
.main-title {
    text-align: center;
    padding: 20px 0;
    border-bottom: 2px solid #8b6914;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ── 初始化 RAG 系统 ──────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ 正在初始化历史知识图谱系统...")
def load_rag_system():
    from config import DEFAULT_CONFIG
    from rag_modules import GraphDataPreparationModule, GenerationIntegrationModule
    from rag_modules.faiss_index_construction import FaissIndexConstructionModule
    from rag_modules.hybrid_retrieval import HybridRetrievalModule
    from rag_modules.graph_rag_retrieval_new import GraphRAGRetrieval
    from rag_modules.intelligent_query_router import IntelligentQueryRouter

    cfg = DEFAULT_CONFIG

    data_module  = GraphDataPreparationModule(
        uri=cfg.neo4j_uri, user=cfg.neo4j_user,
        password=cfg.neo4j_password, database=cfg.neo4j_database
    )
    index_module = FaissIndexConstructionModule(
        persist_directory=cfg.faiss_index_path,
        model_name=cfg.embedding_model
    )
    gen_module   = GenerationIntegrationModule(
        model_name=cfg.llm_model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens
    )
    trad_retrieval  = HybridRetrievalModule(
        config=cfg, index_module=index_module,
        data_module=data_module, llm_client=gen_module.client
    )
    graph_retrieval = GraphRAGRetrieval(config=cfg, llm_client=gen_module.client)
    router          = IntelligentQueryRouter(
        traditional_retrieval=trad_retrieval,
        graph_rag_retrieval=graph_retrieval,
        llm_client=gen_module.client,
        config=cfg
    )

    # 构建知识库
    data_module.load_graph_data()
    data_module.build_recipe_documents()
    chunks = data_module.chunk_documents(
        chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap
    )

    if not chunks:
        logger.warning("数据库无数据，提供虚拟文档避免崩溃")
        from langchain_core.documents import Document
        chunks = [Document(
            page_content="欢迎使用东周列国历史知识图谱！请先运行导入脚本填充数据库。",
            metadata={"source": "system_init", "recipe_name": "系统初始化"}
        )]

    try:
        if index_module.has_collection() and index_module.load_collection():
            pass
        else:
            index_module.build_vector_index(chunks)
    except Exception as e:
        logger.warning(f"向量库初始化失败，系统降级为仅图谱模式: {e}")

    trad_retrieval.initialize(chunks)
    graph_retrieval.initialize()

    return {
        "data_module":  data_module,
        "index_module": index_module,
        "gen_module":   gen_module,
        "router":       router,
        "config":       cfg,
    }


# ── 知识图谱可视化 ────────────────────────────────────────────
def build_pyvis_graph(triples: List[Dict[str, Any]], highlight_nodes: List[str] = None) -> str:
    try:
        from pyvis.network import Network
    except ImportError:
        return "<p style='color:#d4af37'>请安装 pyvis：pip install pyvis</p>"

    highlight_nodes = set(highlight_nodes or [])

    net = Network(
        height="420px", width="100%",
        bgcolor="#0a0800", font_color="#ffd700",
        directed=True
    )
    net.set_options(json.dumps({
        "nodes": {
            "shape": "dot",
            "size": 18,
            "font": {"size": 13, "color": "#ffd700"},
            "borderWidth": 2
        },
        "edges": {
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.7}},
            "color": {"color": "#d4af37", "opacity": 0.85},
            "font": {"size": 10, "color": "#a08030", "align": "middle"},
            "smooth": {"type": "curvedCW", "roundness": 0.2}
        },
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -60,
                "centralGravity": 0.01,
                "springLength": 110
            },
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 150}
        },
        "interaction": {"hover": True, "tooltipDelay": 200}
    }))

    # 节点颜色 - 东周历史主题
    COLOR_MAP = {
        "Person":  "#ffd700",    # 金色 - 人物
        "Event":   "#c0392b",    # 暗红 - 战争/事件
        "State":   "#1abc9c",    # 青色 - 诸侯国
        "Concept": "#9b59b6",    # 紫色 - 概念
    }

    # 关系中文翻译
    ZH_REL_MAP = {
        "BELONGS_TO":         "效力于",
        "ATTACKED_IN":        "主攻参与",
        "DEFENDED_IN":        "主守参与",
        "ASSISTED_ATTACK_IN": "协助进攻",
        "ASSISTED_DEFEND_IN": "协助防守",
        "FRIEND_OF":          "好友",
        "ALLY_OF":            "盟友",
        "RIVAL_OF":           "对手",
        "ENEMY_OF":           "敌人",
        "SIBLING":            "兄弟姐妹",
        "FATHER_SON":         "父子",
        "FATHER_DAUGHTER":    "父女",
        "TEACHER_STUDENT":    "师生",
        "TEACHER_OF":         "老师",
        "DISCIPLE_OF":        "弟子",
        "LORD_MINISTER":      "君臣",
        "MASTER_SERVANT":     "主仆",
        "SPOUSE":             "夫妻",
        "LOVER_OF":           "情人",
        "MOTHER_SON":         "母子",
        "RELATED_TO":         "相关",
        # 旧菜谱关系（保留兼容）
        "REQUIRES":             "需要",
        "BELONGS_TO_CATEGORY":  "属于",
        "HAS_STEP":             "步骤",
    }

    RELATION_COLOR = {
        "RIVAL_OF":   "#c0392b",
        "ENEMY_OF":   "#e74c3c",
        "FRIEND_OF":  "#2ecc71",
        "ALLY_OF":    "#27ae60",
        "BELONGS_TO": "#1abc9c",
        "ATTACKED_IN":"#e67e22",
        "DEFENDED_IN":"#3498db",
        "FATHER_SON": "#9b59b6",
        "SIBLING":    "#8e44ad",
        "LORD_MINISTER": "#f39c12",
        "TEACHER_STUDENT": "#16a085",
        "SPOUSE":     "#e91e8c",
    }

    def guess_node_type(name: str) -> str:
        states = {"齐国","鲁国","晋国","楚国","秦国","燕国","宋国","卫国",
                  "郑国","陈国","蔡国","吴国","越国","赵国","魏国","韩国"}
        if name in states or name.endswith("国"):
            return "State"
        events = {"之战","之盟","合纵","连横","变法","之乱"}
        if any(e in name for e in events):
            return "Event"
        return "Person"

    added_nodes = set()

    def add_node(name: str, ntype: str = None, description: str = ""):
        if name in added_nodes:
            return
        added_nodes.add(name)
        if ntype is None:
            ntype = guess_node_type(name)
        color  = COLOR_MAP.get(ntype, "#aaaaaa")
        border = "#ffffff" if name in highlight_nodes else color
        size   = 24 if name in highlight_nodes else 16
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

    for edge in triples:
        head = edge.get("source")
        rel  = edge.get("relation")
        tail = edge.get("target")
        if not head or not tail:
            continue
        add_node(head, edge.get("source_type"), edge.get("source_desc", ""))
        add_node(tail, edge.get("target_type"), edge.get("target_desc", ""))
        label = ZH_REL_MAP.get(rel, rel)
        color = RELATION_COLOR.get(rel, "#8b6914")
        net.add_edge(head, tail, label=label, color=color)

    return net.generate_html()


# ── 来源面板渲染 ──────────────────────────────────────────────
def render_source_panel(source_list: List[Dict], triples: List[Any]):
    if not source_list and not triples:
        st.info("暂无知识溯源信息")
        return

    tab1, tab2 = st.tabs(["📄 文献片段", "🕸️ 知识图谱"])

    with tab1:
        if source_list:
            for src in source_list[:5]:
                search_badge = {
                    "graph_rag":          "🕸️ 图RAG",
                    "graph_path":         "🕸️ 图路径",
                    "knowledge_subgraph": "🕸️ 子图",
                    "dual_level":         "🔍 双层检索",
                    "vector_enhanced":    "📐 向量",
                    "hybrid_traditional": "🔍 混合",
                }.get(src.get("search_type", ""), "❓")
                score_val = src.get('score', 0)
                score_pct = f"{score_val:.1%}" if score_val <= 1 else f"{score_val:.3f}"
                entity_name = src.get("recipe_name", src.get("entity_name", "未知"))
                st.markdown(
                    f'<div class="source-card">'
                    f'<b>[{src["index"]}] {entity_name}</b> '
                    f'{search_badge} · 相关度: {score_pct}<br>'
                    f'<span style="color:#c8a85a">{src["content_preview"][:200]}…</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        else:
            st.info("无文献片段引用")

    with tab2:
        if triples:
            st.markdown("**检索到的知识三元组：**")
            ZH_REL_MAP = {
                "BELONGS_TO": "效力于", "ATTACKED_IN": "主攻参与", "DEFENDED_IN": "主守参与",
                "FRIEND_OF": "好友", "ALLY_OF": "盟友", "RIVAL_OF": "对手",
                "ENEMY_OF": "敌人", "SIBLING": "兄弟姐妹", "FATHER_SON": "父子",
                "TEACHER_STUDENT": "师生", "LORD_MINISTER": "君臣", "SPOUSE": "夫妻",
                "RELATED_TO": "相关", "ASSISTED_ATTACK_IN": "协助进攻", "ASSISTED_DEFEND_IN": "协助防守"
            }
            tags = []
            for edge in triples[:20]:
                if isinstance(edge, dict):
                    h = edge.get("source", "")
                    r = edge.get("relation", "")
                    t = edge.get("target", "")
                elif isinstance(edge, (list, tuple)) and len(edge) >= 3:
                    h, r, t = edge[0], edge[1], edge[2]
                else:
                    continue
                r_zh = ZH_REL_MAP.get(r, r)
                tags.append(f'<span class="triple-tag">({h}, {r_zh}, {t})</span>')

            st.markdown(" ".join(tags), unsafe_allow_html=True)
            st.markdown("---")

            st.markdown("**知识图谱子图（可拖动节点）：**")
            graph_html = build_pyvis_graph(triples)
            import streamlit.components.v1 as components
            components.html(graph_html, height=440, scrolling=False)

            # 图例
            col1, col2, col3 = st.columns(3)
            col1.markdown('<span style="color:#ffd700">● 人物</span>', unsafe_allow_html=True)
            col2.markdown('<span style="color:#c0392b">● 战争/事件</span>', unsafe_allow_html=True)
            col3.markdown('<span style="color:#1abc9c">● 诸侯国</span>', unsafe_allow_html=True)
        else:
            st.info(
                "该查询未触发图谱检索，图谱子图为空。\n\n"
                "💡 尝试问「孙膑和庞涓的关系？」或「长平之战有哪些人参与？」以触发图RAG。"
            )


# ── 主界面 ────────────────────────────────────────────────────
def main():
    # ── 侧边栏 ───────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚔️ 东周列国")
        st.markdown("**历史知识图谱问答系统**")
        st.markdown("---")
        st.markdown("### ⚙️ 系统配置")

        explain_routing  = st.toggle("显示路由决策",      value=False, help="显示系统选择检索策略的原因")
        force_graph_rag  = st.toggle("强制使用知识图谱",  value=False, help="强制使用图谱检索，关闭智能路由")
        show_triples     = st.toggle("展示图谱三元组",    value=True)
        top_k            = st.slider("检索结果数 (Top-K)", 3, 10, 5)

        st.markdown("---")
        st.markdown("### 📊 系统状态")

        if "rag_loaded" not in st.session_state:
            st.session_state.rag_loaded = False
            st.session_state.rag_error  = None

        if st.button("🚀 初始化系统", type="primary", use_container_width=True):
            try:
                with st.spinner("正在连接Neo4j和加载历史数据..."):
                    st.session_state.rag        = load_rag_system()
                    st.session_state.rag_loaded = True
                    st.session_state.rag_error  = None
                st.success("✅ 历史知识图谱系统就绪！")
            except Exception as e:
                st.session_state.rag_loaded = False
                st.session_state.rag_error  = str(e)
                st.error(f"❌ 初始化失败：{e}")

        if st.session_state.get("rag_loaded"):
            rag = st.session_state.rag
            try:
                stats = rag["data_module"].get_statistics()
                st.metric("历史人物", stats.get("total_persons", stats.get("total_recipes", 0)))
                st.metric("历史事件", stats.get("total_events",  stats.get("total_ingredients", 0)))
                st.metric("诸侯国",   stats.get("total_states",  stats.get("total_cooking_steps", 0)))
                route_stats = rag["router"].get_route_statistics()
                st.metric("总查询次数", route_stats.get("total_queries", 0))
            except Exception:
                pass
        elif st.session_state.get("rag_error"):
            st.error("系统未就绪")
        else:
            st.info("点击上方按钮启动系统")

        st.markdown("---")
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.messages       = []
            st.session_state.source_history = {}
            st.rerun()

        st.markdown("---")
        st.caption("💡 **推荐提问：**")
        demo_questions = [
            "孙膑和庞涓是什么关系？",
            "齐桓公的主要事迹是什么？",
            "长平之战有哪些人参与？",
            "管仲辅佐了哪位君主？",
            "春秋时期有哪些著名战争？",
        ]
        for q in demo_questions:
            if st.button(q, use_container_width=True, key=f"demo_{q}"):
                st.session_state.pending_question = q

    # ── 主区域 ──────────────────────────────────────────────────
    st.markdown("""
    <div class="main-title">
        <h1 style="font-size:2em; color:#ffd700; letter-spacing:0.1em;">⚔️ 东周列国 · 历史知识图谱</h1>
        <p style="color:#a08030;">基于 <b>知识图谱 + 向量检索</b> 的春秋战国历史问答系统 ·
        图数据库: Neo4j · 向量库: FAISS · 大模型: DeepSeek</p>
    </div>
    """, unsafe_allow_html=True)

    if "messages"       not in st.session_state:
        st.session_state.messages       = []
    if "source_history" not in st.session_state:
        st.session_state.source_history = {}

    chat_col, source_col = st.columns([3, 2])

    with chat_col:
        # 显示历史消息
        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and i in st.session_state.source_history:
                    if st.button(f"📚 查看知识溯源", key=f"src_btn_{i}"):
                        st.session_state.active_source_idx = i

        # 用户输入
        pre_fill   = st.session_state.pop("pending_question", None)
        user_input = st.chat_input("请输入您的历史问题，例如：孙膑和庞涓是什么关系？")
        if pre_fill and not user_input:
            user_input = pre_fill

        if user_input:
            if not st.session_state.get("rag_loaded"):
                with st.chat_message("assistant"):
                    st.error("❌ 系统未初始化，请点击侧边栏的「🚀 初始化系统」按钮")
            else:
                rag = st.session_state.rag
                st.session_state.messages.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)

                with st.chat_message("assistant"):
                    with st.spinner("🔍 正在查阅历史文献与知识图谱..."):
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
                                answer      = "抱歉，知识图谱中没有找到与您问题相关的历史信息。\n\n请尝试：\n- 使用更具体的人物名或事件名\n- 确认Neo4j数据库已导入东周列国数据"
                                sources     = []
                                triples_list= []
                            else:
                                answer  = gen_module.generate_adaptive_answer(user_input, docs)

                                sources = []
                                for i, doc in enumerate(docs):
                                    entity_name = doc.metadata.get("entity_name",
                                                  doc.metadata.get("recipe_name", "未知"))
                                    sources.append({
                                        "index":          i + 1,
                                        "recipe_name":    entity_name,
                                        "entity_name":    entity_name,
                                        "search_type":    doc.metadata.get("search_type", "未知"),
                                        "score":          doc.metadata.get("score", 0),
                                        "content_preview":doc.page_content.strip()
                                    })

                                # 提取图谱三元组
                                raw_triples  = []
                                entity_names = []
                                for doc in docs:
                                    nm = doc.metadata.get("entity_name", doc.metadata.get("recipe_name", ""))
                                    if nm:
                                        entity_names.append(nm)
                                    for e in doc.metadata.get("subgraph_edges", []):
                                        raw_triples.append(e)

                                if not raw_triples and entity_names and show_triples:
                                    try:
                                        db_triples = data_module.export_triples(
                                            recipe_names=entity_names[:5], limit=40
                                        )
                                        for s, r, t in db_triples:
                                            raw_triples.append({"source": s, "relation": r, "target": t})
                                    except Exception:
                                        pass

                                triples_list = raw_triples

                            st.markdown(answer)

                            msg_idx = len(st.session_state.messages)
                            st.session_state.source_history[msg_idx] = {
                                "sources": sources,
                                "triples": triples_list
                            }
                            st.session_state.active_source_idx = msg_idx
                            st.session_state.messages.append({
                                "role":    "assistant",
                                "content": answer
                            })

                        except Exception as e:
                            err_msg = f"抱歉，处理问题时出现错误：{str(e)}"
                            st.error(err_msg)
                            st.session_state.messages.append({
                                "role":    "assistant",
                                "content": err_msg
                            })

    # ── 右侧知识溯源面板 ─────────────────────────────────────────
    with source_col:
        st.markdown("### 📚 知识溯源")
        active_idx = st.session_state.get("active_source_idx")
        if active_idx is not None and active_idx in st.session_state.source_history:
            hist = st.session_state.source_history[active_idx]
            render_source_panel(hist.get("sources", []), hist.get("triples", []))
        else:
            st.info(
                "💡 提问后，此处将显示：\n\n"
                "- 📄 引用的历史文献片段\n"
                "- 🕸️ 检索到的知识图谱子图（可交互）\n"
                "- 知识三元组（人物, 关系, 人物/事件）"
            )
        st.markdown("---")
        st.markdown("""
        <div style="color:#6b5020; font-size:0.75em; text-align:center;">
        ⚔️ 数据来源：东周列国知识图谱 · 春秋战国 BCE770-BCE221
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
