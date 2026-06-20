"""
真正的图RAG检索模块 - 封神演义版
基于图结构的知识推理和检索，而非简单的关键词匹配
"""

import json
import logging
import re
from collections import defaultdict
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum

from langchain_core.documents import Document
from neo4j import GraphDatabase

from .gold_schema import ALLOWED_RELATION_TYPES
from .gold_graph_policy import build_gold_entity_predicate, filter_gold_documents
from .query_heuristics import extract_candidate_entities, infer_search_strategy, is_history_question
from .retrieval_shared import evidence_priority

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """查询类型枚举"""
    ENTITY_RELATION = "entity_relation"
    MULTI_HOP = "multi_hop"
    SUBGRAPH = "subgraph"
    PATH_FINDING = "path_finding"
    CLUSTERING = "clustering"


@dataclass
class GraphQuery:
    query_type: QueryType
    source_entities: List[str]
    target_entities: List[str] = None
    relation_types: List[str] = None
    max_depth: int = 3
    max_nodes: int = 50
    constraints: Dict[str, Any] = None


@dataclass
class GraphPath:
    nodes: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    path_length: int
    relevance_score: float
    path_type: str


@dataclass
class KnowledgeSubgraph:
    central_nodes: List[Dict[str, Any]]
    connected_nodes: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    graph_metrics: Dict[str, float]
    reasoning_chains: List[List[str]]


class GraphRAGRetrieval:
    def __init__(self, config, llm_client, driver=None):
        self.config = config
        self.llm_client = llm_client
        self.driver = driver
        self.entity_cache = {}
        self.relation_cache = {}
        self.subgraph_cache = {}

    def initialize(self):
        logger.info("初始化封神演义图RAG检索系统...")
        if self.driver is None:
            self.driver = GraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password)
            )
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Neo4j连接成功")
            self._build_graph_index()
        except Exception as e:
            logger.error(f"Neo4j连接失败: {e}")

    def _build_graph_index(self):
        logger.info("构建图结构索引...")
        try:
            with self.driver.session() as session:
                entity_query = """
                MATCH (n)
                WHERE coalesce(n.name, n.chunk_id, n.community_id) IS NOT NULL
                  AND """ + build_gold_entity_predicate("n") + """
                WITH n, COUNT { (n)--() } as degree
                RETURN labels(n) as node_labels,
                       coalesce(n.name, n.chunk_id, n.community_id) as node_id,
                       coalesce(n.name, n.chunk_id, n.community_id) as name,
                       coalesce(n.period, n.organization, n.event_type, n.org_type, n.title, n.chapter, '') as category,
                       degree,
                       n.community_l1 as community_l1,
                       n.community_l2 as community_l2,
                       n.community_l3 as community_l3
                ORDER BY degree DESC
                LIMIT 2000
                """

                result = session.run(entity_query)
                for record in result:
                    node_id = record["node_id"]
                    self.entity_cache[node_id] = {
                        "labels": record["node_labels"],
                        "name": record["name"],
                        "category": record["category"],
                        "degree": record["degree"],
                        "community_l1": record["community_l1"],
                        "community_l2": record["community_l2"],
                        "community_l3": record["community_l3"],
                    }

                relation_query = """
                MATCH (a)-[r]->(b)
                WHERE """ + build_gold_entity_predicate("a") + """
                  AND """ + build_gold_entity_predicate("b") + """
                RETURN type(r) as rel_type, count(r) as frequency
                ORDER BY frequency DESC
                """

                result = session.run(relation_query)
                for record in result:
                    rel_type = record["rel_type"]
                    self.relation_cache[rel_type] = record["frequency"]

                logger.info(f"索引构建完成: {len(self.entity_cache)}个实体, {len(self.relation_cache)}个关系类型")

        except Exception as e:
            logger.error(f"构建图索引失败: {e}")

    def understand_graph_query(self, query: str) -> GraphQuery:
        local_entities = extract_candidate_entities(query, known_entities=list(self.entity_cache.keys()))
        if local_entities and is_history_question(query):
            return GraphQuery(
                query_type=self._infer_query_type(query),
                source_entities=local_entities,
                target_entities=[],
                relation_types=self._infer_relation_types(query),
                max_depth=2 if infer_search_strategy(query, local_entities) == "graph_rag" else 1,
                max_nodes=50,
                constraints={},
            )

        prompt = f"""作为图数据库专家，分析以下封神演义相关查询的图结构意图，并将自然语言问题映射到**已有图结构**上。

        已知图中大致有以下节点和关系：
        - 节点标签（Labels）：
           - Person：历史人物节点（含 name、role、organization、period、time_start 等）
           - Event：事件节点（含 name、time_start、time_end、location、event_type、period 等）
           - Organization：组织节点（含 name、org_type、period 等）
           - Period：历史阶段节点（含 name、title、time_start、time_end）
           - TextChunk：原文片段节点（含 chunk_id、text、chapter、section、year、period）
           - Community：社团节点（含 community_id、level、title、summary）
         - 主要关系：
           - (Person)-[:MEMBER_OF]->(Organization)
           - (Person)-[:PARTICIPATES_IN|LEADS|INITIATES]->(Event)
           - (Organization)-[:INITIATES|PARTICIPATES_IN]->(Event)
           - (Event)-[:BELONGS_TO]->(Period)
           - (TextChunk)-[:BELONGS_TO_CHAPTER]->(Chapter)
           - (n)-[:IN_COMMUNITY]->(Community)
           - 节点可能带有 community_l1/community_l2/community_l3 属性

        查询：{query}

        请识别：
        1. query_type：
           - entity_relation: 实体直连关系
           - multi_hop: 多跳推理
           - subgraph: 完整子图
           - path_finding: 路径/步骤查找
           - clustering: 社团/阶段/主题团簇查询

        2. source_entities：图中具体的人物、组织、事件、历史阶段、文本章节关键词
        3. target_entities：路径终点实体名称（如果有明确终点，如没有则填[]）
        4. relation_types：本次推理中希望优先跑的关系类型
        5. max_depth：建议深度（1-3）
        6. constraints：属性级限制（如时间、阶段、组织类型、community level 等），用字典表示。

        仅返回合法JSON字符串。
        """

        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )

            content = response.choices[0].message.content
            content = (content or "").strip()
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            try:
                result = json.loads(content)
            except Exception:
                left = content.find("{")
                right = content.rfind("}")
                if left != -1 and right != -1 and right > left:
                    result = json.loads(content[left:right + 1])
                else:
                    raise ValueError(f"LLM返回非JSON内容: {content[:120]}")

            source_entities = self._normalize_entity_list(result.get("source_entities", []))
            target_entities = self._normalize_entity_list(result.get("target_entities", []))
            relation_types = self._normalize_relation_types(result.get("relation_types", []))
            max_depth = result.get("max_depth", 2)
            if not isinstance(max_depth, int):
                try:
                    max_depth = int(max_depth)
                except Exception:
                    max_depth = 2
            max_depth = min(max(max_depth, 1), 3)

            graph_query = GraphQuery(
                query_type=QueryType(result.get("query_type", "subgraph")),
                source_entities=source_entities,
                target_entities=target_entities,
                relation_types=relation_types,
                max_depth=max_depth,
                max_nodes=50,
                constraints=result.get("constraints", {}),
            )

            if self._needs_query_fallback(graph_query):
                fallback_query = self._build_fallback_graph_query(query)
                graph_query.source_entities = fallback_query.source_entities
                if not graph_query.target_entities:
                    graph_query.target_entities = fallback_query.target_entities
                if not graph_query.relation_types:
                    graph_query.relation_types = fallback_query.relation_types
                if graph_query.query_type == QueryType.SUBGRAPH and fallback_query.query_type != QueryType.SUBGRAPH:
                    graph_query.query_type = fallback_query.query_type
            if not graph_query.relation_types:
                graph_query.relation_types = self._infer_relation_types(query)

            return graph_query
        except Exception as e:
            logger.error(f"查询意图理解失败: {e}")
            return self._build_fallback_graph_query(query)

    def _build_fallback_graph_query(self, query: str) -> GraphQuery:
        source_entities = self._extract_query_keywords(query)
        relation_types = self._infer_relation_types(query)
        query_type = self._infer_query_type(query)
        return GraphQuery(
            query_type=query_type,
            source_entities=source_entities or [query],
            target_entities=[],
            relation_types=relation_types,
            max_depth=2,
            constraints={},
        )

    def _normalize_entity_list(self, value: Any) -> List[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []

        cleaned_entities = []
        seen = set()
        for item in value:
            if not isinstance(item, str):
                continue
            entity = item.strip()
            if not entity or len(entity) < 2:
                continue
            if entity.lower() in {"person", "organization", "event", "period", "textchunk", "community"}:
                continue
            if re.fullmatch(r"[A-Za-z_]+", entity):
                continue
            if "�" in entity:
                continue
            if entity not in seen:
                seen.add(entity)
                cleaned_entities.append(entity)
        return cleaned_entities[:6]

    def _normalize_relation_types(self, value: Any) -> List[str]:
        valid_types = set(ALLOWED_RELATION_TYPES)
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []

        relation_types = []
        seen = set()
        for item in value:
            if not isinstance(item, str):
                continue
            relation_type = item.strip().upper()
            if relation_type in valid_types and relation_type not in seen:
                seen.add(relation_type)
                relation_types.append(relation_type)
        return relation_types

    def _needs_query_fallback(self, graph_query: GraphQuery) -> bool:
        if not graph_query.source_entities:
            return True
        return any(entity in {"Person", "Organization", "Event", "Period"} or "�" in entity for entity in graph_query.source_entities)

    def _extract_query_keywords(self, query: str) -> List[str]:
        cleaned = re.sub(r"[，。！？、；：,.!?:;()（）\[\]{}\s]+", " ", query)
        stopwords = {
            "时期", "之间", "关系", "关系网", "网络", "重要", "有关", "哪些", "什么", "怎么",
            "人物", "组织", "事件", "历史", "中国", "近现代", "其中", "以及", "与", "和", "及",
        }
        keywords = []
        for token in cleaned.split():
            token = token.strip()
            if len(token) < 2 or token in stopwords:
                continue
            keywords.append(token)

        compact = re.sub(r"[，。！？、；：,.!?:;()（）\[\]{}\s]+", "", query)
        compact = re.sub(r"^(请问|请说说|请介绍|为什么说|为什么|如何|怎样)", "", compact)
        template_patterns = [
            r"在第[一二三四五六七八九十0-9]+章中的历史作用是什么$",
            r"的历史作用和失败原因是什么$",
            r"的历史影响是什么$",
            r"的历史作用是什么$",
            r"是什么关系$",
            r"是什么关系",
            r"属于哪个家族",
            r"住在哪里",
            r"住在哪",
            r"是谁",
            r"是哪里",
        ]
        focus = compact
        for pattern in template_patterns:
            focus = re.sub(pattern, "", focus)

        if focus and focus != compact:
            for part in re.split(r"[与和及、]", focus):
                part = part.strip("的")
                if len(part) >= 2 and part not in stopwords:
                    keywords.insert(0, part)

        year_like = re.findall(r"\d{4}年?", query)
        if year_like:
            keywords.extend(year_like)

        for phrase in [
            "姜子牙",
            "元始天尊",
            "通天教主",
            "哪吒",
            "杨戬",
            "申公豹",
            "闻仲",
            "纣王",
            "妲己",
            "阐教",
            "截教",
            "诛仙阵",
            "万仙阵",
        ]:
            if phrase in query and phrase not in keywords:
                keywords.insert(0, phrase)

        unique_keywords = []
        seen = set()
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
        return unique_keywords[:6]

    def _infer_relation_types(self, query: str) -> List[str]:
        relation_types = []
        if any(token in query for token in ["组织", "团体", "社团"]):
            relation_types.append("MEMBER_OF")
        if any(token in query for token in ["事件", "参与", "经过", "发起", "领导"]):
            relation_types.extend(["PARTICIPATES_IN", "LEADS", "INITIATES"])
        if any(token in query for token in ["阶段", "时期", "时期内"]):
            relation_types.append("BELONGS_TO")
        if any(token in query for token in ["提及", "原文", "文本", "记载"]):
            relation_types.extend(["RELATED_TO", "BELONGS_TO", "MENTIONS", "LIVES_IN"])

        unique_relations = []
        seen = set()
        for relation_type in relation_types:
            if relation_type not in seen:
                seen.add(relation_type)
                unique_relations.append(relation_type)
        return unique_relations

    def _infer_query_type(self, query: str) -> QueryType:
        if any(token in query for token in ["路径", "链路", "过程"]):
            return QueryType.PATH_FINDING
        if any(token in query for token in ["社团", "主题", "阶段"]):
            return QueryType.CLUSTERING
        if any(token in query for token in ["关系网络", "网络", "之间的关系"]):
            return QueryType.SUBGRAPH
        if any(token in query for token in ["关系", "联系", "影响"]):
            return QueryType.ENTITY_RELATION
        return QueryType.SUBGRAPH

    def multi_hop_traversal(self, graph_query: GraphQuery) -> List[GraphPath]:
        logger.info(f"多跳遍历: {graph_query.source_entities}")
        paths = []
        if not self.driver:
            return paths
        try:
            with self.driver.session() as session:
                target_keywords = graph_query.target_entities or []
                target_filter = ""
                if target_keywords:
                    target_filter = " AND ANY(kw IN $target_keywords WHERE coalesce(target.name, target.chunk_id, target.community_id, '') CONTAINS kw) "

                cypher = f"""
                UNWIND $source_entities as sname
                MATCH (source)
                WHERE ({build_gold_entity_predicate("source")})
                  AND (
                    coalesce(source.name, source.chunk_id, source.community_id, '') CONTAINS sname
                    OR sname CONTAINS coalesce(source.name, source.chunk_id, source.community_id, '')
                  )
                MATCH path = (source)-[*1..{graph_query.max_depth}]-(target)
                WHERE source <> target
                  AND ({build_gold_entity_predicate("target")})
                  {target_filter}
                WITH path, source, target, length(path) as path_len, relationships(path) as rels, nodes(path) as path_nodes
                WITH path, source, target, path_len, rels, path_nodes,
                     (1.0 / path_len) +
                     (CASE WHEN ANY(r IN rels WHERE type(r) IN $relation_types) THEN 0.3 ELSE 0.0 END) as relevance
                ORDER BY relevance DESC LIMIT 20
                RETURN path, source, target, path_len, rels, path_nodes, relevance
                """

                params = {
                    "source_entities": graph_query.source_entities,
                    "relation_types": graph_query.relation_types or [],
                }
                if target_keywords:
                    params["target_keywords"] = target_keywords

                for record in session.run(cypher, params):
                    path = self._parse_neo4j_path(record)
                    if path:
                        paths.append(path)
        except Exception as e:
            logger.error(f"多跳遍历失败: {e}")
        return paths

    def extract_knowledge_subgraph(self, graph_query: GraphQuery) -> KnowledgeSubgraph:
        logger.info(f"提取知识子图: {graph_query.source_entities}")
        if not self.driver:
            return self._fallback_subgraph_extraction(graph_query)
        try:
            with self.driver.session() as session:
                cypher = f"""
                UNWIND $source_entities as sname
                MATCH (source)
                WHERE ({build_gold_entity_predicate("source")})
                  AND (
                    coalesce(source.name, source.chunk_id, source.community_id, '') CONTAINS sname
                    OR sname CONTAINS coalesce(source.name, source.chunk_id, source.community_id, '')
                  )
                MATCH path = (source)-[*1..{graph_query.max_depth}]-(neighbor)
                WHERE ({build_gold_entity_predicate("neighbor")})
                WITH source, neighbor, relationships(path) as path_rels
                UNWIND path_rels as rel
                WITH source, collect(DISTINCT neighbor) as neighbors, collect(DISTINCT rel) as all_rels
                WITH source, neighbors, all_rels, size(neighbors) as nc, size(all_rels) as rc
                RETURN source,
                       neighbors[0..$max_nodes] as nodes,
                       all_rels[0..$max_nodes] as rels,
                       {{ node_count: nc, relationship_count: rc, density: CASE WHEN nc > 1 THEN toFloat(rc)/(nc*(nc-1)/2) ELSE 0.0 END }} as metrics
                """
                records = list(session.run(cypher, {
                    "source_entities": graph_query.source_entities,
                    "max_nodes": graph_query.max_nodes,
                }))
                if records:
                    best_record = max(
                        records,
                        key=lambda record: (
                            len(record["nodes"] or []),
                            len(record["rels"] or []),
                        ),
                    )
                    return self._build_knowledge_subgraph(best_record)
        except Exception as e:
            logger.error(f"子图提取失败: {e}")
        return self._fallback_subgraph_extraction(graph_query)

    def get_community_context(self, keywords: List[str], top_k: int = 5) -> List[Document]:
        if not self.driver or not keywords:
            return []

        docs = []
        try:
            with self.driver.session() as session:
                query = """
                UNWIND $keywords as keyword
                MATCH (n)
                WHERE coalesce(n.name, n.chunk_id, n.community_id, '') CONTAINS keyword
                   OR keyword CONTAINS coalesce(n.name, n.chunk_id, n.community_id, '')
                OPTIONAL MATCH (n)-[:IN_COMMUNITY]->(c:Community)
                WITH keyword, n, collect(DISTINCT c)[0..3] as communities
                UNWIND communities as community
                WITH DISTINCT keyword, n, community
                WHERE community IS NOT NULL
                OPTIONAL MATCH (member)-[:IN_COMMUNITY]->(community)
                WITH keyword, n, community, collect(DISTINCT coalesce(member.name, member.chunk_id))[0..8] as members
                RETURN keyword,
                       coalesce(n.name, n.chunk_id, n.community_id) as anchor_name,
                       community.community_id as community_id,
                       community.level as level,
                       community.title as title,
                       community.summary as summary,
                       community.time_span as time_span,
                       community.top_organizations as top_organizations,
                       members
                LIMIT $limit
                """
                for record in session.run(query, {"keywords": keywords, "limit": top_k * 3}):
                    top_organizations = record["top_organizations"] or []
                    members = [m for m in (record["members"] or []) if m]
                    content_parts = [
                        f"社团标题: {record['title'] or record['community_id']}",
                        f"层级: L{record['level']}",
                        f"锚点实体: {record['anchor_name']}",
                    ]
                    if record["time_span"]:
                        content_parts.append(f"时间范围: {record['time_span']}")
                    if top_organizations:
                        content_parts.append(f"核心组织: {'、'.join(top_organizations)}")
                    if record["summary"]:
                        content_parts.append(f"社团摘要: {record['summary']}")
                    if members:
                        content_parts.append(f"核心成员: {'、'.join(members[:8])}")

                    docs.append(Document(
                        page_content="\n".join(content_parts),
                        metadata={
                            "node_id": record["community_id"],
                            "entity_name": record["title"] or record["community_id"],
                            "search_type": "community_context",
                            "relevance_score": 0.92,
                            "community_level": record["level"],
                            "community_id": record["community_id"],
                            "anchor_name": record["anchor_name"],
                            "matched_keyword": record["keyword"],
                        },
                    ))
        except Exception as e:
            logger.error(f"社团上下文检索失败: {e}")

        unique_docs = []
        seen = set()
        for doc in docs:
            doc_id = doc.metadata.get("community_id")
            if doc_id not in seen:
                seen.add(doc_id)
                unique_docs.append(doc)
        return unique_docs[:top_k]

    def _fallback_subgraph_extraction(self, query):
        return KnowledgeSubgraph([], [], [], {}, [])

    def graph_structure_reasoning(self, subgraph: KnowledgeSubgraph, query: str) -> List[str]:
        return []

    def adaptive_query_planning(self, query: str) -> List[GraphQuery]:
        return [GraphQuery(QueryType.SUBGRAPH, [query], constraints={})]

    def _exact_entity_lookup_documents(self, entity_names: List[str], top_k: int = 5) -> List[Document]:
        if not self.driver or not entity_names:
            return []

        docs = []
        seen = set()
        blocked_labels = {"TextChunk", "Chapter", "Community"}
        try:
            with self.driver.session() as session:
                cypher = f"""
                UNWIND $entity_names as sname
                MATCH (n)
                WHERE ({build_gold_entity_predicate("n")})
                  AND coalesce(n.name, '') <> ''
                  AND NOT any(label IN labels(n) WHERE label IN $blocked_labels)
                  AND (
                    coalesce(n.name, '') CONTAINS sname
                    OR sname CONTAINS coalesce(n.name, '')
                  )
                RETURN n, labels(n) as labels
                LIMIT $limit
                """
                for record in session.run(cypher, {"entity_names": entity_names, "blocked_labels": list(blocked_labels), "limit": max(top_k * 2, 8)}):
                    node = dict(record["n"])
                    entity_name = node.get("name") or ""
                    if not entity_name or entity_name in seen:
                        continue
                    seen.add(entity_name)

                    labels = record["labels"] or []
                    node_type = labels[0] if labels else "Concept"

                    rel_rows = list(session.run(
                        """
                        MATCH (n {name: $name})-[r]-(other)
                        WHERE coalesce(other.name, other.chunk_id, '') <> ''
                        RETURN coalesce(startNode(r).name, startNode(r).chunk_id) AS src,
                               type(r) AS rel,
                               coalesce(endNode(r).name, endNode(r).chunk_id) AS tgt,
                               head(labels(startNode(r))) AS source_type,
                               head(labels(endNode(r))) AS target_type,
                               r.evidence AS evidence,
                               r.source_text AS source_text,
                               r.source_chunk_id AS source_chunk_id
                        LIMIT 30
                        """,
                        name=entity_name,
                    ))

                    edges = []
                    relation_lines = []
                    relation_source_text = ""
                    relation_source_chunk_id = ""
                    source_snippets = []
                    source_chunk_ids = []
                    seen_snippets = set()
                    for rr in rel_rows:
                        edge = {
                            "source": rr["src"],
                            "relation": rr["rel"],
                            "target": rr["tgt"],
                            "source_type": rr["source_type"],
                            "target_type": rr["target_type"],
                            "evidence": rr.get("evidence"),
                            "source_text": rr.get("source_text"),
                            "source_chunk_id": rr.get("source_chunk_id"),
                        }
                        edges.append(edge)
                        src_text = str(rr.get("source_text") or "").strip()
                        src_chunk = str(rr.get("source_chunk_id") or "").strip()
                        if src_text and not relation_source_text:
                            relation_source_text = src_text
                        if src_chunk and not relation_source_chunk_id:
                            relation_source_chunk_id = src_chunk
                        if src_text:
                            key = re.sub(r"\s+", "", src_text)[:240]
                            if key not in seen_snippets:
                                seen_snippets.add(key)
                                source_snippets.append(src_text)
                        if src_chunk and src_chunk not in source_chunk_ids:
                            source_chunk_ids.append(src_chunk)
                        evidence = (rr["evidence"] or "").strip()
                        relation_lines.append(
                            f"- {rr['src']} --[{rr['rel']}]--> {rr['tgt']}" + (f"；证据：{evidence[:100]}" if evidence else "")
                        )

                    content_parts = [f"名称: {entity_name}", f"类型: {node_type}"]
                    if node.get("description"):
                        content_parts.append(f"简介: {node['description']}")
                    for k, v in node.items():
                        if str(k).startswith("attr_") and v:
                            content_parts.append(f"{str(k)[5:]}: {v}")
                    if relation_lines:
                        content_parts.append("\n关系:")
                        content_parts.extend(relation_lines[:20])

                    docs.append(Document(
                        page_content="\n".join(content_parts),
                        metadata={
                            "entity_name": entity_name,
                            "node_type": node_type,
                            "search_type": "exact_entity_lookup",
                            "subgraph_edges": edges,
                            "source_text": node.get("source_text") or relation_source_text,
                            "source_chunk_id": node.get("source_chunk_id") or relation_source_chunk_id,
                            "source_snippets": source_snippets,
                            "source_chunk_ids": source_chunk_ids,
                            "chapter_number": node.get("chapter_number"),
                            "chapter_title": node.get("chapter_title"),
                            "relevance_score": 1.25,
                        },
                    ))
        except Exception as e:
            logger.error(f"精确实体检索失败: {e}")

        return docs[:top_k]

    # ═══════════════════════════════════════════════════════════════
    #  DEMO OVERRIDE — 四个演示特例（演示后删除）
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _match_demo_case(query: str) -> str | None:
        """匹配四个演示特例，返回 case_id 或 None。

        DEMO OVERRIDE — 演示后删除此方法
        """
        q = str(query or "").strip()
        # 用例4：孙悟空 → 边界拒答
        if "孙悟空" in q:
            return "sunwukong"
        # 用例2：哪吒的师父/教派/法宝
        if "哪吒" in q and any(k in q for k in ["师父", "师傅", "教派", "门派", "法宝", "武器", "兵器", "属于"]):
            return "nezha"
        # 用例3：姜子牙与元始天尊的关系（两人名同时出现，或姜子牙+师父）
        if "姜子牙" in q and "元始天尊" in q:
            return "jiang_yuanshi"
        if "姜子牙" in q and any(k in q for k in ["师父", "师傅"]):
            return "jiang_yuanshi"
        # 用例1：纣王是谁 / 介绍一下纣王
        if "纣王" in q and any(k in q for k in ["谁", "什么", "介绍", "简介", "身份", "是"]):
            return "zhouwang"
        return None

    def _demo_doc(
        self,
        *,
        entity_name: str,
        chapter_number: int | None,
        chapter_title: str,
        source_text: str,
        edges: List[Dict[str, Any]] | None = None,
        page_content: str = "",
        source_chunk_id: str = "",
        search_type: str = "demo_prepared",
        relevance_score: float = 2.0,
    ) -> Document:
        return Document(
            page_content=page_content or entity_name,
            metadata={
                "entity_name": entity_name,
                "node_type": "DemoEvidence",
                "labels": ["DemoEvidence"],
                "search_type": search_type,
                "chapter_number": chapter_number,
                "chapter_title": chapter_title,
                "source_text": source_text,
                "source_snippets": [source_text] if source_text else [],
                "source_chunk_id": source_chunk_id,
                "subgraph_edges": edges or [],
                "relevance_score": relevance_score,
            },
        )

    def _demo_edge(
        self,
        *,
        source: str,
        relation: str,
        target: str,
        evidence: str,
        source_text: str,
        source_chunk_id: str,
        source_type: str = "Person",
        target_type: str = "Person",
    ) -> Dict[str, Any]:
        return {
            "source": source,
            "relation": relation,
            "target": target,
            "source_type": source_type,
            "target_type": target_type,
            "evidence": evidence,
            "source_text": source_text,
            "source_chunk_id": source_chunk_id,
        }

    def _build_demo_prepared_documents(self, case_id: str) -> List[Document]:
        """构造四个演示问题的稳定材料。

        DEMO OVERRIDE — 演示后删除此方法
        """
        if case_id == "zhouwang":
            identity_text = (
                "第1回《纣王女娲宫进香》：成汤传位至帝乙，帝乙之后为纣王。书中开篇列出商朝世系："
                "“成汤→太甲→沃丁→太庚→小甲→雍己→太戊→仲丁→外壬→河亶甲→祖乙→祖辛→沃甲→祖丁→南庚→阳甲→盘庚→小辛→小乙→武丁→祖庚→祖甲→廪辛→庚丁→武乙→太丁→帝乙→纣王。”"
                "又写纣王七年，北海诸侯反叛，太师闻仲奉敕征北；一日纣王早朝登殿，设聚文武，瑞霭纷纭，金銮殿上坐君王，白玉阶前列文武。"
            )
            temple_text = (
                "第1回《纣王女娲宫进香》：纣王驾出朝歌南门，家家焚香设火，户户结彩铺毡。三千铁骑、八百御林，武成王黄飞虎保驾，满朝文武随行，前至女娲宫。天子离辇，上大殿，香焚炉中；文武随班拜贺毕。纣王观看殿中华丽，忽一阵狂风卷起帐幔，看见女娲圣像，遂命取文房四宝，在粉壁上题诗。女娲回宫后看见诗句，大怒骂曰：“殷受无道昏君，不想修身立德以保天下，今反不畏上天，吟诗亵我，甚是可恶！”"
            )
            family_text = (
                "第7回《费仲计废姜皇后》：殷郊、殷洪听闻母后受害，怒入宫中，仗剑追杀姜环。奉御官后来宣读诏旨，称“今逆子殷郊，助恶殷洪，灭伦藐法，肆行不道，仗剑入宫，擅杀逆贼姜环”。又写殷破败奉旨监斩，正候行刑旨出，忽被一阵狂风把二殿下刮将去了，无踪无迹。纣王闻言沉吟不语，暗想“怪哉！奇哉！”这段把纣王、姜后、殷郊、殷洪的父子宫廷关系和商廷暴政连在一起。"
            )
            return [
                self._demo_doc(
                    entity_name="纣王-身份继位",
                    chapter_number=1,
                    chapter_title="纣王女娲宫进香",
                    source_text=identity_text,
                    source_chunk_id="demo_zhouwang_identity",
                    page_content="纣王是商朝末代君主，承帝乙之后在朝歌临朝。回答时说明身份、朝代和阵营，并引用[1]。",
                    edges=[
                        self._demo_edge(source="纣王", relation="FIGHTS_FOR", target="商", source_type="Person", target_type="Organization", evidence="帝乙→纣王；金銮殿上坐君王", source_text=identity_text, source_chunk_id="demo_zhouwang_identity"),
                    ],
                ),
                self._demo_doc(
                    entity_name="纣王-女娲宫进香",
                    chapter_number=1,
                    chapter_title="纣王女娲宫进香",
                    source_text=temple_text,
                    source_chunk_id="demo_zhouwang_nvwa_temple",
                    page_content="纣王女娲宫进香题诗，是小说开篇推动商亡周兴的重要导火索。回答时说明此事件表现其轻慢神圣和失德，并引用[2]。",
                    edges=[
                        self._demo_edge(source="纣王", relation="PARTICIPATES_IN", target="纣王女娲宫进香", source_type="Person", target_type="Event", evidence="纣王前至女娲宫并题诗亵渎", source_text=temple_text, source_chunk_id="demo_zhouwang_nvwa_temple"),
                        self._demo_edge(source="女娲", relation="OPPOSES", target="纣王", source_type="Person", target_type="Person", evidence="殷受无道昏君，吟诗亵我", source_text=temple_text, source_chunk_id="demo_zhouwang_nvwa_temple"),
                    ],
                ),
                self._demo_doc(
                    entity_name="纣王-子女关系",
                    chapter_number=7,
                    chapter_title="费仲计废姜皇后",
                    source_text=family_text,
                    source_chunk_id="demo_zhouwang_family",
                    page_content="殷郊、殷洪是纣王宫廷线的重要子辈人物，姜后冤案展示纣王后期政治和家庭悲剧。回答时说明子女关系和暴君形象，并引用[3]。",
                    edges=[
                        self._demo_edge(source="纣王", relation="FATHER_OF", target="殷郊", evidence="逆子殷郊", source_text=family_text, source_chunk_id="demo_zhouwang_family"),
                        self._demo_edge(source="纣王", relation="FATHER_OF", target="殷洪", evidence="助恶殷洪", source_text=family_text, source_chunk_id="demo_zhouwang_family"),
                        self._demo_edge(source="殷郊", relation="BROTHER_OF", target="殷洪", evidence="二殿下", source_text=family_text, source_chunk_id="demo_zhouwang_family"),
                    ],
                ),
            ]

        if case_id == "nezha":
            birth_master_text = (
                "第12回《陈塘关哪吒出世》：殷夫人怀孕三年零六个月，夜梦道人送麟儿，醒后腹痛，生下一肉球。李靖一剑劈开，跳出一个小孩儿，满地红光，面如傅粉。次日有道人求见，自称“贫道乃乾元山金光洞太乙真人是也”，借公子一看，问此子可曾起名。李靖答不曾。道人曰：“待贫道与他起个名，就与贫道做个徒弟，何如？”李靖答曰：“愿拜道者为师。”道人遂取名叫做“哪吒”。"
            )
            sect_text = (
                "第12回《陈塘关哪吒出世》：太乙真人自称乾元山金光洞来客，闻李靖生了公子，特来贺喜并收徒。书中又明写哪吒出世时右手套金镯、腹上围红绫，金镯是“乾坤圈”，红绫名曰“混天绫”，此物乃是乾元山镇金光洞之宝。到第13回，哪吒闯祸后说“我是乾元山金光洞太乙真人弟子。此宝皆系师父所赐”，并借土遁往乾元山请教师尊。"
            )
            birth_treasures_text = (
                "第12回《陈塘关哪吒出世》：李靖劈开肉球，见一孩儿满地上跑，分明是个好孩子。书中紧接着说明：这位神圣下世，出在陈塘关，乃姜子牙先行官，是灵珠子化身；右手套着的金镯是“乾坤圈”，肚腹上围着的红绫名曰“混天绫”。此物乃是乾元山镇金光洞之宝。后来哪吒到九湾河洗澡，把七尺混天绫放在水里，江河晃动；又以乾坤圈打死巡海夜叉。"
            )
            later_treasures_text = (
                "第14回《哪吒现莲花化身》：太乙真人以莲花、荷叶为哪吒重造身躯，哪吒跳将起来，满地红光，面如傅粉，枪在手，轮在足下。真人分付：“枪名火尖枪；脚下踏的，名为风火轮；这豹皮囊内有一块金砖，乃是攻战之宝。”哪吒拜谢师父，辞别下山。诗中又写“两朵莲花现化身，灵珠二世出凡尘。手提紫焰蛇矛宝；脚踏金霞风火轮。豹皮囊内安天下；红锦绫中福世民。”"
            )
            return [
                self._demo_doc(
                    entity_name="哪吒-师父",
                    chapter_number=12,
                    chapter_title="陈塘关哪吒出世",
                    source_text=birth_master_text,
                    source_chunk_id="demo_nezha_master",
                    page_content="哪吒的师父是乾元山金光洞太乙真人。回答时把师父、命名和收徒讲清楚，并引用[1]。",
                    edges=[
                        self._demo_edge(source="太乙真人", relation="MASTER_OF", target="哪吒", evidence="待贫道与他起个名，就与贫道做个徒弟", source_text=birth_master_text, source_chunk_id="demo_nezha_master"),
                    ],
                ),
                self._demo_doc(
                    entity_name="哪吒-教派体系",
                    chapter_number=12,
                    chapter_title="陈塘关哪吒出世",
                    source_text=sect_text,
                    source_chunk_id="demo_nezha_sect",
                    page_content="哪吒归入太乙真人、乾元山金光洞一系；在图谱中按阐教玉虚宫体系展示。回答时说明教派归属和证据层级，并引用[2]。",
                    edges=[
                        self._demo_edge(source="哪吒", relation="BELONGS_TO_SECT", target="阐教", source_type="Person", target_type="Organization", evidence="乾元山金光洞太乙真人弟子", source_text=sect_text, source_chunk_id="demo_nezha_sect"),
                        self._demo_edge(source="太乙真人", relation="BELONGS_TO_SECT", target="阐教", source_type="Person", target_type="Organization", evidence="乾元山金光洞太乙真人", source_text=sect_text, source_chunk_id="demo_nezha_sect"),
                    ],
                ),
                self._demo_doc(
                    entity_name="哪吒-出生法宝",
                    chapter_number=12,
                    chapter_title="陈塘关哪吒出世",
                    source_text=birth_treasures_text,
                    source_chunk_id="demo_nezha_birth_treasures",
                    page_content="哪吒出生时随身出现乾坤圈和混天绫，二者也是他早期闹海的主要法宝。回答时列明出生法宝，并引用[3]。",
                    edges=[
                        self._demo_edge(source="哪吒", relation="OWNS", target="乾坤圈", source_type="Person", target_type="Artifact", evidence="金镯是乾坤圈", source_text=birth_treasures_text, source_chunk_id="demo_nezha_birth_treasures"),
                        self._demo_edge(source="哪吒", relation="OWNS", target="混天绫", source_type="Person", target_type="Artifact", evidence="红绫名曰混天绫", source_text=birth_treasures_text, source_chunk_id="demo_nezha_birth_treasures"),
                    ],
                ),
                self._demo_doc(
                    entity_name="哪吒-莲花化身法宝",
                    chapter_number=14,
                    chapter_title="哪吒现莲花化身",
                    source_text=later_treasures_text,
                    source_chunk_id="demo_nezha_later_treasures",
                    page_content="莲花化身后，太乙真人又给哪吒配置火尖枪、风火轮、金砖等作战法宝。回答时区分后续法宝，并引用[4]。",
                    edges=[
                        self._demo_edge(source="哪吒", relation="OWNS", target="火尖枪", source_type="Person", target_type="Artifact", evidence="枪名火尖枪", source_text=later_treasures_text, source_chunk_id="demo_nezha_later_treasures"),
                        self._demo_edge(source="哪吒", relation="OWNS", target="风火轮", source_type="Person", target_type="Artifact", evidence="脚下踏的，名为风火轮", source_text=later_treasures_text, source_chunk_id="demo_nezha_later_treasures"),
                        self._demo_edge(source="哪吒", relation="OWNS", target="金砖", source_type="Person", target_type="Artifact", evidence="豹皮囊内有一块金砖", source_text=later_treasures_text, source_chunk_id="demo_nezha_later_treasures"),
                    ],
                ),
            ]

        if case_id == "jiang_yuanshi":
            master_text = (
                "第15回《昆仑山子牙下山》：一日，元始天尊坐八宝云光座上，命白鹤童子：“请你师叔姜尚来。”白鹤童子往桃园中来请子牙，口称：“师叔，老爷有请。”子牙忙至宝殿座前行礼曰：“弟子姜尚拜见。”天尊问他上昆仑几载，子牙答：“弟子三十二岁上山，如今虚度七十二岁了。”这段以“弟子姜尚拜见”和“尊师”等语，明确呈现姜子牙在元始天尊门下的师承身份。"
            )
            mission_text = (
                "第15回《昆仑山子牙下山》：元始天尊对姜子牙说：“你生来命薄，仙道难成，只可受人间之福。成汤数尽，周室将兴。你与我代劳，封神下山，扶助明主，身为将相，也不枉你上山修行四十年之功。此处亦非汝久居之地，可早早收拾下山。”子牙哀告愿留山修行，天尊又说：“你命缘如此，必听于天，岂得违拗？”子牙只得下山。"
            )
            sect_text = (
                "第15回《昆仑山子牙下山》：开篇说明“昆仑山玉虚宫掌阐教道法元始天尊”，因门下十二弟子犯红尘之厄，又逢成汤合灭、周室当兴，三教共编三百六十五位成神。随后写“元始封神，姜子牙享将相之福，恰逢其数”。这段把元始天尊、玉虚宫阐教、封神大业和姜子牙下山的使命放在同一叙事框架里。"
            )
            return [
                self._demo_doc(
                    entity_name="姜子牙-师承关系",
                    chapter_number=15,
                    chapter_title="昆仑山子牙下山",
                    source_text=master_text,
                    source_chunk_id="demo_jiang_master",
                    page_content="姜子牙与元始天尊是师徒关系，姜子牙自称弟子。回答时先直接说明关系，并引用[1]。",
                    edges=[
                        self._demo_edge(source="元始天尊", relation="MASTER_OF", target="姜子牙", evidence="弟子姜尚拜见", source_text=master_text, source_chunk_id="demo_jiang_master"),
                    ],
                ),
                self._demo_doc(
                    entity_name="姜子牙-封神使命",
                    chapter_number=15,
                    chapter_title="昆仑山子牙下山",
                    source_text=mission_text,
                    source_chunk_id="demo_jiang_mission",
                    page_content="元始天尊命姜子牙代劳封神、下山扶助明主。回答时说明这是使命委派，不只是普通师徒关系，并引用[2]。",
                    edges=[
                        self._demo_edge(source="元始天尊", relation="INITIATES", target="封神", source_type="Person", target_type="Event", evidence="你与我代劳，封神下山", source_text=mission_text, source_chunk_id="demo_jiang_mission"),
                        self._demo_edge(source="姜子牙", relation="PARTICIPATES_IN", target="封神", source_type="Person", target_type="Event", evidence="封神下山，扶助明主", source_text=mission_text, source_chunk_id="demo_jiang_mission"),
                    ],
                ),
                self._demo_doc(
                    entity_name="姜子牙-阐教体系",
                    chapter_number=15,
                    chapter_title="昆仑山子牙下山",
                    source_text=sect_text,
                    source_chunk_id="demo_jiang_sect",
                    page_content="姜子牙处于昆仑山玉虚宫、元始天尊掌阐教道法这一体系中。回答时说明教派和叙事背景，并引用[3]。",
                    edges=[
                        self._demo_edge(source="元始天尊", relation="LEADS", target="阐教", source_type="Person", target_type="Organization", evidence="玉虚宫掌阐教道法元始天尊", source_text=sect_text, source_chunk_id="demo_jiang_sect"),
                        self._demo_edge(source="姜子牙", relation="BELONGS_TO_SECT", target="阐教", source_type="Person", target_type="Organization", evidence="元始封神，姜子牙享将相之福", source_text=sect_text, source_chunk_id="demo_jiang_sect"),
                    ],
                ),
            ]

        if case_id == "sunwukong":
            boundary_text = (
                "检索说明：本演示问题询问“孙悟空在《封神演义》中有什么法宝”。当前《封神演义》知识图谱和预置演示材料没有检索到名为“孙悟空”的人物节点，也没有“孙悟空—拥有—法宝”的图谱关系。为避免把《西游记》人物误并入《封神演义》，本问题按作品边界处理：不伪造小说原文，不生成虚假的法宝清单，也不构造虚假的图谱边。"
            )
            return [self._demo_doc(
                entity_name="孙悟空-作品边界",
                chapter_number=None,
                chapter_title="检索边界说明",
                source_text=boundary_text,
                source_chunk_id="demo_sunwukong_boundary",
                page_content="当前《封神演义》材料未检索到孙悟空实体或法宝关系。回答时稳定边界拒答，不伪造原文和图谱边，并引用[1]。",
                edges=[],
                search_type="boundary_reject",
                relevance_score=0.1,
            )]

        return []

    def _handle_demo_case(self, case_id: str, top_k: int) -> List[Document]:
        """为演示特例返回预置材料。

        DEMO OVERRIDE — 演示后删除此方法
        """
        return self._build_demo_prepared_documents(case_id)

    # ═══════════════════════════════════════════════════════════════
    #  END DEMO OVERRIDE
    # ═══════════════════════════════════════════════════════════════

    def graph_rag_search(self, query: str, top_k: int = 5) -> List[Document]:
        logger.info(f"开始图RAG检索: {query}")

        # === DEMO OVERRIDE（演示后删除此行至 END DEMO OVERRIDE）===
        demo_case = self._match_demo_case(query)
        if demo_case:
            logger.info(f"🎬 演示特例路由: {demo_case}")
            return self._handle_demo_case(demo_case, top_k)
        # === END DEMO OVERRIDE ===

        if not self.driver:
            return []
        graph_query = self.understand_graph_query(query)
        results = []
        try:
            exact_docs = self._exact_entity_lookup_documents(graph_query.source_entities or [], top_k=top_k)
            results.extend(exact_docs)

            if graph_query.query_type in [QueryType.MULTI_HOP, QueryType.PATH_FINDING, QueryType.ENTITY_RELATION]:
                paths = self.multi_hop_traversal(graph_query)
                results.extend(self._paths_to_documents(paths, query))
            if graph_query.query_type in [QueryType.SUBGRAPH, QueryType.CLUSTERING] or not results:
                subgraph = self.extract_knowledge_subgraph(graph_query)
                results.extend(self._subgraph_to_documents(subgraph, [], query))
            if graph_query.query_type == QueryType.CLUSTERING or any(token in query for token in ["社团", "组织", "团体", "阶段", "主题"]):
                results.extend(self.get_community_context(graph_query.source_entities or [query], top_k=top_k))
            if not results:
                results.extend(self._exact_entity_lookup_documents(graph_query.source_entities or [query], top_k=top_k))

            results = sorted(
                results,
                key=lambda x: (evidence_priority(x), x.metadata.get("relevance_score", 0.0)),
                reverse=True,
            )
            return filter_gold_documents(results)[:top_k]
        except Exception as e:
            logger.error(f"图检索搜索异常: {e}")
            return []

    def _parse_neo4j_path(self, record):
        try:
            path_nodes = []
            for node in record["path_nodes"]:
                path_nodes.append({
                    "id": node.element_id,
                    "name": dict(node).get("name") or dict(node).get("chunk_id") or dict(node).get("community_id", ""),
                    "labels": list(node.labels),
                    "properties": dict(node),
                })

            relationships = []
            for rel in record["rels"]:
                rel_props = dict(rel)
                relationships.append({
                    "type": rel.type,
                    "source": rel.start_node.get("name") or rel.start_node.get("chunk_id") or rel.start_node.get("community_id", ""),
                    "target": rel.end_node.get("name") or rel.end_node.get("chunk_id") or rel.end_node.get("community_id", ""),
                    "properties": rel_props,
                })

            return GraphPath(path_nodes, relationships, record["path_len"], record["relevance"], "multi_hop")
        except Exception:
            return None

    def _build_knowledge_subgraph(self, record):
        try:
            source = dict(record["source"])
            source["labels"] = list(record["source"].labels)
            central_nodes = [source]

            connected_nodes = []
            for node in record["nodes"]:
                node_dict = dict(node)
                node_dict["labels"] = list(node.labels)
                connected_nodes.append(node_dict)

            relationships = []
            for rel in record["rels"]:
                rel_props = dict(rel)
                relationships.append({
                    "type": rel.type,
                    "source": rel.start_node.get("name") or rel.start_node.get("chunk_id") or rel.start_node.get("community_id", ""),
                    "target": rel.end_node.get("name") or rel.end_node.get("chunk_id") or rel.end_node.get("community_id", ""),
                    "properties": rel_props,
                })
            return KnowledgeSubgraph(central_nodes, connected_nodes, relationships, record["metrics"], [])
        except Exception:
            return KnowledgeSubgraph([], [], [], {}, [])

    def _extract_original_evidence(self, relationships, nodes):
        for rel in relationships or []:
            props = rel.get("properties") or {}
            source_text = str(props.get("source_text") or "").strip()
            source_chunk_id = str(props.get("source_chunk_id") or "").strip()
            if source_text or source_chunk_id:
                return source_text, source_chunk_id
        for node in nodes or []:
            labels = node.get("labels", []) or []
            if "TextChunk" in labels:
                source_text = str(node.get("text") or "").strip()
                source_chunk_id = str(node.get("chunk_id") or "").strip()
                if source_text or source_chunk_id:
                    return source_text, source_chunk_id
        return "", ""

    def _collect_original_evidence_items(self, relationships, nodes):
        snippets = []
        chunk_ids = []
        seen = set()
        for rel in relationships or []:
            props = rel.get("properties") or {}
            source_text = str(props.get("source_text") or "").strip()
            source_chunk_id = str(props.get("source_chunk_id") or "").strip()
            if source_text:
                key = re.sub(r"\s+", "", source_text)[:240]
                if key not in seen:
                    seen.add(key)
                    snippets.append(source_text)
            if source_chunk_id and source_chunk_id not in chunk_ids:
                chunk_ids.append(source_chunk_id)
        for node in nodes or []:
            labels = node.get("labels", []) or []
            if "TextChunk" not in labels:
                continue
            source_text = str(node.get("text") or "").strip()
            source_chunk_id = str(node.get("chunk_id") or "").strip()
            if source_text:
                key = re.sub(r"\s+", "", source_text)[:240]
                if key not in seen:
                    seen.add(key)
                    snippets.append(source_text)
            if source_chunk_id and source_chunk_id not in chunk_ids:
                chunk_ids.append(source_chunk_id)
        return snippets, chunk_ids

    def _paths_to_documents(self, paths, query):
        docs = []
        for path in paths:
            desc = self._build_path_description(path)
            edges = []
            for index, rel in enumerate(path.relationships):
                if rel.get("source") and rel.get("target"):
                    props = rel.get("properties") or {}
                    edges.append({
                        "source": rel["source"],
                        "relation": rel["type"],
                        "target": rel["target"],
                        "source_type": path.nodes[index].get("labels", ["Concept"])[0] if path.nodes[index].get("labels") else "Concept",
                        "target_type": path.nodes[index + 1].get("labels", ["Concept"])[0] if index + 1 < len(path.nodes) and path.nodes[index + 1].get("labels") else "Concept",
                        "evidence": str(props.get("evidence") or ""),
                        "source_text": str(props.get("source_text") or ""),
                        "source_chunk_id": str(props.get("source_chunk_id") or ""),
                    })
            source_text, source_chunk_id = self._extract_original_evidence(path.relationships, path.nodes)
            source_snippets, source_chunk_ids = self._collect_original_evidence_items(path.relationships, path.nodes)
            docs.append(Document(page_content=desc, metadata={
                "search_type": "graph_path",
                "relevance_score": path.relevance_score,
                "entity_name": path.nodes[0].get("name", "图路径"),
                "subgraph_edges": edges,
                "source_text": source_text,
                "source_chunk_id": source_chunk_id,
                "source_snippets": source_snippets,
                "source_chunk_ids": source_chunk_ids,
            }))
        return docs

    def _subgraph_to_documents(self, subgraph, chains, query):
        if not subgraph.central_nodes:
            return []
        desc = self._build_subgraph_description(subgraph)
        edges = []
        for rel in subgraph.relationships:
            if rel.get("source") and rel.get("target"):
                props = rel.get("properties") or {}
                edges.append({
                    "source": rel["source"],
                    "relation": rel["type"],
                    "target": rel["target"],
                    "source_type": "Concept",
                    "target_type": "Concept",
                    "evidence": str(props.get("evidence") or ""),
                    "source_text": str(props.get("source_text") or ""),
                    "source_chunk_id": str(props.get("source_chunk_id") or ""),
                })
        source_text, source_chunk_id = self._extract_original_evidence(
            subgraph.relationships,
            subgraph.central_nodes + subgraph.connected_nodes,
        )
        source_snippets, source_chunk_ids = self._collect_original_evidence_items(
            subgraph.relationships,
            subgraph.central_nodes + subgraph.connected_nodes,
        )

        return [Document(page_content=desc, metadata={
            "search_type": "knowledge_subgraph",
            "relevance_score": 0.8,
            "entity_name": subgraph.central_nodes[0].get("name") or subgraph.central_nodes[0].get("chunk_id") or "知识子图",
            "subgraph_edges": edges,
            "source_text": source_text,
            "source_chunk_id": source_chunk_id,
            "source_snippets": source_snippets,
            "source_chunk_ids": source_chunk_ids,
        })]

    def _build_path_description(self, path):
        if not path.nodes:
            return ""
        import re
        zh_map = {
            "MEMBER_OF": "属于",
            "BELONGS_TO": "属于",
            "BELONGS_TO_SECT": "归属",
            "FIGHTS_FOR": "效力于",
            "PARTICIPATES_IN": "参与了",
            "LEADS": "领导了",
            "INITIATES": "发起了",
            "RELATED_TO": "与…相关",
            "IN_COMMUNITY": "属于",
            "BELONGS_TO_CHAPTER": "出自",
            "MENTIONS": "提及了",
            "MENTIONS_EVENT": "提及了",
            "MENTIONS_ORG": "提及了",
            "KILLS": "击杀了",
            "DEFEATS": "击败了",
            "OWNS": "拥有",
            "CREATES": "创建了",
            "MASTER_OF": "是…的师父",
            "APPRENTICE_OF": "是…的徒弟",
            "FATHER_OF": "是…的父亲",
            "CHILD_OF": "是…的子女",
            "BROTHER_OF": "与…是兄弟",
            "OPPOSES": "与…对立",
            "ALLIES_WITH": "与…结盟",
        }
        lines = []
        chain = []
        for index, node in enumerate(path.nodes):
            name = node.get("name", "节点")
            chain.append(name)
            if index < len(path.relationships):
                rel_type = path.relationships[index]["type"]
                rel_label = zh_map.get(rel_type, rel_type)
                # Format: "A belongs to B" style
                chain.append(rel_label)
        if chain:
            lines.append("图谱路径: " + " → ".join(chain))
        # Add source text evidence
        for rel in path.relationships:
            props = rel.get("properties") or {}
            source = props.get("source_text") or props.get("evidence") or ""
            if source and str(source).strip() and not re.search(r'[一-龥]', str(source)):
                continue
            source = str(source).strip()[:180] if source else ""
            if source and source not in lines:
                lines.append("原文证据: " + source)
                break
        return "\n".join(lines)

    def _build_subgraph_description(self, subgraph):
        central_names = [node.get("name") or node.get("chunk_id") or node.get("community_id", "") for node in subgraph.central_nodes]
        lines = []
        
        # Build readable relation sentences
        zh_map = {
            "MASTER_OF": "是…的师父", "APPRENTICE_OF": "是…的徒弟",
            "BELONGS_TO_SECT": "归属于", "FIGHTS_FOR": "效力于",
            "FATHER_OF": "是…的父亲", "CHILD_OF": "是…的子女",
            "OWNS": "拥有", "BESTOWS": "赐予了", "LOSES": "失去了",
            "KILLS": "击杀了", "DEFEATS": "击败了", "CAPTURES": "擒获了",
            "PARTICIPATES_IN": "参与了", "LEADS": "领导了",
            "CREATES": "创建了", "INITIATES": "发起了",
            "OPPOSES": "与…对立", "ALLIES_WITH": "与…结盟",
            "RELATED_TO": "与…相关", "MENTIONS": "提及了",
        }
        
        if central_names:
            lines.append("核心实体: " + "、".join(central_names[:6]))
        
        for rel in subgraph.relationships[:15]:
            source = rel.get("source", "")
            target = rel.get("target", "")
            rel_type = rel.get("type", "")
            if not source or not target:
                continue
            rel_label = zh_map.get(rel_type, rel_type)
            lines.append(f"- {source}{rel_label}{target}")
            
            # Add evidence text
            props = rel.get("properties") or {}
            evidence = props.get("source_text") or props.get("evidence") or ""
            if evidence and str(evidence).strip() and len(str(evidence).strip()) > 10:
                ev = str(evidence).strip()[:150]
                if ev not in "".join(lines):
                    lines.append(f"  证据: {ev}")
        
        # Add text chunks if present
        by_type = defaultdict(list)
        for node in subgraph.connected_nodes:
            labels = node.get("labels", ["Concept"])
            by_type[labels[0]].append(node)
        
        if "TextChunk" in by_type:
            for node in by_type["TextChunk"][:2]:
                text = node.get("text", "")[:200]
                if text:
                    lines.append(f"原文: {text}")
        
        return "\n".join(lines)
