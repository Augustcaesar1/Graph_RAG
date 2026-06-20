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
    def __init__(self, config, llm_client):
        self.config = config
        self.llm_client = llm_client
        self.driver = None
        self.entity_cache = {}
        self.relation_cache = {}
        self.subgraph_cache = {}

    def initialize(self):
        logger.info("初始化封神演义图RAG检索系统...")
        try:
            self.driver = GraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password)
            )
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
                               r.evidence AS evidence
                        LIMIT 30
                        """,
                        name=entity_name,
                    ))

                    edges = []
                    relation_lines = []
                    for rr in rel_rows:
                        edge = {
                            "source": rr["src"],
                            "relation": rr["rel"],
                            "target": rr["tgt"],
                            "source_type": rr["source_type"],
                            "target_type": rr["target_type"],
                        }
                        edges.append(edge)
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
                            "relevance_score": 1.25,
                        },
                    ))
        except Exception as e:
            logger.error(f"精确实体检索失败: {e}")

        return docs[:top_k]

    def graph_rag_search(self, query: str, top_k: int = 5) -> List[Document]:
        logger.info(f"开始图RAG检索: {query}")
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

            results = sorted(results, key=lambda x: x.metadata.get("relevance_score", 0.0), reverse=True)
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

    def _paths_to_documents(self, paths, query):
        docs = []
        for path in paths:
            desc = self._build_path_description(path)
            edges = []
            for index, rel in enumerate(path.relationships):
                if rel.get("source") and rel.get("target"):
                    edges.append({
                        "source": rel["source"],
                        "relation": rel["type"],
                        "target": rel["target"],
                        "source_type": path.nodes[index].get("labels", ["Concept"])[0] if path.nodes[index].get("labels") else "Concept",
                        "target_type": path.nodes[index + 1].get("labels", ["Concept"])[0] if index + 1 < len(path.nodes) and path.nodes[index + 1].get("labels") else "Concept",
                    })
            source_text, source_chunk_id = self._extract_original_evidence(path.relationships, path.nodes)
            docs.append(Document(page_content=desc, metadata={
                "search_type": "graph_path",
                "relevance_score": path.relevance_score,
                "entity_name": path.nodes[0].get("name", "图路径"),
                "subgraph_edges": edges,
                "source_text": source_text,
                "source_chunk_id": source_chunk_id,
            }))
        return docs

    def _subgraph_to_documents(self, subgraph, chains, query):
        if not subgraph.central_nodes:
            return []
        desc = self._build_subgraph_description(subgraph)
        edges = []
        for rel in subgraph.relationships:
            if rel.get("source") and rel.get("target"):
                edges.append({
                    "source": rel["source"],
                    "relation": rel["type"],
                    "target": rel["target"],
                    "source_type": "Concept",
                    "target_type": "Concept",
                })
        source_text, source_chunk_id = self._extract_original_evidence(
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
        })]

    def _build_path_description(self, path):
        if not path.nodes:
            return ""
        parts = []
        zh_map = {
            "MEMBER_OF": "属于组织",
            "BELONGS_TO": "属于阶段",
            "PARTICIPATES_IN": "参与事件",
            "LEADS": "领导",
            "INITIATES": "发起",
            "RELATED_TO": "相关",
            "IN_COMMUNITY": "属于社团",
            "BELONGS_TO_CHAPTER": "所属章节",
            "MENTIONS": "提及",
            "MENTIONS_EVENT": "提及事件",
            "MENTIONS_ORG": "提及组织",
        }
        for index, node in enumerate(path.nodes):
            parts.append(node.get("name", "节点"))
            if index < len(path.relationships):
                rel_type = path.relationships[index]["type"]
                parts.append(f" --({zh_map.get(rel_type, rel_type)})--> ")
        return "".join(parts)

    def _build_subgraph_description(self, subgraph):
        central_names = [node.get("name") or node.get("chunk_id") or node.get("community_id", "") for node in subgraph.central_nodes]
        parts = [f"### 知识子图核心：{', '.join(central_names)}"]

        by_type = defaultdict(list)
        for node in subgraph.central_nodes + subgraph.connected_nodes:
            labels = node.get("labels", ["Concept"])
            label = labels[0] if labels else "Concept"
            by_type[label].append(node)

        if "Person" in by_type:
            parts.append("\n人物:")
            for node in by_type["Person"][:12]:
                parts.append(f"- {node.get('name', '')} ({node.get('organization', '') or node.get('role', '')})")
        if "Event" in by_type:
            parts.append("\n事件:")
            for node in by_type["Event"][:12]:
                parts.append(f"- {node.get('name', '')}")
        if "Organization" in by_type:
            parts.append("\n组织:")
            for node in by_type["Organization"][:10]:
                parts.append(f"- {node.get('name', '')}")
        if "Period" in by_type:
            parts.append("\n历史阶段:")
            for node in by_type["Period"][:8]:
                parts.append(f"- {node.get('name', '') or node.get('title', '')}")
        if "TextChunk" in by_type:
            parts.append("\n原文片段:")
            for node in by_type["TextChunk"][:3]:
                text = node.get("text", "")
                parts.append(f"- {node.get('chapter', '章节未详')} / {node.get('section', '小节未详')}：{text[:80]}")
        if "Community" in by_type:
            parts.append("\n相关社团:")
            for node in by_type["Community"][:6]:
                parts.append(f"- {node.get('title') or node.get('community_id', '')}: {node.get('summary', '')}")

        if subgraph.relationships:
            zh_map = {
                "FAMILY_OF": "亲属",
                "PARENT_OF": "父母",
                "CHILD_OF": "子女",
                "SIBLING_OF": "手足",
                "SPOUSE_OF": "配偶",
                "CONCUBINE_OF": "妾室",
                "SERVES": "服侍",
                "MASTER_OF": "主子",
                "FRIEND_OF": "朋友",
                "LOVES": "爱慕",
                "RIVAL_OF": "对手",
                "CONFLICT_WITH": "冲突",
                "BELONGS_TO": "归属",
                "LIVES_IN": "居于",
                "VISITS": "到访",
                "PARTICIPATES_IN": "参与",
                "CREATES": "创作",
                "OWNS": "拥有",
                "GIVES": "赠予",
                "RELATED_TO": "相关",
                "MENTIONS": "提及",
                "BELONGS_TO_CHAPTER": "所属章节",
            }
            parts.append("\n关系:")
            for rel in subgraph.relationships[:20]:
                rel_type = zh_map.get(rel["type"], rel["type"])
                detail = ""
                props = rel.get("properties") or {}
                if props.get("source_text"):
                    detail = f" | 证据: {str(props['source_text'])[:60]}"
                parts.append(f"- {rel['source']} <{rel_type}> {rel['target']}{detail}")

        return "\n".join(parts)
