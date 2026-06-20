"""
混合检索模块
基于双层检索范式：实体级 + 主题级检索
结合图结构检索和向量检索，使用 Round-robin 轮询策略
"""

import json
import logging
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from neo4j import GraphDatabase

from .graph_indexing import GraphIndexingModule
from .gold_graph_policy import build_gold_entity_predicate, filter_gold_documents
from .query_heuristics import derive_topic_keywords, extract_candidate_entities
from .retrieval_shared import evidence_priority

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """检索结果数据结构"""
    content: str
    node_id: str
    node_type: str
    relevance_score: float
    retrieval_level: str
    metadata: Dict[str, Any]


class HybridRetrievalModule:
    """混合检索模块"""

    def __init__(self, config, index_module, data_module, llm_client, driver=None):
        self.config = config
        self.index_module = index_module
        self.data_module = data_module
        self.llm_client = llm_client
        self.driver = driver
        self.bm25_retriever = None
        self.graph_indexing = GraphIndexingModule(config, llm_client)
        self.graph_indexed = False
        self._keyword_cache = {}

    def initialize(self, chunks: List[Document]):
        logger.info("初始化混合检索模块...")
        if not self.driver:
            self.driver = GraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password)
            )

        if chunks:
            self.bm25_retriever = BM25Retriever.from_documents(chunks)
            logger.info(f"BM25检索器初始化完成，文档数量: {len(chunks)}")

        self._build_graph_index()

    def _build_graph_index(self):
        if self.graph_indexed:
            return

        logger.info("开始构建图索引...")
        try:
            persons = self.data_module.persons
            events = self.data_module.events
            organizations = getattr(self.data_module, "organizations", [])

            self.graph_indexing.create_entity_key_values(persons, events, organizations)
            relationships = self._extract_relationships_from_graph()
            self.graph_indexing.create_relation_key_values(relationships)
            self.graph_indexing.deduplicate_entities_and_relations()

            self.graph_indexed = True
            stats = self.graph_indexing.get_statistics()
            logger.info(f"图索引构建完成: {stats}")
        except Exception as e:
            logger.error(f"构建图索引失败: {e}")

    def _extract_relationships_from_graph(self) -> List[Tuple[str, str, str]]:
        relationships = []
        try:
            with self.driver.session() as session:
                query = """
                MATCH (source)-[r]->(target)
                WHERE source.gold_source = 'manual_gold'
                  AND target.gold_source = 'manual_gold'
                  AND any(label IN labels(source) WHERE label IN ['Person', 'Organization', 'Event', 'Period'])
                  AND any(label IN labels(target) WHERE label IN ['Person', 'Organization', 'Event', 'Period'])
                RETURN coalesce(source.name, source.chunk_id, source.community_id) as source_id,
                       type(r) as relation_type,
                       coalesce(target.name, target.chunk_id, target.community_id) as target_id
                LIMIT 2000
                """
                for record in session.run(query):
                    if record["source_id"] and record["target_id"]:
                        relationships.append((
                            record["source_id"],
                            record["relation_type"],
                            record["target_id"],
                        ))
        except Exception as e:
            logger.error(f"提取图关系失败: {e}")
        return relationships

    def extract_query_keywords(self, query: str) -> Tuple[List[str], List[str]]:
        cached = self._keyword_cache.get(query)
        if cached:
            return cached

        known_entities = []
        for entity in getattr(self.graph_indexing, "entity_kv_store", {}).values():
            entity_name = getattr(entity, "entity_name", "")
            if entity_name:
                known_entities.append(entity_name)

        local_entity_keywords = extract_candidate_entities(query, known_entities=known_entities)
        local_topic_keywords = derive_topic_keywords(query)
        if local_entity_keywords or local_topic_keywords:
            result = (local_entity_keywords[:6], local_topic_keywords[:6])
            self._keyword_cache[query] = result
            return result

        prompt = f"""
        作为封神演义知识助手，请分析以下查询并提取关键词，分为两个层次：

        查询：{query}

        1. 实体级关键词：具体人物、教派/势力、地点、法宝、坐骑、阵法、战役事件、神位
        2. 主题级关键词：师承、教派、阵营、法宝、封神、上榜、肉身成圣、破阵等主题词

        请严格输出 JSON：
        {{
            "entity_keywords": ["关键词1", "关键词2"],
            "topic_keywords": ["关键词1", "关键词2"]
        }}
        """

        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            content = (response.choices[0].message.content or "").strip()
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            result = json.loads(content.strip())
            entity_keywords = result.get("entity_keywords", [])
            topic_keywords = result.get("topic_keywords", [])
            logger.info(f"关键词提取完成 - 实体级: {entity_keywords}, 主题级: {topic_keywords}")
            result = (entity_keywords, topic_keywords)
            self._keyword_cache[query] = result
            return result
        except Exception as e:
            logger.error(f"关键词提取失败: {e}")
            keywords = [token for token in query.replace("，", " ").replace("？", " ").split() if token]
            result = (keywords[:3], keywords[:6])
            self._keyword_cache[query] = result
            return result

    def entity_level_retrieval(self, entity_keywords: List[str], top_k: int = 5) -> List[RetrievalResult]:
        results = []
        for keyword in entity_keywords:
            entities = self.graph_indexing.get_entities_by_key(keyword)
            for entity in entities:
                neighbors = self._get_node_neighbors(entity.metadata["node_id"], max_neighbors=3)
                content = entity.value_content
                if neighbors:
                    content += f"\n相关节点: {', '.join(neighbors)}"
                results.append(RetrievalResult(
                    content=content,
                    node_id=entity.metadata["node_id"],
                    node_type=entity.entity_type,
                    relevance_score=0.9,
                    retrieval_level="entity",
                    metadata={
                        "entity_name": entity.entity_name,
                        "entity_type": entity.entity_type,
                        "index_keys": entity.index_keys,
                        "matched_keyword": keyword,
                    },
                ))

        if len(results) < top_k:
            results.extend(self._neo4j_entity_level_search(entity_keywords, top_k - len(results)))

        results.sort(key=lambda x: x.relevance_score, reverse=True)
        logger.info(f"实体级检索完成，返回 {len(results)} 个结果")
        return results[:top_k]

    def _neo4j_entity_level_search(self, keywords: List[str], limit: int) -> List[RetrievalResult]:
        results = []
        try:
            with self.driver.session() as session:
                cypher_query = """
                UNWIND $keywords as keyword
                MATCH (n)
                WHERE """ + build_gold_entity_predicate("n") + """
                  AND (
                    coalesce(n.name, n.chunk_id, n.community_id, '') CONTAINS keyword
                    OR coalesce(n.description, n.summary, n.text, '') CONTAINS keyword
                  )
                RETURN coalesce(n.name, n.chunk_id, n.community_id) as node_id,
                       coalesce(n.name, n.chunk_id, n.community_id) as name,
                       coalesce(n.description, n.summary, n.text, '') as description,
                       labels(n) as labels,
                       1.0 as score
                LIMIT $limit
                """
                for record in session.run(cypher_query, {"keywords": keywords, "limit": limit}):
                    labels = record["labels"] or []
                    node_type = labels[0] if labels else "Concept"
                    content_parts = [f"名称: {record['name']}"]
                    if record["description"]:
                        content_parts.append(f"内容: {record['description'][:240]}")
                    results.append(RetrievalResult(
                        content="\n".join(content_parts),
                        node_id=record["node_id"],
                        node_type=node_type,
                        relevance_score=float(record["score"]) * 0.7,
                        retrieval_level="entity",
                        metadata={
                            "name": record["name"],
                            "labels": labels,
                            "source": "neo4j_fallback",
                        },
                    ))
        except Exception as e:
            logger.error(f"Neo4j补充检索失败: {e}")
        return results

    def topic_level_retrieval(self, topic_keywords: List[str], top_k: int = 5) -> List[RetrievalResult]:
        results = []

        for keyword in topic_keywords:
            relations = self.graph_indexing.get_relations_by_key(keyword)
            for relation in relations:
                source_entity = self.graph_indexing.entity_kv_store.get(relation.source_entity)
                target_entity = self.graph_indexing.entity_kv_store.get(relation.target_entity)
                if source_entity and target_entity:
                    content_parts = [
                        f"主题: {keyword}",
                        relation.value_content,
                        f"相关实体: {source_entity.entity_name}",
                        f"关联对象: {target_entity.entity_name}",
                    ]
                    results.append(RetrievalResult(
                        content="\n".join(content_parts),
                        node_id=relation.relation_id,
                        node_type="Relation",
                        relevance_score=0.95,
                        retrieval_level="topic",
                        metadata={
                            "relation_id": relation.relation_id,
                            "relation_type": relation.relation_type,
                            "source_name": source_entity.entity_name,
                            "target_name": target_entity.entity_name,
                            "matched_keyword": keyword,
                            "index_keys": relation.index_keys,
                        },
                    ))

        if len(results) < top_k:
            results.extend(self._neo4j_topic_level_search(topic_keywords, top_k - len(results)))

        results.sort(key=lambda x: x.relevance_score, reverse=True)
        logger.info(f"主题级检索完成，返回 {len(results)} 个结果")
        return results[:top_k]

    def community_level_retrieval(self, keywords: List[str], top_k: int = 5) -> List[RetrievalResult]:
        results = []
        if not keywords:
            return results
        try:
            with self.driver.session() as session:
                cypher_query = """
                UNWIND $keywords as keyword
                MATCH (c:Community)
                WHERE coalesce(c.title, '') CONTAINS keyword
                   OR coalesce(c.summary, '') CONTAINS keyword
                   OR keyword IN coalesce(c.top_organizations, [])
                OPTIONAL MATCH (member)-[:IN_COMMUNITY]->(c)
                WITH keyword, c, collect(DISTINCT coalesce(member.name, member.chunk_id))[0..8] as members
                RETURN keyword,
                       c.community_id as community_id,
                       c.level as level,
                       c.title as title,
                       c.summary as summary,
                       c.time_span as time_span,
                       c.top_organizations as top_organizations,
                       members
                LIMIT $limit
                """
                for record in session.run(cypher_query, {"keywords": keywords, "limit": top_k * 2}):
                    top_organizations = record["top_organizations"] or []
                    members = [m for m in (record["members"] or []) if m]
                    content_parts = [
                        f"社团: {record['title'] or record['community_id']}",
                        f"层级: L{record['level']}",
                    ]
                    if top_organizations:
                        content_parts.append(f"核心组织: {'、'.join(top_organizations)}")
                    if record["time_span"]:
                        content_parts.append(f"时间范围: {record['time_span']}")
                    if record["summary"]:
                        content_parts.append(f"摘要: {record['summary']}")
                    if members:
                        content_parts.append(f"核心成员: {'、'.join(members[:8])}")
                    results.append(RetrievalResult(
                        content="\n".join(content_parts),
                        node_id=record["community_id"],
                        node_type="Community",
                        relevance_score=0.93,
                        retrieval_level="community",
                        metadata={
                            "community_id": record["community_id"],
                            "community_level": record["level"],
                            "matched_keyword": record["keyword"],
                            "name": record["title"] or record["community_id"],
                        },
                    ))
        except Exception as e:
            logger.error(f"社团检索失败: {e}")
        return results[:top_k]

    def _neo4j_topic_level_search(self, keywords: List[str], limit: int) -> List[RetrievalResult]:
        results = []
        try:
            with self.driver.session() as session:
                cypher_query = """
                UNWIND $keywords as keyword
                MATCH (n)
                WHERE """ + build_gold_entity_predicate("n") + """
                  AND (
                    coalesce(n.organization, '') CONTAINS keyword
                    OR coalesce(n.period, '') CONTAINS keyword
                    OR coalesce(n.event_type, '') CONTAINS keyword
                    OR coalesce(n.org_type, '') CONTAINS keyword
                    OR coalesce(n.location, '') CONTAINS keyword
                    OR coalesce(n.chapter, '') CONTAINS keyword
                    OR coalesce(n.section, '') CONTAINS keyword
                    OR coalesce(n.community_l1, '') CONTAINS keyword
                    OR coalesce(n.community_l2, '') CONTAINS keyword
                    OR coalesce(n.community_l3, '') CONTAINS keyword
                    OR coalesce(n.summary, '') CONTAINS keyword
                  )
                OPTIONAL MATCH (n)-[r]-(m)
                WHERE """ + build_gold_entity_predicate("m") + """
                WITH n, keyword, collect(DISTINCT coalesce(m.name, m.chunk_id, m.community_id))[0..4] as neighbors
                RETURN coalesce(n.name, n.chunk_id, n.community_id) as node_id,
                       coalesce(n.name, n.chunk_id, n.community_id) as name,
                       coalesce(n.period, n.organization, n.event_type, n.org_type, n.chapter, n.section, n.location, '') as category,
                       labels(n) as labels,
                       neighbors,
                       keyword as matched_keyword
                LIMIT $limit
                """
                for record in session.run(cypher_query, {"keywords": keywords, "limit": limit}):
                    labels = record["labels"] or []
                    node_type = labels[0] if labels else "Concept"
                    content_parts = [f"名称: {record['name']}"]
                    if record["category"]:
                        content_parts.append(f"主题属性: {record['category']}")
                    if record["neighbors"]:
                        content_parts.append(f"相关节点: {', '.join([n for n in record['neighbors'] if n])}")
                    results.append(RetrievalResult(
                        content="\n".join(content_parts),
                        node_id=record["node_id"],
                        node_type=node_type,
                        relevance_score=0.75,
                        retrieval_level="topic",
                        metadata={
                            "name": record["name"],
                            "category": record["category"],
                            "matched_keyword": record["matched_keyword"],
                            "source": "neo4j_fallback",
                        },
                    ))
        except Exception as e:
            logger.error(f"Neo4j主题级检索失败: {e}")
        return results

    def dual_level_retrieval(self, query: str, top_k: int = 5) -> List[Document]:
        logger.info(f"开始双层检索: {query}")
        entity_keywords, topic_keywords = self.extract_query_keywords(query)
        entity_results = self.entity_level_retrieval(entity_keywords, top_k)
        topic_results = self.topic_level_retrieval(topic_keywords, top_k)
        all_results = entity_results + topic_results

        seen_nodes = set()
        unique_results = []
        for result in sorted(all_results, key=lambda x: x.relevance_score, reverse=True):
            if result.node_id not in seen_nodes:
                seen_nodes.add(result.node_id)
                unique_results.append(result)

        documents = []
        for result in unique_results[:top_k]:
            entity_name = result.metadata.get("name") or result.metadata.get("entity_name", "未知节点")
            documents.append(Document(
                page_content=result.content,
                metadata={
                    "node_id": result.node_id,
                    "node_type": result.node_type,
                    "retrieval_level": result.retrieval_level,
                    "relevance_score": result.relevance_score,
                    "score": result.relevance_score,
                    "entity_name": entity_name,
                    "search_type": "dual_level",
                    **result.metadata,
                },
            ))

        logger.info(f"双层检索完成，返回 {len(documents)} 个文档")
        return filter_gold_documents(documents)

    def _normalize_vector_score(self, raw_score: float) -> float:
        try:
            distance = float(raw_score)
        except (TypeError, ValueError):
            return 0.0

        if distance < 0:
            return 0.0
        return 1.0 / (1.0 + distance)

    def _get_document_identity(self, doc: Document) -> str:
        return str(doc.metadata.get("node_id") or hash(doc.page_content[:200]))

    def vector_search_enhanced(self, query: str, top_k: int = 5) -> List[Document]:
        try:
            vector_docs_with_scores = self.index_module.similarity_search_with_score(query, k=top_k * 2)
            enhanced_docs = []
            for result, raw_score in vector_docs_with_scores:
                metadata = dict(result.metadata)
                node_id = metadata.get("node_id")
                neighbors = self._get_node_neighbors(node_id) if node_id else []
                entity_name = metadata.get("entity_name", "未知节点")
                similarity_score = self._normalize_vector_score(raw_score)
                enhanced_docs.append(Document(
                    page_content=result.page_content,
                    metadata={
                        **metadata,
                        "entity_name": entity_name,
                        "score": similarity_score,
                        "vector_distance": float(raw_score),
                        "related_neighbors": neighbors[:3],
                        "search_type": "vector_enhanced",
                    },
                ))
            return filter_gold_documents(enhanced_docs)[:top_k]
        except Exception as e:
            logger.error(f"增强向量检索失败: {e}")
            return []

    def _get_node_neighbors(self, node_id: str, max_neighbors: int = 3) -> List[str]:
        try:
            with self.driver.session() as session:
                query = """
                MATCH (n)-[r]-(neighbor)
                WHERE """ + build_gold_entity_predicate("n") + """
                  AND """ + build_gold_entity_predicate("neighbor") + """
                  AND coalesce(n.name, n.chunk_id, n.community_id) = $node_id
                RETURN coalesce(neighbor.name, neighbor.chunk_id, neighbor.community_id) as name
                LIMIT $limit
                """
                result = session.run(query, {"node_id": node_id, "limit": max_neighbors})
                return [record["name"] for record in result if record["name"]]
        except Exception as e:
            logger.error(f"获取邻居节点失败: {e}")
            return []


    def hybrid_search(self, query: str, top_k: int = 5) -> List[Document]:
        logger.info(f"开始混合检索: {query}")
        dual_docs = self.dual_level_retrieval(query, top_k)
        vector_docs = self.vector_search_enhanced(query, top_k)

        candidates = []
        for doc in dual_docs:
            doc.metadata["search_method"] = "dual_level"
            doc.metadata["final_score"] = float(doc.metadata.get("score", doc.metadata.get("relevance_score", 0.0)))
            candidates.append(doc)

        for doc in vector_docs:
            doc.metadata["search_method"] = "vector_enhanced"
            doc.metadata["final_score"] = float(doc.metadata.get("score", 0.0))
            candidates.append(doc)

        best_docs: Dict[str, Document] = {}
        for doc in candidates:
            doc_id = self._get_document_identity(doc)
            existing = best_docs.get(doc_id)
            current_tuple = (
                evidence_priority(doc),
                float(doc.metadata.get("final_score", 0.0)),
            )
            if existing is None:
                best_docs[doc_id] = doc
                continue
            existing_tuple = (
                evidence_priority(existing),
                float(existing.metadata.get("final_score", 0.0)),
            )
            if current_tuple > existing_tuple:
                best_docs[doc_id] = doc

        ranked_docs = sorted(
            best_docs.values(),
            key=lambda item: (evidence_priority(item), float(item.metadata.get("final_score", 0.0))),
            reverse=True,
        )

        for index, doc in enumerate(ranked_docs):
            doc.metadata["result_rank"] = index

        final_docs = ranked_docs[:top_k]
        logger.info(f"质量优先合并：从总共{len(candidates)}个结果筛选为{len(final_docs)}个文档")
        logger.info(f"混合检索完成，返回 {len(final_docs)} 个文档")
        return filter_gold_documents(final_docs)

    def close(self):
        if self.driver:
            self.driver.close()
            logger.info("Neo4j连接已关闭")
