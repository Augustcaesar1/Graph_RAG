"""
封神演义图数据库数据准备模块
从 Neo4j 读取 Person / Faction / Artifact / Beast / Formation / Event / DeityPosition / Location / TextChunk 节点，转换为 RAG 文档
"""

import csv
import json
import logging
from collections import defaultdict
from io import StringIO
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from neo4j import GraphDatabase

from .gold_schema import ENTITY_TYPES, RELATION_LABELS_ZH

logger = logging.getLogger(__name__)


class GraphNode:
    def __init__(self, node_id: str, labels: List[str], name: str, properties: Dict[str, Any]):
        self.node_id = node_id
        self.labels = labels
        self.name = name
        self.properties = properties


class GraphDataPreparationModule:
    """从 Neo4j 读取封神演义图数据并转换为 RAG 文档"""

    def __init__(self, uri: str = None, user: str = None, password: str = None, database: str = "neo4j", driver=None, config=None):
        if config is not None:
            self.uri = config.neo4j_uri
            self.user = config.neo4j_user
            self.password = config.neo4j_password
            self.database = config.neo4j_database
        else:
            self.uri = uri
            self.user = user
            self.password = password
            self.database = database
        self.documents: List[Document] = []
        self.chunks: List[Document] = []

        self.nodes_by_type: Dict[str, List[GraphNode]] = {etype: [] for etype in ENTITY_TYPES}
        self.text_chunks: List[GraphNode] = []
        self.chapters: List[GraphNode] = []
        self.communities: List[GraphNode] = []
        if driver is not None:
            self.driver = driver
        else:
            self.driver = None
            self._connect()

    def _connect(self):
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            with self.driver.session(database=self.database) as session:
                if session.run("RETURN 1 AS test").single():
                    logger.info("Neo4j 连接测试成功")
        except Exception as e:
            logger.error("连接Neo4j失败: %s", e)
            raise

    def close(self):
        if getattr(self, "driver", None):
            self.driver.close()

    @property
    def persons(self): return self.nodes_by_type.get("Person", [])
    @property
    def factions(self): return self.nodes_by_type.get("Faction", [])
    @property
    def artifacts(self): return self.nodes_by_type.get("Artifact", [])
    @property
    def beasts(self): return self.nodes_by_type.get("Beast", [])
    @property
    def formations(self): return self.nodes_by_type.get("Formation", [])
    @property
    def events(self): return self.nodes_by_type.get("Event", [])
    @property
    def deity_positions(self): return self.nodes_by_type.get("DeityPosition", [])
    @property
    def locations(self): return self.nodes_by_type.get("Location", [])

    def load_graph_data(self) -> Dict[str, Any]:
        logger.info("从Neo4j加载封神演义图数据...")
        with self.driver.session(database=self.database) as session:
            for etype in ENTITY_TYPES:
                self.nodes_by_type[etype] = self._load_nodes(session, etype, "name")
            self.text_chunks = self._load_nodes(session, "TextChunk", "chunk_id")
            self.chapters = self._load_nodes(session, "Chapter", "title", order_by="n.chapter_number")
            self.communities = self._load_nodes(session, "Community", "community_id") if self._label_exists(session, "Community") else []

        return {
            **{etype.lower() + "s": len(nodes) for etype, nodes in self.nodes_by_type.items()},
            "text_chunks": len(self.text_chunks),
            "chapters": len(self.chapters),
            "communities": len(self.communities),
        }

    def _label_exists(self, session, label: str) -> bool:
        rec = session.run("CALL db.labels() YIELD label RETURN collect(label) AS labels").single()
        return label in (rec["labels"] or [])

    def _load_nodes(self, session, label: str, id_field: str, alias: str = "n", order_by: str = None) -> List[GraphNode]:
        order_clause = f"ORDER BY {order_by}" if order_by else f"ORDER BY {alias}.{id_field}"
        query = f"""
            MATCH ({alias}:{label})
            RETURN coalesce({alias}.{id_field}, {alias}.name, {alias}.chunk_id, elementId({alias})) as node_id,
                   coalesce({alias}.title, {alias}.name, {alias}.chunk_id, {alias}.{id_field}) as name,
                   labels({alias}) as labels,
                   properties({alias}) as props
            {order_clause}
        """
        nodes = []
        try:
            for rec in session.run(query):
                nodes.append(GraphNode(rec["node_id"], rec["labels"], rec["name"] or rec["node_id"], dict(rec["props"])))
        except Exception as e:
            logger.debug("加载 %s 节点失败: %s", label, e)
        logger.info("加载了 %s 个 %s 节点", len(nodes), label)
        return nodes

    def _serialize_value(self, value: Any) -> Any:
        if isinstance(value, list):
            return "、".join(str(item) for item in value)
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
        return value

    def _node_to_row(self, node: GraphNode, fields: List[str]) -> Dict[str, Any]:
        row = {"node_id": node.node_id, "name": node.name, "labels": "|".join(node.labels)}
        for field in fields:
            row[field] = self._serialize_value(node.properties.get(field, ""))
        return row

    def export_dataset_rows(self, dataset_name: str, relation_limit: int = 5000) -> List[Dict[str, Any]]:
        if dataset_name == "relations":
            return self.export_relation_rows(relation_limit)
        if dataset_name == "text_chunks":
            return [self._node_to_row(c, ["chapter_number", "chapter_title", "chunk_index", "text", "text_length"]) for c in self.text_chunks]
        label_map = {
            "persons": "Person", "factions": "Faction", "artifacts": "Artifact", "beasts": "Beast",
            "formations": "Formation", "events": "Event", "deity_positions": "DeityPosition", "locations": "Location"
        }
        etype = label_map.get(dataset_name)
        if not etype:
            raise ValueError(f"不支持的数据集: {dataset_name}")
        return [self._node_to_row(n, ["alias", "description", "entity_type"]) for n in self.nodes_by_type.get(etype, [])]

    def export_relation_rows(self, limit: int = 5000) -> List[Dict[str, Any]]:
        rows = []
        try:
            with self.driver.session(database=self.database) as session:
                query = """
                MATCH (a)-[r]->(b)
                RETURN coalesce(a.name, a.chunk_id) AS source,
                       head(labels(a)) AS source_type,
                       type(r) AS relation,
                       coalesce(b.name, b.chunk_id) AS target,
                       head(labels(b)) AS target_type,
                       properties(r) AS props
                LIMIT $limit
                """
                for rec in session.run(query, {"limit": limit}):
                    props = dict(rec["props"] or {})
                    rows.append({
                        "source": rec["source"], "source_type": rec["source_type"],
                        "relation": rec["relation"], "target": rec["target"], "target_type": rec["target_type"],
                        "evidence": self._serialize_value(props.get("evidence", "")),
                        "confidence": self._serialize_value(props.get("confidence", "")),
                    })
        except Exception as e:
            logger.error("导出关系表失败: %s", e)
        return rows

    def rows_to_csv(self, rows: List[Dict[str, Any]]) -> str:
        if not rows:
            return ""
        buffer = StringIO()
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: self._serialize_value(value) for key, value in row.items()})
        return buffer.getvalue()

    def build_entity_documents(self, etype: str) -> List[Document]:
        docs = []
        nodes = self.nodes_by_type.get(etype, [])
        with self.driver.session(database=self.database) as session:
            for node in nodes:
                name = node.name
                props = node.properties
                rels = []
                rel_result = session.run(
                    """
                    MATCH (n {name: $name})-[r]-(other)
                    WHERE coalesce(other.name, other.chunk_id) IS NOT NULL
                    RETURN type(r) AS rtype,
                           coalesce(other.name, other.chunk_id) AS other_name,
                           head(labels(other)) AS other_type,
                           r.evidence AS evidence,
                           r.source_text AS source_text,
                           r.source_chunk_id AS source_chunk_id
                    LIMIT 50
                    """,
                    name=name,
                )
                relation_source_text = ""
                relation_source_chunk_id = ""
                for rr in rel_result:
                    rel_zh = RELATION_LABELS_ZH.get(rr["rtype"], rr["rtype"])
                    otype = ENTITY_TYPES.get(rr["other_type"], rr["other_type"])
                    item = f"{rel_zh} → {rr['other_name']}（{otype}）"
                    if rr.get("evidence"):
                        item += f"；证据：{str(rr['evidence'])[:80]}"
                    if not relation_source_text and rr.get("source_text"):
                        relation_source_text = rr.get("source_text")
                    if not relation_source_chunk_id and rr.get("source_chunk_id"):
                        relation_source_chunk_id = rr.get("source_chunk_id")
                    rels.append(item)

                alias = props.get("alias") or []
                if isinstance(alias, str):
                    alias = [alias]
                entity_label = ENTITY_TYPES.get(etype, etype)
                parts = [f"# {name}", f"类型：{entity_label}"]
                if alias:
                    parts.append(f"别名：{'、'.join(alias)}")
                if props.get("description"):
                    parts.append(f"简介：{props.get('description')}")
                attr_lines = []
                for k, v in props.items():
                    if k.startswith("attr_") and v:
                        attr_lines.append(f"{k[5:]}：{v}")
                if attr_lines:
                    parts.append("\n## 属性")
                    parts.extend(attr_lines)
                if rels:
                    parts.append(f"\n## 关联关系（{len(rels)}条）")
                    parts.extend(f"- {r}" for r in rels[:30])
                content = "\n".join(parts)
                docs.append(Document(
                    page_content=content,
                    metadata={
                        "node_id": name,
                        "entity_name": name,
                        "node_type": etype,
                        "doc_type": etype.lower(),
                        "source_chunk_id": props.get("source_chunk_id") or relation_source_chunk_id,
                        "source_text": props.get("source_text") or relation_source_text,
                        "chapter_number": props.get("chapter_number"),
                        "chapter_title": props.get("chapter_title"),
                        "content_length": len(content),
                    },
                ))
        return docs

    def build_text_chunk_documents(self) -> List[Document]:
        docs = []
        for chunk in self.text_chunks:
            props = chunk.properties
            text = props.get("text", "")
            ch_num = props.get("chapter_number", "?")
            title = props.get("chapter_title", "?")
            header = f"# 第{ch_num}回 {title}（片段{props.get('chunk_index', '?')}）"
            content = f"{header}\n\n{text}"
            docs.append(Document(
                page_content=content,
                metadata={
                    "node_id": chunk.node_id,
                    "chunk_id": chunk.node_id,
                    "source_chunk_id": chunk.node_id,
                    "source_text": text,
                    "entity_name": header,
                    "node_type": "TextChunk",
                    "chapter_number": ch_num,
                    "chapter_title": title,
                    "doc_type": "text_chunk",
                    "content_length": len(content),
                },
            ))
        return docs

    def build_history_documents(self) -> List[Document]:
        docs = []
        for etype in ENTITY_TYPES:
            part = self.build_entity_documents(etype)
            logger.info("构建 %s 文档 %s 个", etype, len(part))
            docs.extend(part)
        chunk_docs = self.build_text_chunk_documents()
        docs.extend(chunk_docs)
        self.documents = docs
        logger.info("共构建 %s 个文档（含 %s 个原文片段）", len(docs), len(chunk_docs))
        return docs

    def chunk_documents(self, chunk_size: int = 800, chunk_overlap: int = 100) -> List[Document]:
        if not self.documents:
            raise ValueError("请先构建文档")
        chunks = []
        chunk_id = 0
        for doc in self.documents:
            content = doc.page_content
            source_doc_type = doc.metadata.get("doc_type", "chunk")
            if len(content) <= chunk_size:
                chunks.append(Document(page_content=content, metadata={**doc.metadata, "chunk_id": f"{doc.metadata['node_id']}_chunk_{chunk_id}", "parent_id": doc.metadata["node_id"], "chunk_index": 0, "total_chunks": 1, "chunk_size": len(content), "source_doc_type": source_doc_type, "is_chunk": True}))
                chunk_id += 1
                continue
            total = (len(content) - 1) // (chunk_size - chunk_overlap) + 1
            for i in range(total):
                start = i * (chunk_size - chunk_overlap)
                end = min(start + chunk_size, len(content))
                chunk_content = content[start:end]
                chunks.append(Document(page_content=chunk_content, metadata={**doc.metadata, "chunk_id": f"{doc.metadata['node_id']}_chunk_{chunk_id}", "parent_id": doc.metadata["node_id"], "chunk_index": i, "total_chunks": total, "chunk_size": len(chunk_content), "source_doc_type": source_doc_type, "is_chunk": True}))
                chunk_id += 1
        self.chunks = chunks
        logger.info("文档分块完成，共 %s 个块", len(chunks))
        return chunks

    def get_statistics(self) -> Dict[str, Any]:
        stats = {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
            "total_text_chunks": len(self.text_chunks),
            "total_chapters": len(self.chapters),
            "total_communities": len(self.communities),
        }
        for etype, nodes in self.nodes_by_type.items():
            stats[f"total_{etype.lower()}s"] = len(nodes)
        return stats

    def export_triples(self, entity_names: List[str] = None, limit: int = 50) -> List[tuple]:
        triples = []
        try:
            with self.driver.session(database=self.database) as session:
                if entity_names:
                    query = """
                    UNWIND $names AS nm
                    MATCH (a)-[r]->(b)
                    WHERE coalesce(a.name, a.chunk_id, '') CONTAINS nm OR coalesce(b.name, b.chunk_id, '') CONTAINS nm
                    RETURN coalesce(a.name, a.chunk_id) AS src, type(r) AS rel, coalesce(b.name, b.chunk_id) AS tgt
                    LIMIT $limit
                    """
                    result = session.run(query, {"names": entity_names, "limit": limit})
                else:
                    query = """
                    MATCH (a)-[r]->(b)
                    RETURN coalesce(a.name, a.chunk_id) AS src, type(r) AS rel, coalesce(b.name, b.chunk_id) AS tgt
                    LIMIT $limit
                    """
                    result = session.run(query, {"limit": limit})
                for rec in result:
                    if rec["src"] and rec["tgt"]:
                        triples.append((rec["src"], rec["rel"], rec["tgt"]))
        except Exception as e:
            logger.error("导出三元组失败: %s", e)
        return triples

    def __del__(self):
        self.close()
