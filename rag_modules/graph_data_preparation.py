"""
东周列国图数据库数据准备模块
从Neo4j读取 Person / Event / State 节点，转换为 RAG 文档
"""

import logging
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from neo4j import GraphDatabase
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

@dataclass
class GraphNode:
    """图节点数据结构"""
    node_id: str
    labels: List[str]
    name: str
    properties: Dict[str, Any]

@dataclass
class GraphRelation:
    """图关系数据结构"""
    start_node_id: str
    end_node_id: str
    relation_type: str
    properties: Dict[str, Any]


class GraphDataPreparationModule:
    """图数据库数据准备模块 - 从Neo4j读取东周列国数据并转换为文档"""

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None
        self.documents: List[Document] = []
        self.chunks: List[Document] = []

        # 节点列表（对应原来的 recipes/ingredients/steps）
        self.persons: List[GraphNode] = []
        self.events:  List[GraphNode] = []
        self.states:  List[GraphNode] = []

        # 兼容旧接口（app.py 统计使用）
        self.recipes    = self.persons
        self.ingredients= self.events

        self._connect()

    def _connect(self):
        """建立Neo4j连接"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                database=self.database
            )
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                if result.single():
                    logger.info("Neo4j 连接测试成功")
        except Exception as e:
            logger.error(f"连接Neo4j失败: {e}")
            raise

    def close(self):
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()

    # ─────────────────────────────────────────────────────────────
    # 加载图数据
    # ─────────────────────────────────────────────────────────────
    def load_graph_data(self) -> Dict[str, Any]:
        """从Neo4j加载人物、事件、国家节点"""
        logger.info("从Neo4j加载东周列国图数据...")

        with self.driver.session() as session:
            # 加载 Person 节点
            result = session.run("""
                MATCH (p:Person)
                RETURN p.name as name, labels(p) as labels, properties(p) as props
                ORDER BY p.name
            """)
            self.persons = []
            for rec in result:
                self.persons.append(GraphNode(
                    node_id=rec["name"],
                    labels=rec["labels"],
                    name=rec["name"],
                    properties=dict(rec["props"])
                ))
            logger.info(f"加载了 {len(self.persons)} 个 Person 节点")

            # 加载 Event 节点
            result = session.run("""
                MATCH (e:Event)
                RETURN e.event_id as eid, e.name as name, labels(e) as labels, properties(e) as props
                ORDER BY e.time_start
            """)
            self.events = []
            for rec in result:
                self.events.append(GraphNode(
                    node_id=rec["eid"] or rec["name"],
                    labels=rec["labels"],
                    name=rec["name"],
                    properties=dict(rec["props"])
                ))
            logger.info(f"加载了 {len(self.events)} 个 Event 节点")

            # 加载 State 节点
            result = session.run("""
                MATCH (s:State)
                RETURN s.name as name, labels(s) as labels, properties(s) as props
                ORDER BY s.name
            """)
            self.states = []
            for rec in result:
                self.states.append(GraphNode(
                    node_id=rec["name"],
                    labels=rec["labels"],
                    name=rec["name"],
                    properties=dict(rec["props"])
                ))
            logger.info(f"加载了 {len(self.states)} 个 State 节点")

        # 同步兼容别名
        self.recipes     = self.persons
        self.ingredients = self.events

        return {
            'persons': len(self.persons),
            'events':  len(self.events),
            'states':  len(self.states)
        }

    # ─────────────────────────────────────────────────────────────
    # 构建人物文档
    # ─────────────────────────────────────────────────────────────
    def build_person_documents(self) -> List[Document]:
        """为每个Person构建RAG文档（包含其关系和参与事件）"""
        logger.info("构建人物文档...")
        documents = []

        with self.driver.session() as session:
            for person in self.persons:
                pname = person.name
                props = person.properties

                # 获取该人物的关系
                rel_result = session.run("""
                    MATCH (p:Person {name: $name})-[r]->(other)
                    RETURN type(r) as rtype, r.label as rlabel, other.name as other_name, labels(other) as other_labels
                    LIMIT 30
                """, {"name": pname})

                relations = []
                for rr in rel_result:
                    other = rr["other_name"]
                    label = rr["rlabel"] or rr["rtype"]
                    other_labels = rr["other_labels"] or []
                    if "State" in other_labels:
                        relations.append(f"效力于：{other}")
                    elif "Event" in other_labels:
                        rel_map = {
                            "ATTACKED_IN": "主攻方参与战争",
                            "DEFENDED_IN": "主守方参与战争",
                            "ASSISTED_ATTACK_IN": "协助进攻",
                            "ASSISTED_DEFEND_IN": "协助防守"
                        }
                        rel_desc = rel_map.get(rr["rtype"], "参与")
                        relations.append(f"{rel_desc}：{other}")
                    else:
                        relations.append(f"{label}：{other}")

                # 获取参与的事件
                event_result = session.run("""
                    MATCH (p:Person {name: $name})-[r]->(e:Event)
                    RETURN e.name as ename, e.time_start as ts, e.result as result, type(r) as rtype
                    ORDER BY e.time_start
                    LIMIT 10
                """, {"name": pname})
                events_info = []
                for er in event_result:
                    role_map = {
                        "ATTACKED_IN": "主攻",
                        "DEFENDED_IN": "主守",
                        "ASSISTED_ATTACK_IN": "助攻",
                        "ASSISTED_DEFEND_IN": "助守"
                    }
                    role = role_map.get(er["rtype"], "参与")
                    ts = er["ts"]
                    year_str = f"（约公元前{abs(ts)}年）" if ts and ts < 0 else (f"（{ts}年）" if ts else "")
                    events_info.append(f"  - {er['ename']}{year_str}，以{role}方参战，结果：{er['result'] or '不详'}")

                # 构建文档
                life_year = props.get("life_year")
                year_display = f"约公元前{abs(life_year)}年" if life_year and life_year < 0 else str(life_year or "不详")
                is_king = "国君" if props.get("is_king") == "是" else "臣子/将领"

                content_parts = [f"# {pname}"]
                content_parts.append(f"\n**人物简介**")
                content_parts.append(f"姓名：{pname}（姓{props.get('xing','')} 氏{props.get('shi','')} 名{props.get('ming','')}）")
                content_parts.append(f"所属诸侯国：{props.get('state', '不详')}")
                content_parts.append(f"身份：{is_king}")
                content_parts.append(f"生活时代：{year_display}")
                content_parts.append(f"工作时间：{props.get('work_time', '不详')}")
                if props.get("note"):
                    content_parts.append(f"备注：{props['note']}")

                if relations:
                    content_parts.append(f"\n**人物关系**")
                    for rel in relations[:15]:
                        content_parts.append(f"  - {rel}")

                if events_info:
                    content_parts.append(f"\n**参与历史事件**")
                    content_parts.extend(events_info)

                full_content = "\n".join(content_parts)

                doc = Document(
                    page_content=full_content,
                    metadata={
                        "node_id":      pname,
                        "entity_name":  pname,
                        "recipe_name":  pname,   # 兼容旧接口
                        "node_type":    "Person",
                        "state":        props.get("state", ""),
                        "is_king":      props.get("is_king", ""),
                        "life_year":    props.get("life_year", ""),
                        "doc_type":     "person",
                        "content_length": len(full_content)
                    }
                )
                documents.append(doc)

        logger.info(f"成功构建 {len(documents)} 个人物文档")
        return documents

    # ─────────────────────────────────────────────────────────────
    # 构建事件文档
    # ─────────────────────────────────────────────────────────────
    def build_event_documents(self) -> List[Document]:
        """为每个Event构建RAG文档"""
        logger.info("构建事件文档...")
        documents = []

        with self.driver.session() as session:
            for event in self.events:
                eid   = event.node_id
                ename = event.name
                props = event.properties

                # 获取参与人物
                persons_result = session.run("""
                    MATCH (p:Person)-[r]->(e:Event {name: $name})
                    RETURN p.name as pname, p.state as pstate, type(r) as rtype
                    LIMIT 20
                """, {"name": ename})

                attackers, defenders, helpers = [], [], []
                for pr in persons_result:
                    rt = pr["rtype"]
                    entry = f"{pr['pname']}（{pr['pstate'] or ''}）"
                    if rt == "ATTACKED_IN":
                        attackers.append(entry)
                    elif rt == "DEFENDED_IN":
                        defenders.append(entry)
                    else:
                        helpers.append(entry)

                ts = props.get("time_start")
                te = props.get("time_end")
                year_str = ""
                if ts:
                    year_str = f"约公元前{abs(ts)}年" if ts < 0 else f"{ts}年"
                    if te:
                        ye = f"公元前{abs(te)}年" if te < 0 else f"{te}年"
                        year_str += f"至{ye}"

                content_parts = [f"# {ename}"]
                content_parts.append(f"\n**战争基本信息**")
                content_parts.append(f"事件编号：{eid}")
                content_parts.append(f"时间：{year_str or '不详'}")
                content_parts.append(f"地点：{props.get('location', '不详')}")
                content_parts.append(f"事件类型：{props.get('event_type', '战争')}")

                content_parts.append(f"\n**交战双方**")
                content_parts.append(f"主攻方：{props.get('attacker', '不详')}")
                content_parts.append(f"主守方：{props.get('defender', '不详')}")
                if props.get("atk_help"):
                    content_parts.append(f"助攻方：{props['atk_help']}")
                if props.get("def_help"):
                    content_parts.append(f"助守方：{props['def_help']}")

                if attackers:
                    content_parts.append(f"\n**参战人物（进攻方）**：{', '.join(attackers)}")
                if defenders:
                    content_parts.append(f"**参战人物（防守方）**：{', '.join(defenders)}")
                if helpers:
                    content_parts.append(f"**其他参战人物**：{', '.join(helpers)}")

                content_parts.append(f"\n**战役详情**")
                content_parts.append(f"起因：{props.get('cause', '不详')}")
                content_parts.append(f"结果：{props.get('result', '不详')}")

                if props.get("atk_force"):
                    content_parts.append(f"主攻方兵力：{props['atk_force']}")
                if props.get("def_force"):
                    content_parts.append(f"主守方兵力：{props['def_force']}")
                if props.get("atk_loss"):
                    content_parts.append(f"主攻方伤亡：{props['atk_loss']}")
                if props.get("def_loss"):
                    content_parts.append(f"主守方伤亡：{props['def_loss']}")
                if props.get("content"):
                    content_parts.append(f"\n**详细记载**\n{props['content']}")

                full_content = "\n".join(content_parts)

                doc = Document(
                    page_content=full_content,
                    metadata={
                        "node_id":     eid,
                        "entity_name": ename,
                        "recipe_name": ename,   # 兼容旧接口
                        "node_type":   "Event",
                        "location":    props.get("location", ""),
                        "time_start":  props.get("time_start", ""),
                        "result":      props.get("result", ""),
                        "doc_type":    "event",
                        "content_length": len(full_content)
                    }
                )
                documents.append(doc)

        logger.info(f"成功构建 {len(documents)} 个事件文档")
        return documents

    # ─────────────────────────────────────────────────────────────
    # 统一构建（原接口保留）
    # ─────────────────────────────────────────────────────────────
    def build_recipe_documents(self) -> List[Document]:
        """兼容旧接口，构建全部文档（人物 + 事件）"""
        person_docs = self.build_person_documents()
        event_docs  = self.build_event_documents()
        self.documents = person_docs + event_docs
        logger.info(f"共构建 {len(self.documents)} 个文档（{len(person_docs)} 人物 + {len(event_docs)} 事件）")
        return self.documents

    # ─────────────────────────────────────────────────────────────
    # 文档分块
    # ─────────────────────────────────────────────────────────────
    def chunk_documents(self, chunk_size: int = 600, chunk_overlap: int = 60) -> List[Document]:
        """对文档进行分块"""
        logger.info(f"文档分块，块大小: {chunk_size}, 重叠: {chunk_overlap}")
        if not self.documents:
            raise ValueError("请先构建文档")

        chunks = []
        chunk_id = 0

        for doc in self.documents:
            content = doc.page_content

            if len(content) <= chunk_size:
                chunk = Document(
                    page_content=content,
                    metadata={
                        **doc.metadata,
                        "chunk_id":    f"{doc.metadata['node_id']}_chunk_{chunk_id}",
                        "parent_id":   doc.metadata["node_id"],
                        "chunk_index": 0,
                        "total_chunks":1,
                        "chunk_size":  len(content),
                        "doc_type":    "chunk"
                    }
                )
                chunks.append(chunk)
                chunk_id += 1
            else:
                # 按二级标题分块
                sections = content.split('\n## ')
                if len(sections) <= 1:
                    sections = content.split('\n**')
                if len(sections) <= 1:
                    # 按长度强制分块
                    total_chunks = (len(content) - 1) // (chunk_size - chunk_overlap) + 1
                    for i in range(total_chunks):
                        start = i * (chunk_size - chunk_overlap)
                        end   = min(start + chunk_size, len(content))
                        chunk_content = content[start:end]
                        chunk = Document(
                            page_content=chunk_content,
                            metadata={
                                **doc.metadata,
                                "chunk_id":    f"{doc.metadata['node_id']}_chunk_{chunk_id}",
                                "parent_id":   doc.metadata["node_id"],
                                "chunk_index": i,
                                "total_chunks":total_chunks,
                                "chunk_size":  len(chunk_content),
                                "doc_type":    "chunk"
                            }
                        )
                        chunks.append(chunk)
                        chunk_id += 1
                else:
                    total_chunks = len(sections)
                    for i, section in enumerate(sections):
                        chunk_content = section if i == 0 else f"## {section}"
                        chunk = Document(
                            page_content=chunk_content,
                            metadata={
                                **doc.metadata,
                                "chunk_id":    f"{doc.metadata['node_id']}_chunk_{chunk_id}",
                                "parent_id":   doc.metadata["node_id"],
                                "chunk_index": i,
                                "total_chunks":total_chunks,
                                "chunk_size":  len(chunk_content),
                                "doc_type":    "chunk",
                                "section_title": section.split('\n')[0][:30] if i > 0 else "主标题"
                            }
                        )
                        chunks.append(chunk)
                        chunk_id += 1

        self.chunks = chunks
        logger.info(f"文档分块完成，共 {len(chunks)} 个块")
        return chunks

    # ─────────────────────────────────────────────────────────────
    # 统计 & 导出
    # ─────────────────────────────────────────────────────────────
    def get_statistics(self) -> Dict[str, Any]:
        stats = {
            'total_recipes':        len(self.persons),
            'total_ingredients':    len(self.events),
            'total_cooking_steps':  len(self.states),
            'total_documents':      len(self.documents),
            'total_chunks':         len(self.chunks),
            # 东周专用键
            'total_persons':        len(self.persons),
            'total_events':         len(self.events),
            'total_states':         len(self.states),
        }

        if self.documents:
            types = {}
            states_cnt = {}
            for doc in self.documents:
                t = doc.metadata.get('doc_type', '未知')
                types[t] = types.get(t, 0) + 1
                s = doc.metadata.get('state', '未知')
                if s:
                    states_cnt[s] = states_cnt.get(s, 0) + 1
            stats['doc_types'] = types
            stats['state_distribution'] = states_cnt

        return stats

    def export_triples(self, recipe_names: List[str] = None, limit: int = 50) -> List[tuple]:
        """导出三元组（兼容旧接口），用于知识图谱可视化"""
        triples = []
        try:
            with self.driver.session() as session:
                if recipe_names:
                    # 按实体名查询相关三元组
                    query = """
                    UNWIND $names as nm
                    MATCH (a)-[r]->(b)
                    WHERE (a.name CONTAINS nm OR b.name CONTAINS nm)
                      AND NOT type(r) IN ['BELONGS_TO']
                    RETURN a.name as src, type(r) as rel, b.name as tgt
                    LIMIT $limit
                    """
                    result = session.run(query, {"names": recipe_names, "limit": limit})
                else:
                    query = """
                    MATCH (a)-[r]->(b)
                    WHERE NOT type(r) = 'BELONGS_TO'
                    RETURN a.name as src, type(r) as rel, b.name as tgt
                    LIMIT $limit
                    """
                    result = session.run(query, {"limit": limit})

                for rec in result:
                    if rec["src"] and rec["tgt"]:
                        triples.append((rec["src"], rec["rel"], rec["tgt"]))
        except Exception as e:
            logger.error(f"导出三元组失败: {e}")
        logger.info(f"导出三元组完成，共 {len(triples)} 条")
        return triples

    def __del__(self):
        self.close()