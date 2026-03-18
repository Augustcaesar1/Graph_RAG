"""
真正的图RAG检索模块 - 东周列国版
基于图结构的知识推理和检索，而非简单的关键词匹配
"""

import json
import logging
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum

from langchain_core.documents import Document
from neo4j import GraphDatabase

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
        logger.info("初始化东周列国图RAG检索系统...")
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
                WHERE n.name IS NOT NULL
                WITH n, COUNT { (n)--() } as degree
                RETURN labels(n) as node_labels, n.name as node_id, 
                       n.name as name, n.state as category, degree
                ORDER BY degree DESC
                LIMIT 1000
                """
                
                result = session.run(entity_query)
                for record in result:
                    node_id = record["node_id"]
                    self.entity_cache[node_id] = {
                        "labels": record["node_labels"],
                        "name": record["name"],
                        "category": record["category"],
                        "degree": record["degree"]
                    }
                
                relation_query = """
                MATCH ()-[r]->()
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
        prompt = f"""作为图数据库专家，分析以下东周列国查询的图结构意图，并将自然语言问题映射到**已有图结构**上。
        
        已知图中大致有以下节点和关系：
        - 节点标签（Labels）：
           - Person：历史人物节点（含 name、state、is_king、life_year等）
           - Event：事件/战争节点（含 name、time_start、location、attacker、defender、result等）
           - State：诸侯国节点（含 name）
         - 主要关系：
           - (Person)-[:BELONGS_TO]->(State)
           - (Person)-[:ATTACKED_IN|DEFENDED_IN|ASSISTED_ATTACK_IN|ASSISTED_DEFEND_IN]->(Event)
           - (Person)-[:FRIEND_OF|ALLY_OF|RIVAL_OF|ENEMY_OF|SIBLING|FATHER_SON|TEACHER_STUDENT|LORD_MINISTER|SPOUSE|RELATED_TO]->(Person)
        
        查询：{query}
        
        请识别：
        1. query_type：
           - entity_relation: 实体直连关系（如：孙膑和庞涓什么关系？）
           - multi_hop: 多跳推理（如：管仲和晏婴都辅佐过哪些君主？）
           - subgraph: 完整子图（如：有关齐桓公的所有事迹？）
           - path_finding: 路径/步骤查找（如：长平之战的参战双方有哪些人？）
           - clustering: 聚类查询
        
        2. source_entities：
           - 图中具体的人物、国家、事件名称（如"齐桓公"、"齐国"、"长勺之战"）
        
        3. target_entities：路径终点实体名称（如果有明确终点，如没有则填[]）
        
        4. relation_types：本次推理中希望优先跑的关系类型
        
        5. max_depth：建议深度（1-3）
        
        6. constraints：属性级限制（如时间、国别等），用字典表示。
        
        示例：
        查询："长勺之战是谁和谁打的？"
        返回JSON：
        {{
          "query_type": "subgraph",
          "source_entities": ["长勺之战"],
          "target_entities": [],
          "relation_types": ["ATTACKED_IN", "DEFENDED_IN", "ASSISTED_ATTACK_IN", "ASSISTED_DEFEND_IN"],
          "max_depth": 1,
          "constraints": {{}}
        }}

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
            if content.startswith("```json"): content = content[7:]
            elif content.startswith("```"): content = content[3:]
            if content.endswith("```"): content = content[:-3]
            content = content.strip()

            try:
                result = json.loads(content)
            except Exception:
                l = content.find("{")
                r = content.rfind("}")
                if l != -1 and r != -1 and r > l:
                    result = json.loads(content[l : r + 1])
                else:
                    raise ValueError(f"LLM返回非JSON内容: {content[:120]}")
            return GraphQuery(
                query_type=QueryType(result.get("query_type", "subgraph")),
                source_entities=result.get("source_entities", []),
                target_entities=result.get("target_entities", []),
                relation_types=result.get("relation_types", []),
                max_depth=result.get("max_depth", 2),
                max_nodes=50
            )
        except Exception as e:
            logger.error(f"查询意图理解失败: {e}")
            return GraphQuery(QueryType.SUBGRAPH, [query], max_depth=2)
    
    def multi_hop_traversal(self, graph_query: GraphQuery) -> List[GraphPath]:
        logger.info(f"多跳遍历: {graph_query.source_entities}")
        paths = []
        if not self.driver: return paths
        try:
            with self.driver.session() as session:
                source_entities = graph_query.source_entities
                target_keywords = graph_query.target_entities or []
                max_depth = graph_query.max_depth
                
                target_filter = ""
                if target_keywords:
                    target_filter = " AND ANY(kw IN $target_keywords WHERE target.name CONTAINS kw OR target.state CONTAINS kw) "
                
                cypher = f"""
                UNWIND $source_entities as sname
                MATCH (source)
                WHERE source.name CONTAINS sname OR sname CONTAINS source.name
                MATCH path = (source)-[*1..{max_depth}]-(target)
                WHERE NOT source = target {target_filter}
                WITH path, source, target, length(path) as path_len, relationships(path) as rels, nodes(path) as path_nodes
                WITH path, source, target, path_len, rels, path_nodes,
                     (1.0 / path_len) + 
                     (CASE WHEN ANY(r IN rels WHERE type(r) IN $relation_types) THEN 0.3 ELSE 0.0 END) as relevance
                ORDER BY relevance DESC LIMIT 20
                RETURN path, source, target, path_len, rels, path_nodes, relevance
                """
                
                params = {"source_entities": source_entities, "relation_types": graph_query.relation_types or []}
                if target_keywords: params["target_keywords"] = target_keywords
                
                for record in session.run(cypher, params):
                    p = self._parse_neo4j_path(record)
                    if p: paths.append(p)
        except Exception as e:
            logger.error(f"多跳遍历失败: {e}")
        return paths
    
    def extract_knowledge_subgraph(self, graph_query: GraphQuery) -> KnowledgeSubgraph:
        logger.info(f"提取知识子图: {graph_query.source_entities}")
        if not self.driver: return self._fallback_subgraph_extraction(graph_query)
        try:
            with self.driver.session() as session:
                cypher = f"""
                UNWIND $source_entities as sname
                MATCH (source)
                WHERE source.name CONTAINS sname OR sname CONTAINS source.name
                MATCH path = (source)-[*1..{graph_query.max_depth}]-(neighbor)
                WITH source, neighbor, relationships(path) as path_rels
                UNWIND path_rels as rel
                WITH source, collect(DISTINCT neighbor) as neighbors, collect(DISTINCT rel) as all_rels
                WITH source, neighbors, all_rels, size(neighbors) as nc, size(all_rels) as rc
                RETURN source, neighbors[0..$max_nodes] as nodes, all_rels[0..$max_nodes] as rels,
                       {{ node_count: nc, relationship_count: rc, density: CASE WHEN nc > 1 THEN toFloat(rc)/(nc*(nc-1)/2) ELSE 0.0 END }} as metrics
                """
                record = session.run(cypher, {"source_entities": graph_query.source_entities, "max_nodes": graph_query.max_nodes}).single()
                if record: return self._build_knowledge_subgraph(record)
        except Exception as e:
            logger.error(f"子图提取失败: {e}")
        return self._fallback_subgraph_extraction(graph_query)

    def _fallback_subgraph_extraction(self, q):
        return KnowledgeSubgraph([], [], [], {}, [])
    
    def graph_structure_reasoning(self, subgraph: KnowledgeSubgraph, query: str) -> List[str]:
        return []
    
    def adaptive_query_planning(self, query: str) -> List[GraphQuery]:
        return [GraphQuery(QueryType.SUBGRAPH, [query])]
    
    def graph_rag_search(self, query: str, top_k: int = 5) -> List[Document]:
        logger.info(f"开始图RAG检索: {query}")
        if not self.driver: return []
        graph_query = self.understand_graph_query(query)
        results = []
        try:
            if graph_query.query_type in [QueryType.MULTI_HOP, QueryType.PATH_FINDING, QueryType.ENTITY_RELATION]:
                paths = self.multi_hop_traversal(graph_query)
                results.extend(self._paths_to_documents(paths, query))
            if graph_query.query_type in [QueryType.SUBGRAPH, QueryType.CLUSTERING] or not results:
                subgraph = self.extract_knowledge_subgraph(graph_query)
                results.extend(self._subgraph_to_documents(subgraph, [], query))
            
            results = sorted(results, key=lambda x: x.metadata.get("relevance_score", 0.0), reverse=True)
            return results[:top_k]
        except Exception as e:
            logger.error(f"图检索搜索异常: {e}")
            return []
            
    # -- 辅助 --
    def _parse_neo4j_path(self, record):
        try:
            path_nodes = []
            for n in record["path_nodes"]:
                path_nodes.append({"id": n.get("nodeId", ""), "name": n.get("name", ""), "labels": list(n.labels), "properties": dict(n)})
            
            rels = []
            for r in record["rels"]:
                rels.append({"type": r.type, "source": dict(r.start_node).get("name", ""), "target": dict(r.end_node).get("name", "")})
            
            return GraphPath(path_nodes, rels, record["path_len"], record["relevance"], "multi_hop")
        except:
            return None

    def _build_knowledge_subgraph(self, record):
        try:
            cn = [dict(record["source"])]
            con_n = [dict(n) for n in record["nodes"]]
            rels = []
            for r in record["rels"]:
                rels.append({"type": r.type, "source": dict(r.start_node).get("name", ""), "target": dict(r.end_node).get("name", "")})
            return KnowledgeSubgraph(cn, con_n, rels, record["metrics"], [])
        except:
            return KnowledgeSubgraph([], [], [], {}, [])

    def _paths_to_documents(self, paths, query):
        docs = []
        for path in paths:
            desc = self._build_path_description(path)
            edges = []
            for j, rel in enumerate(path.relationships):
                if rel.get("source") and rel.get("target"):
                    edges.append({
                        "source": rel["source"], "relation": rel["type"], "target": rel["target"],
                        "source_type": path.nodes[j].get("labels", ["Concept"])[0] if path.nodes[j].get("labels") else "Concept",
                        "target_type": path.nodes[j+1].get("labels", ["Concept"])[0] if j+1 < len(path.nodes) and path.nodes[j+1].get("labels") else "Concept"
                    })
            docs.append(Document(page_content=desc, metadata={
                "search_type": "graph_path", "relevance_score": path.relevance_score,
                "recipe_name": path.nodes[0].get("name", "图路径"), "subgraph_edges": edges
            }))
        return docs

    def _subgraph_to_documents(self, subgraph, chains, query):
        if not subgraph.central_nodes: return []
        desc = self._build_subgraph_description(subgraph)
        edges = []
        for rel in subgraph.relationships:
            if rel.get("source") and rel.get("target"):
                edges.append({"source": rel["source"], "relation": rel["type"], "target": rel["target"], "source_type": "Concept", "target_type": "Concept"})
        
        return [Document(page_content=desc, metadata={
            "search_type": "knowledge_subgraph", "relevance_score": 0.8,
            "recipe_name": subgraph.central_nodes[0].get("name", "知识子图"),
            "subgraph_edges": edges
        })]

    def _build_path_description(self, path):
        if not path.nodes: return ""
        parts = []
        ZH_MAP = {"BELONGS_TO":"效力于", "ATTACKED_IN":"主攻", "DEFENDED_IN":"主守", "FRIEND_OF":"好友", "RIVAL_OF":"对手"}
        for i, n in enumerate(path.nodes):
            parts.append(n.get("name", "节点"))
            if i < len(path.relationships):
                rt = path.relationships[i]["type"]
                parts.append(f" --({ZH_MAP.get(rt, rt)})--> ")
        return "".join(parts)

    def _build_subgraph_description(self, subgraph):
        cnames = [n.get("name", "") for n in subgraph.central_nodes]
        parts = [f"### 知识子图核心：{', '.join(cnames)}"]
        
        by_type = defaultdict(list)
        for n in subgraph.central_nodes + subgraph.connected_nodes:
            lbl = n.get("labels", ["Concept"])
            l = lbl[0] if lbl else "Concept"
            by_type[l].append(n)
            
        if "Person" in by_type:
            parts.append("\n人物:")
            for n in by_type["Person"]:
                parts.append(f"- {n.get('name', '')} ({n.get('state', '')})")
        if "Event" in by_type:
            parts.append("\n事件:")
            for n in by_type["Event"]:
                parts.append(f"- {n.get('name', '')}")
                
        if subgraph.relationships:
            ZH_MAP = {"BELONGS_TO":"效力于", "ATTACKED_IN":"主攻", "DEFENDED_IN":"主守", "FRIEND_OF":"好友", "RIVAL_OF":"对手", "ALLY_OF":"盟友", "ENEMY_OF":"敌人"}
            parts.append("\n关系:")
            for r in subgraph.relationships[:20]:
                rt = ZH_MAP.get(r['type'], r['type'])
                parts.append(f"- {r['source']} <{rt}> {r['target']}")
                
        return "\n".join(parts)
