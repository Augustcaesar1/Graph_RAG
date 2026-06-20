"""
封神演义分层社团检测模块
基于Neo4j中的人物、教派、法宝、阵法、事件、文本片段构建加权图，并写回 community_l1/l2/l3
"""

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Any, Tuple

from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

try:
    import networkx as nx
except ImportError:
    nx = None


@dataclass
class CommunitySummary:
    level: int
    community_id: str
    size: int
    node_names: List[str]
    node_types: List[str]
    top_organizations: List[str]
    top_relations: List[str]
    time_span: str
    title: str
    summary: str


class CommunityDetectionModule:
    def __init__(self, config, driver=None):
        self.config = config
        if driver is not None:
            self.driver = driver
        else:
            self.driver = GraphDatabase.driver(
                config.neo4j_uri,
                auth=(config.neo4j_user, config.neo4j_password),
            )

    def close(self):
        if self.driver:
            self.driver.close()

    def run(self) -> Dict[str, int]:
        if nx is None:
            logger.warning("networkx 未安装，跳过社团检测")
            return {"l1": 0, "l2": 0, "l3": 0}

        weighted_graph, node_metadata = self._build_weighted_graph()
        if weighted_graph.number_of_nodes() == 0:
            logger.warning("图为空，跳过社团检测")
            return {"l1": 0, "l2": 0, "l3": 0}

        l1 = self._detect_level1(weighted_graph)
        l2 = self._detect_higher_level(weighted_graph, l1, level=2)
        l3 = self._detect_higher_level(weighted_graph, l2, level=3)

        self._clear_existing_communities()
        self._write_node_communities(node_metadata, l1, "community_l1")
        self._write_node_communities(node_metadata, l2, "community_l2")
        self._write_node_communities(node_metadata, l3, "community_l3")

        self._write_community_nodes(node_metadata, l1, 1)
        self._write_community_nodes(node_metadata, l2, 2)
        self._write_community_nodes(node_metadata, l3, 3)

        return {
            "l1": len(set(l1.values())),
            "l2": len(set(l2.values())),
            "l3": len(set(l3.values())),
        }

    def _build_weighted_graph(self):
        graph = nx.Graph()
        node_metadata: Dict[str, Dict[str, Any]] = {}

        with self.driver.session(database=self.config.neo4j_database) as session:
            node_query = """
            MATCH (n)
            WHERE any(label IN labels(n) WHERE label IN ['Person', 'Organization', 'Event', 'Period', 'TextChunk'])
            RETURN elementId(n) AS node_key,
                   coalesce(n.name, n.chunk_id, n.title, elementId(n)) AS name,
                   labels(n) AS labels,
                   properties(n) AS props
            LIMIT $limit
            """
            for record in session.run(node_query, {"limit": self.config.community_max_nodes}):
                node_key = record["node_key"]
                props = dict(record["props"])
                labels = list(record["labels"])
                node_metadata[node_key] = {
                    "node_key": node_key,
                    "name": record["name"],
                    "labels": labels,
                    "organization": props.get("organization", ""),
                    "period": props.get("period", props.get("title", "")),
                    "time_start": props.get("time_start"),
                    "time_end": props.get("time_end"),
                    "year": props.get("year"),
                    "chapter": props.get("chapter", ""),
                    "section": props.get("section", ""),
                }
                graph.add_node(node_key)

            edge_query = """
            MATCH (a)-[r]->(b)
            WHERE any(label IN labels(a) WHERE label IN ['Person', 'Organization', 'Event', 'Period', 'TextChunk'])
              AND any(label IN labels(b) WHERE label IN ['Person', 'Organization', 'Event', 'Period', 'TextChunk'])
            RETURN elementId(a) AS source_key,
                   elementId(b) AS target_key,
                   type(r) AS rel_type,
                   properties(r) AS props
            """
            for record in session.run(edge_query):
                source_key = record["source_key"]
                target_key = record["target_key"]
                if source_key not in node_metadata or target_key not in node_metadata:
                    continue
                weight = self._relation_weight(record["rel_type"], dict(record["props"]))
                self._add_weighted_edge(graph, source_key, target_key, weight, record["rel_type"])

            cooccurrence_query = """
            MATCH (tc:TextChunk)-[:MENTIONS_PERSON|MENTIONS_EVENT|MENTIONS_ORG|DESCRIBES_EVENT]->(n)
            WITH tc, collect(DISTINCT elementId(n)) AS nodes
            WHERE size(nodes) > 1
            RETURN elementId(tc) AS chunk_key, nodes
            LIMIT $limit
            """
            for record in session.run(cooccurrence_query, {"limit": self.config.community_max_nodes}):
                nodes = [node_key for node_key in record["nodes"] if node_key in node_metadata]
                for left, right in combinations(nodes, 2):
                    self._add_weighted_edge(graph, left, right, 1.0, "CO_OCCUR")

        return graph, node_metadata

    def _relation_weight(self, rel_type: str, props: Dict[str, Any]) -> float:
        base_weights = {
            "MEMBER_OF": 3.5,
            "BELONGS_TO": 3.0,
            "PARTICIPATES_IN": 3.0,
            "LEADS": 3.5,
            "INITIATES": 3.5,
            "RELATED_TO": 2.5,
            "MENTIONS_PERSON": 1.5,
            "MENTIONS_EVENT": 1.5,
            "MENTIONS_ORG": 1.5,
            "DESCRIBES_EVENT": 2.0,
            "IN_COMMUNITY": 1.0,
        }
        confidence = props.get("confidence", 1.0)
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 1.0
        return max(0.5, base_weights.get(rel_type, 1.2) * confidence)

    def _add_weighted_edge(self, graph, source_key: str, target_key: str, weight: float, relation_type: str):
        if source_key == target_key:
            return
        if graph.has_edge(source_key, target_key):
            graph[source_key][target_key]["weight"] += weight
            graph[source_key][target_key]["relations"].append(relation_type)
        else:
            graph.add_edge(source_key, target_key, weight=weight, relations=[relation_type])

    def _detect_level1(self, graph):
        communities = nx.algorithms.community.greedy_modularity_communities(graph, weight="weight")
        return self._community_mapping(communities, prefix="L1", min_size=self.config.community_level1_min_size)

    def _detect_higher_level(self, graph, lower_mapping: Dict[str, str], level: int):
        reverse_map = defaultdict(list)
        for node_key, community_id in lower_mapping.items():
            reverse_map[community_id].append(node_key)

        super_graph = nx.Graph()
        for community_id in reverse_map:
            super_graph.add_node(community_id)

        for left, right, data in graph.edges(data=True):
            left_group = lower_mapping.get(left)
            right_group = lower_mapping.get(right)
            if not left_group or not right_group or left_group == right_group:
                continue
            weight = float(data.get("weight", 1.0))
            if super_graph.has_edge(left_group, right_group):
                super_graph[left_group][right_group]["weight"] += weight
            else:
                super_graph.add_edge(left_group, right_group, weight=weight)

        if super_graph.number_of_nodes() == 0 or super_graph.number_of_edges() == 0:
            return {node_key: f"L{level}_0" for node_key in lower_mapping}

        communities = nx.algorithms.community.greedy_modularity_communities(super_graph, weight="weight")
        super_mapping = self._community_mapping(
            communities,
            prefix=f"L{level}",
            min_size=getattr(self.config, f"community_level{level}_min_size", 1),
        )

        expanded = {}
        for node_key, lower_group in lower_mapping.items():
            expanded[node_key] = super_mapping.get(lower_group, f"L{level}_0")
        return expanded

    def _community_mapping(self, communities, prefix: str, min_size: int):
        mapping = {}
        fallback_index = 0
        for index, community_nodes in enumerate(communities):
            node_list = list(community_nodes)
            if len(node_list) < min_size:
                for node_key in node_list:
                    mapping[node_key] = f"{prefix}_small_{fallback_index}"
                    fallback_index += 1
                continue
            community_id = f"{prefix}_{index}"
            for node_key in node_list:
                mapping[node_key] = community_id
        return mapping

    def _clear_existing_communities(self):
        with self.driver.session(database=self.config.neo4j_database) as session:
            session.run("MATCH ()-[r:IN_COMMUNITY]->(:Community) DELETE r")
            session.run("MATCH (c:Community) DETACH DELETE c")

    def _write_node_communities(self, node_metadata: Dict[str, Dict[str, Any]], mapping: Dict[str, str], property_name: str):
        with self.driver.session(database=self.config.neo4j_database) as session:
            rows = []
            for node_key, community_id in mapping.items():
                meta = node_metadata.get(node_key, {})
                rows.append({
                    "node_key": node_key,
                    "community_id": community_id,
                    "name": meta.get("name", ""),
                })
            session.run(
                f"""
                UNWIND $rows AS row
                MATCH (n) WHERE elementId(n) = row.node_key
                SET n.{property_name} = row.community_id
                """,
                {"rows": rows},
            )

    def _write_community_nodes(self, node_metadata: Dict[str, Dict[str, Any]], mapping: Dict[str, str], level: int):
        grouped = defaultdict(list)
        for node_key, community_id in mapping.items():
            grouped[community_id].append(node_metadata[node_key])

        summaries = [self._build_summary(level, community_id, members) for community_id, members in grouped.items()]

        with self.driver.session(database=self.config.neo4j_database) as session:
            session.run(
                """
                UNWIND $rows AS row
                MERGE (c:Community {community_id: row.community_id})
                SET c.level = row.level,
                    c.title = row.title,
                    c.summary = row.summary,
                    c.size = row.size,
                    c.top_organizations = row.top_organizations,
                    c.top_relations = row.top_relations,
                    c.time_span = row.time_span,
                    c.node_names = row.node_names
                WITH c, row
                MATCH (n) WHERE elementId(n) IN row.node_keys
                MERGE (n)-[:IN_COMMUNITY]->(c)
                """,
                {
                    "rows": [
                        {
                            "community_id": summary.community_id,
                            "level": summary.level,
                            "title": summary.title,
                            "summary": summary.summary,
                            "size": summary.size,
                            "top_organizations": summary.top_organizations,
                            "top_relations": summary.top_relations,
                            "time_span": summary.time_span,
                            "node_names": summary.node_names[:20],
                            "node_keys": [member["node_key"] for member in grouped[summary.community_id]],
                        }
                        for summary in summaries
                    ]
                },
            )

    def _build_summary(self, level: int, community_id: str, members: List[Dict[str, Any]]) -> CommunitySummary:
        node_names = [member.get("name", "") for member in members if member.get("name")]
        node_types = [member.get("labels", ["Unknown"])[0] for member in members]
        organizations = [member.get("organization", "") for member in members if member.get("organization")]
        organizations.extend(member.get("name", "") for member in members if "Organization" in member.get("labels", []))
        organization_counter = Counter(org for org in organizations if org)
        top_organizations = [name for name, _ in organization_counter.most_common(3)]

        years = []
        for member in members:
            for key in ("time_start", "time_end", "year"):
                value = member.get(key)
                if isinstance(value, int):
                    years.append(value)
                elif isinstance(value, str) and value.isdigit():
                    years.append(int(value))
        if years:
            time_span = f"{min(years)} 至 {max(years)}"
        else:
            time_span = "时间未详"

        type_counter = Counter(node_types)
        dominant_type = type_counter.most_common(1)[0][0] if type_counter else "节点群"
        period_counter = Counter(member.get("period", "") for member in members if member.get("period"))
        dominant_period = period_counter.most_common(1)[0][0] if period_counter else ""

        title_core = dominant_period or ("/".join(top_organizations) if top_organizations else dominant_type)
        title = f"L{level}社团：{title_core}"
        summary = (
            f"该社团包含{len(members)}个节点，"
            f"以{dominant_type}为主，"
            f"核心成员包括：{'、'.join(node_names[:8]) or '未详'}。"
            f"核心组织：{'、'.join(top_organizations) if top_organizations else '未识别'}。"
            f"历史阶段：{dominant_period or '未识别'}。"
        )

        return CommunitySummary(
            level=level,
            community_id=community_id,
            size=len(members),
            node_names=node_names,
            node_types=node_types,
            top_organizations=top_organizations,
            top_relations=[],
            time_span=time_span,
            title=title,
            summary=summary,
        )
