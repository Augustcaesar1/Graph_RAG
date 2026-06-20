"""
图索引模块
实现实体和关系的键值对结构 (K,V)
K: 索引键（简短词汇或短语）
V: 详细描述段落（包含相关文本片段）
"""

import json
import logging
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class EntityKeyValue:
    """实体键值对"""
    entity_name: str
    index_keys: List[str]
    value_content: str
    entity_type: str
    metadata: Dict[str, Any]


@dataclass
class RelationKeyValue:
    """关系键值对"""
    relation_id: str
    index_keys: List[str]
    value_content: str
    relation_type: str
    source_entity: str
    target_entity: str
    metadata: Dict[str, Any]


class GraphIndexingModule:
    def __init__(self, config, llm_client):
        self.config = config
        self.llm_client = llm_client
        self.entity_kv_store: Dict[str, EntityKeyValue] = {}
        self.relation_kv_store: Dict[str, RelationKeyValue] = {}
        self.key_to_entities: Dict[str, List[str]] = defaultdict(list)
        self.key_to_relations: Dict[str, List[str]] = defaultdict(list)

    def create_entity_key_values(
        self,
        persons: List[Any],
        events: List[Any],
        organizations: List[Any],
    ) -> Dict[str, EntityKeyValue]:
        logger.info("开始创建实体键值对...")
        self.entity_kv_store.clear()
        self.key_to_entities.clear()

        entity_groups = [
            (persons, "Person"),
            (events, "Event"),
            (organizations, "Organization"),
        ]

        for entities, default_type in entity_groups:
            for entity in entities:
                entity_id = entity.node_id
                entity_name = entity.name or str(entity_id)
                props = getattr(entity, "properties", {}) or {}
                labels = getattr(entity, "labels", []) or []
                entity_type = labels[0] if labels else default_type

                content_parts = [f"名称: {entity_name}", f"类型: {entity_type}"]
                for key in [
                    "description", "organization", "period", "location", "event_type", "org_type",
                    "chapter", "section", "year", "time_start", "time_end", "summary",
                ]:
                    value = props.get(key)
                    if value not in (None, "", []):
                        content_parts.append(f"{key}: {value}")
                for key in ["community_l1", "community_l2", "community_l3"]:
                    if props.get(key):
                        content_parts.append(f"{key}: {props[key]}")

                index_keys = [entity_name]
                for key in [
                    "organization", "period", "event_type", "org_type", "chapter", "section",
                    "community_l1", "community_l2", "community_l3", "year",
                ]:
                    value = props.get(key)
                    if isinstance(value, str) and value.strip():
                        index_keys.append(value.strip())
                if entity_type == "Community":
                    level = props.get("level")
                    if level:
                        index_keys.append(f"L{level}社团")
                    for organization in props.get("top_organizations", []) or []:
                        index_keys.append(organization)

                entity_kv = EntityKeyValue(
                    entity_name=entity_name,
                    index_keys=list(dict.fromkeys(index_keys)),
                    value_content="\n".join(content_parts),
                    entity_type=entity_type,
                    metadata={
                        "node_id": entity_id,
                        "properties": props,
                    },
                )
                self.entity_kv_store[entity_id] = entity_kv
                for key in entity_kv.index_keys:
                    self.key_to_entities[key].append(entity_id)

        logger.info(f"实体键值对创建完成，共 {len(self.entity_kv_store)} 个实体")
        return self.entity_kv_store

    def create_relation_key_values(self, relationships: List[Tuple[str, str, str]]) -> Dict[str, RelationKeyValue]:
        logger.info("开始创建关系键值对...")
        self.relation_kv_store.clear()
        self.key_to_relations.clear()

        for i, (source_id, relation_type, target_id) in enumerate(relationships):
            source_entity = self.entity_kv_store.get(source_id)
            target_entity = self.entity_kv_store.get(target_id)
            if not source_entity or not target_entity:
                continue

            relation_id = f"rel_{i}_{source_id}_{target_id}_{relation_type}"
            content_parts = [
                f"关系类型: {relation_type}",
                f"源实体: {source_entity.entity_name} ({source_entity.entity_type})",
                f"目标实体: {target_entity.entity_name} ({target_entity.entity_type})",
            ]
            index_keys = self._generate_relation_index_keys(source_entity, target_entity, relation_type)

            relation_kv = RelationKeyValue(
                relation_id=relation_id,
                index_keys=index_keys,
                value_content="\n".join(content_parts),
                relation_type=relation_type,
                source_entity=source_id,
                target_entity=target_id,
                metadata={
                    "source_name": source_entity.entity_name,
                    "target_name": target_entity.entity_name,
                    "created_from_graph": True,
                },
            )
            self.relation_kv_store[relation_id] = relation_kv
            for key in index_keys:
                self.key_to_relations[key].append(relation_id)

        logger.info(f"关系键值对创建完成，共 {len(self.relation_kv_store)} 个关系")
        return self.relation_kv_store

    def _generate_relation_index_keys(self, source_entity: EntityKeyValue, target_entity: EntityKeyValue, relation_type: str) -> List[str]:
        keys = [relation_type, source_entity.entity_name, target_entity.entity_name]

        relation_topics = {
            "MEMBER_OF": ["组织归属", "成员关系", "组织成员"],
            "LEADS": ["领导", "关键人物", "事件领导"],
            "INITIATES": ["发起", "倡导", "事件发起"],
            "PARTICIPATES_IN": ["事件参与", "人物事件", "组织参与"],
            "BELONGS_TO": ["归属", "历史阶段", "章节归类"],
            "RELATED_TO": ["相关", "历史联系", "影响关系"],
            "MENTIONS_PERSON": ["原文人物", "文本证据"],
            "MENTIONS_EVENT": ["原文事件", "文本证据"],
            "MENTIONS_ORG": ["原文组织", "文本证据"],
            "DESCRIBES_EVENT": ["事件描述", "文本证据"],
            "IN_COMMUNITY": ["社团", "社区", "主题团簇"],
        }
        keys.extend(relation_topics.get(relation_type, ["历史关系"]))

        source_props = source_entity.metadata.get("properties", {})
        target_props = target_entity.metadata.get("properties", {})
        for key_name in ["community_l1", "community_l2", "community_l3", "organization", "period", "chapter", "section"]:
            for props in [source_props, target_props]:
                value = props.get(key_name)
                if isinstance(value, str) and value.strip():
                    keys.append(value.strip())

        return list(dict.fromkeys(keys))

    def deduplicate_entities_and_relations(self):
        logger.info("开始去重实体和关系...")
        name_to_entities = defaultdict(list)
        for entity_id, entity_kv in self.entity_kv_store.items():
            name_to_entities[entity_kv.entity_name].append(entity_id)

        entities_to_remove = []
        for _, entity_ids in name_to_entities.items():
            if len(entity_ids) > 1:
                primary_id = entity_ids[0]
                primary_entity = self.entity_kv_store[primary_id]
                for entity_id in entity_ids[1:]:
                    duplicate_entity = self.entity_kv_store[entity_id]
                    primary_entity.value_content += f"\n\n补充信息: {duplicate_entity.value_content}"
                    for key in duplicate_entity.index_keys:
                        if key not in primary_entity.index_keys:
                            primary_entity.index_keys.append(key)
                    entities_to_remove.append(entity_id)

        for entity_id in entities_to_remove:
            del self.entity_kv_store[entity_id]

        relation_signature_to_ids = defaultdict(list)
        for relation_id, relation_kv in self.relation_kv_store.items():
            signature = f"{relation_kv.source_entity}_{relation_kv.target_entity}_{relation_kv.relation_type}"
            relation_signature_to_ids[signature].append(relation_id)

        relations_to_remove = []
        for _, relation_ids in relation_signature_to_ids.items():
            if len(relation_ids) > 1:
                relations_to_remove.extend(relation_ids[1:])

        for relation_id in relations_to_remove:
            del self.relation_kv_store[relation_id]

        self._rebuild_key_mappings()
        logger.info(f"去重完成 - 删除了 {len(entities_to_remove)} 个重复实体，{len(relations_to_remove)} 个重复关系")

    def _rebuild_key_mappings(self):
        self.key_to_entities.clear()
        self.key_to_relations.clear()
        for entity_id, entity_kv in self.entity_kv_store.items():
            for key in entity_kv.index_keys:
                self.key_to_entities[key].append(entity_id)
        for relation_id, relation_kv in self.relation_kv_store.items():
            for key in relation_kv.index_keys:
                self.key_to_relations[key].append(relation_id)

    def get_entities_by_key(self, key: str) -> List[EntityKeyValue]:
        entity_ids = self.key_to_entities.get(key, [])
        return [self.entity_kv_store[eid] for eid in entity_ids if eid in self.entity_kv_store]

    def get_relations_by_key(self, key: str) -> List[RelationKeyValue]:
        relation_ids = self.key_to_relations.get(key, [])
        return [self.relation_kv_store[rid] for rid in relation_ids if rid in self.relation_kv_store]

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_entities": len(self.entity_kv_store),
            "total_relations": len(self.relation_kv_store),
            "total_entity_keys": sum(len(kv.index_keys) for kv in self.entity_kv_store.values()),
            "total_relation_keys": sum(len(kv.index_keys) for kv in self.relation_kv_store.values()),
            "entity_types": defaultdict(int, {
                entity_kv.entity_type: len([kv for kv in self.entity_kv_store.values() if kv.entity_type == entity_kv.entity_type])
                for entity_kv in self.entity_kv_store.values()
            }),
        }
