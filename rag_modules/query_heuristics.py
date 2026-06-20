"""查询启发式规则 - 封神演义版"""

from __future__ import annotations

import re
from typing import Iterable, List


QUESTION_MARKERS = (
    "什么", "为何", "为什么", "如何", "怎样", "关系", "影响", "原因", "作用", "是谁",
    "哪", "哪个", "师父", "徒弟", "法宝", "阵营", "教派", "封神", "上榜", "成圣", "破阵",
)
GRAPH_MARKERS = (
    "关系", "联系", "为什么", "为何", "路径", "过程", "师父", "徒弟", "法宝", "阵营",
    "教派", "上榜", "封神", "成圣", "破阵", "击败", "杀", "属于",
)
LOOKUP_MARKERS = ("是谁", "是什么人", "简介", "介绍", "生平", "是哪", "在哪")
TOPIC_MARKERS = ("关系", "师承", "法宝", "阵营", "教派", "封神", "上榜", "成圣", "破阵", "结局")


def _unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def is_fengshen_question(query: str) -> bool:
    text = str(query or "").strip()
    return len(text) >= 4 and any(marker in text for marker in QUESTION_MARKERS)


is_history_question = is_fengshen_question


def extract_candidate_entities(query: str, known_entities: Iterable[str] | None = None, max_keywords: int = 6) -> List[str]:
    text = str(query or "").strip()
    if not text:
        return []
    entities: List[str] = []
    matched = []
    for entity in set(known_entities or []):
        if entity and entity in text:
            matched.append((text.index(entity), -len(entity), entity))
    for _, _, entity in sorted(matched):
        entities.append(entity)

    compact = re.sub(r"[，。！？、；：,.!?:;()（）\[\]{}\s]+", "", text)
    compact = re.sub(r"^(请问|请说说|请介绍|请分析|请说明)", "", compact)
    patterns = [
        r"和(.+)是什么关系", r"与(.+)是什么关系", r"是谁的(.+)", r"属于哪个(.+)",
        r"的师父是谁", r"的徒弟是谁", r"有什么法宝", r"使用什么法宝", r"是什么关系", r"是谁",
        r"属于(.+)阵营", r"属于(.+)教派", r"最后封为什么", r"结局是什么",
    ]
    focus = compact
    for pattern in patterns:
        focus = re.sub(pattern, "", focus)
    if focus and focus != compact:
        for part in re.split(r"[与和及、]", focus):
            part = part.strip("的")
            if len(part) >= 2:
                entities.append(part)
    return _unique_preserve_order(entities)[:max_keywords]


def derive_topic_keywords(query: str) -> List[str]:
    text = str(query or "").strip()
    return _unique_preserve_order([m for m in TOPIC_MARKERS if m in text])[:6]


def is_simple_lookup_query(query: str, entity_keywords: Iterable[str] | None = None) -> bool:
    text = str(query or "").strip()
    entities = list(entity_keywords or [])
    if any(marker in text for marker in GRAPH_MARKERS if marker not in LOOKUP_MARKERS):
        return False
    if any(marker in text for marker in LOOKUP_MARKERS):
        return True
    return len(entities) == 1 and len(text) <= 10


def infer_search_strategy(query: str, entity_keywords: Iterable[str] | None = None) -> str:
    text = str(query or "").strip()
    entities = list(entity_keywords or [])
    if is_simple_lookup_query(text, entities):
        return "hybrid_traditional"
    if any(marker in text for marker in GRAPH_MARKERS):
        return "graph_rag"
    if len(entities) >= 2:
        return "graph_rag"
    return "hybrid_traditional"
