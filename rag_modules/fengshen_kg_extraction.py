"""
封神演义知识图谱 LLM 抽取模块

使用 LLM 进行命名实体识别(NER)和关系抽取(RE)，输出标准三元组格式：
(头实体, 关系, 尾实体)。
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from .gold_schema import (
    ALLOWED_RELATION_TYPES,
    ENTITY_TYPES,
    extraction_cache_path,
    load_json_file,
    save_json_file,
)

logger = logging.getLogger(__name__)
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")


NER_RE_SYSTEM_PROMPT = """你是一位《封神演义》知识图谱构建专家，负责从原文中抽取实体和关系。

请从文本片段中识别以下实体类型：
1. Person（人物）：姜子牙、哪吒、杨戬、纣王、妲己、元始天尊、通天教主等
2. Faction（教派/势力）：商、周、阐教、截教、西方教、龙族、轩辕坟等
3. Location（地点）：朝歌、西岐、玉虚宫、碧游宫、陈塘关、佳梦关等
4. Artifact（法宝/物件）：打神鞭、乾坤圈、混天绫、风火轮、翻天印等
5. Beast（坐骑/灵兽）：四不像、哮天犬、五色神牛、墨麒麟等
6. Formation（阵法）：十绝阵、九曲黄河阵、诛仙阵、万仙阵等
7. Event（事件/战役）：哪吒闹海、武王伐纣、破诛仙阵、万仙阵大战等
8. DeityPosition（神位/封号）：三坛海会大神、九天应元雷声普化天尊、文曲星等

允许的关系类型：
- MASTER_OF / APPRENTICE_OF：师徒关系
- BELONGS_TO_SECT：人物属于教派/势力
- FIGHTS_FOR：效力于商/周等阵营
- FATHER_OF / CHILD_OF / BROTHER_OF / MARRIED_TO：亲属婚姻关系
- KILLS / DEFEATS / CAPTURES / OPPOSES：战斗对抗关系
- OWNS / BESTOWS / LOSES / STEALS：拥有/赐予/失去/盗取法宝坐骑
- CREATES / DEPLOYS / BREAKS / PARTICIPATES_IN / OCCURS_IN / LEADS / INITIATES：阵法与事件关系
- LISTED_ON / BECOMES：封神榜/神位结局
- ALLIES_WITH / BETRAYS / RELATED_TO / MENTIONS：通用关系

抽取规则：
1. 只抽取文本片段中明确出现或可严格从片段推断的事实。
2. 不要凭常识补充文本中没有出现的关系。
3. 每条关系必须给出 evidence 原文证据。
4. 人物别名可以放到 alias 数组中，如“子牙”“姜尚”“飞熊”。
5. 输出必须是严格 JSON，不要输出解释。

输出格式：
{
  "entities": {
    "Person": [
      {"name": "姜子牙", "alias": ["姜尚", "子牙"], "description": "元始天尊弟子，辅佐周室伐商", "attributes": {"sect": "阐教", "faction": "周"}}
    ],
    "Faction": [
      {"name": "阐教", "alias": [], "description": "元始天尊门下教派", "attributes": {"leader": "元始天尊"}}
    ],
    "Location": [],
    "Artifact": [],
    "Beast": [],
    "Formation": [],
    "Event": [],
    "DeityPosition": []
  },
  "relations": [
    {"source_entity": "元始天尊", "relation": "MASTER_OF", "target_entity": "姜子牙", "evidence": "原文证据句", "confidence": "explicit"}
  ]
}
"""

NER_RE_USER_PROMPT_TEMPLATE = """请从以下《封神演义》文本片段中抽取实体和关系。

章节：第{chapter_number}回 {chapter_title}
文本片段：
{text}

请严格按照系统要求的JSON格式输出。"""


@dataclass
class ExtractedEntity:
    name: str
    entity_type: str
    alias: List[str] = field(default_factory=list)
    description: str = ""
    attributes: Dict[str, str] = field(default_factory=dict)
    source_chunk_id: str = ""
    source_text: str = ""
    chapter_number: int = 0
    chapter_title: str = ""
    chunk_index: int = 0


@dataclass
class ExtractedRelation:
    source_entity: str
    relation: str
    target_entity: str
    evidence: str = ""
    confidence: str = "explicit"
    source_chunk_id: str = ""
    source_text: str = ""
    chapter_number: int = 0
    chapter_title: str = ""
    chunk_index: int = 0


@dataclass
class ChapterExtraction:
    chapter_number: int
    chapter_title: str
    entities: Dict[str, List[ExtractedEntity]] = field(default_factory=dict)
    relations: List[ExtractedRelation] = field(default_factory=list)


def chinese_num_to_int(s: str) -> int:
    digits = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "零": 0}
    units = {"十": 10, "百": 100}
    if s.isdigit():
        return int(s)
    total = 0
    num = 0
    for ch in s:
        if ch in digits:
            num = digits[ch]
        elif ch in units:
            if num == 0:
                num = 1
            total += num * units[ch]
            num = 0
    return total + num


def parse_chapters(text_path: str) -> List[Tuple[int, str, int, int]]:
    """解析章节，返回 [(回号, 标题, start, end), ...]"""
    text = Path(text_path).read_text(encoding="utf-8")
    pattern = re.compile(r"^第([一二三四五六七八九十百0-9]+)回[\s　]*(.+)$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    chapters = []
    for i, m in enumerate(matches):
        num = chinese_num_to_int(m.group(1))
        title = m.group(2).strip()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chapters.append((num, title, start, end))
    return chapters


def split_chapter_text(text: str, chunk_size: int = 2500, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            for sep in ["。", "！", "？", "\n", "；"]:
                pos = text.rfind(sep, start, end)
                if pos > start + chunk_size // 2:
                    end = pos + 1
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if end < len(text) else end
    return chunks


class FengshenKGExtractor:
    """使用LLM从《封神演义》中抽取知识图谱"""

    def __init__(
        self,
        text_path: str = "./封神演义.txt",
        api_base: str = "https://api.siliconflow.cn/v1",
        model: str = "deepseek-ai/DeepSeek-V3",
        chunk_size: int = 2500,
        chunk_overlap: int = 200,
    ):
        self.text_path = text_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = model
        api_key = os.getenv("SILICONFLOW_API_KEY") or os.getenv("MOONSHOT_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("请设置 SILICONFLOW_API_KEY 环境变量")
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.cache_dir = extraction_cache_path()

    def _cache_path(self, chapter_num: int, chunk_idx: int) -> Path:
        return self.cache_dir / f"ch{chapter_num:03d}_chunk{chunk_idx:03d}.json"

    def _extract_chunk(self, chunk_text: str, chapter_num: int, chapter_title: str, retries: int = 3) -> dict:
        user_prompt = NER_RE_USER_PROMPT_TEMPLATE.format(
            chapter_number=chapter_num,
            chapter_title=chapter_title,
            text=chunk_text,
        )
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": NER_RE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=4096,
                    timeout=120,
                )
                content = response.choices[0].message.content.strip()
                content = re.sub(r"^```(?:json)?\s*", "", content)
                content = re.sub(r"\s*```$", "", content)
                start = content.find("{")
                end = content.rfind("}") + 1
                if start != -1 and end > start:
                    content = content[start:end]
                return json.loads(content)
            except Exception as e:
                logger.warning("LLM抽取失败 %s/%s: %s", attempt + 1, retries, e)
                if attempt < retries - 1:
                    time.sleep(2 * (attempt + 1))
        return {"entities": {}, "relations": []}

    def extract_chapter(self, chapter_num: int, chapter_title: str, chapter_text: str) -> ChapterExtraction:
        result = ChapterExtraction(
            chapter_number=chapter_num,
            chapter_title=chapter_title,
            entities={t: [] for t in ENTITY_TYPES},
            relations=[],
        )
        chunks = split_chapter_text(chapter_text, self.chunk_size, self.chunk_overlap)
        logger.info("第%s回《%s》分为 %s 个块", chapter_num, chapter_title, len(chunks))
        entity_map: Dict[str, Dict[str, ExtractedEntity]] = {t: {} for t in ENTITY_TYPES}
        relation_map: Dict[Tuple[str, str, str], ExtractedRelation] = {}

        for idx, chunk in enumerate(chunks):
            source_chunk_id = f"extract_ch{chapter_num:03d}_{idx:04d}"
            cache_path = self._cache_path(chapter_num, idx)
            if cache_path.exists():
                try:
                    data = load_json_file(cache_path)[0]
                except Exception:
                    data = self._extract_chunk(chunk, chapter_num, chapter_title)
                    save_json_file(cache_path, [data])
            else:
                data = self._extract_chunk(chunk, chapter_num, chapter_title)
                save_json_file(cache_path, [data])

            entities_data = data.get("entities", {}) if isinstance(data, dict) else {}
            for etype in ENTITY_TYPES:
                for ent in entities_data.get(etype, []) or []:
                    if not isinstance(ent, dict):
                        continue
                    name = str(ent.get("name") or "").strip()
                    if not name:
                        continue
                    if name not in entity_map[etype]:
                        entity_map[etype][name] = ExtractedEntity(
                            name=name,
                            entity_type=etype,
                            alias=list(ent.get("alias", []) or []),
                            description=str(ent.get("description") or ""),
                            attributes=dict(ent.get("attributes") or {}),
                            source_chunk_id=source_chunk_id,
                            source_text=chunk,
                            chapter_number=chapter_num,
                            chapter_title=chapter_title,
                            chunk_index=idx,
                        )
                    else:
                        old = entity_map[etype][name]
                        old.alias = list(dict.fromkeys(old.alias + list(ent.get("alias", []) or [])))
                        if not old.description and ent.get("description"):
                            old.description = str(ent.get("description"))
                        old.attributes.update(dict(ent.get("attributes") or {}))

            for rel in data.get("relations", []) if isinstance(data, dict) else []:
                if not isinstance(rel, dict):
                    continue
                src = str(rel.get("source_entity") or "").strip()
                rtype = str(rel.get("relation") or "").strip()
                tgt = str(rel.get("target_entity") or "").strip()
                if not src or not tgt or rtype not in ALLOWED_RELATION_TYPES:
                    continue
                key = (src, rtype, tgt)
                if key not in relation_map:
                    relation_map[key] = ExtractedRelation(
                        source_entity=src,
                        relation=rtype,
                        target_entity=tgt,
                        evidence=str(rel.get("evidence") or ""),
                        confidence=str(rel.get("confidence") or "explicit"),
                        source_chunk_id=source_chunk_id,
                        source_text=chunk,
                        chapter_number=chapter_num,
                        chapter_title=chapter_title,
                        chunk_index=idx,
                    )
            time.sleep(0.2)

        result.entities = {t: list(v.values()) for t, v in entity_map.items()}
        result.relations = list(relation_map.values())
        logger.info("第%s回抽取完成：实体%s，关系%s", chapter_num, sum(len(v) for v in result.entities.values()), len(result.relations))
        return result

    def extract_all_chapters(self, chapter_limit: Optional[int] = None, start_chapter: int = 1) -> List[ChapterExtraction]:
        chapters = parse_chapters(self.text_path)
        text = Path(self.text_path).read_text(encoding="utf-8")
        results = []
        for num, title, start, end in chapters:
            if num < start_chapter:
                continue
            if chapter_limit and num >= start_chapter + chapter_limit:
                break
            chapter_text = text[start:end].strip()
            chapter_text = re.sub(r"^第[一二三四五六七八九十百0-9]+回[\s　]*.+$", "", chapter_text, count=1, flags=re.MULTILINE).strip()
            results.append(self.extract_chapter(num, title, chapter_text))
        return results

    def merge_chapter_extractions(self, extractions: List[ChapterExtraction]):
        merged_entities: Dict[str, Dict[str, ExtractedEntity]] = {t: {} for t in ENTITY_TYPES}
        merged_relations: Dict[Tuple[str, str, str], ExtractedRelation] = {}
        for ext in extractions:
            for etype, entities in ext.entities.items():
                for ent in entities:
                    if ent.name not in merged_entities[etype]:
                        merged_entities[etype][ent.name] = ent
                    else:
                        old = merged_entities[etype][ent.name]
                        old.alias = list(dict.fromkeys(old.alias + ent.alias))
                        if not old.description and ent.description:
                            old.description = ent.description
                        old.attributes.update(ent.attributes)
                        if not old.source_chunk_id and ent.source_chunk_id:
                            old.source_chunk_id = ent.source_chunk_id
                            old.source_text = ent.source_text
                            old.chapter_number = ent.chapter_number
                            old.chapter_title = ent.chapter_title
                            old.chunk_index = ent.chunk_index
            for rel in ext.relations:
                key = (rel.source_entity, rel.relation, rel.target_entity)
                if key not in merged_relations:
                    merged_relations[key] = rel
        return {t: list(v.values()) for t, v in merged_entities.items()}, list(merged_relations.values())

    def save_merged_results(self, all_entities, all_relations) -> Path:
        output_path = self.cache_dir / "merged_kg.json"
        data = {
            "entities": {
                etype: [
                    {
                        "name": e.name,
                        "alias": e.alias,
                        "description": e.description,
                        "attributes": e.attributes,
                        "source_chunk_id": e.source_chunk_id,
                        "source_text": e.source_text,
                        "chapter_number": e.chapter_number,
                        "chapter_title": e.chapter_title,
                        "chunk_index": e.chunk_index,
                    }
                    for e in entities
                ]
                for etype, entities in all_entities.items()
            },
            "relations": [
                {
                    "source_entity": r.source_entity,
                    "relation": r.relation,
                    "target_entity": r.target_entity,
                    "evidence": r.evidence,
                    "confidence": r.confidence,
                    "source_chunk_id": r.source_chunk_id,
                    "source_text": r.source_text,
                    "chapter_number": r.chapter_number,
                    "chapter_title": r.chapter_title,
                    "chunk_index": r.chunk_index,
                }
                for r in all_relations
            ],
            "statistics": {etype: len(entities) for etype, entities in all_entities.items()},
            "total_relations": len(all_relations),
        }
        save_json_file(output_path, [data])
        return output_path

    def load_merged_results(self):
        path = self.cache_dir / "merged_kg.json"
        data = load_json_file(path)[0]
        all_entities = {}
        for etype, ent_list in data.get("entities", {}).items():
            all_entities[etype] = [
                ExtractedEntity(
                    name=e["name"],
                    entity_type=etype,
                    alias=e.get("alias", []),
                    description=e.get("description", ""),
                    attributes=e.get("attributes", {}),
                    source_chunk_id=e.get("source_chunk_id", ""),
                    source_text=e.get("source_text", ""),
                    chapter_number=int(e.get("chapter_number") or 0),
                    chapter_title=e.get("chapter_title", ""),
                    chunk_index=int(e.get("chunk_index") or 0),
                )
                for e in ent_list
            ]
        all_relations = [
            ExtractedRelation(
                source_entity=r["source_entity"],
                relation=r["relation"],
                target_entity=r["target_entity"],
                evidence=r.get("evidence", ""),
                confidence=r.get("confidence", "explicit"),
                source_chunk_id=r.get("source_chunk_id", ""),
                source_text=r.get("source_text", ""),
                chapter_number=int(r.get("chapter_number") or 0),
                chapter_title=r.get("chapter_title", ""),
                chunk_index=int(r.get("chunk_index") or 0),
            )
            for r in data.get("relations", [])
        ]
        return all_entities, all_relations


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    extractor = FengshenKGExtractor()
    results = extractor.extract_all_chapters(chapter_limit=3)
    entities, relations = extractor.merge_chapter_extractions(results)
    path = extractor.save_merged_results(entities, relations)
    print("saved", path)
