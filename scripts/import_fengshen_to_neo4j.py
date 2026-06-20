"""
封神演义知识图谱导入脚本

流程：
1. 使用 LLM 从 封神演义.txt 抽取实体和关系（可缓存）
2. 写入 Neo4j 图数据库
3. 按回/段落创建 TextChunk，满足 500+ 非结构化文本块要求
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

from config import DEFAULT_CONFIG, GraphRAGConfig
from rag_modules.gold_schema import ENTITY_TYPES, ENTITY_ID_FIELDS, ALLOWED_RELATION_TYPES
from rag_modules.fengshen_kg_extraction import FengshenKGExtractor, parse_chapters

logger = logging.getLogger(__name__)


class FengshenNeo4jImporter:
    def __init__(self, config: GraphRAGConfig = None, connect_neo4j: bool = True):
        self.config = config or DEFAULT_CONFIG
        self.driver = None
        if connect_neo4j:
            self.driver = GraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password),
            )

    def close(self):
        if self.driver:
            self.driver.close()

    def _clear_graph_data(self, session):
        legacy_labels = [
            "Evidence", "Organization", "Period", "State", "Concept",
            "Dish", "Ingredient", "Recipe", "CookingStep", "Seasoning", "Technique", "Tool", "Place",
        ]
        labels = list(dict.fromkeys(list(ENTITY_TYPES.keys()) + ["TextChunk", "Chapter", "Community"] + legacy_labels))
        for label in labels:
            try:
                session.run(f"MATCH (n:{label}) DETACH DELETE n")
                logger.info("已清空 %s", label)
            except Exception as e:
                logger.debug("清空 %s 失败: %s", label, e)

    def _create_indexes(self, session):
        for label, id_field in ENTITY_ID_FIELDS.items():
            session.run(f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.{id_field})")
            session.run(f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.name)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (n:TextChunk) ON (n.chunk_id)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (n:Chapter) ON (n.chapter_id)")

    def _persist_entities(self, session, all_entities: dict) -> Counter:
        stats = Counter()
        for etype, entities in all_entities.items():
            if etype not in ENTITY_TYPES:
                continue
            id_field = ENTITY_ID_FIELDS.get(etype, "entity_id")
            for ent in entities:
                name = ent.name.strip()
                if not name:
                    continue
                entity_id = f"fs_{etype.lower()}_{name.replace(' ', '_').replace('/', '_')}"
                props = {
                    id_field: entity_id,
                    "name": name,
                    "alias": ent.alias,
                    "description": ent.description,
                    "entity_type": etype,
                    "source_chunk_id": getattr(ent, "source_chunk_id", ""),
                    "source_text": getattr(ent, "source_text", ""),
                    "chapter_number": getattr(ent, "chapter_number", 0),
                    "chapter_title": getattr(ent, "chapter_title", ""),
                    "source_chunk_index": getattr(ent, "chunk_index", 0),
                    **{f"attr_{k}": v for k, v in (ent.attributes or {}).items() if isinstance(k, str)},
                }
                try:
                    session.run(
                        f"""
                        MERGE (n:{etype} {{name: $name}})
                        SET n += $props
                        """,
                        name=name,
                        props=props,
                    )
                    stats[f"entity_{etype}"] += 1
                except Exception as e:
                    logger.warning("实体写入失败 %s:%s - %s", etype, name, e)
        return stats

    def _find_entity_label(self, session, name: str) -> str | None:
        rec = session.run(
            """
            MATCH (n {name: $name})
            RETURN head(labels(n)) AS label
            LIMIT 1
            """,
            name=name,
        ).single()
        return rec["label"] if rec else None


    def _merge_aliased_entities(self, session) -> int:
        """Merge entities that share an alias (entity alignment).

        When two entities of the same type have one's name appear in
        the other's alias list, merge properties into the canonical
        (shorter-name) node and delete the duplicate.
        """
        import logging
        merged = 0
        for etype in ENTITY_TYPES:
            records = list(session.run(
                f"MATCH (n:{etype}) WHERE n.alias IS NOT NULL AND size(coalesce(n.alias, [])) > 0 "
                f"RETURN n.name AS name, n.alias AS aliases, elementId(n) AS nid"
            ))
            alias_map = {}
            for rec in records:
                for alias in rec["aliases"]:
                    alias_clean = alias.strip()
                    if alias_clean and alias_clean != rec["name"]:
                        alias_map.setdefault(alias_clean, []).append(rec)

            for alias, sources in alias_map.items():
                if len(sources) < 2:
                    continue
                names = {s["name"] for s in sources}
                try:
                    session.run(
                        f"MATCH (n:{etype}) WHERE n.name IN  "
                        f"WITH n ORDER BY size(n.name) "
                        f"WITH collect(n) AS nodes "
                        f"WITH nodes[0] AS keep, nodes[1..] AS remove_list "
                        f"UNWIND remove_list AS rem "
                        f"OPTIONAL MATCH (rem)-[r]-() "
                        f"DELETE r "
                        f"WITH keep, rem "
                        f"SET keep.alias = coalesce(keep.alias, []) + [rem.name] "
                        f"DELETE rem",
                        {"names": list(names)},
                    )
                    merged += len(names) - 1
                except Exception:
                    pass
        logging.getLogger(__name__).info(f"Entity alignment merged {merged} duplicate nodes across all types")
        return merged


    def _persist_relations(self, session, all_relations: list) -> int:
        count = 0
        for rel in all_relations:
            if rel.relation not in ALLOWED_RELATION_TYPES:
                continue
            if not rel.source_entity or not rel.target_entity:
                continue
            try:
                rec = session.run(
                    """
                    MATCH (a {name: $source})
                    MATCH (b {name: $target})
                    WITH a, b
                    MERGE (a)-[r:%s]->(b)
                    SET r.evidence = $evidence,
                        r.confidence = $confidence,
                        r.source = 'llm_extraction',
                        r.source_chunk_id = $source_chunk_id,
                        r.source_text = $source_text,
                        r.chapter_number = $chapter_number,
                        r.chapter_title = $chapter_title,
                        r.source_chunk_index = $source_chunk_index
                    RETURN count(r) AS c
                    """ % rel.relation,
                    source=rel.source_entity,
                    target=rel.target_entity,
                    evidence=rel.evidence,
                    confidence=rel.confidence,
                    source_chunk_id=getattr(rel, "source_chunk_id", ""),
                    source_text=getattr(rel, "source_text", ""),
                    chapter_number=getattr(rel, "chapter_number", 0),
                    chapter_title=getattr(rel, "chapter_title", ""),
                    source_chunk_index=getattr(rel, "chunk_index", 0),
                ).single()
                if rec and rec["c"]:
                    count += 1
            except Exception as e:
                logger.debug("关系写入失败 %s-[%s]->%s: %s", rel.source_entity, rel.relation, rel.target_entity, e)
        return count

    def _persist_text_chunks(self, session, text_path: str) -> Counter:
        stats = Counter()
        text = Path(text_path).read_text(encoding="utf-8")
        chapters = parse_chapters(text_path)
        chunk_total = 0

        for i, (ch_num, title, start, end) in enumerate(chapters):
            chapter_text = text[start:end].strip()
            session.run(
                """
                MERGE (ch:Chapter {chapter_id: $chapter_id})
                SET ch.chapter_number = $chapter_number,
                    ch.title = $title,
                    ch.text_length = $text_length
                """,
                chapter_id=f"chapter_{ch_num:03d}",
                chapter_number=ch_num,
                title=title,
                text_length=len(chapter_text),
            )
            stats["chapters"] += 1

            paragraphs = [p.strip() for p in chapter_text.split("\n") if p.strip()]
            merged = []
            buf = ""
            for p in paragraphs:
                if len(buf) + len(p) < 650:
                    buf += "\n" + p if buf else p
                else:
                    if buf:
                        merged.append(buf)
                    buf = p
            if buf:
                merged.append(buf)

            for idx, chunk in enumerate(merged):
                if len(chunk) < 30:
                    continue
                chunk_id = f"fs_ch{ch_num:03d}_{idx:04d}"
                session.run(
                    """
                    MERGE (t:TextChunk {chunk_id: $chunk_id})
                    SET t.chapter_number = $chapter_number,
                        t.chapter_title = $chapter_title,
                        t.chunk_index = $chunk_index,
                        t.text = $text,
                        t.text_length = $text_length
                    WITH t
                    MATCH (ch:Chapter {chapter_id: $chapter_id})
                    MERGE (t)-[:BELONGS_TO_CHAPTER]->(ch)
                    """,
                    chunk_id=chunk_id,
                    chapter_number=ch_num,
                    chapter_title=title,
                    chapter_id=f"chapter_{ch_num:03d}",
                    chunk_index=idx,
                    text=chunk,
                    text_length=len(chunk),
                )
                chunk_total += 1
                stats["text_chunks"] += 1

        logger.info("写入文本块：%s 章，%s 块", stats["chapters"], stats["text_chunks"])
        return stats

    def import_all(
        self,
        text_path: str = None,
        chapter_limit: int = None,
        start_chapter: int = 1,
        skip_extraction: bool = False,
        extraction_only: bool = False,
    ) -> dict:
        text_path = text_path or self.config.fengshen_text_path
        if not Path(text_path).exists():
            raise FileNotFoundError(text_path)

        extractor = FengshenKGExtractor(
            text_path=text_path,
            api_base=self.config.llm_api_base,
            model=self.config.llm_model,
            chunk_size=self.config.fengshen_extraction_chunk_size,
            chunk_overlap=self.config.fengshen_extraction_chunk_overlap,
        )

        if skip_extraction:
            all_entities, all_relations = extractor.load_merged_results()
        else:
            extractions = extractor.extract_all_chapters(chapter_limit=chapter_limit, start_chapter=start_chapter)
            all_entities, all_relations = extractor.merge_chapter_extractions(extractions)
            extractor.save_merged_results(all_entities, all_relations)

        if extraction_only or not self.driver:
            return {
                "entities_total": sum(len(v) for v in all_entities.values()),
                "relations_total": len(all_relations),
                **{f"entity_{k}": len(v) for k, v in all_entities.items()},
            }

        stats = Counter()
        with self.driver.session(database=self.config.neo4j_database) as session:
            self._clear_graph_data(session)
            self._create_indexes(session)
            stats.update(self._persist_entities(session, all_entities))
            stats["relations"] = self._persist_relations(session, all_relations)
            stats["merged_entities"] = self._merge_aliased_entities(session)
            stats.update(self._persist_text_chunks(session, text_path))
        return dict(stats)


def main():
    parser = argparse.ArgumentParser(description="封神演义知识图谱导入")
    parser.add_argument("--text-path", default="./封神演义.txt")
    parser.add_argument("--chapter-limit", type=int, default=None)
    parser.add_argument("--start-chapter", type=int, default=1)
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--extraction-only", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(message)s")
    importer = FengshenNeo4jImporter(connect_neo4j=not args.extraction_only)
    try:
        stats = importer.import_all(
            text_path=args.text_path,
            chapter_limit=args.chapter_limit,
            start_chapter=args.start_chapter,
            skip_extraction=args.skip_extraction,
            extraction_only=args.extraction_only,
        )
        print("\n=== 导入统计 ===")
        for k, v in sorted(stats.items()):
            print(f"  {k}: {v}")
    finally:
        importer.close()


if __name__ == "__main__":
    main()
