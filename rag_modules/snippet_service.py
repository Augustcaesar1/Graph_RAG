from __future__ import annotations

import re
from typing import List


class SourceSnippetService:
    def __init__(self, data_module):
        self.data_module = data_module

    def fetch_original_text_snippets(self, entity_name: str, limit: int = 2) -> List[str]:
        if not self.data_module or not getattr(self.data_module, "driver", None):
            return []

        def extract_terms(value: str) -> List[str]:
            text = str(value or "")
            stop = {"名称", "类型", "简介", "关系", "证据", "相关", "未知", "Person", "Event", "Location", "Artifact"}
            terms: List[str] = []
            for token in re.findall(r"[一-鿿]{2,8}", text):
                token = re.sub(r"(为什么|是什么|是谁|有哪些|做了什么|之间|关系|事件)$", "", token)
                if len(token) >= 2 and token not in stop and token not in terms:
                    terms.append(token)
                if len(token) >= 4:
                    for i in range(0, len(token) - 1):
                        subterm = token[i:i + 2]
                        if subterm not in stop and subterm not in terms:
                            terms.append(subterm)
            return terms[:12]

        def fmt(row) -> str:
            chapter = (row.get("chapter") or "").strip()
            chapter_number = row.get("chapter_number")
            chapter_title = (row.get("chapter_title") or "").strip()
            if chapter_title and chapter_number:
                chapter = f"第{chapter_number}回 {chapter_title}"
            elif chapter_title:
                chapter = chapter_title
            section = (row.get("section") or "").strip()
            year = (row.get("year") or "").strip()
            text = (row.get("text") or "").strip()
            prefix = " / ".join([p for p in [chapter, section, year] if p])
            return f"{prefix}：{text}" if prefix else text

        def direct_chunk_lookup(session, lookup_key: str) -> List[str]:
            queries = [
                """
                MATCH (t:TextChunk {chunk_id: $id})
                RETURN t.chapter as chapter, t.chapter_number as chapter_number, t.chapter_title as chapter_title, t.section as section, coalesce(t.year,'') as year, t.text as text
                LIMIT 1
                """,
                """
                MATCH (t:TextChunk)
                WHERE coalesce(t.chunk_id, '') = $id OR elementId(t) = $id
                RETURN t.chapter as chapter, t.chapter_number as chapter_number, t.chapter_title as chapter_title, t.section as section, coalesce(t.year,'') as year, t.text as text
                LIMIT 1
                """,
            ]
            for query in queries:
                row = session.run(query, {"id": lookup_key}).single()
                if row and row.get("text"):
                    return [fmt(dict(row))][:limit]
            return []

        def direct_evidence_lookup(session, lookup_key: str) -> List[str]:
            queries = [
                """
                MATCH (e:Evidence)
                WHERE coalesce(e.source_chunk_id, e.source_anchor, '') = $id
                RETURN e.chapter as chapter, e.section as section, coalesce(e.year,'') as year, e.source_text as text
                LIMIT $limit
                """,
                """
                MATCH (e:Evidence)
                WHERE e.owner_name = $name
                   OR e.owner_id = $name
                   OR e.source_id = $name
                   OR e.target_id = $name
                RETURN e.chapter as chapter, e.section as section, coalesce(e.year,'') as year, e.source_text as text
                LIMIT $limit
                """,
            ]
            for query, params in [
                (queries[0], {"id": lookup_key, "limit": int(limit)}),
                (queries[1], {"name": lookup_key, "limit": int(limit)}),
            ]:
                rows = [
                    fmt(dict(result))
                    for result in session.run(query, params)
                    if (result.get("text") or "").strip()
                ]
                if rows:
                    return rows[:limit]
            return []

        try:
            with self.data_module.driver.session(database=self.data_module.database) as session:
                direct_hits = direct_chunk_lookup(session, entity_name)
                if direct_hits:
                    return direct_hits

                evidence_hits = direct_evidence_lookup(session, entity_name)
                if evidence_hits:
                    return evidence_hits

                queries = [
                    """
                    MATCH (t:TextChunk)
                    WHERE t.text CONTAINS $name
                    RETURN t.chapter as chapter, t.chapter_number as chapter_number, t.chapter_title as chapter_title, t.section as section, coalesce(t.year,'') as year, t.text as text
                    ORDER BY coalesce(t.chapter_number, 999), coalesce(t.chunk_index, 999)
                    LIMIT $limit
                    """,
                    """
                    MATCH (t:TextChunk)-[r:MENTIONS_PERSON]->(p:Person {name: $name})
                    RETURN t.chapter as chapter, t.chapter_number as chapter_number, t.chapter_title as chapter_title, t.section as section, coalesce(t.year,'') as year, t.text as text
                    ORDER BY coalesce(r.confidence, 0) DESC
                    LIMIT $limit
                    """,
                    """
                    MATCH (t:TextChunk)-[r:MENTIONS_ORG]->(o:Organization {name: $name})
                    RETURN t.chapter as chapter, t.chapter_number as chapter_number, t.chapter_title as chapter_title, t.section as section, coalesce(t.year,'') as year, t.text as text
                    ORDER BY coalesce(r.confidence, 0) DESC
                    LIMIT $limit
                    """,
                    """
                    MATCH (t:TextChunk)-[r:DESCRIBES_EVENT]->(e:Event {name: $name})
                    RETURN t.chapter as chapter, t.chapter_number as chapter_number, t.chapter_title as chapter_title, t.section as section, coalesce(t.year,'') as year, t.text as text
                    ORDER BY coalesce(r.confidence, 0) DESC
                    LIMIT $limit
                    """,
                    """
                    MATCH (t:TextChunk)-[r:MENTIONS_EVENT]->(e:Event {name: $name})
                    RETURN t.chapter as chapter, t.chapter_number as chapter_number, t.chapter_title as chapter_title, t.section as section, coalesce(t.year,'') as year, t.text as text
                    ORDER BY coalesce(r.confidence, 0) DESC
                    LIMIT $limit
                    """,
                    """
                    MATCH (t:TextChunk)-[r:RELATED_TO]->(p:Period {name: $name})
                    RETURN t.chapter as chapter, t.chapter_number as chapter_number, t.chapter_title as chapter_title, t.section as section, coalesce(t.year,'') as year, t.text as text
                    ORDER BY coalesce(r.confidence, 0) DESC
                    LIMIT $limit
                    """,
                ]

                for query in queries:
                    rows = [
                        fmt(dict(result))
                        for result in session.run(query, {"name": entity_name, "limit": int(limit)})
                        if (result.get("text") or "").strip()
                    ]
                    if rows:
                        return rows[:limit]

                terms = extract_terms(entity_name)
                if terms:
                    rows = [
                        fmt(dict(result))
                        for result in session.run(
                            """
                            MATCH (t:TextChunk)
                            WHERE any(term IN $terms WHERE t.text CONTAINS term)
                            WITH t,
                                 size([term IN $terms WHERE t.text CONTAINS term]) AS hits
                            WHERE hits > 0
                            RETURN t.chapter as chapter,
                                   t.chapter_number as chapter_number,
                                   t.chapter_title as chapter_title,
                                   t.section as section,
                                   coalesce(t.year,'') as year,
                                   t.text as text,
                                   hits
                            ORDER BY hits DESC, coalesce(t.chapter_number, 999), coalesce(t.chunk_index, 999)
                            LIMIT $limit
                            """,
                            {"terms": terms, "limit": int(limit)},
                        )
                        if (result.get("text") or "").strip()
                    ]
                    if rows:
                        return rows[:limit]
        except Exception:
            return []

        return []
