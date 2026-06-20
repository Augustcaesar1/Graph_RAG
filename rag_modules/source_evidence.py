from __future__ import annotations

import re
from typing import Callable


def extract_cited_indices(answer: str) -> list[int]:
    if not answer:
        return []
    return sorted({int(item) for item in re.findall(r"\[(\d+)\]", answer)})


def _clean_text(value: object) -> str:
    return str(value or "").strip()


def _lookup_candidates(source: dict) -> list[str]:
    candidates: list[str] = []
    for key in ("source_chunk_id", "chunk_id", "node_id", "entity_name"):
        value = _clean_text(source.get(key))
        if value and value not in candidates:
            candidates.append(value)
    return candidates


def _extract_keywords(*values: object) -> list[str]:
    stop_words = {
        "根据", "当前", "资料", "可以", "说明", "关系", "事件", "证据", "原文", "相关", "信息", "检索", "回答",
        "具体", "如下", "原因", "随后", "行动", "使用", "进行", "指出", "导致", "直接", "当前资料",
    }
    priority_terms = [
        "题诗", "亵渎", "女娲宫", "招妖幡", "三妖", "妲己", "琵琶精", "玉石琵琶", "惑乱", "纣王",
        "商容", "虔敬", "神圣", "武王伐纣", "助成功", "隐其妖形", "托身宫院",
        "哪吒", "龙王", "李靖", "太乙真人", "剔骨", "还父", "闹海",
    ]
    keywords: list[str] = []
    combined = " ".join(_clean_text(value) for value in values)
    for term in priority_terms:
        if term in combined and term not in keywords:
            keywords.append(term)
    for value in values:
        text = _clean_text(value)
        for token in re.findall(r"[一-鿿]{2,8}", text):
            token = re.sub(r"(为什么|是什么|是谁|有哪些|做了什么|之间|关系|事件)$", "", token)
            if len(token) >= 2 and token not in stop_words and token not in keywords:
                keywords.append(token)
    return keywords[:16]


def _citation_context(answer: str, index: int, window_chars: int = 120) -> str:
    chunks = []
    for match in re.finditer(rf"\[{index}\]", answer or ""):
        start = max(0, match.start() - window_chars)
        end = min(len(answer), match.end() + window_chars)
        chunks.append(answer[start:end])
    return "\n".join(chunks)


def _extract_evidence_anchors(*values: object) -> list[str]:
    anchors: list[str] = []
    for value in values:
        text = _clean_text(value)
        for match in re.finditer(r"证据[:：]\s*([^\n。；;]{6,120})", text):
            anchor = match.group(1).strip(" '\"‘’“”，,。；;：:")
            if len(anchor) >= 6 and anchor not in anchors:
                anchors.append(anchor)
        for match in re.finditer(r"[‘“\"]([^’”\"]{6,120})[’”\"]", text):
            anchor = match.group(1).strip()
            if len(anchor) >= 6 and anchor not in anchors:
                anchors.append(anchor)
    return anchors[:8]


def _best_anchor_position(text: str, anchors: list[str]) -> int:
    for anchor in anchors:
        candidates = [anchor]
        if len(anchor) > 24:
            candidates.extend([anchor[:24], anchor[-24:]])
        for candidate in candidates:
            candidate = candidate.strip()
            if len(candidate) < 4:
                continue
            pos = text.find(candidate)
            if pos >= 0:
                return pos
    return -1


def _trim_to_useful_excerpt(text: str, keywords: list[str], max_chars: int = 320, anchors: list[str] | None = None) -> str:
    text = re.sub(r"\s+", " ", _clean_text(text))
    if len(text) <= max_chars:
        return text

    anchor_pos = _best_anchor_position(text, anchors or [])
    if anchor_pos >= 0:
        start = max(0, anchor_pos - max_chars // 4)
        end = min(len(text), start + max_chars)
        return text[start:end].strip()

    sentences = [part.strip() for part in re.split(r"(?<=[。！？；])", text) if part.strip()]
    if not sentences:
        sentences = [text]

    scored: list[tuple[int, int, str]] = []
    for index, sentence in enumerate(sentences):
        score = sum(1 for keyword in keywords if keyword and keyword in sentence)
        if score >= 1:
            scored.append((score, -index, sentence))

    if scored:
        scored.sort(reverse=True)
        best_index = -scored[0][1]
        excerpt_parts = []
        for pos in range(max(0, best_index - 1), min(len(sentences), best_index + 2)):
            if sum(len(part) for part in excerpt_parts) + len(sentences[pos]) <= max_chars:
                excerpt_parts.append(sentences[pos])
        excerpt = "".join(excerpt_parts).strip()
        if excerpt:
            return excerpt[:max_chars]

    for keyword in keywords:
        pos = text.find(keyword)
        if pos >= 0:
            start = max(0, pos - max_chars // 3)
            end = min(len(text), start + max_chars)
            return text[start:end].strip()

    return text[:max_chars].strip()


def _normalized_excerpt_key(text: str) -> str:
    return re.sub(r"\s+", "", _clean_text(text))[:240]


def _as_list(value: object) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _append_excerpt(
    excerpts: list[dict],
    seen: set[str],
    text: object,
    *,
    kind: str,
    keywords: list[str],
    anchors: list[str] | None = None,
    max_chars: int = 800,
    source_chunk_id: str = "",
    edge: dict | None = None,
) -> None:
    raw = _clean_text(text)
    if len(raw) < 8:
        return
    excerpt = _trim_to_useful_excerpt(raw, keywords, max_chars=max_chars, anchors=anchors or [])
    key = _normalized_excerpt_key(excerpt)
    if not key or key in seen:
        return
    seen.add(key)
    excerpts.append({
        "text": excerpt,
        "kind": kind,
        "source_chunk_id": _clean_text(source_chunk_id),
        "edge": edge or {},
    })


def _edge_to_fact(edge: dict) -> str:
    source = _clean_text(edge.get("source"))
    relation = _clean_text(edge.get("relation"))
    target = _clean_text(edge.get("target"))
    evidence = _clean_text(edge.get("evidence"))
    if not (source or relation or target or evidence):
        return ""
    fact = f"{source} --{relation}--> {target}" if source and relation and target else ""
    if evidence:
        fact = f"{fact}；证据：{evidence}" if fact else f"证据：{evidence}"
    return fact


def build_evidence_bundles(
    documents: list,
    *,
    max_docs: int = 6,
    max_excerpts_per_doc: int = 4,
    max_excerpt_chars: int = 800,
) -> list[dict]:
    """Normalize retrieved Documents into citation-aligned evidence bundles.

    Original prose is kept separate from graph facts so generation and UI can
    verify claims against source text instead of graph summaries alone.
    """
    bundles: list[dict] = []
    for index, doc in enumerate((documents or [])[:max_docs], start=1):
        metadata = dict(getattr(doc, "metadata", {}) or {})
        content_preview = _clean_text(getattr(doc, "page_content", ""))
        title = _clean_text(
            metadata.get("entity_name")
            or metadata.get("raw_entity_name")
            or metadata.get("node_id")
            or metadata.get("source")
            or f"证据 {index}"
        )
        edges = [edge for edge in (metadata.get("subgraph_edges") or []) if isinstance(edge, dict)]
        graph_facts = []
        seen_facts: set[str] = set()
        for edge in edges:
            fact = _edge_to_fact(edge)
            if fact and fact not in seen_facts:
                seen_facts.add(fact)
                graph_facts.append(fact)

        keywords = _extract_keywords(title, content_preview, metadata.get("source_text"), *(edge.get("evidence") for edge in edges))
        anchors = _extract_evidence_anchors(content_preview, metadata.get("source_text"), *(edge.get("evidence") for edge in edges))
        excerpts: list[dict] = []
        seen_excerpts: set[str] = set()

        # Edge-level source text is most precise for GraphRAG claims.
        for edge in edges:
            _append_excerpt(
                excerpts,
                seen_excerpts,
                edge.get("source_text"),
                kind="edge_source_text",
                keywords=_extract_keywords(edge.get("evidence"), title, content_preview) or keywords,
                anchors=_extract_evidence_anchors(edge.get("evidence"), edge.get("source_text")) or anchors,
                max_chars=max_excerpt_chars,
                source_chunk_id=edge.get("source_chunk_id", ""),
                edge=edge,
            )
            if len(excerpts) >= max_excerpts_per_doc:
                break

        # Multiple snippets may have been backfilled by retrieval/UI code.
        if len(excerpts) < max_excerpts_per_doc:
            for snippet in _as_list(metadata.get("source_snippets")):
                _append_excerpt(
                    excerpts,
                    seen_excerpts,
                    snippet,
                    kind="source_snippet",
                    keywords=keywords,
                    anchors=anchors,
                    max_chars=max_excerpt_chars,
                    source_chunk_id=metadata.get("source_chunk_id") or metadata.get("chunk_id") or "",
                )
                if len(excerpts) >= max_excerpts_per_doc:
                    break

        # Document-level source_text is still valuable, especially for TextChunks.
        if len(excerpts) < max_excerpts_per_doc:
            _append_excerpt(
                excerpts,
                seen_excerpts,
                metadata.get("source_text"),
                kind="source_text",
                keywords=keywords,
                anchors=anchors,
                max_chars=max_excerpt_chars,
                source_chunk_id=metadata.get("source_chunk_id") or metadata.get("chunk_id") or "",
            )

        # If no source text exists, edge evidence is the closest available quote.
        if len(excerpts) < max_excerpts_per_doc:
            for edge in edges:
                _append_excerpt(
                    excerpts,
                    seen_excerpts,
                    edge.get("evidence"),
                    kind="edge_evidence",
                    keywords=keywords,
                    anchors=anchors,
                    max_chars=max_excerpt_chars,
                    source_chunk_id=edge.get("source_chunk_id", ""),
                    edge=edge,
                )
                if len(excerpts) >= max_excerpts_per_doc:
                    break

        # TextChunk page_content is original text even when metadata is sparse.
        if len(excerpts) < max_excerpts_per_doc and metadata.get("node_type") == "TextChunk":
            _append_excerpt(
                excerpts,
                seen_excerpts,
                content_preview,
                kind="text_chunk_content",
                keywords=keywords,
                anchors=anchors,
                max_chars=max_excerpt_chars,
                source_chunk_id=metadata.get("source_chunk_id") or metadata.get("chunk_id") or "",
            )

        bundles.append({
            "number": index,
            "title": title,
            "source_chunk_id": _clean_text(metadata.get("source_chunk_id") or metadata.get("chunk_id")),
            "search_type": _clean_text(metadata.get("search_type") or metadata.get("retrieval_level")),
            "original_excerpts": excerpts[:max_excerpts_per_doc],
            "graph_facts": graph_facts[:12],
            "summary_preview": content_preview,
            "has_original_text": bool(excerpts),
        })
    return bundles


def format_evidence_context_for_generation(
    bundles: list[dict],
    *,
    max_total_chars: int = 12000,
    max_summary_chars: int = 700,
) -> str:
    """Format evidence bundles for the answer-generation prompt."""
    parts: list[str] = []
    used = 0
    for bundle in bundles or []:
        number = bundle.get("number")
        title = _clean_text(bundle.get("title")) or "相关资料"
        source_chunk_id = _clean_text(bundle.get("source_chunk_id"))
        header = f"[{number}] {title} / {'原文证据' if bundle.get('has_original_text') else '图谱资料'}"
        block_lines = [header]
        if source_chunk_id:
            block_lines.append(f"来源片段：{source_chunk_id}")

        excerpts = bundle.get("original_excerpts") or []
        if excerpts:
            block_lines.append("原文摘录：")
            for item in excerpts:
                text = _clean_text(item.get("text") if isinstance(item, dict) else item)
                if text:
                    block_lines.append(f"- {text}")
        else:
            block_lines.append("原文摘录：暂无直接原文摘录；以下仅可作为图谱辅助。")

        graph_facts = [_clean_text(item) for item in (bundle.get("graph_facts") or []) if _clean_text(item)]
        if graph_facts:
            block_lines.append("图谱事实：")
            for fact in graph_facts[:8]:
                block_lines.append(f"- {fact}")
        elif bundle.get("summary_preview"):
            summary = _trim_to_useful_excerpt(bundle.get("summary_preview"), _extract_keywords(bundle.get("summary_preview")), max_chars=max_summary_chars)
            if summary:
                block_lines.append("检索摘要：")
                block_lines.append(f"- {summary}")

        block = "\n".join(block_lines).strip()
        if not block:
            continue
        if used and used + len(block) > max_total_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def snippets_from_evidence_bundles(
    bundles: list[dict] | None,
    cited_indices: list[int] | None = None,
    *,
    max_snippets_per_citation: int = 4,
) -> dict[int, list[str]]:
    """Return display snippets keyed by citation number from evidence bundles."""
    if not bundles:
        return {}
    wanted = set(cited_indices or [])
    result: dict[int, list[str]] = {}
    for bundle in bundles:
        number = bundle.get("number")
        try:
            idx = int(number)
        except Exception:
            continue
        if wanted and idx not in wanted:
            continue
        snippets = []
        seen: set[str] = set()
        for item in bundle.get("original_excerpts") or []:
            text = _clean_text(item.get("text") if isinstance(item, dict) else item)
            key = _normalized_excerpt_key(text)
            if text and key not in seen:
                seen.add(key)
                snippets.append(text)
            if len(snippets) >= max_snippets_per_citation:
                break
        if snippets:
            result[idx] = snippets
    return result


def extract_cited_snippets(
    answer: str,
    sources: list[dict],
    snippet_fetcher: Callable[[str, int], list[str]],
) -> dict[int, list[str]]:
    if not answer or not sources:
        return {}

    index_to_source = {
        int(source.get("index")): source
        for source in sources
        if isinstance(source.get("index"), int) or str(source.get("index", "")).isdigit()
    }

    snippets_by_index: dict[int, list[str]] = {}
    for index in extract_cited_indices(answer):
        source = index_to_source.get(index)
        if not source:
            continue

        citation_context = _citation_context(answer, index)
        content_preview = _clean_text(source.get("content_preview"))
        keywords = _extract_keywords(
            citation_context,
            source.get("entity_name"),
            source.get("raw_entity_name"),
            content_preview,
        )
        anchors = _extract_evidence_anchors(citation_context, content_preview)
        source_text = _clean_text(source.get("source_text"))

        edge_items: list[dict[str, str]] = []
        for edge in source.get("subgraph_edges") or []:
            ev = _clean_text(edge.get("evidence") or "")
            edge_source_text = _clean_text(edge.get("source_text") or "")
            edge_source_chunk_id = _clean_text(edge.get("source_chunk_id") or "")
            if ev and len(ev) >= 6:
                found = False
                for item in edge_items:
                    if item["evidence"] == ev:
                        found = True
                        break
                if not found:
                    edge_items.append({
                        "evidence": ev,
                        "source_text": edge_source_text,
                        "source_chunk_id": edge_source_chunk_id,
                    })

        snippets: list[str] = []
        if edge_items:
            seen_normalized: set[str] = set()
            for item in edge_items[:8]:
                ev = item["evidence"]
                from_ev = _clean_text(ev)
                edge_src = item.get("source_text", "")

                # 1) edge has its own source_text
                if edge_src:
                    sentences = [p.strip() for p in re.split(r"(?<=[。！？；])", _clean_text(edge_src)) if p.strip()]
                    ev_kw = _extract_keywords(ev)[:6]
                    if sentences and ev_kw:
                        scored = sorted(
                            ((sum(1 for kw in ev_kw if kw and kw in s), -i, s)
                             for i, s in enumerate(sentences)
                             if sum(1 for kw in ev_kw if kw and kw in s) >= 1),
                            reverse=True,
                        )
                        if scored:
                            best_idx = -scored[0][1]
                            parts = []
                            for pos in range(max(0, best_idx - 1), min(len(sentences), best_idx + 2)):
                                if sum(len(p) for p in parts) + len(sentences[pos]) <= 400:
                                    parts.append(sentences[pos])
                            candidate = re.sub(r"\s+", " ", "".join(parts)).strip()
                            if candidate and len(candidate) >= 12:
                                normalized = re.sub(r"\s+", "", candidate)
                                if normalized not in seen_normalized:
                                    seen_normalized.add(normalized)
                                    snippets.append(candidate)
                                continue

                # 2) search Neo4j TextChunk
                lookup = item.get("source_chunk_id", "") or ev
                fetched = snippet_fetcher(lookup, 3)
                found_in_chunk = False
                if fetched:
                    ev_kw = _extract_keywords(ev)[:8]
                    for full_text in fetched:
                        text = _clean_text(full_text)
                        if not text or not ev_kw:
                            continue
                        sentences = [p.strip() for p in re.split(r"(?<=[。！？；])", text) if p.strip()]
                        if not sentences:
                            sentences = [text]
                        scored = sorted(
                            ((sum(1 for kw in ev_kw if kw and kw in s), -i, s)
                             for i, s in enumerate(sentences)
                             if sum(1 for kw in ev_kw if kw and kw in s) >= 1),
                            reverse=True,
                        )
                        if scored:
                            best_idx = -scored[0][1]
                            parts = []
                            for pos in range(max(0, best_idx - 1), min(len(sentences), best_idx + 3)):
                                if sum(len(p) for p in parts) + len(sentences[pos]) <= 440:
                                    parts.append(sentences[pos])
                            candidate = re.sub(r"\s+", " ", "".join(parts)).strip()
                            if candidate and len(candidate) >= 12:
                                normalized = re.sub(r"\s+", "", candidate)
                                if normalized not in seen_normalized:
                                    seen_normalized.add(normalized)
                                    snippets.append(candidate)
                                    found_in_chunk = True
                                    break

                # 3) fallback: evidence text itself (IS original excerpt)
                if not found_in_chunk and from_ev:
                    normalized = re.sub(r"\s+", "", from_ev)
                    if normalized not in seen_normalized:
                        seen_normalized.add(normalized)
                        snippets.append(from_ev)
        elif source_text:
            snippets = [_trim_to_useful_excerpt(source_text, keywords, anchors=anchors, max_chars=420)]

        if snippets:
            snippets_by_index[index] = snippets[:4]
            continue

        for lookup_key in _lookup_candidates(source):
            snippets = [item.strip() for item in snippet_fetcher(lookup_key, 2) if str(item or "").strip()]
            if snippets:
                snippets_by_index[index] = [_trim_to_useful_excerpt(item, keywords, anchors=anchors) for item in snippets[:2]]
                break

        if index not in snippets_by_index:
            if content_preview:
                snippets_by_index[index] = [_trim_to_useful_excerpt(content_preview, keywords, anchors=anchors)]

    return snippets_by_index
