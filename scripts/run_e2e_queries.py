"""Run a few Fengshen GraphRAG end-to-end queries."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from rag_modules.bootstrap import build_rag_system
from rag_modules.source_evidence import extract_cited_snippets


def main():
    questions = [
        "纣王和女娲有什么关系？",
        "女娲做了什么？",
        "纣王是谁？他的儿子有哪些？",
    ]
    rag = build_rag_system()
    router = rag.router
    gen = rag.gen_module
    data = rag.data_module
    snippet_service = rag.snippet_service

    for q in questions:
        print("\n" + "=" * 90)
        print("Q:", q)
        docs, analysis = router.route_query(q, top_k=5)
        docs = docs or []
        print("Strategy:", analysis.recommended_strategy.value, "complexity=", f"{analysis.query_complexity:.2f}", "relation=", f"{analysis.relationship_intensity:.2f}")
        print("Docs:", len(docs))

        answer = gen.generate_adaptive_answer(q, docs)
        print("\nANSWER:")
        print(answer[:1200])

        sources = []
        entity_names = []
        triples = []
        for i, doc in enumerate(docs):
            md = doc.metadata or {}
            entity_name = md.get("entity_name", "未知")
            entity_names.append(entity_name)
            sources.append({
                "index": i + 1,
                "entity_name": entity_name,
                "node_id": md.get("node_id", ""),
                "chunk_id": md.get("chunk_id", ""),
                "source_chunk_id": md.get("source_chunk_id", md.get("chunk_id", "")),
                "source_text": md.get("source_text", ""),
                "content_preview": doc.page_content,
            })
            triples.extend(md.get("subgraph_edges", []) or [])

        if not triples and entity_names:
            for s, r, t in data.export_triples(entity_names=entity_names[:5], limit=12):
                triples.append({"source": s, "relation": r, "target": t})

        cited = extract_cited_snippets(answer, sources, snippet_service.fetch_original_text_snippets)
        print("\nINLINE SOURCE SNIPPETS:")
        if cited:
            for idx, snippets in cited.items():
                print(f"[{idx}]")
                for snip in snippets[:1]:
                    print(snip[:400].replace("\n", " "))
        else:
            shown = 0
            for src in sources:
                snips = snippet_service.fetch_original_text_snippets(str(src.get("entity_name") or ""), 1)
                if snips:
                    print(f"[{src['index']}]")
                    print(snips[0][:400].replace("\n", " "))
                    shown += 1
                if shown >= 2:
                    break

        print("\nTRIPLES:")
        for edge in triples[:12]:
            print(f"{edge.get('source')} --[{edge.get('relation')}]--> {edge.get('target')}")

    data.close()


if __name__ == "__main__":
    main()
