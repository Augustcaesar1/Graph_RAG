"""
预生成四个演示用例的答案（含原文引用编号）。
输出到文件，避免终端乱码。
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["NEO4J_PASSWORD"] = "12345678"

from dotenv import load_dotenv
load_dotenv()

out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "pregen_results.txt")
f = open(out_path, "w", encoding="utf-8")

from config import DEFAULT_CONFIG
from rag_modules.bootstrap import build_rag_system

config = DEFAULT_CONFIG

rag = build_rag_system(config=config)
generator = rag.gen_module
router = rag.router

DEMO_QUERIES = [
    "纣王是谁？",
    "哪吒的师父是谁？他属于哪个教派？有哪些法宝？",
    "姜子牙和元始天尊是什么关系？",
    "孙悟空在《封神演义》中有什么法宝？",
]

for qi, question in enumerate(DEMO_QUERIES):
    f.write(f"\n{'='*80}\n")
    f.write(f"用例 {qi+1}: {question}\n")
    f.write(f"{'='*80}\n\n")

    # 路由 + 检索
    documents, analysis = router.route_query(question, top_k=5)
    f.write(f"策略: {analysis.recommended_strategy.value} | 结果数: {len(documents)}\n\n")

    # 展示每个 Document 的关键信息
    for i, doc in enumerate(documents):
        meta = doc.metadata or {}
        f.write(f"--- Doc[{i}] entity={meta.get('entity_name')} type={meta.get('search_type')} "
                f"score={meta.get('relevance_score')}\n")
        edges = meta.get("subgraph_edges") or []
        for e in edges[:10]:
            f.write(f"  {e.get('source')} --[{e.get('relation')}]--> {e.get('target')}\n")
            ev = str(e.get("evidence") or "")
            if ev: f.write(f"    证据: {ev[:150]}\n")
            st = str(e.get("source_text") or "")
            if st: f.write(f"    原文: {st[:200]}\n")
            cid = str(e.get("source_chunk_id") or "")
            if cid: f.write(f"    chunk: {cid}\n")

    # 生成答案
    answer = generator.generate_adaptive_answer(question, documents)
    f.write(f"\n===== 生成答案 =====\n")
    f.write(answer + "\n")
    f.write(f"=====================\n")

f.close()
print(f"结果已写入: {out_path}")
