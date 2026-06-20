"""
查四个演示案例的 Neo4j 实际数据（精简版：只取关键字段，截断长文本）。
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GraphRAGConfig
from neo4j import GraphDatabase

config = GraphRAGConfig()
config.neo4j_password = os.getenv("NEO4J_PASSWORD", "12345678")

driver = GraphDatabase.driver(
    config.neo4j_uri,
    auth=(config.neo4j_user, config.neo4j_password),
)

def short(v, n=120):
    s = str(v)
    return s if len(s) <= n else s[:n] + "..."

with driver.session() as session:

    # ====== 用例1: 纣王 ======
    print("="*70)
    print("用例1: 纣王是谁？")
    print("="*70)
    r = session.run("""
        MATCH (n:Person {name: '纣王'})
        RETURN n.name AS name, n.description AS desc, n.alias AS alias,
               n.chapter_number AS ch, n.chapter_title AS ch_title
    """).single()
    if r:
        print(f"名称: {r['name']}")
        print(f"简介: {r['desc']}")
        print(f"别名: {short(r['alias'], 200)}")
        print(f"出场: 第{r['ch']}回 {r['ch_title']}")

    print("\n-- 核心关系 (含evidence) --")
    rows = list(session.run("""
        MATCH (n:Person {name: '纣王'})-[r]-(other)
        WHERE coalesce(other.name, '') <> ''
        RETURN coalesce(startNode(r).name, '') AS src,
               type(r) AS rel,
               coalesce(endNode(r).name, '') AS tgt,
               labels(startNode(r)) AS src_lbl,
               labels(endNode(r)) AS tgt_lbl,
               r.evidence AS evidence,
               r.source_chunk_id AS cid
        LIMIT 50
    """))
    for row in rows:
        ev = short(row['evidence'], 100)
        print(f"  {row['src']} --[{row['rel']}]--> {row['tgt']}  | ev: {ev} | cid: {row['cid']}")

    # ====== 用例2: 哪吒 ======
    print("\n" + "="*70)
    print("用例2: 哪吒的师父是谁？属于哪个教派？有哪些法宝？")
    print("="*70)
    r = session.run("""
        MATCH (n:Person {name: '哪吒'})
        RETURN n.name AS name, n.description AS desc, n.alias AS alias,
               n.chapter_number AS ch, n.chapter_title AS ch_title
    """).single()
    if r:
        print(f"名称: {r['name']}")
        print(f"简介: {r['desc']}")
        print(f"别名: {short(r['alias'], 200)}")
        print(f"出场: 第{r['ch']}回 {r['ch_title']}")

    print("\n-- 全部关系 --")
    rows = list(session.run("""
        MATCH (n:Person {name: '哪吒'})-[r]-(other)
        WHERE coalesce(other.name, '') <> ''
        RETURN coalesce(startNode(r).name, '') AS src,
               type(r) AS rel,
               coalesce(endNode(r).name, '') AS tgt,
               labels(startNode(r)) AS src_lbl,
               labels(endNode(r)) AS tgt_lbl,
               r.evidence AS evidence,
               r.source_chunk_id AS cid
        LIMIT 50
    """))
    for row in rows:
        ev = short(row['evidence'], 100)
        print(f"  {row['src']} --[{row['rel']}]--> {row['tgt']} | ev: {ev} | cid: {row['cid']}")

    # 太乙真人相关
    print("\n-- 太乙真人与哪吒的关系 --")
    rows = list(session.run("""
        MATCH (a:Person {name: '太乙真人'})-[r]-(b:Person {name: '哪吒'})
        RETURN coalesce(startNode(r).name,'') AS src,
               type(r) AS rel,
               coalesce(endNode(r).name,'') AS tgt,
               r.evidence AS evidence,
               r.source_chunk_id AS cid
    """))
    for row in rows:
        print(f"  {row['src']} --[{row['rel']}]--> {row['tgt']} | ev: {short(row['evidence'], 200)} | cid: {row['cid']}")

    # ====== 用例3: 姜子牙和元始天尊 ======
    print("\n" + "="*70)
    print("用例3: 姜子牙和元始天尊是什么关系？")
    print("="*70)
    for name in ['姜子牙', '元始天尊']:
        r = session.run("""
            MATCH (n:Person {name: $name})
            RETURN n.name AS name, n.description AS desc, n.alias AS alias
        """, name=name).single()
        if r:
            print(f"名称: {r['name']}")
            print(f"简介: {r['desc']}")
            print(f"别名: {short(r['alias'], 200)}")

    print("\n-- 两人之间的直接关系 --")
    rows = list(session.run("""
        MATCH (a:Person {name: '姜子牙'})-[r]-(b:Person {name: '元始天尊'})
        RETURN coalesce(startNode(r).name,'') AS src,
               type(r) AS rel,
               coalesce(endNode(r).name,'') AS tgt,
               r.evidence AS evidence,
               r.source_chunk_id AS cid
    """))
    for row in rows:
        print(f"  {row['src']} --[{row['rel']}]--> {row['tgt']}")
        print(f"  证据: {short(row['evidence'], 300)}")
        print(f"  cid: {row['cid']}")

    print("\n-- 姜子牙全部关系 --")
    rows = list(session.run("""
        MATCH (n:Person {name: '姜子牙'})-[r]-(other)
        WHERE coalesce(other.name, '') <> ''
        RETURN coalesce(startNode(r).name,'') AS src,
               type(r) AS rel,
               coalesce(endNode(r).name,'') AS tgt,
               r.evidence AS evidence
        LIMIT 25
    """))
    for row in rows:
        print(f"  {row['src']} --[{row['rel']}]--> {row['tgt']} | ev: {short(row['evidence'], 80)}")

    # ====== 用例4: 孙悟空 ======
    print("\n" + "="*70)
    print("用例4: 孙悟空在《封神演义》中有什么法宝？")
    print("="*70)
    for search_type in ['name', 'alias']:
        r = session.run("""
            MATCH (n)
            WHERE coalesce(n.name, '') CONTAINS '孙悟空'
            RETURN coalesce(n.name,'') AS name, labels(n) AS labels
            LIMIT 5
        """).single()
        print(f"  按name搜索孙悟空: {'找到' if r else '无结果'}")
    r = session.run("""
        MATCH (n)
        WHERE coalesce(n.name, '') CONTAINS '孙' AND coalesce(n.name, '') CONTAINS '悟'
        RETURN coalesce(n.name,'') AS name, labels(n) AS labels
        LIMIT 5
    """).single()
    print(f"  模糊搜索'孙'+'悟': {'找到' if r else '无结果'}")

driver.close()
print("\n完成。")
