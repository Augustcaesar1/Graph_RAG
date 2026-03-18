"""
东周列国知识图谱 → Neo4j 导入脚本
数据来源：
  - 人物信息.xlsx  ：436 个 Person 节点
  - 事件信息.xlsx  ：21 个 War/Event 节点
  - 人物关系.xlsx  ：613 条人物关系边
"""

import os
import re
import pandas as pd
from neo4j import GraphDatabase

# ─── 配置 ───────────────────────────────────────────────────
NEO4J_URI      = "bolt://127.0.0.1:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "12345678"
DATA_DIR       = r"C:\Users\21707\Desktop\C9\东周列国知识图谱"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# ─── 工具函数 ────────────────────────────────────────────────
def safe_str(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip()

def safe_int(v):
    try:
        return int(v)
    except Exception:
        return None

def clean_name(raw: str) -> str:
    """去掉 '齐国-齐桓公' 格式中的国名前缀，只保留人名"""
    if not raw:
        return ""
    parts = [p.strip() for p in raw.split("-")]
    return parts[-1] if len(parts) > 1 else parts[0]

# ─── Step 0：清空旧数据 ───────────────────────────────────────
def clear_database(session):
    print("[WARN] 清除旧节点与关系...")
    session.run("MATCH (n) DETACH DELETE n")
    print("[OK] 数据库已清空")

# ─── Step 1：导入人物 Person 节点 ────────────────────────────
def import_persons(session, df: pd.DataFrame):
    print(f"\n[LOAD] 导入人物节点（{len(df)} 行）...")
    created = 0
    for _, row in df.iterrows():
        xing    = safe_str(row.get("姓", ""))
        shi     = safe_str(row.get("氏", ""))
        ming    = safe_str(row.get("名", ""))
        hao     = safe_str(row.get("谥号", ""))      # 谥号/称呼作为主显示名
        guo     = safe_str(row.get("国籍", ""))
        danwei  = safe_str(row.get("工作单位", ""))
        shishi  = safe_str(row.get("生活时间", ""))
        gongshi = safe_str(row.get("工作时间", ""))
        mingzhu = safe_str(row.get("是否国君", ""))
        beizhu  = safe_str(row.get("备注", ""))

        # 优先用谥号作为主名，其次用"氏+名"
        display_name = hao if hao else (shi + ming if shi or ming else xing + ming)
        if not display_name:
            continue  # 跳过无名节点

        # 时间转int
        life_year = safe_int(shishi)

        session.run("""
            MERGE (p:Person {name: $name})
            SET p.xing      = $xing,
                p.shi       = $shi,
                p.ming      = $ming,
                p.hao       = $hao,
                p.nationality = $guo,
                p.state     = $guo,
                p.danwei    = $danwei,
                p.life_year = $life_year,
                p.work_time = $gongshi,
                p.is_king   = $mingzhu,
                p.note      = $beizhu,
                p.description = $desc
        """, {
            "name":    display_name,
            "xing":    xing,
            "shi":     shi,
            "ming":    ming,
            "hao":     hao,
            "guo":     guo,
            "danwei":  danwei,
            "life_year": life_year,
            "gongshi": gongshi,
            "mingzhu": mingzhu,
            "beizhu":  beizhu,
            "desc":    f"{display_name}，{guo}人物，{'国君' if mingzhu=='是' else '臣子'}。{beizhu}"
        })
        created += 1

    print(f"[OK] 人物节点导入完成，共 {created} 个 Person")
    return created


# ─── Step 2：创建 State（国家）节点 ──────────────────────────
def import_states(session, df: pd.DataFrame):
    print("\n[LOAD] 提取并创建国家节点...")
    states = set()
    for _, row in df.iterrows():
        g = safe_str(row.get("国籍", ""))
        if g:
            states.add(g)

    for state in states:
        session.run("""
            MERGE (s:State {name: $name})
            SET s.description = $name + '是东周时期一个重要的诸侯国。'
        """, {"name": state})

    # Person -[BELONGS_TO]-> State
    session.run("""
        MATCH (p:Person), (s:State)
        WHERE p.state = s.name AND NOT (p)-[:BELONGS_TO]->(s)
        CREATE (p)-[:BELONGS_TO]->(s)
    """)
    print(f"[OK] 国家节点创建完成，共 {len(states)} 个 State，已建立 BELONGS_TO 关系")


# ─── Step 3：导入战争/事件 Event 节点 ────────────────────────
def import_events(session, df: pd.DataFrame):
    print(f"\n[LOAD] 导入事件节点（{len(df)} 行）...")
    created = 0
    for _, row in df.iterrows():
        event_id   = safe_str(row.get("事件编码", ""))
        name       = safe_str(row.get("战争名称", ""))
        time_start = safe_int(row.get("时间起", None))
        time_end   = safe_int(row.get("时间止", None))
        location   = safe_str(row.get("地点", ""))
        attacker   = safe_str(row.get("主攻方", ""))
        defender   = safe_str(row.get("主守方", ""))
        atk_help   = safe_str(row.get("主攻-帮手", ""))
        def_help   = safe_str(row.get("主守-帮手", ""))
        atk_force  = safe_str(row.get("主攻方-兵力", ""))
        def_force  = safe_str(row.get("主守方-兵力", ""))
        atk_loss   = safe_str(row.get("主攻方-伤亡", ""))
        def_loss   = safe_str(row.get("主守方-伤亡", ""))
        cause      = safe_str(row.get("起因", ""))
        result     = safe_str(row.get("结果", ""))
        content    = safe_str(row.get("内容", ""))

        if not name:
            continue

        desc = f"{name}（{time_start}年{'至'+str(time_end)+'年' if time_end else ''}）发生于{location}。{cause}。结果：{result}。{content}"

        session.run("""
            MERGE (e:Event {event_id: $event_id})
            SET e.name        = $name,
                e.time_start  = $time_start,
                e.time_end    = $time_end,
                e.location    = $location,
                e.attacker    = $attacker,
                e.defender    = $defender,
                e.atk_help    = $atk_help,
                e.def_help    = $def_help,
                e.atk_force   = $atk_force,
                e.def_force   = $def_force,
                e.atk_loss    = $atk_loss,
                e.def_loss    = $def_loss,
                e.cause       = $cause,
                e.result      = $result,
                e.content     = $content,
                e.description = $desc,
                e.event_type  = '战争'
        """, {
            "event_id": event_id, "name": name,
            "time_start": time_start, "time_end": time_end,
            "location": location, "attacker": attacker, "defender": defender,
            "atk_help": atk_help, "def_help": def_help,
            "atk_force": atk_force, "def_force": def_force,
            "atk_loss": atk_loss, "def_loss": def_loss,
            "cause": cause, "result": result, "content": content, "desc": desc
        })
        created += 1

        # ── 建立 Person-[:ATTACKED_IN]->Event  Person-[:DEFENDED_IN]->Event
        def link_persons_to_event(raw_str: str, rel_type: str):
            """解析主攻/主守方字符串，建立人物-事件关系"""
            if not raw_str:
                return
            # 格式: "齐国-齐桓公；宋国-宋闵公" 或 "齐国-齐桓公"
            parts = re.split(r'[；;,，]', raw_str)
            for part in parts:
                person_name = clean_name(part.strip())
                if not person_name:
                    continue
                session.run(f"""
                    MATCH (p:Person {{name: $pname}}), (e:Event {{event_id: $eid}})
                    MERGE (p)-[:{rel_type}]->(e)
                """, {"pname": person_name, "eid": event_id})

        link_persons_to_event(attacker, "ATTACKED_IN")
        link_persons_to_event(defender, "DEFENDED_IN")
        link_persons_to_event(atk_help, "ASSISTED_ATTACK_IN")
        link_persons_to_event(def_help, "ASSISTED_DEFEND_IN")

    print(f"[OK] 事件节点导入完成，共 {created} 个 Event")


# ─── Step 4：导入人物关系 ─────────────────────────────────────
RELATION_MAP = {
    "朋友": "FRIEND_OF",
    "好友": "FRIEND_OF",
    "盟友": "ALLY_OF",
    "师生": "TEACHER_STUDENT",
    "老师": "TEACHER_OF",
    "弟子": "DISCIPLE_OF",
    "父子": "FATHER_SON",
    "父女": "FATHER_DAUGHTER",
    "兄弟": "SIBLING",
    "姐妹": "SIBLING",
    "兄妹": "SIBLING",
    "姐弟": "SIBLING",
    "君臣": "LORD_MINISTER",
    "主仆": "MASTER_SERVANT",
    "对手": "RIVAL_OF",
    "敌人": "ENEMY_OF",
    "夫妻": "SPOUSE",
    "妻子": "SPOUSE",
    "丈夫": "SPOUSE",
    "情人": "LOVER_OF",
    "母子": "MOTHER_SON",
    "母女": "MOTHER_DAUGHTER",
}

def get_rel_type(rel_str: str) -> str:
    rel_str = rel_str.strip()
    for k, v in RELATION_MAP.items():
        if k in rel_str:
            return v
    return "RELATED_TO"

def import_relations(session, df: pd.DataFrame):
    print(f"\n[LOAD] 导入人物关系（{len(df)} 行）...")
    created = 0
    skipped = 0
    for _, row in df.iterrows():
        p1  = safe_str(row.get("人物1", ""))
        p2  = safe_str(row.get("人物2", ""))
        rel = safe_str(row.get("关系", ""))

        if not p1 or not p2 or not rel:
            skipped += 1
            continue

        # 支持多关系，如 "兄妹|情人"
        rel_parts = [r.strip() for r in rel.split("|")]
        for rp in rel_parts:
            rel_type = get_rel_type(rp)
            session.run(f"""
                MATCH (a:Person {{name: $p1}}), (b:Person {{name: $p2}})
                MERGE (a)-[r:{rel_type} {{label: $label}}]->(b)
            """, {"p1": p1, "p2": p2, "label": rp})
            created += 1

    print(f"[OK] 人物关系导入完成，共 {created} 条关系（跳过 {skipped} 条空数据）")


# ─── Step 5：创建全文索引（加速检索）────────────────────────
def create_indexes(session):
    print("\n📐 创建索引...")
    indexes = [
        "CREATE INDEX person_name_idx IF NOT EXISTS FOR (p:Person) ON (p.name)",
        "CREATE INDEX event_name_idx IF NOT EXISTS FOR (e:Event) ON (e.name)",
        "CREATE INDEX state_name_idx IF NOT EXISTS FOR (s:State) ON (s.name)",
        "CREATE INDEX person_state_idx IF NOT EXISTS FOR (p:Person) ON (p.state)",
    ]
    for idx in indexes:
        try:
            session.run(idx)
        except Exception as ex:
            print(f"  索引已存在或创建失败: {ex}")
    print("[OK] 索引创建完成")


# ─── 主流程 ──────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  东周列国知识图谱 → Neo4j 导入工具")
    print("=" * 60)

    df_persons  = pd.read_excel(os.path.join(DATA_DIR, "人物信息.xlsx"))
    df_events   = pd.read_excel(os.path.join(DATA_DIR, "事件信息.xlsx"))
    df_relations= pd.read_excel(os.path.join(DATA_DIR, "人物关系.xlsx"))

    print(f"[OK] 读取数据：人物 {len(df_persons)} 行，事件 {len(df_events)} 行，关系 {len(df_relations)} 行")

    with driver.session() as session:
        clear_database(session)
        import_persons(session, df_persons)
        import_states(session, df_persons)
        import_events(session, df_events)
        import_relations(session, df_relations)
        create_indexes(session)

        # ── 汇总统计 ───────────────────────────────────────────
        print("\n📊 数据库最终统计：")
        for label in ["Person", "Event", "State"]:
            cnt = session.run(f"MATCH (n:{label}) RETURN count(n) as c").single()["c"]
            print(f"  {label}: {cnt} 个节点")
        rel_cnt = session.run("MATCH ()-[r]->() RETURN count(r) as c").single()["c"]
        print(f"  关系总计: {rel_cnt} 条")

    driver.close()
    print("\n🎉 导入完成！")

if __name__ == "__main__":
    main()
