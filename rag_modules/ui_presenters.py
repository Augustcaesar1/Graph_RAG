from __future__ import annotations

from typing import Any, Dict, List


def build_hero_metrics(rag_loaded: bool, stats: Dict[str, Any] | None = None, route_stats: Dict[str, Any] | None = None) -> List[Dict[str, str]]:
    stats = stats or {}
    route_stats = route_stats or {}

    people = int(stats.get("total_persons", 0) or 0)
    events = int(stats.get("total_events", 0) or 0)
    orgs = int(stats.get("total_organizations", 0) or 0)
    total_queries = int(route_stats.get("total_queries", 0) or 0)

    return [
        {"label": "系统状态", "value": "已连接" if rag_loaded else "未启动"},
        {"label": "人物 / 事件 / 组织", "value": f"{people} / {events} / {orgs}"},
        {"label": "累计查询", "value": str(total_queries)},
    ]


def build_answer_badges(strategy: str, docs_count: int, answer_text: str) -> List[Dict[str, str]]:
    answer_text = str(answer_text or "")
    mode = "本地兜底" if "根据当前知识图谱证据" in answer_text else "模型生成"
    return [
        {"label": "策略", "value": str(strategy or "unknown")},
        {"label": "证据", "value": str(int(docs_count or 0))},
        {"label": "模式", "value": mode},
    ]


def build_source_summary(source_list: List[Dict[str, Any]] | None = None, triples: List[Any] | None = None) -> Dict[str, int]:
    return {
        "source_count": len(source_list or []),
        "triple_count": len(triples or []),
    }


def build_workspace_panels() -> List[Dict[str, str]]:
    return [
        {
            "title": "问题范围",
            "body": "优先提问前三章中的关键人物、事件、组织与历史分期，系统会优先调用 manual gold 图谱证据。",
        },
        {
            "title": "证据导向",
            "body": "适合提问历史作用、关系、影响、失败原因、为什么说这类需要组织证据的问题。",
        },
        {
            "title": "核验方式",
            "body": "左侧阅读回答，右侧对照引用片段、图谱关系和检索元信息，适合做复习与事实核验。",
        },
    ]
