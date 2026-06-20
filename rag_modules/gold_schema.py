"""Shared schema helpers for the 封神演义 knowledge graph."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

# ── 封神演义实体类型 ──
ENTITY_TYPES = {
    "Person":        "人物",
    "Faction":       "教派/势力",
    "Location":      "地点",
    "Artifact":      "法宝",
    "Beast":         "坐骑/灵兽",
    "Formation":     "阵法",
    "Event":         "事件/战役",
    "DeityPosition": "神位/封号",
}

# ── 封神演义关系类型 ──
ALLOWED_RELATION_TYPES = {
    # 师承关系（分层核心）
    "MASTER_OF",
    "APPRENTICE_OF",
    # 教派归属（分层核心）
    "BELONGS_TO_SECT",
    # 阵营归属
    "FIGHTS_FOR",
    # 血缘关系
    "FATHER_OF",
    "CHILD_OF",
    "BROTHER_OF",
    "MARRIED_TO",
    # 对抗关系
    "KILLS",
    "DEFEATS",
    "CAPTURES",
    "OPPOSES",
    # 法宝/坐骑流转
    "OWNS",
    "BESTOWS",
    "LOSES",
    "STEALS",
    # 阵法/事件相关
    "CREATES",
    "DEPLOYS",
    "BREAKS",
    "PARTICIPATES_IN",
    "OCCURS_IN",
    "LEADS",
    "INITIATES",
    # 封神结局
    "LISTED_ON",    # 上榜封神
    "BECOMES",      # 肉身成圣/成神
    # 通用
    "ALLIES_WITH",
    "BETRAYS",
    "RELATED_TO",
    "MENTIONS",
}

ALLOWED_ASSERTION_TYPES = {
    "explicit",
    "inferred_strict",
    "inferred_loose",
}

ENTITY_ID_FIELDS = {
    "Person":        "person_id",
    "Faction":       "faction_id",
    "Location":      "location_id",
    "Artifact":      "artifact_id",
    "Beast":         "beast_id",
    "Formation":     "formation_id",
    "Event":         "event_id",
    "DeityPosition": "deity_position_id",
}

ENTITY_NAME_FIELDS = {
    "Person":        "name",
    "Faction":       "name",
    "Location":      "name",
    "Artifact":      "name",
    "Beast":         "name",
    "Formation":     "name",
    "Event":         "name",
    "DeityPosition": "name",
}

# ── 中文关系标签 ──
RELATION_LABELS_ZH = {
    "MASTER_OF":        "师父",
    "APPRENTICE_OF":    "徒弟",
    "BELONGS_TO_SECT":  "教派归属",
    "FIGHTS_FOR":       "效力于",
    "FATHER_OF":        "父亲",
    "CHILD_OF":         "子女",
    "BROTHER_OF":       "兄弟",
    "MARRIED_TO":       "婚配",
    "KILLS":            "击杀",
    "DEFEATS":          "击败",
    "CAPTURES":         "擒获",
    "OPPOSES":           "对抗",
    "OWNS":             "拥有",
    "BESTOWS":          "赐予",
    "LOSES":            "失去",
    "STEALS":           "盗取",
    "CREATES":          "布阵",
    "DEPLOYS":          "部署",
    "BREAKS":           "破阵",
    "PARTICIPATES_IN":  "参战",
    "OCCURS_IN":        "发生于",
    "LEADS":            "率领",
    "INITIATES":        "发起",
    "LISTED_ON":        "上榜封神",
    "BECOMES":          "肉身成圣",
    "ALLIES_WITH":      "同盟",
    "BETRAYS":          "背叛",
    "RELATED_TO":       "相关",
    "MENTIONS":         "提及",
}

# ── 结局类型 ──
FATE_TYPES = {
    "listed":          "上榜封神",
    "ascended":        "肉身成圣",
    "absorbed_west":   "归入西方教",
    "died_mortal":     "战死/凡人",
    "unknown":         "未详",
}


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def fengshen_data_root() -> Path:
    return project_root() / "data" / "fengshen"


def extraction_cache_path() -> Path:
    p = fengshen_data_root() / "extraction_results"
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json_file(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return data


def save_json_file(path: Path, data: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
