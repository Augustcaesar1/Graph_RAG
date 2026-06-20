"""
基于图数据库的RAG系统配置文件 - 封神演义版
"""

import os
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class GraphRAGConfig:
    """基于图数据库的RAG系统配置类"""

    # Neo4j数据库配置
    neo4j_uri: str = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")

    # FAISS配置
    vector_store_type: str = "faiss"
    faiss_index_path: str = "./fengshen_faiss_index"

    # 模型配置 (硅基流动 API)
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    llm_api_base: str = os.getenv("LLM_API_BASE", "https://api.siliconflow.cn/v1")
    llm_model: str = os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-V3")

    # 检索配置
    top_k: int = 5
    graph_top_k: int = 10

    # 生成配置
    temperature: float = 0.1
    max_tokens: int = 2048

    # 图数据处理配置
    chunk_size: int = 800
    chunk_overlap: int = 100
    max_graph_depth: int = 3

    # 封神演义文本配置
    fengshen_text_path: str = "./封神演义.txt"
    fengshen_extraction_chunk_size: int = 2500
    fengshen_extraction_chunk_overlap: int = 200

    # 社区检测配置
    enable_community_detection: bool = True
    community_max_nodes: int = 1500
    community_level1_min_size: int = 3
    community_level2_min_size: int = 2
    community_level3_min_size: int = 1

    def __post_init__(self):
        pass

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GraphRAGConfig':
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'neo4j_uri': self.neo4j_uri,
            'neo4j_user': self.neo4j_user,
            'neo4j_password': self.neo4j_password,
            'neo4j_database': self.neo4j_database,
            'vector_store_type': self.vector_store_type,
            'faiss_index_path': self.faiss_index_path,
            'embedding_model': self.embedding_model,
            'llm_api_base': self.llm_api_base,
            'llm_model': self.llm_model,
            'top_k': self.top_k,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'max_graph_depth': self.max_graph_depth,
            'fengshen_text_path': self.fengshen_text_path,
            'fengshen_extraction_chunk_size': self.fengshen_extraction_chunk_size,
            'fengshen_extraction_chunk_overlap': self.fengshen_extraction_chunk_overlap,
            'enable_community_detection': self.enable_community_detection,
        }


DEFAULT_CONFIG = GraphRAGConfig()
