"""
基于图数据库的RAG系统配置文件 - 东周列国版
"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class GraphRAGConfig:
    """基于图数据库的RAG系统配置类"""

    # Neo4j数据库配置
    neo4j_uri: str = "bolt://127.0.0.1:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "12345678"
    neo4j_database: str = "neo4j"

    # FAISS配置
    vector_store_type: str = "faiss"
    faiss_index_path: str = "./dongzhou_faiss_index"

    # 模型配置 (硅基流动 API)
    embedding_model: str = "BAAI/bge-m3"
    llm_api_base: str = "https://api.siliconflow.cn/v1"
    llm_model: str = "deepseek-ai/DeepSeek-V3"

    # 检索配置
    top_k: int = 5

    # 生成配置
    temperature: float = 0.1
    max_tokens: int = 2048

    # 图数据处理配置
    chunk_size: int = 600
    chunk_overlap: int = 60
    max_graph_depth: int = 3  # 历史关系多跳遍历更有价值

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
            'max_graph_depth': self.max_graph_depth
        }

# 默认配置实例
DEFAULT_CONFIG = GraphRAGConfig()