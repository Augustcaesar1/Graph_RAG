from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import DEFAULT_CONFIG, GraphRAGConfig
from rag_modules import GraphDataPreparationModule, GenerationIntegrationModule
from rag_modules.faiss_index_construction import FaissIndexConstructionModule
from rag_modules.graph_rag_retrieval import GraphRAGRetrieval
from rag_modules.hybrid_retrieval import HybridRetrievalModule
from rag_modules.intelligent_query_router import IntelligentQueryRouter
from rag_modules.question_service import DemoQuestionService
from rag_modules.snippet_service import SourceSnippetService


@dataclass
class RAGSystem:
    data_module: Any
    index_module: Any
    gen_module: Any
    router: Any
    snippet_service: Any
    question_service: Any
    config: GraphRAGConfig

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key: str):
        return getattr(self, key)

    def close(self):
        """Close the shared Neo4j driver."""
        driver = getattr(self, '_shared_driver', None)
        if driver:
            driver.close()
            import logging
            logging.getLogger(__name__).info("Shared Neo4j driver closed")


def build_rag_system(config: GraphRAGConfig | None = None) -> RAGSystem:
    cfg = config or DEFAULT_CONFIG

    # Create a single shared Neo4j driver for all modules
    from neo4j import GraphDatabase
    shared_driver = GraphDatabase.driver(
        cfg.neo4j_uri,
        auth=(cfg.neo4j_user, cfg.neo4j_password),
    )

    data_module = GraphDataPreparationModule(
        config=cfg,
        driver=shared_driver,
    )
    gen_module = GenerationIntegrationModule(
        model_name=cfg.llm_model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        api_base=cfg.llm_api_base,
        embedding_model=cfg.embedding_model,
    )
    index_module = FaissIndexConstructionModule(
        persist_directory=cfg.faiss_index_path,
        embeddings=gen_module.embeddings,
    )
    trad_retrieval = HybridRetrievalModule(
        config=cfg,
        index_module=index_module,
        data_module=data_module,
        llm_client=gen_module.client,
        driver=shared_driver,
    )
    graph_retrieval = GraphRAGRetrieval(config=cfg, llm_client=gen_module.client, driver=shared_driver)
    router = IntelligentQueryRouter(
        traditional_retrieval=trad_retrieval,
        graph_rag_retrieval=graph_retrieval,
        llm_client=gen_module.client,
        config=cfg,
        driver=shared_driver,
    )

    data_module.load_graph_data()
    data_module.build_history_documents()
    chunks = data_module.chunk_documents(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )

    if not chunks:
        from langchain_core.documents import Document

        chunks = [
            Document(
                page_content="欢迎使用《封神演义》知识图谱！请先运行导入脚本填充数据库：python scripts/import_fengshen_to_neo4j.py",
                metadata={"source": "system_init", "entity_name": "系统初始化"},
            )
        ]

    try:
        if not (
            index_module.has_collection(chunks)
            and index_module.load_collection(expected_documents=chunks)
        ):
            index_module.build_vector_index(chunks)
    except Exception as _e:
        import logging
        logging.getLogger(__name__).warning(
            f"FAISS index build/load failed, vector retrieval will be unavailable: {_e}"
        )

    trad_retrieval.initialize(chunks)
    graph_retrieval.initialize()

    system = RAGSystem(
        data_module=data_module,
        index_module=index_module,
        gen_module=gen_module,
        router=router,
        snippet_service=SourceSnippetService(data_module),
        question_service=DemoQuestionService(data_module),
        config=cfg,
    )
    system._shared_driver = shared_driver
    return system
