import os
import logging
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from rag_modules.generation_integration import GenerationIntegrationModule

logger = logging.getLogger(__name__)

class FaissIndexConstructionModule:
    """FAISS 向量索引构建与检索模块 (替代 Milvus)"""
    
    def __init__(
        self,
        persist_directory: str = "./faiss_index",
        model_name: str = "moonshot-v1-8k"
    ):
        self.persist_directory = persist_directory
        self.Gen_module = GenerationIntegrationModule(model_name=model_name)
        self.embeddings = self.Gen_module.embeddings
        self.vector_store = None
        
        # 尝试加载已有索引
        self.load_collection()
            
    def load_collection(self) -> bool:
        """从本地加载FAISS索引"""
        if os.path.exists(self.persist_directory):
            try:
                self.vector_store = FAISS.load_local(
                    self.persist_directory, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"成功从 {self.persist_directory} 加载 FAISS 索引")
                return True
            except Exception as e:
                logger.warning(f"未能加载 FAISS 索引: {e}")
                return False
        return False
        
    def has_collection(self) -> bool:
        """检查是否有可用索引"""
        return self.vector_store is not None or os.path.exists(self.persist_directory)
        
    def build_vector_index(self, documents: List[Document]) -> bool:
        """构建向量索引并保存"""
        logger.info(f"开始使用 FAISS 构建向量索引，文档数量: {len(documents)}")
        if not documents:
            logger.warning("没有提供用于构建索引的文档")
            return False
            
        try:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            self.vector_store.save_local(self.persist_directory)
            logger.info(f"构建完成，索引已保存至 {self.persist_directory}")
            return True
        except Exception as e:
            logger.error(f"构建 FAISS 索引失败: {e}")
            return False

    def get_retriever(self, search_kwargs: dict = None):
        """获取LangChain兼容的检索器"""
        if not self.vector_store:
            raise ValueError("FAISS 索引还未初始化")
            
        search_kwargs = search_kwargs or {"k": 3}
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)
        
    def similarity_search_with_score(self, query: str, k: int = 3) -> List[tuple]:
        """执行带得分的相似度搜索"""
        if not self.vector_store:
            logger.warning("FAISS 检索器尚未初始化")
            return []
            
        return self.vector_store.similarity_search_with_score(query, k=k)
