import hashlib
import json
import logging
import os
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


logger = logging.getLogger(__name__)


class FaissIndexConstructionModule:
    """FAISS 向量索引构建与检索模块。"""

    def __init__(
        self,
        persist_directory: str = "./faiss_index",
        embeddings=None,
    ):
        self.persist_directory = persist_directory
        self.embeddings = embeddings
        self.vector_store = None

        self.load_collection()

    def _scope_manifest_path(self) -> Path:
        return Path(self.persist_directory) / "scope_manifest.json"

    def _build_scope_fingerprint(self, documents: List[Document]) -> str:
        normalized = []
        for doc in documents or []:
            metadata = doc.metadata or {}
            normalized.append(
                {
                    "node_id": str(metadata.get("node_id") or metadata.get("entity_name") or ""),
                    "node_type": str(metadata.get("node_type") or metadata.get("doc_type") or ""),
                    "gold_source": str(metadata.get("gold_source") or ""),
                    "chapter": str(metadata.get("chapter") or ""),
                    "period": str(metadata.get("period") or ""),
                    "source_doc_type": str(metadata.get("source_doc_type") or metadata.get("doc_type") or ""),
                }
            )

        payload = json.dumps(
            sorted(
                normalized,
                key=lambda item: (
                    item["node_id"],
                    item["node_type"],
                    item["chapter"],
                    item["period"],
                    item["source_doc_type"],
                ),
            ),
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def write_scope_manifest(self, documents: List[Document]) -> None:
        manifest_path = self._scope_manifest_path()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "scope": "manual_gold_first_three_chapters",
            "document_count": len(documents or []),
            "fingerprint": self._build_scope_fingerprint(documents),
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    def scope_matches_documents(self, documents: List[Document]) -> bool:
        manifest_path = self._scope_manifest_path()
        if not manifest_path.exists():
            return False

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"读取 FAISS 作用域清单失败: {exc}")
            return False

        expected_fingerprint = self._build_scope_fingerprint(documents)
        return (
            manifest.get("scope") == "manual_gold_first_three_chapters"
            and int(manifest.get("document_count", -1)) == len(documents or [])
            and manifest.get("fingerprint") == expected_fingerprint
        )

    def load_collection(self, expected_documents: List[Document] | None = None) -> bool:
        """从本地加载 FAISS 索引。"""
        if os.path.exists(self.persist_directory):
            if expected_documents is not None and not self.scope_matches_documents(expected_documents):
                logger.warning("检测到 FAISS 索引与当前前三章 manual_gold 作用域不一致，将拒绝复用旧索引")
                self.vector_store = None
                return False

            try:
                self.vector_store = FAISS.load_local(
                    self.persist_directory,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                logger.info(f"成功从 {self.persist_directory} 加载 FAISS 索引")
                return True
            except Exception as e:
                logger.warning(f"未能加载 FAISS 索引: {e}")
                self.vector_store = None
                return False
        return False

    def has_collection(self, expected_documents: List[Document] | None = None) -> bool:
        """检查是否已有可用索引。"""
        if expected_documents is not None:
            return os.path.exists(self.persist_directory) and self.scope_matches_documents(expected_documents)
        return self.vector_store is not None or os.path.exists(self.persist_directory)

    def build_vector_index(self, documents: List[Document]) -> bool:
        """构建向量索引并保存。"""
        logger.info(f"开始使用 FAISS 构建向量索引，文档数量: {len(documents)}")
        if not documents:
            logger.warning("没有提供用于构建索引的文档")
            self.vector_store = None
            return False

        try:
            self.vector_store = None
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            self.vector_store.save_local(self.persist_directory)
            self.write_scope_manifest(documents)
            logger.info(f"构建完成，索引已保存至 {self.persist_directory}")
            return True
        except Exception as e:
            logger.error(f"构建 FAISS 索引失败: {e}")
            self.vector_store = None
            return False

    def get_retriever(self, search_kwargs: dict = None):
        """获取 LangChain 兼容的检索器。"""
        if not self.vector_store:
            raise ValueError("FAISS 索引还未初始化")

        search_kwargs = search_kwargs or {"k": 3}
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)

    def similarity_search_with_score(self, query: str, k: int = 3) -> List[tuple]:
        """执行带分数的相似度搜索。"""
        if not self.vector_store:
            logger.warning("FAISS 检索器尚未初始化")
            return []

        return self.vector_store.similarity_search_with_score(query, k=k)
