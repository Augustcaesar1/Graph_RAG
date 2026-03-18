"""
生成集成模块 - 东周列国历史知识图谱版
"""

import logging
import os
import time
from typing import List, Tuple, Dict

from openai import OpenAI
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class GenerationIntegrationModule:
    """生成集成模块 - 负责答案生成"""

    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-V3", temperature: float = 0.1, max_tokens: int = 2048):
        from config import DEFAULT_CONFIG

        self.model_name  = getattr(DEFAULT_CONFIG, 'llm_model', model_name)
        self.temperature = getattr(DEFAULT_CONFIG, 'temperature', temperature)
        self.max_tokens  = getattr(DEFAULT_CONFIG, 'max_tokens', max_tokens)

        api_key = os.getenv("SILICONFLOW_API_KEY") or os.getenv("MOONSHOT_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("请设置 SILICONFLOW_API_KEY 环境变量")

        api_base = getattr(DEFAULT_CONFIG, 'llm_api_base', "https://api.siliconflow.cn/v1")

        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            default_headers={
                "User-Agent": "Mozilla/5.0",
                "Accept":     "application/json"
            }
        )

        from langchain_openai import OpenAIEmbeddings
        embedding_model = getattr(DEFAULT_CONFIG, 'embedding_model', "BAAI/bge-m3")
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=api_key,
            openai_api_base=api_base,
            model=embedding_model,
            chunk_size=64
        )

        logger.info(f"生成模块初始化完成，API: {api_base}, 模型: {self.model_name}")

    def _build_structured_context_with_citations(self, documents: List[Document]) -> Tuple[str, List[Dict]]:
        """构建带引用编号的结构化上下文"""
        context_parts = []
        citations     = []

        for i, doc in enumerate(documents):
            content = doc.page_content.strip()
            if content:
                citation_number = i + 1
                level  = doc.metadata.get('retrieval_level', '')
                source = doc.metadata.get('source', '未知来源')
                context_parts.append(f"[{citation_number}] {content}")
                citations.append({
                    "number": citation_number,
                    "content": content,
                    "source": source,
                    "retrieval_level": level
                })

        return "\n\n".join(context_parts), citations

    # ─────────────────────────────────────────────────────────────
    def generate_adaptive_answer(self, question: str, documents: List[Document]) -> str:
        """智能统一答案生成 - 东周列国历史专家"""
        context, citations = self._build_structured_context_with_citations(documents)

        prompt = f"""你是一位博学的东周列国历史学家，精通春秋战国时期的人物、事件与政治格局。
请基于以下检索到的知识图谱信息，准确、生动地回答用户的问题。

严格遵守以下规则：
1. 仅使用提供的"检索到的相关信息"来回答，不引入未经证实的外部知识。
2. 信息不足以完整回答时，请明确说明。
3. 直接引用时使用方括号引用编号，例如 [1]、[2] 等。
4. 确保历史严谨性和生动性。
5. 根据问题性质：
   - 询问人物 → 描述生平、所属国家、身份地位及重要事迹
   - 询问战争/事件 → 说明时间、地点、交战双方、起因与结果
   - 询问人物关系 → 详细说明关系类型和历史背景
   - 综合性问题 → 提供全面的历史分析

检索到的相关信息：
{context}

用户问题：{question}

回答："""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"抱歉，生成回答时出现错误：{str(e)}"

    # ─────────────────────────────────────────────────────────────
    def generate_adaptive_answer_stream(self, question: str, documents: List[Document], max_retries: int = 3):
        """流式答案生成（带重试）"""
        context_parts = []
        for doc in documents:
            content = doc.page_content.strip()
            if content:
                level = doc.metadata.get('retrieval_level', '')
                context_parts.append(f"[{level.upper()}] {content}" if level else content)
        context = "\n\n".join(context_parts)

        prompt = f"""作为一位博学的东周列国历史学家，请基于以下知识图谱信息回答用户的问题。

检索到的相关信息：
{context}

用户问题：{question}

请提供准确、生动的历史回答，根据问题性质：
- 询问人物 → 描述生平、国籍、身份和事迹
- 询问战争 → 说明时间、交战双方、起因和结果
- 询问关系 → 说明关系类型和历史背景

回答："""

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True,
                    timeout=60
                )
                if attempt == 0:
                    print("开始流式生成回答...\n")
                else:
                    print(f"第{attempt + 1}次尝试流式生成...\n")

                for chunk in response:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        yield content
                return

            except Exception as e:
                logger.warning(f"流式生成第{attempt + 1}次失败: {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"连接中断，{wait_time}秒后重试...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error("流式生成完全失败，切换到标准模式")
                    try:
                        fallback = self.generate_adaptive_answer(question, documents)
                        yield fallback
                    except Exception as fe:
                        yield f"抱歉，生成回答时出现网络错误，请稍后重试。错误信息：{str(e)}"
                    return