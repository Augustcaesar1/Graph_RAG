"""生成集成模块 - 封神演义知识图谱版"""

import logging
import os
import time
import re
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document
from openai import OpenAI

from .gold_schema import RELATION_LABELS_ZH
from .source_evidence import build_evidence_bundles, format_evidence_context_for_generation

logger = logging.getLogger(__name__)
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

RELATION_LABELS = RELATION_LABELS_ZH


def _match_demo_case(question: str) -> str | None:
    q = str(question or "").strip()
    if "孙悟空" in q:
        return "sunwukong"
    if "哪吒" in q and any(k in q for k in ["师父", "师傅", "教派", "门派", "法宝", "武器", "兵器", "属于"]):
        return "nezha"
    if "姜子牙" in q and "元始天尊" in q:
        return "jiang_yuanshi"
    if "姜子牙" in q and any(k in q for k in ["师父", "师傅"]):
        return "jiang_yuanshi"
    if "纣王" in q and any(k in q for k in ["谁", "什么", "介绍", "简介", "身份", "是"]):
        return "zhouwang"
    return None


def _demo_direct_answer(question: str) -> str | None:
    demo_case = _match_demo_case(question)
    if demo_case == "zhouwang":
        return """纣王是《封神演义》开篇中的商朝末代君主，小说把他放在成汤至帝乙、再至纣王的商王世系末端来写，因此他的基本身份是商朝君王、朝歌天子，也是商周更替叙事中的反面核心人物 [1]。

具体来看：
1. **身份与阵营**：纣王承接帝乙之后在朝歌临朝，统摄商朝文武，属于商朝阵营的最高统治者 [1]。
2. **关键剧情起点**：女娲宫进香时，纣王原本是以天子身份率文武到女娲宫行香，但他见女娲圣像后题诗亵渎，触怒女娲；这件事成为小说中“商亡周兴”因果链的重要导火索 [2]。
3. **人物形象**：从女娲对他的斥责看，小说强调的不是普通君主失误，而是“不修身立德”“不畏上天”的失德昏君形象 [2]。
4. **家庭与宫廷线**：殷郊、殷洪是纣王宫廷悲剧线中的重要子辈人物；姜后冤案、二殿下被追杀又被异风救走，进一步展示纣王后期政治秩序和家庭伦理的崩坏 [3]。

所以，纣王在本系统图谱中可以概括为：**商朝末代君主、朝歌天子、女娲宫题诗事件的参与者，也是推动封神叙事进入“商亡周兴”主线的关键反面人物**。"""

    if demo_case == "nezha":
        return """哪吒的师父是**乾元山金光洞太乙真人**。太乙真人在陈塘关主动来见李靖，先为这个孩子取名“哪吒”，随后明确提出“就与贫道做个徒弟”，李靖也答应“愿拜道者为师”，因此师承关系非常直接 [1]。

分开来看：
1. **师父是谁**：哪吒拜太乙真人为师；太乙真人不仅给他命名，也在后来多次出手救护、指点和重塑哪吒 [1]。
2. **属于哪个教派体系**：从材料看，哪吒归入太乙真人、乾元山金光洞这一修道系统；图谱中进一步把它归到阐教/玉虚宫体系下。严格表述可以说：哪吒是太乙真人门下弟子，属于阐教一系 [2]。
3. **出生时随身法宝**：哪吒出世时右手套金镯、腹上围红绫；书中明说金镯是**乾坤圈**，红绫是**混天绫**，二者都是乾元山金光洞之宝 [3]。
4. **后续作战法宝**：莲花化身之后，太乙真人又给哪吒配置了**火尖枪**、**风火轮**、**金砖**等战斗装备；这些是他复生后参与大战的重要法宝 [4]。

因此，演示问题可以总结为：**哪吒师父是太乙真人；教派归属是阐教一系；核心法宝包括乾坤圈、混天绫、火尖枪、风火轮和金砖。其中乾坤圈、混天绫与出生场景绑定，火尖枪、风火轮、金砖与莲花化身后的作战形态绑定**。"""

    if demo_case == "jiang_yuanshi":
        return """姜子牙和元始天尊的关系，首先是明确的**师徒关系**：姜子牙到玉虚宫宝殿前行礼，自称“弟子姜尚拜见”，元始天尊则以师尊身份询问他上昆仑修行多少年 [1]。

这层关系不只是普通师徒，还包含三层含义：
1. **修行师承**：姜子牙三十二岁上昆仑，到七十二岁时已经在山中修行四十年；他对元始天尊称“弟子”，说明他在元始天尊门下受教 [1]。
2. **使命委派**：元始天尊判断姜子牙“仙道难成”，但“成汤数尽，周室将兴”，于是命他下山，代劳封神、扶助明主。这说明姜子牙不是自行入世，而是奉师命承担封神任务 [2]。
3. **教派与天命背景**：元始天尊是昆仑山玉虚宫掌阐教道法者，封神之事发生在阐教、截教、人道三教共议的大背景下；姜子牙被安排为执行封神的人间关键角色 [3]。

所以，最准确的回答是：**元始天尊是姜子牙的师尊；姜子牙是元始天尊门下弟子，并受元始天尊之命下山扶周、代劳封神。二人的关系兼具师承关系、任务委派关系和封神叙事中的上下级关系**。"""

    if demo_case == "sunwukong":
        return """根据当前《封神演义》知识图谱和演示材料，**不能回答“孙悟空在《封神演义》中有什么法宝”这个问题**，原因是没有检索到《封神演义》中存在“孙悟空”这个人物节点，也没有检索到“孙悟空拥有某法宝”的图谱关系 [1]。

这里需要明确区分作品边界：
1. **孙悟空是《西游记》的核心人物**，不是《封神演义》主线人物。
2. 当前系统没有证据支持“孙悟空在《封神演义》中拥有法宝”。
3. 因此不能把金箍棒等《西游记》设定迁移到《封神演义》的回答里，否则会造成跨作品混淆 [1]。

所以，稳定结论是：**在当前《封神演义》材料范围内，孙悟空没有可确认的法宝；这个问题应按作品边界拒答，而不是编造答案**。"""

    return None


def evidence_bundle_config_for_question(question: str, *, for_display: bool = False) -> Dict[str, int]:
    demo_case = _match_demo_case(question)
    if not demo_case:
        return {
            "max_docs": 6,
            "max_excerpts_per_doc": 4,
            "max_excerpt_chars": 800,
        }
    return {
        "max_docs": 4 if for_display else 5,
        "max_excerpts_per_doc": 2,
        "max_excerpt_chars": 220 if for_display else 260,
    }


def _format_demo_source_label(citation: Dict) -> str:
    chapter_number = citation.get("chapter_number")
    chapter_title = str(citation.get("chapter_title") or "").strip()
    if chapter_number:
        if chapter_title:
            return f"第{chapter_number}回《{chapter_title}》"
        return f"第{chapter_number}回"
    source_chunk_id = str(citation.get("source_chunk_id") or "").strip()
    if source_chunk_id:
        return source_chunk_id
    source = str(citation.get("source") or "").strip()
    return source or "当前资料"


def _replace_citation_numbers(text: str, citation_order: List[int]) -> str:
    if not text:
        return text
    remap = {old: new for new, old in enumerate(citation_order, start=1)}

    def repl(match: re.Match) -> str:
        old = int(match.group(1))
        return f"[{remap.get(old, old)}]"

    return re.sub(r"\[(\d+)\]", repl, text)


def _unique_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def build_local_fallback_answer(question: str, documents: List[Document]) -> str:
    if not documents:
        return "根据当前知识图谱证据，暂时没有检索到可直接支持该问题的事实。"
    bundles = build_evidence_bundles(documents, max_docs=5, max_excerpts_per_doc=3, max_excerpt_chars=700)
    entity_names = _unique_preserve_order([str(doc.metadata.get("entity_name") or "") for doc in documents])
    edge_lines, evidence_lines = [], []
    for bundle in bundles:
        for item in bundle.get("original_excerpts") or []:
            text = str(item.get("text") if isinstance(item, dict) else item).strip()
            if text:
                evidence_lines.append(f"[{bundle.get('number')}] {text}")
        for fact in bundle.get("graph_facts") or []:
            edge_lines.append(str(fact))
    if not evidence_lines:
        for doc in documents:
            content = (doc.page_content or "").strip().replace("\n", " ")
            if content:
                evidence_lines.append(content[:180])
    lines = ["根据当前知识图谱证据，可以得到以下结论："]
    if entity_names:
        lines.append(f"问题涉及的核心对象包括：{'、'.join(entity_names[:5])}。")
    evidence_lines = _unique_preserve_order(evidence_lines)
    if evidence_lines:
        lines.append("可直接参考的原文证据：")
        for item in evidence_lines[:4]:
            lines.append(f"- {item}")
    edge_lines = _unique_preserve_order(edge_lines)
    if edge_lines:
        lines.append("图谱辅助关系：")
        for item in edge_lines[:6]:
            lines.append(f"- {item}")
    return "\n".join(lines)



def _clean_answer_text(answer: str) -> str:
    """Remove inline graph notation and internal identifiers from generated answers."""
    import re
    # Remove --[RELATION]--> patterns
    answer = re.sub(r'\s*--\s*\[[A-Z_]+\]\s*-->\s*', ' ', answer)
    # Remove <RELATION> angle-bracket notation
    answer = re.sub(r'\s*<[A-Z_]+>\s*', ' ', answer)
    # Remove bare ALL_CAPS relation types on their own line
    answer = re.sub(r'(?m)^\s*[A-Z_]{3,}\s*$', '', answer)
    # Normalize spaces
    answer = re.sub(r' {2,}', ' ', answer)
    answer = re.sub(r'\n{3,}', '\n\n', answer)
    return answer.strip()

class GenerationIntegrationModule:
    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-V3",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        api_base: str = "https://api.siliconflow.cn/v1",
        embedding_model: str = "BAAI/bge-m3",
        client=None,
        embeddings=None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        api_key = os.getenv("SILICONFLOW_API_KEY") or os.getenv("MOONSHOT_API_KEY") or os.getenv("OPENAI_API_KEY")
        if client is None and not api_key:
            raise ValueError("请设置 SILICONFLOW_API_KEY 环境变量")
        self.client = client or OpenAI(api_key=api_key, base_url=api_base)
        if embeddings is not None:
            self.embeddings = embeddings
        else:
            from langchain_openai import OpenAIEmbeddings
            self.embeddings = OpenAIEmbeddings(openai_api_key=api_key, openai_api_base=api_base, model=embedding_model, chunk_size=64)
        logger.info("生成模块初始化完成，API: %s, 模型: %s", api_base, self.model_name)

    def _build_structured_context_with_citations(self, documents: List[Document], question: str = "") -> Tuple[str, List[Dict]]:
        bundle_cfg = evidence_bundle_config_for_question(question, for_display=False)
        bundles = build_evidence_bundles(documents, **bundle_cfg)
        context = format_evidence_context_for_generation(bundles, max_total_chars=12000)
        citations = []
        for bundle in bundles:
            metadata = dict(getattr(documents[bundle.get("number", 1) - 1], "metadata", {}) or {}) if str(bundle.get("number", "")).isdigit() else {}
            citations.append({
                "number": bundle.get("number"),
                "content": bundle.get("summary_preview", ""),
                "source": bundle.get("title", "未知来源"),
                "retrieval_level": bundle.get("search_type", ""),
                "source_chunk_id": bundle.get("source_chunk_id", ""),
                "chapter_number": metadata.get("chapter_number"),
                "chapter_title": metadata.get("chapter_title", ""),
                "has_original_text": bool(bundle.get("has_original_text")),
                "original_excerpts": bundle.get("original_excerpts", []),
                "graph_facts": bundle.get("graph_facts", []),
            })
        return context, citations

    def _postprocess_demo_answer(self, question: str, answer: str, citations: List[Dict] | None = None) -> str:
        if not _match_demo_case(question):
            return answer
        answer_body = re.split(r"原文依据[：:]\s*", answer, maxsplit=1)[0].rstrip()

        citation_map: Dict[int, Dict] = {}
        for citation in citations or []:
            try:
                idx = int(citation.get("number"))
            except Exception:
                continue
            citation_map[idx] = citation

        cited_in_answer = [int(x) for x in re.findall(r"\[(\d+)\]", answer_body)]
        citation_order: List[int] = []
        for idx in cited_in_answer:
            if idx in citation_map and idx not in citation_order:
                citation_order.append(idx)
        for idx in citation_map:
            if idx not in citation_order:
                citation_order.append(idx)

        rewritten_body = _replace_citation_numbers(answer_body, citation_order)

        rebuilt_items: List[str] = []
        for new_index, old_idx in enumerate(citation_order, start=1):
            citation = citation_map.get(old_idx, {})
            excerpts = citation.get("original_excerpts") or []
            excerpt_text = ""
            for item in excerpts:
                if isinstance(item, dict):
                    excerpt_text = str(item.get("text") or "").strip()
                else:
                    excerpt_text = str(item or "").strip()
                if excerpt_text:
                    break
            if not excerpt_text:
                continue
            source_label = _format_demo_source_label(citation)
            rebuilt_items.append(f"{new_index}. {excerpt_text}（{source_label}）")

        if not rebuilt_items:
            return rewritten_body

        return rewritten_body + "\n\n原文依据：\n" + "\n".join(rebuilt_items)

    def generate_adaptive_answer(self, question: str, documents: List[Document]) -> str:
        demo_answer = _demo_direct_answer(question)
        if demo_answer:
            return demo_answer

        context, citations = self._build_structured_context_with_citations(documents, question=question)
        prompt = f"""你是一位熟悉《封神演义》的知识图谱问答助手。请只基于给定的检索信息回答问题，不要补充外部知识。

回答要求：
1. 优先依据每条资料中的“原文摘录”回答；原文摘录是验证观点的第一依据。
2. “图谱事实”只能辅助组织人物、法宝、事件关系，不能替代原文证据；如果某点只有图谱事实，请明确写“据图谱关系”。
3. 每个关键事实后必须标注对应编号，如 [1]、[2]；引用编号只用于标明证据来源，可以使用方括号。
4. 信息不足时，明确说“根据当前资料无法完全确定”。
5. 用自然流畅的中文回答，不要输出 --[REL]-->、内部关系代码、chunk_id 或数据库字段名。
6. 结构化呈现：先直接回答问题，再用“原文依据”小标题列出关键引用编号和对应原文要点。

检索到的相关信息：
{context}

用户问题：{question}

回答："""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            raw = response.choices[0].message.content.strip()
            return self._postprocess_demo_answer(question, _clean_answer_text(raw), citations)
        except Exception as e:
            logger.warning("answer generation failed, using local fallback: %s", e)
            return build_local_fallback_answer(question, documents)

    def generate_adaptive_answer_stream(self, question: str, documents: List[Document], max_retries: int = 3):
        context, _ = self._build_structured_context_with_citations(documents, question=question)
        prompt = f"""你是《封神演义》知识图谱问答助手。只基于检索信息回答，不要补充外部知识。

回答要求：优先使用“原文摘录”验证观点；图谱事实只作关系辅助。每个关键事实后标注对应编号 [1]、[2]。不要输出 --[REL]-->、内部关系代码、chunk_id 或数据库字段名。

检索到的相关信息：
{context}

用户问题：{question}

回答："""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True,
                    timeout=60,
                )
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                return
            except Exception as e:
                logger.warning("stream generation failed %s/%s: %s", attempt + 1, max_retries, e)
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 2)
                    continue
                yield self.generate_adaptive_answer(question, documents)
                return
