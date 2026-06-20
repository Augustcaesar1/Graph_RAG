from pathlib import Path
import sys

from langchain_core.documents import Document


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_modules.generation_integration import (
    GenerationIntegrationModule,
    evidence_bundle_config_for_question,
)
from rag_modules.source_evidence import build_evidence_bundles, format_evidence_context_for_generation


def _build_module(client=None):
    module = GenerationIntegrationModule.__new__(GenerationIntegrationModule)
    module.client = client
    module.model_name = "test-model"
    module.temperature = 0.1
    module.max_tokens = 256
    return module


def test_evidence_context_includes_document_source_text():
    doc = Document(
        page_content="图谱摘要：纣王与女娲存在对立关系。",
        metadata={
            "entity_name": "女娲",
            "source_text": "女娲娘娘看见粉壁上诗句，大怒骂曰：今反不畏上天，吟诗亵我，甚是可恶！",
            "source_chunk_id": "fs_ch001_0002",
        },
    )
    module = _build_module()

    context, citations = module._build_structured_context_with_citations([doc])

    assert "原文摘录" in context
    assert "吟诗亵我" in context
    assert "fs_ch001_0002" in context
    assert citations[0]["has_original_text"] is True


def test_evidence_context_prefers_edge_source_text_and_keeps_multiple_excerpts():
    doc = Document(
        page_content="知识子图：女娲、纣王、武王伐纣。",
        metadata={
            "entity_name": "女娲",
            "source_text": "无关背景文本。" * 50,
            "subgraph_edges": [
                {
                    "source": "纣王",
                    "relation": "OPPOSES",
                    "target": "女娲",
                    "evidence": "作诗亵渎圣明",
                    "source_text": "商容启奏曰：今陛下作诗亵渎圣明，毫无虔敬之诚，是获罪于神圣。",
                },
                {
                    "source": "女娲",
                    "relation": "INITIATES",
                    "target": "武王伐纣",
                    "evidence": "三妖可隐其妖形，托身宫院，惑乱君心",
                    "source_text": "娘娘曰：你三妖可隐其妖形，托身宫院，惑乱君心；俟武王伐纣，以助成功。",
                },
            ],
        },
    )

    bundles = build_evidence_bundles([doc], max_excerpts_per_doc=4)
    context = format_evidence_context_for_generation(bundles)

    assert "作诗亵渎圣明" in context
    assert "惑乱君心" in context
    assert context.index("作诗亵渎圣明") < context.index("无关背景文本")
    assert len(bundles[0]["original_excerpts"]) >= 2


class _CapturingCompletions:
    def __init__(self):
        self.prompt = ""

    def create(self, **kwargs):
        self.prompt = kwargs["messages"][0]["content"]
        message = type("Message", (), {"content": "女娲因纣王题诗亵渎而怒[1]。"})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


class _CapturingClient:
    def __init__(self):
        self.completions = _CapturingCompletions()
        self.chat = type("Chat", (), {"completions": self.completions})()


def test_generate_prompt_contains_original_excerpt():
    client = _CapturingClient()
    module = _build_module(client)
    doc = Document(
        page_content="图谱摘要：纣王 OPPOSES 女娲。",
        metadata={
            "entity_name": "女娲",
            "source_text": "今陛下作诗亵渎圣明，毫无虔敬之诚，是获罪于神圣。",
        },
    )

    answer = module.generate_adaptive_answer("女娲为什么生气？", [doc])

    assert "题诗亵渎" in answer
    assert "原文摘录" in client.completions.prompt
    assert "作诗亵渎圣明" in client.completions.prompt


def test_demo_case_uses_compact_evidence_bundle_config():
    config = evidence_bundle_config_for_question(
        "哪吒的师父是谁？他属于哪个教派？有哪些法宝？",
        for_display=True,
    )

    assert config["max_excerpts_per_doc"] == 2
    assert config["max_excerpt_chars"] <= 220


def test_demo_case_answer_postprocess_reindexes_and_adds_sources():
    module = _build_module()
    raw = """纣王是商朝末代君主。

原文依据：
1. 出身与继位：纣王乃帝乙之三子，因托梁换柱展现神力被立为太子，帝乙崩后继位 [1]。
2. 暴政表现：
- 发明虿盆之刑，将宫女与忠臣胶鬲喂蛇蝎 [1]；
- 听信妲己建造酒池肉林，残害宫人取乐 [1]；
3. 统治背景：初期商朝国力强盛，但后期因昏庸失政 [1]。
4. 其他补充：他是殷郊、殷洪的父亲 [1]。
"""

    citations = [
        {
            "number": 1,
            "chapter_number": 1,
            "chapter_title": "纣王女娲宫进香",
            "source_chunk_id": "extract_ch001_0000",
            "source": "纣王",
            "original_excerpts": [{"text": "纣王乃帝乙之三子，因托梁换柱展现神力被立为太子，帝乙崩后继位。"}],
        }
    ]

    answer = module._postprocess_demo_answer("纣王是谁？", raw, citations)

    assert "纣王是商朝末代君主。" in answer
    assert "原文依据：" in answer
    assert "1. 纣王乃帝乙之三子，因托梁换柱展现神力被立为太子，帝乙崩后继位。" in answer
    assert "（第1回《纣王女娲宫进香》）" in answer


def test_demo_case_answer_postprocess_handles_unordered_sections():
    module = _build_module()
    raw = """哪吒的师父是太乙真人。

原文依据：
师父与教派：太乙真人为哪吒重塑莲花化身，并见师父拜倒在地 [5]。
法宝：乾坤圈与混天绫见于哪吒出世情节 [1]；火尖枪、风火轮、金砖见于莲花化身后授宝情节 [5]。
"""

    citations = [
        {
            "number": 1,
            "chapter_number": 12,
            "chapter_title": "陈塘关哪吒出世",
            "original_excerpts": [{"text": "金镯是乾坤圈，红绫名曰混天绫。"}],
        },
        {
            "number": 5,
            "chapter_number": 14,
            "chapter_title": "哪吒现莲花化身",
            "original_excerpts": [{"text": "此乃哪吒莲花化身，见师父拜倒在地。"}],
        },
    ]

    answer = module._postprocess_demo_answer(
        "哪吒的师父是谁？他属于哪个教派？有哪些法宝？",
        raw,
        citations,
    )

    assert "哪吒的师父是太乙真人。" in answer
    assert "1. 金镯是乾坤圈，红绫名曰混天绫。" in answer or "1. 此乃哪吒莲花化身，见师父拜倒在地。" in answer
    assert "（第12回《陈塘关哪吒出世》）" in answer
    assert "（第14回《哪吒现莲花化身》）" in answer


def test_demo_case_answer_postprocess_rebuilds_evidence_numbering_from_citations():
    module = _build_module()
    raw = """哪吒的师父是太乙真人[5]，主要法宝包括乾坤圈、混天绫、火尖枪、风火轮和金砖[1][5]。

原文依据：
师父与教派：太乙真人为哪吒重塑莲花化身，并见师父拜倒在地 [5]。
法宝：乾坤圈与混天绫见于哪吒出世情节 [1]；火尖枪、风火轮、金砖见于莲花化身后授宝情节 [5]。
"""

    citations = [
        {
            "number": 1,
            "chapter_number": 12,
            "chapter_title": "陈塘关哪吒出世",
            "original_excerpts": [{"text": "金镯是乾坤圈，红绫名曰混天绫。"}],
        },
        {
            "number": 5,
            "chapter_number": 14,
            "chapter_title": "哪吒现莲花化身",
            "original_excerpts": [{"text": "此乃哪吒莲花化身，见师父拜倒在地。"}],
        },
    ]

    answer = module._postprocess_demo_answer(
        "哪吒的师父是谁？他属于哪个教派？有哪些法宝？",
        raw,
        citations,
    )

    assert "哪吒的师父是太乙真人[1]" in answer
    assert "火尖枪、风火轮和金砖[2][1]" in answer
    assert "师父与教派：太乙真人为哪吒重塑莲花化身" not in answer
    assert "1. 此乃哪吒莲花化身，见师父拜倒在地。" in answer
    assert "（第14回《哪吒现莲花化身》）" in answer
    assert "2. 金镯是乾坤圈，红绫名曰混天绫。" in answer
    assert "（第12回《陈塘关哪吒出世》）" in answer
