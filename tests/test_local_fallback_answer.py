from pathlib import Path
import sys

from langchain_core.documents import Document


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from rag_modules.generation_integration import build_local_fallback_answer
from rag_modules.generation_integration import GenerationIntegrationModule


SUN = "\u5b59\u4e2d\u5c71"
XINGZHONGHUI = "\u5174\u4e2d\u4f1a"
QUESTION = f"{SUN}\u4e0e{XINGZHONGHUI}\u662f\u4ec0\u4e48\u5173\u7cfb\uff1f"
EVIDENCE_HEADER = "\u6839\u636e\u5f53\u524d\u77e5\u8bc6\u56fe\u8c31\u8bc1\u636e"
NETWORK_ERROR = "\u7f51\u7edc\u9519\u8bef"


def _sample_docs():
    return [
        Document(
            page_content=f"\u540d\u79f0: {SUN}\n\u76f8\u5173\u7ec4\u7ec7: {XINGZHONGHUI}",
            metadata={
                "entity_name": SUN,
                "node_type": "Person",
                "search_type": "exact_entity_lookup",
            },
        ),
        Document(
            page_content=f"{SUN}\u521b\u5efa{XINGZHONGHUI}\u3002",
            metadata={
                "entity_name": SUN,
                "node_type": "Person",
                "search_type": "graph_path",
                "subgraph_edges": [
                    {
                        "source": SUN,
                        "relation": "FOUNDS",
                        "target": XINGZHONGHUI,
                        "source_type": "Person",
                        "target_type": "Organization",
                    }
                ],
            },
        ),
    ]


def test_local_fallback_answer_uses_graph_docs_and_evidence():
    answer = build_local_fallback_answer(QUESTION, _sample_docs())

    assert SUN in answer
    assert XINGZHONGHUI in answer
    assert "FOUNDS" in answer or "\u521b\u5efa" in answer
    assert EVIDENCE_HEADER in answer


class _RaisingCompletions:
    def create(self, **kwargs):
        raise RuntimeError("connection error")


class _FakeClient:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": _RaisingCompletions()})()


def _build_module():
    module = GenerationIntegrationModule.__new__(GenerationIntegrationModule)
    module.client = _FakeClient()
    module.model_name = "test-model"
    module.temperature = 0.1
    module.max_tokens = 256
    return module


def test_generate_adaptive_answer_falls_back_to_local_answer_on_llm_error():
    module = _build_module()

    answer = module.generate_adaptive_answer(QUESTION, _sample_docs())

    assert SUN in answer
    assert XINGZHONGHUI in answer
    assert EVIDENCE_HEADER in answer
    assert "connection error" not in answer


def test_generate_adaptive_answer_stream_falls_back_to_local_answer_on_llm_error():
    module = _build_module()

    answer = "".join(
        module.generate_adaptive_answer_stream(
            QUESTION,
            _sample_docs(),
            max_retries=1,
        )
    )

    assert SUN in answer
    assert XINGZHONGHUI in answer
    assert EVIDENCE_HEADER in answer
    assert NETWORK_ERROR not in answer
