"""封神演义知识图谱问答验收测试用例"""

ACCEPTANCE_QUERIES = [
    "哪吒是谁？",
    "姜子牙和元始天尊是什么关系？",
    "哪吒的师父是谁？他属于哪个教派？有哪些法宝？",
    "申公豹和姜子牙是什么关系？",
    "孙悟空在封神演义中有什么法宝？",
]


def test_acceptance_query_list_is_stable():
    assert len(ACCEPTANCE_QUERIES) == 5


def test_acceptance_queries_are_fengshen():
    keywords = ["哪吒", "姜子牙", "元始天尊", "申公豹", "封神演义"]
    assert sum(any(kw in query for kw in keywords) for query in ACCEPTANCE_QUERIES) >= 4
