from __future__ import annotations

import re


class DemoQuestionService:
    def __init__(self, data_module):
        self.data_module = data_module

    def question_has_answer(self, question: str) -> bool:
        if not self.data_module or not getattr(self.data_module, "driver", None):
            return True

        stop_words = {"什么", "为什么", "怎么", "如何", "是谁", "哪些", "之间", "关系", "事件", "分别", "发生", "做了", "什么关系"}
        keywords = []
        for token in re.findall(r"[一-鿿]{2,8}", question):
            token = re.sub(r"(为什么|是什么|是谁|有哪些|做了什么|之间|关系|事件)$", "", token)
            if len(token) >= 2 and token not in stop_words:
                keywords.append(token)
        keywords = list(dict.fromkeys(keywords))[:8]
        if not keywords:
            return True

        try:
            with self.data_module.driver.session(database=self.data_module.database) as session:
                query = """
                UNWIND $kw as kw
                MATCH (n)
                WHERE (n:TextChunk OR n:Event OR n:Organization OR n:Person)
                  AND (
                    coalesce(n.text,'') CONTAINS kw OR
                    coalesce(n.description,'') CONTAINS kw OR
                    coalesce(n.name,'') CONTAINS kw OR
                    coalesce(n.chapter,'') CONTAINS kw OR
                    coalesce(n.section,'') CONTAINS kw
                  )
                RETURN count(n) as c
                """
                count = session.run(query, {"kw": keywords}).single()["c"]
                return int(count) > 0
        except Exception:
            return True
