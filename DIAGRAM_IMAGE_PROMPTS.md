# 项目示意图详细图片生成 Prompt

以下 Prompt 可用于 DALL·E、Midjourney、Stable Diffusion、通义万相、即梦、可灵、Canva AI、PPT AI 或其他图像生成工具。建议统一要求：**中文标签清晰、学术论文风格、浅色背景、蓝金配色、模块化架构图、不要生成真实人物、不要生成代码截图**。

---

## 1. 系统总体架构图 Prompt

```text
请生成一张中文技术架构图，主题为“《封神演义》GraphRAG 问答系统总体架构”。

画面风格：学术论文/课程项目报告风格，横向布局，浅米白或浅灰背景，蓝色与金色为主色，模块使用圆角矩形，箭头清晰，层次分明，中文标签必须清晰可读，不要使用复杂装饰，不要生成真实人物，不要生成代码截图。

整体从左到右分为六个大区域：
1. 数据层
2. 知识图谱构建层
3. 存储层
4. 检索与路由层
5. 生成与证据层
6. 展示层

左侧“数据层”包含两个模块：
- 《封神演义.txt》：99回原文语料，约60万字
- LLM抽取缓存：data/fengshen/extraction_results

第二列“知识图谱构建层”包含四个模块，并用箭头串联：
- 章节解析与文本切块：parse_chapters / split_chapter_text
- LLM知识抽取：NER + RE Prompt
- Schema约束：Person、Faction、Artifact、Event等实体类型
- 实体对齐：alias合并重复实体

第三列“存储层”包含两个数据库形状的模块：
- Neo4j知识图谱：实体、关系、TextChunk、索引
- FAISS向量索引：fengshen_faiss_index

第四列“检索与路由层”包含四个模块：
- GraphDataPreparation：Neo4j转RAG文档
- Hybrid Retrieval：BM25 + FAISS + 图谱补充
- GraphRAG Retrieval：实体链接、多跳、子图、路径
- Intelligent Query Router：问题复杂度分析与策略选择

第五列“生成与证据层”包含三个模块：
- 上下文融合：原文片段 + 图谱三元组
- LLM答案生成：只基于证据回答
- 引用与溯源：source_evidence / snippet_service

最右侧“展示层”包含一个大的 Streamlit Web UI 框，内部有三个小模块：
- 答案输出
- 原文片段引用
- 知识图谱子图可视化

箭头关系：
《封神演义.txt》进入章节解析与文本切块；切块进入LLM知识抽取；抽取结果进入Schema约束和实体对齐；实体与关系写入Neo4j；TextChunk同时进入Neo4j；Neo4j经过GraphDataPreparation生成RAG文档并构建FAISS；用户问题进入Intelligent Query Router；Router根据问题选择Hybrid Retrieval或GraphRAG Retrieval；检索结果进入上下文融合；再进入LLM答案生成；最后在Streamlit UI展示答案、引用和子图。

请保持图面整洁，模块不要太拥挤，每个模块最多两行文字，重要关键词加粗或使用深色。输出为高清 16:9 架构图。
```

---

## 2. 图谱构建流程图 Prompt

```text
请生成一张中文流程图，主题为“《封神演义》知识图谱构建流程”。

画面风格：适合课程设计报告的流程图，纵向布局，浅色背景，蓝金配色，流程节点为圆角矩形，判断节点为菱形，数据库节点为圆柱体，箭头清晰，中文标签清楚，不要生成代码截图，不要生成复杂插画。

流程从上到下展示以下步骤：

1. 开始：读取《封神演义.txt》
2. 解析章节标题：按“第X回”定位章节边界
3. 文本清洗与切块：按段落、句号、问号、感叹号、换行等断点切分文本
4. 调用LLM抽取知识：NER实体识别 + RE关系抽取
5. 判断：抽取结果是否为合法JSON？
   - 否：记录错误 / 跳过或重试，然后返回“调用LLM抽取知识”
   - 是：进入实体与关系解析
6. 解析实体 Entities
7. 解析关系 Relations
8. 按 gold_schema 校验实体类型
9. 按 ALLOWED_RELATION_TYPES 校验关系类型
10. 写入Neo4j实体节点
11. 写入Neo4j关系边
12. 写入TextChunk原文片段节点
13. TextChunk关联Chapter章节节点
14. 创建节点索引：name、id、chunk_id、chapter_id
15. 基于alias进行实体对齐，合并重复实体
16. 保存或复用抽取缓存
17. 完成：生成可供GraphRAG检索的知识图谱

图中请将“实体处理”和“关系处理”作为两个并行分支展示：
- 实体分支：解析实体 → 校验实体类型 → 写入实体节点
- 关系分支：解析关系 → 校验关系类型 → 写入关系边

同时从“文本清洗与切块”额外引出一条箭头到“写入TextChunk原文片段节点”，表示原文证据单独入库。

在底部用一个 Neo4j 圆柱体图标表示最终知识图谱，旁边标注：“实体、关系、章节、原文片段、索引、别名合并”。

输出为高清竖版或 4:3 图，适合放入论文/报告中。
```

---

## 3. 用户问答时序图 Prompt

```text
请生成一张中文时序图，主题为“《封神演义》GraphRAG 用户问答时序”。

画面风格：软件工程时序图，白色或浅灰背景，蓝色线条，参与者使用竖向生命线，箭头从左到右交互，文字清晰，适合课程报告，不要生成代码，不要生成真实人物。

请从左到右放置以下参与者：
1. 用户
2. Streamlit UI app.py
3. IntelligentQueryRouter
4. HybridRetrieval
5. GraphRAGRetrieval
6. Neo4j知识图谱
7. FAISS向量索引
8. GenerationIntegration
9. SourceEvidence / SnippetService
10. 大模型API

时序流程如下：

第一阶段：用户提问
- 用户 → Streamlit UI：输入问题
- Streamlit UI → IntelligentQueryRouter：route_query(question, top_k)
- IntelligentQueryRouter 自处理：分析问题复杂度、候选实体、关系强度、是否需要多跳推理

第二阶段：策略分支
画出三个可选分支，并用浅色区域框标注：

分支A：简单事实或文本证据优先
- Router → HybridRetrieval：执行传统混合检索
- HybridRetrieval → FAISS：Top-K向量检索
- HybridRetrieval → Neo4j：实体/主题补充查询
- FAISS → HybridRetrieval：返回相似原文片段
- Neo4j → HybridRetrieval：返回相关节点与关系
- HybridRetrieval → Router：返回文档候选

分支B：多跳关系或图谱推理问题
- Router → GraphRAGRetrieval：执行图谱检索
- GraphRAGRetrieval → Neo4j：实体链接 + 多跳路径/子图查询
- Neo4j → GraphRAGRetrieval：返回路径、邻居、三元组、原文证据
- GraphRAGRetrieval → Router：返回图谱增强文档

分支C：组合策略
- Router 同时调用 HybridRetrieval 和 GraphRAGRetrieval
- 两者分别返回文本片段和图谱事实

第三阶段：生成答案
- Router → Streamlit UI：返回检索文档和路由解释
- Streamlit UI → GenerationIntegration：generate_adaptive_answer(question, docs)
- GenerationIntegration → SourceEvidence/SnippetService：构造带编号证据上下文
- SourceEvidence/SnippetService → GenerationIntegration：返回原文摘录 + 图谱事实
- GenerationIntegration → 大模型API：只基于证据生成答案
- 大模型API → GenerationIntegration：返回带引用编号的回答
- GenerationIntegration → Streamlit UI：返回答案

第四阶段：展示溯源
- Streamlit UI → SourceEvidence/SnippetService：提取引用片段和三元组
- SourceEvidence/SnippetService → Streamlit UI：返回来源卡片和子图数据
- Streamlit UI → 用户：展示答案、原文溯源、知识图谱子图

请让时序图结构清晰，分支区域不要太复杂，使用“alt/else”或分支框表达不同检索策略。输出为高清横版图。
```

---

## 4. 检索融合流程图 Prompt

```text
请生成一张中文流程图，主题为“GraphRAG 检索融合流程”。

画面风格：横向或上到下混合布局，适合技术报告，浅色背景，蓝色、绿色、金色区分不同路径。文本检索路径使用蓝色，图谱检索路径使用金色，融合与生成路径使用绿色。所有标签用中文，简洁清晰，不要生成代码。

流程起点：用户问题。

第一部分：问题分析
从“用户问题”进入“问题分析”模块，问题分析模块拆成三个结果：
1. 抽取候选实体：人物、教派、法宝、事件
2. 判断查询类型：简单查询、多跳查询、路径查询、子图查询
3. 判断是否需要图谱推理

第二部分：三种策略分支
从“判断是否需要图谱推理”分出三条路径：

路径A：文本证据优先
- 进入“FAISS向量检索 Top-K”
- 输出“相似原文片段”

路径B：关系推理优先
- 进入“Neo4j图谱检索”
- 进入“实体链接”
- 进入“一跳/多跳邻居查询”
- 进入“路径与子图提取”
- 输出“图谱三元组事实”

路径C：组合策略
- 同时连接到“FAISS向量检索 Top-K”和“Neo4j图谱检索”
- 表示文本和图谱并行检索

第三部分：融合与生成
将“相似原文片段”和“图谱三元组事实”汇入“上下文融合”模块。
上下文融合之后依次连接：
- 去重、排序、证据优先
- 构造问答Prompt：原文摘录 + 图谱事实 + 引用编号
- LLM生成答案
- 展示引用片段与知识子图

请在图中强调：
- 原文片段用于事实依据
- 图谱三元组用于关系推理
- LLM只基于融合后的证据上下文回答

输出为高清 16:9 流程图，适合直接放到项目答辩PPT中。
```

---

## 5. 知识图谱 Schema 示意图 Prompt

```text
请生成一张中文知识图谱 Schema 示意图，主题为“《封神演义》知识图谱 Schema”。

画面风格：图数据库关系模型图，浅色背景，节点为圆角矩形或圆形，实体节点使用不同柔和颜色，关系用带箭头的连线表示。整体清晰、对称、适合学术报告。不要生成复杂背景，不要生成真实人物插画。

中心节点放置 Person（人物），因为人物是知识图谱核心。围绕 Person 分布以下节点：
- Faction（教派/势力）
- Location（地点）
- Artifact（法宝）
- Beast（坐骑/灵兽）
- Formation（阵法）
- Event（事件/战役）
- DeityPosition（神位/封号）
- Chapter（章节）
- TextChunk（原文片段）

请画出以下关系：

Person 与 Person 之间：
- MASTER_OF / APPRENTICE_OF：师徒关系
- FATHER_OF / CHILD_OF / BROTHER_OF / MARRIED_TO：亲属婚姻关系
- KILLS / DEFEATS / CAPTURES / OPPOSES：战斗对抗关系

Person 与 Faction：
- BELONGS_TO_SECT：教派归属
- FIGHTS_FOR：效力阵营

Person 与 Artifact：
- OWNS / BESTOWS / LOSES / STEALS：法宝拥有与流转

Person 与 Beast：
- OWNS：拥有坐骑或灵兽

Person 与 Event：
- PARTICIPATES_IN / LEADS / INITIATES：参与、率领或发起事件

Person 与 Formation：
- CREATES / DEPLOYS / BREAKS：布阵、部署、破阵

Person 与 DeityPosition：
- LISTED_ON / BECOMES：封神榜或成神结局

Event 与 Location：
- OCCURS_IN：事件发生地点

TextChunk 与 Chapter：
- BELONGS_TO_CHAPTER：原文片段所属章节

TextChunk 与 Person/Event/Artifact：
- MENTIONS：原文片段提及实体

请在图底部增加一句说明：“TextChunk 提供原文证据，实体和关系提供图谱推理结构。”

输出为高清横版图，适合放在系统设计章节。
```

---

## 6. 报告用系统流程简图 Prompt

```text
请生成一张简洁的中文系统流程图，主题为“《封神演义》GraphRAG 系统流程”。

用途：这张图要适合放在项目报告或答辩PPT的一页中，是简洁版总览图，不要过多细节。

画面风格：横向流程，白色或浅米色背景，蓝金配色，模块为圆角矩形，数据库用圆柱体，箭头清晰，中文标签清晰。整体不要超过12个模块。

请从左到右绘制以下流程：

第一段：离线构建流程
1. 《封神演义》原文
2. 文本清洗 / 章节解析 / 切块
3. LLM实体识别与关系抽取
4. Neo4j知识图谱
5. TextChunk文档
6. FAISS向量索引

其中：
- 《封神演义》原文 → 文本清洗/章节解析/切块
- 文本清洗/章节解析/切块 → LLM实体识别与关系抽取 → Neo4j知识图谱
- 文本清洗/章节解析/切块 → TextChunk文档 → FAISS向量索引

第二段：在线问答流程
7. 用户问题
8. 智能路由
9. 相关原文片段
10. 实体链接 / 多跳路径 / 子图三元组
11. 证据上下文融合
12. LLM基于证据生成答案
13. Streamlit展示：答案 + 引用 + 子图

其中：
- 用户问题 → 智能路由
- 智能路由连接 FAISS向量索引，输出相关原文片段
- 智能路由连接 Neo4j知识图谱，输出实体链接/多跳路径/子图三元组
- 相关原文片段和图谱三元组共同进入证据上下文融合
- 证据上下文融合 → LLM基于证据生成答案 → Streamlit展示

请在图中用虚线或背景分区区分“离线构建”和“在线问答”。
请确保布局简洁，适合一眼看懂系统流程。输出高清16:9图片。
```

---

## 7. Graphviz DOT 风格架构图 Prompt

```text
请生成一张类似 Graphviz DOT 渲染效果的中文技术架构图，主题为“Fengshen GraphRAG Pipeline”。

画面风格：极简工程图，白色背景，模块为浅米色圆角矩形，数据库模块为浅蓝色圆柱体，箭头为棕金色，整体类似自动布局的 Graphviz 架构图。不要插画，不要复杂背景，不要人物。

请使用从左到右 rankdir=LR 的布局逻辑，包含以下节点：

1. 封神演义.txt：原始语料
2. 章节解析 / 文本切块
3. LLM NER + RE：实体关系抽取
4. Neo4j知识图谱：实体、关系、TextChunk、索引
5. FAISS向量索引：Top-K文本检索
6. 智能查询路由：复杂度、关系强度、策略选择
7. GraphRAG检索：实体链接、多跳、子图
8. 混合检索：BM25、FAISS、图谱补充
9. 上下文融合：原文片段 + 图谱三元组
10. LLM证据生成：防幻觉Prompt
11. Streamlit UI：答案、引用、知识子图

箭头关系：
- 封神演义.txt → 章节解析 / 文本切块
- 章节解析 / 文本切块 → LLM NER + RE
- LLM NER + RE → Neo4j知识图谱
- 章节解析 / 文本切块 → FAISS向量索引
- 智能查询路由 → GraphRAG检索 → Neo4j知识图谱
- 智能查询路由 → 混合检索 → FAISS向量索引
- 混合检索 → Neo4j知识图谱
- GraphRAG检索 → 上下文融合
- 混合检索 → 上下文融合
- 上下文融合 → LLM证据生成 → Streamlit UI

请让所有中文文字清晰可读，节点间距均匀，输出高清横版图。
```

---

## 8. 答辩PPT封面式架构图 Prompt

```text
请生成一张适合课程答辩PPT使用的视觉化架构图，主题为“基于大模型 + 知识图谱的《封神演义》增强型 RAG 问答系统”。

画面风格：现代科技感但不过度花哨，深蓝或浅米背景均可，主色调为蓝色、金色、白色。整体像一张高质量答辩PPT中的系统架构页。中文标签清晰，模块边界明确，不要生成真实人物，不要生成复杂古风插画，不要生成代码。

画面中心是一个大标题：
“《封神演义》GraphRAG 问答系统”
副标题：
“原文证据 + 知识图谱 + 多跳推理 + 可视化溯源”

图中分为三条主线：

左侧：数据与构建
- 《封神演义》原文
- 文本清洗与章节切块
- LLM实体识别与关系抽取
- Neo4j知识图谱

中间：检索与推理
- FAISS向量检索
- 实体链接
- 多跳路径检索
- 子图三元组提取
- 智能路由融合

右侧：生成与展示
- 证据上下文融合
- 大模型防幻觉回答
- 引用编号
- 原文片段溯源
- 知识图谱子图可视化
- Streamlit Web界面

请用箭头连接三条主线：
原文进入构建流程，构建出Neo4j和FAISS；用户问题进入智能路由；路由连接Neo4j和FAISS；检索结果进入证据上下文；大模型生成答案；最后进入Streamlit展示。

请在图底部用三个亮点标签突出：
1. 500+ 文本片段
2. Neo4j + FAISS 混合检索
3. 答案引用与图谱子图溯源

输出为高清16:9图片，适合直接作为PPT中的架构页。
```

---

## 9. 更适合 AI 绘图的统一负面约束

如果使用图像生成模型，建议在每个 Prompt 后追加：

```text
负面要求：不要生成英文乱码，不要生成不可读的小字，不要生成真实人物，不要生成古风人物插画，不要生成代码截图，不要生成过度复杂背景，不要把箭头画乱，不要出现与《封神演义》无关的现代城市或机器人形象，不要使用低分辨率，不要出现错误拼写。
```

---

## 10. 推荐使用顺序

- 报告正文优先用：Prompt 6 简洁系统流程图
- 系统设计章节用：Prompt 1 总体架构图
- 数据处理/图谱构建章节用：Prompt 2 图谱构建流程图
- 检索算法章节用：Prompt 4 检索融合流程图
- Schema 设计章节用：Prompt 5 知识图谱 Schema 图
- 答辩 PPT 首页或亮点页用：Prompt 8 PPT封面式架构图
