# 封神演义 Graph RAG

基于 **大模型 + Neo4j 知识图谱 + FAISS 向量检索 + Streamlit** 的《封神演义》增强型 RAG 问答系统。

本项目对照 `项目要求.docx` 实现：从非结构化文本中使用 LLM 抽取实体与关系，构建知识图谱，结合向量检索与图检索进行问答，并在 Web UI 中展示原文片段与知识图谱子图溯源。

## 数据源

- 原始语料：`./封神演义.txt`
- 规模：99 回，约 60 万字，超过 500 个非结构化文本片段
- 编码：已转换为 UTF-8

数据处理链路：

- `scripts/import_fengshen_to_neo4j.py` 读取 `封神演义.txt`。
- `parse_chapters()` 按“第X回”解析章节边界。
- `split_chapter_text()` 按句号、问号、感叹号、换行等断点切分文本，避免硬截断语义。
- `_persist_text_chunks()` 将每回正文整理为 TextChunk 节点，并与 Chapter 节点相连，用于原文溯源和向量检索。

## 核心功能

- LLM 知识抽取：NER + 关系抽取，输出标准三元组 `(头实体, 关系, 尾实体)`
- Neo4j 图谱存储：人物、教派、法宝、阵法、事件等节点
- 实体对齐：基于 alias 合并重复实体，降低同名/别名带来的图谱冗余
- FAISS 向量检索：全文文本块向量化，支持 Top-K 相似文本块检索
- GraphRAG 检索：实体链接、多跳邻居检索、路径/子图提取
- 混合检索融合：文本片段 + 图谱三元组合并为 LLM 上下文
- 防幻觉回答：回答只基于检索上下文，信息不足时明确说明无法确定
- Streamlit 可视化：问答、原文溯源、知识图谱子图展示

## Schema 设计

### 实体类型

| 类型 | 说明 | 示例 |
|------|------|------|
| Person | 人物 | 姜子牙、哪吒、杨戬、纣王、妲己 |
| Faction | 教派/势力 | 商、周、阐教、截教、西方教 |
| Location | 地点 | 朝歌、西岐、玉虚宫、碧游宫 |
| Artifact | 法宝 | 打神鞭、乾坤圈、混天绫、翻天印 |
| Beast | 坐骑/灵兽 | 四不像、哮天犬、五色神牛 |
| Formation | 阵法 | 十绝阵、九曲黄河阵、诛仙阵、万仙阵 |
| Event | 事件/战役 | 哪吒闹海、武王伐纣、破诛仙阵 |
| DeityPosition | 神位/封号 | 三坛海会大神、文曲星 |
| TextChunk | 原文片段 | 第X回片段，用于溯源 |

### 关系类型

- 师承：`MASTER_OF`, `APPRENTICE_OF`
- 归属：`BELONGS_TO_SECT`, `FIGHTS_FOR`
- 亲属：`FATHER_OF`, `CHILD_OF`, `BROTHER_OF`, `MARRIED_TO`
- 对抗：`KILLS`, `DEFEATS`, `CAPTURES`, `OPPOSES`
- 法宝：`OWNS`, `BESTOWS`, `LOSES`, `STEALS`
- 事件/阵法：`CREATES`, `DEPLOYS`, `BREAKS`, `PARTICIPATES_IN`, `OCCURS_IN`, `LEADS`, `INITIATES`
- 封神结局：`LISTED_ON`, `BECOMES`
- 通用：`ALLIES_WITH`, `BETRAYS`, `RELATED_TO`, `MENTIONS`

## 系统架构

```text
封神演义.txt
  └─ scripts/import_fengshen_to_neo4j.py
       ├─ parse_chapters / split_chapter_text：章节解析、切块
       ├─ FengshenKGExtractor：LLM NER/RE 抽取
       ├─ Neo4j：实体、关系、TextChunk、索引、别名合并
       └─ data/fengshen/extraction_results：抽取缓存

Streamlit app.py
  └─ rag_modules/bootstrap.py
       ├─ graph_data_preparation.py：Neo4j → RAG 文档
       ├─ faiss_index_construction.py：FAISS 向量索引
       ├─ hybrid_retrieval.py：BM25/FAISS/图谱混合检索
       ├─ graph_rag_retrieval.py：图路径、多跳、子图检索
       ├─ intelligent_query_router.py：查询路由
       └─ generation_integration.py：基于证据上下文生成答案
```

## 项目结构

```text
app.py                              # Streamlit Web UI
config.py                           # 配置
封神演义.txt                         # 原始语料
.env.example                        # 环境变量模板
rag_modules/
  fengshen_kg_extraction.py          # LLM 知识抽取模块
  gold_schema.py                     # Schema 定义
  graph_data_preparation.py          # Neo4j → RAG 文档
  faiss_index_construction.py        # FAISS 索引
  hybrid_retrieval.py                # 混合检索
  graph_rag_retrieval.py             # 图检索
  generation_integration.py          # LLM 答案生成
  intelligent_query_router.py        # 智能路由
scripts/
  import_fengshen_to_neo4j.py        # 导入入口
  run_e2e_queries.py                 # 端到端问答检查
tests/                              # 自动化测试
ui/styles.py                        # Streamlit 样式
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，填写：

```bash
SILICONFLOW_API_KEY=sk-xxx
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=12345678
```

### 3. 导入知识图谱

建议先测试前 3 回：

```bash
python scripts/import_fengshen_to_neo4j.py --chapter-limit 3
```

全量导入：

```bash
python scripts/import_fengshen_to_neo4j.py
```

如果已有 LLM 抽取缓存，只写入 Neo4j：

```bash
python scripts/import_fengshen_to_neo4j.py --skip-extraction
```

只做抽取，不写数据库：

```bash
python scripts/import_fengshen_to_neo4j.py --extraction-only --chapter-limit 3
```

### 4. 启动问答系统

```bash
streamlit run app.py
```

打开页面后点击「初始化系统」。

## 核心 Prompt

知识抽取 Prompt 位于：

- `rag_modules/fengshen_kg_extraction.py` 的 `NER_RE_SYSTEM_PROMPT`

该 Prompt 要求模型识别 Person、Faction、Location、Artifact、Beast、Formation、Event、DeityPosition 等实体，并抽取师承、阵营、亲属、对抗、法宝、事件、封神结局等关系。每条关系必须包含 evidence 原文证据，且只抽取文本中明确出现或可严格推断的事实。

问答生成 Prompt 位于：

- `rag_modules/generation_integration.py` 的 `generate_adaptive_answer()`

该 Prompt 要求模型只基于检索信息回答，优先使用原文摘录验证观点，每个关键事实标注 `[1]`、`[2]` 等证据编号；信息不足时必须说明“根据当前资料无法完全确定”。

## 测试案例与验收建议

| 类型 | 问题 | 重点验证 |
|------|------|----------|
| 简单查询 | 哪吒是谁？ | 人物简介、原文证据、引用编号 |
| 多跳推理 | 哪吒的师父是谁？他属于哪个教派？有哪些法宝？ | 师承 → 教派 → 法宝链路 |
| 图谱关系 | 姜子牙和元始天尊是什么关系？ | 人物关系与师承图谱 |
| 边界问题 | 孙悟空在封神演义中有什么法宝？ | 资料不足时拒绝编造 |

端到端脚本：

```bash
python scripts/run_e2e_queries.py
```

自动化测试：

```bash
pytest
```

### 纯向量 RAG vs GraphRAG 对比建议

项目验收报告中建议列出以下对比表。纯向量 RAG 只使用 FAISS 文本块；GraphRAG 使用智能路由后的图谱检索/混合检索结果。

| 问题 | 纯向量 RAG 预期特点 | GraphRAG 预期特点 |
|------|---------------------|-------------------|
| 哪吒是谁？ | 可从相似文本块回答基础介绍 | 可同时给出人物节点、原文片段和相关关系 |
| 哪吒的师父是谁？他属于哪个教派？有哪些法宝？ | 可能召回分散片段，需要模型自行拼接 | 可沿师承、教派、法宝关系进行多跳组织 |
| 孙悟空在封神演义中有什么法宝？ | 可能因相似神魔文本产生误答 | 更容易基于图谱无实体/无证据而拒答 |

## 验收对照

- ≥500 条非结构化文本：由全书按回/段落切分 TextChunk 满足
- 文本清洗与分句：`parse_chapters()`、`split_chapter_text()`、`_persist_text_chunks()`
- LLM NER/RE：`rag_modules/fengshen_kg_extraction.py`
- 标准三元组：关系写入 Neo4j，UI/检索中以 source-relation-target 形式展示
- 图数据库：Neo4j 节点与关系 + 索引
- 实体对齐：`_merge_aliased_entities()` 基于 alias 合并重复实体
- 向量数据库：FAISS
- 实体链接、多跳、子图检索：`graph_rag_retrieval.py`
- 混合检索：`hybrid_retrieval.py` + `intelligent_query_router.py`
- QA Pipeline：`generation_integration.py`
- Web UI 和图谱溯源：`app.py`
- 自动化验证：`tests/` + `scripts/run_e2e_queries.py`

## 交付物建议

最终提交建议包含：

- 源代码、`requirements.txt`、`.env.example`
- `封神演义.txt` 或说明数据获取方式
- LLM 抽取缓存样例或 `data/fengshen/extraction_results/merged_kg.json`
- 项目报告：需求、设计、实现、Prompt、测试、纯向量 RAG vs GraphRAG 对比
- 5 分钟演示视频或 PPT：展示图谱构建、问答交互、原文溯源、知识图谱子图

不建议提交：`.env`、`__pycache__/`、`.pytest_cache/`、`~$*.docx`、`.claude/`、`.cursor/`、`.trellis/`、无关小说文本和本地模板文件。
