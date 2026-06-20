# 项目示意图 / 流程图

以下图均使用 Mermaid 编写，可直接粘贴到支持 Mermaid 的 Markdown 编辑器、Typora、Obsidian、GitHub、Mermaid Live Editor 或项目报告中渲染。

---

## 1. 系统总体架构图

```mermaid
flowchart LR
    subgraph Data[数据层]
        TXT[封神演义.txt\n99回原文语料]
        CACHE[data/fengshen/extraction_results\nLLM抽取缓存]
    end

    subgraph KGBuild[知识图谱构建层]
        SPLIT[章节解析与文本切块\nparse_chapters / split_chapter_text]
        EXTRACT[LLM知识抽取\nNER + RE Prompt]
        SCHEMA[Schema约束\nPerson/Faction/Artifact/Event/...]
        ALIGN[实体对齐\nalias合并重复实体]
    end

    subgraph Store[存储层]
        NEO4J[(Neo4j知识图谱\n实体/关系/TextChunk/索引)]
        FAISS[(FAISS向量索引\nfengshen_faiss_index)]
    end

    subgraph Retrieval[检索与路由层]
        DATA_PREP[GraphDataPreparation\nNeo4j转RAG文档]
        HYBRID[Hybrid Retrieval\nBM25 + FAISS + 图谱补充]
        GRAPH[GraphRAG Retrieval\n实体链接/多跳/子图/路径]
        ROUTER[Intelligent Query Router\n问题复杂度分析与策略选择]
    end

    subgraph Generation[生成与证据层]
        CONTEXT[上下文融合\n原文片段 + 图谱三元组]
        LLM[LLM答案生成\n只基于证据回答]
        EVIDENCE[引用与溯源\nsource_evidence/snippet_service]
    end

    subgraph UI[展示层]
        APP[Streamlit Web UI\napp.py]
        ANSWER[答案输出]
        SOURCES[原文片段引用]
        SUBGRAPH[知识图谱子图可视化]
    end

    TXT --> SPLIT
    SPLIT --> EXTRACT
    EXTRACT --> SCHEMA
    SCHEMA --> CACHE
    SCHEMA --> ALIGN
    ALIGN --> NEO4J
    SPLIT --> NEO4J

    NEO4J --> DATA_PREP
    DATA_PREP --> FAISS
    DATA_PREP --> HYBRID
    FAISS --> HYBRID
    NEO4J --> GRAPH
    HYBRID --> ROUTER
    GRAPH --> ROUTER

    ROUTER --> CONTEXT
    CONTEXT --> LLM
    LLM --> EVIDENCE

    APP --> ROUTER
    EVIDENCE --> ANSWER
    EVIDENCE --> SOURCES
    GRAPH --> SUBGRAPH
    ANSWER --> APP
    SOURCES --> APP
    SUBGRAPH --> APP
```

---

## 2. 图谱构建流程图

```mermaid
flowchart TD
    A[开始：读取 封神演义.txt] --> B[解析章节标题\n按“第X回”定位章节边界]
    B --> C[按段落与标点切分文本\n生成可处理文本块]
    C --> D[调用LLM抽取知识]
    D --> E{抽取结果是否为合法JSON?}
    E -- 否 --> F[记录错误/跳过或重试]
    F --> D
    E -- 是 --> G[解析实体 Entities]
    E -- 是 --> H[解析关系 Relations]

    G --> I[按 gold_schema 校验实体类型]
    H --> J[按 ALLOWED_RELATION_TYPES 校验关系类型]
    I --> K[写入Neo4j实体节点]
    J --> L[写入Neo4j关系边]
    C --> M[写入TextChunk原文片段节点]
    M --> N[TextChunk关联Chapter]

    K --> O[创建节点索引\nname/id/chunk_id/chapter_id]
    L --> O
    N --> O
    O --> P[基于alias做实体对齐]
    P --> Q[保存/复用抽取缓存]
    Q --> R[完成：可供GraphRAG检索]
```

---

## 3. 用户问答时序图

```mermaid
sequenceDiagram
    actor User as 用户
    participant UI as Streamlit UI app.py
    participant Router as IntelligentQueryRouter
    participant Hybrid as HybridRetrieval
    participant Graph as GraphRAGRetrieval
    participant Neo4j as Neo4j知识图谱
    participant Faiss as FAISS向量索引
    participant Gen as GenerationIntegration
    participant Evidence as SourceEvidence/SnippetService
    participant LLM as 大模型API

    User->>UI: 输入问题
    UI->>Router: route_query(question, top_k)
    Router->>Router: 分析问题复杂度/实体/关系强度

    alt 简单事实或文本证据优先
        Router->>Hybrid: 传统混合检索
        Hybrid->>Faiss: Top-K向量检索
        Hybrid->>Neo4j: 实体/主题补充查询
        Faiss-->>Hybrid: 相似原文片段
        Neo4j-->>Hybrid: 相关节点与关系
        Hybrid-->>Router: 文档候选
    else 多跳关系/图谱推理问题
        Router->>Graph: 图谱检索
        Graph->>Neo4j: 实体链接 + 多跳路径/子图查询
        Neo4j-->>Graph: 路径、邻居、三元组、原文证据
        Graph-->>Router: 图谱增强文档
    else 组合策略
        Router->>Hybrid: 文本检索
        Router->>Graph: 图谱检索
        Hybrid-->>Router: 文本片段
        Graph-->>Router: 图谱事实
    end

    Router-->>UI: 检索文档 + 路由解释
    UI->>Gen: generate_adaptive_answer(question, docs)
    Gen->>Evidence: 构造带编号证据上下文
    Evidence-->>Gen: 原文摘录 + 图谱事实
    Gen->>LLM: 只基于证据生成答案
    LLM-->>Gen: 带引用编号的回答
    Gen-->>UI: 答案
    UI->>Evidence: 提取引用片段/三元组
    Evidence-->>UI: 来源卡片与子图数据
    UI-->>User: 展示答案、原文溯源、知识图谱子图
```

---

## 4. 检索融合流程图

```mermaid
flowchart TD
    Q[用户问题] --> A[问题分析]
    A --> B[抽取候选实体\n人物/教派/法宝/事件]
    A --> C[判断查询类型\n简单查询/多跳/路径/子图]
    A --> D[判断是否需要图谱推理]

    D -->|文本证据优先| V[向量检索 FAISS Top-K]
    D -->|关系推理优先| G[图谱检索 Neo4j]
    D -->|组合| BOTH[并行执行文本检索与图谱检索]

    V --> VDOC[相似原文片段]
    G --> GLINK[实体链接]
    GLINK --> GHOP[一跳/多跳邻居]
    GHOP --> GPATH[路径与子图]
    GPATH --> GFACT[三元组事实]

    BOTH --> VDOC
    BOTH --> GLINK

    VDOC --> FUSE[上下文融合]
    GFACT --> FUSE
    FUSE --> RANK[去重、排序、证据优先]
    RANK --> PROMPT[构造问答Prompt\n原文摘录 + 图谱事实 + 引用编号]
    PROMPT --> ANSWER[LLM生成答案]
    ANSWER --> SOURCE[展示引用片段与知识子图]
```

---

## 5. 知识图谱 Schema 示意图

```mermaid
graph TD
    Person[Person\n人物]
    Faction[Faction\n教派/势力]
    Location[Location\n地点]
    Artifact[Artifact\n法宝]
    Beast[Beast\n坐骑/灵兽]
    Formation[Formation\n阵法]
    Event[Event\n事件/战役]
    Deity[DeityPosition\n神位/封号]
    Chapter[Chapter\n章节]
    TextChunk[TextChunk\n原文片段]

    Person -- MASTER_OF / APPRENTICE_OF --> Person
    Person -- FATHER_OF / CHILD_OF / BROTHER_OF / MARRIED_TO --> Person
    Person -- BELONGS_TO_SECT / FIGHTS_FOR --> Faction
    Person -- OWNS / BESTOWS / LOSES / STEALS --> Artifact
    Person -- OWNS --> Beast
    Person -- KILLS / DEFEATS / CAPTURES / OPPOSES --> Person
    Person -- PARTICIPATES_IN / LEADS / INITIATES --> Event
    Person -- CREATES / DEPLOYS / BREAKS --> Formation
    Person -- LISTED_ON / BECOMES --> Deity

    Event -- OCCURS_IN --> Location
    Event -- PARTICIPATES_IN --> Faction
    Formation -- OCCURS_IN --> Location
    TextChunk -- BELONGS_TO_CHAPTER --> Chapter
    TextChunk -- MENTIONS --> Person
    TextChunk -- MENTIONS --> Event
    TextChunk -- MENTIONS --> Artifact
```

---

## 6. 适合放在报告里的系统流程简图

如果报告只放一张图，建议使用下面这张更紧凑的版本：

```mermaid
flowchart LR
    A[封神演义原文] --> B[文本清洗/章节解析/切块]
    B --> C[LLM实体识别与关系抽取]
    C --> D[(Neo4j知识图谱)]
    B --> E[TextChunk文档]
    E --> F[(FAISS向量索引)]

    Q[用户问题] --> R[智能路由]
    R --> F
    R --> D
    F --> H[相关原文片段]
    D --> I[实体链接/多跳路径/子图三元组]
    H --> J[证据上下文融合]
    I --> J
    J --> K[LLM基于证据生成答案]
    K --> L[Streamlit展示\n答案 + 引用 + 子图]
```

---

## 7. 如果改用 Graphviz DOT

```dot
digraph FengshenGraphRAG {
  rankdir=LR;
  node [shape=box, style="rounded,filled", fillcolor="#F8F5EC", color="#8B6F47", fontname="Microsoft YaHei"];
  edge [color="#8B6F47", fontname="Microsoft YaHei"];

  Text [label="封神演义.txt\n原始语料"];
  Split [label="章节解析 / 文本切块"];
  Extract [label="LLM NER + RE\n实体关系抽取"];
  Neo4j [label="Neo4j知识图谱\n实体/关系/TextChunk/索引", shape=cylinder, fillcolor="#E8F1FA"];
  Faiss [label="FAISS向量索引\nTop-K文本检索", shape=cylinder, fillcolor="#E8F1FA"];
  Router [label="智能查询路由\n复杂度/关系强度/策略选择"];
  Graph [label="GraphRAG检索\n实体链接/多跳/子图"];
  Hybrid [label="混合检索\nBM25/FAISS/图谱补充"];
  Context [label="上下文融合\n原文片段 + 图谱三元组"];
  LLM [label="LLM证据生成\n防幻觉Prompt"];
  UI [label="Streamlit UI\n答案/引用/知识子图"];

  Text -> Split -> Extract -> Neo4j;
  Split -> Faiss;
  Router -> Graph -> Neo4j;
  Router -> Hybrid -> Faiss;
  Hybrid -> Neo4j;
  Graph -> Context;
  Hybrid -> Context;
  Context -> LLM -> UI;
}
```

---

## 8. 图片生成描述词

如果需要用 AI 绘图工具生成一张更美观的架构图，可使用以下描述：

> 生成一张中文技术架构图，主题为“《封神演义》GraphRAG 问答系统”。画面从左到右分为五层：数据层、知识图谱构建层、存储层、检索生成层、展示层。左侧是《封神演义.txt》原文数据，经过章节解析、文本切块、LLM实体识别和关系抽取，进入 Neo4j 知识图谱；文本块同时进入 FAISS 向量索引。中间是智能查询路由，根据用户问题选择向量检索、图谱多跳检索或混合检索。右侧是大模型基于“原文片段 + 图谱三元组”生成带引用的答案，最终在 Streamlit 页面展示答案、原文溯源和知识图谱子图。整体风格为学术论文架构图，浅色背景，蓝金配色，清晰箭头，模块边框圆角，中文标签简洁。不要生成代码，不要生成复杂背景。
