# 熹茗电商 AI 客服系统

基于 RAG + Tool Use 的茶叶电商客服 Agent，覆盖**产品推荐 / 冲泡指导 / 售后处理**三大场景。

## 与题目的对照

| 题目要求 | 实现 |
|---|---|
| 1) 梳理 6 大类核心知识节点 | `ingestion/schema.py`：24 个 NodeSpec（产品 5 / 工艺 4 / 冲泡 4 / 饮用建议 4 / 售后 5 / 品牌 2），每类含必填+可选字段契约 |
| 2) **用 AI 工具获取资料，不用传统爬取** | 搜索 5 后端 / 抓取 3 后端混合（Jina Reader + Tavily + Firecrawl + SerpAPI + Jina-via-Bing/DDG），自动 fallback；详见 §架构 |
| 3) 基于 RAG 的架构 | Milvus Lite（生产换 Milvus 只改 URI）+ bge-small-zh-v1.5 中文 embedding + BM25 双路混合 + Tool Use（订单/物流/库存/工单/转人工）+ 出站合规过滤 + Agent 编排（Claude Opus 4.7，system 含 prompt cache） |
| 4) 三类场景 Prompt | `prompts/` 下 4 份 markdown：system_base + recommend / brewing / aftersale |
| 真实业务场景 | sources.yaml = 18 条品牌 URL + 28 条扩散 query + 12 条行业兜底 query；**实跑 107 候选 → 54 抓回 → 437 节点入 Milvus** |
| 茶行业常见问题 | 已实测覆盖：产品推荐 / 冲泡水温 / 7 天无理由退换 / 茶饼发霉 / 孕妇能否喝茶 / 白茶储存 / 品牌创立人 7 类查询全部命中相关节点 |
| 准确性保障 | 入库前敏感词分流（33 条进 pending_review）+ LLM 抽取契约校验 + 生成时 system red lines + 出站正则过滤 + 检索置信度兜底转人工 |

> **数据来源**：本仓库**不携带任何品牌数据**，全部知识由 `ingestion/` 流水线从公开互联网真实获取（经 Jina Reader API），按 24 节点 schema LLM 抽取后入 Milvus。

## 架构

```
渠道接入 → 会话编排 → 混合检索（BM25+TF-IDF+元数据过滤+Rerank）
                  ↘ Tool Use（订单/物流/库存/工单/转人工）
                  ↘ Claude Opus 4.7（带 Prompt Caching 的 system）
                  ↘ 出站合规过滤 → 用户
```

目录：

```
xim/
├── ingestion/        # 资料获取 + LLM 结构化抽取流水线
│   ├── sources.yaml      # 18 条已核实熹茗 URL + 16 条搜索 query + 敏感词
│   ├── schema.py         # 6 大类 24 节点 NodeSpec 契约
│   ├── discover.py       # SerpAPI / DuckDuckGo 搜索发现 + 品牌相关性过滤
│   ├── fetch.py          # Jina Reader API（不传统爬取）
│   ├── extract.py        # Claude 按 schema.py 契约抽取 + 字段校验
│   └── pipeline.py       # 主入口
├── retrieval/        # 混合检索
│   ├── vector_store.py   # Milvus Lite（文件存储；生产换 Milvus 只改 URI）+ bge-small-zh-v1.5
│   └── hybrid_search.py  # 向量召回 + BM25 + 同义词扩展 + 元数据过滤 + 加权融合
├── safety/           # 出站合规过滤（功效/绝对化/竞品贬损正则）
├── tools/            # Tool Use：订单/物流/库存/售后工单/转人工（OMS 未配置时返回 NOT_CONFIGURED 自然降级）
├── prompts/          # system 基座 + 推荐/冲泡/售后三场景
├── agent/            # 意图分类 → 检索 → 工具循环 → 出站过滤
├── eval/             # 黄金集（10 例） + 自动回归脚本
├── data/             # vector.db (Milvus Lite) + knowledge/{nodes,pending_review,manifest}.json
└── cli.py            # 交互式 / 单条问答 / debug
```

## 快速开始

```bash
# 1) 安装
pip install -r requirements.txt

# 2) 配置 key
cp .env.example .env
# 编辑 .env 填入 OPENROUTER_API_KEY（推荐）或 ANTHROPIC_API_KEY；
# 可选：FIRECRAWL_API_KEY（强反爬源兜底）/ JINA_API_KEY（提升配额）/ SERPAPI_API_KEY

# 3) 配置采集源（必做）
# 编辑 ingestion/sources.yaml：
#   - 把示例占位 URL 换成熹茗实际官网 / 旗舰店
#   - 增删 search_queries
$EDITOR ingestion/sources.yaml

# 4) 跑一次采集（discover → Jina Reader 取干净 markdown → Claude 按 schema 抽取 → 推 Milvus）
python -m ingestion.pipeline --dry-run                                # 列候选 URL
python -m ingestion.pipeline --only-fetch https://www.ximingcha.com/  # 只抓不抽（验证 URL 可达）
python -m ingestion.pipeline                                          # 真实采集
# 产物：data/knowledge/nodes.jsonl            ← 已通过初筛，可入库
#       data/knowledge/pending_review.jsonl   ← 敏感/低置信，待人审
#       data/knowledge/manifest.json          ← 本次运行 metadata

# 5) 进对话
python cli.py
python cli.py --once "想买茶送领导，预算800，他喝岩茶多" --debug

# 6) 黄金集回归
python eval/run_eval.py
```

## 关键设计

### 1) 数据采集——多 API 混合 AI 资料获取（不做传统爬取）

**搜索发现层** `ingestion/search_backends.py`：5 后端按优先级 fallback
| 优先级 | 后端 | 是否需 key | 备注 |
|---|---|---|---|
| 1 | Tavily Search | TAVILY_API_KEY（免费 1000/月） | AI 摘要，最稳 |
| 2 | SerpAPI | SERPAPI_API_KEY | 百度/Google |
| 3 | **Jina Reader 抓 DuckDuckGo HTML 结果页** | 无（免费） | 实测最常命中 |
| 4 | **Jina Reader 抓 Bing 搜索结果页** | 无 | 兜底 |
| 5 | duckduckgo-search 库 | 无 | 偶尔限速 |

**资料获取层** `ingestion/fetch_backends.py`：3 后端按优先级 fallback
| 优先级 | 后端 | 是否需 key | 备注 |
|---|---|---|---|
| 1 | **Jina Reader** `https://r.jina.ai/<url>` | 无（免费） | 返回干净 markdown，反爬/JS/编码全兜底 |
| 2 | Tavily Extract | TAVILY_API_KEY | 反爬严的页面更稳 |
| 3 | Firecrawl | FIRECRAWL_API_KEY | JS 渲染最强 |

**抽取双轨**：
- `--mode llm` 默认：Claude 按 `schema.py` 24 节点契约结构化抽取
- `--mode section`：按 markdown 标题切原始段（无 LLM key 也能填库）

**合规分流**：命中疾病 / 绝对化 / 竞品贬损 → `pending_review.jsonl`，运营+法务白名单后才入库

### 2) 检索——混合而非纯向量

**当前实现没有部署外部向量数据库**。检索栈是：

| 层 | 实现 | 角色 |
|---|---|---|
| 关键词召回 | `rank_bm25` (BM25Okapi) | 中文术语精确匹配（如 SKU 名、产区） |
| "向量"召回 | `sklearn.TfidfVectorizer` 稀疏向量 + 余弦相似度 | 语义近似项（小型库无需外部服务即可端到端跑通） |
| 分词 | `jieba` 中文分词 | 给 BM25 / TF-IDF 共用 |
| 同义词扩展 | 领域词典（`大红袍↔DHP`、`老枞↔老丛`、`退货↔退款↔退换`…） | 召回扩散 |
| 元数据过滤 | category / node_type 硬过滤 | 避免问"白茶"召回"岩茶"；问"怎么泡"优先 brewing 节点 |
| 置信度 | top1 分数 + (top1 − top2) gap | 低于阈值直接转人工 |

为什么不上来就接 Milvus/PGVector：节点数 < 几万条时，BM25+TF-IDF 在中文专名（"肉桂""老枞"等同形异义词多）上的精度反而**优于纯稠密向量**；而且零外部依赖，`pip install -r requirements.txt` 后立即能跑。

**生产升级路径**（数据量 > 5 万节点 / 需要跨语言召回 / 需要语义模糊匹配）：

| 项 | 当前 | 推荐升级 |
|---|---|---|
| 向量库 | sklearn 内存稀疏矩阵 | **Milvus**（国内云成熟、支持亿级）/ **PGVector**（已用 PostgreSQL 时最省运维）/ **Qdrant**（轻量自托管）/ **Chroma**（demo/小项目） |
| Embedding | TF-IDF 稀疏 | **bge-large-zh-v1.5**（开源、中文 SOTA）或 Voyage AI `voyage-3` |
| Rerank | 加权融合 | **bge-reranker-v2-m3** 精排 top-20 → top-5 |
| 关键词 | rank_bm25 | **Elasticsearch / OpenSearch** 同义词词典 |

切换只需替换 `retrieval/hybrid_search.py` 中的 `_tfidf` 与 `_mat` 两处即可，外部接口（`HybridIndex.search`）不变。

### 3) 编排——Tool Use 与 RAG 分离
- 静态知识（产品介绍、冲泡参数、政策）走 RAG
- **动态数据（订单、物流、库存、价格）必须走工具调用**——绝不让 LLM 凭检索结果里的旧价/旧库存作答
- 5 类工具：`order_query` / `logistics_query` / `stock_check` / `aftersale_ticket` / `handoff_to_human`
- OMS 未配置时返回 `NOT_CONFIGURED`，Agent 自动降级为"暂未连通系统，请提供订单号截图"

### 4) 合规——三道防线
1. **入库前**：采集流水线敏感词 → `pending_review.jsonl` 人审
2. **生成时**：System Prompt 写死红线（"不得宣称疗效""不用绝对化用语"）
3. **出站时**：`safety/compliance_filter` 正则双层拦截，命中即重写为安全话术 + 转人工

### 5) 升级转人工——多重触发
- 关键词命中（315/曝光/起诉/媒体/赔我/没人管…）→ 立即转
- 工具调用决定（金额>500 在 `aftersale_ticket` 时升级）
- 合规违规重写 → 自动转
- 检索置信度 < 阈值 → 兜底转

### 6) Prompt Caching
System 基座 + 三场景 prompt 拼接为单块带 `cache_control: ephemeral` —— 跨请求复用，降低首响时延与成本。

## 验证

```bash
# 离线回归（需 ANTHROPIC_API_KEY；OMS 不需要）
python eval/run_eval.py
```

期望指标：
- 通过率 ≥ 90%
- 合规违规命中数 = 0
- 升级关键词样例 100% 转人工

## 部署提示

- 生产可把 `XIMING_MODEL` 切成 `claude-sonnet-4-6`（吞吐+成本更优）或 `claude-haiku-4-5`（最快）
- 生产检索建议升级：`bge-large-zh-v1.5` 替换 TF-IDF + Milvus 向量库 + `bge-rerank` 精排
- 生产采集建议升级：动态站点用 Playwright；评测样本用 `client.messages.batches`（5 折扣）
- 与平台对接：把 `cli.py` 的入口换为千牛/抖店 webhook handler，用同一个 `Agent` 实例

## Clone-and-run 五步开箱

**别人 clone 这个仓库后，5 步内能直接对话**——KB（973 节点 100% 官方源）已随仓库提交。

```bash
# 1. 克隆
git clone https://github.com/<your>/xim.git && cd xim

# 2. 装依赖（约 4GB，含 torch + sentence-transformers + bge 模型）
pip install -r requirements.txt

# 3. 配 LLM key（OpenRouter 推荐——国内可达，按量付费）
cp .env.example .env
# 在 .env 里填 OPENROUTER_API_KEY=sk-or-v1-xxx
# 或 ANTHROPIC_API_KEY=sk-ant-xxx

# 4. 启动 Web（uvicorn 自动检测 nodes.jsonl 并重建 Milvus 索引，~20 秒）
python -m uvicorn app:app --port 8000

# 5. 浏览器开 http://localhost:8000
```

**就是这样**——不需要先跑 ingestion，KB 已随仓库 git 跟踪（`data/knowledge/nodes.jsonl` 584KB），首次启动自动 reindex 到 Milvus Lite。

如要刷新 KB（采集新数据）：`python -m ingestion.pipeline --mode llm`（需 LLM key + ~30 分钟）。

---

## 部署到 GitHub & Docker

仓库已经包含完整部署套件，clone 后开箱即用：

### A. 本地 Docker 一键启动

```bash
git clone https://github.com/<your>/xim.git && cd xim
cp .env.example .env       # 填 OPENROUTER_API_KEY（推荐）或 ANTHROPIC_API_KEY

# ★ 推荐：启动 Web UI （http://localhost:8000）
docker compose up web

# 其他命令
docker compose run --rm ingest          # 跑采集（约 3-5 min，113 SKU + 33 品牌页）
docker compose run --rm chat            # 进交互式 CLI
docker compose run --rm chat python cli.py --once "牛魁肉桂多少钱" --debug
docker compose run --rm eval            # 黄金集回归
```

### A.bis. 不用 Docker 直接跑 Web UI

```bash
pip install -r requirements.txt
python -m uvicorn app:app --reload --port 8000
# 浏览器打开 http://localhost:8000
```

UI 特性：
- 单文件 HTML，无构建步骤；移动端响应式；中式茶色配色
- 无 ANTHROPIC_API_KEY 时**自动降级**到展示 RAG top-3 命中（依然能 demo 检索能力）
- 顶部实时显示 KB 节点数 / SKU 数 / embedding 模型 / Anthropic key 状态
- 「开发者模式」开关：把 intent / confidence / tool_calls / 合规违规命中都显示出来
- 5 个预置 query pill 一键测试

REST API：

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/healthz` | 存活探针，返回 `{ok, kb_nodes}` |
| GET | `/stats` | KB 详细统计（按品类/host/embed model/llm/key 状态） |
| POST | `/chat` | 完整客服回复（非流式，含工具循环 + 合规过滤）|
| **POST** | **`/chat/stream`** | **SSE 流式（首字 ~2.5s，体感流畅）—— 客服默认走这条** |
| GET | `/admin` | KB 浏览器（搜索 / 筛选 / 浏览全部 970 节点） |
| GET | `/nodes` | 列表 / 关键词检索 / 多维过滤 |
| GET | `/nodes/{doc_id}` | 单节点详情 |
| GET | `/pending` | 待人审节点（命中敏感词，未入库） |
| GET | `/` | 内置 Web UI |

镜像构建时已**预下载 bge-small-zh 模型** (`pip install` 后 `RUN python -c "SentenceTransformer(...)"`)，运行时不再拉网络模型。`./data` 挂载持久化 Milvus 库。

### B. GitHub Actions 自动化

| Workflow | 触发 | 作用 |
|---|---|---|
| [`ci.yml`](.github/workflows/ci.yml) | push / PR | 语法检查、模块导入、24 节点 schema 完整性、合规过滤器单测、sources.yaml 校验（≥100 SKU + 必有白名单） |
| [`ingest.yml`](.github/workflows/ingest.yml) | 每周日 03:00 北京时间 + 手动触发 | 跑采集流水线，新 `manifest.json` 提交回主分支，`nodes.jsonl` 作为 artifact 保留 30 天 |

仓库需配置的 Secrets（`Settings → Secrets → Actions`）：
- `OPENROUTER_API_KEY` —— **推荐**，国内可达、支持 Anthropic / OpenAI / DeepSeek / Qwen 全系
- `ANTHROPIC_API_KEY` —— 备选，Anthropic 直连
- `XIMING_MODEL` —— 抽取模型，默认 `anthropic/claude-sonnet-4.5`（OpenRouter）/ `claude-opus-4-7`（Anthropic）
- `XIMING_CHAT_MODEL` —— 客服回复模型，默认 `anthropic/claude-haiku-4.5`（更快，TTFT ~2.5s）
- `JINA_API_KEY` —— 可选，提升 Jina Reader 配额
- `TAVILY_API_KEY` —— 可选，反爬严的页面 fallback
- `FIRECRAWL_API_KEY` —— 可选，JS 重渲染兜底

### C. Codespaces / 云端 IDE

GitHub 上点 `Code → Codespaces → Create on main`，60s 内得到完整开发环境（已装 deps、bge 模型已就位）。

### D. 进一步部署到 Web（不在本套件，下一步可加）

| 目标 | 方案 |
|---|---|
| HTTP API（千牛 / 抖店 webhook 接入） | 加 `app.py` FastAPI（200 行）→ `Dockerfile` 改 `CMD ["uvicorn","app:app"]` → push GHCR → Railway/Render 一键部署 |
| 大流量 | Milvus URI 切到云集群（`MILVUS_URI=http://...:19530`），LLM 模型切 `claude-sonnet-4-6` 或 `claude-haiku-4-5` |

## License

仅作面试 / 演示用途。
