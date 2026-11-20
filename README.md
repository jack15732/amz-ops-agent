# AMZ Ops Agent

[![CI](https://github.com/jack15732/amz-ops-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/jack15732/amz-ops-agent/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

亚马逊跨境 Listing 上架前合规预审系统：**规则引擎判违规 + 混合 RAG 溯源 + Function Calling Agent 分流**。

面向运营经验零散、新人上手慢、人工合规漏检率高的问题，把「词库检测 → 风险分级 → 知识溯源 → 人工转接 → 审计留档」做成可运行、可评测、可接入上架流程的闭环。

> 个人项目，代码量刻意保持克制：约 5k 行 Python（核心约 4.4k，含评测与测试），无微服务、无消息队列，一条命令即可本地跑通。

---

## 1. 核心能力

| 模块 | 实现 |
| --- | --- |
| 多站点合规词库 | YAML 维护 US / EU / JP 三套规则，支持词条、正则、白名单上下文、严重级别、版本号 |
| 边界与误报控制 | NFKC 归一化 + 原文 offset 映射；ASCII 词边界；`allow_contexts` 放行兼容性/售后等合法语境 |
| 分级风险判定 | 高危 `blocked` 直接拦截；模糊 `review` 交 LLM 复核并打标；无依据不直接放行 |
| 多格式知识入库 | Markdown / Word / PDF / Excel；标题层级切块 + 滑动窗口；chunk 携带 heading path、页码/工作表/行号 |
| 混合检索 | Qdrant 向量（BGE-M3）+ BM25（中文 jieba + CJK bigram）；RRF 融合；跨境缩写双向扩展 |
| 无依据守卫 | 向量阈值、BM25 阈值、OOV 比例三重判断；无可靠依据时检索结果标记 `sufficient=false` |
| Agent 工具编排 | `search_knowledge` / `check_compliance` / `lookup_clause` / `request_human_review` |
| 强约束 | 结论以规则引擎为准；必须用原文检测；引用必须来自检索片段；无依据转人工；模型不能把 blocked 改成 passed |
| 工程落地 | FastAPI、Streamlit 内部界面、SQLite 全链路审计、Docker Compose、GitHub Actions |
| 评测 | 自建 82 条合规回归 + 23 条留出集 + 32 条检索 + 18 条 Agent 轨迹；含首轮/迭代后对比 |

---

## 2. 架构

```mermaid
flowchart LR
    U[运营 / Listing 上架流程] --> API[FastAPI]
    UI[Streamlit 内部界面] --> API
    API --> RE[规则引擎]
    API --> AG[Agent Orchestrator]
    AG --> T[工具层]
    T --> RE
    T --> HR[Hybrid Retriever]
    HR --> Q[(Qdrant / BGE-M3)]
    HR --> BM[BM25 + 缩写扩展]
    RE --> Y[YAML 词库 US/EU/JP]
    AG --> LLM[LLM Function Calling]
    API --> AU[(SQLite 审计)]
    T --> AU
```

一次预审的路径：

1. 规则引擎对原文做归一化、边界匹配、白名单过滤，输出命中项与风险分。
2. `blocked` 直接拦截；`review` 触发 LLM 结合上下文复核并打标；未命中则进入检索核验。
3. Agent 强制调用 `check_compliance`，再调用 `search_knowledge`；结论、命中项、引用、工具轨迹一起返回。
4. 无可靠检索依据、引用无效、涉及法律定性时调用 `request_human_review`。
5. 全流程写入 SQLite：trace、tool_calls、compliance_flags，可按 `trace_id` 回放。

---

## 3. 目录结构

```text
.
├── apps/streamlit_app.py          # 内部可视化界面
├── config/
│   ├── settings.yaml              # 主配置
│   ├── lexicon.yaml               # 缩写/术语双向扩展
│   └── compliance/{us,eu,jp}.yaml # 三站点词库
├── data/
│   ├── knowledge/                 # 12 篇种子知识文档
│   └── eval/                      # 合规/检索/Agent 标注集
├── scripts/retrieval_ablation.py  # BM25 vs 向量 vs 混合 消融
├── src/amz_agent/
│   ├── compliance/                # 规则加载、边界匹配、风险分级
│   ├── rag/                       # 切块、入库、embedding、Qdrant、BM25、RRF
│   ├── agent/                     # 工具、Function Calling、护栏
│   ├── llm/                       # OpenAI-compatible 客户端与 Prompt
│   ├── api/                       # FastAPI
│   ├── audit/                     # SQLite 审计
│   ├── eval/                      # 指标与评测 runner
│   ├── service.py                 # 依赖装配
│   └── cli.py                     # amz-agent 命令
├── tests/                         # 24 个单测/集成测试
├── reports/                       # 评测报告（已提交）
├── Dockerfile / docker-compose.yml
└── .github/workflows/ci.yml
```

---

## 4. 快速开始

### 4.1 环境

- Python 3.11+
- 可选：Docker（Qdrant server / 一键起服务）
- 可选：SiliconFlow API Key（BGE-M3 + Qwen）；没有 Key 会自动降级到本地 HuggingFace 或 hash embedder

```bash
git clone https://github.com/jack15732/amz-ops-agent.git
cd amz-ops-agent

python -m venv .venv
source .venv/bin/activate
pip install -e ".[ui,hf,eval,dev]"
# 国内网络可用：
# pip install -i https://mirrors.aliyun.com/pypi/simple/ -e ".[ui,hf,eval,dev]"
```

### 4.2 配置

```bash
cp .env.example .env
# 编辑 .env，至少填入 SILICONFLOW_API_KEY（如果要使用 BGE-M3/Qwen）
```

关键变量：

```dotenv
SILICONFLOW_API_KEY=sk-...
EMBEDDING_PROVIDER=siliconflow      # siliconflow | huggingface | hash
EMBEDDING_MODEL=BAAI/bge-m3
LLM_PROVIDER=siliconflow            # siliconflow | none
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
AMZ_QDRANT_MODE=local               # local | server | memory
```

没有 Key 时的降级顺序：

1. `EMBEDDING_PROVIDER=huggingface` → 本地 `sentence-transformers`（默认 fallback `moka-ai/m3e-base`，已缓存可离线）。
2. 本地模型不可用 → 确定性 `hash` embedder（只用于测试/CI，不代表语义检索效果）。

### 4.3 启动

```bash
# 1) 入库（首次会自动执行；也可手动）
amz-agent ingest data/knowledge --reset

# 2) API
amz-agent serve --port 8000
# 打开 http://127.0.0.1:8000/docs

# 3) 内部界面（另开终端）
streamlit run apps/streamlit_app.py
# 打开 http://127.0.0.1:8501
```

如果没安装 console script，用 `python -m amz_agent.cli` 替代 `amz-agent`。

### 4.4 Docker（Qdrant server + API + UI）

```bash
cp .env.example .env
docker compose up --build
# API: http://127.0.0.1:8000/docs
# UI : http://127.0.0.1:8501
```

Compose 会把 `AMZ_QDRANT_MODE=server`、`QDRANT_URL=http://qdrant:6333` 注入 API；本地开发默认用 Qdrant embedded 模式，不需要 Docker。

---

## 5. 使用示例

### 5.1 规则检测

```bash
amz-agent check --site US --text "Our yoga mat is the best seller and guaranteed to stay flat."
# decision=blocked risk=93
# [US-ABS-002] best / [US-ABS-001] best seller / [US-ABS-001] guaranteed
```

### 5.2 混合检索

```bash
amz-agent search --site EU --query "CE certified 为什么不行" --top-k 3
```

### 5.3 Agent 预审

```bash
amz-agent agent --site JP --text "業界最高の美白クリーム。必ず効果があります。" \
  --question "日本站能上架吗？"
```

### 5.4 HTTP API

```bash
curl -s http://127.0.0.1:8000/v1/compliance/check \
  -H 'Content-Type: application/json' \
  -d '{"text":"100% cure for acne","site":"US"}' | jq

curl -s http://127.0.0.1:8000/v1/agent/query \
  -H 'Content-Type: application/json' \
  -d '{"text":"Compatible with iPhone 15","site":"US","question":"可以上架吗？"}' | jq
```

主要端点：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 依赖/索引/规则状态 |
| POST | `/v1/compliance/check` | 纯规则检测 |
| POST | `/v1/compliance/precheck` | 结构化 Listing 预审（上架流程固定环节） |
| POST | `/v1/knowledge/search` | 混合检索 |
| POST | `/v1/agent/query` | Agent 预审问答 |
| POST | `/v1/knowledge/ingest` | 上传 PDF/Word/Excel/Markdown 入库 |
| GET | `/v1/rules?site=US` | 查看站点规则 |
| GET | `/v1/audit/{trace_id}` | 回放审计轨迹 |
| GET | `/v1/stats` | 服务 + 审计统计 |

---

## 6. 评测结果

评测集全部在 `data/eval/`，口径与字段见 [`data/eval/README.md`](data/eval/README.md)。最终报告：[`reports/eval_report.md`](reports/eval_report.md)。

> 本机无 SiliconFlow Key，最终报告使用本地 `moka-ai/m3e-base` 作为 BGE-M3 的可复现替代；配置 Key 后用同一条命令即可换成 `BAAI/bge-m3`。

### 6.1 总览

| 模块 | 指标 | 结果 |
| --- | --- | --- |
| 合规回归（82 条） | 三分类准确率 / blocked F1 | **100.00% / 100.00%** |
| 合规回归 | 高危漏检率 / 通过样本误报率 | **0.00% / 0.00%** |
| 独立留出集（23 条） | 准确率 | **91.30%** |
| 独立留出集 | 高危漏检率 | **0.00%** |
| 检索（32 条） | Recall@6 / MRR@6 / nDCG@6 | **100.00% / 0.8778 / 0.8784** |
| 检索 | 无答案识别率 | **100.00%** |
| Agent（18 条） | 任务成功率 / 决策准确率 | **100.00% / 100.00%** |
| Agent | 工具召回 / 引用覆盖 / 护栏通过 | **100.00% / 100.00% / 100.00%** |

### 6.2 检索消融

详见 [`reports/retrieval_comparison.md`](reports/retrieval_comparison.md)：

| 模式 | Recall@6 | Precision@6 | MRR@6 | nDCG@6 | 无答案识别 |
| --- | --- | --- | --- | --- | --- |
| BM25 | 100.00% | 37.56% | 0.8028 | 0.8388 | 0.00% |
| 向量 | 100.00% | 45.89% | 0.8722 | 0.8386 | 0.00% |
| **混合（RRF）** | **100.00%** | **41.06%** | **0.8778** | **0.8784** | **100.00%** |

结论：单路检索在“无答案”问题上都会误召回；RRF 融合后 MRR/nDCG 最优，叠加 OOV/阈值守卫后无答案识别提升到 100%。

### 6.3 留出集迭代记录

- 首轮留出集只补了 23 条语义改写/边界样本，初始准确率 **43.48%**，暴露 8 条高危漏检（Top of the line、CE self-certified、世界で最も…）。
- 补充高频变体与 allowlist 作用域修复后复测 **91.30%**，高危漏检 **0**；仍保留 2 条“review 被判 blocked”的保守误拦截（doctor-formulated / clinically studied）。
- 这说明纯词表规则对语义改写的覆盖有边界，生产环境必须保留人工抽检与规则迭代。

### 6.4 复现

```bash
# 本地 HF（无需 API Key，报告中的配置）
amz-agent eval --suite all \
  --embedding-provider huggingface --embedding-model moka-ai/m3e-base \
  --report-dir reports

# SiliconFlow BGE-M3（配置 Key 后）
amz-agent eval --suite all \
  --embedding-provider siliconflow --embedding-model BAAI/bge-m3

# 消融
python scripts/retrieval_ablation.py \
  --embedding-provider huggingface --embedding-model moka-ai/m3e-base

# 可选：把四份评测集发布到 Hugging Face Dataset
python scripts/push_eval_to_hf.py --repo-id <your-account>/amz-ops-eval
```

---

## 7. 关键设计

### 7.1 规则引擎：边界与白名单

- NFKC 归一化 + 原文 offset 映射，命中片段可回到原文位置，避免“模型说命中但指不出哪里”。
- 只在 ASCII 边缘加词边界：`CE` 不会命中 `certificate`，`100%`、`#1` 仍可命中。
- `allow_contexts` 支持合法语境：`Compatible with iPhone`、`Satisfaction guarantee`、`Not a medical device`、`Apple cider vinegar`。
- 白名单作用域是“规则级”的：同一个 allow context 不会掩盖同规则下另一处真实违规（留出集 EU 医疗宣称案例就是据此修复的）。
- 高危规则 `block`，模糊规则 `review`；LLM 只能升级拦截或对低危内容高置信放行，不能静默放过中高危。

### 7.2 RAG：可溯源切块

- Markdown 按标题栈切；PDF 按页；Word 按 Heading 样式；Excel 按工作表 + 行区间。
- 章节超过 `max_chars` 后按标点滑动切窗，重叠 `chunk_overlap`；每个 chunk 携带 `heading_path / source / page / sheet / row_start / row_end`。
- 引用返回 `chunk_id + doc_id + heading_path + source`，可直接定位原文。

### 7.3 混合检索与缩写扩展

- 向量：BGE-M3（SiliconFlow）或本地 sentence-transformers；Qdrant cosine。
- 关键词：jieba + CJK bigram + 拉丁词；BM25 参数 `k1=1.5, b=0.75`。
- RRF：`w_v/(k+rank_v) + w_b/(k+rank_b)`，默认 `k=60, w_v=1.2, w_b=1.0`（由消融实验调参）。
- `config/lexicon.yaml` 做双向扩展：BSR / ACOS / GPSR / 薬機法 / 景表法 等。
- 无依据守卫：向量/BM25 阈值 + query OOV 比例，避免把“火星站”“2099 量子条款”硬答成政策。

### 7.4 Agent 强约束

- 工具集固定四个：检索、合规检测、条款查询、人工转接。
- 运行时自动用**原文**执行一次 `check_compliance`，模型改写文本也绕不过去。
- 最终 decision 由规则引擎决定；模型输出“可以上架”但规则 blocked 时，系统会纠正并标记 override。
- 引用编号必须能解析到本轮检索片段；没有引用时自动补 `[S1]...`；没有可靠依据则转人工。
- 审计记录 tool_calls、arguments、result、latency、trace_id，可回放。

---

## 8. 测试

```bash
pytest -q
# 24 passed
```

覆盖：归一化/offset、边界与白名单、三站点规则、全角/连字符变体、切块稳定性、BM25/混合检索、无答案守卫、Agent 注入与决策纠正、无依据转人工、API 校验、SQLite 审计、Qdrant local 模式、docx/xlsx 解析。

CI：`.github/workflows/ci.yml` 在 Python 3.11/3.12 上安装核心依赖并跑测试（不下载 HuggingFace 模型）。

---

## 9. 已知限制与下一步

- **规则覆盖是有限集**：留出集仍有 2 个保守误拦截；语义改写、图片内文字、视频口播尚未覆盖。
- **评测集为自建**：82 条回归与 23 条留出集来自团队口径，不等于平台官方基准；上线后应持续抽样人工复核。
- **PDF 只做文本层**：扫描件需要 OCR；表格结构未做单元格级还原。
- **LLM 复核未纳入最终报告**：报告默认关闭 LLM 复核，以衡量规则引擎本身；开启后需记录模型版本、成本与漂移。
- **Qdrant embedded 单进程**：本地模式下 API 与 Streamlit 不要同时直连同一个 Qdrant 路径；生产用 server 模式。
- **下一步**：图片 OCR 预审、规则灰度与回滚、人工复核工作台、按 ASIN 的历史趋势看板、LLM 评审的 gold set。

---

## 10. 设计参考

本项目为独立实现，设计与以下公开资料/开源实践对照：

- [Qdrant Hybrid Search](https://qdrant.tech/documentation/concepts/hybrid-queries/)：向量 + 稀疏检索融合思路
- [BGE-M3](https://huggingface.co/BAAI/bge-m3)：多语言 embedding
- [FastAPI](https://fastapi.tiangolo.com/) / [Streamlit](https://streamlit.io/)
- [Amazon Seller Central 政策](https://sellercentral.amazon.com/)（业务口径，非代码）
- RAGFlow / LlamaIndex 的“标题分层 + 来源元数据”切块实践（仅借鉴思路，未复制代码）

---

## 11. 仓库地址

已上传：**https://github.com/jack15732/amz-ops-agent**

CI 在 Python 3.11 / 3.12 上运行；`reports/eval_report.*` 与 `reports/retrieval_comparison.*` 随仓库提交，便于面试/复盘时直接展示。`.env` 未提交（已在 `.gitignore` 中）。

---

## License

MIT
