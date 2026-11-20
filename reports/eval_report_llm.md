# AMZ Ops Agent 评测报告

- 生成时间：2026-11-20T17:17:50.388679+00:00
- Embedding：`huggingface` / `moka-ai/m3e-base`
- Agent 模式：`llm_function_calling`
- 知识库 chunk 数：96；规则数：35
- 规则版本：`{"EU": "2026.11", "JP": "2026.11", "US": "2026.11"}`
- 检索模式：`hybrid`；K=6

## 总览

| 模块 | 指标 | 数值 |
| --- | --- | --- |
| 合规 | 三分类准确率 | 90.24% |
| 合规 | blocked F1 | 88.24% |
| 合规 | 规则命中率 | 100.00% |
| 合规 | 高危漏检率 | 0.00% |
| 合规 | 通过样本误报率 | 0.00% |
| 留出集 | 三分类准确率 | 78.26% |
| 留出集 | 高危漏检率 | 0.00% |
| 留出集 | 通过样本误报率 | 0.00% |
| 联网 | 有结果率 | 100.00% |
| 联网 | 官方域名命中率 | 70.00% |
| 联网 | 期望域名命中率 | 70.00% |
| 检索 | Recall@6 | 100.00% |
| 检索 | MRR@6 | 0.8833 |
| 检索 | nDCG@6 | 0.8984 |
| 检索 | 无答案识别率 | 100.00% |
| Agent | 任务成功率 | 83.33% |
| Agent | 决策准确率 | 83.33% |
| Agent | 工具召回 | 100.00% |
| Agent | 引用覆盖率 | 100.00% |
| Agent | 护栏通过率 | 100.00% |

## 合规识别

样本 82 条；高危漏检 0 条；通过样本误报 0 条。

| 站点 | 样本 | 准确率 | blocked P | blocked R | blocked F1 | 规则命中率 |
| --- | --- | --- | --- | --- | --- | --- |
| EU | 24 | 91.67% | 81.82% | 100.00% | 90.00% | 100.00% |
| JP | 26 | 88.46% | 75.00% | 100.00% | 85.71% | 100.00% |
| US | 32 | 90.62% | 80.00% | 100.00% | 88.89% | 100.00% |

## 独立留出集（泛化检查）

样本 23 条。首轮用于发现规则缺口，补充高频变体后复测：准确率 78.26%，高危漏检 0 条，通过样本误报 0 条。

| 站点 | 样本 | 准确率 | 规则命中率 |
| --- | --- | --- | --- |
| EU | 6 | 83.33% | 100.00% |
| JP | 7 | 85.71% | 100.00% |
| US | 10 | 70.00% | 100.00% |

## 联网检索（Tavily）

样本 10 条，max_results=5。

| 指标 | 数值 |
| --- | --- |
| 有结果率 | 100.00% |
| 官方域名命中率 | 70.00% |
| 期望域名命中率 | 70.00% |
| 平均结果数 | 5.0 |
| 平均延迟 | 397.32 ms |

**未命中期望域名**

- `FDA cosmetic claims cure acne official guidance` → ['arrsys.com', 'formul8cosmetics.com', 'www.afdo.org', 'ww3.lawschool.cornell.edu', 'cosmetic.chemlinked.com']
- `REACH regulation official ECHA SVHC` → ['blog.sourceintelligence.com', 'getenviropass.com', 'www.3eco.com', 'www.chemradar.com', 'getreadycompliance.eu']
- `Japan PSE mark electrical appliances METI official` → ['payapress.com', 'www.ul.com', 'www.jqa.jp', 'www.tuv.com', 'www.emc-directory.com']

## 混合检索

模式 `hybrid`，K=6，样本 32 条。

| 指标 | 数值 |
| --- | --- |
| Recall@K | 100.00% |
| Precision@K | 42.39% |
| MRR@K | 0.8833 |
| nDCG@K | 0.8984 |
| 无答案识别率 | 100.00% |
| Top-1 标题命中 | 96.67% |

## Agent 评测

模式 `llm_function_calling`，样本 18 条。

| 指标 | 数值 |
| --- | --- |
| 任务成功率 | 83.33% |
| 决策准确率 | 83.33% |
| 转人工判断准确率 | 83.33% |
| 必需工具召回 | 100.00% |
| 引用覆盖率 | 100.00% |
| 护栏通过率 | 100.00% |
| 平均工具步数 | 3.0 |
| 平均延迟 | 5960.58 ms |

**失败样例**

- `ag-us-003` expected=review predicted=blocked tools=['check_compliance', 'lookup_clause', 'search_knowledge'] handoff_ok=False guardrail=True：## 结论：**拦截（blocked）**，不转人工

规则引擎判定 `blocked`（risk_score 90），命中 2 处。按决策口径，`blocked` 明确拦截，**不调用** `request_human_review`。

### 一、命中规则与原文片段

| 规则 ID | 命中词 | 严重度 |
- `ag-eu-005` expected=review predicted=blocked tools=['check_compliance', 'search_knowledge'] handoff_ok=False guardrail=True：**结论：拦截（blocked），不可上架。** 规则引擎判定 `blocked`，风险分 90，命中规则 EU-ENV-001（2 处）。

## 一、规则引擎结果（原文一致传入）
| 命中词 | 规则 | 严重度 | 原文片段 |
|---|---|---|---|
| eco-friendly | EU-ENV-
- `ag-jp-003` expected=review predicted=blocked tools=['check_compliance', 'lookup_clause', 'search_knowledge'] handoff_ok=False guardrail=True：## 结论：**拦截（blocked）**，不可直接上架

规则引擎判定 `blocked`（risk_score 90），命中 **JP-MED-002**（医疗/功效宣称，severity: medium）。虽然该条款的默认 action 为 `review`，但规则引擎本次返回的决策是 `blocked`，按口径

## 口径与局限

- 合规集为本项目自建标注样本，覆盖美/欧/日站点的高频规则，不代表平台全部政策。
- 合规指标在关闭 LLM 复核时衡量规则引擎本身；开启后需单独记录模型版本与成本。
- 检索集以 doc_id 为相关性标签，Recall@K 衡量“至少命中一个正确文档”，不是段落级标注。
- Agent 集为离线可复现的任务集；生产环境应继续抽样人工复核并监控漏检。
