# 评测数据集说明

这些文件是项目自建的“可复现回归集”，不是亚马逊官方基准。目标是让每次改规则、改切块、改融合参数后，都能量化是否退化。

## 文件

| 文件 | 条数 | 用途 |
| --- | --- | --- |
| `compliance_cases.jsonl` | 82 | 规则回归集：三站点高频违规/合法/模糊样本，含全角、连字符、白名单等边界 |
| `compliance_holdout.jsonl` | 23 | 留出集：语义改写与未见变体，用于暴露词表规则盲区；不并入回归集 |
| `retrieval_cases.jsonl` | 32 | 检索集：问题 → 正确 doc_id / 标题关键词；含 2 条无答案问题 |
| `agent_cases.jsonl` | 18 | Agent 轨迹集：期望决策、必需工具、是否转人工、是否应有引用 |
| `web_cases.jsonl` | 10 | Tavily 联网集：问题 → 期望官方域名；衡量有结果率/官方域名命中率 |

## 字段

### 合规

```json
{
  "id": "us-block-001",
  "site": "US",
  "text": "Our yoga mat is the best seller...",
  "expected_decision": "blocked",
  "expected_rule_ids": ["US-ABS-001", "US-ABS-002"],
  "tags": ["absolute"],
  "note": "自建标注样本"
}
```

- `expected_decision`：`blocked` / `review` / `passed`。
- `expected_rule_ids`：至少应命中的规则；用于规则召回率，不要求“只命中这些”。
- 标注口径：高危违规 → blocked；需要证据/上下文判断 → review；合法边界 → passed。

### 检索

```json
{
  "id": "ret-001",
  "query": "美国站标题里能不能写 Best Seller？",
  "site": "US",
  "expected_doc_ids": ["amazon_us_listing_policy"],
  "expected_heading_keywords": ["绝对化"],
  "answerable": true
}
```

- 相关性标签是文档级，不是段落级；Recall@K = 至少命中一个正确文档。
- `answerable=false` 用于检验“无答案识别”，期望 `sufficient=false`。

### Agent

```json
{
  "id": "ag-us-001",
  "site": "US",
  "text": "Our yoga mat is the best seller...",
  "question": "请预审并给出修改建议。",
  "expected_decision": "blocked",
  "required_tools": ["check_compliance", "search_knowledge"],
  "should_handoff": false,
  "expect_citation": true
}
```

- `required_tools` 是必须出现的工具调用（不要求顺序）。
- `should_handoff` 单独衡量是否转人工；`review` 用例不强制调用 `request_human_review`，因为系统在 `_finalize` 会自动转人工。
- 任务成功 = 决策正确 + 转人工正确 + 工具召回 100% + 引用满足 + 护栏通过。

### 联网

```json
{
  "id": "web-001",
  "query": "FTC 100% guaranteed advertising substantiation official",
  "expected_domains": ["ftc.gov"]
}
```

- 只验证“能否搜到 + 是否命中期望官方域名”，不判断网页内容是否支持某个具体结论。
- 需要 `TAVILY_API_KEY`；无 Key 时该 suite 不参与评测。

## 指标

- 合规：三分类准确率、blocked P/R/F1、规则命中率、高危漏检率、通过样本误报率。
- 检索：Recall@K、Precision@K、MRR@K、nDCG@K、无答案识别率、Top-1 标题命中。
- Agent：任务成功率、决策准确率、转人工判断准确率、工具召回、引用覆盖率、护栏通过率、平均步数/延迟。
- 联网：有结果率、官方域名命中率、期望域名命中率、平均结果数/延迟。

## 使用与限制

```bash
amz-agent eval --suite all --embedding-provider huggingface --embedding-model moka-ai/m3e-base
amz-agent eval --suite holdout
amz-agent eval --suite web        # 需要 TAVILY_API_KEY
```

- 82 条回归集与词库同源，指标用于防退化，不应对外宣称“平台合规准确率”。
- 23 条留出集首轮 43.48%，补充变体后 91.30%，仍有 2 个保守误拦截；它只用于发现盲区。
- 检索集只有 32 条，K=6；扩大标注量后应重新报告置信区间。
- Agent 集默认跑 rule-based 模式，保证无 Key 可复现；LLM Agent 需单独跑并记录模型版本。
- 联网集只有 10 条，且搜索结果会随时间变化；它衡量“检索可达性”，不是内容正确性。
