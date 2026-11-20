# AMZ Ops Agent 评测报告

- 生成时间：2026-11-20T17:13:12.318489+00:00
- Embedding：`huggingface` / `moka-ai/m3e-base`
- Agent 模式：`rule_based`
- 知识库 chunk 数：96；规则数：35
- 规则版本：`{"EU": "2026.11", "JP": "2026.11", "US": "2026.11"}`
- 检索模式：`hybrid`；K=6

## 总览

| 模块 | 指标 | 数值 |
| --- | --- | --- |
| 联网 | 有结果率 | 100.00% |
| 联网 | 官方域名命中率 | 70.00% |
| 联网 | 期望域名命中率 | 70.00% |

## 联网检索（Tavily）

样本 10 条，max_results=5。

| 指标 | 数值 |
| --- | --- |
| 有结果率 | 100.00% |
| 官方域名命中率 | 70.00% |
| 期望域名命中率 | 70.00% |
| 平均结果数 | 5.0 |
| 平均延迟 | 439.59 ms |

**未命中期望域名**

- `FDA cosmetic claims cure acne official guidance` → ['arrsys.com', 'formul8cosmetics.com', 'www.afdo.org', 'ww3.lawschool.cornell.edu', 'cosmetic.chemlinked.com']
- `REACH regulation official ECHA SVHC` → ['blog.sourceintelligence.com', 'getenviropass.com', 'www.3eco.com', 'www.chemradar.com', 'getreadycompliance.eu']
- `Japan PSE mark electrical appliances METI official` → ['payapress.com', 'www.ul.com', 'www.jqa.jp', 'www.tuv.com', 'www.emc-directory.com']

## 口径与局限

- 合规集为本项目自建标注样本，覆盖美/欧/日站点的高频规则，不代表平台全部政策。
- 合规指标在关闭 LLM 复核时衡量规则引擎本身；开启后需单独记录模型版本与成本。
- 检索集以 doc_id 为相关性标签，Recall@K 衡量“至少命中一个正确文档”，不是段落级标注。
- Agent 集为离线可复现的任务集；生产环境应继续抽样人工复核并监控漏检。
