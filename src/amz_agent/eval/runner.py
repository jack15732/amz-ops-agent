from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..agent import AgentService, RuleBasedAgent
from ..config import PROJECT_ROOT
from urllib.parse import urlparse
from ..models import AgentRequest, Decision, RetrievedChunk, Site
from ..rag.bm25 import tokenize
from ..service import AppService
from .metrics import agent_summary, classification_metrics, retrieval_metrics

EVAL_DIR = PROJECT_ROOT / "data" / "eval"


def load_cases(name: str) -> list[dict]:
    path = EVAL_DIR / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"eval cases not found: {path}")
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            cases.append(json.loads(line))
    return cases


def _compliance(service: AppService, cases: list[dict], llm_review: bool) -> dict:
    rows = []
    for case in cases:
        result = service.engine.check(
            case["text"], Site(case["site"]), use_llm_review=llm_review
        )
        expected_rules = set(case.get("expected_rule_ids", []))
        matched_rules = {match.rule_id for match in result.matches}
        rows.append(
            {
                "id": case["id"],
                "site": case["site"],
                "expected": case["expected_decision"],
                "predicted": result.decision.value,
                "decision_ok": result.decision.value == case["expected_decision"],
                "expected_rules": sorted(expected_rules),
                "matched_rules": sorted(matched_rules),
                "rule_hit": expected_rules.issubset(matched_rules) if expected_rules else True,
                "text": case["text"][:160],
            }
        )
    overall = classification_metrics(
        [row["expected"] for row in rows], [row["predicted"] for row in rows]
    )
    per_site = {}
    for site in sorted({row["site"] for row in rows}):
        subset = [row for row in rows if row["site"] == site]
        per_site[site] = {
            **classification_metrics(
                [row["expected"] for row in subset], [row["predicted"] for row in subset]
            ),
            "cases": len(subset),
            "rule_hit_rate": round(
                sum(row["rule_hit"] for row in subset) / len(subset), 4
            ),
        }
    high_risk_miss = [
        row for row in rows if row["expected"] == "blocked" and row["predicted"] != "blocked"
    ]
    false_positive = [
        row for row in rows if row["expected"] == "passed" and row["predicted"] != "passed"
    ]
    return {
        "cases": len(rows),
        "metrics": overall,
        "rule_hit_rate": round(sum(row["rule_hit"] for row in rows) / len(rows), 4),
        "high_risk_miss": len(high_risk_miss),
        "high_risk_miss_rate": round(len(high_risk_miss) / max(1, sum(r["expected"] == "blocked" for r in rows)), 4),
        "false_positive": len(false_positive),
        "false_positive_rate": round(len(false_positive) / max(1, sum(r["expected"] == "passed" for r in rows)), 4),
        "per_site": per_site,
        "failures": [row for row in rows if not row["decision_ok"]][:12],
        "high_risk_miss_examples": high_risk_miss[:8],
        "false_positive_examples": false_positive[:8],
    }


def _retrieval_rankings(
    service: AppService, cases: list[dict], k: int, mode: str, rerank: bool | None = None
) -> tuple[dict[str, list[str]], dict[str, bool], dict[str, str]]:
    rankings: dict[str, list[str]] = {}
    sufficient: dict[str, bool] = {}
    top_texts: dict[str, str] = {}
    retriever = service.retriever
    for case in cases:
        query = case["query"]
        site = Site(case["site"]) if case.get("site") else None
        if mode == "hybrid":
            results, _ = retriever.search(
                query, site=site, top_k=k, use_rerank=rerank
            )
            sufficient[case["id"]] = retriever.is_sufficient(results, query)
        elif mode == "vector":
            rewritten, _ = (
                retriever.expander.expand(query)
                if retriever.expander is not None
                else (query, [])
            )
            vector = service.embedder.embed([rewritten])[0]
            pairs = service.store.search(
                vector, retriever.settings.candidate_k, site=site.value if site else None
            )
            results = [
                RetrievedChunk(chunk=chunk, vector_score=score)
                for chunk, score in pairs[:k]
            ]
            sufficient[case["id"]] = bool(results and results[0].vector_score >= retriever.settings.min_vector_score)
        elif mode == "bm25":
            rewritten, _ = (
                retriever.expander.expand(query)
                if retriever.expander is not None
                else (query, [])
            )
            pairs = retriever.bm25.search(
                tokenize(rewritten),
                retriever.settings.candidate_k,
                site=site.value if site else None,
            )
            results = [
                RetrievedChunk(chunk=chunk, bm25_score=score, matched_terms=matched)
                for chunk, score, matched in pairs[:k]
            ]
            sufficient[case["id"]] = bool(results and results[0].bm25_score >= retriever.settings.min_bm25_score)
        else:
            raise ValueError(f"unknown retrieval mode: {mode}")

        seen: list[str] = []
        for item in results:
            if item.chunk.doc_id not in seen:
                seen.append(item.chunk.doc_id)
        rankings[case["id"]] = seen
        if results:
            top = results[0].chunk
            top_texts[case["id"]] = f"{' > '.join(top.heading_path)} {top.text[:400]}"
    return rankings, sufficient, top_texts


def _retrieval(
    service: AppService,
    cases: list[dict],
    k: int,
    mode: str,
    rerank: bool | None = None,
) -> dict:
    rankings, sufficient, top_texts = _retrieval_rankings(service, cases, k, mode, rerank)
    # Heading hit needs the top chunk text; encode it as a doc id ranking for the
    # metric, then evaluate doc-level recall separately.
    enriched = []
    for case in cases:
        copy = dict(case)
        if top_texts.get(case["id"]):
            copy["expected_heading_keywords"] = case.get("expected_heading_keywords", [])
        enriched.append(copy)
    metrics = retrieval_metrics(enriched, rankings, k, sufficient, top_texts)
    for case in cases:
        ranking = rankings.get(case["id"], [])
        expected = case.get("expected_doc_ids", [])
        metrics.setdefault("details", {})[case["id"]] = {
            "query": case["query"],
            "ranked": ranking,
            "expected": expected,
            "hit": bool(set(expected) & set(ranking[:k])) if expected else None,
            "sufficient": sufficient.get(case["id"]),
        }
    failures = [
        detail
        for detail in metrics.get("details", {}).values()
        if detail["hit"] is False
    ][:12]
    return {
        "mode": mode,
        "k": k,
        "cases": len(cases),
        "metrics": metrics,
        "failures": failures,
    }


def _web(service: AppService, cases: list[dict], max_results: int = 5) -> dict:
    if not service.web.available:
        return {"available": False, "cases": len(cases), "message": "Tavily is not configured"}
    rows = []
    for case in cases:
        response = service.web.search(case["query"], max_results=max_results)
        hosts = []
        for result in response.results:
            host = (urlparse(result.url).hostname or "").lower()
            if host:
                hosts.append(host)
        expected = [domain.lower() for domain in case.get("expected_domains", [])]
        domain_hit = any(
            host == domain or host.endswith("." + domain)
            for host in hosts
            for domain in expected
        )
        rows.append(
            {
                "id": case["id"],
                "query": case["query"],
                "results": len(response.results),
                "sufficient": response.sufficient,
                "official": any(result.official for result in response.results),
                "domain_hit": domain_hit,
                "hosts": hosts,
                "latency_ms": response.latency_ms,
                "error": response.error,
            }
        )
    total = len(rows) or 1
    return {
        "available": True,
        "cases": len(rows),
        "metrics": {
            "sufficient_rate": round(sum(row["sufficient"] for row in rows) / total, 4),
            "official_rate": round(sum(row["official"] for row in rows) / total, 4),
            "expected_domain_hit_rate": round(sum(row["domain_hit"] for row in rows) / total, 4),
            "avg_results": round(sum(row["results"] for row in rows) / total, 2),
            "avg_latency_ms": round(sum(row["latency_ms"] for row in rows) / total, 2),
        },
        "failures": [row for row in rows if not row["domain_hit"]][:10],
        "rows": rows,
    }


def _agent(
    service: AppService, cases: list[dict], rule_based: bool, llm_review: bool = False
) -> dict:
    runner = (
        RuleBasedAgent(
            service.retriever,
            service.engine,
            web=service.web,
            use_llm_review=llm_review,
        )
        if rule_based
        else AgentService(
            service.retriever,
            service.engine,
            llm=service.llm,
            settings=service.settings.agent,
            web=service.web,
            use_llm_review=llm_review,
        )
    )
    mode = "rule_based" if rule_based else service.agent.mode
    rows = []
    for case in cases:
        request = AgentRequest(
            text=case["text"],
            site=Site(case["site"]),
            question=case.get("question", "请对这段 Listing 文案做上架前合规预审，并给出修改建议。"),
            trace_id=case["id"],
        )
        response = runner.run(request)
        called = {call.name for call in response.tool_calls}
        required = set(case.get("required_tools", ["check_compliance"]))
        tool_recall = len(required & called) / len(required) if required else 1.0
        decision_ok = response.decision.value == case["expected_decision"]
        handoff_ok = bool(response.handoff.required) == bool(case.get("should_handoff", False))
        has_citation = bool(response.citations)
        citation_ok = has_citation if case.get("expect_citation", True) else True
        guardrail_ok = response.guardrails.passed
        rows.append(
            {
                "id": case["id"],
                "site": case["site"],
                "expected": case["expected_decision"],
                "predicted": response.decision.value,
                "decision_ok": decision_ok,
                "handoff_ok": handoff_ok,
                "tool_recall": round(tool_recall, 4),
                "called_tools": sorted(called),
                "required_tools": sorted(required),
                "has_citation": has_citation,
                "citation_ok": citation_ok,
                "guardrail_ok": guardrail_ok,
                "steps": len(response.tool_calls),
                "latency_ms": response.latency_ms,
                "success": decision_ok and handoff_ok and tool_recall == 1.0 and citation_ok and guardrail_ok,
                "answer": response.answer[:600],
                "matches": sorted({m.rule_id for m in response.matches}),
            }
        )
    return {
        "mode": mode,
        "cases": len(rows),
        "summary": agent_summary(rows),
        "failures": [row for row in rows if not row["success"]][:12],
        "rows": rows,
    }


def evaluate(
    service: AppService,
    *,
    suite: str = "all",
    k: int = 6,
    retrieval_mode: str = "hybrid",
    rule_based: bool = True,
    llm_review: bool = False,
    rerank: bool | None = None,
) -> dict:
    started_at = datetime.now(timezone.utc).isoformat()
    report: dict = {
        "meta": {
            "generated_at": started_at,
            "embedding_provider": service.embedder.name,
            "embedding_model": service.embedder.model,
            "agent_mode": "rule_based" if rule_based else service.agent.mode,
            "qdrant_mode": service.settings.qdrant.mode,
            "chunks": service.store.count(),
            "rules": service.engine.rule_count,
            "rule_versions": service.ruleset.versions,
            "retrieval_mode": retrieval_mode,
            "rerank": service.settings.retrieval.rerank_enabled if rerank is None else rerank,
            "k": k,
        }
    }
    if suite in {"all", "compliance"}:
        report["compliance"] = _compliance(service, load_cases("compliance_cases"), llm_review)
    if suite in {"all", "holdout"}:
        report["compliance_holdout"] = _compliance(
            service, load_cases("compliance_holdout"), llm_review
        )
    if suite in {"all", "web"}:
        report["web"] = _web(service, load_cases("web_cases"))
    if suite in {"all", "retrieval"}:
        report["retrieval"] = _retrieval(
            service, load_cases("retrieval_cases"), k, retrieval_mode, rerank
        )
    if suite in {"all", "agent"}:
        report["agent"] = _agent(
            service, load_cases("agent_cases"), rule_based, llm_review=llm_review
        )
    return report


def render_markdown(report: dict) -> str:
    meta = report["meta"]
    lines = [
        "# AMZ Ops Agent 评测报告",
        "",
        f"- 生成时间：{meta['generated_at']}",
        f"- Embedding：`{meta['embedding_provider']}` / `{meta['embedding_model']}`",
        f"- Agent 模式：`{meta['agent_mode']}`",
        f"- 知识库 chunk 数：{meta['chunks']}；规则数：{meta['rules']}",
        f"- 规则版本：`{json.dumps(meta['rule_versions'], ensure_ascii=False)}`",
        f"- 检索模式：`{meta['retrieval_mode']}`；K={meta['k']}",
        "",
        "## 总览",
        "",
        "| 模块 | 指标 | 数值 |",
        "| --- | --- | --- |",
    ]
    if "compliance" in report:
        comp = report["compliance"]
        m = comp["metrics"]
        lines += [
            f"| 合规 | 三分类准确率 | {m['accuracy']:.2%} |",
            f"| 合规 | blocked F1 | {m['per_label'].get('blocked', {}).get('f1', 0):.2%} |",
            f"| 合规 | 规则命中率 | {comp['rule_hit_rate']:.2%} |",
            f"| 合规 | 高危漏检率 | {comp['high_risk_miss_rate']:.2%} |",
            f"| 合规 | 通过样本误报率 | {comp['false_positive_rate']:.2%} |",
        ]
    if "compliance_holdout" in report:
        hold = report["compliance_holdout"]
        lines += [
            f"| 留出集 | 三分类准确率 | {hold['metrics']['accuracy']:.2%} |",
            f"| 留出集 | 高危漏检率 | {hold['high_risk_miss_rate']:.2%} |",
            f"| 留出集 | 通过样本误报率 | {hold['false_positive_rate']:.2%} |",
        ]
    if "web" in report and report["web"].get("available"):
        wm = report["web"]["metrics"]
        lines += [
            f"| 联网 | 有结果率 | {wm['sufficient_rate']:.2%} |",
            f"| 联网 | 官方域名命中率 | {wm['official_rate']:.2%} |",
            f"| 联网 | 期望域名命中率 | {wm['expected_domain_hit_rate']:.2%} |",
        ]
    if "retrieval" in report:
        rm = report["retrieval"]["metrics"]
        lines += [
            f"| 检索 | Recall@{report['retrieval']['k']} | {rm['recall_at_k']:.2%} |",
            f"| 检索 | MRR@{report['retrieval']['k']} | {rm['mrr_at_k']:.4f} |",
            f"| 检索 | nDCG@{report['retrieval']['k']} | {rm['ndcg_at_k']:.4f} |",
            f"| 检索 | 无答案识别率 | {rm['no_answer_accuracy']:.2%} |",
        ]
    if "agent" in report:
        am = report["agent"]["summary"]
        lines += [
            f"| Agent | 任务成功率 | {am['task_success']:.2%} |",
            f"| Agent | 决策准确率 | {am['decision_accuracy']:.2%} |",
            f"| Agent | 工具召回 | {am['tool_recall']:.2%} |",
            f"| Agent | 引用覆盖率 | {am['citation_rate']:.2%} |",
            f"| Agent | 护栏通过率 | {am['guardrail_pass_rate']:.2%} |",
        ]

    if "compliance" in report:
        comp = report["compliance"]
        lines += [
            "",
            "## 合规识别",
            "",
            f"样本 {comp['cases']} 条；高危漏检 {comp['high_risk_miss']} 条；通过样本误报 {comp['false_positive']} 条。",
            "",
            "| 站点 | 样本 | 准确率 | blocked P | blocked R | blocked F1 | 规则命中率 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for site, data in comp["per_site"].items():
            blocked = data["per_label"].get("blocked", {})
            lines.append(
                f"| {site} | {data['cases']} | {data['accuracy']:.2%} | "
                f"{blocked.get('precision', 0):.2%} | {blocked.get('recall', 0):.2%} | "
                f"{blocked.get('f1', 0):.2%} | {data['rule_hit_rate']:.2%} |"
            )
        if comp["high_risk_miss_examples"]:
            lines += ["", "**高危漏检样例**", ""]
            for row in comp["high_risk_miss_examples"][:5]:
                lines.append(f"- `{row['id']}` expected={row['expected']} predicted={row['predicted']}：{row['text']}")
        if comp["false_positive_examples"]:
            lines += ["", "**通过样本误报样例**", ""]
            for row in comp["false_positive_examples"][:5]:
                lines.append(f"- `{row['id']}` predicted={row['predicted']}：{row['text']}")

    if "compliance_holdout" in report:
        hold = report["compliance_holdout"]
        lines += [
            "",
            "## 独立留出集（泛化检查）",
            "",
            f"样本 {hold['cases']} 条。首轮用于发现规则缺口，补充高频变体后复测："
            f"准确率 {hold['metrics']['accuracy']:.2%}，高危漏检 {hold['high_risk_miss']} 条，"
            f"通过样本误报 {hold['false_positive']} 条。",
            "",
            "| 站点 | 样本 | 准确率 | 规则命中率 |",
            "| --- | --- | --- | --- |",
        ]
        for site, data in hold["per_site"].items():
            lines.append(
                f"| {site} | {data['cases']} | {data['accuracy']:.2%} | {data['rule_hit_rate']:.2%} |"
            )
        misses = hold.get("high_risk_miss_examples", [])[:8]
        if misses:
            lines += ["", "**漏检样例（用于下一轮规则迭代）**", ""]
            for row in misses:
                lines.append(f"- `{row['id']}` expected={row['expected']} predicted={row['predicted']}：{row['text']}")
            lines += [
                "",
                "留出集只用于暴露规则盲区，不并入主回归集；迭代后仍保留 2 个保守误拦截，"
                "说明纯词表规则对语义改写仍有边界。",
            ]

    if "web" in report and report["web"].get("available"):
        web = report["web"]
        wm = web["metrics"]
        lines += [
            "",
            "## 联网检索（Tavily）",
            "",
            f"样本 {web['cases']} 条，max_results=5。",
            "",
            "| 指标 | 数值 |",
            "| --- | --- |",
            f"| 有结果率 | {wm['sufficient_rate']:.2%} |",
            f"| 官方域名命中率 | {wm['official_rate']:.2%} |",
            f"| 期望域名命中率 | {wm['expected_domain_hit_rate']:.2%} |",
            f"| 平均结果数 | {wm['avg_results']} |",
            f"| 平均延迟 | {wm['avg_latency_ms']} ms |",
        ]
        if web["failures"]:
            lines += ["", "**未命中期望域名**", ""]
            for row in web["failures"][:8]:
                lines.append(f"- `{row['query']}` → {row['hosts']}")

    if "retrieval" in report:
        ret = report["retrieval"]
        rm = ret["metrics"]
        lines += [
            "",
            "## 混合检索",
            "",
            f"模式 `{ret['mode']}`，K={ret['k']}，样本 {ret['cases']} 条。",
            "",
            "| 指标 | 数值 |",
            "| --- | --- |",
            f"| Recall@K | {rm['recall_at_k']:.2%} |",
            f"| Precision@K | {rm['precision_at_k']:.2%} |",
            f"| MRR@K | {rm['mrr_at_k']:.4f} |",
            f"| nDCG@K | {rm['ndcg_at_k']:.4f} |",
            f"| 无答案识别率 | {rm['no_answer_accuracy']:.2%} |",
            f"| Top-1 标题命中 | {rm['heading_hit_at_1']:.2%} |",
        ]
        if ret["failures"]:
            lines += ["", "**未命中样例**", ""]
            for row in ret["failures"][:8]:
                lines.append(f"- `{row['query']}` expected={row['expected']} ranked={row['ranked']}")

    if "agent" in report:
        agent = report["agent"]
        am = agent["summary"]
        lines += [
            "",
            "## Agent 评测",
            "",
            f"模式 `{agent['mode']}`，样本 {agent['cases']} 条。",
            "",
            "| 指标 | 数值 |",
            "| --- | --- |",
            f"| 任务成功率 | {am['task_success']:.2%} |",
            f"| 决策准确率 | {am['decision_accuracy']:.2%} |",
            f"| 转人工判断准确率 | {am['handoff_accuracy']:.2%} |",
            f"| 必需工具召回 | {am['tool_recall']:.2%} |",
            f"| 引用覆盖率 | {am['citation_rate']:.2%} |",
            f"| 护栏通过率 | {am['guardrail_pass_rate']:.2%} |",
            f"| 平均工具步数 | {am['avg_steps']} |",
            f"| 平均延迟 | {am['avg_latency_ms']} ms |",
        ]
        if agent["failures"]:
            lines += ["", "**失败样例**", ""]
            for row in agent["failures"][:8]:
                lines.append(
                    f"- `{row['id']}` expected={row['expected']} predicted={row['predicted']} "
                    f"tools={row['called_tools']} handoff_ok={row['handoff_ok']} "
                    f"guardrail={row['guardrail_ok']}：{row['answer'][:160]}"
                )

    lines += [
        "",
        "## 口径与局限",
        "",
        "- 合规集为本项目自建标注样本，覆盖美/欧/日站点的高频规则，不代表平台全部政策。",
        "- 合规指标在关闭 LLM 复核时衡量规则引擎本身；开启后需单独记录模型版本与成本。",
        "- 检索集以 doc_id 为相关性标签，Recall@K 衡量“至少命中一个正确文档”，不是段落级标注。",
        "- Agent 集为离线可复现的任务集；生产环境应继续抽样人工复核并监控漏检。",
        "",
    ]
    return "\n".join(line.rstrip() for line in lines)


def write_report(report: dict, report_dir: str | Path = "reports", stem: str = "eval_report") -> tuple[Path, Path]:
    directory = Path(report_dir)
    if not directory.is_absolute():
        directory = PROJECT_ROOT / directory
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"{stem}.json"
    md_path = directory / f"{stem}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path
