from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Settings, load_settings
from .models import AgentRequest, ComplianceRequest, SearchRequest, Site
from .service import AppService


def _override(settings: Settings, args: argparse.Namespace) -> Settings:
    if getattr(args, "embedding_provider", None):
        settings.embedding.provider = args.embedding_provider
    if getattr(args, "embedding_model", None):
        settings.embedding.model = args.embedding_model
    if getattr(args, "qdrant_mode", None):
        settings.qdrant.mode = args.qdrant_mode
    return settings


def _service(args: argparse.Namespace) -> AppService:
    return AppService(_override(load_settings(getattr(args, "config", None)), args))


def _read_text(args: argparse.Namespace) -> str:
    if getattr(args, "file", None):
        return Path(args.file).read_text(encoding="utf-8")
    return args.text or ""


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from .api.app import create_app

    settings = _override(load_settings(args.config), args)
    app = create_app(settings)
    uvicorn.run(app, host=args.host, port=args.port, log_level=settings.app.log_level.lower())


def cmd_ingest(args: argparse.Namespace) -> None:
    service = _service(args)
    try:
        stats = service.ingest(args.paths, reset=args.reset)
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    finally:
        service.close()


def cmd_check(args: argparse.Namespace) -> None:
    service = _service(args)
    try:
        text = _read_text(args)
        result = service.check(ComplianceRequest(text=text, site=Site(args.site)))
        if args.json:
            print(result.model_dump_json(indent=2))
            return
        print(f"站点: {result.site.value}  规则版本: {result.rule_version}")
        print(f"结论: {result.decision.value}  风险分: {result.risk_score}")
        print(f"说明: {result.summary}")
        for match in result.matches:
            print(f"  - [{match.rule_id}] {match.term} @ {match.start}:{match.end}  {match.snippet}")
            if match.suggestion:
                print(f"    建议: {match.suggestion}")
    finally:
        service.close()


def cmd_search(args: argparse.Namespace) -> None:
    service = _service(args)
    try:
        response = service.search(
            SearchRequest(query=args.query, site=Site(args.site) if args.site else None, top_k=args.top_k)
        )
        print(f"sufficient={response.sufficient}  rewritten={response.rewritten_query}")
        for index, item in enumerate(response.results, start=1):
            heading = " > ".join(item.chunk.heading_path)
            print(
                f"[S{index}] score={item.score:.4f} bm25={item.bm25_score:.2f} "
                f"vec={item.vector_score:.3f} {item.chunk.doc_id} :: {heading}"
            )
            print(f"     {item.chunk.text[:180].replace(chr(10), ' ')}")
    finally:
        service.close()


def cmd_agent(args: argparse.Namespace) -> None:
    service = _service(args)
    try:
        response = service.run_agent(
            AgentRequest(
                text=_read_text(args),
                site=Site(args.site),
                question=args.question,
            )
        )
        print(f"mode={response.agent_mode} decision={response.decision.value} "
              f"risk={response.risk_score} handoff={response.handoff.required}")
        print(response.answer)
        if response.citations:
            print("\n引用：")
            for index, citation in enumerate(response.citations, start=1):
                print(f"  [S{index}] {citation.title} :: {' > '.join(citation.heading_path)}")
        if response.handoff.required:
            print(f"\n转人工：{response.handoff.reason}")
        print(f"\nguardrails={response.guardrails.model_dump(mode='json')}")
    finally:
        service.close()


def cmd_eval(args: argparse.Namespace) -> None:
    from .eval import evaluate
    from .eval.runner import write_report

    service = _service(args)
    try:
        report = evaluate(
            service,
            suite=args.suite,
            k=args.k,
            retrieval_mode=args.retrieval_mode,
            rule_based=not args.use_llm_agent,
            llm_review=args.llm_review,
        )
        json_path, md_path = write_report(report, args.report_dir, args.stem)
        print(f"report: {md_path}")
        print(f"raw:    {json_path}")
        if "compliance" in report:
            comp = report["compliance"]
            print(
                f"compliance accuracy={comp['metrics']['accuracy']:.3f} "
                f"high_risk_miss={comp['high_risk_miss']} false_positive={comp['false_positive']}"
            )
        if "retrieval" in report:
            metrics = report["retrieval"]["metrics"]
            print(
                f"retrieval recall@{args.k}={metrics['recall_at_k']:.3f} "
                f"mrr={metrics['mrr_at_k']:.3f} ndcg={metrics['ndcg_at_k']:.3f} "
                f"no_answer={metrics['no_answer_accuracy']:.3f}"
            )
        if "agent" in report:
            summary = report["agent"]["summary"]
            print(
                f"agent success={summary['task_success']:.3f} "
                f"decision={summary['decision_accuracy']:.3f} "
                f"guardrail={summary['guardrail_pass_rate']:.3f}"
            )
    finally:
        service.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="amz-agent",
        description="Amazon listing pre-review: rules + hybrid RAG + tool-calling agent",
    )
    parser.add_argument("--config", default=None, help="path to settings.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run FastAPI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--embedding-provider")
    serve.add_argument("--embedding-model")
    serve.add_argument("--qdrant-mode", choices=["local", "memory", "server"])
    serve.set_defaults(func=cmd_serve)

    ingest = sub.add_parser("ingest", help="ingest PDF/Word/Excel/Markdown into the knowledge base")
    ingest.add_argument("paths", nargs="+")
    ingest.add_argument("--reset", action="store_true")
    ingest.add_argument("--embedding-provider")
    ingest.add_argument("--embedding-model")
    ingest.add_argument("--qdrant-mode", choices=["local", "memory", "server"])
    ingest.set_defaults(func=cmd_ingest)

    check = sub.add_parser("check", help="run the compliance rule engine")
    group = check.add_mutually_exclusive_group(required=True)
    group.add_argument("--text")
    group.add_argument("--file")
    check.add_argument("--site", choices=["US", "EU", "JP"], default="US")
    check.add_argument("--json", action="store_true")
    check.add_argument("--embedding-provider")
    check.add_argument("--embedding-model")
    check.add_argument("--qdrant-mode", choices=["local", "memory", "server"])
    check.set_defaults(func=cmd_check)

    search = sub.add_parser("search", help="hybrid retrieval")
    search.add_argument("--query", required=True)
    search.add_argument("--site", choices=["US", "EU", "JP"])
    search.add_argument("--top-k", type=int, default=6)
    search.add_argument("--embedding-provider")
    search.add_argument("--embedding-model")
    search.add_argument("--qdrant-mode", choices=["local", "memory", "server"])
    search.set_defaults(func=cmd_search)

    agent = sub.add_parser("agent", help="run the tool-calling agent")
    agroup = agent.add_mutually_exclusive_group(required=True)
    agroup.add_argument("--text")
    agroup.add_argument("--file")
    agent.add_argument("--site", choices=["US", "EU", "JP"], default="US")
    agent.add_argument("--question", default="请对这段 Listing 文案做上架前合规预审，并给出修改建议。")
    agent.add_argument("--embedding-provider")
    agent.add_argument("--embedding-model")
    agent.add_argument("--qdrant-mode", choices=["local", "memory", "server"])
    agent.set_defaults(func=cmd_agent)

    ev = sub.add_parser("eval", help="run compliance / retrieval / agent evaluation")
    ev.add_argument("--suite", choices=["all", "compliance", "holdout", "retrieval", "agent"], default="all")
    ev.add_argument("--k", type=int, default=6)
    ev.add_argument("--retrieval-mode", choices=["hybrid", "vector", "bm25"], default="hybrid")
    ev.add_argument("--report-dir", default="reports")
    ev.add_argument("--stem", default="eval_report")
    ev.add_argument("--use-llm-agent", action="store_true")
    ev.add_argument("--llm-review", action="store_true")
    ev.add_argument("--embedding-provider")
    ev.add_argument("--embedding-model")
    ev.add_argument("--qdrant-mode", choices=["local", "memory", "server"])
    ev.set_defaults(func=cmd_eval)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    try:
        args.func(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
