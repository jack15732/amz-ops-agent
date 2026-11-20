from __future__ import annotations

import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from ..config import Settings, load_settings
from ..models import (
    AgentRequest,
    AgentResponse,
    ComplianceRequest,
    ComplianceResult,
    HealthResponse,
    PrecheckRequest,
    PrecheckResponse,
    Rule,
    SearchRequest,
    SearchResponse,
    Site,
    WebSearchRequest,
    WebSearchResponse,
)
from ..service import AppService
from ..rag.ingest import SUPPORTED


def create_app(settings: Settings | None = None, service: AppService | None = None) -> FastAPI:
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.service is None:
            app.state.service = AppService(app.state.settings)
        yield
        app.state.service.close()

    app = FastAPI(
        title="AMZ Ops Agent",
        version=settings.app.version,
        description="Amazon listing pre-review: rule engine + hybrid RAG + tool-calling agent",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.service = service
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def get_service() -> AppService:
        if app.state.service is None:
            raise HTTPException(status_code=503, detail="service is starting")
        return app.state.service

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/docs")

    @app.get("/health", response_model=HealthResponse)
    def health(svc: AppService = Depends(get_service)):
        return svc.health()

    @app.post("/v1/compliance/check", response_model=ComplianceResult)
    def check_compliance(request: ComplianceRequest, svc: AppService = Depends(get_service)):
        if not request.text.strip():
            raise HTTPException(status_code=422, detail="text must not be empty")
        return svc.check(request)

    @app.post("/v1/compliance/precheck", response_model=PrecheckResponse)
    def precheck(request: PrecheckRequest, svc: AppService = Depends(get_service)):
        if not request.full_text().strip():
            raise HTTPException(status_code=422, detail="at least one listing field is required")
        return svc.precheck(request)

    @app.post("/v1/knowledge/search", response_model=SearchResponse)
    def search(request: SearchRequest, svc: AppService = Depends(get_service)):
        if not request.query.strip():
            raise HTTPException(status_code=422, detail="query must not be empty")
        return svc.search(request)

    @app.post("/v1/web/search", response_model=WebSearchResponse)
    def web_search(request: WebSearchRequest, svc: AppService = Depends(get_service)):
        if not request.query.strip():
            raise HTTPException(status_code=422, detail="query must not be empty")
        if not svc.web.available:
            raise HTTPException(status_code=503, detail="Tavily is not configured")
        return svc.web_search(request)

    @app.post("/v1/agent/query", response_model=AgentResponse)
    def agent(request: AgentRequest, svc: AppService = Depends(get_service)):
        if not request.text.strip():
            raise HTTPException(status_code=422, detail="text must not be empty")
        return svc.run_agent(request)

    @app.post("/v1/knowledge/ingest")
    def ingest(
        files: list[UploadFile] = File(...),
        reset: bool = Query(False),
        svc: AppService = Depends(get_service),
    ):
        upload_dir = settings.data_path / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []
        for upload in files:
            suffix = Path(upload.filename or "").suffix.lower()
            if suffix not in SUPPORTED:
                raise HTTPException(status_code=415, detail=f"unsupported file: {upload.filename}")
            target_dir = upload_dir / uuid.uuid4().hex[:8]
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / Path(upload.filename or "file").name
            with target.open("wb") as handle:
                shutil.copyfileobj(upload.file, handle)
            saved.append(target)
        try:
            stats = svc.ingest(saved, reset=reset)
        finally:
            for upload in files:
                upload.file.close()
        return {"status": "ok", **stats, "uploaded": [str(p) for p in saved]}

    @app.get("/v1/rules", response_model=list[Rule])
    def rules(site: Site | None = None, svc: AppService = Depends(get_service)):
        if site is None:
            return svc.ruleset.rules
        return svc.engine.rules_for_site(site)

    @app.get("/v1/audit/{trace_id}")
    def audit(trace_id: str, svc: AppService = Depends(get_service)):
        record = svc.audit.get_trace(trace_id)
        if record is None:
            raise HTTPException(status_code=404, detail="trace not found")
        return record

    @app.get("/v1/audit")
    def recent_audit(
        limit: int = Query(20, ge=1, le=200), svc: AppService = Depends(get_service)
    ):
        return {"traces": svc.audit.recent_traces(limit)}

    @app.get("/v1/stats")
    def stats(svc: AppService = Depends(get_service)):
        return {
            "service": svc.health().model_dump(mode="json"),
            "audit": svc.audit.stats(),
            "rule_versions": svc.ruleset.versions,
        }

    return app


app = create_app()
