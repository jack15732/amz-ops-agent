from __future__ import annotations

import time
from typing import Protocol

from ..models import (
    ComplianceResult,
    Decision,
    LLMLabel,
    LLMReview,
    MatchEvidence,
    Rule,
    Severity,
    Site,
)
from .loader import RuleSet
from .matcher import RuleMatcher

_SEVERITY_SCORE = {Severity.HIGH: 90, Severity.MEDIUM: 60, Severity.LOW: 30}
_SEVERITY_RANK = {Severity.HIGH: 3, Severity.MEDIUM: 2, Severity.LOW: 1}

_CATEGORY_CN = {
    "absolute": "绝对化用语",
    "medical": "医疗/功效宣称",
    "infringement": "知识产权侵权",
    "safety": "产品安全/认证",
    "pricing": "价格与促销",
    "privacy": "数据与隐私",
    "environment": "环保宣称",
    "origin": "产地与来源",
    "advertising": "广告与促销",
    "labeling": "标签与标识",
    "other": "其他",
}


class Reviewer(Protocol):
    def review_compliance(
        self, text: str, site: Site, matches: list[MatchEvidence]
    ) -> LLMReview | None: ...


class ComplianceEngine:
    def __init__(
        self,
        ruleset: RuleSet,
        *,
        reviewer: Reviewer | None = None,
        llm_review_threshold: float = 0.80,
    ):
        self.ruleset = ruleset
        self.matcher = RuleMatcher(ruleset.rules)
        self.reviewer = reviewer
        self.llm_review_threshold = llm_review_threshold

    @property
    def rule_count(self) -> int:
        return len(self.ruleset.rules)

    def rules_for_site(self, site: Site) -> list[Rule]:
        return self.ruleset.for_site(site)

    def find(self, text: str, site: Site) -> list[MatchEvidence]:
        return self.matcher.find(text, site)

    def check(
        self,
        text: str,
        site: Site,
        *,
        trace_id: str = "",
        use_llm_review: bool = True,
    ) -> ComplianceResult:
        started = time.perf_counter()
        matches = self.matcher.find(text, site)
        decision = self._initial_decision(matches)
        risk_score = self._risk_score(matches)
        review: LLMReview | None = None

        # Ambiguous hits are reviewed by the LLM, but the model can never turn a
        # high/medium hit into a silent pass: it may only upgrade to blocked, or
        # downgrade low-severity hits with very high confidence (still archived).
        if (
            use_llm_review
            and decision is Decision.REVIEW
            and self.reviewer is not None
            and matches
        ):
            try:
                review = self.reviewer.review_compliance(text, site, matches)
            except Exception:  # noqa: BLE001 - review failure must never fail-open
                review = LLMReview(
                    label=LLMLabel.UNCERTAIN,
                    confidence=0.0,
                    reason="LLM 复核调用失败，按风险优先原则保留人工复核",
                )
            if review is not None:
                decision, risk_score = self._apply_review(
                    decision, risk_score, matches, review
                )

        latency_ms = (time.perf_counter() - started) * 1000
        return ComplianceResult(
            decision=decision,
            risk_score=risk_score,
            site=site,
            matches=matches,
            llm_review=review,
            review_required=decision is Decision.REVIEW,
            summary=self._summary(decision, matches, review),
            rule_version=self.ruleset.version_for(site),
            latency_ms=round(latency_ms, 2),
            trace_id=trace_id,
        )

    @staticmethod
    def _initial_decision(matches: list[MatchEvidence]) -> Decision:
        if not matches:
            return Decision.PASSED
        if any(m.action.value == "block" or m.severity is Severity.HIGH for m in matches):
            return Decision.BLOCKED
        return Decision.REVIEW

    @staticmethod
    def _risk_score(matches: list[MatchEvidence]) -> int:
        if not matches:
            return 0
        worst = max(_SEVERITY_SCORE[m.severity] for m in matches)
        unique_rules = len({m.rule_id for m in matches})
        return min(99, worst + min(9, (unique_rules - 1) * 3))

    def _apply_review(
        self,
        decision: Decision,
        risk_score: int,
        matches: list[MatchEvidence],
        review: LLMReview,
    ) -> tuple[Decision, int]:
        medium_or_high = any(m.severity is not Severity.LOW for m in matches)
        if review.label is LLMLabel.VIOLATION and review.confidence >= self.llm_review_threshold:
            return Decision.BLOCKED, max(risk_score, 90)
        if (
            review.label is LLMLabel.BENIGN
            and review.confidence >= 0.90
            and not medium_or_high
        ):
            # Low-risk and confidently benign: released, but the LLM tag stays in
            # the audit trail for later sampling.
            return Decision.PASSED, risk_score
        return decision, risk_score

    @staticmethod
    def _summary(
        decision: Decision, matches: list[MatchEvidence], review: LLMReview | None
    ) -> str:
        if not matches:
            return "未命中当前站点合规词库，可进入下一环节。"
        categories = sorted({_CATEGORY_CN.get(m.category, m.category) for m in matches})
        rule_ids = sorted({m.rule_id for m in matches})
        if decision is Decision.BLOCKED:
            text = f"命中 {len(matches)} 处高风险内容（{'、'.join(categories)}），已拦截。"
        elif decision is Decision.REVIEW:
            text = f"命中 {len(matches)} 处待复核内容（{'、'.join(categories)}），需人工/LLM 复核。"
        else:
            text = f"命中 {len(matches)} 处低风险提示，LLM 复核后放行并留档。"
        text += f" 规则：{', '.join(rule_ids[:6])}"
        if review and review.reason:
            text += f"；复核意见：{review.reason}"
        return text
