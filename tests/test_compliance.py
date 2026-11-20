from __future__ import annotations

from amz_agent.compliance.matcher import normalize_with_map
from amz_agent.models import Decision, Site


def test_normalisation_preserves_offsets():
    text = "ＡＢＣ　100% 保证"
    normalised, mapping = normalize_with_map(text)
    assert "abc" in normalised
    assert "100%" in normalised
    assert len(normalised) == len(mapping)
    assert mapping[0] == 0


def test_boundary_does_not_match_inside_words(service):
    # "best" must not fire inside "best practices" thanks to allow_contexts,
    # and "classic" must not be treated as "class".
    result = service.engine.check("Best practices for storage.", Site.US, use_llm_review=False)
    assert result.decision is Decision.PASSED
    result = service.engine.check("A classic cotton shirt.", Site.US, use_llm_review=False)
    assert result.decision is Decision.PASSED


def test_allow_context_compatibility(service):
    result = service.engine.check("Compatible with iPhone 15 only.", Site.US, use_llm_review=False)
    assert result.decision is Decision.PASSED
    result = service.engine.check("Apple AirPods charging case.", Site.US, use_llm_review=False)
    assert result.decision is Decision.BLOCKED


def test_high_risk_is_blocked_and_spans_are_verbatim(service):
    text = "Our formula is the best seller and cures acne."
    result = service.engine.check(text, Site.US, use_llm_review=False)
    assert result.decision is Decision.BLOCKED
    assert result.matches
    for match in result.matches:
        assert text[match.start : match.end].lower() == match.term.lower()


def test_site_specific_rules(service):
    text = "最高の美白クリームです。"
    assert service.engine.check(text, Site.US, use_llm_review=False).decision is Decision.PASSED
    assert service.engine.check(text, Site.JP, use_llm_review=False).decision is Decision.BLOCKED


def test_fullwidth_and_hyphen_variants(service):
    for text in ["Best-Seller quality", "１００％ cotton", "ｂｅｓｔ ｓｅｌｌｅｒ"]:
        assert service.engine.check(text, Site.US, use_llm_review=False).decision is Decision.BLOCKED


def test_review_and_pass_decisions(service):
    review = service.engine.check("Dermatologist tested formula.", Site.US, use_llm_review=False)
    assert review.decision is Decision.REVIEW
    assert review.review_required
    passed = service.engine.check("Cotton T-shirt, machine washable.", Site.US, use_llm_review=False)
    assert passed.decision is Decision.PASSED
    assert passed.risk_score == 0
