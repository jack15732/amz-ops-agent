from __future__ import annotations

from amz_agent.models import AgentRequest, ComplianceRequest, Site


def test_audit_records_compliance_and_tools(service):
    result = service.check(
        ComplianceRequest(text="100% cure for acne.", site=Site.US)
    )
    trace = service.audit.get_trace(result.trace_id)
    assert trace is not None
    assert trace["output"]["decision"] == "blocked"
    assert trace["compliance_flags"]

    response = service.run_agent(
        AgentRequest(text="Cotton towel.", site=Site.US, question="可以上架吗？")
    )
    trace = service.audit.get_trace(response.trace_id)
    assert trace is not None
    assert trace["endpoint"] == "agent"
    assert trace["tool_calls"]
    assert service.audit.stats()["traces"] >= 2
