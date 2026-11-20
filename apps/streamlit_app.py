"""Internal ops UI for the listing pre-review agent.

Run:
  streamlit run apps/streamlit_app.py
The API is expected at AMZ_API_URL (default http://127.0.0.1:8000).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import streamlit as st

API_URL = os.environ.get("AMZ_API_URL", "http://127.0.0.1:8000").rstrip("/")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SITES = {"US": "美国站", "EU": "欧盟站", "JP": "日本站"}
DECISION_LABEL = {"blocked": "拦截", "review": "转人工复核", "passed": "通过"}

st.set_page_config(page_title="AMZ Ops Agent", page_icon="🛡️", layout="wide")
st.title("🛡️ AMZ Ops Agent")
st.caption("亚马逊 Listing 上架前合规预审 · 规则引擎 + 混合检索 + 工具型 Agent")


@st.cache_data(ttl=10)
def api_get(path: str):
    response = httpx.get(f"{API_URL}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict):
    response = httpx.post(f"{API_URL}{path}", json=payload, timeout=180)
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except json.JSONDecodeError:
            detail = response.text
        raise RuntimeError(f"{response.status_code}: {detail}")
    return response.json()


def show_error(exc: Exception) -> None:
    st.error(f"调用失败：{exc}")
    st.code("uvicorn amz_agent.api.app:app --port 8000", language="bash")


def decision_badge(decision: str) -> None:
    label = DECISION_LABEL.get(decision, decision)
    if decision == "blocked":
        st.error(f"结论：{label}（禁止上架）")
    elif decision == "review":
        st.warning(f"结论：{label}（暂不放行）")
    else:
        st.success(f"结论：{label}")


def render_matches(matches: list[dict]) -> None:
    if not matches:
        st.info("未命中合规词条。")
        return
    for match in matches:
        with st.expander(
            f"[{match['rule_id']}] {match['term']} · {match['severity']} · {match['category']}",
            expanded=match["severity"] == "high",
        ):
            st.write(match["snippet"])
            st.caption(f"位置：{match['start']}–{match['end']}")
            if match.get("suggestion"):
                st.markdown(f"**修改建议：** {match['suggestion']}")
            if match.get("citation"):
                st.caption(f"依据：{match['citation']}")


def render_citations(citations: list[dict]) -> None:
    if not citations:
        st.caption("无引用（按风险优先原则，无依据时不会直接放行）。")
        return
    st.markdown("**可追溯引用**")
    for index, citation in enumerate(citations, start=1):
        if citation.get("source_type") == "web":
            with st.expander(f"[S{index}] 🌐 {citation['title']}"):
                st.markdown(f"[{citation.get('url', '')}]({citation.get('url', '')})")
                st.write(citation.get("snippet", ""))
                st.caption(f"web · score={citation.get('score', 0)}")
            continue
        heading = " > ".join(citation.get("heading_path", []))
        with st.expander(f"[S{index}] {citation['title']} :: {heading}"):
            st.write(citation.get("snippet", ""))
            st.caption(f"chunk_id={citation['chunk_id']} · source={citation['source']}")


# --------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("连接")
    st.code(API_URL, language="text")
    try:
        health = api_get("/health")
        st.success("API 在线")
        st.json(
            {
                "embedding": f"{health['embedding_provider']} / {health['embedding_model']}",
                "llm": health["llm_available"],
                "llm_model": health.get("llm_model", ""),
                "web_search": health.get("web_search_available", False),
                "qdrant": health["qdrant_mode"],
                "chunks": health["chunks"],
                "rules": health["rules"],
            }
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"API 不可用：{exc}")
        st.code("amz-agent serve --port 8000", language="bash")
    st.divider()
    site = st.selectbox("站点", list(SITES), format_func=lambda key: f"{SITES[key]} ({key})")
    st.caption("风控原则：高危直接拦截；模糊内容转人工；无可靠依据不放行。")

tabs = st.tabs(["Listing 预审", "Agent 问答", "知识库检索", "联网检索", "合规词库", "审计与评测"])

# ------------------------------------------------------------------ precheck
with tabs[0]:
    st.subheader("结构化 Listing 预审")
    st.caption("适合接入上架流程的固定预审环节，输出 decision / risk_score / 命中项 / 引用。")
    title = st.text_input("标题", placeholder="Cotton towel, soft and absorbent")
    bullets = st.text_area("五点描述（每行一条）", height=140)
    description = st.text_area("详情页 / A+ 文本", height=120)
    asin = st.text_input("ASIN（可选）", placeholder="B0XXXXXXXX")
    if st.button("提交预审", type="primary", key="precheck"):
        payload = {
            "asin": asin or None,
            "site": site,
            "title": title,
            "bullets": [line for line in bullets.splitlines() if line.strip()],
            "description": description,
        }
        try:
            result = api_post("/v1/compliance/precheck", payload)
        except Exception as exc:  # noqa: BLE001
            show_error(exc)
        else:
            decision_badge(result["decision"])
            col1, col2, col3 = st.columns(3)
            col1.metric("风险分", result["risk_score"])
            col2.metric("命中项", len(result["matches"]))
            col3.metric("引用数", len(result["citations"]))
            st.write(result["summary"])
            if result["handoff"]["required"]:
                st.warning(f"已转人工：{result['handoff']['reason']}")
            render_matches(result["matches"])
            render_citations(result["citations"])
            st.caption(f"trace_id={result['trace_id']} · {result['latency_ms']} ms")
            with st.expander("原始响应"):
                st.json(result)

# --------------------------------------------------------------------- agent
with tabs[1]:
    st.subheader("Agent 预审问答")
    st.caption("Agent 必须先调用合规工具，再检索知识库；无法溯源或需要法律定性时转人工。")
    agent_text = st.text_area(
        "待审文案",
        height=180,
        value="Our yoga mat is the best seller and guaranteed to stay flat.",
        key="agent_text",
    )
    question = st.text_input(
        "问题",
        value="请对这段文案做上架前合规预审，并给出修改建议。",
        key="agent_question",
    )
    if st.button("运行 Agent", type="primary", key="agent"):
        try:
            result = api_post(
                "/v1/agent/query",
                {"text": agent_text, "site": site, "question": question},
            )
        except Exception as exc:  # noqa: BLE001
            show_error(exc)
        else:
            decision_badge(result["decision"])
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("风险分", result["risk_score"])
            col2.metric("工具步数", len(result["tool_calls"]))
            col3.metric("引用", len(result["citations"]))
            col4.metric("护栏", "通过" if result["guardrails"]["passed"] else "未通过")
            st.markdown(result["answer"])
            if result["handoff"]["required"]:
                st.warning(f"转人工：{result['handoff']['reason']}")
            render_citations(result["citations"])
            with st.expander("工具调用轨迹"):
                for call in result["tool_calls"]:
                    st.markdown(f"**{call['seq']}. `{call['name']}`** · {call['latency_ms']} ms")
                    st.json({"arguments": call["arguments"], "result": call["result"]})
            with st.expander("护栏详情"):
                st.json(result["guardrails"])
            st.caption(f"mode={result['agent_mode']} · trace_id={result['trace_id']}")

# -------------------------------------------------------------------- search
with tabs[2]:
    st.subheader("混合检索")
    st.caption("BGE-M3 向量 + BM25 关键词，RRF 融合；缩写会自动扩展（BSR/ACOS/GPSR/薬機法…）。")
    query = st.text_input("问题", value="美国站标题能不能写 Best Seller？")
    top_k = st.slider("Top K", 1, 10, 6)
    if st.button("检索", type="primary", key="search"):
        try:
            result = api_post(
                "/v1/knowledge/search",
                {"query": query, "site": site, "top_k": top_k},
            )
        except Exception as exc:  # noqa: BLE001
            show_error(exc)
        else:
            if result["sufficient"]:
                st.success("检索依据充分")
            else:
                st.warning("检索依据不足，Agent 应转人工")
            st.caption(f"rewritten_query: {result['rewritten_query']} · {result['latency_ms']} ms")
            for index, item in enumerate(result["results"], start=1):
                chunk = item["chunk"]
                heading = " > ".join(chunk["heading_path"])
                with st.expander(f"[S{index}] {chunk['title']} :: {heading}", expanded=index == 1):
                    st.write(chunk["text"])
                    st.caption(
                        f"score={item['score']:.4f} · vector={item['vector_score']:.3f} · "
                        f"bm25={item['bm25_score']:.2f} · doc={chunk['doc_id']}"
                    )

# ------------------------------------------------------------------ web search
with tabs[3]:
    st.subheader("Tavily 联网检索")
    st.caption("用于知识库不足时的实时政策/官方公告兜底；结果必须带 URL，网页内容仅作为证据。")
    web_query = st.text_input("联网问题", value="Amazon US listing best seller policy 2026")
    web_topic = st.selectbox("类型", ["general", "news"])
    web_days = st.number_input("最近天数（news 生效）", min_value=1, max_value=365, value=30)
    if st.button("联网检索", type="primary", key="web_search"):
        try:
            result = api_post(
                "/v1/web/search",
                {
                    "query": web_query,
                    "max_results": 5,
                    "topic": web_topic,
                    "days": int(web_days) if web_topic == "news" else None,
                },
            )
        except Exception as exc:  # noqa: BLE001
            show_error(exc)
        else:
            if result.get("error"):
                st.warning(result["error"])
            elif result.get("sufficient"):
                st.success("联网结果可作为补充依据")
            else:
                st.warning("联网结果不足，Agent 应转人工")
            for index, item in enumerate(result.get("results", []), start=1):
                with st.expander(f"[S{index}] {item['title']}", expanded=index == 1):
                    st.markdown(f"[{item['url']}]({item['url']})")
                    st.write(item.get("content", ""))
                    st.caption(f"score={item['score']} · {item.get('published_date', '')}")
            st.caption(f"{result.get('latency_ms', 0)} ms")

# --------------------------------------------------------------------- rules
with tabs[4]:
    st.subheader("站点合规词库")
    st.caption("YAML 维护，按站点灰度发布；支持词条、正则、白名单上下文与严重级别。")
    try:
        rules = api_get(f"/v1/rules?site={site}")
        st.metric("规则数", len(rules))
        st.dataframe(
            [
                {
                    "规则ID": rule["id"],
                    "类别": rule["category"],
                    "级别": rule["severity"],
                    "动作": rule["action"],
                    "版本": rule["version"],
                    "说明": rule["description"],
                }
                for rule in rules
            ],
            width="stretch",
        )
        with st.expander("查看词条与建议"):
            for rule in rules:
                st.markdown(f"**{rule['id']}** · {rule['severity']} · {rule['category']}")
                st.write("词条：" + "、".join(rule["terms"][:20]))
                if rule["patterns"]:
                    st.code("\n".join(rule["patterns"]), language="regex")
                st.caption(rule["citation"])
                st.write("建议：" + rule["suggestion"])
    except Exception as exc:  # noqa: BLE001
        show_error(exc)

# -------------------------------------------------------------------- audit
with tabs[5]:
    st.subheader("审计与评测")
    try:
        traces = api_get("/v1/audit?limit=20")["traces"]
        st.markdown("**最近预审记录**")
        st.dataframe(traces, width="stretch")
        trace_id = st.text_input("按 trace_id 查询")
        if trace_id:
            st.json(api_get(f"/v1/audit/{trace_id}"))
    except Exception as exc:  # noqa: BLE001
        show_error(exc)

    st.divider()
    report_path = PROJECT_ROOT / "reports" / "eval_report.md"
    st.markdown("**离线评测报告**")
    if report_path.exists():
        st.markdown(report_path.read_text(encoding="utf-8"))
    else:
        st.info("尚未生成报告。运行：`amz-agent eval --suite all --report-dir reports`")
