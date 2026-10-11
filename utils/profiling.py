from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter
from typing import Any, Iterator


PROFILE_EVENTS: list[dict[str, Any]] = []
CURRENT_SAMPLE_ID: ContextVar[str] = ContextVar("current_sample_id", default="")
CURRENT_NODE_NAME: ContextVar[str] = ContextVar("current_node_name", default="")
CURRENT_STEP_ID: ContextVar[int | None] = ContextVar("current_step_id", default=None)


def reset_profile_events() -> None:
    """清空当前样本之前的 profiling 事件。"""
    PROFILE_EVENTS.clear()


def current_context() -> dict[str, Any]:
    """返回当前节点上下文，供 LLM 调用日志绑定 sample/step。"""
    return {
        "sample_id": CURRENT_SAMPLE_ID.get(),
        "node_name": CURRENT_NODE_NAME.get(),
        "step_id": CURRENT_STEP_ID.get(),
    }


@contextmanager
def profile_scope(
    node_name: str,
    sample_id: str = "",
    step_id: int | None = None,
    event_type: str = "node",
) -> Iterator[None]:
    """记录节点/内部 Agent 耗时，并在终端输出开始与结束标记。"""
    parent = current_context()
    actual_sample = sample_id or str(parent.get("sample_id", ""))
    actual_step = step_id if step_id is not None else parent.get("step_id")
    sample_token = CURRENT_SAMPLE_ID.set(actual_sample)
    node_token = CURRENT_NODE_NAME.set(node_name)
    step_token = CURRENT_STEP_ID.set(actual_step)
    started = perf_counter()
    step_text = "" if actual_step is None else f" step_id={actual_step}"
    print(f"[PROFILE][START] type={event_type} node={node_name} sample_id={actual_sample}{step_text}", flush=True)
    try:
        yield
    finally:
        elapsed = round(perf_counter() - started, 4)
        event = {
            "event_type": event_type,
            "node_name": node_name,
            "sample_id": actual_sample,
            "step_id": actual_step,
            "elapsed_seconds": elapsed,
        }
        PROFILE_EVENTS.append(event)
        print(
            f"[PROFILE][END] type={event_type} node={node_name} sample_id={actual_sample}"
            f"{step_text} elapsed_seconds={elapsed}",
            flush=True,
        )
        CURRENT_STEP_ID.reset(step_token)
        CURRENT_NODE_NAME.reset(node_token)
        CURRENT_SAMPLE_ID.reset(sample_token)


def record_llm_call(diagnostic: dict[str, Any]) -> None:
    """把单次 LLM 调用诊断落入 profiling 流。"""
    context = current_context()
    event = {
        "event_type": "llm_call",
        "node_name": context["node_name"],
        "sample_id": context["sample_id"],
        "step_id": context["step_id"],
        "schema_name": diagnostic.get("schema_name", ""),
        "prompt_chars": diagnostic.get("total_prompt_chars", 0),
        "response_chars": diagnostic.get("response_chars", 0),
        "elapsed_seconds": diagnostic.get("elapsed_seconds", 0.0),
        "retry_count": diagnostic.get("retry_count", 0),
        "parse_retry_count": diagnostic.get("parse_retry_count", 0),
        "model_call_success": diagnostic.get("model_call_success", False),
        "fallback_used": diagnostic.get("fallback_used", False),
        "fallback_reason": diagnostic.get("fallback_reason", ""),
        "raw_error_message": diagnostic.get("raw_error_message", ""),
        "call_id": diagnostic.get("call_id", ""),
    }
    replaced = False
    if event["call_id"]:
        for index in range(len(PROFILE_EVENTS) - 1, -1, -1):
            prior = PROFILE_EVENTS[index]
            if prior.get("event_type") == "llm_call" and prior.get("call_id") == event["call_id"]:
                PROFILE_EVENTS[index] = event
                replaced = True
                break
    if not replaced:
        PROFILE_EVENTS.append(event)
    print(
        f"[PROFILE][LLM] schema={event['schema_name']} node={event['node_name']} "
        f"sample_id={event['sample_id']} step_id={event['step_id']} "
        f"prompt_chars={event['prompt_chars']} response_chars={event['response_chars']} "
        f"elapsed_seconds={event['elapsed_seconds']} retry_count={event['retry_count']} "
        f"parse_retry_count={event['parse_retry_count']} success={event['model_call_success']} "
        f"fallback={event['fallback_used']}",
        flush=True,
    )


def summarize_profile_events(total_runtime_seconds: float) -> dict[str, Any]:
    """聚合当前样本耗时，供 debug/report 快速识别瓶颈。"""
    events = list(PROFILE_EVENTS)
    llm_calls = [event for event in events if event.get("event_type") == "llm_call"]
    node_events = [event for event in events if event.get("event_type") in {"node", "agent", "retriever"}]
    breakdown: dict[str, dict[str, Any]] = defaultdict(lambda: {"call_count": 0, "elapsed_seconds": 0.0})
    for event in node_events:
        item = breakdown[str(event.get("node_name", ""))]
        item["call_count"] += 1
        item["elapsed_seconds"] = round(item["elapsed_seconds"] + float(event.get("elapsed_seconds", 0.0)), 4)
    slowest = sorted(llm_calls, key=lambda item: float(item.get("elapsed_seconds", 0.0)), reverse=True)[:10]
    return {
        "total_runtime_seconds": round(total_runtime_seconds, 4),
        "llm_call_count": len(llm_calls),
        "total_llm_time": round(sum(float(item.get("elapsed_seconds", 0.0)) for item in llm_calls), 4),
        "llm_retry_count": sum(int(item.get("retry_count", 0) or 0) for item in llm_calls),
        "parse_retry_count": sum(int(item.get("parse_retry_count", 0) or 0) for item in llm_calls),
        "node_time_breakdown": dict(breakdown),
        "slowest_10_calls": slowest,
        "profile_events": events,
    }
