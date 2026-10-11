from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError, OpenAI

from utils.json_utils import extract_json_block, repair_json_text, safe_json_loads
from utils.profiling import record_llm_call


LLM_CLIENT_LOGS: list[dict[str, Any]] = []
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ENV_PATH = PROJECT_ROOT / ".env"
_PROJECT_ENV_STATUS: dict[str, Any] | None = None


def load_project_env(force: bool = False) -> dict[str, Any]:
    """从当前项目根目录的 .env 加载模型配置，不在日志中暴露密钥值。"""
    global _PROJECT_ENV_STATUS
    if _PROJECT_ENV_STATUS is not None and not force:
        return dict(_PROJECT_ENV_STATUS)

    status: dict[str, Any] = {
        "env_path": str(PROJECT_ENV_PATH),
        "env_file_found": PROJECT_ENV_PATH.exists(),
        "loaded_keys": [],
        "env_load_error": "",
    }
    if PROJECT_ENV_PATH.exists():
        try:
            lines = PROJECT_ENV_PATH.read_text(encoding="utf-8").splitlines()
            for raw_line in lines:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, raw_value = line.split("=", 1)
                key = key.strip()
                if key not in {
                    "DEEPSEEK_API_KEY",
                    "OPENAI_MODEL",
                    "OPENAI_API_KEY",
                    "OPENAI_BASE_URL",
                    "LLM_TIMEOUT_SECONDS",
                    "LLM_MAX_RETRIES",
                    "LLM_RETRY_SLEEP_SECONDS",
                }:
                    continue
                value = raw_value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                    value = value[1:-1]
                if not value:
                    continue
                # 允许命令行临时指定 OPENAI_MODEL/OPENAI_BASE_URL/OPENAI_API_KEY
                # 来切换 Qwen/OpenAI-compatible 后端；未显式设置时才使用 .env。
                if not os.environ.get(key):
                    os.environ[key] = value
                status["loaded_keys"].append(key)
        except OSError as exc:
            status["env_load_error"] = str(exc)

    status["deepseek_api_key_configured"] = bool(os.getenv("DEEPSEEK_API_KEY"))
    status["openai_model_configured"] = bool(os.getenv("OPENAI_MODEL"))
    _PROJECT_ENV_STATUS = status
    return dict(status)


class LLMCallFailure(RuntimeError):
    """严格 API 模式下用于立即中止无效预测的结构化异常。"""

    def __init__(self, diagnostic: dict[str, Any]):
        self.diagnostic = diagnostic
        super().__init__(
            f"{diagnostic.get('schema_name', 'LLM')} failed: "
            f"{diagnostic.get('fallback_reason', 'unknown_error')} - "
            f"{diagnostic.get('raw_error_message', '')}"
        )


def _verbose_llm() -> bool:
    """判断是否开启详细 LLM 请求与响应日志。"""
    return os.getenv("VERBOSE_LLM", "0").strip().lower() in {"1", "true", "yes", "on"}


def _chat_completion_compatibility(model: str) -> dict[str, Any]:
    """Preserve the retired deepseek-chat alias's non-thinking request mode."""
    thinking_mode = os.getenv("DEEPSEEK_THINKING_MODE", "").strip().lower()
    if model in {"deepseek-v4-flash", "deepseek-v4-pro"} and thinking_mode == "disabled":
        return {
            "extra_body": {
                "thinking": {
                    "type": "disabled",
                }
            }
        }
    return {}


def _new_diagnostic(schema_name: str, system_prompt: str, user_prompt: str) -> dict[str, Any]:
    """构造会落入 report/debug 的模型调用状态记录。"""
    return {
        "schema_name": schema_name,
        "model_call_success": False,
        "fallback_used": False,
        "fallback_reason": "",
        "raw_error_message": "",
        "stage": "",
        "system_prompt_chars": len(system_prompt),
        "user_prompt_chars": len(user_prompt),
        "total_prompt_chars": len(system_prompt) + len(user_prompt),
        "response_chars": 0,
        "json_repaired": False,
        "elapsed_seconds": 0.0,
        "retry_count": 0,
        "parse_retry_count": 0,
        "call_id": f"{schema_name}:{time.perf_counter_ns()}",
        "json_mode": False,
        "finish_reason": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "request_sha256": "",
        "response_sha256": "",
    }


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _escape_unquoted_json_string_quotes(value: str) -> str:
    """Escape quote characters that are visibly embedded inside JSON strings."""
    repaired: list[str] = []
    in_string = False
    escaped = False
    for index, char in enumerate(value):
        if not in_string:
            repaired.append(char)
            if char == '"':
                in_string = True
            continue
        if escaped:
            repaired.append(char)
            escaped = False
            continue
        if char == "\\":
            repaired.append(char)
            escaped = True
            continue
        if char != '"':
            if char == "\n":
                repaired.append("\\n")
                continue
            if char == "\r":
                repaired.append("\\r")
                continue
            if char == "\t":
                repaired.append("\\t")
                continue
            if char in {"]", "}"}:
                next_index = index + 1
                while next_index < len(value) and value[next_index].isspace():
                    next_index += 1
                next_char = value[next_index] if next_index < len(value) else ""
                if not next_char or next_char in {",", "}", "]"}:
                    repaired.append('"')
                    in_string = False
            repaired.append(char)
            continue

        next_index = index + 1
        while next_index < len(value) and value[next_index].isspace():
            next_index += 1
        next_char = value[next_index] if next_index < len(value) else ""
        if not next_char or next_char in {":", ",", "}", "]"}:
            repaired.append(char)
            in_string = False
        else:
            repaired.append('\\"')
    return "".join(repaired)


def _persist_raw_call(
    diagnostic: dict[str, Any],
    raw_response: str,
) -> None:
    audit_dir = os.getenv("LLM_RAW_AUDIT_DIR", "").strip()
    if not audit_dir:
        return
    target_dir = Path(audit_dir).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_call_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(diagnostic["call_id"]))
    target = target_dir / f"{safe_call_id}.json"
    temporary = target.with_suffix(".json.tmp")
    payload = {
        **diagnostic,
        "raw_response": raw_response,
    }
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def _record_error(diagnostic: dict[str, Any]) -> None:
    """记录 LLM client 错误，供当前 graph node 合并到 state logs 中。"""
    print(
        f"[LLM][ERROR] schema={diagnostic.get('schema_name')} "
        f"type={diagnostic.get('fallback_reason')} stage={diagnostic.get('stage')} "
        f"prompt_chars={diagnostic.get('total_prompt_chars')} "
        f"error={diagnostic.get('raw_error_message')}",
        flush=True,
    )
    LLM_CLIENT_LOGS.append(
        {
            "agent": "LLMClient",
            **diagnostic,
        }
    )


def _fail(
    diagnostic: dict[str, Any],
    reason: str,
    stage: str,
    message: Exception | str,
    strict_api: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    diagnostic.update(
        {
            "model_call_success": False,
            "fallback_used": not strict_api,
            "fallback_reason": reason,
            "raw_error_message": str(message),
            "stage": stage,
        }
    )
    _record_error(diagnostic)
    record_llm_call(diagnostic)
    if strict_api:
        raise LLMCallFailure(diagnostic)
    return {}, diagnostic


def consume_llm_client_logs() -> list[dict[str, Any]]:
    """返回并清空累计的 LLM client 日志，供 node 级日志合并使用。"""
    logs = list(LLM_CLIENT_LOGS)
    LLM_CLIENT_LOGS.clear()
    return logs


def _client() -> OpenAI:
    """根据环境变量创建 OpenAI-compatible 客户端。"""
    load_project_env()
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not base_url and os.getenv("DEEPSEEK_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        base_url = "https://api.deepseek.com"
    if not api_key:
        raise RuntimeError("Neither OPENAI_API_KEY nor DEEPSEEK_API_KEY is set.")
    timeout_seconds = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds, max_retries=0)


def record_schema_validation_failure(
    diagnostic: dict[str, Any],
    schema_name: str,
    error: Exception | str,
    strict_api: bool,
) -> dict[str, Any]:
    """把节点级 JSON schema 校验失败转成显式错误，而不是“模型不可用”。"""
    if not diagnostic.get("model_call_success"):
        return diagnostic
    diagnostic = {**diagnostic}
    _fail(diagnostic, "schema_validation_failed", "schema_validation", error, strict_api)
    return diagnostic


def chat_json_with_diagnostics(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 2048,
    schema_name: str = "",
    strict_api: bool = False,
    json_mode: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """调用 API 并返回 JSON 与诊断，明确区分配置、调用、空响应和解析失败。"""
    load_project_env()
    started = time.perf_counter()
    diagnostic = _new_diagnostic(schema_name, system_prompt, user_prompt)
    diagnostic["json_mode"] = bool(json_mode)
    content = ""

    def fail(reason: str, stage: str, message: Exception | str) -> tuple[dict[str, Any], dict[str, Any]]:
        diagnostic["elapsed_seconds"] = round(time.perf_counter() - started, 4)
        diagnostic.update(
            {
                "model_call_success": False,
                "fallback_used": not strict_api,
                "fallback_reason": reason,
                "raw_error_message": str(message),
                "stage": stage,
            }
        )
        _persist_raw_call(diagnostic, content)
        return _fail(diagnostic, reason, stage, message, strict_api)

    if not (os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")):
        return fail(
            "key_error",
            "config",
            "Neither OPENAI_API_KEY nor DEEPSEEK_API_KEY is set.",
        )
    model = os.getenv("OPENAI_MODEL") or ("deepseek-chat" if os.getenv("DEEPSEEK_API_KEY") else "")
    if not model:
        return fail(
            "model_config_error",
            "config",
            "OPENAI_MODEL is not set and no DEEPSEEK_API_KEY default model is available.",
        )
    compatibility_kwargs = _chat_completion_compatibility(model)
    diagnostic["model"] = model
    diagnostic["thinking_mode"] = (
        "disabled" if compatibility_kwargs else "provider_default"
    )

    timeout_seconds = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
    max_retries = max(0, int(os.getenv("LLM_MAX_RETRIES", "1")))
    retry_sleep_seconds = max(0.0, float(os.getenv("LLM_RETRY_SLEEP_SECONDS", "1")))
    print(
        f"[PROFILE][LLM_START] schema={schema_name} prompt_chars={diagnostic['total_prompt_chars']} "
        f"timeout_seconds={timeout_seconds} max_retries={max_retries}",
        flush=True,
    )
    if _verbose_llm():
        print(f"[LLM][START] schema={schema_name} model={model}", flush=True)
        print(
            f"[LLM][PROMPT] schema={schema_name} system_chars={len(system_prompt)} user_chars={len(user_prompt)}",
            flush=True,
        )
    diagnostic["request_sha256"] = _sha256_text(
        json.dumps(
            {
                "model": model,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "json_mode": bool(json_mode),
                "thinking_mode": diagnostic["thinking_mode"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    for attempt in range(max_retries + 1):
        try:
            request_kwargs: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                **compatibility_kwargs,
            }
            if json_mode:
                request_kwargs["response_format"] = {"type": "json_object"}
            completion = _client().chat.completions.create(
                **request_kwargs,
            )
            choice = completion.choices[0]
            content = choice.message.content or ""
            diagnostic["response_chars"] = len(content)
            diagnostic["response_sha256"] = _sha256_text(content)
            diagnostic["finish_reason"] = str(getattr(choice, "finish_reason", "") or "")
            usage = getattr(completion, "usage", None)
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                diagnostic[field] = int(getattr(usage, field, 0) or 0)
            _persist_raw_call(diagnostic, content)
            if _verbose_llm():
                print(f"[LLM][RAW] schema={schema_name} chars={len(content)}", flush=True)
                preview = content.replace("\n", " ")[:800]
                print(f"[LLM][PREVIEW] schema={schema_name} {preview}", flush=True)
            break
        except AuthenticationError as exc:
            return fail("key_error", "api_call", exc)
        except (APITimeoutError, APIConnectionError) as exc:
            reason = "timeout" if isinstance(exc, APITimeoutError) else "connection_error"
            if attempt >= max_retries:
                return fail(reason, "api_call", exc)
            diagnostic["retry_count"] += 1
            sleep_seconds = retry_sleep_seconds * (2**attempt)
            print(
                f"[LLM][RETRY] schema={schema_name} reason={reason} "
                f"retry_count={diagnostic['retry_count']} sleep_seconds={sleep_seconds}",
                flush=True,
            )
            time.sleep(sleep_seconds)
        except APIStatusError as exc:
            retryable = exc.status_code in {408, 409, 429} or exc.status_code >= 500
            if not retryable or attempt >= max_retries:
                return fail("api_status_error", "api_call", exc)
            diagnostic["retry_count"] += 1
            sleep_seconds = retry_sleep_seconds * (2**attempt)
            print(
                f"[LLM][RETRY] schema={schema_name} reason=http_{exc.status_code} "
                f"retry_count={diagnostic['retry_count']} sleep_seconds={sleep_seconds}",
                flush=True,
            )
            time.sleep(sleep_seconds)
        except Exception as exc:
            return fail("api_error", "api_call", exc)
    if not content.strip():
        return fail("model_response_empty", "response", "Model response content is empty.")
    if diagnostic["finish_reason"] == "length":
        return fail(
            "response_truncated",
            "response",
            "Model response reached the configured token limit.",
        )

    try:
        parsed = extract_json_block(content)
        repaired_content = repair_json_text(content).strip()
        valid_empty_object = repaired_content == "{}"
        if isinstance(parsed, dict) and (parsed or valid_empty_object):
            if _verbose_llm():
                print(f"[LLM][OK] schema={schema_name} keys={list(parsed.keys())}", flush=True)
            diagnostic.update({"model_call_success": True, "stage": "complete", "elapsed_seconds": round(time.perf_counter() - started, 4)})
            _persist_raw_call(diagnostic, content)
            record_llm_call(diagnostic)
            return parsed, diagnostic
        if isinstance(parsed, list):
            if _verbose_llm():
                print(f"[LLM][OK] schema={schema_name} array_len={len(parsed)}", flush=True)
            diagnostic.update({"model_call_success": True, "stage": "complete", "elapsed_seconds": round(time.perf_counter() - started, 4)})
            _persist_raw_call(diagnostic, content)
            record_llm_call(diagnostic)
            return {"items": parsed}, diagnostic
        raise ValueError("Model output did not contain a JSON object or array.")
    except Exception:
        pass

    diagnostic["parse_retry_count"] += 1
    print(
        f"[LLM][PARSE_RETRY] schema={schema_name} retry_count={diagnostic['parse_retry_count']} sleep_seconds=0",
        flush=True,
    )
    repaired = repair_json_text(content)
    parsed = safe_json_loads(repaired, {})
    if not parsed:
        quote_repaired = _escape_unquoted_json_string_quotes(repaired)
        parsed = safe_json_loads(quote_repaired, {})
        if parsed:
            repaired = quote_repaired
    if isinstance(parsed, dict) and (parsed or repaired.strip() == "{}"):
        diagnostic.update({"model_call_success": True, "stage": "complete", "json_repaired": True, "elapsed_seconds": round(time.perf_counter() - started, 4)})
        _persist_raw_call(diagnostic, content)
        record_llm_call(diagnostic)
        return parsed, diagnostic
    if isinstance(parsed, list):
        diagnostic.update({"model_call_success": True, "stage": "complete", "json_repaired": True, "elapsed_seconds": round(time.perf_counter() - started, 4)})
        _persist_raw_call(diagnostic, content)
        record_llm_call(diagnostic)
        return {"items": parsed}, diagnostic

    return fail("json_parse_error", "json_parse", "Failed to parse or repair JSON.")


def chat_json(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 2048,
    schema_name: str = "",
    strict_api: bool = False,
) -> dict[str, Any]:
    """兼容旧节点的 JSON 调用接口；新核心节点应使用诊断版本。"""
    data, _ = chat_json_with_diagnostics(
        system_prompt,
        user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        schema_name=schema_name,
        strict_api=strict_api,
    )
    return data
