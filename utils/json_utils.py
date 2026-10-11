from __future__ import annotations

import json
import re
from typing import Any


def _strip_markdown_fences(text: str) -> str:
    """移除文本外层的 markdown 代码块标记。"""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json|JSON)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _balanced_json_slice(text: str, start: int) -> str:
    """从对象或数组起始符开始截取一个尽量括号平衡的 JSON 片段。"""
    opening = text[start]
    closing = "}" if opening == "{" else "]"
    stack: list[str] = []
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if not stack:
                break
            expected = "}" if stack[-1] == "{" else "]"
            if char != expected:
                break
            stack.pop()
            if not stack and char == closing:
                return text[start : index + 1]

    return text[start:]


def repair_json_text(text: str) -> str:
    """对模型常见的不规范 JSON 输出做简单修复。"""
    repaired = _strip_markdown_fences(text)

    block = re.search(r"```(?:json|JSON)?\s*(.*?)```", repaired, flags=re.DOTALL)
    if block:
        repaired = block.group(1).strip()

    repaired = (
        repaired.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
        .replace("，", ",")
    )
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    return repaired.strip()


def safe_json_loads(text: str, fallback: dict[str, Any]) -> Any:
    """解析 JSON 文本；若直接解析和修复后解析都失败，则返回 fallback。"""
    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        return json.loads(repair_json_text(text))
    except Exception:
        return fallback


def extract_json_block(text: str) -> Any:
    """从 markdown 或普通文本中提取第一个 JSON 对象或数组。"""
    fallback: dict[str, Any] = {}

    fenced = re.search(r"```(?:json|JSON)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        parsed = safe_json_loads(fenced.group(1), fallback)
        if parsed != fallback:
            return parsed

    raw = _strip_markdown_fences(text)
    first_object = raw.find("{")
    first_array = raw.find("[")
    candidates = [index for index in [first_object, first_array] if index >= 0]
    if not candidates:
        return safe_json_loads(raw, fallback)

    start = min(candidates)
    raw_slice = _balanced_json_slice(raw, start)
    parsed = safe_json_loads(raw_slice, fallback)
    if parsed != fallback:
        return parsed

    repaired = repair_json_text(text)
    first_object = repaired.find("{")
    first_array = repaired.find("[")
    candidates = [index for index in [first_object, first_array] if index >= 0]
    if not candidates:
        return safe_json_loads(repaired, fallback)
    return safe_json_loads(_balanced_json_slice(repaired, min(candidates)), fallback)
