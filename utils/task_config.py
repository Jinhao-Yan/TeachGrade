from __future__ import annotations

import copy
from typing import Any


def validate_task_config(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Task configuration must be a JSON object")
    result = copy.deepcopy(value)
    for field in ("q_id", "course", "question_type", "guideline"):
        types = (str, int) if field == "q_id" else (str,)
        if not isinstance(result.get(field), types) or isinstance(result[field], bool):
            raise ValueError(f"Task configuration requires {field}")
        result[field] = str(result[field]).strip()
        if not result[field]:
            raise ValueError(f"Task configuration requires a nonempty {field}")
    errors = result.get("errors")
    if not isinstance(errors, list) or not errors:
        raise ValueError("Task configuration requires a nonempty errors list")
    names = set()
    for error in errors:
        if not isinstance(error, dict) or not isinstance(error.get("name"), str):
            raise ValueError("Each error category requires a name")
        error["name"] = error["name"].strip()
        if not error["name"] or error["name"] in names:
            raise ValueError("Error category names must be nonempty and unique")
        if not isinstance(error.get("description", ""), str):
            raise ValueError("Error category descriptions must be strings")
        names.add(error["name"])
    return result
