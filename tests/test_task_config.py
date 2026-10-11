from __future__ import annotations

import copy

import pytest

from utils.task_config import validate_task_config


def test_valid_task_config_is_copied_and_normalized(task_config):
    value = copy.deepcopy(task_config)
    value["q_id"] = 123
    result = validate_task_config(value)
    assert result["q_id"] == "123"
    assert value["q_id"] == 123
    result["errors"][0]["name"] = "Changed"
    assert value["errors"] == task_config["errors"]


@pytest.mark.parametrize("field", ["q_id", "course", "question_type", "guideline"])
def test_task_config_requires_nonempty_metadata_and_rubric(task_config, field):
    task_config[field] = "  "
    with pytest.raises(ValueError, match="nonempty"):
        validate_task_config(task_config)


@pytest.mark.parametrize("errors", [[], ["wrong"], [{"name": ""}], [{"name": "same"}, {"name": "same"}]])
def test_task_config_rejects_invalid_error_categories(task_config, errors):
    task_config["errors"] = errors
    with pytest.raises(ValueError):
        validate_task_config(task_config)
