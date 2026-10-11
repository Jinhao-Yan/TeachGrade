from __future__ import annotations

import copy

from scripts import task_adaptation as builder


def test_compiler_preserves_inputs_and_teacher_provenance(compiler_example, monkeypatch):
    example = compiler_example
    rows = {row["id"]: row for row in example["teachers"]}
    monkeypatch.setattr(builder, "dataset_rows", lambda _: rows)
    original = copy.deepcopy(example)
    skill = builder.compile_skill(example["source_skill"], example["specification"])
    rag = builder.compile_rag("fictional", example["source_rag"], example["specification"])
    assert example == original
    assert tuple(skill) == builder.SKILL_FIELDS
    assert all(len(skill[field]) > len(example["source_skill"][field]) for field in skill)
    assert len(rag["cases"]) == len(example["teachers"])
    for before, after in zip(example["source_rag"]["cases"], rag["cases"]):
        for field in ("case_id", "source_sample_id", "step_id"):
            assert before[field] == after[field]


def test_compiler_hides_absolute_teacher_scores(compiler_example, monkeypatch):
    rows = {row["id"]: row for row in compiler_example["teachers"]}
    monkeypatch.setattr(builder, "dataset_rows", lambda _: rows)
    compiled = builder.compile_rag("fictional", compiler_example["source_rag"], compiler_example["specification"])
    for case in compiled["cases"]:
        visible = builder._model_visible_payload(case)
        for hidden in ("教师评分", "来源局部预算", "来源教师绝对分"):
            assert hidden not in visible
        assert 0 <= case["局部完成比例"] <= 1
        assert 0 <= case["归一化检索档位"] <= 10
    assert compiled["judge_raw_teacher_score_visible"] is False


def test_task_specification_accepts_only_teacher_evidence(compiler_example):
    example = compiler_example
    ids = [row["id"] for row in example["teachers"]]
    validated = builder.validate_agent_output("fictional", example["specification"], ids, example["source_skill"])
    assert validated == example["specification"]
    invalid = copy.deepcopy(example["specification"])
    invalid["evidence_support"][0]["supporting_example_ids"].append("sample:evaluation-only:S0")
    import pytest
    with pytest.raises(RuntimeError, match="non-Knowledge20 evidence"):
        builder.validate_agent_output("fictional", invalid, ids, example["source_skill"])
