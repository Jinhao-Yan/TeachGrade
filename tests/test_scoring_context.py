from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import frameworks.skill_augmented_grading as grading


def _sample() -> dict:
    return {
        "id": "political-native-base-contract",
        "question": "QUESTION_MARKER：请结合材料说明政治原理。",
        "total": 20,
        "reference": "REFERENCE_MARKER：参考答案包含两个独立得分点。",
        "analysis": "ANALYSIS_MARKER：专家解析说明两个得分点的边界。",
        "manual_label": 987654,
        "steps": [
            {"response": "STUDENT_STEP_A_MARKER：第一个完整得分点。", "label": 0},
            {"response": "STUDENT_STEP_B_MARKER：第二个完整得分点。", "label": 0},
        ],
    }


def test_global_prejudge_and_final_judge_inherit_native_base_context(monkeypatch):
    sample = _sample()
    skill = {
        "评分原则": ["SKILL_MARKER：先按独立得分点判断，再处理语义等价。"],
        "部分分规则": [],
        "零分情况": [],
        "边界说明": [],
    }
    guideline = "GUIDELINE_MARKER：官方政治评分指南。"
    calls: list[dict] = []

    monkeypatch.setattr(grading, "_official_score_guideline", lambda: guideline)
    monkeypatch.setattr(grading, "_official_error_labels", lambda: ["步骤正确", "答非所问"])
    monkeypatch.setattr(grading, "_task_name_zh", lambda: "政治")
    monkeypatch.setattr(grading, "_task_meta", lambda: {"dataset": "Political ShortAns"})
    monkeypatch.setattr(grading, "_is_math_shortans", lambda: False)

    responses = iter(
        [
            {
                "初步整题分": 13,
                "整题行为摘要": "整题初步判断。",
                "步骤初判": [
                    {"step_id": 0, "初步步骤分": 4, "局部完成百分比": 80, "学生行为摘要": "第一点", "可能错误类型": []},
                    {"step_id": 1, "初步步骤分": 5, "局部完成百分比": 90, "学生行为摘要": "第二点", "可能错误类型": []},
                ],
            },
            {
                "pred_score": 13,
                "step_scores": [
                    {"step_id": 0, "score": 4, "errors": ["步骤正确"], "reason": "第一点得分"},
                    {"step_id": 1, "score": 5, "errors": ["步骤正确"], "reason": "第二点得分"},
                ],
                "overall_reason": "整题依据完整答案评为13分。",
                "non_repetition_check": "无重复给分。",
                "error_chain_check": "无错误链。",
            },
        ]
    )

    def fake_chat(system, user, **kwargs):
        calls.append({"system": system, "user": user, "kwargs": kwargs})
        return next(responses), {"api_calls": 1, "usage": {}}

    monkeypatch.setattr(grading, "chat_json_with_diagnostics", fake_chat)

    preliminary = grading._global_preliminary_judge_zh(sample, skill, strict_api=True)
    assert len(calls) == 1
    assert preliminary["prejudge_granularity"] == "one_global_call_per_sample"
    assert [item["preliminary_step_score"] for item in preliminary["step_preliminaries"]] == [4, 5]

    prejudge_prompt = calls[0]["user"]
    for marker in (
        guideline,
        "QUESTION_MARKER",
        "20",
        "REFERENCE_MARKER",
        "ANALYSIS_MARKER",
        "STUDENT_STEP_A_MARKER",
        "STUDENT_STEP_B_MARKER",
        "SKILL_MARKER",
        "答非所问",
    ):
        assert marker in prejudge_prompt
    assert "987654" not in prejudge_prompt

    calibration = [
        {
            "step_id": 0,
            "当前学生步骤": sample["steps"][0]["response"],
            "检索案例评分分布": {"4": 1},
            "检索到的评分边界": [{"案例ID": "RAG_MARKER", "为什么更高": "覆盖完整"}],
        },
        {
            "step_id": 1,
            "当前学生步骤": sample["steps"][1]["response"],
            "检索案例评分分布": {"5": 1},
            "检索到的评分边界": [],
        },
    ]
    prediction = grading._joint_final_judge_zh(
        sample,
        skill,
        calibration,
        strict_api=True,
        fusion_mode="native_base_additive",
    )
    assert len(calls) == 2
    assert prediction["pred_score"] == 13
    assert prediction["step_sum"] == 9
    assert prediction["pred_score_overridden"] is False

    final_prompt = calls[1]["user"]
    for marker in (
        guideline,
        "QUESTION_MARKER",
        "20",
        "REFERENCE_MARKER",
        "ANALYSIS_MARKER",
        "STUDENT_STEP_A_MARKER",
        "STUDENT_STEP_B_MARKER",
        "SKILL_MARKER",
        "RAG_MARKER",
        "答非所问",
    ):
        assert marker in final_prompt
    assert "987654" not in final_prompt
    assert "Skill 是主要评分流程" not in final_prompt
    assert "数学短答题补充规则" not in final_prompt

    audit = prediction["base_scoring_context_audit"]
    assert audit["inherits_native_base_inputs"] is True
    assert audit["official_score_guideline_present"] is True
    assert audit["question_analysis_present"] is True
    assert audit["official_score_guideline_sha256"] == hashlib.sha256(guideline.encode("utf-8")).hexdigest()
    assert audit["question_analysis_sha256"] == hashlib.sha256(sample["analysis"].encode("utf-8")).hexdigest()


def test_main_runner_uses_one_global_prejudge_not_per_step_prejudge():
    source = inspect.getsource(grading._run_joint_step_rag_zh)
    assert "_global_preliminary_judge_zh(" in source
    assert "_step_preliminary_judge_zh(" not in source
    assert '"prejudge_granularity": global_preliminary.get("prejudge_granularity")' in source


def test_custom_dataset_filename_uses_the_supplied_task_rubric(monkeypatch):
    sample = _sample()
    monkeypatch.setattr(grading, "ACTIVE_TASK_CONFIG", {})
    monkeypatch.setattr(grading, "ACTIVE_DATASET_META", {})
    grading.configure_task({
        "q_id": "my-task", "course": "MySubject", "question_type": "ShortAns",
        "guideline": "MY_CUSTOM_RUBRIC", "errors": [{"name": "Incomplete answer"}],
    })
    meta = grading._infer_dataset_meta(
        Path("my_own_answers.jsonl"),
        [sample],
    )
    assert meta["q_id"] == "my-task"
    assert meta["course"] == "MySubject"
    assert meta["question_type"] == "ShortAns"
    grading._set_active_dataset_context(Path("my_own_answers.jsonl"), [sample])
    assert grading._official_score_guideline() == "MY_CUSTOM_RUBRIC"
    assert grading._official_error_labels() == ["Incomplete answer", "步骤正确"]


def test_final_judge_payload_is_json_serializable(monkeypatch):
    sample = _sample()
    monkeypatch.setattr(grading, "_official_score_guideline", lambda: "官方指南")
    monkeypatch.setattr(grading, "_official_error_labels", lambda: ["步骤正确"])
    monkeypatch.setattr(grading, "_task_name_zh", lambda: "政治")
    monkeypatch.setattr(grading, "_task_meta", lambda: {"dataset": "Political ShortAns"})
    monkeypatch.setattr(grading, "_is_math_shortans", lambda: False)

    def fake_chat(system, user, **kwargs):
        return {
            "pred_score": 10,
            "step_scores": [
                {"step_id": 0, "score": 4, "errors": ["步骤正确"]},
                {"step_id": 1, "score": 4, "errors": ["步骤正确"]},
            ],
        }, {"api_calls": 1, "usage": {}}

    monkeypatch.setattr(grading, "chat_json_with_diagnostics", fake_chat)
    result = grading._joint_final_judge_zh(sample, {"评分原则": ["规则"]}, [], True)
    json.dumps(result["final_judge_input_snapshot"], ensure_ascii=False)
    assert result["pred_score"] == 10
    assert result["step_sum"] == 8
