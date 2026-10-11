from __future__ import annotations

import json
import pytest

import run_teachgrade as public
from frameworks import skill_augmented_grading as grading
from scripts import task_adaptation as builder

def test_prepare_freezes_exact_compiler_outputs_and_checks_cache(tmp_path, monkeypatch, compiler_example, task_config):
    teachers = compiler_example["teachers"]
    source_skill = compiler_example["source_skill"]
    source_rag = compiler_example["source_rag"]
    specification = compiler_example["specification"]
    dataset = tmp_path / "fictional_task.jsonl"
    dataset.write_text("".join(json.dumps(row) + "\n" for row in teachers), encoding="utf-8")
    output = tmp_path / "output"
    public.write_json(output / "split_manifest.json", {"teacher_ids": [row["id"] for row in teachers]})
    for name in ("ACTIVE_DATASET_META", "ACTIVE_TASK_CONFIG"):
        monkeypatch.setattr(grading, name, getattr(grading, name))
    monkeypatch.setenv("OPENAI_MODEL", "offline-test-model")

    def skill(knowledge, strict_api, output_dir):
        assert knowledge == teachers
        public.write_json(output_dir / "evaluation_skill_zh.json", source_skill)
        return source_skill

    def casebank(knowledge, output_dir):
        assert knowledge == teachers
        public.write_json(output_dir / "full_step_casebank_zh_evidence_reason.json", source_rag)
        return source_rag

    def adaptation(task, source, force):
        assert task == dataset.stem
        assert builder.knowledge20_rows(task) == teachers
        assert builder.SPEC_ROOT == output / "specifications"
        assert source["knowledge_ids"] == [row["id"] for row in teachers]
        assert not force
        return specification, {}

    monkeypatch.setattr(grading, "_load_or_build_evaluation_skill_zh", skill)
    monkeypatch.setattr(grading, "_load_or_build_full_step_casebank_zh_evidence_reason", casebank)
    monkeypatch.setattr(builder, "run_task_adaptation", adaptation)
    previous_provider = builder.dataset_rows
    with public.task_context(builder, teachers, teachers, output):
        expected_skill = builder.compile_skill(source_skill, specification)
        expected_rag = builder.compile_rag(dataset.stem, source_rag, specification)
    public.prepare_memories(dataset, teachers, teachers, output, task_config)
    assert builder.dataset_rows is previous_provider
    skill_path = output / "ks4/evaluation_skill_zh.json"
    rag_path = output / "ks4/full_step_casebank_zh_evidence_reason.json"
    assert public.read_json(skill_path) == expected_skill
    assert public.read_json(rag_path) == expected_rag
    manifest = public.read_json(output / "memory_manifest.json")
    assert manifest["case_count"] == len(source_rag["cases"])
    assert manifest["skill_sha256"] == public.path_sha(skill_path)
    assert manifest["casebank_sha256"] == public.path_sha(rag_path)

    def unexpected_call(*args, **kwargs):
        raise AssertionError("Frozen memories must not be rebuilt")

    monkeypatch.setattr(grading, "_load_or_build_evaluation_skill_zh", unexpected_call)
    public.prepare_memories(dataset, teachers, teachers, output, task_config)
    changed = {**task_config, "guideline": "A different task rubric."}
    with pytest.raises(ValueError, match="different split, model, or source"):
        public.prepare_memories(dataset, teachers, teachers, output, changed)
    public.write_json(rag_path, {"cases": []})
    with pytest.raises(ValueError, match="hashes changed"):
        public.prepare_memories(dataset, teachers, teachers, output, task_config)


def test_task_context_restores_bindings_after_failure(tmp_path):
    names = ("dataset_rows", "knowledge20_rows", "SPEC_ROOT", "RAW_ROOT")
    before = {name: getattr(builder, name) for name in names}
    with pytest.raises(RuntimeError, match="interrupted"):
        with public.task_context(builder, [], [], tmp_path):
            raise RuntimeError("interrupted")
    assert {name: getattr(builder, name) for name in names} == before


def test_teachgrade_preparation_and_scoring_without_benchmark_assets(tmp_path, monkeypatch, compiler_example, task_config):
    rows = [
        {**compiler_example["teachers"][index % 4], "id": f"fictional-answer-{index}"}
        for index in range(7)
    ]
    dataset = tmp_path / "my_custom_answers.jsonl"
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "output"
    manifest, evaluation, teachers = public.split_manifest(dataset, rows, 4, 42, None)
    public.write_json(output / "split_manifest.json", manifest)
    for name in ("ACTIVE_DATASET_META", "ACTIVE_TASK_CONFIG"):
        monkeypatch.setattr(grading, name, getattr(grading, name))
    monkeypatch.setenv("OPENAI_MODEL", "offline-test-model")

    def skill(knowledge, strict_api, output_dir):
        assert knowledge == teachers
        public.write_json(output_dir / "evaluation_skill_zh.json", compiler_example["source_skill"])
        return compiler_example["source_skill"]

    def adaptation(task, source, force):
        assert source["knowledge_ids"] == [row["id"] for row in teachers]
        return compiler_example["specification"], {}

    def prejudge(sample, skill, strict_api):
        return {"step_preliminaries": [{
            "step_id": index, "preliminary_step_score": 1,
            "raw_preliminary_step_score": 1, "local_completion_percent": 50,
            "normalized_retrieval_band": 5, "学生行为摘要": step["response"],
            "可能错误类型": [],
        } for index, step in enumerate(sample["steps"])], "prejudge_granularity": "one_global_call_per_sample"}

    def final(sample, skill, calibration, strict_api, fusion_mode):
        assert fusion_mode == "native_base_additive"
        assert grading._official_score_guideline() == task_config["guideline"]
        assert len(calibration) == len(sample["steps"])
        return {"pred_score": 2, "steps": [{"step_score": 2, "errors": []} for _ in sample["steps"]]}

    monkeypatch.setattr(grading, "_load_or_build_evaluation_skill_zh", skill)
    monkeypatch.setattr(builder, "run_task_adaptation", adaptation)
    public.prepare_memories(dataset, rows, teachers, output, task_config)
    monkeypatch.setattr(grading, "_global_preliminary_judge_zh", prejudge)
    monkeypatch.setattr(grading, "_joint_final_judge_zh", final)
    monkeypatch.setattr(grading, "_metrics", lambda records: {"CCS": 98.0, "QWK": 99.0})
    summary = grading._run_mode("teachgrade", limit=len(evaluation), strict_api=True,
                                knowledge_size=4, dataset_path=dataset,
                                eval_size=len(evaluation), output_root=output)
    predictions = public.read_json(output / "ks4/predictions_teachgrade.json")
    assert summary["completed_samples"] == len(evaluation)
    assert "native_baseline" not in summary
    assert "ccs_delta_vs_native" not in summary
    assert [row["sample_id"] for row in predictions] == [row["id"] for row in evaluation]
    assert all(row["pred_score"] == 2 for row in predictions)
    assert (output / "ks4/predictions_teachgrade.jsonl").exists()

    def unexpected_call(*args, **kwargs):
        raise AssertionError("Completed answers must not trigger inference on resume")

    monkeypatch.setattr(grading, "_global_preliminary_judge_zh", unexpected_call)
    resumed = grading._run_mode("teachgrade", limit=len(evaluation), strict_api=True,
                                knowledge_size=4, dataset_path=dataset,
                                eval_size=len(evaluation), output_root=output, resume=True)
    assert resumed == summary
