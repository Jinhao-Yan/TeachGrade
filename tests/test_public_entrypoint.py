from __future__ import annotations

import json
import subprocess
import sys
import pytest

import run_teachgrade as public
from utils.teacher_sampling import split_teacher_examples


def make_dataset(tmp_path):
    rows = [{"id": f"Math_ShortAns2_{index}", "steps": []} for index in range(30)]
    path = tmp_path / "7_Math_ShortAns2.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path, rows


def invoke(dataset, output, *options):
    return subprocess.run(
        [sys.executable, "-S", str(public.ROOT / "run_teachgrade.py"),
         "--dataset", str(dataset), "--output", str(output), "--k", "4", *options],
        capture_output=True, text=True, check=False,
    )


def test_split_command_needs_no_third_party_packages_or_api(tmp_path):
    dataset, rows = make_dataset(tmp_path)
    output = tmp_path / "output"
    result = invoke(dataset, output)
    assert result.returncode == 0, result.stderr
    manifest = public.read_json(output / "split_manifest.json")
    evaluation, teachers = split_teacher_examples(rows, 4, 42)
    assert manifest["teacher_ids"] == [row["id"] for row in teachers]
    assert manifest["evaluation_ids"] == [row["id"] for row in evaluation]
    assert manifest["teacher_test_overlap"] == 0
    assert list(output.iterdir()) == [output / "split_manifest.json"]
    assert invoke(dataset, output).returncode == 0


def test_existing_output_rejects_different_seed(tmp_path):
    dataset, _ = make_dataset(tmp_path)
    output = tmp_path / "output"
    assert invoke(dataset, output).returncode == 0
    before = (output / "split_manifest.json").read_bytes()
    result = invoke(dataset, output, "--seed", "43")
    assert result.returncode != 0
    assert "different dataset/split" in result.stderr
    assert (output / "split_manifest.json").read_bytes() == before


@pytest.mark.parametrize("phase", ["prepare", "evaluate"])
def test_model_phases_require_explicit_run_flag(tmp_path, phase):
    dataset, _ = make_dataset(tmp_path)
    result = invoke(dataset, tmp_path / "output", "--phase", phase)
    assert result.returncode != 0
    assert "require --run" in result.stderr
    assert not (tmp_path / "output").exists()


def test_final_runtime_consumes_the_same_random_split(tmp_path, monkeypatch):
    from frameworks import skill_augmented_grading as grading

    dataset, rows = make_dataset(tmp_path)
    evaluation, teachers = split_teacher_examples(rows, 4, 17, 5)
    observed = {}

    class ReachedPreJudge(BaseException):
        pass

    def skill(knowledge, strict_api, output_dir):
        observed["skill_teachers"] = knowledge
        return {}

    def casebank(knowledge, output_dir):
        observed["casebank_teachers"] = knowledge
        return {"cases": []}

    def prejudge(sample, skill, strict_api):
        observed["first_evaluation"] = sample
        raise ReachedPreJudge

    monkeypatch.setenv("TEACHGRADE_RANDOM_SEED", "17")
    for name in ("ACTIVE_DATASET_META", "ACTIVE_TASK_CONFIG"):
        monkeypatch.setattr(grading, name, getattr(grading, name))
    monkeypatch.setattr(grading, "_load_or_build_evaluation_skill_zh", skill)
    monkeypatch.setattr(grading, "_load_or_build_full_step_casebank_zh_evidence_reason", casebank)
    monkeypatch.setattr(grading, "_audit_full_step_casebank_zh_evidence_reason", lambda *args: {})
    monkeypatch.setattr(grading, "_global_preliminary_judge_zh", prejudge)
    with pytest.raises(ReachedPreJudge):
        grading._run_mode(public.MODE, limit=1, strict_api=True, knowledge_size=4,
                          dataset_path=dataset, eval_size=5, output_root=tmp_path / "runtime")
    assert observed["skill_teachers"] == teachers
    assert observed["casebank_teachers"] == teachers
    assert observed["first_evaluation"] == evaluation[0]


def test_partial_cache_rejects_changed_configuration(tmp_path, monkeypatch, task_config):
    from frameworks import skill_augmented_grading as grading

    dataset, rows = make_dataset(tmp_path)
    output = tmp_path / "output"
    manifest, _, teachers = public.split_manifest(dataset, rows, 4, 42, None)
    public.write_json(output / "split_manifest.json", manifest)
    for name in ("ACTIVE_DATASET_META", "ACTIVE_TASK_CONFIG"):
        monkeypatch.setattr(grading, name, getattr(grading, name))
    monkeypatch.setenv("OPENAI_MODEL", "model-one")

    def interrupt(*args, **kwargs):
        raise RuntimeError("offline interruption")

    monkeypatch.setattr(grading, "_load_or_build_evaluation_skill_zh", interrupt)
    with pytest.raises(RuntimeError, match="offline interruption"):
        public.prepare_memories(dataset, rows, teachers, output, task_config)
    monkeypatch.setenv("OPENAI_MODEL", "model-two")
    with pytest.raises(ValueError, match="different configuration"):
        public.prepare_memories(dataset, rows, teachers, output, task_config)


@pytest.mark.parametrize("phase", ["prepare", "evaluate"])
def test_model_phases_require_a_user_rubric_without_writing_output(tmp_path, phase):
    dataset, _ = make_dataset(tmp_path)
    output = tmp_path / "output"
    result = invoke(dataset, output, "--phase", phase, "--run")
    assert result.returncode != 0
    assert "require --task-config" in result.stderr
    assert not output.exists()


def test_only_teachgrade_mode_is_available():
    from frameworks import skill_augmented_grading as grading

    with pytest.raises(ValueError, match="Only the TeachGrade"):
        grading._run_mode("native_baseline", limit=1, strict_api=True)
