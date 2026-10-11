from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from utils.teacher_sampling import split_teacher_examples
from utils.task_config import validate_task_config

ROOT = Path(__file__).resolve().parent
MODE = "teachgrade"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def path_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_manifest(
    path: Path,
    rows: list[dict[str, Any]],
    k: int,
    seed: int,
    evaluation_count: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    evaluation, teachers = split_teacher_examples(rows, k, seed, evaluation_count)
    selected_ids = {str(row["id"]) for row in evaluation + teachers}
    manifest = {
        "dataset_sha256": path_sha(path),
        "selection": "uniform_without_replacement",
        "seed": seed,
        "k": k,
        "teacher_ids": [str(row["id"]) for row in teachers],
        "evaluation_ids": [str(row["id"]) for row in evaluation],
        "reserved_ids": [str(row["id"]) for row in rows if str(row["id"]) not in selected_ids],
        "teacher_test_overlap": 0,
        "within_split_order": "original_dataset_order",
    }
    return manifest, evaluation, teachers


@contextmanager
def task_context(builder: Any, rows: list[dict[str, Any]], teachers: list[dict[str, Any]], output: Path) -> Iterator[None]:
    # The original compiler receives task-local data instead of private experiment manifests.
    bindings = {
        "dataset_rows": lambda _dataset: {str(row["id"]): row for row in rows},
        "knowledge20_rows": lambda _dataset: teachers,
        "SPEC_ROOT": output / "specifications",
        "RAW_ROOT": output / "private_raw_calls",
    }
    original = {name: getattr(builder, name) for name in bindings}
    try:
        for name, value in bindings.items():
            setattr(builder, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(builder, name, value)


def prepare_memories(
    dataset_path: Path,
    rows: list[dict[str, Any]],
    teachers: list[dict[str, Any]],
    output: Path,
    task_config: dict[str, Any],
) -> None:
    from frameworks import skill_augmented_grading as grading
    from scripts import task_adaptation as builder

    if len(teachers) < 4:
        raise ValueError("The evidence validator requires at least four teacher examples")
    task_config = validate_task_config(task_config)
    grading.configure_task(task_config)
    grading._set_active_dataset_context(dataset_path, rows)
    base = output / "original_memories"
    runtime = output / f"ks{len(teachers)}"
    skill_path = runtime / "evaluation_skill_zh.json"
    rag_path = runtime / "full_step_casebank_zh_evidence_reason.json"
    manifest_path = output / "memory_manifest.json"
    expected = {
        "split_sha256": path_sha(output / "split_manifest.json"),
        "runtime_sha256": path_sha(ROOT / "frameworks/skill_augmented_grading.py"),
        "builder_sha256": path_sha(ROOT / "scripts/task_adaptation.py"),
        "sampling_sha256": path_sha(ROOT / "utils/teacher_sampling.py"),
        "task_config": task_config,
        "model": os.environ.get("OPENAI_MODEL", ""),
        "base_url": os.environ.get("OPENAI_BASE_URL", ""),
        "thinking_mode": os.environ.get("DEEPSEEK_THINKING_MODE", ""),
    }
    if manifest_path.exists():
        recorded = read_json(manifest_path)
        if any(recorded.get(key) != value for key, value in expected.items()):
            raise ValueError("Existing memories use a different split, model, or source; choose a new output directory")
        if not skill_path.is_file() or not rag_path.is_file():
            raise ValueError("Frozen memories are missing; choose a new output directory")
        if recorded["skill_sha256"] != path_sha(skill_path) or recorded["casebank_sha256"] != path_sha(rag_path):
            raise ValueError("Frozen memory hashes changed")
        return
    if skill_path.exists() or rag_path.exists():
        raise ValueError("Unverified compiled memories already exist; choose a new output directory")
    build_path = output / "build_configuration.json"
    if build_path.exists():
        if read_json(build_path) != expected:
            raise ValueError("Partial memories use a different configuration; choose a new output directory")
    elif base.exists() or (output / "specifications").exists():
        raise ValueError("Unverified partial memories exist; choose a new output directory")
    write_json(build_path, expected)
    source_skill = grading._load_or_build_evaluation_skill_zh(teachers, strict_api=True, output_dir=base)
    source_rag = grading._load_or_build_full_step_casebank_zh_evidence_reason(teachers, output_dir=base)
    source = {
        "skill": source_skill,
        "rag": source_rag,
        "knowledge_ids": [str(row["id"]) for row in teachers],
        "paths": {
            "skill": base / "evaluation_skill_zh.json",
            "rag": base / "full_step_casebank_zh_evidence_reason.json",
        },
    }
    with task_context(builder, rows, teachers, output):
        specification, _envelope = builder.run_task_adaptation(dataset_path.stem, source, force=False)
        skill = builder.compile_skill(source_skill, specification)
        rag = builder.compile_rag(dataset_path.stem, source_rag, specification)
    teacher_ids = set(source["knowledge_ids"])
    if not rag.get("cases") or any(str(case.get("source_sample_id")) not in teacher_ids for case in rag["cases"]):
        raise ValueError("CaseBank has an invalid teacher source")
    write_json(skill_path, skill)
    write_json(rag_path, rag)
    write_json(manifest_path, {
        **expected,
        "skill_sha256": path_sha(skill_path),
        "casebank_sha256": path_sha(rag_path),
        "case_count": len(rag["cases"]),
        "teacher_ids": source["knowledge_ids"],
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="TeachGrade: training-free teacher-evidence scoring")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, help="Task rubric and error categories as JSON")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-size", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/teachgrade")
    parser.add_argument("--phase", choices=("split", "prepare", "evaluate"), default="split")
    parser.add_argument("--run", action="store_true", help="Allow real model API calls")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.phase != "split" and not args.run:
        parser.error("prepare/evaluate require --run because they can make billable model calls")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    task_config = None
    if args.phase != "split":
        if args.task_config is None:
            parser.error("prepare/evaluate require --task-config with your task's grading rubric")
        try:
            task_config = validate_task_config(read_json(args.task_config.expanduser().resolve()))
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    dataset_path = args.dataset.expanduser().resolve()
    rows = read_rows(dataset_path)
    manifest, evaluation, teachers = split_manifest(dataset_path, rows, args.k, args.seed, args.eval_size)
    output = args.output.expanduser().resolve()
    manifest_path = output / "split_manifest.json"
    if manifest_path.exists():
        if read_json(manifest_path) != manifest:
            parser.error("Output contains a different dataset/split; choose a new output directory")
    elif output.exists() and any(output.iterdir()):
        parser.error("Output is not empty and has no split manifest; choose a new output directory")
    write_json(manifest_path, manifest)
    if args.phase == "split":
        print(json.dumps({"teacher_count": len(teachers), "evaluation_count": len(evaluation), "overlap": 0, "seed": args.seed}, indent=2))
        return
    from llm.client import load_project_env
    load_project_env()
    if not os.environ.get("OPENAI_MODEL"):
        parser.error("Configure OPENAI_MODEL before running model calls")
    prepare_memories(dataset_path, rows, teachers, output, task_config)
    if args.phase == "prepare":
        print(json.dumps({"status": "prepared", "teacher_count": len(teachers)}, indent=2))
        return
    from frameworks import skill_augmented_grading as grading
    previous_seed = os.environ.get("TEACHGRADE_RANDOM_SEED")
    os.environ["TEACHGRADE_RANDOM_SEED"] = str(args.seed)
    try:
        summary = grading._run_mode(
            MODE,
            limit=min(args.limit or len(evaluation), len(evaluation)),
            strict_api=True,
            knowledge_size=args.k,
            dataset_path=dataset_path,
            eval_size=len(evaluation),
            output_root=output,
            resume=args.resume,
        )
    finally:
        if previous_seed is None:
            os.environ.pop("TEACHGRADE_RANDOM_SEED", None)
        else:
            os.environ["TEACHGRADE_RANDOM_SEED"] = previous_seed
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
