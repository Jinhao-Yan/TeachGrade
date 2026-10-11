from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from llm.client import chat_json_with_diagnostics

ROOT = Path(__file__).resolve().parents[1]
VERSION = "ALL_TASKS_PERCENTAGE_OWNER_REINTERPRETATION_V30"
MODEL_VISIBLE_SCALE = "TEACHER_COMPLETION_PERCENT_ONLY_CURRENT_OWNER_REINTERPRETATION"
SPEC_ROOT = ROOT / "outputs" / "teachgrade" / "specifications"
RAW_ROOT = ROOT / "outputs" / "teachgrade" / "private_raw_calls"
SKILL_FIELDS = (
    "工作流程",
    "评分原则",
    "部分分规则",
    "常见扣分",
    "零分情况",
    "高分情况",
    "边界说明",
)


CONTRACT_FIELDS = (
    "case_selection_rule",
    "current_owner_budget_rule",
    "percentage_reinterpretation_rule",
    "soft_relevance_rule",
    "no_absolute_score_visibility_rule",
    "base_priority_rule",
)


SPECIFICATION_FIELDS = (
    "teacher_behavior_analysis",
    "contrastive_analysis",
    "boundary_induction",
    "ontology_validation",
    "task_ontology",
    "scoring_unit_policy",
    "semantic_equivalence_rule",
    "semantic_non_equivalence_rule",
    "duplicate_credit_rule",
    "dependency_rule",
    "hard_scoring_rules",
    "soft_scoring_guidance",
    "skill_compilation",
    "rag_compilation",
    "percentage_transfer_contract",
    "evidence_support",
)


PRESERVED_CASE_FIELDS = (
    "case_id",
    "source_sample_id",
    "step_id",
    "题目摘要",
    "学生步骤",
    "参考节点匹配",
    "学生行为摘要",
    "错误类型",
    "owner_mapping",
    "credit_ownership",
    "merge_decision",
    "boundary_explanation",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def path_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def compact(value: Any, limit: int = 900) -> str:
    return " ".join(str(value or "").split())[:limit]


def target_paths(dataset: str) -> dict[str, Path]:
    return {"spec": SPEC_ROOT / f"{dataset}.json"}


def dataset_rows(dataset: str) -> dict[str, dict[str, Any]]:
    raise RuntimeError("Bind task-local rows through run_teachgrade.task_context")


def knowledge20_rows(dataset: str) -> list[dict[str, Any]]:
    raise RuntimeError("Bind selected teacher examples through run_teachgrade.task_context")


def knowledge20_payload(dataset: str) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": str(row["id"]),
            "question": compact(row.get("question"), 1200),
            "total": int(row.get("total", 0)),
            "reference": compact(row.get("reference"), 1400),
            "analysis": compact(row.get("analysis"), 1200),
            "teacher_steps": [
                {
                    "step_id": index,
                    "student_response": compact(step.get("response"), 520),
                    "teacher_score": int(step.get("label", 0)),
                    "teacher_error_labels": [str(value) for value in step.get("errors", [])],
                }
                for index, step in enumerate(row.get("steps", []))
                if isinstance(step, dict)
            ],
        }
        for row in knowledge20_rows(dataset)
    ]


def adaptation_prompt(dataset: str, source: dict[str, Any]) -> tuple[str, str]:
    skill = source["skill"]
    rag_cases = source["rag"].get("cases", [])
    payload = {
        "task_id": dataset,
        "adaptation_source": "official Knowledge20 + original SE-Grade Skill/RAG only",
        "knowledge20_teacher_demonstrations": knowledge20_payload(dataset),
        "original_se_grade_skill": skill,
        "original_se_grade_rag_summary": {
            "case_count": len(rag_cases),
            "case_schema": list(rag_cases[0]) if rag_cases else [],
            "representative_cases": rag_cases[:20],
        },
        "consumer_contract": {
            "prejudge": "one global BASE + task Skill call; emits per-step completion percent",
            "retrieval": "unchanged similarity and s-1/s/s+1 normalized percentage bands",
            "final_judge": "one global BASE + task Skill + model-visible teacher boundaries call",
            "artifact_rule": "generate task-specific owner, boundary, dependency, error, and semantic evidence, then preserve it during formal inference",
        },
    }
    system = (
        "你是数据集无关的V30 Task Adaptation Agent。只根据当前任务Knowledge20教师行为和原始"
        "SE-Grade Skill/RAG，重新生成原SE-Grade可消费的完整任务评分表示，不是评分器。只输出JSON。禁止"
        "evaluation、baseline prediction、metric、固定跨题分值、新grader、reject模块、学科名单分支或"
        "pipeline修改。必须先完成teacher behavior analysis、contrastive analysis、boundary induction和"
        "ontology validation，再编译Skill/RAG。不得复用任何旧candidate结论或用generic模板替代任务分析。"
    )
    user = f"""
为输入中的任务重新生成完整、证据支持的V30 Task Grading Specification。Knowledge20不能直接变成Skill：
必须依次经过teacher behavior analysis、supporting/contradicting contrast、boundary induction、ontology
validation。只允许通过validation的claim进入Skill和RAG。百分比只是附加检索与校准坐标，完整产物仍须
包含任务owner、同义/非同义边界、duplicate、错误模式、依赖关系和teacher boundary。

Skill compilation必须为原Skill的七个字段分别生成2-6条简短、可执行、任务专属增量规则：
{json.dumps(list(SKILL_FIELDS), ensure_ascii=False)}
每条必须是{{"rule":"...", "evidence_ids":["sample:<Knowledge20 ID>:S0"], "confidence":0.0}}。
禁止照抄original Skill；必须表达从Knowledge20新归纳出的owner、条件、错误、duplicate或dependency边界。
不要输出新grader；增量规则会被加入原Skill而不是替换原Skill。

RAG compilation必须描述：task_retrieval_objective、owner_mapping_policy、semantic_boundary_policy、
error_pattern_policy、dependency_policy、teacher_boundary_policy。编译器会保留原Knowledge20 case并增加这些
任务专属语义，不改变retriever、similarity或case selection。

固定语义：
1. case_selection_rule：按PreJudge局部完成百分比的s-1/s/s+1档位检索，不改变retriever。
2. current_owner_budget_rule：FinalJudge先从当前题官方评分指南、题干、满分、标准答案和analysis识别
   当前step命中的current-question owner及当前题预算B；禁止使用来源题分值建立B。
3. percentage_reinterpretation_rule：历史教师完成百分比p只表示来源owner的完成度；在当前题中解释为
   当前owner预算上的p×B软边界，再结合当前题完整BASE做全局评分。
4. soft_relevance_rule：不做二元reject；owner/机制越相近，百分比数值参考越强。相关性较弱时仍可参考
   错误模式和higher/lower方向，但降低数值影响。
5. no_absolute_score_visibility_rule：FinalJudge不得看到、恢复、猜测或引用历史教师绝对分和来源局部预算。
6. base_priority_rule：百分比RAG是软参考，不是硬上限、固定单价或BASE替代；冲突时以当前题完整BASE为准。

每项核心claim必须引用当前任务Knowledge20中可观察到的教师行为。evidence_support至少覆盖4个不同
Knowledge20 sample ID；teacher_behavior_analysis必须包含supporting_examples、contradicting_examples和
confidence。无证据的ontology必须拒绝。禁止引用evaluation ID。

严格输出：
{{
  "teacher_behavior_analysis": {{
    "summary": "...", "supporting_examples": ["sample:<Knowledge20 ID>:S0"],
    "contradicting_examples": ["sample:<Knowledge20 ID>:S1"], "confidence": 0.0
  }},
  "contrastive_analysis": [{{"claim": "...", "supporting_examples": ["..."], "contradicting_examples": ["..."], "confidence": 0.0}}],
  "boundary_induction": [{{"boundary": "...", "higher_condition": "...", "lower_condition": "...", "evidence_ids": ["..."], "confidence": 0.0}}],
  "ontology_validation": {{"status": "PASS", "validated_claims": ["..."], "rejected_unsupported_claims": ["..."]}},
  "task_ontology": {{"credit_unit": "...", "owner_definition": "...", "grading_style": "..."}},
  "scoring_unit_policy": "...",
  "semantic_equivalence_rule": "...",
  "semantic_non_equivalence_rule": "...",
  "duplicate_credit_rule": "...",
  "dependency_rule": "...",
  "hard_scoring_rules": ["..."],
  "soft_scoring_guidance": ["..."],
  "skill_compilation": {{
    "工作流程": [{{"rule":"...", "evidence_ids":["..."], "confidence":0.0}}],
    "评分原则": [{{"rule":"...", "evidence_ids":["..."], "confidence":0.0}}],
    "部分分规则": [{{"rule":"...", "evidence_ids":["..."], "confidence":0.0}}],
    "常见扣分": [{{"rule":"...", "evidence_ids":["..."], "confidence":0.0}}],
    "零分情况": [{{"rule":"...", "evidence_ids":["..."], "confidence":0.0}}],
    "高分情况": [{{"rule":"...", "evidence_ids":["..."], "confidence":0.0}}],
    "边界说明": [{{"rule":"...", "evidence_ids":["..."], "confidence":0.0}}]
  }},
  "rag_compilation": {{
    "task_retrieval_objective": "...", "owner_mapping_policy": "...",
    "semantic_boundary_policy": "...", "error_pattern_policy": "...",
    "dependency_policy": "...", "teacher_boundary_policy": "..."
  }},
  "percentage_transfer_contract": {{
    "case_selection_rule": "...",
    "current_owner_budget_rule": "...",
    "percentage_reinterpretation_rule": "...",
    "soft_relevance_rule": "...",
    "no_absolute_score_visibility_rule": "...",
    "base_priority_rule": "..."
  }},
  "evidence_support": [
    {{"claim": "...", "supporting_example_ids": ["sample:<Knowledge20 ID>:S0"], "confidence": 0.0}}
  ]
}}

输入：
{json.dumps(payload, ensure_ascii=False)}
"""
    return system, user


def validate_agent_output(
    dataset: str,
    raw: dict[str, Any],
    knowledge_ids: list[str],
    original_skill: dict[str, Any],
) -> dict[str, Any]:
    if set(raw) != set(SPECIFICATION_FIELDS):
        raise RuntimeError(f"V30 Task Agent schema mismatch: {dataset} {sorted(raw)}")
    contract = raw.get("percentage_transfer_contract")
    if not isinstance(contract, dict) or set(contract) != set(CONTRACT_FIELDS):
        raise RuntimeError(f"V30 transfer contract mismatch: {dataset}")
    for field in CONTRACT_FIELDS:
        if len(compact(contract.get(field), 2000)) < 24:
            raise RuntimeError(f"V30 empty contract rule: {dataset} {field}")
    text = json.dumps(contract, ensure_ascii=False)
    required = ("百分比", "当前", "owner", "预算", "绝对分", "BASE")
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise RuntimeError(f"V30 contract semantics missing: {dataset} {missing}")
    forbidden = (
        "baseline prediction",
        "QWK",
        "CCS",
        "MAE",
        "RMSE",
        "新评分器",
        "veto engine",
        "constraint module",
    )
    found = [marker for marker in forbidden if marker.lower() in text.lower()]
    if found:
        raise RuntimeError(f"V30 prohibited contract semantics: {dataset} {found}")
    analysis = raw.get("teacher_behavior_analysis")
    if not isinstance(analysis, dict) or not {
        "summary",
        "supporting_examples",
        "contradicting_examples",
        "confidence",
    }.issubset(analysis):
        raise RuntimeError(f"V30 teacher behavior analysis mismatch: {dataset}")
    if not 0.0 <= float(analysis.get("confidence", -1)) <= 1.0:
        raise RuntimeError(f"V30 teacher behavior confidence invalid: {dataset}")
    contrastive = raw.get("contrastive_analysis")
    boundaries = raw.get("boundary_induction")
    if not isinstance(contrastive, list) or not contrastive:
        raise RuntimeError(f"V30 contrastive analysis missing: {dataset}")
    if not isinstance(boundaries, list) or not boundaries:
        raise RuntimeError(f"V30 boundary induction missing: {dataset}")
    validation = raw.get("ontology_validation")
    if not isinstance(validation, dict) or validation.get("status") != "PASS":
        raise RuntimeError(f"V30 ontology validation failed: {dataset}")
    skill_compilation = raw.get("skill_compilation")
    if not isinstance(skill_compilation, dict) or set(skill_compilation) != set(SKILL_FIELDS):
        raise RuntimeError(f"V30 Skill compilation schema mismatch: {dataset}")
    legal_ids = set(knowledge_ids)
    for field in SKILL_FIELDS:
        values = skill_compilation[field]
        if not isinstance(values, list) or not 1 <= len(values) <= 8:
            raise RuntimeError(f"V30 Skill compilation count mismatch: {dataset} {field}")
        for value in values:
            if not isinstance(value, dict) or set(value) != {
                "rule",
                "evidence_ids",
                "confidence",
            }:
                raise RuntimeError(f"V30 Skill rule schema mismatch: {dataset} {field}")
            rule = compact(value.get("rule"), 2000)
            if len(rule) < 4:
                raise RuntimeError(f"V30 Skill rule is empty: {dataset} {field}")
            if not 0.0 <= float(value.get("confidence", -1)) <= 1.0:
                raise RuntimeError(f"V30 Skill confidence invalid: {dataset} {field}")
            evidence_ids = value.get("evidence_ids")
            if not isinstance(evidence_ids, list) or not evidence_ids:
                raise RuntimeError(f"V30 Skill evidence missing: {dataset} {field}")
            for evidence_id in evidence_ids:
                sample_id = str(evidence_id).removeprefix("sample:").split(":S", 1)[0]
                if sample_id not in legal_ids:
                    raise RuntimeError(
                        f"V30 Skill non-Knowledge20 evidence: {dataset} {evidence_id}"
                    )
    task_ontology = raw.get("task_ontology")
    hard_rules = raw.get("hard_scoring_rules")
    soft_rules = raw.get("soft_scoring_guidance")
    if not isinstance(task_ontology, dict) or len(task_ontology) < 3:
        raise RuntimeError(f"V30 task ontology insufficient: {dataset}")
    if not isinstance(hard_rules, list) or len(hard_rules) < 2:
        raise RuntimeError(f"V30 hard scoring rules insufficient: {dataset}")
    if not isinstance(soft_rules, list) or len(soft_rules) < 2:
        raise RuntimeError(f"V30 soft scoring guidance insufficient: {dataset}")
    rag_compilation = raw.get("rag_compilation")
    rag_fields = {
        "task_retrieval_objective",
        "owner_mapping_policy",
        "semantic_boundary_policy",
        "error_pattern_policy",
        "dependency_policy",
        "teacher_boundary_policy",
    }
    if not isinstance(rag_compilation, dict) or set(rag_compilation) != rag_fields:
        raise RuntimeError(f"V30 RAG compilation schema mismatch: {dataset}")
    if any(len(compact(rag_compilation[field], 2000)) < 18 for field in rag_fields):
        raise RuntimeError(f"V30 RAG compilation content mismatch: {dataset}")
    cited_samples: set[str] = set()
    evidence = raw.get("evidence_support")
    if not isinstance(evidence, list) or len(evidence) < 2:
        raise RuntimeError(f"V30 evidence support insufficient: {dataset}")
    for item in evidence:
        if not isinstance(item, dict) or not compact(item.get("claim"), 1000):
            raise RuntimeError(f"V30 malformed evidence: {dataset}")
        confidence = float(item.get("confidence", -1))
        if not 0.0 <= confidence <= 1.0:
            raise RuntimeError(f"V30 evidence confidence invalid: {dataset}")
        example_ids = item.get("supporting_example_ids", [])
        if not isinstance(example_ids, list) or not example_ids:
            raise RuntimeError(f"V30 evidence IDs missing: {dataset}")
        for value in example_ids:
            sample_id = str(value).removeprefix("sample:").split(":S", 1)[0]
            if sample_id not in legal_ids:
                raise RuntimeError(f"V30 non-Knowledge20 evidence: {dataset} {value}")
            cited_samples.add(sample_id)
    if len(cited_samples) < 4:
        raise RuntimeError(f"V30 evidence covers fewer than four samples: {dataset}")
    validated = copy.deepcopy(raw)
    validated["percentage_transfer_contract"] = {
        field: compact(contract[field], 1200) for field in CONTRACT_FIELDS
    }
    return validated


def _accepted_raw_event(dataset: str) -> dict[str, Any] | None:
    schema = f"TaskAdaptationAgent:{dataset}:PercentageOwnerReinterpretationV30"
    root = RAW_ROOT / dataset
    for path in sorted(root.glob("*.json"), reverse=True):
        try:
            event = read_json(path)
        except Exception:
            continue
        if (
            event.get("schema_name") == schema
            and event.get("model_call_success") is True
            and not event.get("fallback_used")
        ):
            return event
    return None


def run_task_adaptation(
    dataset: str, source: dict[str, Any], *, force: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    target = target_paths(dataset)["spec"]
    if target.exists() and not force:
        envelope = read_json(target)
        return envelope["task_grading_specification"], envelope
    replay = _accepted_raw_event(dataset) if not force else None
    if replay is not None:
        raw = json.loads(str(replay["raw_response"]))
        diagnostic = {
            "replayed_from_accepted_raw_event": True,
            "request_sha256": replay.get("request_sha256"),
            "response_sha256": replay.get("response_sha256"),
            "prompt_tokens": replay.get("prompt_tokens"),
            "completion_tokens": replay.get("completion_tokens"),
            "api_calls": 0,
        }
    else:
        system, user = adaptation_prompt(dataset, source)
        previous_audit = os.environ.get("LLM_RAW_AUDIT_DIR")
        os.environ["LLM_RAW_AUDIT_DIR"] = str(RAW_ROOT / dataset)
        try:
            raw, diagnostic = chat_json_with_diagnostics(
                system,
                user,
                temperature=0.0,
                max_tokens=12000,
                schema_name=(
                    f"TaskAdaptationAgent:{dataset}:PercentageOwnerReinterpretationV30"
                ),
                strict_api=True,
                json_mode=True,
            )
        finally:
            if previous_audit is None:
                os.environ.pop("LLM_RAW_AUDIT_DIR", None)
            else:
                os.environ["LLM_RAW_AUDIT_DIR"] = previous_audit
    validated = validate_agent_output(
        dataset, raw, source["knowledge_ids"], source["skill"]
    )
    specification = {
        "representation_version": VERSION,
        "adaptation_source": (
            "official Knowledge20 + original SE-Grade Skill/RAG + original consumer contract"
        ),
        "original_skill_sha256": path_sha(source["paths"]["skill"]),
        "original_rag_sha256": path_sha(source["paths"]["rag"]),
        "evaluation_data_used": False,
        "baseline_predictions_used": False,
        **validated,
    }
    envelope = {
        "dataset": dataset,
        "task_adaptation_logical_calls": 1,
        "formal_task_adaptation_calls": 0,
        "formal_skill_rebuilds": 0,
        "formal_rag_rebuilds": 0,
        "knowledge20_ids": source["knowledge_ids"],
        "original_skill_hash": path_sha(source["paths"]["skill"]),
        "original_rag_hash": path_sha(source["paths"]["rag"]),
        "task_grading_specification": specification,
        "diagnostic": diagnostic,
    }
    envelope["sha256"] = canonical_sha(envelope)
    write_json(target, envelope)
    return specification, envelope


def compile_skill(source_skill: dict[str, Any], specification: dict[str, Any]) -> dict[str, Any]:
    skill = copy.deepcopy(source_skill)
    rules = specification["percentage_transfer_contract"]
    original_rules = {
        compact(value, 2000)
        for values in source_skill.values()
        for value in values
    }

    def text(value: Any, limit: int = 900) -> str:
        if isinstance(value, str):
            return compact(value, limit)
        return compact(json.dumps(value, ensure_ascii=False), limit)

    def unique(values: list[Any], limit: int = 6) -> list[str]:
        result: list[str] = []
        for value in values:
            rendered = text(value)
            if rendered and rendered not in result and rendered not in original_rules:
                result.append(rendered)
            if len(result) >= limit:
                break
        return result

    compiled = specification["skill_compilation"]
    advisory = {
        field: unique([value["rule"] for value in compiled[field]], limit=3)
        for field in SKILL_FIELDS
    }
    ontology = specification["task_ontology"]
    hard = unique(specification["hard_scoring_rules"], limit=6)
    soft = unique(specification["soft_scoring_guidance"], limit=6)
    induced_boundaries = [
        (
            f"边界={text(value.get('boundary'))}；higher={text(value.get('higher_condition'))}；"
            f"lower={text(value.get('lower_condition'))}；confidence={value.get('confidence')}"
        )
        for value in specification["boundary_induction"]
        if isinstance(value, dict)
    ]
    task_owner = (
        f"任务评分单位={text(ontology.get('credit_unit'))}；"
        f"owner定义={text(ontology.get('owner_definition'))}；"
        f"评分风格={text(ontology.get('grading_style'))}"
    )
    generated = {
        "工作流程": [
            task_owner,
            f"先按评分单位执行：{text(specification['scoring_unit_policy'])}",
            *hard[:2],
            *advisory["工作流程"],
        ],
        "评分原则": [
            task_owner,
            f"语义等价：{text(specification['semantic_equivalence_rule'])}",
            f"语义非等价：{text(specification['semantic_non_equivalence_rule'])}",
            *hard[:2],
            *advisory["评分原则"],
        ],
        "部分分规则": [
            f"部分分仅按任务边界：{value}" for value in induced_boundaries[:3]
        ]
        + soft[:3]
        + advisory["部分分规则"],
        "常见扣分": hard[:3] + soft[:2] + advisory["常见扣分"],
        "零分情况": hard[:5] + advisory["零分情况"],
        "高分情况": [
            f"达到higher边界：{value}" for value in induced_boundaries[:3]
        ]
        + soft[:3]
        + advisory["高分情况"],
        "边界说明": [
            f"duplicate：{text(specification['duplicate_credit_rule'])}",
            f"dependency：{text(specification['dependency_rule'])}",
            f"语义等价：{text(specification['semantic_equivalence_rule'])}",
            f"语义非等价：{text(specification['semantic_non_equivalence_rule'])}",
            *induced_boundaries[:3],
            *advisory["边界说明"],
        ],
    }
    for field in SKILL_FIELDS:
        task_rules = [
            f"[TASK-ADAPT V30] {value}" for value in unique(generated[field], limit=10)
        ]
        if not task_rules:
            raise RuntimeError(f"V30 Skill compiler produced no task delta: {field}")
        skill[field] = [*task_rules, *skill.get(field, [])]
    skill["工作流程"] = [
        f"[TASK-ADAPT V30 PERCENT] {rules['current_owner_budget_rule']}",
        f"[TASK-ADAPT V30 PERCENT] {rules['percentage_reinterpretation_rule']}",
        *skill["工作流程"],
    ]
    skill["边界说明"] = [
        f"[TASK-ADAPT V30 PERCENT] {rules['case_selection_rule']}",
        f"[TASK-ADAPT V30 PERCENT] {rules['soft_relevance_rule']}",
        f"[TASK-ADAPT V30 PERCENT] {rules['no_absolute_score_visibility_rule']}",
        f"[TASK-ADAPT V30 PERCENT] {rules['base_priority_rule']}",
        *skill["边界说明"],
    ]
    if tuple(skill) != SKILL_FIELDS:
        raise RuntimeError("V30 Skill schema drift")
    return skill


def _case_owner_evidence(
    dataset: str, cases: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    rows = dataset_rows(dataset)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    enriched: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case.get("case_id", ""))
        source_id = str(case.get("source_sample_id", ""))
        if not case_id or source_id not in rows:
            raise RuntimeError(f"V30 case source mismatch: {dataset} {case_id} {source_id}")
        row = rows[source_id]
        question_key = hashlib.sha256(
            compact(row.get("question"), 4000).encode("utf-8")
        ).hexdigest()[:16]
        owner_basis = compact(case.get("参考节点匹配"), 2000)
        if not owner_basis:
            owner_basis = f"step:{int(case.get('step_id', 0))}"
        owner_key = hashlib.sha256(owner_basis.encode("utf-8")).hexdigest()[:16]
        value = {
            "case_id": case_id,
            "source_sample_id": source_id,
            "question_key": question_key,
            "owner_key": owner_key,
            "owner_basis": owner_basis,
            "teacher_score": int(case.get("教师评分", 0)),
            "boundary_explanation": compact(case.get("评分边界"), 1600),
            "merge_decision": "SAME_QUESTION_SAME_REFERENCE_OWNER_ONLY",
        }
        enriched.append(value)
        groups[(question_key, owner_key)].append(value)
    result: dict[str, dict[str, Any]] = {}
    for group_key, values in groups.items():
        credits = [int(value["teacher_score"]) for value in values]
        positive = [value for value in credits if value > 0]
        budget = max(positive, default=0)
        distinct = sorted(set(credits))
        evidence_kind = (
            "TRUE_SAME_QUESTION_SAME_OWNER_CONTRAST"
            if len(distinct) >= 2
            else "OBSERVED_SINGLE_BOUNDARY"
        )
        confidence = 1.0 if len(distinct) >= 2 else 0.65
        for value in values:
            result[str(value["case_id"])] = {
                "domain_id": group_key[0],
                "owner_ids": [group_key[1]],
                "owner_basis": value["owner_basis"],
                "observed_scope_budget": budget,
                "observed_teacher_scores": distinct,
                "boundary_evidence_kind": evidence_kind,
                "boundary_confidence": confidence,
                "merge_decision": value.get("merge_decision", ""),
                "boundary_explanation": value.get("boundary_explanation", ""),
                "credit_ownership": {
                    "observed_teacher_credit": value["teacher_score"],
                    "allocation": "OBSERVED_SAME_QUESTION_REFERENCE_OWNER_SCOPE",
                },
            }
    if set(result) != {str(case.get("case_id", "")) for case in cases}:
        raise RuntimeError(f"V30 case owner materialization mismatch: {dataset}")
    return result


def _clean_model_text(value: Any) -> str:
    text = compact(value, 2200)
    substitutions = (
        (r"教师(?:绝对)?评分\s*[：=]\s*-?\d+(?:\.\d+)?", "教师历史完成边界"),
        (r"(?:来源)?教师分\s*[：=]\s*-?\d+(?:\.\d+)?", "来源教师完成边界"),
        (r"来源局部预算\s*[：=]\s*-?\d+(?:\.\d+)?", "来源owner观测预算仅审计可见"),
        (r"来源题总分\s*[：=]\s*-?\d+(?:\.\d+)?", "来源题总分仅审计可见"),
        (r"当前案例\s*=?.{0,60}?教师分\s*=\s*-?\d+(?:\.\d+)?", "当前历史案例完成边界"),
        (r"当前\s*=\s*-?\d+(?:\.\d+)?\s*分", "当前历史案例完成边界"),
        (r"K\d+_S\d+\s*=\s*-?\d+(?:\.\d+)?\s*分", "相邻历史案例边界"),
        (r"人工(?:给|评)\s*-?\d+(?:\.\d+)?\s*分", "人工确认该完成边界"),
        (r"(?:给|得)\s*-?\d+(?:\.\d+)?\s*分", "达到相应完成边界"),
    )
    for pattern, replacement in substitutions:
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"[；;]{2,}", "；", text)
    return text.strip(" ；;")


def _clean_policy_text(value: Any) -> str:
    text = _clean_model_text(value)
    replacements = {
        "教师评分": "教师给分行为",
        "来源局部预算": "来源owner观测尺度",
        "来源教师绝对分": "来源教师完成边界",
        "来源题总分": "来源题尺度",
    }
    for original, replacement in replacements.items():
        text = text.replace(original, replacement)
    return text


def _model_visible_payload(case: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in case.items()
        if key.startswith("模型可见_") or key == "模型可见评分尺度"
    }


def compile_rag(
    dataset: str,
    source_rag: dict[str, Any],
    specification: dict[str, Any],
) -> dict[str, Any]:
    rag = copy.deepcopy(source_rag)
    cases = rag.get("cases", [])
    evidence = _case_owner_evidence(dataset, cases)
    rows = dataset_rows(dataset)
    rules = specification["percentage_transfer_contract"]
    rag_rules = specification["rag_compilation"]
    visible_rules = {
        key: _clean_policy_text(value) for key, value in rag_rules.items()
    }
    visible_transfer_rules = {
        key: _clean_policy_text(value) for key, value in rules.items()
    }
    for case in cases:
        case_id = str(case["case_id"])
        mapping = evidence[case_id]
        score = int(case.get("教师评分", 0))
        budget = int(mapping["observed_scope_budget"])
        if budget > 0:
            ratio = max(0.0, min(1.0, score / budget))
        else:
            ratio = 0.0
        ratio = round(ratio, 4)
        band = max(0, min(10, round(ratio * 10)))
        percentage = f"{ratio * 100:.0f}%"
        source_id = str(case.get("source_sample_id", ""))
        source_total = int(rows.get(source_id, {}).get("total", 0))
        original_audit = {
            "教师评分": case.get("教师评分"),
            "来源局部预算": budget,
            "来源题总分": source_total,
            "原评分边界": case.get("评分边界", ""),
            "原评分理由短评": case.get("评分理由短评", ""),
            "原检索文本_证据": case.get("检索文本_证据", ""),
        }
        owner_text = compact(mapping["owner_basis"], 700) or "当前案例任务owner"
        adapted_boundary = _clean_model_text(
            mapping.get("boundary_explanation") or case.get("评分边界", "")
        )
        source_evidence = _clean_model_text(case.get("检索文本_证据", ""))
        case.update(
            {
                "来源题总分": source_total,
                "来源局部预算": budget,
                "局部完成比例": ratio,
                "归一化检索档位": band,
                "局部预算证据状态": mapping["boundary_evidence_kind"],
                "边界置信度": mapping["boundary_confidence"],
                "模型可见评分尺度": MODEL_VISIBLE_SCALE,
                "模型可见_教师完成百分比": percentage,
                "模型可见_边界角色": mapping["boundary_evidence_kind"],
                "模型可见_参考得分点": compact(
                    f"owner={owner_text}；{case.get('参考节点匹配', '')}", 700
                ),
                "模型可见_学生作答行为": compact(case.get("学生步骤", ""), 650),
                "模型可见_相对边界": compact(
                    f"{adapted_boundary}；证据状态={mapping['boundary_evidence_kind']}；"
                    f"错误模式={'、'.join(str(value) for value in case.get('错误类型', [])) or '步骤正确'}；"
                    f"合并规则={visible_rules['semantic_boundary_policy']}；"
                    f"依赖规则={visible_rules['dependency_policy']}；"
                    f"teacher boundary={visible_rules['teacher_boundary_policy']}",
                    1000,
                ),
                "模型可见_当前owner预算重解释": visible_transfer_rules[
                    "percentage_reinterpretation_rule"
                ],
                "模型可见_软参考规则": visible_transfer_rules[
                    "soft_relevance_rule"
                ],
                "评分边界": adapted_boundary,
                "评分理由短评": compact(
                    f"{_clean_model_text(case.get('评分理由短评', ''))}；"
                    f"{visible_transfer_rules['soft_relevance_rule']}",
                    900,
                ),
                "检索文本_证据": "\n".join(
                    value
                    for value in (
                        source_evidence,
                        f"任务owner：{owner_text}",
                        f"任务检索目标：{visible_rules['task_retrieval_objective']}",
                        f"owner映射：{visible_rules['owner_mapping_policy']}",
                        f"任务错误模式：{visible_rules['error_pattern_policy']}",
                        f"教师完成百分比：{percentage}",
                        f"边界证据状态：{mapping['boundary_evidence_kind']}",
                    )
                    if value
                ),
                "校准文本_理由": (
                    f"教师完成百分比：{percentage}\n"
                    f"相对边界：{adapted_boundary}\n"
                    f"当前题重解释：{visible_transfer_rules['percentage_reinterpretation_rule']}\n"
                    f"软参考：{visible_transfer_rules['soft_relevance_rule']}"
                ),
                "比例用途": "RETRIEVAL_AND_CURRENT_OWNER_REINTERPRETATION",
                "当前题直接计分用途": "SOFT_PERCENTAGE_BOUNDARY_ONLY",
                "二元拒绝": False,
                "审计_原始教师评分记录": original_audit,
            }
        )
    rag.update(
        {
            "method_version": VERSION,
            "original_se_grade_rag_sha256": canonical_sha(source_rag),
            "model_visible_score_scale": MODEL_VISIBLE_SCALE,
            "judge_raw_teacher_score_visible": False,
            "strict_reject": False,
            "ratio_usage": "RETRIEVAL_AND_CURRENT_OWNER_REINTERPRETATION",
            "direct_scoring_usage": "SOFT_CURRENT_OWNER_PERCENTAGE_CALIBRATION",
            "task_adapted_content_policy": "PRESERVE_AND_ENRICH",
            "task_retrieval_objective": rag_rules["task_retrieval_objective"],
            "owner_mapping_policy": rag_rules["owner_mapping_policy"],
            "semantic_boundary_policy": rag_rules["semantic_boundary_policy"],
            "error_pattern_policy": rag_rules["error_pattern_policy"],
            "dependency_policy": rag_rules["dependency_policy"],
            "teacher_boundary_policy": rag_rules["teacher_boundary_policy"],
        }
    )
    return rag
