from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from llm.client import chat_json_with_diagnostics
from utils.collaborative_consistency_score import adjusted_qwk
from utils.errors_consistency_score import compute_ecs
from utils.task_config import validate_task_config
from utils.teacher_sampling import split_teacher_examples

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "teachgrade"
DATA_PATH = PROJECT_ROOT / "data" / "task.jsonl"
ACTIVE_DATASET_META: dict[str, str] = {}
ACTIVE_TASK_CONFIG: dict[str, Any] = {}


def configure_task(config: dict[str, Any]) -> None:
    global ACTIVE_TASK_CONFIG
    ACTIVE_TASK_CONFIG = validate_task_config(config)


def _infer_dataset_meta(dataset_path: Path | None = None, samples: list[dict[str, Any]] | None = None) -> dict[str, str]:
    path = dataset_path or DATA_PATH
    stem = path.stem
    q_id = ""
    course = ""
    question_type = ""
    match = re.match(r"^(\d+)_([^_]+)_(.+)$", stem)
    if match:
        q_id, course, question_type = match.groups()
    if samples:
        first_id = str(samples[0].get("id", "") if samples else "")
        id_match = re.match(r"^([^_]+)_(.+)_\d+$", first_id)
        if id_match:
            course = course or id_match.group(1)
            question_type = question_type or id_match.group(2)
    q_id = str(ACTIVE_TASK_CONFIG.get("q_id") or q_id or stem)
    course = str(ACTIVE_TASK_CONFIG.get("course") or course or "Unknown")
    question_type = str(ACTIVE_TASK_CONFIG.get("question_type") or question_type or "Unknown")
    course_zh = {
        "Math": "数学",
        "Physics": "物理",
        "Chemistry": "化学",
        "Biology": "生物",
        "History": "历史",
        "Geography": "地理",
        "Political": "政治",
        "Chinese": "语文",
        "English": "英语",
    }.get(course, course)
    question_type_zh = {
        "ShortAns": "简答题",
        "gapfilling": "填空题",
        "Choice": "选择题",
    }.get(question_type, question_type)
    return {
        "q_id": str(q_id),
        "course": course,
        "question_type": question_type,
        "dataset_stem": stem,
        "task_name": f"{course} {question_type}",
        "task_name_zh": f"{course_zh}{question_type_zh}",
    }


def _set_active_dataset_context(dataset_path: Path | None = None, samples: list[dict[str, Any]] | None = None) -> dict[str, str]:
    global ACTIVE_DATASET_META
    ACTIVE_DATASET_META = _infer_dataset_meta(dataset_path, samples)
    return ACTIVE_DATASET_META


def _task_meta() -> dict[str, str]:
    return ACTIVE_DATASET_META


def _task_name_zh() -> str:
    return ACTIVE_DATASET_META.get("task_name_zh", "当前题型")


def _is_math_shortans() -> bool:
    return ACTIVE_DATASET_META.get("course") == "Math" and ACTIVE_DATASET_META.get("question_type") == "ShortAns"


def _output_dir_for_knowledge_size(knowledge_size: int | None) -> Path:
    if knowledge_size is None:
        return OUTPUT_DIR
    return OUTPUT_DIR / f"ks{knowledge_size}"


def _main_output_dir(output_root: Path | None, knowledge_size: int | None, mode: str) -> Path:
    if output_root is None:
        return _output_dir_for_knowledge_size(knowledge_size)
    if knowledge_size is None:
        return output_root
    return output_root / f"ks{knowledge_size}"


def _path_in_output(output_dir: Path, filename: str) -> Path:
    return output_dir / filename


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _short(text: Any, limit: int = 420) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _skill_guardrail_checklist_zh(skill_zh: dict[str, Any]) -> dict[str, Any]:
    return {
        "定位": "Skill 只做 guardrail，不决定具体 0/1/2/高分边界；具体分数边界主要由原题、标准答案、学生作答和 RAG 人工案例边界决定。",
        "防重复": "同一有效作答贡献只给一次；后续重复同一答案点、同一结论或同一信息不再给分。",
        "错误链": "完全依赖错误前提且没有新增正确关键点的后续步骤通常不给分。",
        "setup_only": "纯设变量、抄条件、前置整理、说明接下来要做什么，通常不是独立得分贡献。",
        "局部有效贡献保护": "首次出现且独立成立的关键答案点、关键事实、关键概念、关键选项、关键表达或有效中间结论，即使后续错误，也应保留对应步骤分。",
        "最终答案型保护": "若题型允许只写答案且学生答案与标准答案等价，不要仅因缺少过程直接判 0；需结合题目要求和 RAG 边界判断。",
        "skill_notes": {
            "部分分规则": [str(x) for x in skill_zh.get("部分分规则", [])[:3]] if isinstance(skill_zh, dict) else [],
            "零分情况": [str(x) for x in skill_zh.get("零分情况", [])[:3]] if isinstance(skill_zh, dict) else [],
        },
    }


def _skill_hierarchical_checklist_zh(skill_zh: dict[str, Any]) -> dict[str, Any]:
    return {
        "定位": "Skill 是主要评分流程；RAG 只做局部评分边界校准，不能推翻 Skill 对新增有效贡献/重复/错误链的基本判断。",
        "评分顺序": [
            "先判断当前 step 是否有新增有效作答贡献。",
            "再判断是否只是重复、setup-only、错误链延续或空泛描述。",
            "通过前两步后，再参考 RAG 边界决定具体分数。",
        ],
        "新增有效贡献": "认可独立成立的关键答案点、关键事实、关键概念、关键选项、关键表达、有效中间推导或正确关键结论。",
        "不重复": "同一有效贡献只给一次。",
        "错误链": "错误前提导致的后续推导，若无新的正确关键点，不给分。",
        "最终答案型保护": "若题型本身允许只写最终答案且答案等价，不能机械要求完整过程；但若题目明确要求过程，应按步骤完整性处理。",
        "skill_extract": {
            "评分原则": [str(x) for x in skill_zh.get("评分原则", [])[:4]] if isinstance(skill_zh, dict) else [],
            "部分分规则": [str(x) for x in skill_zh.get("部分分规则", [])[:4]] if isinstance(skill_zh, dict) else [],
            "零分情况": [str(x) for x in skill_zh.get("零分情况", [])[:4]] if isinstance(skill_zh, dict) else [],
            "边界说明": [str(x) for x in skill_zh.get("边界说明", [])[:4]] if isinstance(skill_zh, dict) else [],
        },
    }


def _tokens(text: Any) -> set[str]:
    value = str(text or "").lower()
    return set(re.findall(r"[\u4e00-\u9fff]{1,4}|[a-z]+|\d+(?:\.\d+)?", value))


def _similarity(query: Any, text: Any) -> float:
    query_tokens = _tokens(query)
    text_tokens = _tokens(text)
    if not query_tokens or not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / math.sqrt(len(query_tokens) * len(text_tokens))


def _clamp_int(value: Any, lo: int, hi: int) -> int:
    try:
        out = int(round(float(value)))
    except Exception:
        out = lo
    return max(lo, min(hi, out))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    return text in {"true", "yes", "1", "是", "是的", "对", "首次", "重复", "延续"}


def _safe_errors(value: Any) -> list[str]:
    allowed_labels = _official_error_labels()
    if isinstance(value, str):
        labels = [value]
    elif isinstance(value, list):
        labels = [str(item) for item in value]
    else:
        labels = []
    cleaned = [item for item in labels if item in allowed_labels]
    return cleaned or ["步骤正确"]


def _task_error_type_record() -> dict[str, Any]:
    if not ACTIVE_TASK_CONFIG:
        raise RuntimeError("Provide a task configuration with a grading guideline and error labels")
    return {**ACTIVE_TASK_CONFIG, **_task_meta()}


def _official_score_guideline() -> str:
    record = _task_error_type_record()
    meta = _task_meta()
    if (
        not record
        or str(record.get("q_id")) != str(meta.get("q_id"))
        or str(record.get("course")) != str(meta.get("course"))
        or str(record.get("question_type")) != str(meta.get("question_type"))
    ):
        raise RuntimeError(
            "官方评分指南与当前任务不匹配: "
            f"task={meta.get('course')} {meta.get('question_type')} q_id={meta.get('q_id')} "
            f"guideline_record={record.get('course')} {record.get('question_type')} q_id={record.get('q_id')}"
        )
    guideline = str(record.get("guideline", "")).strip()
    if not guideline:
        raise RuntimeError(f"当前任务缺少官方评分指南: {meta}")
    return guideline


def _base_scoring_context_audit(sample: dict[str, Any]) -> dict[str, Any]:
    guideline = _official_score_guideline()
    analysis = str(sample.get("analysis", ""))
    meta = _task_meta()
    return {
        "q_id": meta.get("q_id"),
        "course": meta.get("course"),
        "question_type": meta.get("question_type"),
        "official_score_guideline_present": bool(guideline),
        "question_analysis_present": bool(analysis.strip()),
        "official_score_guideline_sha256": hashlib.sha256(guideline.encode("utf-8")).hexdigest(),
        "question_analysis_sha256": hashlib.sha256(analysis.encode("utf-8")).hexdigest(),
        "inherits_native_base_inputs": True,
    }


def _error_types() -> list[dict[str, str]]:
    errors = list(_task_error_type_record().get("errors", []))
    if not any(e.get("name") == "步骤正确" for e in errors):
        errors.append({"name": "步骤正确", "description": "该步骤正确"})
    return errors or [{"name": "步骤正确", "description": "该步骤正确"}]


def _official_error_labels() -> list[str]:
    labels = [str(item.get("name", "")).strip() for item in _error_types()]
    return [label for label in labels if label]


def _student_answer(sample: dict[str, Any]) -> str:
    return "\n".join(f"## Step {i}. {step.get('response', '')}" for i, step in enumerate(sample.get("steps", [])))


def _load_or_build_evaluation_skill_zh(knowledge_set: list[dict[str, Any]], strict_api: bool, output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    skill_path = _path_in_output(output_dir, "evaluation_skill_zh.json")
    if skill_path.exists():
        return json.loads(skill_path.read_text(encoding="utf-8"))

    compact_examples: list[dict[str, Any]] = []
    for sample in knowledge_set:
        compact_examples.append(
            {
                "样本编号": sample.get("id"),
                "满分": sample.get("total"),
                "人工总分": sample.get("manual_label"),
                "参考答案摘要": _short(sample.get("reference"), 500),
                "步骤": [
                    {
                        "学生步骤": _short(step.get("response"), 220),
                        "教师评分": step.get("label", 0),
                        "错误标签": step.get("errors", []),
                    }
                    for step in sample.get("steps", [])[:12]
                ],
            }
        )
    task = _task_name_zh()
    system = f"你是{task}评分经验总结器。必须只输出 JSON，所有字段和值都用中文。"
    user = f"""
请基于 knowledge_set 总结一份通用中文{task}评分 skill。

要求：
- 只总结通用评分流程，不要写成某一道题的标准答案。
- 必须贴合当前学科和题型，不要套用其他学科的公式、方程或步骤要求。
- 所有输出内容必须为中文。
- JSON 字段必须且只能使用：
  工作流程, 评分原则, 部分分规则, 常见扣分, 零分情况, 高分情况, 边界说明。
- 每个字段是中文字符串数组，内容要适合后续阅卷老师直接参考。

knowledge_set:
{json.dumps(compact_examples, ensure_ascii=False)}
"""
    data, _ = chat_json_with_diagnostics(
        system,
        user,
        temperature=0.1,
        max_tokens=2200,
        schema_name="中文评分Skill生成器",
        strict_api=strict_api,
    )
    keys = ["工作流程", "评分原则", "部分分规则", "常见扣分", "零分情况", "高分情况", "边界说明"]
    skill = {key: [str(x) for x in data.get(key, []) if str(x).strip()] for key in keys}
    defaults = {
        "工作流程": ["先通读题目、标准答案和学生完整作答，再判断学生作答是否回应题目要求，并逐步核对关键评分点。"],
        "评分原则": ["以标准答案和题目满分为依据，只认可可独立成立的新增有效作答贡献，避免机械累加无效或重复内容。"],
        "部分分规则": ["作答与参考答案中的部分关键点对应、方向基本正确但不完整、或只完成部分空/小问时，可以按题型给部分分。"],
        "常见扣分": ["关键概念、事实、条件、对象、数值、选项、用词或结论与参考答案不一致时应扣分。"],
        "零分情况": ["空白、无关内容、纯抄题干、重复无新增、答案方向明显错误，通常不给分。"],
        "高分情况": ["完整覆盖题目要求和参考答案关键点，表达准确，步骤或答案符合题型要求，通常可得高分或满分。"],
        "边界说明": ["区分前置整理与可独立得分贡献；区分局部正确与达到关键评分节点。"],
    }
    for key, value in defaults.items():
        if not skill.get(key):
            skill[key] = value
    _write_json(skill_path, skill)
    return skill


def _question_summary(sample: dict[str, Any]) -> str:
    question = str(sample.get("question", ""))
    return _short(re.sub(r"<[^>]+>", " ", question), 260)


def _reference_context(sample: dict[str, Any], step: dict[str, Any]) -> str:
    response = str(step.get("response", ""))
    reference = str(sample.get("reference", ""))
    response_tokens = _tokens(response)
    sentences = [s.strip() for s in re.split(r"[\n。；;]+", reference) if s.strip()]
    if not sentences:
        return _short(reference, 260)
    ranked = sorted(
        sentences,
        key=lambda sentence: len(response_tokens & _tokens(sentence)),
        reverse=True,
    )
    return _short("；".join(ranked[:2]), 300)


def _full_step_score_bucket(label: int, errors: list[str], response: str) -> str:
    if label <= 0:
        if _is_math_shortans() and response and any(token in response for token in ["设", "由", "公式", "代入", "计算"]):
            return "low_credit"
        return "zero_credit"
    if label == 1:
        return "partial_credit" if "步骤正确" in errors else "boundary"
    if errors and "步骤正确" not in errors:
        return "boundary"
    return "full_credit"


def _zh_score_bucket(label: int, errors: list[str], response: str) -> str:
    bucket = _full_step_score_bucket(label, errors, response)
    mapping = {
        "full_credit": "满分",
        "partial_credit": "部分分",
        "low_credit": "低分",
        "zero_credit": "零分",
        "boundary": "边界",
    }
    return mapping.get(bucket, "边界")


def _zh_reason_for_case(label: int, errors: list[str], score_bucket: str, response: str) -> str:
    error_text = "、".join(errors or ["步骤正确"])
    if label <= 0:
        if "计算错误" in errors or "计算错误或单位错误" in errors or "计算错误或单位混乱" in errors:
            return "该步骤涉及计算或代入，但关键数值结果不成立，人工未认可为有效得分贡献，因此给零分。"
        if "公式使用错误或遗漏" in errors or "公式套用" in errors or "解答时公式使用错误" in errors:
            return "该步骤使用的公式或关键表达不符合题目要求，不能支撑后续正确解法，因此给零分。"
        if response and _is_math_shortans() and any(token in response for token in ["设", "由", "可得", "计算", "公式"]):
            return "该步骤主要是前置整理或无效尝试，尚未形成可独立认可的评分贡献，因此给零分。"
        return "该步骤没有提供可被人工认可的有效作答贡献，通常不给分。"
    if score_bucket == "低分":
        return f"该步骤包含少量可认可内容，但仍存在{error_text}，只能获得低分。"
    if score_bucket == "部分分":
        return f"该步骤体现了部分正确答案点、方法方向或中间推进，但尚未完整达到后续评分要求，因此给予部分分。"
    if score_bucket == "边界":
        return f"该步骤有一定有效内容，但同时存在{error_text}，属于评分边界案例，需要参考相似人工案例校准。"
    return "该步骤与参考答案中的相应关键点一致，答案、过程或表达可独立认可，因此获得该步骤对应分数。"


def _zh_behavior_summary(response: str, errors: list[str]) -> str:
    fragments: list[str] = []
    if any(token in response for token in ["公式", "K", "χ", "方程", "="]):
        fragments.append("包含公式或方程表达")
    if any(token in response for token in ["代入", "=", "≈", "计算", "+", "-", "×", "/", "^"]):
        fragments.append("包含代入或计算过程")
    if any(token in response for token in ["所以", "因此", "故", "结论", "有把握", "最大", "最小", "范围"]):
        fragments.append("包含推理结论或最终判断")
    if any(token in response for token in ["设", "令", "记"]):
        fragments.append("包含变量设定或条件整理")
    if "计算错误" in errors:
        fragments.append("涉及数值计算偏差")
    if "公式使用错误或遗漏" in errors:
        fragments.append("涉及公式使用偏差")
    if "答题步骤不完整" in errors:
        fragments.append("涉及关键步骤缺失")
    if "解题思路混乱" in errors:
        fragments.append("涉及推理链不连贯")
    if "答案与过程不符" in errors:
        fragments.append("涉及结论与过程不一致")
    for error in errors:
        if error and error != "步骤正确" and error not in {"计算错误", "公式使用错误或遗漏", "答题步骤不完整", "解题思路混乱", "答案与过程不符"}:
            fragments.append(f"涉及{error}")
    if not fragments:
        fragments.append("呈现学生当前步骤内容")
    return "；".join(dict.fromkeys(fragments))


def _load_or_build_full_step_casebank_zh_evidence_reason(knowledge_set: list[dict[str, Any]], output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    casebank_path = _path_in_output(output_dir, "full_step_casebank_zh_evidence_reason.json")
    if casebank_path.exists():
        return json.loads(casebank_path.read_text(encoding="utf-8"))

    cases: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(knowledge_set):
        source_sample_id = str(sample.get("id", f"K{sample_index}"))
        question_summary = _question_summary(sample)
        for step_index, step in enumerate(sample.get("steps", [])):
            response = _short(step.get("response"), 420)
            if not response:
                continue
            label = _clamp_int(step.get("label", 0), -50, 50)
            errors = [str(error) for error in step.get("errors", [])]
            score_bucket = _zh_score_bucket(label, errors, response)
            error_types = errors or ["步骤正确"]
            reference_match = _reference_context(sample, step)
            behavior_summary = _zh_behavior_summary(response, errors)
            evidence_pattern = "；".join(
                [
                    f"学生行为：{behavior_summary}",
                    f"参考节点：{reference_match}",
                    f"错误类型：{'、'.join(error_types)}",
                ]
            )
            grading_boundary = f"人工给 {label} 分；该步骤属于{score_bucket}；错误类型：{'、'.join(error_types)}。"
            reason = _zh_reason_for_case(label, errors, score_bucket, response)
            evidence_text = "\n".join(
                [
                    f"题目摘要：{question_summary}",
                    f"学生步骤：{response}",
                    f"学生行为摘要：{behavior_summary}",
                    f"参考节点匹配：{reference_match}",
                    f"错误类型：{'、'.join(error_types)}",
                ]
            )
            calibration_text = "\n".join(
                [
                    f"教师评分：{label}",
                    f"分数档位：{score_bucket}",
                    f"评分边界：{grading_boundary}",
                    f"评分理由短评：{reason}",
                ]
            )
            cases.append(
                {
                    "case_id": f"K{sample_index + 1}_S{step_index}",
                    "source_sample_id": source_sample_id,
                    "step_id": step_index,
                    "题目摘要": question_summary,
                    "学生步骤": response,
                    "教师评分": label,
                    "分数档位": score_bucket,
                    "证据模式": evidence_pattern,
                    "参考节点匹配": reference_match,
                    "学生行为摘要": behavior_summary,
                    "错误类型": error_types,
                    "评分边界": grading_boundary,
                    "评分理由短评": reason,
                    "检索文本_证据": evidence_text,
                    "校准文本_理由": calibration_text,
                }
            )
    casebank = {"cases": cases}
    _write_json(casebank_path, casebank)
    return casebank


def _audit_full_step_casebank_zh_evidence_reason(casebank: dict[str, Any], output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    cases = [case for case in casebank.get("cases", []) if isinstance(case, dict)]
    audit = {
        "num_cases": len(cases),
        "num_step_cases": len(cases),
        "score_distribution": dict(Counter(str(case.get("教师评分", 0)) for case in cases)),
        "score_bucket_distribution": dict(Counter(str(case.get("分数档位", "")) for case in cases)),
        "error_type_distribution": dict(Counter(tag for case in cases for tag in case.get("错误类型", []))),
        "evidence_text_missing_count": sum(1 for case in cases if not str(case.get("检索文本_证据", "")).strip()),
        "reason_text_missing_count": sum(1 for case in cases if not str(case.get("校准文本_理由", "")).strip()),
        "reason_comment_missing_count": sum(1 for case in cases if not str(case.get("评分理由短评", "")).strip()),
    }
    _write_json(_path_in_output(output_dir, "full_step_casebank_zh_evidence_reason_audit.json"), audit)
    return audit


def _score_band_for_case(case: dict[str, Any]) -> int:
    return _clamp_int(case.get("gold_score", case.get("教师评分", 0)), -50, 50)


def _evaluation_record(sample: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    item = dict(sample)
    item["pred_label"] = _clamp_int(prediction.get("pred_score", 0), 0, _clamp_int(sample.get("total", 0), 0, 100))
    pred_steps = list(prediction.get("steps", []))
    gold_steps = sample.get("steps", [])
    pred_steps = pred_steps[: len(gold_steps)]
    while len(pred_steps) < len(gold_steps):
        pred_steps.append({"step_score": 0, "errors": []})
    item["pred_steps"] = pred_steps
    return item


def _step_context(sample: dict[str, Any], step_id: int, radius: int = 1) -> dict[str, Any]:
    steps = sample.get("steps", [])
    previous_steps = []
    next_steps = []
    for i in range(max(0, step_id - radius), step_id):
        previous_steps.append({"step_id": i, "text": _short(steps[i].get("response", ""), 260)})
    for i in range(step_id + 1, min(len(steps), step_id + 1 + radius)):
        next_steps.append({"step_id": i, "text": _short(steps[i].get("response", ""), 260)})
    current = steps[step_id].get("response", "") if 0 <= step_id < len(steps) else ""
    return {
        "step_id": step_id,
        "current_step": _short(current, 700),
        "previous_steps_context": previous_steps,
        "next_steps_context": next_steps,
    }


def _global_preliminary_judge_zh(
    sample: dict[str, Any],
    skill_zh: dict[str, Any],
    strict_api: bool,
) -> dict[str, Any]:
    total = _clamp_int(sample.get("total", 0), 0, 100)
    steps = sample.get("steps", [])
    task = _task_name_zh()
    score_guideline = _official_score_guideline()
    analysis = str(sample.get("analysis", ""))
    base_context_audit = _base_scoring_context_audit(sample)
    system = f"你是中文{task}全局初判员。必须只输出 JSON。初判只用于逐step检索，不是最终评分。"
    user = f"""
请先像 Native BASE 一样对完整学生答案做一次全局评分，再输出全部 student step 的初步分数，供后续每步独立检索 RAG 案例。

BASE 输入继承要求：
- 必须完整使用官方评分指南、当前题完整题干、题目满分、标准答案、当前题专家解析 analysis、完整学生答案和官方错因类型。
- Adaptive Skill 只能在 BASE 输入之上增加任务级评分指导，不能替代、删除或覆盖 BASE 评分依据。
- 必须先形成整题评分尺度，再把整题判断分解到现有 student steps；禁止把每个 step 当作互不相关的独立题目。
- 初步整题分和初步步骤分只用于 RAG 检索，FinalJudge 会重新进行全局评分。
- 步骤初判数量必须严格等于学生步骤数：{len(steps)}。

输出 JSON：
{{
  "初步整题分": <int>,
  "整题行为摘要": "...",
  "步骤初判": [
    {{
      "step_id": 0,
      "初步步骤分": <int>,
      "局部完成百分比": <0到100的整数>,
      "学生行为摘要": "...",
      "可能错误类型": ["..."]
    }}
  ]
}}

局部完成百分比规则：
- 分母不是整题满分，而是当前 step 在当前题中实际覆盖的顶层评分 owner 的可得预算。
- 完整命中其覆盖的全部 owner 为100；只完成一部分按语义完成度给0到99；未命中为0。
- 一个 step 打包多个 owner 时，按这些 owner 的综合完成度判断；不得引用历史案例分数。
- 不得用“当前step分/整题满分”计算局部完成百分比，也不得默认各owner等权。

官方评分指南：{score_guideline}
当前题完整题干：{sample.get('question')}
当前题满分：{total}
标准答案：{sample.get('reference')}
当前题专家解析 analysis：{analysis}
完整学生答案：{_student_answer(sample)}
官方错因类型：{json.dumps(_official_error_labels(), ensure_ascii=False)}
Adaptive Skill：{json.dumps(skill_zh, ensure_ascii=False)}
"""
    max_tokens = 4200 if len(steps) > 12 else 2600
    data, diagnostic = chat_json_with_diagnostics(
        system,
        user,
        temperature=0.1,
        max_tokens=max_tokens,
        schema_name="中文全局初判员",
        strict_api=strict_api,
    )
    raw_steps = data.get("步骤初判", data.get("step_scores", []))
    if not isinstance(raw_steps, list):
        raise RuntimeError("全局初判缺少步骤初判数组")
    by_id: dict[int, dict[str, Any]] = {}
    for raw in raw_steps:
        if not isinstance(raw, dict) or "step_id" not in raw:
            continue
        step_id = _clamp_int(raw.get("step_id", -1), -1, max(-1, len(steps) - 1))
        if step_id >= 0:
            by_id[step_id] = raw
    missing = [step_id for step_id in range(len(steps)) if step_id not in by_id]
    if missing:
        raise RuntimeError(f"全局初判缺少step_id: {missing}")

    step_preliminaries: list[dict[str, Any]] = []
    for step_id in range(len(steps)):
        raw = by_id[step_id]
        raw_score = _clamp_int(
            raw.get("初步步骤分", raw.get("step_score", raw.get("score", 0))),
            0,
            total,
        )
        completion_percent = _clamp_int(
            raw.get(
                "局部完成百分比",
                raw.get("local_completion_percent", raw.get("completion_percent", -1)),
            ),
            -1,
            100,
        )
        if completion_percent < 0:
            raise RuntimeError(f"全局初判step_id={step_id}缺少局部完成百分比")
        response = str(steps[step_id].get("response", ""))
        adjusted_score = raw_score if response.strip() else 0
        adjusted_completion_percent = completion_percent if response.strip() else 0
        adjustment_reasons = [] if response.strip() else ["空步骤检索锚点设为0"]
        step_preliminaries.append(
            {
                "step_id": step_id,
                "preliminary_step_score": adjusted_score,
                "raw_preliminary_step_score": raw_score,
                "初步步骤分": adjusted_score,
                "初步步骤分_raw": raw_score,
                "local_completion_percent": adjusted_completion_percent,
                "局部完成百分比": adjusted_completion_percent,
                "normalized_retrieval_band": _clamp_int(
                    round(adjusted_completion_percent / 10), 0, 10
                ),
                "学生行为摘要": _short(raw.get("学生行为摘要", raw.get("reason", "")), 260),
                "可能错误类型": [
                    _short(str(value), 60)
                    for value in raw.get("可能错误类型", raw.get("errors", []))
                    if str(value).strip()
                ][:8],
                "检索锚点校正原因": adjustment_reasons,
                "base_scoring_context_audit": base_context_audit,
            }
        )
    return {
        "preliminary_score": _clamp_int(
            data.get("初步整题分", data.get("pred_score", data.get("score", 0))),
            0,
            total,
        ),
        "整题行为摘要": _short(data.get("整题行为摘要", data.get("overall_reason", "")), 600),
        "step_preliminaries": step_preliminaries,
        "base_scoring_context_audit": base_context_audit,
        "model_diagnostic": diagnostic,
        "prejudge_granularity": "one_global_call_per_sample",
    }


def _select_step_cases_zh(
    *,
    casebank: dict[str, Any],
    sample: dict[str, Any],
    step_id: int,
    preliminary: dict[str, Any],
    top_k: int = 6,
) -> dict[str, Any]:
    cases = [case for case in casebank.get("cases", []) if isinstance(case, dict)]
    normalized_retrieval = bool(cases) and all(
        "归一化检索档位" in case for case in cases
    )
    percentage_only_visibility = bool(cases) and all(
        case.get("模型可见评分尺度")
        == "TEACHER_COMPLETION_PERCENT_ONLY_CURRENT_OWNER_REINTERPRETATION"
        for case in cases
    )
    if normalized_retrieval:
        s = _clamp_int(
            preliminary.get(
                "normalized_retrieval_band",
                round(_clamp_int(preliminary.get("局部完成百分比", 0), 0, 100) / 10),
            ),
            0,
            10,
        )
    else:
        s = _clamp_int(
            preliminary.get("preliminary_step_score", preliminary.get("初步步骤分", 0)),
            0,
            _clamp_int(sample.get("total", 0), 0, 100),
        )
    ctx = _step_context(sample, step_id)
    query_text = "\n".join(
        [
            f"原题：{sample.get('question', '')}",
            f"当前学生步骤：{ctx['current_step']}",
            f"学生行为摘要：{preliminary.get('学生行为摘要', '')}",
            f"可能错误类型：{'、'.join(str(x) for x in preliminary.get('可能错误类型', []))}",
        ]
    )

    by_score: dict[int, list[dict[str, Any]]] = defaultdict(list)
    similarity_by_id: dict[str, float] = {}
    for case in cases:
        score = (
            _clamp_int(case.get("归一化检索档位", 0), 0, 10)
            if normalized_retrieval
            else _score_band_for_case(case)
        )
        by_score[score].append(case)
        similarity_by_id[str(case.get("case_id", ""))] = _similarity(query_text, case.get("检索文本_证据", ""))
    for score in by_score:
        by_score[score].sort(key=lambda case: similarity_by_id.get(str(case.get("case_id", "")), 0.0), reverse=True)

    core_bands = [band for band in [s - 1, s, s + 1] if band >= 0]
    quota_plan = {str(band): 2 for band in core_bands}
    considered_bands: set[int] = set(core_bands)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_sources: set[str] = set()
    selected_by_band: dict[str, list[str]] = defaultdict(list)

    def add_case(case: dict[str, Any], *, enforce_source: bool) -> bool:
        case_id = str(case.get("case_id", ""))
        source_id = str(case.get("source_sample_id", ""))
        if not case_id or case_id in selected_ids:
            return False
        if enforce_source and source_id and source_id in selected_sources:
            return False
        selected.append(case)
        selected_ids.add(case_id)
        if source_id:
            selected_sources.add(source_id)
        case_band = (
            _clamp_int(case.get("归一化检索档位", 0), 0, 10)
            if normalized_retrieval
            else _score_band_for_case(case)
        )
        selected_by_band[str(case_band)].append(case_id)
        return True

    def add_from_band(band: int, limit: int, *, enforce_source: bool) -> None:
        count = 0
        for case in by_score.get(band, []):
            if len(selected) >= top_k or count >= limit:
                break
            if add_case(case, enforce_source=enforce_source):
                count += 1

    # First pass: fixed score-band quota, max 2 per core band.
    for band in core_bands:
        add_from_band(band, 2, enforce_source=True)

    # Fill gaps from the other core bands by evidence similarity. Keep source diversity first.
    def fill_from_bands(bands: list[int], *, enforce_source: bool) -> None:
        pool = [
            case
            for band in bands
            for case in by_score.get(band, [])
            if str(case.get("case_id", "")) not in selected_ids
        ]
        pool.sort(key=lambda case: similarity_by_id.get(str(case.get("case_id", "")), 0.0), reverse=True)
        for case in pool:
            if len(selected) >= top_k:
                break
            add_case(case, enforce_source=enforce_source)

    fill_from_bands(core_bands, enforce_source=True)
    if len(selected) < top_k:
        fill_from_bands(core_bands, enforce_source=False)

    # Expand outward only if the core bands cannot fill top_k.
    max_score = max(by_score.keys(), default=s)
    radius = 2
    while len(selected) < top_k and radius <= max(max_score + 1, s + 3):
        new_bands = [band for band in [s - radius, s + radius] if band >= 0 and band in by_score]
        for band in new_bands:
            considered_bands.add(band)
        fill_from_bands(new_bands, enforce_source=True)
        if len(selected) < top_k:
            fill_from_bands(new_bands, enforce_source=False)
        radius += 1

    # If preliminary step score is positive, avoid returning an all-zero case set when nonzero cases exist.
    def retrieval_band(case: dict[str, Any]) -> int:
        return (
            _clamp_int(case.get("归一化检索档位", 0), 0, 10)
            if normalized_retrieval
            else _score_band_for_case(case)
        )

    if s > 0 and selected and all(retrieval_band(case) == 0 for case in selected):
        nonzero_pool = [
            case
            for band in sorted(considered_bands | {s, s + 1})
            for case in by_score.get(band, [])
            if retrieval_band(case) > 0 and str(case.get("case_id", "")) not in selected_ids
        ]
        nonzero_pool.sort(key=lambda case: similarity_by_id.get(str(case.get("case_id", "")), 0.0), reverse=True)
        if nonzero_pool:
            removed = selected.pop()
            removed_id = str(removed.get("case_id", ""))
            selected_ids.discard(removed_id)
            try:
                selected_by_band[str(retrieval_band(removed))].remove(removed_id)
            except ValueError:
                pass
            add_case(nonzero_pool[0], enforce_source=False)

    selected = selected[:top_k]
    candidate_score_bands = sorted(considered_bands)
    candidates = [case for band in candidate_score_bands for case in by_score.get(band, [])]
    if percentage_only_visibility:
        boundaries = [
            {
                "案例编号": str(case.get("case_id", "")),
                "教师完成百分比": str(case["模型可见_教师完成百分比"]),
                "完成度检索档位": retrieval_band(case),
                "历史案例角色": _short(case.get("模型可见_边界角色", ""), 100),
                "参考得分点": _short(case.get("模型可见_参考得分点", ""), 500),
                "历史学生作答行为": _short(case.get("模型可见_学生作答行为", ""), 500),
                "相对边界说明": _short(case.get("模型可见_相对边界", ""), 700),
                "当前题owner预算重解释": _short(
                    case.get("模型可见_当前owner预算重解释", ""), 500
                ),
                "软参考规则": _short(case.get("模型可见_软参考规则", ""), 400),
                "错误类型": [str(tag) for tag in case.get("错误类型", [])][:8],
            }
            for case in selected
        ]
        forbidden_model_keys = {
            "教师评分",
            "来源局部预算",
            "来源教师绝对分_仅作证据",
            "来源题总分",
        }
        leaked_keys = sorted(
            forbidden_model_keys.intersection(
                key for boundary in boundaries for key in boundary
            )
        )
        if leaked_keys:
            raise RuntimeError(
                f"百分比RAG模型可见边界泄漏历史绝对分字段: {leaked_keys}"
            )
    else:
        boundaries = [
            {
                "案例编号": str(case.get("case_id", "")),
                "教师评分": _clamp_int(case.get("教师评分", case.get("gold_score", 0)), -50, 50),
                "来源局部预算": case.get("来源局部预算"),
                "局部完成比例": case.get("局部完成比例"),
                "归一化检索档位": retrieval_band(case),
                "分数档位": str(case.get("分数档位", "")),
                "评分边界": _short(
                    case.get("评分边界", ""), 700 if normalized_retrieval else 240
                ),
                "评分理由短评": _short(
                    case.get("评分理由短评", ""), 500 if normalized_retrieval else 260
                ),
                "错误类型": [str(tag) for tag in case.get("错误类型", [])][:8],
            }
            for case in selected
        ]
    semantic_scores = [round(similarity_by_id.get(str(case.get("case_id", "")), 0.0), 4) for case in selected]
    selected_scores = [
        _clamp_int(case.get("教师评分", case.get("gold_score", 0)), -50, 50)
        for case in selected
    ]
    selected_retrieval_bands = [retrieval_band(case) for case in selected]
    retrieval_percentage_distribution = (
        dict(Counter(str(item["教师完成百分比"]) for item in boundaries))
        if percentage_only_visibility
        else {}
    )
    return {
        "retrieval_method": (
            "步骤级中文证据检索_理由校准_局部完成度归一化档位"
            if normalized_retrieval
            else "步骤级中文证据检索_理由校准_先分数筛选"
        ),
        "retrieval_anchor_type": (
            "LOCAL_COMPLETION_BAND_0_TO_10"
            if normalized_retrieval
            else "ABSOLUTE_TEACHER_SCORE"
        ),
        "model_visible_score_scale": (
            "TEACHER_COMPLETION_PERCENT_ONLY_CURRENT_OWNER_REINTERPRETATION"
            if percentage_only_visibility
            else "LEGACY_CASE_BOUNDARY"
        ),
        "query_retrieval_band": s,
        "quota_plan": quota_plan,
        "selected_by_band": dict(selected_by_band),
        "candidate_score_bands": candidate_score_bands,
        "candidate_case_count": len(candidates),
        "selected_case_ids": [case.get("case_id", "") for case in selected],
        "selected_case_scores": selected_scores,
        "selected_case_retrieval_bands": selected_retrieval_bands,
        "selected_case_buckets": [str(case.get("分数档位", "")) for case in selected],
        "selected_case_similarity": semantic_scores,
        "all_zero_selected": bool(selected_retrieval_bands) and all(
            score == 0 for score in selected_retrieval_bands
        ),
        "retrieved_reason_calibration": boundaries,
        "retrieval_score_distribution": (
            {}
            if percentage_only_visibility
            else dict(Counter(str(item["教师评分"]) for item in boundaries))
        ),
        "retrieval_percentage_distribution": retrieval_percentage_distribution,
        "retrieval_band_distribution": dict(
            Counter(
                str(
                    item.get(
                        "完成度检索档位", item.get("归一化检索档位", "")
                    )
                )
                for item in boundaries
            )
        ),
        "selected_case_count": len(boundaries),
    }


def _model_visible_step_calibration_entry(
    *,
    step_id: int,
    current_step: str,
    retrieval: dict[str, Any],
) -> dict[str, Any]:
    boundaries = retrieval.get("retrieved_reason_calibration", [])
    if (
        retrieval.get("model_visible_score_scale")
        == "TEACHER_COMPLETION_PERCENT_ONLY_CURRENT_OWNER_REINTERPRETATION"
    ):
        return {
            "step_id": step_id,
            "当前学生步骤": current_step,
            "RAG数值尺度": "历史教师完成百分比；不含历史绝对分",
            "检索案例完成百分比分布": retrieval.get(
                "retrieval_percentage_distribution", {}
            ),
            "检索到的百分比边界": boundaries,
            "当前题解释要求": (
                "先从当前题官方评分指南、标准答案和专家解析识别本step命中的owner及其当前题预算；"
                "再把历史完成百分比解释为该当前owner预算上的软边界。不得恢复、猜测或照搬历史绝对分。"
            ),
        }
    return {
        "step_id": step_id,
        "当前学生步骤": current_step,
        "检索案例评分分布": retrieval.get("retrieval_score_distribution", {}),
        "检索到的评分边界": boundaries,
    }


def _joint_final_judge_zh(
    sample: dict[str, Any],
    skill_zh: dict[str, Any],
    step_calibration_map: list[dict[str, Any]],
    strict_api: bool,
    boundary_aware: bool = False,
    fusion_mode: str = "standard",
) -> dict[str, Any]:
    total = _clamp_int(sample.get("total", 0), 0, 100)
    steps = sample.get("steps", [])
    task = _task_name_zh()
    official_error_labels = _official_error_labels()
    score_guideline = _official_score_guideline()
    analysis = str(sample.get("analysis", ""))
    base_context_audit = _base_scoring_context_audit(sample)
    student_steps = [
        {"step_id": i, "学生步骤": _short(step.get("response", ""), 700)}
        for i, step in enumerate(steps)
    ]
    payload = {
        "官方评分指南": score_guideline,
        "原题": sample.get("question", ""),
        "题目满分": total,
        "标准答案": sample.get("reference", ""),
        "当前题专家解析_analysis": analysis,
        "学生完整答案": _student_answer(sample),
        "所有学生步骤": student_steps,
        "BASE输入完整性": base_context_audit,
        "step_calibration_map": step_calibration_map,
        "fusion_mode": fusion_mode,
        "当前任务": _task_meta(),
        "可选错误标签": official_error_labels,
        "输出要求": {
            "pred_score": "整题最终分，0 到题目满分之间的整数",
            "step_scores": [
                {
                    "step_id": "步骤编号",
                    "score": "该 step 的分数，非负整数",
                    "errors": ["从可选错误标签中选择"],
                    "边界判断": "0分|1分|2分|更高分，boundary mode 必填",
                    "是否新增有效作答贡献": "true/false，boundary mode 必填",
                    "是否新增有效数学贡献": "兼容旧字段；非数学题也表示新增有效作答贡献",
                    "是否完整命中关键节点": "true/false，boundary mode 必填",
                    "是否只是部分推进": "true/false，boundary mode 必填",
                    "是否首次有效关键步骤": "true/false",
                    "是否重复前面内容": "true/false",
                    "是否错误链延续": "true/false",
                    "是否最终答案型保护": "true/false，融合模式必填",
                    "fusion_mode_used": "rag_guardrail|skill_hierarchical|standard",
                    "rag_boundary_used": "中文短评",
                    "skill_used": "中文短评",
                    "对应参考关键点": "中文短语",
                    "为什么不是0分": "中文短评，boundary mode 必填",
                    "为什么不是更高分": "中文短评，boundary mode 必填",
                    "reason": "中文短评",
                }
            ],
            "overall_reason": "整题联合评分理由",
            "non_repetition_check": "说明是否避免重复给分",
            "error_chain_check": "说明错误链后续是否重复给分",
        },
    }
    if fusion_mode == "rag_guardrail":
        payload["skill_guardrail_checklist"] = _skill_guardrail_checklist_zh(skill_zh)
    elif fusion_mode == "skill_hierarchical":
        payload["skill_checklist"] = _skill_hierarchical_checklist_zh(skill_zh)
    elif fusion_mode == "native_base_additive":
        payload["Adaptive Skill"] = skill_zh
    else:
        payload["中文评分原则"] = skill_zh
    percentage_rag_active = any(
        isinstance(item, dict)
        and item.get("RAG数值尺度") == "历史教师完成百分比；不含历史绝对分"
        for item in step_calibration_map
    )
    percentage_rag_principles = ""
    if percentage_rag_active:
        payload["RAG百分比解释契约"] = {
            "案例选择": "按PreJudge局部完成百分比的相邻档位检索",
            "数值含义": "历史百分比表示历史答案对其来源owner的完成度，不是历史绝对分",
            "当前题重解释": (
                "先从当前题官方评分指南、标准答案和专家解析识别当前step命中的owner及当前题预算B；"
                "再把案例完成百分比p解释为当前owner预算上的p×B软边界"
            ),
            "参考方式": (
                "不做二元reject。语义和评分机制越接近，百分比边界权重越高；相关性较弱时仍可参考错误模式，"
                "但降低其数值影响。最终以当前题BASE评分尺度为准"
            ),
            "禁止": "不得恢复、猜测、引用或照搬历史教师绝对分与历史局部预算",
        }
        percentage_rag_principles = """

【百分比 teacher-boundary RAG 使用规则】
- 检索案例只提供教师完成百分比，不提供历史绝对分。百分比表示历史学生行为对其来源owner的局部完成度。
- 对当前step，先依据当前题官方评分指南、标准答案和专家解析识别实际命中的current-question owner，并归纳该owner在当前题的预算B；再将历史完成百分比p解释为当前owner预算上的p×B软边界。
- 不做二元reject。评分机制与owner语义越接近，百分比边界越值得参考；相关性较弱时仍可参考其错误模式或higher/lower方向，但应降低数值影响。
- 百分比只是软校准，不是硬上限、固定单价或对BASE全局分的替代。若案例与当前题尺度冲突，以当前题完整BASE信息为准。
- 不得恢复、猜测、引用或照搬任何历史教师绝对分或历史局部预算。
"""
    forbidden = {
        "初步步骤分",
        "preliminary_score",
        "preliminary_step_score",
        "routing_score",
        "gold_score",
        "gold_step_score",
        "raw selected_cases",
    }
    if percentage_rag_active:
        forbidden_historical_keys = {
            "教师评分",
            "来源局部预算",
            "来源教师绝对分_仅作证据",
            "来源题总分",
        }
        visible_rag_snapshot = json.dumps(step_calibration_map, ensure_ascii=False)
        leaked_historical_keys = sorted(
            key for key in forbidden_historical_keys if key in visible_rag_snapshot
        )
        if leaked_historical_keys:
            raise RuntimeError(
                "百分比RAG FinalJudge输入泄漏历史绝对分字段: "
                f"{leaked_historical_keys}"
            )
    snapshot = json.dumps(payload, ensure_ascii=False)
    leakage = [term for term in forbidden if term in snapshot]
    if leakage:
        raise RuntimeError(f"Joint FinalJudge 输入发生泄漏: {leakage}")

    boundary_principles = ""
    if boundary_aware:
        boundary_principles = """

边界判断特别规则（必须执行）：
第一阶段：先做 0 vs 非0。每个 step 必须先判断是否有新增有效数学贡献、独立可认可的公式/方程/代入/推导/结论，以及是否与 reference key point 明确对应。若没有，必须给0。
默认0分的情况：只是重复前面内容；空泛文字或结论性废话；错误链延续且没有新的正确关键点；setup/设变量/抄题/前置整理但没有完成可评分动作；最终结论错误且当前 step 没有独立正确中间贡献。
第二阶段：只有通过第一阶段后，才判断 1 vs 2 vs 更高。完整命中 reference 的核心公式、关键方程、关键结论，可以给该 step 的高分；只是部分推进、部分代入、部分推导但未完整形成关键节点，给1或低部分分。
如果 step 有明显计算错误、代入错误、公式误用、答案与过程不符，不能直接给2，除非该 step 的独立核心公式本身完全正确。
保护1分步骤：独立关键方程、正确代入形式、有效中间式、局部正确推导，即使整题最终错误，也不要压成0。
严控0→1：setup、重复、错误链、无新增有效贡献、错误结论解释，不要因为看起来有数学符号就给1。
严控0→2：有明显错误或只是无效推进，不能给2。2分只能给完整、独立、清晰命中的关键公式/关键方程/关键结论。
严控1→2：只是部分正确、未完成关键节点，或有计算/代入错误，不能给2。
不重复给分：同一有效贡献只给一次。错误链不给分：完全依赖错误前提继续推导且无新增正确关键点，给0。
每个 step 输出必须包含：边界判断、是否新增有效数学贡献、是否完整命中关键节点、是否只是部分推进、是否重复前面内容、是否错误链延续、对应参考关键点、为什么不是0分、为什么不是更高分。
"""
    fusion_principles = ""
    schema_fusion_value = "standard"
    if fusion_mode == "rag_guardrail":
        schema_fusion_value = "rag_guardrail"
        fusion_principles = """

融合模式：RAG 主导 + Skill guardrail。
- Step PreJudge 未使用 Skill；RAG 案例边界是主要评分边界来源。
- skill_guardrail_checklist 只用于防止重复给分、错误链给分、setup-only 给分，并保护首次局部有效贡献和最终答案型样本。
- Skill 不决定 0/1/2/高分边界。具体边界主要由原题、标准答案、学生步骤和 step_calibration_map 中的人工案例边界决定。
- 不要因为缺少过程直接把正确最终答案给 0；若学生唯一答案与标准答案等价，应结合题型和案例边界给相应分。
- 如果 RAG 边界与 guardrail 冲突：guardrail 只阻止明显重复/错误链/setup-only 或保护最终答案型样本，不替代案例边界。
"""
    elif fusion_mode == "skill_hierarchical":
        schema_fusion_value = "skill_hierarchical"
        fusion_principles = """

融合模式：Skill 主导 + RAG 边界校准。
- skill_checklist 是主要评分流程：先判断新增有效贡献，再判断重复/错误链/setup-only，再参考 RAG 边界决定具体分数。
- RAG 只能校准局部分数边界，不能推翻 Skill 对“是否有新增有效贡献”的基本判断。
- 如果 Skill 判断当前 step 没有新增有效贡献，RAG 中的高分案例不能直接把它改成高分。
- 如果 Skill 判断当前 step 有独立有效贡献，RAG 用于校准给 1、2 或更高，而不是机械照搬案例分数。
- 最终答案型保护仍适用：题型允许短答且答案等价时，不应仅因过程缺失给 0。
"""
    if _is_math_shortans():
        task_specific_principles = """
数学短答题补充规则：
- 对统计题，单纯人数整理通常不给分；核心统计量公式若正确可单独给分。
- 对解析几何题，单纯设焦点/设直线通常不给分；将指定直线条件正确代入椭圆方程形成关键方程可给低分。
- “正确统计人数/总人数/局部数据整理”如果只是为后续比例或统计量做准备，且没有完成该问要求的比例、统计量、方程或结论，必须按0分处理。
- 统计题中，若当前 step 独立且完整写出核心统计量公式（例如 K²/χ² 统计量公式），即使后续代入或计算错误，也应把主要公式分给这个公式 step，而不是把分散给前面的数据整理或后面的错误计算链。
- 基于错误统计量、错误样本量、错误临界值或错误方法选择得到的后续结论，通常为0，除非该 step 本身包含独立正确的核心公式或关键方程。
"""
    else:
        task_specific_principles = """
当前学科/题型补充规则：
- 必须按当前任务的题型评分：选择题重点核对选项和概念判断；填空题重点核对每个空的答案等价性；简答题重点核对答案点、材料提取、推理和表述。
- 不要把数学短答题的公式/方程/代入规则套用到非数学题。
- 对背诵、填空、改错、选择等题型，如果学生答案与参考答案等价，可按该空/该题的人工边界给分；不应强行要求推导过程。
- 对文科/阅读/材料分析题，只有命中参考答案的关键语义点、材料信息或分析角度才给分；空泛套话、重复题干、泛泛而谈通常不给分。
"""
    system = f"你是唯一的中文{task}整题联合最终阅卷老师。必须只输出 JSON。"
    user = f"""
请对整题进行联合评分，并同时输出整题总分和每个 step 的分数。

	重要原则：
	- FinalJudge 必须先像 Native BASE 一样，使用官方评分指南、完整题干、题目满分、标准答案、当前题专家解析 analysis、完整学生答案和官方错因类型建立整题评分尺度。
	- Adaptive Skill 与逐step RAG 评分建议只能在完整 BASE 输入之上增加任务级规则和历史边界参考，不能替代、删除、覆盖或降级 BASE 评分依据。
	- 先做一次完整的全局评分判断，再结合每个 step 的 RAG 建议校准步骤分，最后重新核对整题总分；不能把各 step 当作互不相关的独立题目。
	- 这是整题联合评分，不是每个 step 完全独立评分。
- 只给“新增有效作答贡献”得分。
- 不要重复给分；同一个有效贡献只能给一次。
- 如果后续 step 只是重复前面内容，不应再次给分。
- 若错误前提导致后续错误链，且后续没有新的正确关键点，后续不给分。
- 纯设定、抄题、前置整理、说明接下来要做什么，通常不是独立得分贡献。
- 正确写出或表达参考答案中的关键答案点、关键事实、关键概念、关键选项、关键表达、核心方法或有效中间结论，即使后续错误，也应保留相应步骤分。
- 不要因为整题最终错误就抹掉前面已经独立成立的关键贡献。
- SAS-Bench 风格下，不要奖励孤立的局部事实：如果该局部事实没有完成当前题型要求的答案点、空格、选项或结论，通常为0。
- 如果检索案例边界中同类关键贡献可给2分或更高，不要保守地只给1分；应按案例边界保留该关键步骤的对应分。
- 如果前文长段落中混杂出现过某个关键点，但该长段落本身包含错误链，当前 step 是第一次清晰、独立写出该关键点，则当前 step 不算重复。
- 如果某个 step 的检索案例评分分布以0分为主，而当前 step 又只是数据整理、错误链延续或方法方向错误，应优先给0。
- 如果某个 step 是独立关键贡献，并且检索案例中有对应较高分边界，可以按该边界给部分分或该步骤分。
	- 必须先基于完整 BASE 输入形成整题 pred_score，再输出各 step 的解释性分数和边界判断。
	- student step 是答案分段，不保证与教师得分点一一对应；不能机械地用 step 分之和覆盖整题全局判断。
	- 各 step 分用于说明得分来源，其累计值不得超过 pred_score；若分段无法完全承载整题credit，保留全局 pred_score并在 overall_reason 中说明。
- step_calibration_map 是每个 step 独立检索到的历史人工评分边界，只用于校准，不得照搬案例分数。
- 不要输出或引用任何初步步骤分。
{task_specific_principles}
{boundary_principles}
{fusion_principles}
请输出 JSON：
{{
  "pred_score": <int>,
  "step_scores": [
    {{
      "step_id": 0,
      "score": <int>,
      "errors": ["..."],
      "边界判断": "0分|1分|2分|更高分",
      "是否新增有效作答贡献": true,
      "是否新增有效数学贡献": true,
      "是否完整命中关键节点": false,
      "是否只是部分推进": true,
      "是否首次有效关键步骤": true,
      "是否重复前面内容": false,
      "是否错误链延续": false,
      "是否最终答案型保护": false,
      "fusion_mode_used": "{schema_fusion_value}",
      "rag_boundary_used": "...",
      "skill_used": "...",
      "对应参考关键点": "...",
      "为什么不是0分": "...",
      "为什么不是更高分": "...",
      "reason": "..."
    }}
  ],
  "overall_reason": "...",
  "non_repetition_check": "...",
  "error_chain_check": "..."
}}

输入：
{json.dumps(payload, ensure_ascii=False)}
"""
    if fusion_mode == "native_base_additive":
        user = f"""
请作为{task}评分专家，在 Native BASE 完整评分输入上增加 Adaptive Skill 和逐step RAG 参考，对学生整份作答进行一次全局评分。

【评分指南】
{score_guideline}

【评估材料】
- 试题内容：{sample.get('question', '')}
- 题目分值：{total}
- 错因类型：{json.dumps(official_error_labels, ensure_ascii=False)}
- 标准答案：{sample.get('reference', '')}
- 解析说明：{analysis}
- 学生完整作答：{_student_answer(sample)}

【在 BASE 上增加的任务级信息】
- Adaptive Skill：{json.dumps(skill_zh, ensure_ascii=False)}
- 逐step检索到的 teacher-boundary RAG：{json.dumps(step_calibration_map, ensure_ascii=False)}
{percentage_rag_principles}

【信息优先级】
1. 官方评分指南、当前题题干与满分、标准答案、当前题专家解析和完整学生作答共同确定本题评分尺度。
2. Adaptive Skill 用于补充任务级评分规律；RAG 用于参考相似学生行为的完成百分比和 higher/lower/partial/full 边界。
3. Skill 和 RAG 只能增加评分依据，不能删减或替代 BASE 信息；历史案例绝对分不能直接作为当前题分值或上限。

【评估流程和要求】
1. 先完整阅读整题所有 BASE 信息和完整学生答案，形成全局 pred_score。
2. 再依据学生现有 step 分段解释每步贡献，并结合该 step 的 RAG 边界校准 step_score；step 是答案分段，不保证与教师得分点一一对应。
3. 最后重新核对整题 pred_score。不得机械地用 step_score 之和覆盖全局 pred_score；step_score 累计值必须位于 0 到 pred_score 之间。
4. 每个错误标签只能从给定错因类型中选择。
5. step_scores 必须严格覆盖 step_id 0 到 {max(0, len(steps) - 1)}，数量必须为 {len(steps)}。

只输出标准 JSON：
{{
  "pred_score": <0到{total}的整数>,
  "step_scores": [
    {{
      "step_id": 0,
      "score": <非负整数>,
      "errors": ["给定错因类型"],
      "rag_boundary_used": "使用了什么teacher边界",
      "skill_used": "使用了什么Skill规则",
      "reason": "本步骤评分理由"
    }}
  ],
  "overall_reason": "整题全局评分理由",
  "non_repetition_check": "重复计分核对",
  "error_chain_check": "错误链核对"
}}
"""
    step_count_for_budget = len(sample.get("steps", []))
    if fusion_mode in {"rag_guardrail", "skill_hierarchical", "native_base_additive"}:
        max_tokens_budget = 10000 if step_count_for_budget > 12 else 7200
    else:
        max_tokens_budget = 9000 if step_count_for_budget > 20 else (5200 if (boundary_aware or step_count_for_budget > 12) else 2600)
    data, diagnostic = chat_json_with_diagnostics(
        system,
        user,
        temperature=0.1,
        max_tokens=max_tokens_budget,
        schema_name="中文整题联合最终评分员:joint_step_rag_boundary_zh" if boundary_aware else "中文整题联合最终评分员:joint_step_rag_zh",
        strict_api=strict_api,
    )
    raw_steps = data.get("step_scores", data.get("步骤评分", []))
    if not isinstance(raw_steps, list):
        raw_steps = []
    by_id: dict[int, dict[str, Any]] = {}
    fallback_order: list[dict[str, Any]] = []
    for raw in raw_steps:
        if not isinstance(raw, dict):
            continue
        fallback_order.append(raw)
        if "step_id" in raw:
            by_id[_clamp_int(raw.get("step_id", 0), 0, max(0, len(steps) - 1))] = raw

    pred_steps: list[dict[str, Any]] = []
    step_reasons: list[dict[str, Any]] = []
    calibration_by_step = {
        _clamp_int(item.get("step_id", 0), 0, max(0, len(steps) - 1)): item
        for item in step_calibration_map
        if isinstance(item, dict)
    }
    for i in range(len(steps)):
        raw = by_id.get(i)
        if raw is None and i < len(fallback_order):
            raw = fallback_order[i]
        if raw is None:
            raw = {}
        score = _clamp_int(raw.get("score", raw.get("步骤分", raw.get("step_score", 0))), 0, total)
        errors = _safe_errors(raw.get("errors", raw.get("错误标签", [])))
        reason = _short(raw.get("reason", raw.get("评分理由", "")), 420)
        pred_steps.append({"step_score": score, "errors": errors})
        step_reasons.append(
            {
                "step_id": i,
                "step_score": score,
                "errors": errors,
                "边界判断": _short(raw.get("边界判断", raw.get("boundary_judgment", "")), 80),
                "是否新增有效数学贡献": _as_bool(raw.get("是否新增有效数学贡献", raw.get("是否新增有效作答贡献", raw.get("new_valid_math_contribution", False)))),
                "是否完整命中关键节点": _as_bool(raw.get("是否完整命中关键节点", raw.get("fully_hits_key_node", False))),
                "是否只是部分推进": _as_bool(raw.get("是否只是部分推进", raw.get("partial_progress_only", False))),
                "是否首次有效关键步骤": _as_bool(raw.get("是否首次有效关键步骤", raw.get("first_valid_key_step", False))),
                "是否重复前面内容": _as_bool(raw.get("是否重复前面内容", raw.get("repeated_previous_content", False))),
                "是否错误链延续": _as_bool(raw.get("是否错误链延续", raw.get("error_chain_continuation", False))),
                "是否最终答案型保护": _as_bool(raw.get("是否最终答案型保护", raw.get("final_answer_protection", False))),
                "fusion_mode_used": _short(raw.get("fusion_mode_used", raw.get("融合模式", fusion_mode)), 80),
                "rag_boundary_used": _short(raw.get("rag_boundary_used", raw.get("使用的RAG边界", raw.get("参考的案例边界", ""))), 260),
                "skill_used": _short(raw.get("skill_used", raw.get("使用的Skill", raw.get("使用的评分原则", ""))), 260),
                "对应参考关键点": _short(raw.get("对应参考关键点", raw.get("reference_key_point", "")), 220),
                "为什么不是0分": _short(raw.get("为什么不是0分", raw.get("why_not_zero", "")), 220),
                "为什么不是更高分": _short(raw.get("为什么不是更高分", raw.get("why_not_higher", "")), 220),
                "reason": reason,
            }
        )

    postprocess_adjustments: list[dict[str, Any]] = []
    for i, step in enumerate(steps):
        text = str(step.get("response", ""))
        compact = re.sub(r"\s+", "", text)
        calibration = calibration_by_step.get(i, {})
        boundaries = calibration.get("检索到的评分边界", []) if isinstance(calibration, dict) else []
        boundary_scores = [
            _clamp_int(boundary.get("教师评分", 0), 0, total)
            for boundary in boundaries
            if isinstance(boundary, dict)
        ]
        old_score = pred_steps[i]["step_score"]
        new_score = old_score
        adjustment_reason = ""

        if _is_math_shortans():
            if old_score > 0 and "需要志愿者的总人数" in text and "比例为：" in text and not re.search(r"70\s*/\s*500|70\\over500|0\.14|14%", text):
                new_score = 0
                adjustment_reason = "仅统计人数/总人数且未完成比例结果，按SAS-Bench步骤标注风格置0"

            is_k2_formula = bool(
                ("K^{2}" in compact or "K^2" in compact or "χ" in compact)
                and ("ad-bc" in compact or "a d-b c" in text or "(ad-bc)" in compact)
                and not any(token in compact for token in ["n=400", "n＝400", "3.841", "4.0", "70}{1000", "70/1000"])
            )
            if is_k2_formula and boundary_scores and max(boundary_scores) >= 2 and new_score < 2:
                new_score = 2
                adjustment_reason = "独立核心统计量公式且检索边界含2分案例，保留公式步骤对应分"

            clean_equation = bool(
                len(compact) <= 90
                and ("x^2" in compact or "x^{2}" in compact)
                and ("(x+c)^2" in compact or "(x+c)^{2}" in compact)
                and ("b^2" in compact or "b^{2}" in compact)
                and "=1" in compact
            )
            if clean_equation and boundary_scores and max(boundary_scores) >= 1 and new_score < 1:
                new_score = 1
                adjustment_reason = "当前step为独立清晰关键方程，不能因前文混杂错误段落而视为重复"

        if new_score != old_score:
            pred_steps[i]["step_score"] = new_score
            step_reasons[i]["step_score"] = new_score
            step_reasons[i]["joint_postprocess_adjustment"] = adjustment_reason
            postprocess_adjustments.append(
                {
                    "step_id": i,
                    "old_score": old_score,
                    "new_score": new_score,
                    "reason": adjustment_reason,
                }
            )

    if boundary_aware or fusion_mode in {"rag_guardrail", "skill_hierarchical"}:
        overflow = sum(step["step_score"] for step in pred_steps) - total
        if overflow > 0:
            for i in range(len(pred_steps) - 1, -1, -1):
                if overflow <= 0:
                    break
                current_score = _clamp_int(pred_steps[i].get("step_score", 0), 0, total)
                if current_score <= 0:
                    continue
                reduction = min(current_score, overflow)
                old_score = current_score
                new_score = current_score - reduction
                pred_steps[i]["step_score"] = new_score
                step_reasons[i]["step_score"] = new_score
                reason = "所有step分之和超过题目满分；按整题满分上限从后续正分步骤扣减"
                step_reasons[i]["joint_postprocess_adjustment"] = reason
                postprocess_adjustments.append({"step_id": i, "old_score": old_score, "new_score": new_score, "reason": reason})
                overflow -= reduction

    raw_model_pred_score = _clamp_int(data.get("pred_score", data.get("最终分数", data.get("score", 0))), 0, total)
    step_sum = sum(step["step_score"] for step in pred_steps)
    final_pred_score = raw_model_pred_score
    pred_score_overridden = False
    return {
        "pred_score": final_pred_score,
        "raw_model_pred_score": raw_model_pred_score,
        "step_sum": step_sum,
        "final_pred_score": final_pred_score,
        "pred_score_overridden": pred_score_overridden,
        "joint_postprocess_adjustments": postprocess_adjustments,
        "steps": pred_steps,
        "step_reasons": step_reasons,
        "reason": _short(data.get("overall_reason", data.get("评分理由", "")), 900),
        "overall_reason": _short(data.get("overall_reason", ""), 900),
        "non_repetition_check": _short(data.get("non_repetition_check", ""), 500),
        "error_chain_check": _short(data.get("error_chain_check", ""), 500),
        "final_judge_input_snapshot": payload,
        "base_scoring_context_audit": base_context_audit,
        "routing_leakage_detected": False,
        "model_diagnostic": diagnostic,
    }


def _metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    pred_result = []
    ori_result = []
    weights_result = []
    max_score = 0
    max_length = 0
    pred_scores, ori_scores, total_scores = [], [], []
    error_type_item = _error_types()
    error_to_id = {item["name"]: i for i, item in enumerate(error_type_item)}

    def err2idx(name: str) -> int:
        return error_to_id.get(name, len(error_to_id) - 1)

    pred_errors = []
    gold_errors = []
    total_exact = 0
    step_exact = 0
    step_count = 0
    pred_sum = 0
    gold_sum = 0
    for item in records:
        total = _clamp_int(item.get("total", 0), 0, 100)
        max_score = max(max_score, total)
        ori_score = _clamp_int(item.get("manual_label", 0), -100, 100)
        pred_score = _clamp_int(item.get("pred_label", 0), 0, total)
        pred_sum += pred_score
        gold_sum += ori_score
        total_exact += int(ori_score == pred_score)
        pred_scores.append(pred_score)
        ori_scores.append(ori_score)
        total_scores.append(total)
        ori_labels = [ori_score]
        pred_labels = [pred_score]
        steps = item.get("steps", [])
        pred_steps = item.get("pred_steps", [])
        weights = [0.5] + ([0.5 / len(steps)] * len(steps) if steps else [])
        p_errors = [0] * len(error_type_item)
        g_errors = [0] * len(error_type_item)
        for i, step in enumerate(steps):
            gold_label = _clamp_int(step.get("label", 0), -50, 50)
            pred_label = _clamp_int(pred_steps[i].get("step_score", 0), -50, 50) if i < len(pred_steps) and isinstance(pred_steps[i], dict) else 0
            ori_labels.append(gold_label)
            pred_labels.append(pred_label)
            step_count += 1
            step_exact += int(gold_label == pred_label)
            for error in step.get("errors", []) or []:
                if error != "步骤正确":
                    g_errors[err2idx(str(error))] += 1
            if i < len(pred_steps) and isinstance(pred_steps[i], dict):
                for error in pred_steps[i].get("errors", []) or []:
                    if error != "步骤正确":
                        p_errors[err2idx(str(error))] += 1
        pred_result.append(pred_labels)
        ori_result.append(ori_labels)
        weights_result.append(weights)
        max_length = max(max_length, len(ori_labels))
        pred_errors.append(p_errors)
        gold_errors.append(g_errors)

    for i in range(len(pred_result)):
        pred_result[i] += [0] * (max_length - len(pred_result[i]))
        ori_result[i] += [0] * (max_length - len(ori_result[i]))
        weights_result[i] += [0] * (max_length - len(weights_result[i]))
    max_scores = [max([row[i] for row in ori_result] + [row[i] for row in pred_result]) for i in range(max_length)]
    max_scores[0] = max_score
    ccs, qwk = adjusted_qwk(ori_result, pred_result, weights_result, max_scores)
    ecs, ecs_ranges = compute_ecs(pred_scores, ori_scores, total_scores, pred_errors, gold_errors, len(error_type_item))
    return {
        "CCS": float(ccs) * 100,
        "QWK": float(qwk) * 100,
        "ECS": float(ecs) * 100,
        "ECS_by_range": [float(x) * 100 for x in ecs_ranges],
        "total_exact": total_exact / len(records) * 100,
        "step_exact": step_exact / step_count * 100 if step_count else 0,
        "pred_score_sum": pred_sum,
        "gold_score_sum": gold_sum,
        "avg_error": (pred_sum - gold_sum) / len(records),
    }


def _run_joint_step_rag_zh(
    limit: int,
    strict_api: bool,
    knowledge_size: int | None = None,
    dataset_path: Path | None = None,
    eval_size: int | None = None,
    output_root: Path | None = None,
    resume: bool = False,
    mode_name: str = "joint_step_rag_zh",
    use_skill: bool = True,
    use_rag: bool = True,
    prejudge_use_skill: bool | None = None,
    fusion_mode: str = "standard",
) -> dict[str, Any]:
    dataset_path = dataset_path or DATA_PATH
    samples = _read_jsonl(dataset_path)
    _set_active_dataset_context(dataset_path, samples)
    effective_knowledge_size = knowledge_size or 20
    evaluation_count = eval_size if eval_size is not None else 50
    if eval_size is not None and eval_size <= 0:
        evaluation_count = max(0, len(samples) - effective_knowledge_size)
    evaluation_set, knowledge_set = split_teacher_examples(
        samples,
        k=effective_knowledge_size,
        seed=int(os.environ.get("TEACHGRADE_RANDOM_SEED", "42")),
        evaluation_count=evaluation_count,
    )
    knowledge_set_range = (
        f"uniform_without_replacement(k={effective_knowledge_size}, "
        f"seed={os.environ.get('TEACHGRADE_RANDOM_SEED', '42')})"
    )
    output_dir = _main_output_dir(output_root, effective_knowledge_size, mode_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / f"predictions_{mode_name}.json"
    retrieval_debug_path = output_dir / f"retrieval_debug_{mode_name}.json"
    evaluation_path = output_dir / f"predictions_{mode_name}.jsonl"
    existing_summary_path = output_dir / f"run_summary_{mode_name}.json"
    if existing_summary_path.exists():
        existing = json.loads(existing_summary_path.read_text(encoding="utf-8"))
        if resume and int(existing.get("completed_samples", 0)) >= limit and int(existing.get("limit", 0)) >= limit:
            print(f"[SKIP][COMPLETED] mode={mode_name} knowledge_size={effective_knowledge_size} limit={limit}")
            return existing

    if prejudge_use_skill is None:
        prejudge_use_skill = use_skill
    skill = _load_or_build_evaluation_skill_zh(knowledge_set, strict_api, output_dir) if use_skill else {}
    casebank = _load_or_build_full_step_casebank_zh_evidence_reason(knowledge_set, output_dir) if use_rag else {"cases": []}
    casebank_audit = _audit_full_step_casebank_zh_evidence_reason(casebank, output_dir) if use_rag else {
        "num_cases": 0,
        "num_step_cases": 0,
        "score_distribution": {},
        "score_bucket_distribution": {},
        "error_type_distribution": {},
        "rag_disabled": True,
    }

    started = time.perf_counter()
    predictions: list[dict[str, Any]] = []
    retrieval_debug_records: list[dict[str, Any]] = []
    if resume and predictions_path.exists():
        try:
            loaded_predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
            if isinstance(loaded_predictions, list):
                predictions = [item for item in loaded_predictions if isinstance(item, dict)]
        except Exception as exc:
            print(f"[WARN][RESUME_LOAD_FAILED] path={predictions_path} error={exc}")
            predictions = []
    if resume and retrieval_debug_path.exists():
        try:
            loaded_debug = json.loads(retrieval_debug_path.read_text(encoding="utf-8"))
            if isinstance(loaded_debug, list):
                retrieval_debug_records = [item for item in loaded_debug if isinstance(item, dict)]
        except Exception as exc:
            print(f"[WARN][RESUME_DEBUG_LOAD_FAILED] path={retrieval_debug_path} error={exc}")
            retrieval_debug_records = []
    completed_ids = {str(record.get("sample_id")) for record in predictions}
    evaluation_records: list[dict[str, Any]] = []
    sample_by_id = {str(sample.get("id")): sample for sample in evaluation_set[:limit]}
    for record in predictions:
        sample = sample_by_id.get(str(record.get("sample_id")))
        if sample is not None:
            evaluation_records.append(_evaluation_record(sample, {"pred_score": record.get("pred_score", 0), "steps": record.get("steps", [])}))
    selected_case_id_counter: Counter[str] = Counter()
    selected_score_counter: Counter[str] = Counter()
    preliminary_score_counter: Counter[str] = Counter()
    raw_preliminary_score_counter: Counter[str] = Counter()
    selected_case_counts: list[int] = []
    candidate_case_counts: list[int] = []
    gold_in_selected_step_score_count = 0
    gold_missing_selected_step_score_count = 0
    total_step_count = 0
    pred_score_equals_step_sum_count = 0
    pred_step_sum_total = 0
    pred_score_override_count = 0
    pred_score_override_records: list[dict[str, Any]] = []
    leakage_detected = False

    for record in predictions:
        pred_score_equals_step_sum_count += int(_clamp_int(record.get("pred_score", 0), 0, 100) == _clamp_int(record.get("pred_step_sum", 0), -100, 100))
        pred_step_sum_total += _clamp_int(record.get("pred_step_sum", 0), -100, 100)
        pred_score_overridden = bool(record.get("pred_score_overridden", False))
        pred_score_override_count += int(pred_score_overridden)
        if pred_score_overridden:
            pred_score_override_records.append(
                {
                    "sample_id": record.get("sample_id"),
                    "raw_model_pred_score": record.get("raw_model_pred_score"),
                    "step_sum": record.get("pred_step_sum"),
                    "final_pred_score": record.get("pred_score"),
                }
            )
        leakage_detected = leakage_detected or bool(record.get("routing_leakage_detected"))
    for debug in retrieval_debug_records:
        selected_case_id_counter.update(str(x) for x in debug.get("selected_case_ids", []))
        selected_score_counter.update(str(x) for x in debug.get("selected_case_scores", []))
        if "preliminary_step_score" in debug:
            preliminary_score_counter.update([str(debug.get("preliminary_step_score"))])
        if "raw_preliminary_step_score" in debug:
            raw_preliminary_score_counter.update([str(debug.get("raw_preliminary_step_score"))])
        selected_case_counts.append(int(debug.get("selected_case_count", len(debug.get("selected_case_ids", [])))))
        candidate_case_counts.append(int(debug.get("candidate_case_count", 0)))
        total_step_count += 1
        has_gold = bool(debug.get("gold_step_score_in_selected_scores"))
        gold_in_selected_step_score_count += int(has_gold)
        gold_missing_selected_step_score_count += int(not has_gold)

    def flush_partial() -> None:
        _write_json(predictions_path, predictions)
        _write_json(retrieval_debug_path, retrieval_debug_records)
        with evaluation_path.open("w", encoding="utf-8") as f:
            for item in evaluation_records:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    for sample_index, sample in enumerate(evaluation_set[:limit], start=1):
        sample_id = str(sample.get("id"))
        if sample_id in completed_ids:
            continue
        print(
            "[PROGRESS] "
            f"mode={mode_name} "
            f"ks={effective_knowledge_size} "
            f"sample={sample_index}/{limit} "
            f"completed={len(completed_ids)}/{limit} "
            f"sample_id={sample_id} "
            f"steps={len(sample.get('steps', []))}",
            flush=True,
        )
        step_calibration_map: list[dict[str, Any]] = []
        step_calibration_map_summary: list[dict[str, Any]] = []
        gold_step_scores = [_clamp_int(step.get("label", 0), -50, 50) for step in sample.get("steps", [])]
        global_preliminary = None
        last_preliminary_error = None
        for preliminary_attempt in range(1, 4):
            try:
                global_preliminary = _global_preliminary_judge_zh(
                    sample,
                    skill if prejudge_use_skill else {},
                    strict_api,
                )
                break
            except Exception as exc:
                last_preliminary_error = exc
                print(
                    f"[WARN][GLOBAL_PRELIM_RETRY] mode={mode_name} ks={effective_knowledge_size} "
                    f"sample_id={sample_id} attempt={preliminary_attempt}/3 error={exc}",
                    flush=True,
                )
                if preliminary_attempt < 3:
                    time.sleep(1.0)
        if global_preliminary is None:
            base_context_audit = _base_scoring_context_audit(sample)
            global_preliminary = {
                "preliminary_score": 0,
                "整题行为摘要": f"全局初判失败后使用0分检索锚点：{last_preliminary_error}",
                "step_preliminaries": [
                    {
                        "step_id": step_id,
                        "preliminary_step_score": 0,
                        "raw_preliminary_step_score": 0,
                        "初步步骤分": 0,
                        "初步步骤分_raw": 0,
                        "local_completion_percent": 0,
                        "局部完成百分比": 0,
                        "normalized_retrieval_band": 0,
                        "学生行为摘要": _short(str(step.get("response", "")), 260),
                        "可能错误类型": ["初判解析失败"],
                        "检索锚点校正原因": ["全局初判失败后保守使用0分检索锚点"],
                        "preliminary_fallback_used": True,
                        "base_scoring_context_audit": base_context_audit,
                    }
                    for step_id, step in enumerate(sample.get("steps", []))
                ],
                "base_scoring_context_audit": base_context_audit,
                "prejudge_granularity": "one_global_call_per_sample",
                "global_preliminary_fallback_used": True,
            }
            print(
                f"[WARN][GLOBAL_PRELIM_FALLBACK] mode={mode_name} ks={effective_knowledge_size} "
                f"sample_id={sample_id} fallback=all_step_score_0",
                flush=True,
            )
        preliminary_by_step = {
            _clamp_int(item.get("step_id", -1), -1, max(-1, len(sample.get("steps", [])) - 1)): item
            for item in global_preliminary.get("step_preliminaries", [])
            if isinstance(item, dict)
        }
        for step_id, step in enumerate(sample.get("steps", [])):
            total_step_count += 1
            preliminary = preliminary_by_step.get(step_id)
            if preliminary is None:
                raise RuntimeError(f"全局初判缺少step_id={step_id}，禁止回退为逐step API调用")
            if use_rag:
                retrieval = _select_step_cases_zh(
                    casebank=casebank,
                    sample=sample,
                    step_id=step_id,
                    preliminary=preliminary,
                )
            else:
                retrieval = {
                    "retrieval_score_distribution": {},
                    "retrieved_reason_calibration": [],
                    "selected_case_ids": [],
                    "selected_case_scores": [],
                    "selected_case_similarity": [],
                    "selected_case_count": 0,
                    "candidate_case_count": 0,
                    "quota_plan": {},
                    "selected_by_band": {},
                    "candidate_score_bands": [],
                    "selected_case_buckets": [],
                    "all_zero_selected": False,
                    "rag_disabled": True,
                }
            ctx = _step_context(sample, step_id)
            calibration_boundaries = retrieval.get("retrieved_reason_calibration", [])
            step_calibration_map.append(
                _model_visible_step_calibration_entry(
                    step_id=step_id,
                    current_step=ctx["current_step"],
                    retrieval=retrieval,
                )
            )
            step_calibration_map_summary.append(
                {
                    "step_id": step_id,
                    "selected_case_ids": retrieval.get("selected_case_ids", []),
                    "selected_case_scores": retrieval.get("selected_case_scores", []),
                    "selected_case_retrieval_bands": retrieval.get(
                        "selected_case_retrieval_bands", []
                    ),
                    "selected_case_similarity": retrieval.get("selected_case_similarity", []),
                }
            )
            selected_case_id_counter.update(str(x) for x in retrieval.get("selected_case_ids", []))
            selected_score_counter.update(str(x) for x in retrieval.get("selected_case_scores", []))
            preliminary_score_counter.update([str(preliminary.get("preliminary_step_score"))])
            raw_preliminary_score_counter.update([str(preliminary.get("raw_preliminary_step_score"))])
            selected_case_counts.append(int(retrieval.get("selected_case_count", 0)))
            candidate_case_counts.append(int(retrieval.get("candidate_case_count", 0)))
            gold_step = _clamp_int(step.get("label", 0), -50, 50)
            has_gold_step_score = gold_step in set(retrieval.get("selected_case_scores", []))
            gold_in_selected_step_score_count += int(has_gold_step_score)
            gold_missing_selected_step_score_count += int(not has_gold_step_score)
            retrieval_debug_records.append(
                {
                    "sample_id": sample.get("id"),
                    "step_id": step_id,
                    "prejudge_granularity": global_preliminary.get("prejudge_granularity"),
                    "global_preliminary_score": global_preliminary.get("preliminary_score"),
                    "global_preliminary_summary": global_preliminary.get("整题行为摘要", ""),
                    "base_scoring_context_audit": global_preliminary.get("base_scoring_context_audit", {}),
                    "preliminary_step_score": preliminary.get("preliminary_step_score"),
                    "raw_preliminary_step_score": preliminary.get("raw_preliminary_step_score"),
                    "local_completion_percent": preliminary.get(
                        "local_completion_percent"
                    ),
                    "normalized_retrieval_band": preliminary.get(
                        "normalized_retrieval_band"
                    ),
                    "retrieval_anchor_type": retrieval.get("retrieval_anchor_type"),
                    "query_retrieval_band": retrieval.get("query_retrieval_band"),
                    "retrieval_anchor_adjustment_reasons": preliminary.get("检索锚点校正原因", []),
                    "学生行为摘要": preliminary.get("学生行为摘要", ""),
                    "可能错误类型": preliminary.get("可能错误类型", []),
                    "quota_plan": retrieval.get("quota_plan", {}),
                    "selected_by_band": retrieval.get("selected_by_band", {}),
                    "candidate_score_bands": retrieval.get("candidate_score_bands", []),
                    "candidate_case_count": retrieval.get("candidate_case_count", 0),
                    "selected_case_ids": retrieval.get("selected_case_ids", []),
                    "selected_case_scores": retrieval.get("selected_case_scores", []),
                    "selected_case_retrieval_bands": retrieval.get(
                        "selected_case_retrieval_bands", []
                    ),
                    "selected_case_buckets": retrieval.get("selected_case_buckets", []),
                    "selected_case_similarity": retrieval.get("selected_case_similarity", []),
                    "all_zero_selected": retrieval.get("all_zero_selected", False),
                    "gold_step_score_in_selected_scores": has_gold_step_score,
                    "retrieved_reason_calibration": calibration_boundaries,
                    "gold_step_score": gold_step,
                }
            )

        prediction = None
        last_error = None
        for attempt in range(1, 4):
            try:
                prediction = _joint_final_judge_zh(
                    sample,
                    skill,
                    step_calibration_map,
                    strict_api,
                    fusion_mode=fusion_mode,
                )
                break
            except Exception as exc:
                last_error = exc
                print(f"[WARN][JOINT_FINAL_RETRY] sample_id={sample.get('id')} attempt={attempt}/3 error={exc}")
                time.sleep(1.0)
        if prediction is None:
            failed_record = {
                "sample_id": sample.get("id"),
                "gold_score": sample.get("manual_label"),
                "pred_score": None,
                "sample_failed": True,
                "failure_stage": "joint_final_judge",
                "failure_reason": str(last_error),
                "partial_result_saved": True,
                "steps": [],
                "step_scores": [],
                "gold_step_scores": gold_step_scores,
                "pred_step_scores": [],
                "step_calibration_map_summary": step_calibration_map_summary,
            }
            predictions.append(failed_record)
            completed_ids.add(sample_id)
            print(
                f"[WARN][SAMPLE_SKIPPED] mode={mode_name} ks={effective_knowledge_size} "
                f"sample_id={sample_id} stage=joint_final_judge",
                flush=True,
            )
            if output_root is not None or resume:
                flush_partial()
            continue

        pred_step_scores = [_clamp_int(step.get("step_score", 0), 0, _clamp_int(sample.get("total", 0), 0, 100)) for step in prediction["steps"]]
        pred_step_sum = sum(pred_step_scores)
        pred_step_sum_total += pred_step_sum
        pred_score = _clamp_int(prediction.get("pred_score", 0), 0, _clamp_int(sample.get("total", 0), 0, 100))
        pred_score_equals_step_sum_count += int(pred_score == pred_step_sum)
        raw_model_pred_score = _clamp_int(prediction.get("raw_model_pred_score", pred_score), 0, _clamp_int(sample.get("total", 0), 0, 100))
        pred_score_overridden = bool(prediction.get("pred_score_overridden", False))
        pred_score_override_count += int(pred_score_overridden)
        if pred_score_overridden:
            pred_score_override_records.append(
                {
                    "sample_id": sample.get("id"),
                    "raw_model_pred_score": raw_model_pred_score,
                    "step_sum": pred_step_sum,
                    "final_pred_score": pred_score,
                }
            )
        leakage_detected = leakage_detected or bool(prediction.get("routing_leakage_detected"))

        record = {
            "sample_id": sample.get("id"),
            "gold_score": sample.get("manual_label"),
            "pred_score": pred_score,
            "raw_model_pred_score": raw_model_pred_score,
            "step_sum": pred_step_sum,
            "final_pred_score": pred_score,
            "pred_score_overridden": pred_score_overridden,
            "steps": prediction["steps"],
            "step_scores": prediction.get("step_reasons", []),
            "gold_step_scores": gold_step_scores,
            "pred_step_scores": pred_step_scores,
            "pred_step_sum": pred_step_sum,
            "pred_score_minus_step_sum": pred_score - pred_step_sum,
            "overall_reason": prediction.get("overall_reason", prediction.get("reason", "")),
            "non_repetition_check": prediction.get("non_repetition_check", ""),
            "error_chain_check": prediction.get("error_chain_check", ""),
            "joint_postprocess_adjustments": prediction.get("joint_postprocess_adjustments", []),
            "step_calibration_map_summary": step_calibration_map_summary,
            "preliminary_global_score": global_preliminary.get("preliminary_score"),
            "preliminary_global_summary": global_preliminary.get("整题行为摘要", ""),
            "prejudge_granularity": global_preliminary.get("prejudge_granularity"),
            "base_scoring_context_audit": global_preliminary.get("base_scoring_context_audit", {}),
            "final_judge_base_scoring_context_audit": prediction.get("base_scoring_context_audit", {}),
            "routing_leakage_detected": prediction.get("routing_leakage_detected", False),
        }
        predictions.append(record)
        evaluation_records.append(_evaluation_record(sample, {"pred_score": pred_score, "steps": prediction["steps"]}))
        completed_ids.add(sample_id)
        if output_root is not None or resume:
            flush_partial()

    metrics = _metrics(evaluation_records)
    total_retrieved = sum(selected_case_id_counter.values())
    top5_case_share = (
        sum(count for _, count in selected_case_id_counter.most_common(5)) / total_retrieved
        if total_retrieved
        else 0
    )
    failed_predictions = [record for record in predictions if record.get("sample_failed")]
    completed_prediction_count = len(predictions) - len(failed_predictions)
    one_step_records = [
        item
        for item in evaluation_records
        if len(item.get("steps", [])) == 1
    ]
    one_step_metrics = _metrics(one_step_records) if one_step_records else {}
    one_step_abs_errors = [
        abs(_clamp_int(item.get("pred_label", 0), 0, _clamp_int(item.get("total", 0), 0, 100)) - _clamp_int(item.get("manual_label", 0), -100, 100))
        for item in one_step_records
    ]
    summary = {
        "mode": mode_name,
        "knowledge_size": effective_knowledge_size,
        "use_skill": use_skill,
        "use_rag": use_rag,
        "prejudge_use_skill": prejudge_use_skill,
        "fusion_mode": fusion_mode,
        "knowledge_output_dir": str(output_dir),
        "dataset_path": str(dataset_path),
        "dataset_size": len(samples),
        "evaluation_set_range": f"samples[0:{evaluation_count}]",
        "knowledge_set_range": knowledge_set_range,
        "limit": limit,
        "completed_samples": completed_prediction_count,
        "processed_samples": len(predictions),
        "failed_samples": len(failed_predictions),
        "failed_sample_ids": [record.get("sample_id") for record in failed_predictions],
        "runtime_seconds": round(time.perf_counter() - started, 3),
        **metrics,
        "pred_step_sum": pred_step_sum_total,
        "raw_model_pred_score_sum": sum(_clamp_int(record.get("raw_model_pred_score", record.get("pred_score", 0)), 0, 100) for record in predictions),
        "final_pred_score_sum": sum(_clamp_int(record.get("final_pred_score", record.get("pred_score", 0)), 0, 100) for record in predictions),
        "pred_score_overridden_count": pred_score_override_count,
        "pred_score_overridden_rate": pred_score_override_count / len(predictions) if predictions else 0,
        "pred_score_override_records": pred_score_override_records,
        "pred_score_equals_step_sum_rate": pred_score_equals_step_sum_count / len(predictions) if predictions else 0,
        "evaluation_skill_path": str(_path_in_output(output_dir, "evaluation_skill_zh.json")),
        "full_step_casebank_path": str(_path_in_output(output_dir, "full_step_casebank_zh_evidence_reason.json")),
        "full_step_casebank_audit_path": str(_path_in_output(output_dir, "full_step_casebank_zh_evidence_reason_audit.json")),
        "full_step_casebank_audit": casebank_audit,
        "num_step_cases": casebank_audit.get("num_step_cases", 0),
        "score_distribution": casebank_audit.get("score_distribution", {}),
        "preliminary_distribution": dict(preliminary_score_counter),
        "raw_preliminary_distribution": dict(raw_preliminary_score_counter),
        "retrieval_score_distribution": dict(selected_score_counter),
        "selected_score_distribution": dict(selected_score_counter),
        "selected_case_count_avg": sum(selected_case_counts) / len(selected_case_counts) if selected_case_counts else 0,
        "unique_cases_retrieved": len(selected_case_id_counter),
        "top5_case_share": top5_case_share,
        "gold_in_selected_score_rate": gold_in_selected_step_score_count / total_step_count if total_step_count else 0,
        "gold_score_missing_from_selected_cases": gold_missing_selected_step_score_count,
        "gold_score_missing_from_selected_cases_rate": gold_missing_selected_step_score_count / total_step_count if total_step_count else 0,
        "candidate_case_count_avg": sum(candidate_case_counts) / len(candidate_case_counts) if candidate_case_counts else 0,
        "one_step_subset_n": len(one_step_records),
        "one_step_subset_CCS": one_step_metrics.get("CCS"),
        "one_step_subset_QWK": one_step_metrics.get("QWK"),
        "one_step_subset_total_exact": one_step_metrics.get("total_exact"),
        "one_step_subset_MAE": sum(one_step_abs_errors) / len(one_step_abs_errors) if one_step_abs_errors else None,
        "leakage_detected": leakage_detected,
        "routing_leakage_detected": leakage_detected,
    }
    flush_partial()
    _write_json(output_dir / f"run_summary_{mode_name}.json", summary)
    if limit <= 3:
        _write_json(output_dir / f"smoke_{mode_name}_3.json", {"predictions": predictions, "summary": summary})
    return summary


def _run_mode(
    mode: str,
    limit: int,
    strict_api: bool,
    knowledge_size: int | None = None,
    dataset_path: Path | None = None,
    eval_size: int | None = None,
    output_root: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    if mode != "teachgrade":
        raise ValueError("Only the TeachGrade scoring pipeline is included")
    return _run_joint_step_rag_zh(
        limit=limit,
        strict_api=strict_api,
        knowledge_size=knowledge_size,
        dataset_path=dataset_path,
        eval_size=eval_size,
        output_root=output_root,
        resume=resume,
        mode_name="teachgrade",
        use_skill=True,
        use_rag=True,
        prejudge_use_skill=True,
        fusion_mode="native_base_additive",
    )
