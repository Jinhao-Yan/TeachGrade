from __future__ import annotations

import pytest


@pytest.fixture
def compiler_example():
    fields = ("工作流程", "评分原则", "部分分规则", "常见扣分", "零分情况", "高分情况", "边界说明")
    teachers = [
        {
            "id": f"fictional-{index}",
            "question": "What is two plus two?",
            "total": 2,
            "reference": "Four.",
            "analysis": "Add the two quantities.",
            "manual_label": score,
            "steps": [{"response": response, "label": score, "errors": []}],
        }
        for index, (score, response) in enumerate(((2, "Four."), (0, "Five."), (1, "Add two."), (2, "2 + 2 = 4.")))
    ]
    source_skill = {field: ["Fictional original rule."] for field in fields}
    source_rag = {"cases": [
        {
            "case_id": f"K{index}_S0",
            "source_sample_id": row["id"],
            "step_id": 0,
            "学生步骤": row["steps"][0]["response"],
            "参考节点匹配": "addition result",
            "教师评分": row["manual_label"],
            "错误类型": [],
            "评分边界": "Correct addition result completes the answer point.",
            "评分理由短评": "Fictional teacher evidence.",
            "检索文本_证据": "An addition answer point.",
        }
        for index, row in enumerate(teachers)
    ]}
    specification = {
        "percentage_transfer_contract": {
            "case_selection_rule": "按完成百分比档位检索当前任务相关owner案例。",
            "current_owner_budget_rule": "当前owner预算来自当前题官方评分指南和完整BASE，不使用来源题分值。",
            "percentage_reinterpretation_rule": "历史完成百分比只在当前owner预算上作为软边界，结合当前题BASE重解释。",
            "soft_relevance_rule": "当前owner相关性越强，百分比参考越强；弱相关时降低数值影响。",
            "no_absolute_score_visibility_rule": "当前评分不得看到、恢复、猜测或引用历史教师绝对分和来源局部预算。",
            "base_priority_rule": "当前题完整BASE优先，百分比只是软参考而非固定单价或硬上限。",
        },
        "task_ontology": {"credit_unit": "answer point", "owner_definition": "addition result", "grading_style": "partial credit"},
        "scoring_unit_policy": "Score each independently correct point.",
        "semantic_equivalence_rule": "Equivalent arithmetic expressions match.",
        "semantic_non_equivalence_rule": "A different result does not match.",
        "duplicate_credit_rule": "Do not count a repeated point twice.",
        "dependency_rule": "Check the prerequisite reasoning.",
        "hard_scoring_rules": ["Never exceed the question budget.", "Do not reward a wrong result."],
        "soft_scoring_guidance": ["Retain valid partial credit.", "Use teacher evidence as a soft boundary."],
        "boundary_induction": [{"boundary": "Correct result", "higher_condition": "Complete addition", "lower_condition": "Incomplete addition", "confidence": 0.8}],
        "skill_compilation": {field: [{"rule": "Check the addition result.", "evidence_ids": ["sample:fictional-0:S0"], "confidence": 0.8}] for field in fields},
        "rag_compilation": {field: "Use task-specific answer points and teacher completion evidence." for field in (
            "task_retrieval_objective", "owner_mapping_policy", "semantic_boundary_policy",
            "error_pattern_policy", "dependency_policy", "teacher_boundary_policy",
        )},
        "teacher_behavior_analysis": {"summary": "Fictional teacher examples.", "supporting_examples": ["sample:fictional-0:S0"], "contradicting_examples": ["sample:fictional-1:S0"], "confidence": 0.8},
        "contrastive_analysis": [{"claim": "Correct completion earns more credit.", "supporting_examples": ["sample:fictional-0:S0"], "contradicting_examples": ["sample:fictional-1:S0"], "confidence": 0.8}],
        "ontology_validation": {"status": "PASS", "validated_claims": ["addition result"], "rejected_unsupported_claims": []},
        "evidence_support": [{"claim": "Completion controls credit.", "supporting_example_ids": [f"sample:fictional-{index}:S0" for index in range(4)], "confidence": 0.8}, {"claim": "Equivalent results match.", "supporting_example_ids": ["sample:fictional-3:S0"], "confidence": 0.8}],
    }
    return {"teachers": teachers, "source_skill": source_skill, "source_rag": source_rag, "specification": specification}


@pytest.fixture
def task_config():
    return {
        "q_id": "fictional-task", "course": "Example", "question_type": "ShortAns",
        "guideline": "Award credit for correct answer points and retain valid partial credit.",
        "errors": [{"name": "Incorrect concept", "description": "A conceptual error."}],
    }
