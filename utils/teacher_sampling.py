from __future__ import annotations

import random
from typing import Any


def split_teacher_examples(
    rows: list[dict[str, Any]],
    k: int = 20,
    seed: int = 42,
    evaluation_count: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if isinstance(k, bool) or not isinstance(k, int) or not 0 < k < len(rows):
        raise ValueError("k must be a positive integer smaller than the dataset size")
    ids = [str(row.get("id", "")) if row.get("id") is not None else "" for row in rows]
    if any(not sample_id.strip() for sample_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("Dataset IDs must be nonempty and unique")
    teacher_indices = set(random.Random(seed).sample(range(len(rows)), k))
    teachers = [row for index, row in enumerate(rows) if index in teacher_indices]
    evaluation = [row for index, row in enumerate(rows) if index not in teacher_indices]
    if evaluation_count is not None:
        if isinstance(evaluation_count, bool) or not isinstance(evaluation_count, int):
            raise ValueError("evaluation_count must be an integer")
        if not 0 < evaluation_count <= len(evaluation):
            raise ValueError("evaluation_count exceeds the non-teacher pool or is nonpositive")
        evaluation = evaluation[:evaluation_count]
    return evaluation, teachers
