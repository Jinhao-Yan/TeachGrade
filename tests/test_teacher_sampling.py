from __future__ import annotations

import copy
import random

import pytest

from utils.teacher_sampling import split_teacher_examples


def rows(count=80):
    return [{"id": f"sample_{index}", "steps": []} for index in range(count)]


def test_seeded_selection_is_uniform_sample_not_tail():
    source = rows()
    evaluation, teachers = split_teacher_examples(source, k=20, seed=42)
    indices = set(random.Random(42).sample(range(len(source)), 20))
    assert teachers == [row for index, row in enumerate(source) if index in indices]
    assert teachers != source[-20:]
    assert len(teachers) == 20
    assert len(evaluation) == 60
    assert {row["id"] for row in teachers}.isdisjoint(row["id"] for row in evaluation)
    assert {row["id"] for row in teachers + evaluation} == {row["id"] for row in source}


def test_reproducible_and_seed_sensitive():
    source = rows()
    assert split_teacher_examples(source, 20, 42) == split_teacher_examples(source, 20, 42)
    assert split_teacher_examples(source, 20, 42)[1] != split_teacher_examples(source, 20, 43)[1]


def test_sampling_preserves_input_order_and_global_random_state():
    source = rows()
    original = copy.deepcopy(source)
    state = random.getstate()
    evaluation, teachers = split_teacher_examples(source, 20, 42)
    assert random.getstate() == state
    assert source == original
    assert teachers == sorted(teachers, key=lambda row: source.index(row))
    assert evaluation == sorted(evaluation, key=lambda row: source.index(row))


def test_evaluation_limit_does_not_change_teachers():
    all_evaluation, teachers = split_teacher_examples(rows(), 20, 42)
    limited, limited_teachers = split_teacher_examples(rows(), 20, 42, 10)
    assert limited == all_evaluation[:10]
    assert limited_teachers == teachers


@pytest.mark.parametrize("k", [0, -1, 80, 81, True, 2.5])
def test_invalid_k_is_rejected(k):
    with pytest.raises(ValueError, match="k must"):
        split_teacher_examples(rows(), k)


@pytest.mark.parametrize("count", [0, -1, 61, True, 2.5])
def test_invalid_evaluation_count_is_rejected(count):
    with pytest.raises(ValueError, match="evaluation_count"):
        split_teacher_examples(rows(), 20, 42, count)


@pytest.mark.parametrize("invalid", [[{}, {"id": "x"}], [{"id": None}, {"id": "x"}], [{"id": " "}, {"id": "x"}], [{"id": "x"}, {"id": "x"}], [{"id": 1}, {"id": "1"}]])
def test_invalid_ids_are_rejected(invalid):
    with pytest.raises(ValueError, match="IDs"):
        split_teacher_examples(invalid, 1)
