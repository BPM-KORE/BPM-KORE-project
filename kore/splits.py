"""질문 분할과 누출 점검."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import replace

from kore.schemas import EvalQuestion
from kore.tokenize import tokenize


def assign_splits(
    questions: list[EvalQuestion],
    seed: int = 0,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> list[EvalQuestion]:
    """분야별로 층화해서 train/dev/test를 결정적으로 배정한다.

    같은 seed와 같은 question_id면 언제나 같은 결과가 나온다.
    분야 안의 질문이 3개 미만이면 전부 train으로 둔다.
    """
    if abs(sum(ratios) - 1.0) > 1e-6 or any(r < 0 for r in ratios):
        raise ValueError(f"ratios 합은 1이어야 한다: {ratios}")
    by_domain: dict[str, list[EvalQuestion]] = defaultdict(list)
    for q in questions:
        by_domain[q.domain].append(q)

    assigned: dict[str, str] = {}
    for domain in sorted(by_domain):
        ordered = sorted(
            by_domain[domain],
            key=lambda q: hashlib.sha256(f"{seed}|{q.question_id}".encode()).hexdigest(),
        )
        n = len(ordered)
        if n < 3:
            for q in ordered:
                assigned[q.question_id] = "train"
            continue
        n_test = max(1, round(n * ratios[2]))
        n_dev = max(1, round(n * ratios[1]))
        for idx, q in enumerate(ordered):
            if idx < n_test:
                assigned[q.question_id] = "test"
            elif idx < n_test + n_dev:
                assigned[q.question_id] = "dev"
            else:
                assigned[q.question_id] = "train"
    return [replace(q, split=assigned[q.question_id]) for q in questions]


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def near_duplicate_pairs(
    questions: list[EvalQuestion], threshold: float = 0.8
) -> list[tuple[str, str, float]]:
    """서로 다른 분할에 걸쳐 거의 같은 질문이 있는지 찾는다(분할 간 누출 후보)."""
    sets = [set(tokenize(q.question, "word")) for q in questions]
    pairs: list[tuple[str, str, float]] = []
    for i in range(len(questions)):
        for j in range(i + 1, len(questions)):
            if questions[i].split == questions[j].split:
                continue
            sim = jaccard(sets[i], sets[j])
            if sim >= threshold:
                pairs.append((questions[i].question_id, questions[j].question_id, sim))
    return pairs
