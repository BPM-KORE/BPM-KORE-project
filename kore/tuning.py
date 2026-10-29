"""선택기 가중치 탐색. test 분할은 구조적으로 막아 둔다."""

from __future__ import annotations

import itertools
from dataclasses import replace
from typing import Any

from kore.pipeline import PipelineConfig, build_retriever
from kore.schemas import EvalQuestion, KnowledgeItem
from kore.selector import FEATURES, SelectorWeights, select_knowledge
from kore.evaluation import recall_at_k

DEFAULT_GRID: dict[str, list[float]] = {
    "coverage": [0.0, 0.25, 0.5],
    "diversity": [0.0, 0.2],
    "redundancy": [0.0, 0.5, 1.0],
    "reliability": [0.0, 0.2],
}
SMALL_SAMPLE = 30


def grid_search_weights(
    items: list[KnowledgeItem],
    questions: list[EvalQuestion],
    grid: dict[str, list[float]] | None = None,
    config: PipelineConfig | None = None,
    splits: tuple[str, ...] = ("train", "dev"),
) -> dict[str, Any]:
    """train/dev 질문에서 평균 Recall이 가장 높은 가중치를 찾는다.

    'test'가 splits에 들어 있으면 ValueError를 낸다. 최종 test는 튜닝에 쓰지 않는다.
    목적함수가 근거 문서 Recall이라서 답변 품질 기여도를 직접 최적화하지는 않는다.
    """
    if "test" in splits:
        raise ValueError("test 분할은 튜닝에 쓸 수 없다. train 또는 dev만 지정한다")
    cfg = config or PipelineConfig()
    cfg.validate()
    grid = grid or DEFAULT_GRID
    for name in grid:
        if name not in FEATURES:
            raise ValueError(f"grid의 키는 {FEATURES} 중 하나여야 한다: {name!r}")
    scope = [q for q in questions if q.split in splits]
    if not scope:
        raise ValueError(f"{splits}에 해당하는 질문이 없다")

    retriever = build_retriever(items, cfg)
    cands = {q.question_id: retriever.retrieve(q.question, k=cfg.candidate_k) for q in scope}

    def mean_recall(strategy: str, weights: SelectorWeights | None) -> float:
        vals = []
        for q in scope:
            picked = select_knowledge(
                q.question, cands[q.question_id], k=cfg.k, strategy=strategy, weights=weights,
                token_budget=cfg.token_budget, tokenizer_mode=cfg.tokenizer_mode,
                term_weights=retriever.term_weights(q.question),
            )
            r = recall_at_k([p.id for p in picked], q.expected_source_ids)
            if r is not None:
                vals.append(r)
        return sum(vals) / len(vals) if vals else 0.0

    names = list(grid)
    table: list[dict[str, Any]] = []
    for combo in itertools.product(*(grid[n] for n in names)):
        w = replace(cfg.weights, **dict(zip(names, combo)))
        table.append({"weights": w.as_dict(), "recall": mean_recall("kore", w)})
    best = max(table, key=lambda row: row["recall"])  # 동점이면 먼저 나온 조합
    baseline = mean_recall("similarity", None)
    return {
        "n_questions": len(scope),
        "splits": list(splits),
        "baseline_similarity_recall": baseline,
        "best": best,
        "table": table,
        "warning": (
            f"질문이 {len(scope)}개뿐이라 과적합 위험이 크다. 최적 가중치를 곧바로 일반화하지 않는다"
            if len(scope) < SMALL_SAMPLE
            else ""
        ),
    }
