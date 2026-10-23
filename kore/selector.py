"""지식 선택기.

similarity: 검색 점수 상위 k개를 그대로 쓴다(vanilla RAG 기준선).
kore: 유사도, 질문어 커버리지, 분야 다양성, 중복 패널티, 출처 신뢰도를 합친 가중 점수로
      욕심쟁이(greedy) 방식으로 한 개씩 고른다. 분야 다양성 보너스는 후보의 정규화 유사도에
      비례한다(무관한 문서를 다양성만으로 끌어들이지 않기 위해서다).

kore 점수식은 설계 가설이다. 가중치가 맞는지, 유사도 순위보다 나은지는 실험 전에는 모른다.
두 전략은 같은 후보 목록, 같은 k, 같은 토큰 예산을 받아서 공정하게 비교한다.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from kore.schemas import KnowledgeItem
from kore.splits import jaccard
from kore.tokenize import estimate_item_tokens, tokenize

STRATEGIES = ("similarity", "kore")
FEATURES = ("coverage", "diversity", "redundancy", "reliability")


@dataclass(frozen=True)
class SelectorWeights:
    sim: float = 1.0
    coverage: float = 0.5
    diversity: float = 0.2
    redundancy: float = 0.5
    reliability: float = 0.2

    def ablate(self, feature: str) -> "SelectorWeights":
        if feature not in FEATURES:
            raise ValueError(f"feature는 {FEATURES} 중 하나여야 한다: {feature!r}")
        return replace(self, **{feature: 0.0})

    def as_dict(self) -> dict[str, float]:
        return {
            "sim": self.sim,
            "coverage": self.coverage,
            "diversity": self.diversity,
            "redundancy": self.redundancy,
            "reliability": self.reliability,
        }


def _sim(item: KnowledgeItem) -> float:
    # retrieval_score가 비어 있으면 계약의 score 필드를 쓴다.
    return item.retrieval_score if item.retrieval_score != 0.0 else item.score


def select_knowledge(
    question: str,
    candidates: list[KnowledgeItem],
    k: int = 3,
    strategy: str = "kore",
    weights: SelectorWeights | None = None,
    token_budget: int | None = None,
    tokenizer_mode: str = "ngram",
    term_weights: dict[str, float] | None = None,
) -> list[KnowledgeItem]:
    """candidates에서 최대 k개를 고른다. 입력 객체는 바꾸지 않고 복사본을 돌려준다.

    token_budget을 주면 남은 예산보다 큰 문서는 건너뛴다. 후보가 모자라면 k보다 적게 돌려준다.
    term_weights는 질문 토큰별 가중치(보통 코퍼스 IDF)다. 주면 커버리지를 가중해서 계산해
    불용어가 점수를 올리지 못하게 한다. 없으면 모든 토큰을 같은 가중치 1로 본다.
    """
    if k < 1:
        raise ValueError("k는 1 이상이어야 한다")
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy는 {STRATEGIES} 중 하나여야 한다: {strategy!r}")
    if token_budget is not None and token_budget < 0:
        raise ValueError("token_budget은 0 이상이어야 한다")
    cands = list(candidates)
    if strategy == "similarity":
        return _select_similarity(cands, k, token_budget)
    return _select_kore(
        question, cands, k, weights or SelectorWeights(), token_budget, tokenizer_mode, term_weights
    )


def _select_similarity(cands: list[KnowledgeItem], k: int, budget: int | None) -> list[KnowledgeItem]:
    order = sorted(range(len(cands)), key=lambda i: (-_sim(cands[i]), i))
    remaining = budget
    picked: list[KnowledgeItem] = []
    for i in order:
        if len(picked) >= k:
            break
        cost = estimate_item_tokens(cands[i])
        if remaining is not None:
            if cost > remaining:
                continue
            remaining -= cost
        s = _sim(cands[i])
        picked.append(cands[i].copy_with(score=s, selection_score=s))
    return picked


def _select_kore(
    question: str,
    cands: list[KnowledgeItem],
    k: int,
    w: SelectorWeights,
    budget: int | None,
    mode: str,
    term_weights: dict[str, float] | None = None,
) -> list[KnowledgeItem]:
    q_terms = set(tokenize(question, mode))
    tw = {t: (term_weights.get(t, 0.0) if term_weights is not None else 1.0) for t in q_terms}
    tw_total = sum(tw.values())
    tok_sets = [set(tokenize(f"{c.title} {c.text}", mode)) for c in cands]
    costs = [estimate_item_tokens(c) for c in cands]
    max_sim = max((_sim(c) for c in cands), default=0.0)

    chosen: list[int] = []
    scores: dict[int, float] = {}
    covered: set[str] = set()
    chosen_domains: set[str] = set()
    remaining = budget

    while len(chosen) < k:
        best_idx = -1
        best_score = 0.0
        for i, c in enumerate(cands):
            if i in scores:
                continue
            if remaining is not None and costs[i] > remaining:
                continue
            sim = _sim(c) / max_sim if max_sim > 0.0 else 0.0
            gained = (tok_sets[i] & q_terms) - covered
            cov = sum(tw[t] for t in gained) / tw_total if tw_total > 0.0 else 0.0
            # 다양성 보너스는 관련도에 비례시킨다. 그렇지 않으면 무관한 다른 분야 문서가 점수만으로 끌려 들어온다.
            div = 0.0 if c.domain in chosen_domains else sim
            red = max((jaccard(tok_sets[i], tok_sets[j]) for j in chosen), default=0.0)
            score = (
                w.sim * sim
                + w.coverage * cov
                + w.diversity * div
                - w.redundancy * red
                + w.reliability * (c.reliability - 0.5)
            )
            if best_idx < 0 or score > best_score + 1e-12:  # 동점이면 앞선 후보를 유지
                best_idx, best_score = i, score
        if best_idx < 0:
            break
        chosen.append(best_idx)
        scores[best_idx] = best_score
        covered |= tok_sets[best_idx] & q_terms
        chosen_domains.add(cands[best_idx].domain)
        if remaining is not None:
            remaining -= costs[best_idx]

    return [cands[i].copy_with(score=scores[i], selection_score=scores[i]) for i in chosen]
