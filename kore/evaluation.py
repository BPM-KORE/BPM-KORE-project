"""평가 지표와 통계 도구.

검색 지표(Recall@k, Precision, Hit, MRR)는 '정답 근거 문서를 골랐는가'를 잰다.
답변 품질 기여도와는 다른 값이라서 둘을 섞어 해석하지 않는다.
"""

from __future__ import annotations

import math
import random
import re
from statistics import NormalDist
from typing import Any

from kore.schemas import KnowledgeItem
from kore.tokenize import estimate_item_tokens, normalize

_DIGITS_RE = re.compile(r"^[0-9][0-9.,]*$")


def recall_at_k(selected_ids: list[str], expected_ids: list[str]) -> float | None:
    """정답 근거 중 선택된 비율. 정답 근거가 없으면 None."""
    if not expected_ids:
        return None
    return len(set(selected_ids) & set(expected_ids)) / len(set(expected_ids))


def precision_at_k(selected_ids: list[str], expected_ids: list[str]) -> float:
    if not selected_ids:
        return 0.0
    return len(set(selected_ids) & set(expected_ids)) / len(selected_ids)


def hit_at_k(selected_ids: list[str], expected_ids: list[str]) -> float:
    return 1.0 if set(selected_ids) & set(expected_ids) else 0.0


def mrr(ranked_ids: list[str], expected_ids: list[str]) -> float:
    expected = set(expected_ids)
    for rank, sid in enumerate(ranked_ids, start=1):
        if sid in expected:
            return 1.0 / rank
    return 0.0


def _alt_matches(answer_norm: str, alt: str) -> bool:
    alt_norm = normalize(alt)
    if _DIGITS_RE.match(alt_norm):  # '7'이 '17'에 걸리지 않게 숫자 경계를 본다
        return re.search(rf"(?<![0-9]){re.escape(alt_norm)}(?![0-9])", answer_norm) is not None
    return alt_norm in answer_norm


def rubric_score(answer: str, rubric_items: list[list[str]]) -> float | None:
    """루브릭 항목 중 답변에 들어 있는 비율. 키워드 일치로 재는 값이라 의미 평가의 대용치일 뿐이다."""
    if not rubric_items:
        return None
    answer_norm = normalize(answer)
    hit = sum(1 for alts in rubric_items if any(_alt_matches(answer_norm, a) for a in alts))
    return hit / len(rubric_items)


def evaluate(
    question: str,
    selected: list[KnowledgeItem],
    reference: dict[str, Any],
    answer: str | None = None,
) -> dict[str, Any]:
    """reference: expected_source_ids(list[str]), rubric_items(list[list[str]], 선택)."""
    ids = [s.id for s in selected]
    expected = list(reference.get("expected_source_ids", []))
    result: dict[str, Any] = {
        "recall": recall_at_k(ids, expected),
        "precision": precision_at_k(ids, expected),
        "hit": hit_at_k(ids, expected),
        "mrr": mrr(ids, expected),
        "n_selected": len(ids),
        "context_tokens": sum(estimate_item_tokens(s) for s in selected),
        "rubric_score": None,
    }
    if answer is not None and reference.get("rubric_items"):
        result["rubric_score"] = rubric_score(answer, reference["rubric_items"])
    return result


def bootstrap_ci(
    values: list[float | None], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> tuple[float | None, float | None, float | None]:
    """평균과 부트스트랩 백분위 구간. None은 건너뛴다. 같은 seed면 같은 값이 나온다."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None, None
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return mean, mean, mean
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return mean, lo, hi


def paired_bootstrap_diff(
    a: list[float | None], b: list[float | None], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> dict[str, float | int | None]:
    """같은 질문끼리 짝지은 평균 차이(a - b)와 부트스트랩 구간. 둘 중 하나라도 None인 짝은 뺀다."""
    if len(a) != len(b):
        raise ValueError("a와 b는 같은 질문 순서로 같은 길이여야 한다")
    diffs = [x - y for x, y in zip(a, b) if x is not None and y is not None]
    mean, lo, hi = bootstrap_ci(diffs, n_boot=n_boot, seed=seed, alpha=alpha)
    return {"n": len(diffs), "mean_diff": mean, "ci_lo": lo, "ci_hi": hi}


def approx_mde(n: int, sd_diff: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """쌍체 비교에서 검출 가능한 최소 평균 차이의 근사값. sd_diff는 질문별 차이의 표준편차."""
    if n < 2:
        raise ValueError("n은 2 이상이어야 한다")
    nd = NormalDist()
    z = nd.inv_cdf(1 - alpha / 2) + nd.inv_cdf(power)
    return z * sd_diff / math.sqrt(n)


def _ranks(values: list[float]) -> list[float]:
    """평균 순위(동점은 평균). 가장 작은 값이 1."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    """스피어만 순위상관. 한쪽이 전부 같은 값이면 정의되지 않으므로 None."""
    if len(x) != len(y) or len(x) < 3:
        return None
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return None
    return cov / (vx * vy) ** 0.5
