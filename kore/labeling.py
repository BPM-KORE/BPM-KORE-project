"""문서 기여도(ΔQ) 라벨링과 H1·H2 예비 분석.

질문 하나에 대해 후보 문서의 부분집합마다 '정답 문장의 평균 로그확률'을 재서
- 문서 하나가 답에 얼마나 보탬이 되는지(단일 기여도 = 점수(그 문서) - 점수(근거 없음)),
- 두 문서를 같이 줄 때 합과 다른지(시너지 = 점수(a,b) - 점수(a) - 점수(b) + 점수(없음))
를 기록한다. 이 값으로 "유사도 순위와 실제 기여도가 어긋나는가"(H1)와 "조합 효과가 있는가"(H2)를 예비로 본다.

한계(분석 결과를 읽을 때 같이 적는다)
- 로그확률은 답변 품질의 대용치다. 모델이 정답 문장을 얼마나 그럴듯하게 보느냐를 재는 값이다.
- 한 모델, 한 프롬프트, 한 정답 문장 표현의 결과다. 다른 모델에서 같다는 보장이 없다.
- 로그 척도에서는 확률이 1에 가까워질수록 보탬이 줄어든다. 그래서 시너지 값이 음수로 치우치는 것은
  문서끼리 간섭해서가 아니라 척도 때문일 수 있다.
- 부분집합 안의 문서 순서는 검색 순위순으로 고정했다. 순서 효과는 재지 않는다.
- test 분할은 라벨링하지 않는다(선택기 학습과 튜닝에 쓰일 값이라서).
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

from kore.evaluation import bootstrap_ci, spearman
from kore.generation import AnswerScorer
from kore.pipeline import KOREPipeline, PipelineConfig, _canonical_hash, _read_log
from kore.schemas import EvalQuestion, KnowledgeItem

LABEL_SPLITS = ("train", "dev")
EMPTY_KEY = ""


def subset_key(ids: list[str] | tuple[str, ...]) -> str:
    """부분집합의 키. id 순서와 상관없이 같은 집합은 같은 키가 되게 정렬해서 잇는다."""
    return "|".join(sorted(ids))


def enumerate_subsets(ids: list[str], max_size: int) -> list[tuple[str, ...]]:
    """크기 0부터 max_size까지 모든 부분집합. 순서는 결정적이다(크기, 그다음 입력 순서)."""
    out: list[tuple[str, ...]] = []
    for size in range(0, min(max_size, len(ids)) + 1):
        out.extend(combinations(ids, size))
    return out


def count_calls(n_candidates: int, max_size: int) -> int:
    """질문 하나에 필요한 점수 계산 횟수(공집합 포함)."""
    from math import comb

    return sum(comb(n_candidates, s) for s in range(0, min(max_size, n_candidates) + 1))


def label_question(
    q: EvalQuestion,
    candidates: list[KnowledgeItem],
    scorer: AnswerScorer,
    max_cands: int = 8,
    max_subset: int = 3,
) -> dict[str, Any]:
    """질문 하나의 모든 부분집합 점수를 계산한다. 후보는 검색 순위 앞쪽 max_cands개만 쓴다."""
    if not q.reference_answer.strip():
        raise ValueError(f"{q.question_id}: reference_answer가 비어 있어 라벨링할 수 없다")
    cands = candidates[:max_cands]
    by_id = {c.id: c for c in cands}
    order = {c.id: n for n, c in enumerate(cands)}
    scores: dict[str, float] = {}
    for subset in enumerate_subsets([c.id for c in cands], max_subset):
        evidence = [by_id[i] for i in sorted(subset, key=order.__getitem__)]  # 검색 순위순으로 고정
        scores[subset_key(subset)] = float(scorer.score(q.question, evidence, q.reference_answer))
    return {
        "question_id": q.question_id,
        "domain": q.domain,
        "split": q.split,
        "difficulty": q.difficulty,
        "candidate_ids": [c.id for c in cands],
        "expected_source_ids": list(q.expected_source_ids),
        "scores": scores,
    }


def run_labeling(
    items: list[KnowledgeItem],
    questions: list[EvalQuestion],
    out_dir: str | Path,
    scorer: AnswerScorer,
    config: PipelineConfig | None = None,
    splits: tuple[str, ...] = LABEL_SPLITS,
    max_cands: int = 8,
    max_subset: int = 3,
    resume: bool = True,
    limit: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """질문별 기여도 라벨을 dq_log.jsonl에 쌓는다. 끊겨도 같은 명령으로 이어서 돈다."""
    if "test" in splits:
        raise ValueError("test 분할은 라벨링하지 않는다(선택기 학습·튜닝에 쓰일 값이라서)")
    bad = [s for s in splits if s not in ("train", "dev")]
    if bad:
        raise ValueError(f"알 수 없는 분할: {bad}")
    if max_cands < 2 or max_subset < 1:
        raise ValueError("max_cands는 2 이상, max_subset은 1 이상이어야 한다")
    config = config or PipelineConfig()
    scope = [q for q in questions if q.split in splits and q.reference_answer.strip()]
    skipped = [q.question_id for q in questions if q.split in splits and not q.reference_answer.strip()]
    if limit is not None:
        scope = scope[:limit]
    if not scope:
        raise ValueError("라벨링할 질문이 없다(reference_answer가 있는 train/dev 질문이 필요하다)")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path, env_path = out / "dq_log.jsonl", out / "dq_env.json"
    fingerprint = {
        "config_hash": _canonical_hash(config.to_dict()),
        "data_hash": _canonical_hash(
            {"items": [i.to_dict() for i in items], "questions": [q.to_dict() for q in questions]}
        ),
        "scorer": scorer.name,
        "max_cands": max_cands,
        "max_subset": max_subset,
    }
    if resume and env_path.exists():
        prev = json.loads(env_path.read_text(encoding="utf-8"))
        diff = [k for k, v in fingerprint.items() if prev.get(k) != v]
        if diff:
            raise ValueError(f"이전 라벨링과 {', '.join(diff)}가 다르다. 새 out_dir를 쓰거나 resume=False로 다시 시작한다")
    if not resume:
        for p in (log_path, env_path):
            if p.exists():
                p.unlink()
    env_path.write_text(json.dumps(fingerprint, ensure_ascii=False, indent=2), encoding="utf-8")

    done = {r["question_id"] for r in _read_log(log_path)}
    pipeline = KOREPipeline(items, config=config)
    if log_path.exists() and log_path.stat().st_size > 0 and not log_path.read_bytes().endswith(b"\n"):
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write("\n")
    with log_path.open("a", encoding="utf-8") as fh:
        for n, q in enumerate(scope, start=1):
            if q.question_id not in done:
                cands = pipeline.candidates(q.question)
                if len(cands) >= 2:  # 후보가 하나뿐이면 순위 비교가 안 된다
                    rec = label_question(q, cands, scorer, max_cands=max_cands, max_subset=max_subset)
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
            if progress:
                progress(n, len(scope))
    rows = [r for r in _read_log(log_path) if r["question_id"] in {q.question_id for q in scope}]
    return {"out_dir": str(out), "n_labeled": len(rows), "skipped_no_reference": skipped, "rows": rows}


# ---------------------------------------------------------------- 분석


def analyze_question(row: dict[str, Any], k: int = 3, eps: float = 1e-9) -> dict[str, Any]:
    """질문 하나의 H1·H2 지표."""
    cands: list[str] = row["candidate_ids"]
    scores: dict[str, float] = row["scores"]
    base = scores[EMPTY_KEY]
    gains = {c: scores[c] - base for c in cands if c in scores}
    ordered = [gains[c] for c in cands]
    rank_score = [-float(n) for n in range(len(cands))]  # 검색 순위가 높을수록 큰 값
    max_size = max((len(key.split("|")) for key in scores if key), default=0)
    k_eff = min(k, max_size, len(cands))

    best_single = max(cands, key=lambda c: gains[c])
    best_gain = gains[best_single]
    top1_gain = gains[cands[0]]
    top1_mismatch = None
    if best_gain - top1_gain > eps:
        top1_mismatch = 1.0
    elif len(cands) >= 2:
        top1_mismatch = 0.0

    sim_key = subset_key(cands[:k_eff])
    size_k = [key for key in scores if key and len(key.split("|")) == k_eff]
    oracle_key = max(size_k, key=lambda key: scores[key]) if size_k else None
    gap = (scores[oracle_key] - scores[sim_key]) if oracle_key and sim_key in scores else None
    # 유사도 상위 k가 '무작위 k개'와 '최선 k개' 사이 어디쯤인지(0=무작위 수준, 1=최선). 기준이 있어야 해석할 수 있다.
    capture = None
    if size_k and sim_key in scores:
        rand_mean = sum(scores[key] for key in size_k) / len(size_k)
        span = scores[oracle_key] - rand_mean
        if span > eps:
            capture = (scores[sim_key] - rand_mean) / span

    synergies: list[float] = []
    for a, b in combinations(cands, 2):
        key = subset_key((a, b))
        if key in scores:
            synergies.append(scores[key] - scores[a] - scores[b] + base)
    mean_abs_single = sum(abs(g) for g in ordered) / len(ordered) if ordered else 0.0

    # 간섭: 가장 도움이 되는 문서에 다른 문서를 더했더니 오히려 나빠진 비율
    hurt = []
    for c in cands:
        if c == best_single:
            continue
        pair = subset_key((best_single, c))
        if pair in scores:
            hurt.append(1.0 if scores[pair] < scores[best_single] - eps else 0.0)

    return {
        "question_id": row["question_id"],
        "domain": row.get("domain", ""),
        "split": row.get("split", ""),
        "n_candidates": len(cands),
        "spearman_rank_vs_gain": spearman(rank_score, ordered),
        "top1_mismatch": top1_mismatch,
        "k_eff": k_eff,
        "oracle_gap_at_k": gap,
        "capture_ratio": capture,
        "top1_mismatch_null": (1.0 - 1.0 / len(cands)) if cands else None,
        "sim_topk_gain": (scores[sim_key] - base) if sim_key in scores else None,
        "frac_candidates_harmful": sum(1 for g in ordered if g < -eps) / len(ordered) if ordered else None,
        "mean_synergy": (sum(synergies) / len(synergies)) if synergies else None,
        "mean_abs_synergy": (sum(abs(s) for s in synergies) / len(synergies)) if synergies else None,
        "mean_abs_single_gain": mean_abs_single,
        "synergy_to_single_ratio": (
            (sum(abs(s) for s in synergies) / len(synergies)) / mean_abs_single
            if synergies and mean_abs_single > eps
            else None
        ),
        "frac_add_hurts_best": (sum(hurt) / len(hurt)) if hurt else None,
        "best_is_gold": 1.0 if best_single in set(row.get("expected_source_ids", [])) else 0.0,
    }


SUMMARY_METRICS = (
    "spearman_rank_vs_gain",
    "top1_mismatch",
    "top1_mismatch_null",
    "capture_ratio",
    "oracle_gap_at_k",
    "sim_topk_gain",
    "frac_candidates_harmful",
    "mean_synergy",
    "mean_abs_synergy",
    "synergy_to_single_ratio",
    "frac_add_hurts_best",
    "best_is_gold",
)


def analyze_labels(rows: list[dict[str, Any]], k: int = 3, seed: int = 0) -> dict[str, Any]:
    """질문별 지표와 전체 요약(평균과 부트스트랩 95% 구간)을 돌려준다.

    해석 가이드
    - spearman_rank_vs_gain이 1에 가까우면 유사도 순위가 곧 기여도 순위다(H1 지지 안 함).
    - top1_mismatch가 높으면 검색 1위가 가장 도움이 되는 문서가 아닌 질문이 많다.
    - top1_mismatch는 후보가 m개일 때 기여도와 순위가 무관해도 평균 1 - 1/m 만큼 나온다(top1_mismatch_null). 이 값과 비교해서 읽는다.
    - capture_ratio는 유사도 상위 k가 '무작위 k개'(0)와 '최선 k개'(1) 사이 어디쯤인지 나타낸다. 1보다 한참 낮으면 선택으로 줄일 여지가 있고, 0이거나 음수이면 유사도가 무작위보다 나을 게 없다는 뜻이다.
    - oracle_gap_at_k는 최선 k개와 유사도 상위 k개의 차이다. 여러 조합 중 최댓값이라서 점수에 잡음만 있어도 0보다 크게 나온다.
      그래서 이 값이 0보다 크다는 것만으로는 H1의 증거가 되지 않는다. 헤드룸(줄일 수 있는 최대치)을 가늠하는 용도로만 쓴다.
    - synergy_to_single_ratio가 0에 가까우면 문서들이 거의 더하기로 작동한다(H2 지지 안 함).
    """
    per_q = [analyze_question(r, k=k) for r in rows]
    summary: dict[str, Any] = {"n_questions": len(per_q), "k": k}
    for metric in SUMMARY_METRICS:
        vals = [p[metric] for p in per_q]
        mean, lo, hi = bootstrap_ci(vals, seed=seed)
        summary[metric] = {"n": sum(v is not None for v in vals), "mean": mean, "ci_lo": lo, "ci_hi": hi}
    return {"summary": summary, "per_question": per_q}


def format_report(analysis: dict[str, Any]) -> str:
    s = analysis["summary"]
    lines = [f"질문 {s['n_questions']}개, k={s['k']}", ""]
    labels = {
        "spearman_rank_vs_gain": "순위상관(검색 순위 vs 단일 기여도)",
        "top1_mismatch": "검색 1위가 최고 기여 문서가 아닌 비율",
        "top1_mismatch_null": "  (순위와 기여도가 무관할 때의 기대값)",
        "capture_ratio": "유사도 상위 k의 위치(0=무작위, 1=최선)",
        "oracle_gap_at_k": "최선 선택 대비 유사도 상위 k 손실(항상 ≥0, 증거로 쓰지 않음)",
        "sim_topk_gain": "유사도 상위 k의 기여도(근거 없음 대비)",
        "frac_candidates_harmful": "해가 되는 후보 비율",
        "mean_synergy": "쌍 시너지 평균(부호 포함)",
        "mean_abs_synergy": "쌍 시너지 절댓값 평균",
        "synergy_to_single_ratio": "시너지 / 단일 기여도 크기 비",
        "frac_add_hurts_best": "최고 문서에 추가하면 나빠지는 비율",
        "best_is_gold": "최고 기여 문서가 정답 근거인 비율",
    }
    for key in SUMMARY_METRICS:
        v = s[key]
        if v["mean"] is None:
            lines.append(f"{labels[key]}: 계산할 수 없음")
        else:
            lines.append(f"{labels[key]}: {v['mean']:+.3f}  (95% 구간 {v['ci_lo']:+.3f} ~ {v['ci_hi']:+.3f}, n={v['n']})")
    lines += [
        "",
        "이 값은 예비 신호다. 한 모델·한 프롬프트·로그확률 대용치의 결과이고, 구간이 넓으면 결론을 내지 않는다.",
        "시너지가 음수로 치우친 것은 로그 척도의 포화 때문일 수 있다. 간섭의 증거로 바로 쓰지 않는다.",
    ]
    return "\n".join(lines)

