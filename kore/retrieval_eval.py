"""검색기 설정끼리 후보 검색 성능을 비교한다(train·dev만).

선택기와 생성기를 고치기 전에 "정답 근거가 후보에 들어오는가"부터 본다. 후보에 없으면 어떤 선택기도 고를 수 없다.
같은 질문끼리 짝지어 비교하고 부트스트랩 95% 구간을 낸다. 여러 설정을 보고 가장 좋은 것을 고르면
그 자체가 dev에 맞춘 튜닝이므로, 시도한 설정 수를 함께 적는다.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from kore.evaluation import mrr, paired_bootstrap_diff, recall_at_k
from kore.pipeline import PipelineConfig, build_retriever
from kore.schemas import EvalQuestion, KnowledgeItem

EVAL_SPLITS = ("train", "dev")
KS = (1, 3, 10)


def parse_variant(spec: str, base: PipelineConfig) -> PipelineConfig:
    """'bm25', 'bm25:ngram', 'hybrid:ngram_stem:0.3'(검색기:토크나이저:alpha) 형식."""
    parts = spec.split(":")
    if not parts[0]:
        raise ValueError(f"설정 이름이 비었다: {spec!r}")
    changes: dict[str, Any] = {"retriever": parts[0]}
    if len(parts) > 1 and parts[1]:
        changes["tokenizer_mode"] = parts[1]
    if len(parts) > 2 and parts[2]:
        changes["hybrid_alpha"] = float(parts[2])
    if len(parts) > 3:
        raise ValueError(f"형식은 검색기[:토크나이저[:alpha]]: {spec!r}")
    cfg = replace(base, **changes)
    cfg.validate()
    from kore.tokenize import MODES

    if cfg.tokenizer_mode not in MODES:
        raise ValueError(f"토크나이저는 {MODES} 중 하나: {cfg.tokenizer_mode!r}")
    return cfg


def compare_retrievers(
    items: list[KnowledgeItem],
    questions: list[EvalQuestion],
    variants: list[str],
    base: PipelineConfig | None = None,
    splits: tuple[str, ...] = EVAL_SPLITS,
    seed: int = 0,
) -> dict[str, Any]:
    if "test" in splits:
        raise ValueError("test 분할로 검색기를 고르지 않는다. train 또는 dev만 지정한다")
    if len(variants) < 1:
        raise ValueError("비교할 설정이 하나 이상 필요하다")
    base = base or PipelineConfig()
    scope = [q for q in questions if q.split in splits and q.expected_source_ids]
    if not scope:
        raise ValueError(f"{splits}에 정답 근거가 있는 질문이 없다")
    depth = max(max(KS), base.candidate_k)
    per: dict[str, dict[str, list[float | None]]] = {}
    for spec in variants:
        cfg = parse_variant(spec, base)
        retriever = build_retriever(items, cfg)
        cols: dict[str, list[float | None]] = {f"recall@{k}": [] for k in KS}
        cols["candidate_recall"] = []
        cols["mrr"] = []
        for q in scope:
            ranked = [c.id for c in retriever.retrieve(q.question, k=depth)]
            for k in KS:
                cols[f"recall@{k}"].append(recall_at_k(ranked[:k], q.expected_source_ids))
            cols["candidate_recall"].append(recall_at_k(ranked[: cfg.candidate_k], q.expected_source_ids))
            cols["mrr"].append(mrr(ranked, q.expected_source_ids))
        per[spec] = cols

    def mean(xs: list[float | None]) -> float:
        vals = [x for x in xs if x is not None]
        return sum(vals) / len(vals) if vals else 0.0

    first = variants[0]
    summary = []
    for spec in variants:
        row: dict[str, Any] = {"variant": spec, **{m: mean(v) for m, v in per[spec].items()}}
        if spec != first:
            for m in ("recall@3", "candidate_recall"):
                row[f"diff_{m}"] = paired_bootstrap_diff(per[spec][m], per[first][m], seed=seed)
        summary.append(row)
    return {"n_questions": len(scope), "splits": list(splits), "baseline": first, "rows": summary,
            "n_variants": len(variants), "per_question": per, "question_ids": [q.question_id for q in scope]}


def format_comparison(res: dict[str, Any]) -> str:
    lines = [f"검색기 비교 — {'·'.join(res['splits'])} 질문 {res['n_questions']}개, 기준 {res['baseline']}", ""]
    lines.append(f"{'설정':<26} {'R@1':>6} {'R@3':>6} {'R@10':>6} {'후보':>6} {'MRR':>6}")
    for r in res["rows"]:
        lines.append(f"{r['variant']:<26} {r['recall@1']:>6.3f} {r['recall@3']:>6.3f} {r['recall@10']:>6.3f} "
                     f"{r['candidate_recall']:>6.3f} {r['mrr']:>6.3f}")
    lines.append("")
    for r in res["rows"]:
        for m in ("recall@3", "candidate_recall"):
            d = r.get(f"diff_{m}")
            if d and d["mean_diff"] is not None:
                verdict = "나음" if d["ci_lo"] > 0 else ("나쁨" if d["ci_hi"] < 0 else "구분 못 함")
                lines.append(f"{r['variant']} − 기준, {m}: {d['mean_diff']:+.3f} [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}] → {verdict}")
    lines.append("")
    lines.append(f"시도한 설정 {res['n_variants']}개. 가장 좋은 것을 고르는 일도 dev에 맞춘 튜닝이다. 고른 뒤 test로 다시 고르지 않는다.")
    if res["n_questions"] < 30:
        lines.append(f"질문이 {res['n_questions']}개뿐이라 구간이 넓다. 차이가 작으면 기준 설정을 그대로 둔다.")
    return "\n".join(lines)
