"""닫힌북 필터와 분야 게이트.

no_retrieval 결과로 '모델이 근거 없이도 아는 질문'을 걸러내고,
gold_context 결과로 '정답 근거를 줘도 못 맞히는 질문(질문·루브릭·모델 문제)'을 걸러낸다.
분야별로 세 숫자를 모아 그 분야를 연구에 쓸지 결정하는 근거로 삼는다.

vanilla_rag도 돌렸다면 개선 여지(gold_context − vanilla_rag)를 본다. 이 차이가 작으면 유사도 상위 k개가 이미
정답 근거를 거의 다 주고 있다는 뜻이라, 어떤 선택기도 그 분야에서 나아질 여지가 없다. 방해 문서(비슷하지만
틀린 문서)를 늘리거나 근거가 둘 이상 필요한 질문을 늘린다. 기준값 0.1은 처음 정한 값이고 검증하지 않았다.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from kore.pipeline import _read_log

KNOWN_THRESHOLD = 0.67  # no_retrieval 루브릭 점수가 이 이상이면 '모델이 이미 안다'
GOLD_LOW = 0.5  # gold_context 점수가 이 미만이면 '정답 근거를 줘도 못 맞힌다'
GATE = {"no_retrieval_max": 0.5, "gold_context_min": 0.6, "candidate_recall_min": 0.8, "headroom_min": 0.1}
MIN_N_FOR_GATE = 5
SCREEN_COLUMNS = [
    "question_id", "domain", "split", "no_retrieval_score", "vanilla_rag_score", "gold_context_score", "candidate_recall",
    "flags",
]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def screen_rows(rows: list[dict[str, Any]], known: float = KNOWN_THRESHOLD, gold_low: float = GOLD_LOW) -> list[dict[str, Any]]:
    per_q: dict[str, dict[str, Any]] = {}
    for r in rows:
        d = per_q.setdefault(
            r["question_id"],
            {"question_id": r["question_id"], "domain": r["domain"], "split": r["split"],
             "no_retrieval_score": None, "vanilla_rag_score": None, "gold_context_score": None,
             "candidate_recall": r.get("candidate_recall")},
        )
        if r["condition"] == "no_retrieval":
            d["no_retrieval_score"] = r.get("rubric_score")
        elif r["condition"] == "gold_context":
            d["gold_context_score"] = r.get("rubric_score")
        elif r["condition"] == "vanilla_rag":
            d["vanilla_rag_score"] = r.get("rubric_score")
    out = []
    for d in per_q.values():
        flags = []
        if d["no_retrieval_score"] is not None and d["no_retrieval_score"] >= known:
            flags.append("모델이_이미_안다")
        if d["gold_context_score"] is not None and d["gold_context_score"] < gold_low:
            flags.append("정답근거를_줘도_못맞힘")
        if d["candidate_recall"] is not None and d["candidate_recall"] == 0:
            flags.append("후보에_정답근거_없음")
        d["flags"] = ";".join(flags)
        out.append(d)
    return sorted(out, key=lambda d: d["question_id"])


def domain_gate(screened: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """분야별 게이트. 세 조건을 모두 넘어야 PASS. 표본이 5개 미만이면 판정을 미룬다."""
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for d in screened:
        by_domain[d["domain"]].append(d)
    result = []
    for domain, qs in sorted(by_domain.items()):
        nr = _mean([q["no_retrieval_score"] for q in qs if q["no_retrieval_score"] is not None])
        gc = _mean([q["gold_context_score"] for q in qs if q["gold_context_score"] is not None])
        rc = _mean([q["candidate_recall"] for q in qs if q["candidate_recall"] is not None])
        pairs = [(q["gold_context_score"], q["vanilla_rag_score"]) for q in qs
                 if q.get("gold_context_score") is not None and q.get("vanilla_rag_score") is not None]
        headroom = _mean([g - v for g, v in pairs])
        n_scored = sum(1 for q in qs if q["no_retrieval_score"] is not None and q["gold_context_score"] is not None)
        reasons = []
        if nr is not None and nr > GATE["no_retrieval_max"]:
            reasons.append(f"근거 없이도 평균 {nr:.2f} 맞힘(모델이 이미 안다)")
        if gc is not None and gc < GATE["gold_context_min"]:
            reasons.append(f"정답 근거를 줘도 평균 {gc:.2f}(모델 한국어 능력, 질문, 루브릭 점검)")
        if rc is not None and rc < GATE["candidate_recall_min"]:
            reasons.append(f"후보 재현율 {rc:.2f}(검색부터 고친다)")
        if headroom is not None and headroom < GATE["headroom_min"]:
            reasons.append(f"개선 여지 {headroom:+.2f}(vanilla_rag가 이미 정답 근거 수준이다. 방해 문서·다중 근거 질문을 늘린다)")
        if n_scored < MIN_N_FOR_GATE:
            verdict = "판정 보류(표본 부족)"
        else:
            verdict = "PASS" if not reasons else "FAIL"
        result.append(
            {"domain": domain, "n_questions": len(qs), "n_scored": n_scored, "no_retrieval": nr,
             "gold_context": gc, "candidate_recall": rc, "headroom": headroom, "verdict": verdict, "reasons": reasons}
        )
    return result


def write_screen_csv(path: str | Path, screened: list[dict[str, Any]]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SCREEN_COLUMNS)
        writer.writeheader()
        for d in screened:
            writer.writerow({c: ("" if d.get(c) is None else d.get(c)) for c in SCREEN_COLUMNS})


def screen_run_dir(run_dir: str | Path, known: float = KNOWN_THRESHOLD, gold_low: float = GOLD_LOW) -> dict[str, Any]:
    log = Path(run_dir) / "run_log.jsonl"
    rows = _read_log(log)
    if not rows:
        raise ValueError(f"{log}에 결과가 없다. 먼저 no_retrieval과 gold_context 조건으로 run을 돌린다")
    if not any(r["condition"] == "no_retrieval" for r in rows):
        raise ValueError("no_retrieval 결과가 없다. --conditions no_retrieval gold_context 로 돌린다")
    if not any(r.get("rubric_score") is not None for r in rows):
        raise ValueError("루브릭 점수가 없다. 생성기를 켜고(--generator) 질문에 rubric을 채워서 돌린다")
    screened = screen_rows(rows, known=known, gold_low=gold_low)
    return {"screened": screened, "gate": domain_gate(screened)}


def format_gate(gate: list[dict[str, Any]]) -> str:
    def f(v: float | None) -> str:
        return "-" if v is None else f"{v:.2f}"

    lines = [
        f"분야 게이트 기준: no_retrieval ≤ {GATE['no_retrieval_max']}, gold_context ≥ {GATE['gold_context_min']}, "
        f"후보 재현율 ≥ {GATE['candidate_recall_min']}, 개선 여지(gold_context − vanilla_rag) ≥ {GATE['headroom_min']} "
        f"(표본 {MIN_N_FOR_GATE}개 미만이면 보류, vanilla_rag를 안 돌렸으면 개선 여지는 건너뜀)",
        "",
    ]
    for g in gate:
        lines.append(
            f"{g['domain']:<10} {g['verdict']:<18} 질문 {g['n_questions']}개  no_retrieval {f(g['no_retrieval'])}  "
            f"gold_context {f(g['gold_context'])}  재현율 {f(g['candidate_recall'])}  개선 여지 {f(g.get('headroom'))}"
        )
        for r in g["reasons"]:
            lines.append(f"    - {r}")
    return "\n".join(lines)
