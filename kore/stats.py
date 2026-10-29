"""데이터 건강검진. B와 C가 올린 파일이 실험에 쓸 만한지 숫자로 보여 준다.

validate는 '형식이 맞는가'를 본다. 이 모듈은 '연구에 쓸 수 있는가'를 본다:
규모, 분야 균형, 난이도 분포, 근거 둘 이상 질문 비율, 후보 재현율, 질문 베끼기, 중복 문서, 출처 확인 상태.
통과해도 내용이 사실인지는 알 수 없다. 그건 사람이 원문을 열어서 확인한다.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

from kore.io import UNKNOWN_LICENSES
from kore.retrieval import BM25Retriever
from kore.schemas import EvalQuestion, KnowledgeItem
from kore.splits import jaccard, near_duplicate_pairs
from kore.tokenize import normalize, tokenize

SHINGLE = 8  # 질문 베끼기를 볼 때 쓰는 글자 조각 길이
COPY_THRESHOLD = 0.5  # 질문 조각의 이 비율 이상이 근거 문서에 그대로 있으면 의심한다
NEAR_DUP_DOC = 0.85

# 단계별 목표. week1·final의 숫자는 마스터 문서 9절과 1절에서 왔고, week2는 그 사이를 내가 정한 값이다.
STAGES: dict[str, dict[str, int]] = {
    "week1": {"docs": 50, "questions": 30, "test": 0, "dev": 0},
    "week2": {"docs": 100, "questions": 60, "test": 0, "dev": 0},
    "final": {"docs": 150, "questions": 100, "test": 24, "dev": 12},
}
SOURCE_LOG_COLUMNS = ["id", "source_url", "accessed_date", "license_checked", "verified_by", "note"]


def read_source_log(path: str | Path) -> dict[str, dict[str, str]]:
    """출처 기록을 id로 찾을 수 있게 읽는다. 파일이 없으면 빈 사전."""
    p = Path(path)
    if not p.exists():
        return {}
    with p.open(newline="", encoding="utf-8-sig") as fh:
        return {r["id"].strip(): {k: (v or "").strip() for k, v in r.items() if k} for r in csv.DictReader(fh) if r.get("id")}


def _shingles(text: str, n: int = SHINGLE) -> set[str]:
    compact = "".join(normalize(text).split())
    return {compact[i : i + n] for i in range(max(0, len(compact) - n + 1))}


def copy_risk(question: EvalQuestion, by_id: dict[str, KnowledgeItem]) -> float:
    """질문의 글자 조각 중 정답 근거 문서에 그대로 들어 있는 비율(0~1). 높으면 BM25가 부풀려진다."""
    q = _shingles(question.question)
    if not q:
        return 0.0
    best = 0.0
    for sid in question.expected_source_ids:
        item = by_id.get(sid)
        if item is not None:
            best = max(best, len(q & _shingles(item.text)) / len(q))
    return best


def _pct(n: int, d: int) -> float:
    return (n / d) if d else 0.0


def data_health(
    items: list[KnowledgeItem],
    questions: list[EvalQuestion],
    stage: str = "final",
    source_log: dict[str, dict[str, str]] | None = None,
    candidate_k: int = 10,
) -> dict[str, Any]:
    """점검 항목 목록과 요약. 항목마다 level이 '필수'이면 실험 전에 반드시 통과해야 한다."""
    if stage not in STAGES:
        raise ValueError(f"stage는 {sorted(STAGES)} 중 하나여야 한다: {stage!r}")
    target = STAGES[stage]
    final = stage == "final"
    by_id = {i.id: i for i in items}
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str, level: str = "필수") -> None:
        checks.append({"name": name, "ok": bool(ok), "level": level, "detail": detail})

    # ---- 문서
    domains = Counter(i.domain for i in items)
    add("문서 수", len(items) >= target["docs"], f"{len(items)}개 (목표 {target['docs']}개 이상)")
    if items:
        small = [d for d, n in domains.items() if _pct(n, len(items)) < 0.2]
        add("분야 균형(문서)", not small and len(domains) >= 2, f"{dict(domains)}; 20% 미만 분야: {small or '없음'}", "권고")
    unknown = [i.id for i in items if i.license.strip().lower() in UNKNOWN_LICENSES]
    add("라이선스 미확인 문서", not unknown, f"{len(unknown)}개 {unknown[:5]}")
    if source_log is not None:
        unchecked = [i.id for i in items if source_log.get(i.id, {}).get("license_checked", "").lower() != "yes"]
        add(
            "출처 기록(license_checked=yes)",
            not unchecked,
            f"{len(unchecked)}개가 아직 yes가 아니다 {unchecked[:5]}",
            "필수" if final else "권고",
        )
    lengths = sorted(len(i.text) for i in items)
    if lengths:
        outside = sum(1 for n in lengths if n < 100 or n > 3000)
        add(
            "문서 길이",
            _pct(outside, len(lengths)) <= 0.1,
            f"중앙값 {lengths[len(lengths) // 2]}자, 100자 미만 또는 3000자 초과 {outside}개({_pct(outside, len(lengths)):.0%})",
            "권고",
        )
    sets = [set(tokenize(i.text, "word")) for i in items]
    dup_docs = [
        (items[a].id, items[b].id)
        for a in range(len(items))
        for b in range(a + 1, len(items))
        if jaccard(sets[a], sets[b]) >= NEAR_DUP_DOC
    ]
    add("거의 같은 문서 쌍", not dup_docs, f"{len(dup_docs)}쌍 {dup_docs[:3]}", "권고")

    # ---- 질문
    if questions:
        splits = Counter(q.split for q in questions)
        diffs = Counter(q.difficulty for q in questions)
        q_domains = Counter(q.domain for q in questions)
        multi = sum(1 for q in questions if len(q.expected_source_ids) >= 2)
        with_ref = sum(1 for q in questions if q.reference_answer.strip())
        with_rub = sum(1 for q in questions if q.rubric.strip())
        add("질문 수", len(questions) >= target["questions"], f"{len(questions)}개 (목표 {target['questions']}개 이상)")
        add(
            "분할 크기",
            splits["test"] >= target["test"] and splits["dev"] >= target["dev"],
            f"{dict(splits)} (test {target['test']}, dev {target['dev']} 이상)",
        )
        small_q = [d for d, n in q_domains.items() if _pct(n, len(questions)) < 0.2]
        add("분야 균형(질문)", not small_q, f"{dict(q_domains)}; 20% 미만 분야: {small_q or '없음'}", "권고")
        if final:
            thin = [d for d in ("easy", "medium", "hard") if _pct(diffs[d], len(questions)) < 0.15]
            add("난이도 분포", not thin, f"{dict(diffs)}; 15% 미만: {thin or '없음'}")
            add("근거 둘 이상 질문(H2용)", _pct(multi, len(questions)) >= 0.25, f"{multi}개({_pct(multi, len(questions)):.0%}), 25% 이상 필요")
            add("reference_answer 작성률", _pct(with_ref, len(questions)) >= 0.95, f"{with_ref}/{len(questions)} (기여도 라벨링에 필요)")
            add("rubric 작성률", _pct(with_rub, len(questions)) >= 0.95, f"{with_rub}/{len(questions)} (자동 채점에 필요)")
        else:
            add("근거 둘 이상 질문(H2용)", multi >= 1, f"{multi}개", "권고")
        leaks = near_duplicate_pairs(questions)
        add("분할 간 거의 같은 질문", not leaks, f"{len(leaks)}쌍 {[(a, b) for a, b, _ in leaks[:3]]}")
        risky = sorted(
            ((copy_risk(q, by_id), q.question_id) for q in questions), reverse=True
        )
        risky = [(round(r, 2), qid) for r, qid in risky if r >= COPY_THRESHOLD]
        add("질문이 근거 문장을 베낀 의심", not risky, f"{len(risky)}개 {risky[:5]} (겹침 비율, 높으면 검색 성능이 부풀려진다)", "권고")
        retriever = BM25Retriever(items) if items else None
        if retriever is not None:
            hits = []
            for q in questions:
                cands = {c.id for c in retriever.retrieve(q.question, k=candidate_k)}
                hits.append(len(cands & set(q.expected_source_ids)) / len(set(q.expected_source_ids)))
            recall = sum(hits) / len(hits)
            add(
                f"후보 재현율(BM25 상위 {candidate_k})",
                recall >= 0.8,
                f"{recall:.2f} (0.8 미만이면 선택기 이전에 검색부터 고쳐야 한다)",
                "필수" if final else "권고",
            )
        used = {sid for q in questions for sid in q.expected_source_ids}
        unused = len(set(by_id) - used)
        add("정답 근거로 안 쓰인 문서", True, f"{unused}개({_pct(unused, len(items)):.0%}) — 방해 문서 역할을 한다. 정보용", "권고")
    else:
        add("질문 수", False, f"0개 (목표 {target['questions']}개 이상)")

    required_failed = [c for c in checks if c["level"] == "필수" and not c["ok"]]
    return {
        "stage": stage,
        "n_docs": len(items),
        "n_questions": len(questions),
        "checks": checks,
        "ready": not required_failed,
        "required_failed": [c["name"] for c in required_failed],
    }


def format_health(report: dict[str, Any]) -> str:
    lines = [f"데이터 점검 (단계: {report['stage']}) — 문서 {report['n_docs']}개, 질문 {report['n_questions']}개", ""]
    for c in report["checks"]:
        mark = "통과" if c["ok"] else ("실패" if c["level"] == "필수" else "주의")
        lines.append(f"[{mark}] ({c['level']}) {c['name']}: {c['detail']}")
    lines.append("")
    if report["ready"]:
        lines.append("필수 항목을 모두 통과했다. 단, 내용이 사실인지는 이 도구가 알 수 없다. 사람이 원문으로 확인한다.")
    else:
        lines.append("필수 항목 미통과: " + ", ".join(report["required_failed"]))
    return "\n".join(lines)
