"""C 질문 CSV 점검 스크립트.

하는 일
1) expected_source_ids의 구분자를 `;`/`|` 모두 `|`로 통일하고 domain을 knowledge의 분야로 맞춘 사본을 쓴다(원본은 건드리지 않는다).
2) 근거 id가 knowledge에 있는지 본다.
3) reference_answer를 실제 채점 함수(kore.evaluation.rubric_score)에 넣어 점수를 낸다.
   정답 문장이 자기 루브릭을 다 채우지 못하면 루브릭이 키워드가 아니라 설명문이라는 뜻이다.

실행(저장소 맨 위 폴더에서):
  python3 -I team/C_전달_2026-11-01/A_검증/check_c_questions.py \
    --questions team/C_전달_2026-11-01/data/evaluation_questions_20_provisional.csv \
    --knowledge team/provisional/law/knowledge.csv \
    --out team/C_전달_2026-11-01/A_검증/evaluation_questions_20_계약형식.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from kore.evaluation import rubric_score  # noqa: E402

FIELDS = ["question_id", "question", "expected_source_ids", "domain", "difficulty", "split", "reference_answer", "rubric"]


def items_of(rubric: str) -> list[list[str]]:
    return [[a.strip() for a in it.split("/") if a.strip()] for it in rubric.split("|") if it.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--knowledge", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    kb = {r["id"]: r for r in csv.DictReader(open(a.knowledge, encoding="utf-8-sig"))}
    domains = {r["domain"] for r in kb.values()}
    domain = next(iter(domains)) if len(domains) == 1 else None
    rows = list(csv.DictReader(open(a.questions, encoding="utf-8-sig")))

    fixed = []
    for r in rows:
        r = dict(r)
        r["expected_source_ids"] = "|".join(x.strip() for x in re.split(r"[;|]", r["expected_source_ids"]) if x.strip())
        if domain:
            r["domain"] = domain
        fixed.append({k: r[k] for k in FIELDS})
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(fixed)

    print(f"질문 {len(fixed)}개, 사본 저장: {a.out}")
    print("id   | 근거수 | 없는id | 정답문장의 자기루브릭 점수 | 루브릭항목수")
    scores = []
    for r in fixed:
        ids = r["expected_source_ids"].split("|")
        missing = [i for i in ids if i not in kb]
        items = items_of(r["rubric"])
        s = rubric_score(r["reference_answer"], items)
        scores.append(s)
        print(f"{r['question_id']} | {len(ids)} | {missing or '-'} | {s:.2f} | {len(items)}")
    n_full = sum(1 for s in scores if s == 1.0)
    n_zero = sum(1 for s in scores if s == 0.0)
    print(f"평균 {sum(scores)/len(scores):.3f} / 만점 {n_full}개 / 0점 {n_zero}개")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
