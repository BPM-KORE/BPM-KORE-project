"""v2 세트 검증: (1) 참조답 자기채점 20/20=1.0 (2) 반례(wrong<1, partial<1) (3) 빈 답·질문 베끼기 =0 (4) 질문 누출(알트가 질문에 포함)."""
from __future__ import annotations
import csv, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from kore.evaluation import rubric_score, normalize  # noqa: E402

def items(r): return [[a.strip() for a in it.split("/") if a.strip()] for it in r["rubric"].split("|") if it.strip()]

qs = {r["question_id"]: r for r in csv.DictReader(open(HERE/"evaluation_questions_20_v2.csv", encoding="utf-8"))}
ce = list(csv.DictReader(open(HERE/"counterexamples.csv", encoding="utf-8")))
bad = 0
print("qid  self  wrong partial echo empty  n_items")
by = {}
for c in ce: by.setdefault(c["question_id"], {})[c["kind"]] = rubric_score(c["answer"], items(qs[c["question_id"]]))
for qid, r in qs.items():
    it = items(r); s = rubric_score(r["reference_answer"], it)
    echo = rubric_score(r["question"], it); empty = rubric_score("모르겠다", it)
    w, p = by[qid]["wrong"], by[qid]["partial"]
    ok = s == 1.0 and w < 1 and p < 1 and w <= 0.25 and p < 1 and echo == 0 and empty == 0 and 0 < p
    leak = [a for g in it for a in g if len(a) >= 2 and normalize(a) in normalize(r["question"])]
    flag = "" if ok and not leak else "  <-- 확인"
    if flag: bad += 1
    print(f"{qid}  {s:.2f}  {w:.2f}  {p:.2f}    {echo:.2f}  {empty:.2f}   {len(it)}{flag}{'  누출:'+str(leak) if leak else ''}")
n = len(qs)
print(f"\n자기채점 평균 {sum(rubric_score(r['reference_answer'], items(r)) for r in qs.values())/n:.3f} | wrong 평균 {sum(v['wrong'] for v in by.values())/n:.3f} | partial 평균 {sum(v['partial'] for v in by.values())/n:.3f}")
print("문제 문항:", bad)
sys.exit(1 if bad else 0)
