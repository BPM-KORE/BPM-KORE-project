"""질문 CSV 자기채점·누출 점검 (로컬 전용). 사용: python team/배정_2026-11-03/self_score.py 내_질문.csv
참조답이 자기 루브릭에서 1.00을 받는지, 루브릭 단어가 질문에 그대로 들어 있지 않은지 본다.
test 질문 파일은 저장소 밖에 두고 이 스크립트만 로컬에서 돌린다. 출력에는 질문 ID와 점수만 나온다."""
from __future__ import annotations
import csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kore.evaluation import rubric_score, normalize  # noqa: E402

def items(r):
    return [[a.strip() for a in it.split("/") if a.strip()] for it in r["rubric"].split("|") if it.strip()]

bad = 0
for r in csv.DictReader(open(sys.argv[1], encoding="utf-8-sig")):
    it = items(r)
    s = rubric_score(r["reference_answer"], it)
    echo = rubric_score(r["question"], it)
    leak = [a for g in it for a in g if len(a) >= 2 and normalize(a) in normalize(r["question"])]
    ok = s == 1.0 and (echo or 0) == 0 and not leak
    bad += 0 if ok else 1
    print(r["question_id"], f"자기채점={s:.2f}", f"질문베끼기={echo:.2f}", "누출" if leak else "", "" if ok else "<-- 고칠 것")
print("고칠 문항:", bad)
sys.exit(1 if bad else 0)
