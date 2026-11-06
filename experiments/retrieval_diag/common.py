"""공통 로더. kore는 읽기 전용으로 import만 한다(소스 수정 없음)."""
from __future__ import annotations
import csv, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kore.io import load_knowledge  # noqa: E402
from kore.retrieval import BM25Retriever  # noqa: E402
from kore.tokenize import tokenize  # noqa: E402

KB = ROOT / "team/provisional/law/knowledge.csv"
QS = ROOT / "experiments/eval_normalization/evaluation_questions_20_v2.csv"

def load():
    items = load_knowledge(KB)
    qs = []
    with open(QS, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            r["gold"] = r["expected_source_ids"].split("|")
            qs.append(r)
    return items, qs

def rank_of(order_ids, gold):
    pos = {i: n + 1 for n, i in enumerate(order_ids)}
    return {g: pos.get(g) for g in gold}  # None = 점수 0으로 후보 밖
