"""a 보강) 오답 1위와 정답의 BM25 항별 기여 상위 토큰을 비교한다."""
from __future__ import annotations
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import *

items, qs = load()
bm = BM25Retriever(items)
idx = {it.id: n for n, it in enumerate(items)}

def contrib(question, doc_id):
    n = idx[doc_id]; out = {}
    for term in dict.fromkeys(tokenize(question)):
        for i, tf in bm._postings.get(term, ()):
            if i == n:
                norm = bm.k1 * (1 - bm.b + bm.b * bm._lengths[n] / bm._avgdl)
                out[term] = bm._idf[term] * tf * (bm.k1 + 1) / (tf + norm)
    return out

for qid in ["Q01", "Q02", "Q03", "Q05", "Q10", "Q12", "Q14"]:
    q = next(x for x in qs if x["question_id"] == qid)
    top = bm.retrieve(q["question"], k=1)[0]
    print(f"\n=== {qid}  질문: {q['question']}")
    for did in [top.id] + [g for g in q["gold"]]:
        c = contrib(q["question"], did)
        tot = sum(c.values())
        best = sorted(c.items(), key=lambda kv: -kv[1])[:6]
        print(f"  {did} ({items[idx[did]].title[:30]}) 합={tot:.1f} 길이토큰={bm._lengths[idx[did]]} 상위기여: " + ", ".join(f"{t}:{v:.1f}" for t, v in best))
print("\n평균 문서 토큰 길이:", round(bm._avgdl, 1))
