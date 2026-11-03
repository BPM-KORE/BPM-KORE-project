"""a) 10위 밖 정답 근거의 원인 진단. 출력만 한다."""
from __future__ import annotations
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import *

items, qs = load()
by = {i.id: i for i in items}
bm = BM25Retriever(items)
N = len(items)
total = 0; ranks_all = []
for q in qs:
    full = bm.retrieve(q["question"], k=N)
    order = [x.id for x in full]
    rk = rank_of(order, q["gold"])
    ranks_all += [(q["question_id"], g, r) for g, r in rk.items()]
    qt = set(tokenize(q["question"]))
    print(f"\n=== {q['question_id']} 순위: " + ", ".join(f"{g}={r}" for g, r in rk.items()))
    for g, r in rk.items():
        if r is not None and r <= 10:
            continue
        it = by[g]
        dt = set(tokenize(f"{it.title} {it.text}"))
        ov = sorted(qt & dt, key=lambda t: -bm._idf.get(t, 0))
        # 질문 어휘 중 코퍼스에는 있으나 해당 조문에 없는 상위 IDF 어절(원형 토큰만, 2-gram 제외 어려우므로 길이>=2 전체)
        miss = sorted((t for t in qt - dt if t in bm._idf), key=lambda t: -bm._idf[t])[:8]
        print(f"  [10위 밖] {g} rank={r} | {it.title} | len={len(dt)}")
        print(f"    겹친 토큰({len(ov)}/{len(qt)}): {ov[:12]}")
        print(f"    조문에 없는 질문 토큰(idf순): {miss}")
        if g.endswith(tuple('0123456789')) and '_decree_' in g:
            print("    (시행령 조문)")
    print("  상위 5:")
    for n, x in enumerate(full[:5], 1):
        mark = "*" if x.id in q["gold"] else " "
        print(f"   {mark}{n}. {x.id} | {x.title} | {x.score:.2f}")
