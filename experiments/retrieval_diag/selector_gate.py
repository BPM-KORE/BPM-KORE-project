"""c) 검색 상한 vs Selector. 생성 없이 계산되는 지표만 쓴다."""
from __future__ import annotations
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import random
from common import *
from kore.selector import select_knowledge, SelectorWeights
from variants import run, MyBM25, family

items, qs, V, ranks, N = run()
by = {i.id: i for i in items}
base_names = list(V)
kb = BM25Retriever(items)

def macro(r, k):  # 문항 평균 recall (문항별 recall의 평균)
    return sum(sum(r[(q['question_id'], g)] <= k for g in q['gold']) / len(q['gold']) for q in qs) / len(qs)
b = ranks[base_names[0]]
print("기준 R@k  micro(44개 기준) / macro(문항 평균)")
for k in (5, 10, 20, 50):
    micro = sum(v <= k for v in b.values()) / len(b)
    print(f"  R@{k}: micro {micro:.3f} ({sum(v<=k for v in b.values())}/44)  macro {macro(b,k):.3f}")

print("\n오답 상위10 구성: 정답 법과 같은 법 가족인 비정답 / 다른 법")
same = diff = 0
for q in qs:
    fams = {family(g) for g in q["gold"]}
    for x in kb.retrieve(q["question"], k=10):
        if x.id in q["gold"]: continue
        if family(x.id) in fams: same += 1
        else: diff += 1
print(f"  같은 법 가족 {same}개, 다른 법 {diff}개 (문항 20개 x 후보 10개 중 비정답)")

def evaluate(cands_fn, label, ks=(3, 4, 5), tw_fn=None):
    print(f"\n[{label}] 후보 10 -> Selector")
    # 상한
    cand_hit = []
    for q in qs:
        cs = cands_fn(q); cand_hit.append(len(set(c.id for c in cs) & set(q["gold"])))
    ng = [len(q["gold"]) for q in qs]
    print(f"  후보10 안의 정답 {sum(cand_hit)}/{sum(ng)} (= 검색 단계 상한 R@10 {sum(cand_hit)/sum(ng):.3f}), 후보10에 정답이 하나도 없는 문항 {sum(h==0 for h in cand_hit)}/20")
    rows = {}
    for k in ks:
        oracle = [min(k, h) for h in cand_hit]  # 후보 10에서 고를 수 있는 이론 최대
        res = {}
        for strat in ("similarity", "kore"):
            hits = []
            for q in qs:
                cs = cands_fn(q)
                tw = tw_fn(q) if tw_fn else kb.term_weights(q["question"])
                sel = select_knowledge(q["question"], cs, k=k, strategy=strat, weights=SelectorWeights(), term_weights=tw)
                hits.append(len({s.id for s in sel} & set(q["gold"])))
            res[strat] = hits
        # 전체 정답 수 대비 건진 비율, 고른 k개 중 정답 비율
        tot = sum(ng)
        print(f"  k={k}: 이론상한(후보10,k개) {sum(oracle)}/{tot}={sum(oracle)/tot:.3f} | similarity {sum(res['similarity'])}/{tot}={sum(res['similarity'])/tot:.3f} | kore {sum(res['kore'])}/{tot}={sum(res['kore'])/tot:.3f} | 상한 대비 sim {sum(res['similarity'])/sum(oracle):.2f}, kore {sum(res['kore'])/sum(oracle):.2f}")
        print(f"        문항 단위 '정답 1개 이상 포함': 후보10 {sum(h>0 for h in cand_hit)}/20, similarity {sum(h>0 for h in res['similarity'])}/20, kore {sum(h>0 for h in res['kore'])}/20")
        d = [(q['question_id'], res['similarity'][i], res['kore'][i], oracle[i]) for i, q in enumerate(qs) if res['similarity'][i] != res['kore'][i]]
        print(f"        sim과 kore가 다른 문항: " + (", ".join(f"{a}(sim {s}, kore {kk}, 상한 {o})" for a, s, kk, o in d) or "없음"))
        rows[k] = res
    return rows

# 기준 BM25 후보
evaluate(lambda q: kb.retrieve(q["question"], k=10), "기준 BM25 (실제 파이프라인 후보)")

# 개선 검색 후보(제목x3 + 예측 법 부스트 아님: 단순히 ngram 변형과 E2) - 가장 낮은 위험 변형과 과적합 상한
def cands_from(bm, qfn=lambda s: s):
    def f(q):
        order = bm.order(qfn(q["question"]))[:10]
        sc = bm.scores(qfn(q["question"]))
        idx = {it.id: n for n, it in enumerate(bm.items)}
        return [by[i].copy_with(score=sc[idx[i]], retrieval_score=sc[idx[i]]) for i in order]
    return f
from variants import expand
ng = MyBM25(items, tok=lambda s: tokenize(s, "ngram"), title_w=3)
evaluate(cands_from(ng), "변형: tok=ngram + 제목x3 (과적합 위험 낮은 쪽, 잡음 수준)",
         tw_fn=lambda q: {t: ng.idf[t] for t in set(tokenize(q['question'], 'ngram')) if t in ng.idf})
e2 = MyBM25(items, title_w=2)
evaluate(cands_from(e2, expand), "변형: 수동 질의확장 + 제목x2 (20문항에 맞춘 상한 추정, 과적합)",
         tw_fn=lambda q: {t: e2.idf[t] for t in set(tokenize(expand(q['question']))) if t in e2.idf})
