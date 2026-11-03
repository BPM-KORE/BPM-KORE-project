"""b) 검색 변형 비교. kore는 읽기 전용 import만 하고 BM25는 여기서 다시 구현한다(표준 라이브러리만)."""
from __future__ import annotations
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import math, random, re, csv, json
from collections import Counter, defaultdict
from common import *
from kore.tokenize import normalize

OUT = pathlib.Path(__file__).resolve().parent

class MyBM25:
    def __init__(self, items, k1=1.5, b=0.75, tok=lambda s: tokenize(s, "ngram_stem"), title_w=1):
        self.items, self.k1, self.b, self.tok = items, k1, b, tok
        self.len, self.post = [], defaultdict(list)
        for n, it in enumerate(items):
            toks = tok(f"{it.title} {it.text}") + tok(it.title) * (title_w - 1)
            self.len.append(len(toks))
            for t, tf in Counter(toks).items():
                self.post[t].append((n, tf))
        N = len(items); self.avg = sum(self.len) / N
        self.idf = {t: math.log(1 + (N - len(p) + .5) / (len(p) + .5)) for t, p in self.post.items()}
    def scores(self, q):
        s = defaultdict(float)
        for t in dict.fromkeys(self.tok(q)):
            p = self.post.get(t)
            if not p: continue
            for n, tf in p:
                norm = self.k1 * (1 - self.b + self.b * self.len[n] / self.avg)
                s[n] += self.idf[t] * tf * (self.k1 + 1) / (tf + norm)
        return s
    def order(self, q, boost=None):
        s = self.scores(q)
        if boost:
            for n in list(s): s[n] *= boost(self.items[n])
        return [self.items[n].id for n, v in sorted(s.items(), key=lambda kv: (-kv[1], kv[0])) if v > 0]

def family(i):  # law_pipa_decree_020 -> pipa
    return i.split("_")[1]
LAWNAME = {"pipa": "개인정보 보호법", "foia": "공공기관의 정보공개에 관한 법률",
           "opendata": "공공데이터의 제공 및 이용 활성화에 관한 법률", "egov": "전자정부법",
           "ictconstruction": "정보통신공사업법", "pubinst": "공공기관의 운영에 관한 법률"}

# 20문항 진단에서 직접 뽑은 수동 사전이다. 과적합 위험이 크다(상한 추정용).
EXPAND = [("건강", "민감정보"), ("꼭 필요하지", "최소한 수집"), ("더 이상 필요하지", "파기"),
          ("고치거나", "정정"), ("확인하고", "열람"), ("공개 가능한", "부분 공개"),
          ("청구할 수", "청구방법"), ("이름과 연락처", "개인정보"), ("다른 업체에서", "수집 출처 통지"),
          ("외부 업체에 맡기", "위탁")]
def expand(q):
    extra = [e for k, e in EXPAND if k in q]
    return q + " " + " ".join(extra) if extra else q

def bigram_only(text):
    out = []
    for w in re.findall(r"[가-힣]+|[A-Za-z0-9]+", normalize(text)):
        if re.fullmatch(r"[가-힣]+", w) and len(w) >= 2:
            out += [w[i:i+2] for i in range(len(w)-1)]
        else:
            out.append(w)
    return out

def run():
    items, qs = load()
    ids = [i.id for i in items]; N = len(ids)
    base = MyBM25(items)
    # kore 구현과 순위가 같은지 확인
    kb = BM25Retriever(items)
    for q in qs:
        a = [x.id for x in kb.retrieve(q["question"], k=N)]
        assert a == base.order(q["question"]), q["question_id"]
    cache = {}
    def mk(**kw):
        key = json.dumps({k: (v if isinstance(v, (int, float)) else str(v)) for k, v in kw.items()})
        return MyBM25(items, **kw)

    def routed_family(q, topn=3, nfam=1):
        s = base.scores(q); per = defaultdict(list)
        for n, v in s.items(): per[family(items[n].id)].append(v)
        tot = {f: sum(sorted(v, reverse=True)[:topn]) for f, v in per.items()}
        return [f for f, _ in sorted(tot.items(), key=lambda kv: -kv[1])[:nfam]]

    V = {}  # name -> (order_fn)
    def add(name, bm, qfn=lambda q: q["question"], boost=None):
        V[name] = lambda q, bm=bm, qfn=qfn, boost=boost: bm.order(qfn(q), boost(q) if boost else None)
    add("0 기준(ngram_stem k1=1.5 b=.75)", base)
    add("T1 tok=ngram", mk(tok=lambda s: tokenize(s, "ngram")))
    add("T2 tok=word", mk(tok=lambda s: tokenize(s, "word")))
    add("T3 tok=bigram_only(kore 밖)", mk(tok=bigram_only))
    for k1 in (0.9, 1.2, 2.0):
        add(f"P k1={k1}", mk(k1=k1))
    for b in (0.3, 0.5, 0.9):
        add(f"P b={b}", mk(b=b))
    add("P k1=1.2 b=0.5", mk(k1=1.2, b=0.5))
    add("P k1=2.0 b=0.9", mk(k1=2.0, b=0.9))
    for w in (2, 3):
        add(f"TW 제목x{w}", mk(title_w=w))
    # 법령명 힌트: 예측(라우팅 1위 법) / oracle(정답 법)
    add("L1 법령명 힌트-예측 top1", base, qfn=lambda q: q["question"] + " " + LAWNAME[routed_family(q["question"], 3, 1)[0]])
    add("L2 법령명 힌트-예측 top2", base, qfn=lambda q: q["question"] + " " + " ".join(LAWNAME[f] for f in routed_family(q["question"], 3, 2)))
    add("L3 법령명 힌트-oracle(정답 법)", base, qfn=lambda q: q["question"] + " " + " ".join(LAWNAME[f] for f in sorted({family(g) for g in q["gold"]})))
    add("L4 법 부스트x1.5-예측 top1(점수 곱)", base, boost=lambda q: (lambda it, f=routed_family(q["question"],3,1)[0]: 1.5 if family(it.id) == f else 1.0))
    add("L5 법 부스트x1.5-oracle", base, boost=lambda q: (lambda it, fs={family(g) for g in q["gold"]}: 1.5 if family(it.id) in fs else 1.0))
    add("E1 수동 질의확장(10쌍, 과적합)", base, qfn=lambda q: expand(q["question"]))
    add("E2 확장+제목x2", mk(title_w=2), qfn=lambda q: expand(q["question"]))
    add("C1 제목x2 + k1=1.2 b=0.5", mk(title_w=2, k1=1.2, b=0.5))
    add("C2 E1+L1(과적합)", base, qfn=lambda q: expand(q["question"]) + " " + LAWNAME[routed_family(q["question"],3,1)[0]])

    ranks = {}  # name -> {(qid,gold): rank}
    for name, fn in V.items():
        r = {}
        for q in qs:
            order = fn(q); pos = {i: n + 1 for n, i in enumerate(order)}
            for g in q["gold"]: r[(q["question_id"], g)] = pos.get(g, N + 1)
        ranks[name] = r
    return items, qs, V, ranks, N

def recall(r, k, qs=None, qids=None):
    keys = [x for x in r if (qids is None or x[0] in qids)]
    return sum(r[x] <= k for x in keys) / len(keys)

if __name__ == "__main__":
    items, qs, V, ranks, N = run()
    names = list(V); b = ranks[names[0]]
    gold_keys = list(b)
    print(f"gold 근거 수 {len(gold_keys)}, 문항 {len(qs)}")
    print(f"{'변형':<44} R@5   R@10  R@20  R@50  | 문항단위 any@10 all@10 | 평균 순위(중앙)")
    for n in names:
        r = ranks[n]
        anyk = sum(any(r[(q['question_id'], g)] <= 10 for g in q['gold']) for q in qs)
        allk = sum(all(r[(q['question_id'], g)] <= 10 for g in q['gold']) for q in qs)
        rs = sorted(r.values()); med = rs[len(rs)//2]
        print(f"{n:<44} {recall(r,5):.2f}  {recall(r,10):.2f}  {recall(r,20):.2f}  {recall(r,50):.2f}  | {anyk:>2}/20 {allk:>2}/20 | {med}")
    # 부트스트랩: 문항 단위 재표집으로 R@10 차이의 95% 구간
    rng = random.Random(0); qids = [q["question_id"] for q in qs]
    print("\n기준 대비 R@10 차이 (문항 단위 부트스트랩 2000회, 95% 구간) -> 0을 포함하면 잡음과 구별 안 됨")
    def r10_by_q(r):
        return {qid: (sum(r[(qid, g)] <= 10 for g in next(q for q in qs if q['question_id']==qid)['gold']), len(next(q for q in qs if q['question_id']==qid)['gold'])) for qid in qids}
    bq = r10_by_q(b)
    for n in names[1:]:
        vq = r10_by_q(ranks[n]); diffs = []
        for _ in range(2000):
            s = [rng.choice(qids) for _ in qids]
            tot = sum(bq[x][1] for x in s)
            diffs.append((sum(vq[x][0] for x in s) - sum(bq[x][0] for x in s)) / tot)
        diffs.sort(); pt = (sum(v[0] for v in vq.values()) - sum(v[0] for v in bq.values())) / len(gold_keys)
        print(f"{n:<44} 차이 {pt:+.3f}  [{diffs[50]:+.3f}, {diffs[1949]:+.3f}]")
    # 문항별 순위 표 (csv + 주요 변형 출력)
    with open(OUT / "rank_table.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["question_id", "gold_id"] + names)
        for k in gold_keys: w.writerow(list(k) + [ranks[n][k] for n in names])
    show = [n for n in names if n.split()[0] in ("0", "T1", "T2", "P", "TW", "L1", "L3", "E1", "C1")][:0]
    pick = ["0 기준(ngram_stem k1=1.5 b=.75)", "T2 tok=word", "P k1=1.2 b=0.5", "TW 제목x2", "L1 법령명 힌트-예측 top1", "L3 법령명 힌트-oracle(정답 법)", "E1 수동 질의확장(10쌍, 과적합)"]
    print("\n문항별 정답 순위 (777=후보 밖)  열: " + " | ".join(p.split()[0] + (" "+p.split()[1] if p.startswith(('P','TW')) else '') for p in pick))
    for k in gold_keys:
        if any(ranks[p][k] > 10 for p in pick):
            print(f"{k[0]} {k[1]:<26} " + " ".join(f"{ranks[p][k]:>4}" for p in pick))
