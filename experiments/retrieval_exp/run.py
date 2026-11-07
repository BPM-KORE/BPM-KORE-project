"""문서 확장 변형 + 후보 수 변화. kore는 읽기 전용 import. 실행: python3 -I experiments/retrieval_exp/run.py"""
from __future__ import annotations
import sys, re, csv, math, random, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kore.io import load_knowledge
from kore.retrieval import BM25Retriever
from kore.schemas import KnowledgeItem

OUT = pathlib.Path(__file__).resolve().parent
import os as _os
KB = pathlib.Path(_os.environ.get("KORE_KB", str(ROOT / "team/provisional/law/knowledge.csv")))
QS = ROOT / "experiments/eval_normalization/evaluation_questions_20_v2.csv"
KS = (5, 10, 20, 30)

items = load_knowledge(KB)
qs = []
with open(QS, encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        assert r["split"] == "dev", r["question_id"]
        r["gold"] = r["expected_source_ids"].split("|")
        qs.append(r)

# 본법 조문 색인: (가족, 조번호, 가지번호) -> 항목. 가족은 id에서 'law_' 뒤 ~ '_decree'/번호 앞.
def fam(i): return i.split("_")[1]
def is_decree(i): return "_decree_" in i
ART = re.compile(r"제(\d+)조(?:의(\d+))?")
parent = {}
for it in items:
    if is_decree(it.id): continue
    m = ART.search(it.title)
    if m: parent.setdefault((fam(it.id), int(m.group(1)), int(m.group(2) or 0)), it)
LAWNAME = {}
for it in items:
    if not is_decree(it.id):
        LAWNAME[fam(it.id)] = it.title[: ART.search(it.title).start()].strip()
# 시행령 본문의 '법 제N조(의M)' (바로 앞이 '법'이고 '시행' '같은' 등 영 참조가 아닌 것). 「타법」 제N조는 제외(보수적).
REF = re.compile(r"(?<![가-힣」])법 제(\d+)조(?:의(\d+))?|(?<=」) ?제(\d+)조(?:의(\d+))?")
REF = re.compile(r"(?<![가-힣])법 제(\d+)조(?:의(\d+))?")

def refs(it):
    out = []
    for m in REF.finditer(it.text):
        key = (fam(it.id), int(m.group(1)), int(m.group(2) or 0))
        p = parent.get(key)
        if p and p.id not in [x.id for x in out]: out.append(p)
    return out

def paren(title):
    m = re.search(r"\((.*)\)\s*$", title)
    return m.group(1) if m else title

R = {it.id: refs(it) for it in items if is_decree(it.id)}
n_dec = len(R); n_with = sum(1 for v in R.values() if v)
print(f"시행령 행 {n_dec}, 법 조문 참조 해소된 행 {n_with}, 평균 참조 수 {sum(len(v) for v in R.values())/max(n_with,1):.2f}")

def build(mode):
    if mode == "base": return items
    out = []
    for it in items:
        ps = R.get(it.id, [])
        if not ps: out.append(it); continue
        if mode == "D1":   # 법 이름 + 조 제목(괄호 안)
            add = " ".join(f"{LAWNAME[fam(it.id)]} {paren(p.title)}" for p in ps)
        elif mode == "D2": # 조 제목 전체(법 이름·조번호 포함)
            add = " ".join(p.title for p in ps)
        elif mode == "D3": # 조 제목 + 본법 조문 첫 200자
            add = " ".join(f"{p.title} {p.text[:200]}" for p in ps)
        out.append(it.copy_with(text=it.text + "\n" + add))
    return out

MODES = ["base", "D1", "D2", "D3"]
def ranks(mode):
    ret = BM25Retriever(build(mode))
    res = []
    for q in qs:
        order = [x.id for x in ret.retrieve(q["question"], k=len(items))]
        pos = {i: n + 1 for n, i in enumerate(order)}
        res.append({g: pos.get(g, 10**6) for g in q["gold"]})  # 점수 0 = 후보 밖
    return res

RK = {m: ranks(m) for m in MODES}
flat = lambda rk: [v for d in rk for v in d.values()]
def micro(rk, k): f = flat(rk); return sum(v <= k for v in f) / len(f)
def macro_q(rk, k): return [sum(v <= k for v in d.values()) / len(d) for d in rk]   # evaluation.recall_at_k 와 같은 정의(문항별)
def boot_micro_diff(a, b, k, n_boot=5000, seed=0):
    # 문항 단위 재표집(문항 안 정답 묶음 유지), micro 차이
    rng = random.Random(seed); n = len(a); ds = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        tot = sum(len(a[i]) for i in idx)
        ds.append((sum(v <= k for i in idx for v in a[i].values()) - sum(v <= k for i in idx for v in b[i].values())) / tot)
    ds.sort(); return ds[int(.025 * n_boot)], ds[int(.975 * n_boot)]
def boot_micro(a, k, n_boot=5000, seed=0):
    rng = random.Random(seed); n = len(a); ds = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        tot = sum(len(a[i]) for i in idx)
        ds.append(sum(v <= k for i in idx for v in a[i].values()) / tot)
    ds.sort(); return ds[int(.025 * n_boot)], ds[int(.975 * n_boot)]

lines = []
P = lambda s="": (print(s), lines.append(s))
P("## Recall@k micro (정답 44개 기준) [95% 문항 부트스트랩]")
P("| 변형 | " + " | ".join(f"R@{k}" for k in KS) + " | 정답 44개 중 @10 안 | 문항평균(macro) R@10 |")
P("|---|" + "---|" * (len(KS) + 2))
for m in MODES:
    cells = []
    for k in KS:
        lo, hi = boot_micro(RK[m], k)
        cells.append(f"{micro(RK[m], k):.3f} [{lo:.2f}, {hi:.2f}]")
    mq = macro_q(RK[m], 10)
    P(f"| {m} | " + " | ".join(cells) + f" | {sum(v<=10 for v in flat(RK[m]))} | {sum(mq)/len(mq):.3f} |")
P()
P("## 기준 대비 변화 (정답 단위, 같은 순서의 44개)")
P("| 변형 | 순위 좋아짐 | 나빠짐 | 같음 | 후보10 진입 | 후보10 이탈 | R@10 순증(개) | R@10 차이 [95%] | R@20 순증 | R@30 순증 |")
P("|---|---|---|---|---|---|---|---|---|---|")
base = RK["base"]
detail = []
for m in MODES[1:]:
    up = dn = eq = enter = leave = 0
    for qi, (a, b) in enumerate(zip(RK[m], base)):
        for g in a:
            if a[g] < b[g]: up += 1
            elif a[g] > b[g]: dn += 1
            else: eq += 1
            if a[g] <= 10 < b[g]: enter += 1
            if b[g] <= 10 < a[g]: leave += 1
            detail.append((m, qs[qi]["question_id"], g, b[g], a[g]))
    net = lambda k: sum(v <= k for v in flat(RK[m])) - sum(v <= k for v in flat(base))
    lo, hi = boot_micro_diff(RK[m], base, 10)
    P(f"| {m} | {up} | {dn} | {eq} | {enter} | {leave} | {net(10):+d} | {micro(RK[m],10)-micro(base,10):+.3f} [{lo:+.3f}, {hi:+.3f}] | {net(20):+d} | {net(30):+d} |")
P()
P("판정 기준 '탐색 통과' = R@10 순증 5개 이상. 판정: " + ", ".join(
    f"{m}={'통과' if sum(v<=10 for v in flat(RK[m]))-sum(v<=10 for v in flat(base))>=5 else '미달'}" for m in MODES[1:]))
(OUT / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
with open(OUT / "rank_changes.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f); w.writerow(["variant", "question", "gold", "base_rank", "new_rank"])
    for r in detail: w.writerow(["%s" % r[0], r[1], r[2], r[3] if r[3] < 10**6 else "OUT", r[4] if r[4] < 10**6 else "OUT"])
