"""검색 재현율 탐색(CPU, dev 20문항). 문서 쪽만 확장하고 질의는 그대로 둔다.
실행: python3 -I experiments/retrieval_exp/run_exp.py
kore/, data/, team/ 은 읽기만 한다. test 분할은 열지 않는다(질문 파일의 split이 dev뿐임을 assert)."""
from __future__ import annotations
import csv, random, re, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kore.io import load_knowledge  # noqa: E402
from kore.retrieval import BM25Retriever  # noqa: E402

import os as _os
KB = Path(_os.environ.get("KORE_KB", str(ROOT / "team/provisional/law/knowledge.csv")))
QS = ROOT / "experiments/eval_normalization/evaluation_questions_20_v2.csv"
OUT = Path(__file__).resolve().parent
KS = (5, 10, 20, 30)
ABSENT = 10**6  # 점수 0이라 후보에 없음

items = load_knowledge(KB)
with open(QS, encoding="utf-8-sig", newline="") as f:
    qs = list(csv.DictReader(f))
assert len(qs) == 20 and {q["split"] for q in qs} == {"dev"}
for q in qs:
    q["gold"] = q["expected_source_ids"].split("|")

# ---- 법 조문 사전 ---------------------------------------------------------
TITLE_RE = re.compile(r"^(?P<law>.+?)\s*제(?P<no>\d+)조(?P<sub>의\d+)?\((?P<name>[^)]*)\)")
def norm(s: str) -> str:
    return re.sub(r"\s+", "", s)
laws: dict[tuple[str, str], tuple[str, str]] = {}
law_text: dict[tuple[str, str], str] = {}  # (정규화 법명, "33" 또는 "24의2") -> (법명, 조문제목)
for it in items:
    m = TITLE_RE.match(it.title)
    if m and not m["law"].endswith("시행령"):
        laws[(norm(m["law"]), m["no"] + (m["sub"] or ""))] = (m["law"], m["name"])
        law_text[(norm(m["law"]), m["no"] + (m["sub"] or ""))] = it.text
def parent_law(it) -> str | None:
    m = TITLE_RE.match(it.title)
    if m and m["law"].endswith("시행령"):
        return norm(m["law"][: -len("시행령")])
    return None

REF_RE = re.compile(r"(?:「(?P<law>[^」]+)」\s*|(?P<bare>(?<![가-힣])법)\s*)제(?P<no>\d+)조(?P<sub>의\d+)?")
def refs(it) -> list[tuple[str, str]]:
    """시행령 행이 인용한 (법명, 조문제목) 목록. 등장 순서, 중복 제거. 사전에 없는 조문은 버린다."""
    parent = parent_law(it)
    if parent is None:
        return []
    out = []
    for m in REF_RE.finditer(it.text):
        law = norm(m["law"]) if m["law"] else parent
        hit = laws.get((law, m["no"] + (m["sub"] or "")))
        if hit and (hit, m["no"] + (m["sub"] or "")) not in [(o[:2], o[2]) for o in out]:
            out.append((hit[0], hit[1], m["no"] + (m["sub"] or "")))
    return out

ref_map = {it.id: refs(it) for it in items}
n_dec = sum(1 for it in items if parent_law(it))
n_ref = sum(1 for v in ref_map.values() if v)

def expand(it, mode: str) -> str:
    r = ref_map[it.id]
    if not r:
        return it.text
    if mode.endswith("_top1"):
        cnt = Counter(it_[2] for it_ in refs_all[it.id])
        r = [max(r, key=lambda x: (cnt[x[2]], -r.index(x)))]
    if mode == "fulltext_top1":
        r = [max(r, key=lambda x: (Counter(i[2] for i in refs_all[it.id])[x[2]], -r.index(x)))]
        return it.text + "\n[위임 근거] " + " ".join(law_text[(norm(x[0]), x[2])] for x in r)
    if mode.startswith("title"):
        add = " ".join(x[1] for x in r)
    elif mode.startswith("lawtitle"):
        add = " ".join(f"{x[0]} 제{x[2].replace('의', '조의') if '의' in x[2] else x[2] + '조'}({x[1]})" for x in r)
    else:
        raise ValueError(mode)
    return it.text + "\n[위임 근거] " + add

# top1 용: 인용 횟수(중복 포함) 계산
refs_all: dict[str, list] = {}
for it in items:
    parent = parent_law(it)
    lst = []
    if parent:
        for m in REF_RE.finditer(it.text):
            law = norm(m["law"]) if m["law"] else parent
            key = m["no"] + (m["sub"] or "")
            hit = laws.get((law, key))
            if hit:
                lst.append((hit[0], hit[1], key))
    refs_all[it.id] = lst

def build(mode: str | None) -> BM25Retriever:
    if mode is None:
        return BM25Retriever(items)
    return BM25Retriever([it.copy_with(text=expand(it, mode)) for it in items])

VARIANTS = [
    ("0 기준(확장 없음)", None),
    ("D1 법 조문 제목만 덧붙임(인용 전부)", "title"),
    ("D2 법명+조번호+제목 덧붙임(인용 전부)", "lawtitle"),
    ("D3 D1을 가장 많이 인용된 조문 1개만", "title_top1"),
    ("D4 D2를 가장 많이 인용된 조문 1개만", "lawtitle_top1"),
    ("D5 가장 많이 인용된 조문 1개의 본문 전체", "fulltext_top1"),
]

def ranks(rt: BM25Retriever) -> dict[tuple[str, str], int]:
    out = {}
    for q in qs:
        order = [c.id for c in rt.retrieve(q["question"], k=len(items))]
        pos = {i: n + 1 for n, i in enumerate(order)}
        for g in q["gold"]:
            out[(q["question_id"], g)] = pos.get(g, ABSENT)
    return out

def recall_macro(rk, k, idx=None):
    vals = []
    for q in qs if idx is None else [qs[i] for i in idx]:
        vals.append(sum(rk[(q["question_id"], g)] <= k for g in q["gold"]) / len(q["gold"]))
    return sum(vals) / len(vals)

def recall_micro(rk, k):
    keys = list(rk)
    return sum(rk[x] <= k for x in keys) / len(keys)

def boot_diff(rb, rv, k, n=5000, seed=0):
    rnd = random.Random(seed)
    d = []
    for _ in range(n):
        idx = [rnd.randrange(len(qs)) for _ in qs]
        d.append(recall_macro(rv, k, idx) - recall_macro(rb, k, idx))
    d.sort()
    return d[int(0.025 * n)], d[int(0.975 * n) - 1]

def boot_ci(rk, k, n=5000, seed=0):
    rnd = random.Random(seed)
    d = sorted(recall_macro(rk, k, [rnd.randrange(len(qs)) for _ in qs]) for _ in range(n))
    return d[int(0.025 * n)], d[int(0.975 * n) - 1]

lines = []
P = lines.append
P(f"시행령 행 {n_dec}개 중 법 조문 인용을 사전에서 찾은 행 {n_ref}개. 조문 사전 {len(laws)}개.")
res = {}
for name, mode in VARIANTS:
    res[name] = ranks(build(mode))
base = res[VARIANTS[0][0]]
gold_n = len(base)

P("")
P("표1. Recall@k (문항 평균 macro = kore stats의 '후보 재현율' 정의 / 근거 44개 전체 micro). 괄호는 macro의 문항 부트스트랩 95% 구간(5000회, seed 0)")
P("| 변형 | R@5 | R@10 | R@20 | R@30 | R@10 macro 95% 구간 | R@30 micro |")
P("|---|---|---|---|---|---|---|")
for name, _ in VARIANTS:
    rk = res[name]
    lo, hi = boot_ci(rk, 10)
    P("| " + name + " | " + " | ".join(f"{recall_macro(rk,k):.3f}" for k in KS) + f" | [{lo:.3f}, {hi:.3f}] | {recall_micro(rk,30):.3f} |")
P("")
P("표2. 기준 대비 정답 근거(44개)별 순위 변화와 후보 안 순증(micro 개수). 순위 상승은 순위 숫자가 작아진 것이다.")
P("| 변형 | 순위 상승(숫자 작아짐) | 순위 하락(숫자 커짐) | 같음 | 상위10 순증 | 상위20 순증 | 상위30 순증 | R@10 차이 macro [95% 구간] | R@20 차이 [95% 구간] |")
P("|---|---|---|---|---|---|---|---|---|")
for name, _ in VARIANTS[1:]:
    rk = res[name]
    up = sum(rk[x] < base[x] for x in base); dn = sum(rk[x] > base[x] for x in base)
    eq = gold_n - up - dn
    net = {k: sum(rk[x] <= k for x in base) - sum(base[x] <= k for x in base) for k in (10, 20, 30)}
    d10 = recall_macro(rk, 10) - recall_macro(base, 10); lo10, hi10 = boot_diff(base, rk, 10)
    d20 = recall_macro(rk, 20) - recall_macro(base, 20); lo20, hi20 = boot_diff(base, rk, 20)
    P(f"| {name} | {up} | {dn} | {eq} | {net[10]:+d} | {net[20]:+d} | {net[30]:+d} | {d10:+.3f} [{lo10:+.3f}, {hi10:+.3f}] | {d20:+.3f} [{lo20:+.3f}, {hi20:+.3f}] |")
P("")
P("표3. 후보 수만 바꿀 때(기준 BM25, 순위는 그대로) 정답 근거 44개 중 후보 안에 든 개수")
P("| candidate_k | 개수 | micro | macro | 후보에 못 든 근거 |")
P("|---|---|---|---|---|")
for k in (10, 20, 30, 50):
    n = sum(base[x] <= k for x in base)
    P(f"| {k} | {n} | {n/gold_n:.3f} | {recall_macro(base,k):.3f} | {gold_n-n} |")
P("")
P("표4. 정답 근거별 순위(기준 -> 변형). 점수 0이라 후보 밖이면 '-'")
names = [v[0] for v in VARIANTS]
P("| 문항 | 근거 id | " + " | ".join(n.split()[0] for n in names) + " |")
P("|---|---|" + "---|" * len(names))
f = lambda r: "-" if r >= ABSENT else str(r)
for (qid, g) in base:
    row = [res[n][(qid, g)] for n in names]
    if len(set(row)) > 1:
        P(f"| {qid} | {g} | " + " | ".join(f(r) for r in row) + " |")
P("")
P("표5. 정답 근거 중 확장 대상(시행령 행 + 인용 사전 적중) 현황")
tgt = [x for x in base if ref_map.get(x[1])]
P(f"정답 근거 {gold_n}개 중 시행령이고 인용을 찾은 것: {len(tgt)}개. 그중 기준에서 상위10 밖: {sum(base[x] > 10 for x in tgt)}개.")
P("")
P("부록. 확장 예(정답 근거에 해당하는 행)")
for (qid, g) in tgt:
    P(f"- {g}: " + "; ".join(f"{a} 제{c.replace('의','조의') if '의' in c else c+'조'}({b})" for a, b, c in ref_map[g]))
txt = "\n".join(lines)
(OUT / "result.txt").write_text(txt + "\n", encoding="utf-8")
print(txt)
