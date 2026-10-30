"""블라인드 사람 채점 도구 (담당: C).

자동 루브릭 점수는 키워드 일치라서 의미를 재지 못한다. 그래서 일부 답변을 사람이 조건 이름을 모른 채 채점하고,
자동 점수와 얼마나 맞는지, 조건 사이에 사람 점수 차이가 있는지 본다.

흐름
1. export: 조건 이름을 지우고 순서를 섞은 채점표(sheet)와, 어느 답이 어느 조건인지 적힌 열쇠(key)를 만든다.
   같은 질문에서 두 조건이 똑같은 답을 냈으면 한 줄로 합쳐서 한 번만 채점한다(일관성을 위해서).
2. 채점자는 sheet의 score 열만 채운다. 0=틀림, 1=일부 맞음, 2=정답. key 파일은 채점이 끝나기 전에는 열지 않는다.
3. score: 채점표와 열쇠를 합쳐 조건별 사람 점수, 조건 간 쌍체 차이, 자동 점수와의 일치도를 계산한다.
   채점자가 둘이면 서로 얼마나 일치했는지(가중 카파)도 낸다.

생성 모델이 자기 답을 채점하게 하지 않는다. 사람이 채점한다.
"""

from __future__ import annotations

import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from kore.evaluation import bootstrap_ci, paired_bootstrap_diff, spearman
from kore.schemas import EvalQuestion
from kore.tokenize import normalize

SHEET_COLUMNS = ["item_id", "question", "reference_answer", "answer", "score", "note"]
KEY_COLUMNS = ["item_id", "question_id", "domain", "conditions", "rubric_score"]
SCORES = (0, 1, 2)


def export_blind(
    rows: list[dict[str, Any]],
    questions: list[EvalQuestion],
    conditions: list[str],
    n: int = 30,
    seed: int = 0,
    include_test: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(채점표 행, 열쇠 행)을 만든다. 모든 대상 조건의 답이 있는 질문 중에서 n개를 결정적으로 뽑는다."""
    if len(conditions) < 1:
        raise ValueError("조건이 하나 이상 필요하다")
    qmap = {q.question_id: q for q in questions}
    answers: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        if r["condition"] in conditions and r.get("answer") is not None:
            answers[r["question_id"]][r["condition"]] = r
    eligible = [
        qid
        for qid, by_cond in answers.items()
        if len(by_cond) == len(conditions) and qid in qmap and (include_test or qmap[qid].split != "test")
    ]
    if not eligible:
        raise ValueError(
            "채점할 답변이 없다. 대상 조건을 모두 돌렸는지, 생성기를 켰는지(--generator) 확인한다"
            + ("" if include_test else " (test 분할은 --final-test 없이는 제외된다)")
        )
    ranked = sorted(eligible, key=lambda qid: hashlib.sha256(f"{seed}|{qid}".encode()).hexdigest())
    chosen = ranked[: min(n, len(ranked))]

    entries: list[dict[str, Any]] = []
    for qid in chosen:
        q = qmap[qid]
        groups: dict[str, dict[str, Any]] = {}
        for cond in conditions:
            r = answers[qid][cond]
            key = normalize(r["answer"])
            g = groups.setdefault(key, {"answer": r["answer"], "conditions": [], "rubric": []})
            g["conditions"].append(cond)
            g["rubric"].append(r.get("rubric_score"))
        for g in groups.values():
            scores = [s for s in g["rubric"] if s is not None]
            entries.append(
                {
                    "question_id": qid,
                    "question": q.question,
                    "reference_answer": q.reference_answer,
                    "domain": q.domain,
                    "answer": g["answer"],
                    "conditions": g["conditions"],
                    "rubric_score": (sum(scores) / len(scores)) if scores else None,
                }
            )
    random.Random(seed).shuffle(entries)
    sheet, key = [], []
    for n_item, e in enumerate(entries, start=1):
        item_id = f"b{n_item:03d}"
        sheet.append(
            {"item_id": item_id, "question": e["question"], "reference_answer": e["reference_answer"],
             "answer": e["answer"], "score": "", "note": ""}
        )
        key.append(
            {"item_id": item_id, "question_id": e["question_id"], "domain": e["domain"],
             "conditions": ";".join(e["conditions"]),
             "rubric_score": "" if e["rubric_score"] is None else f"{e['rubric_score']:.4f}"}
        )
    return sheet, key


def _write(path: str | Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8-sig") as fh:  # 엑셀에서 한글이 깨지지 않게 BOM을 붙인다
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_sheet(path: str | Path, sheet: list[dict[str, Any]]) -> None:
    _write(path, SHEET_COLUMNS, sheet)


def write_key(path: str | Path, key: list[dict[str, Any]]) -> None:
    _write(path, KEY_COLUMNS, key)


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as fh:
        return [{(k or "").strip(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(fh)]


def parse_scores(sheet_rows: list[dict[str, str]]) -> tuple[dict[str, int], list[str]]:
    """채점표에서 item_id별 점수를 읽는다. 반환: (점수, 문제 목록). 비었거나 0/1/2가 아니면 문제로 적는다."""
    scores: dict[str, int] = {}
    problems: list[str] = []
    for r in sheet_rows:
        item = r.get("item_id", "")
        raw = r.get("score", "")
        if raw == "":
            problems.append(f"{item}: score가 비었다")
            continue
        try:
            value = int(float(raw))
        except ValueError:
            problems.append(f"{item}: score '{raw}'를 숫자로 읽을 수 없다")
            continue
        if value not in SCORES or float(raw) != value:
            problems.append(f"{item}: score는 0, 1, 2 중 하나여야 한다 (받은 값 {raw})")
            continue
        scores[item] = value
    return scores, problems


def weighted_kappa(a: list[int], b: list[int], categories: tuple[int, ...] = SCORES) -> float | None:
    """제곱 가중 카파. 두 채점자가 우연보다 얼마나 더 일치하는지. 한쪽이 한 가지 값만 줬으면 정의되지 않아 None."""
    if len(a) != len(b) or not a:
        return None
    k = len(categories)
    idx = {c: i for i, c in enumerate(categories)}
    n = len(a)
    obs = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[idx[x]][idx[y]] += 1.0 / n
    row = [sum(obs[i]) for i in range(k)]
    col = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = ((i - j) / (k - 1)) ** 2
            num += w * obs[i][j]
            den += w * row[i] * col[j]
    if den == 0:
        return None
    return 1.0 - num / den


def score_blind(
    sheets: list[list[dict[str, str]]], key_rows: list[dict[str, str]], seed: int = 0, baseline: str | None = None
) -> dict[str, Any]:
    """채점표(1~2명분)와 열쇠로 조건별 사람 점수를 낸다. 점수는 0~1(= score/2)로 바꿔서 자동 점수와 같은 척도로 본다."""
    if not 1 <= len(sheets) <= 2:
        raise ValueError("채점표는 1개 또는 2개(채점자 2명)를 받는다")
    parsed = [parse_scores(s) for s in sheets]
    problems = [p for _, probs in parsed for p in probs]
    key = {r["item_id"]: r for r in key_rows}
    unknown = sorted({i for sc, _ in parsed for i in sc} - set(key))
    if unknown:
        raise ValueError(f"열쇠에 없는 item_id가 있다: {unknown[:5]}")
    items = sorted(set.intersection(*[set(sc) for sc, _ in parsed]))
    if not items:
        raise ValueError("모든 채점자가 점수를 매긴 항목이 없다")
    human: dict[str, float] = {i: sum(sc[i] for sc, _ in parsed) / len(parsed) / 2.0 for i in items}

    out: dict[str, Any] = {"n_items": len(items), "problems": problems}
    if len(parsed) == 2:
        a = [parsed[0][0][i] for i in items]
        b = [parsed[1][0][i] for i in items]
        out["grader_agreement"] = {
            "exact_rate": sum(1 for x, y in zip(a, b) if x == y) / len(items),
            "weighted_kappa": weighted_kappa(a, b),
        }

    per_cond_q: dict[str, dict[str, float]] = defaultdict(dict)
    auto_pairs: list[tuple[float, float]] = []
    for i in items:
        qid = key[i]["question_id"]
        for cond in key[i]["conditions"].split(";"):
            per_cond_q[cond][qid] = human[i]
        if key[i].get("rubric_score", "") != "":
            auto_pairs.append((float(key[i]["rubric_score"]), human[i]))
    conds = sorted(per_cond_q)
    out["by_condition"] = {}
    for c in conds:
        mean, lo, hi = bootstrap_ci(list(per_cond_q[c].values()), seed=seed)
        out["by_condition"][c] = {"n_questions": len(per_cond_q[c]), "mean": mean, "ci_lo": lo, "ci_hi": hi}
    base = baseline if baseline in conds else ("vanilla_rag" if "vanilla_rag" in conds else None)
    out["baseline"] = base
    out["paired"] = {}
    if base:
        for c in conds:
            if c == base:
                continue
            shared = sorted(set(per_cond_q[c]) & set(per_cond_q[base]))
            out["paired"][c] = paired_bootstrap_diff(
                [per_cond_q[c][q] for q in shared], [per_cond_q[base][q] for q in shared], seed=seed
            )
    if auto_pairs:
        autos, humans = [p[0] for p in auto_pairs], [p[1] for p in auto_pairs]
        out["auto_vs_human"] = {
            "n": len(auto_pairs),
            "spearman": spearman(autos, humans),
            "mean_abs_diff": sum(abs(x - y) for x, y in auto_pairs) / len(auto_pairs),
        }
    return out


def format_blind_report(res: dict[str, Any]) -> str:
    lines = [f"채점된 항목 {res['n_items']}개"]
    if res["problems"]:
        lines.append(f"주의: 채점 문제 {len(res['problems'])}건 (예: {res['problems'][0]})")
    if "grader_agreement" in res:
        g = res["grader_agreement"]
        kappa = "계산 불가(한 쪽이 한 가지 점수만 줌)" if g["weighted_kappa"] is None else f"{g['weighted_kappa']:.2f}"
        lines.append(f"채점자 일치: 정확히 같은 비율 {g['exact_rate']:.0%}, 가중 카파 {kappa}")
    lines.append("")
    lines.append("조건별 사람 점수(0~1):")
    for c, v in res["by_condition"].items():
        lines.append(f"  {c:<14} {v['mean']:.3f}  (95% 구간 {v['ci_lo']:.3f} ~ {v['ci_hi']:.3f}, 질문 {v['n_questions']}개)")
    if res["paired"]:
        lines.append(f"기준선 {res['baseline']} 대비 쌍체 차이:")
        for c, d in res["paired"].items():
            if d["mean_diff"] is None:
                lines.append(f"  {c}: 계산할 수 없음")
            else:
                lines.append(f"  {c:<14} {d['mean_diff']:+.3f}  (95% 구간 {d['ci_lo']:+.3f} ~ {d['ci_hi']:+.3f}, 질문 {d['n']}개)")
    if "auto_vs_human" in res:
        a = res["auto_vs_human"]
        sp = "계산 불가" if a["spearman"] is None else f"{a['spearman']:.2f}"
        lines.append(f"자동 루브릭 점수와 사람 점수: 스피어만 {sp}, 평균 절대 차이 {a['mean_abs_diff']:.2f} (n={a['n']})")
    lines.append("")
    lines.append("표본이 작으면 구간이 넓다. 구간이 0을 포함하면 '차이가 없다'가 아니라 '구분하지 못했다'로 쓴다.")
    return "\n".join(lines)
