import json

import pytest

from kore.labeling import (
    analyze_labels, analyze_question, count_calls, enumerate_subsets, format_report, label_question, run_labeling,
    subset_key,
)
from kore.schemas import EvalQuestion
from tests.helpers import mk_item


class FakeScorer:
    """부분집합 점수를 미리 정한 표로 돌려주는 점수기. 호출 횟수를 센다."""

    def __init__(self, table=None, name="fake-scorer", fn=None):
        self.name = name
        self.table = table or {}
        self.fn = fn
        self.calls = 0

    def score(self, question, evidence, answer):
        self.calls += 1
        key = subset_key([e.id for e in evidence])
        if self.fn is not None:
            return self.fn(key)
        return self.table[key]


def make_q(qid="q1", split="train", ref="답"):
    return EvalQuestion(qid, "질문", ["a"], "d", "easy", split, reference_answer=ref, rubric="답")


# ---------------------------------------------------------------- 부분집합 계산


def test_subset_key_ignores_order_and_empty_is_empty_string():
    assert subset_key(["b", "a"]) == subset_key(["a", "b"]) == "a|b"
    assert subset_key([]) == ""


def test_enumerate_subsets_and_call_counts_match_the_report_table():
    subs = enumerate_subsets(["a", "b", "c"], 2)
    assert subs[0] == () and ("a", "b") in subs and ("a", "b", "c") not in subs and len(subs) == 7
    # docs/REPORT의 표: 후보 6/8/10개, 크기 2까지 22/37/56, 크기 3까지 42/93/176
    assert [count_calls(n, 2) for n in (6, 8, 10)] == [22, 37, 56]
    assert [count_calls(n, 3) for n in (6, 8, 10)] == [42, 93, 176]
    assert len(enumerate_subsets([str(i) for i in range(8)], 3)) == 93


def test_label_question_scores_every_subset_in_retrieval_order():
    cands = [mk_item("a", "x"), mk_item("b", "y"), mk_item("c", "z")]
    seen = []

    def fn(key):
        seen.append(key)
        return -float(len(key))

    scorer = FakeScorer(fn=fn)
    rec = label_question(make_q(), cands, scorer, max_cands=3, max_subset=2)
    assert scorer.calls == 7 and set(rec["scores"]) == {"", "a", "b", "c", "a|b", "a|c", "b|c"}
    assert rec["candidate_ids"] == ["a", "b", "c"]
    # 증거는 검색 순위순으로 건네진다(키는 정렬되므로 같은 집합은 한 번만 계산된다)
    assert len(seen) == len(set(seen))


def test_label_question_caps_candidates_and_requires_reference():
    cands = [mk_item(f"c{i}", "t") for i in range(6)]
    scorer = FakeScorer(fn=lambda key: 0.0)
    rec = label_question(make_q(), cands, scorer, max_cands=4, max_subset=1)
    assert rec["candidate_ids"] == ["c0", "c1", "c2", "c3"] and scorer.calls == 5
    with pytest.raises(ValueError, match="reference_answer"):
        label_question(make_q(ref=""), cands, scorer)


# ---------------------------------------------------------------- 분석


def row_from(base, singles, pairs=None, ids=("a", "b", "c"), gold=("a",)):
    scores = {"": base}
    for i, g in zip(ids, singles):
        scores[i] = base + g
    for key, extra in (pairs or {}).items():
        a, b = key.split("|")
        scores[key] = scores[a] + scores[b] - base + extra  # extra = 시너지
    return {"question_id": "q", "domain": "d", "split": "train", "candidate_ids": list(ids),
            "expected_source_ids": list(gold), "scores": scores}


def test_analysis_detects_rank_gain_disagreement():
    # 검색 1위(a)는 기여가 작고 2위(b)가 가장 크다. c는 해롭다.
    row = row_from(-5.0, [0.1, 0.5, -0.2], pairs={"a|b": 0.3, "a|c": 0.0, "b|c": 0.0})
    m = analyze_question(row, k=2)
    assert m["top1_mismatch"] == 1.0
    assert m["spearman_rank_vs_gain"] == pytest.approx(0.5)  # 순위(3,2,1 의미) vs 이득(0.1,0.5,-0.2)
    assert m["frac_candidates_harmful"] == pytest.approx(1 / 3)
    # 최선의 쌍은 a|b(0.1+0.5+0.3=0.9), 유사도 상위 2개도 a|b라서 손실이 없다
    assert m["oracle_gap_at_k"] == pytest.approx(0.0)
    assert m["mean_synergy"] == pytest.approx(0.1)
    assert m["best_is_gold"] == 0.0  # 최고 기여 문서는 b인데 정답 근거는 a


def test_analysis_oracle_gap_when_similarity_top_k_is_worse():
    row = row_from(0.0, [0.1, 0.0, 0.6], pairs={"a|b": 0.0, "a|c": 0.0, "b|c": 0.0})
    m = analyze_question(row, k=2)
    # 유사도 상위 2개 = a|b = 0.1, 최선 = a|c 또는 b|c ... a|c=0.7
    assert m["oracle_gap_at_k"] == pytest.approx(0.7 - 0.1)
    assert m["top1_mismatch"] == 1.0 and m["top1_mismatch_null"] == pytest.approx(2 / 3)
    # 쌍 점수 a|b=0.1, a|c=0.7, b|c=0.6 → 무작위 평균 0.4667, 최선 0.7, 유사도 0.1 → 무작위보다도 나쁘다
    assert m["capture_ratio"] == pytest.approx((0.1 - 1.4 / 3) / (0.7 - 1.4 / 3))
    assert m["capture_ratio"] < 0


def test_analysis_agreeing_rankings_and_additive_documents():
    row = row_from(0.0, [0.9, 0.5, 0.1], pairs={"a|b": 0.0, "a|c": 0.0, "b|c": 0.0})
    m = analyze_question(row, k=2)
    assert m["spearman_rank_vs_gain"] == pytest.approx(1.0)
    assert m["top1_mismatch"] == 0.0 and m["oracle_gap_at_k"] == pytest.approx(0.0)
    assert m["capture_ratio"] == pytest.approx(1.0)  # 유사도 상위 k가 곧 최선이다
    assert m["mean_abs_synergy"] == pytest.approx(0.0) and m["synergy_to_single_ratio"] == pytest.approx(0.0)
    assert m["best_is_gold"] == 1.0


def test_analysis_handles_constant_scores_without_crashing():
    row = row_from(1.0, [0.0, 0.0, 0.0], pairs={"a|b": 0.0, "a|c": 0.0, "b|c": 0.0})
    m = analyze_question(row, k=2)
    assert m["spearman_rank_vs_gain"] is None and m["top1_mismatch"] == 0.0
    assert m["capture_ratio"] is None  # 최선과 무작위가 같으면 위치를 정의할 수 없다
    assert m["synergy_to_single_ratio"] is None


def test_add_hurts_best_detects_interference():
    # b가 최고. b에 a를 더하면 b 혼자보다 나빠진다(간섭)
    row = row_from(0.0, [0.1, 0.8, 0.0], pairs={"a|b": -0.5, "b|c": 0.0, "a|c": 0.0})
    m = analyze_question(row, k=2)
    assert m["frac_add_hurts_best"] == pytest.approx(0.5)  # 후보 a는 해치고 c는 아니다


def test_analyze_labels_summary_and_report_text():
    rows = [row_from(0.0, [0.9, 0.5, 0.1], pairs={"a|b": 0.0, "a|c": 0.0, "b|c": 0.0}),
            row_from(0.0, [0.1, 0.5, 0.0], pairs={"a|b": 0.1, "a|c": 0.0, "b|c": 0.0})]
    res = analyze_labels(rows, k=2)
    s = res["summary"]
    assert s["n_questions"] == 2 and s["top1_mismatch"]["mean"] == pytest.approx(0.5)
    text = format_report(res)
    assert "예비 신호" in text and "순위상관" in text and "기대값" in text and "증거로 쓰지 않음" in text


# ---------------------------------------------------------------- 러너


def test_run_labeling_writes_resumes_and_guards(tmp_path, items, questions):
    scorer = FakeScorer(fn=lambda key: -float(len(key)))
    qs = [q for q in questions if q.reference_answer.strip()]
    out = tmp_path / "dq"
    res = run_labeling(items, qs, out, scorer, max_cands=4, max_subset=2)
    assert res["n_labeled"] > 0
    assert all(r["split"] in ("train", "dev") for r in res["rows"])
    first_calls = scorer.calls
    again = run_labeling(items, qs, out, scorer, max_cands=4, max_subset=2)
    assert scorer.calls == first_calls and again["n_labeled"] == res["n_labeled"]  # 이어 돌리면 다시 계산하지 않는다
    with pytest.raises(ValueError, match="다르"):
        run_labeling(items, qs, out, FakeScorer(fn=lambda k: 0.0, name="other"), max_cands=4, max_subset=2)
    lines = (out / "dq_log.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == res["n_labeled"] and json.loads(lines[0])["scores"][""] == 0.0


def test_run_labeling_refuses_test_split_and_missing_references(tmp_path, items, questions):
    scorer = FakeScorer(fn=lambda key: 0.0)
    with pytest.raises(ValueError, match="test"):
        run_labeling(items, questions, tmp_path / "x", scorer, splits=("train", "test"))
    no_ref = [EvalQuestion("n1", "Which planet is largest?", ["ast_001"], "astronomy", "easy", "train", rubric="x")]
    with pytest.raises(ValueError, match="라벨링할 질문이 없다"):
        run_labeling(items, no_ref, tmp_path / "y", scorer)
    with pytest.raises(ValueError):
        run_labeling(items, questions, tmp_path / "z", scorer, max_cands=1)


def test_run_labeling_limit_and_fresh(tmp_path, items, questions):
    scorer = FakeScorer(fn=lambda key: 0.0)
    out = tmp_path / "lim"
    res = run_labeling(items, questions, out, scorer, max_cands=3, max_subset=1, limit=1)
    assert res["n_labeled"] <= 1
    res2 = run_labeling(items, questions, out, scorer, max_cands=3, max_subset=1, limit=1, resume=False)
    assert res2["n_labeled"] <= 1
