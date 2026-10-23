import pytest

from kore.evaluation import (
    approx_mde, bootstrap_ci, evaluate, hit_at_k, mrr, paired_bootstrap_diff, precision_at_k,
    recall_at_k, rubric_score,
)
from tests.helpers import mk_item


def test_retrieval_metrics():
    sel, exp = ["a", "b", "c"], ["b", "z"]
    assert recall_at_k(sel, exp) == 0.5
    assert precision_at_k(sel, exp) == pytest.approx(1 / 3)
    assert hit_at_k(sel, exp) == 1.0
    assert hit_at_k(["a"], exp) == 0.0
    assert mrr(sel, exp) == 0.5
    assert mrr(["x"], exp) == 0.0


def test_metric_edge_cases():
    assert recall_at_k(["a"], []) is None
    assert precision_at_k([], ["a"]) == 0.0
    assert recall_at_k([], ["a"]) == 0.0
    assert recall_at_k(["a", "a"], ["a"]) == 1.0  # 중복 선택은 한 번으로 센다


def test_rubric_score():
    items = [["jupiter"], ["mars", "red planet"]]
    assert rubric_score("Jupiter and the Red Planet", items) == 1.0
    assert rubric_score("only jupiter", items) == 0.5
    assert rubric_score("nothing", items) == 0.0
    assert rubric_score("anything", []) is None


def test_rubric_digit_boundary():
    assert rubric_score("The pH is 17.", [["7"]]) == 0.0
    assert rubric_score("The pH is 7.", [["7"]]) == 1.0
    assert rubric_score("pH of 7.0", [["7"]]) == 1.0


def test_rubric_is_case_insensitive_and_handles_symbols():
    assert rubric_score("it is O(LOG N)", [["o(log n)"]]) == 1.0


def test_evaluate_collects_metrics():
    sel = [mk_item("a", "one two"), mk_item("b", "three")]
    ref = {"expected_source_ids": ["a", "z"], "rubric_items": [["one"]]}
    res = evaluate("q", sel, ref)
    assert res["recall"] == 0.5 and res["hit"] == 1.0 and res["n_selected"] == 2
    assert res["context_tokens"] > 0 and res["rubric_score"] is None  # 답변이 없으면 rubric은 None
    assert evaluate("q", sel, ref, answer="one")["rubric_score"] == 1.0
    assert evaluate("q", [], ref)["context_tokens"] == 0


def test_bootstrap_ci_properties():
    mean, lo, hi = bootstrap_ci([0.0, 1.0, 1.0, 0.0, 1.0, 1.0], seed=3)
    assert mean == pytest.approx(4 / 6) and lo <= mean <= hi
    assert bootstrap_ci([0.0, 1.0, 1.0, 0.0, 1.0, 1.0], seed=3) == (mean, lo, hi)
    assert bootstrap_ci([0.4, 0.4, 0.4]) == (pytest.approx(0.4), pytest.approx(0.4), pytest.approx(0.4))
    assert bootstrap_ci([None, 0.5, None])[0] == 0.5
    assert bootstrap_ci([]) == (None, None, None)
    assert bootstrap_ci([None]) == (None, None, None)


def test_bootstrap_ci_narrows_with_more_data():
    small = bootstrap_ci([0, 1] * 5, seed=1)
    large = bootstrap_ci([0, 1] * 100, seed=1)
    assert (large[2] - large[1]) < (small[2] - small[1])


def test_paired_bootstrap_diff():
    a = [1.0, 1.0, 0.5, None, 1.0]
    b = [0.5, 0.5, 0.0, 1.0, None]
    d = paired_bootstrap_diff(a, b, seed=0)
    assert d["n"] == 3 and d["mean_diff"] == pytest.approx(0.5) and d["ci_lo"] > 0
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1.0], [1.0, 2.0])
    assert paired_bootstrap_diff([None], [1.0])["mean_diff"] is None


def test_approx_mde_matches_hand_calculation():
    # (1.96 + 0.8416) * sqrt(0.3) / sqrt(30) = 0.28
    assert approx_mde(30, 0.3 ** 0.5) == pytest.approx(0.28, abs=0.005)
    assert approx_mde(120, 0.3 ** 0.5) < approx_mde(30, 0.3 ** 0.5)
    with pytest.raises(ValueError):
        approx_mde(1, 0.5)
