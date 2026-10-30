import pytest

from kore.selector import SelectorWeights, select_knowledge
from kore.tokenize import estimate_item_tokens
from tests.helpers import mk_item


def ids(items):
    return [i.id for i in items]


def test_similarity_takes_top_k_by_score_and_ties_keep_order():
    cands = [mk_item("a", "x", 1.0), mk_item("b", "x", 3.0), mk_item("c", "x", 3.0), mk_item("d", "x", 2.0)]
    assert ids(select_knowledge("q", cands, k=3, strategy="similarity")) == ["b", "c", "d"]


def test_kore_with_only_sim_weight_equals_similarity():
    cands = [mk_item("a", "x y", 1.0), mk_item("b", "z", 3.0, domain="e"), mk_item("c", "x y", 2.0)]
    w = SelectorWeights(sim=1.0, coverage=0.0, diversity=0.0, redundancy=0.0, reliability=0.0)
    assert ids(select_knowledge("x", cands, k=3, strategy="kore", weights=w)) == ids(
        select_knowledge("x", cands, k=3, strategy="similarity")
    )


def test_redundancy_penalty_prefers_a_different_document():
    a = mk_item("A", "alpha beta gamma delta epsilon", 10.0)
    a2 = mk_item("A2", "alpha beta gamma delta epsilon zeta", 9.5)  # A의 거의 복사본
    b = mk_item("B", "alpha omega sigma tau", 7.0)
    cands = [a, a2, b]
    assert ids(select_knowledge("alpha", cands, k=2, strategy="similarity", tokenizer_mode="word")) == ["A", "A2"]
    strong = SelectorWeights(redundancy=0.5)
    assert ids(select_knowledge("alpha", cands, k=2, strategy="kore", tokenizer_mode="word", weights=strong)) == ["A", "B"]
    # 기본값(0.1)은 약하다. KorQuAD에서 0.5가 근거 하나짜리 질문의 정답 문단을 밀어냈기 때문이다(docs/REPORT 3차).
    # 거의 같은 문서는 kore stats의 중복 점검으로 데이터에서 먼저 없앤다.
    assert SelectorWeights().redundancy == 0.1


def test_diversity_bonus_must_not_promote_an_irrelevant_document():
    # 회귀: 다양성 보너스가 관련도와 무관하게 붙던 때는 무관한 다른 분야 문서 I가 S를 이겼다.
    f = mk_item("F", "alpha b c", 10.0, domain="x")
    s = mk_item("S", "alpha b d", 3.5, domain="x")  # 같은 분야, 관련도 보통
    i = mk_item("I", "omega", 0.5, domain="y")  # 다른 분야, 거의 무관
    assert ids(select_knowledge("alpha", [f, s, i], k=2, strategy="kore", tokenizer_mode="word")) == ["F", "S"]


def test_diversity_still_breaks_ties_among_relevant_candidates():
    f = mk_item("F", "alpha one", 10.0, domain="x")
    s = mk_item("S", "alpha two", 8.0, domain="x")
    o = mk_item("O", "alpha three", 8.0, domain="y")  # 같은 관련도지만 새 분야
    assert ids(select_knowledge("alpha", [f, s, o], k=2, strategy="kore", tokenizer_mode="word")) == ["F", "O"]


def test_idf_weighted_coverage_ignores_stopword_matches():
    p = mk_item("P", "zzz", 10.0)
    x = mk_item("X", "the common", 5.0)  # 불용어 'the'만 덮는다
    y = mk_item("Y", "alpha rare", 5.0)  # 희귀어 'alpha'를 덮는다
    cands = [p, x, y]
    # 가중치가 없으면 두 토큰이 같은 값이라 동점이고 입력 순서가 앞선 X가 이긴다(결함의 재현).
    assert ids(select_knowledge("the alpha", cands, k=2, strategy="kore", tokenizer_mode="word")) == ["P", "X"]
    tw = {"the": 0.1, "alpha": 2.0}
    assert ids(
        select_knowledge("the alpha", cands, k=2, strategy="kore", tokenizer_mode="word", term_weights=tw)
    ) == ["P", "Y"]


def test_reliability_weight_breaks_equal_similarity():
    low = mk_item("low", "x", 5.0, reliability=0.1)
    high = mk_item("high", "x", 5.0, reliability=0.9)
    w = SelectorWeights(sim=1.0, coverage=0.0, diversity=0.0, redundancy=0.0, reliability=1.0)
    assert ids(select_knowledge("x", [low, high], k=1, strategy="kore", weights=w)) == ["high"]


@pytest.mark.parametrize("strategy", ["similarity", "kore"])
def test_token_budget_is_respected(strategy):
    big = mk_item("big", " ".join(["w"] * 100), 10.0)
    small = mk_item("small", " ".join(["w"] * 10), 5.0)
    out = select_knowledge("w", [big, small], k=2, strategy=strategy, token_budget=50)
    assert ids(out) == ["small"]
    assert sum(estimate_item_tokens(i) for i in out) <= 50
    assert select_knowledge("w", [big, small], k=2, strategy=strategy, token_budget=0) == []


@pytest.mark.parametrize("strategy", ["similarity", "kore"])
def test_k_larger_than_pool_and_empty_pool(strategy):
    cands = [mk_item("a", "x", 1.0), mk_item("b", "y", 2.0)]
    assert len(select_knowledge("x", cands, k=10, strategy=strategy)) == 2
    assert select_knowledge("x", [], k=3, strategy=strategy) == []


@pytest.mark.parametrize("strategy", ["similarity", "kore"])
def test_inputs_are_not_mutated(strategy):
    cands = [mk_item("a", "x", 1.0), mk_item("b", "y", 2.0)]
    select_knowledge("x", cands, k=2, strategy=strategy)
    assert [c.selection_score for c in cands] == [0.0, 0.0]
    assert [c.score for c in cands] == [1.0, 2.0]


def test_selection_score_is_set_and_deterministic():
    cands = [mk_item("a", "alpha", 4.0), mk_item("b", "beta", 3.0)]
    first = select_knowledge("alpha", cands, k=2, strategy="kore")
    again = select_knowledge("alpha", cands, k=2, strategy="kore")
    assert [(i.id, i.selection_score) for i in first] == [(i.id, i.selection_score) for i in again]
    assert all(i.score == i.selection_score for i in first)


def test_falls_back_to_contract_score_field():
    a = mk_item("a", "x", 0.0)
    b = mk_item("b", "x", 0.0)
    a.score, b.score = 1.0, 2.0  # retrieval_score 없이 계약의 score만 채워진 입력
    assert ids(select_knowledge("x", [a, b], k=1, strategy="similarity")) == ["b"]


def test_invalid_arguments():
    c = [mk_item("a", "x")]
    with pytest.raises(ValueError):
        select_knowledge("x", c, k=0)
    with pytest.raises(ValueError):
        select_knowledge("x", c, strategy="bogus")
    with pytest.raises(ValueError):
        select_knowledge("x", c, token_budget=-1)


def test_ablate_zeroes_exactly_one_feature():
    w = SelectorWeights()
    assert w.ablate("diversity").diversity == 0.0
    assert w.ablate("diversity").coverage == w.coverage
    with pytest.raises(ValueError):
        w.ablate("sim")
