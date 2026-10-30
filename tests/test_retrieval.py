import pytest

from kore.retrieval import BM25Retriever
from tests.helpers import mk_item


def test_bm25_matches_hand_computed_values():
    # 손계산(구현과 독립): idf(apple)=ln(1+(3-2+.5)/(2+.5))=0.470004, avgdl=7/3
    items = [mk_item("d1", "apple banana"), mk_item("d2", "banana cherry"), mk_item("d3", "apple apple durian")]
    r = BM25Retriever(items, tokenizer_mode="word")
    out = r.retrieve("apple", k=10)
    assert [i.id for i in out] == ["d3", "d1"]  # d2는 점수 0이라 빠진다
    assert out[0].retrieval_score == pytest.approx(0.614958, abs=1e-5)
    assert out[1].retrieval_score == pytest.approx(0.502294, abs=1e-5)
    assert out[0].score == out[0].retrieval_score


def test_ranking_on_sample(items):
    r = BM25Retriever(items)
    assert r.retrieve("Which planet is the largest in the Solar System?")[0].id == "ast_001"
    assert r.retrieve("What is the pH of pure water at 25 C?")[0].id == "che_001"
    assert r.retrieve("What mechanism does TCP use to establish a connection?")[0].id == "com_005"


def test_korean_query_with_particles(items):
    r = BM25Retriever(items)
    assert r.retrieve("정렬된 배열에서 이진 탐색의 시간 복잡도는 얼마인가?")[0].id == "com_002"
    assert r.retrieve("스택과 큐의 차이는 무엇인가?")[0].id == "com_003"


def test_k_limits_results_and_ties_keep_input_order(items):
    assert len(BM25Retriever(items).retrieve("planet", k=2)) == 2
    tie = [mk_item("x1", "same words"), mk_item("x2", "same words")]
    assert [i.id for i in BM25Retriever(tie, tokenizer_mode="word").retrieve("same")] == ["x1", "x2"]


def test_domain_filter(items):
    r = BM25Retriever(items)
    out = r.retrieve("acid water", k=10, domains=["chemistry"])
    assert out and {i.domain for i in out} == {"chemistry"}


def test_no_match_and_empty_question(items):
    r = BM25Retriever(items)
    assert r.retrieve("zzzzqqqq") == []
    assert r.retrieve("") == []
    assert r.retrieve("   ?!  ") == []


def test_invalid_k_and_duplicate_ids(items):
    with pytest.raises(ValueError):
        BM25Retriever(items).retrieve("x", k=0)
    with pytest.raises(ValueError):
        BM25Retriever([mk_item("a", "x"), mk_item("a", "y")])


def test_empty_corpus_returns_nothing():
    assert BM25Retriever([]).retrieve("anything") == []


def test_retrieve_does_not_mutate_corpus_items(items):
    r = BM25Retriever(items)
    r.retrieve("planet")
    assert all(i.retrieval_score == 0.0 and i.score == 0.0 for i in r.items)
    assert all(i.retrieval_score == 0.0 for i in items)


def test_retrieve_is_deterministic(items):
    r = BM25Retriever(items)
    a = [(i.id, i.retrieval_score) for i in r.retrieve("gas giant planet")]
    b = [(i.id, i.retrieval_score) for i in r.retrieve("gas giant planet")]
    assert a == b


def test_domain_scores_routes_to_right_domain(items):
    scores = BM25Retriever(items).domain_scores("neutron star supernova density")
    assert next(iter(scores)) == "astronomy"
    assert max(scores.values()) == pytest.approx(1.0)
    assert BM25Retriever(items).domain_scores("zzzzqqqq") == {}


def test_term_weights_drop_unseen_terms_and_rank_rare_higher(items):
    w = BM25Retriever(items).term_weights("saturn zzzzqqqq the")
    assert "zzzzqqqq" not in w
    assert w["saturn"] > w["the"]
