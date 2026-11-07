from __future__ import annotations

import random

from kore import selector
from kore.selector import SelectorWeights, select_knowledge
from tests.helpers import mk_item

TEXTS = [
    "Jupiter is the largest planet with a Great Red Spot storm",
    "대한민국 헌법은 국민의 기본권을 보장한다",
    "행정기관은 민원 처리 기한을 지켜야 한다",
    "Mars looks red because of iron oxide on its surface",
    "소득세는 개인의 소득에 부과되는 세금이다",
    "Neutron stars are extremely dense remnants of supernovae",
]
QUESTIONS = ["Jupiter storm planet", "헌법 기본권", "민원 처리 기한", "red Mars", "소득세 신고"]


def _pool():
    return [
        mk_item(f"d{i}", t, score=5.0 - 0.4 * i, domain=("a", "b", "c")[i % 3],
                reliability=0.3 + 0.1 * i, title=f"T{i}")
        for i, t in enumerate(TEXTS)
    ]


def _ids_scores(picked):
    return [(p.id, p.score, p.selection_score) for p in picked]


def test_cached_selection_equals_cold_selection_on_many_inputs():
    rng = random.Random(1)
    pool = _pool()
    for n in range(60):
        cands = rng.sample(pool, rng.randint(1, len(pool)))
        kw = dict(
            k=rng.choice([1, 2, 3, 5]),
            token_budget=rng.choice([None, 10, 40, 200]),
            tokenizer_mode=rng.choice(["ngram_stem", "word", "ngram"]),
            weights=rng.choice([None, SelectorWeights().ablate("redundancy"), SelectorWeights(sim=0.3, redundancy=0.9)]),
            term_weights=rng.choice([None, {"jupiter": 2.0, "헌법": 3.0}]),
        )
        q = QUESTIONS[n % len(QUESTIONS)]
        selector._doc_tokens.cache_clear()
        cold = _ids_scores(select_knowledge(q, cands, **kw))
        warm1 = _ids_scores(select_knowledge(q, cands, **kw))  # 캐시가 찬 뒤
        warm2 = _ids_scores(select_knowledge(q, [c.copy_with() for c in cands], **kw))  # 복사본(다른 객체, 같은 내용)
        assert cold == warm1 == warm2


def test_changed_document_content_is_not_served_from_stale_cache():
    a = mk_item("a", "Jupiter storm planet", score=1.0, domain="x")
    b = mk_item("b", "unrelated cooking recipe", score=1.0, domain="y")
    first = select_knowledge("Jupiter storm", [a, b], k=1)
    assert first[0].id == "a"
    # 같은 id, 내용만 바뀐 문서: 새 내용 기준으로 점수가 달라져야 한다.
    a2 = a.copy_with(text="unrelated cooking recipe")
    b2 = b.copy_with(text="Jupiter storm planet")
    second = select_knowledge("Jupiter storm", [a2, b2], k=1)
    assert second[0].id == "b"


def test_tokenizer_mode_is_part_of_cache_key():
    cands = [mk_item("a", "대한민국은 민주공화국이다", title="헌법")]
    selector._doc_tokens.cache_clear()
    select_knowledge("민주공화국", cands, k=1, tokenizer_mode="word")
    select_knowledge("민주공화국", cands, k=1, tokenizer_mode="ngram_stem")
    assert selector._doc_tokens.cache_info().currsize == 2


def test_documents_are_tokenized_once_across_calls(monkeypatch):
    calls = []
    real = selector.tokenize

    def counting(text, mode="ngram_stem"):
        calls.append(text)
        return real(text, mode)

    monkeypatch.setattr(selector, "tokenize", counting)
    selector._doc_tokens.cache_clear()
    pool = _pool()
    select_knowledge("Jupiter", pool, k=3)
    first = len(calls)  # 질문 1회 + 문서 6개
    select_knowledge("Jupiter", pool, k=3)
    assert len(calls) - first == 1  # 두 번째는 질문만 다시 토큰화한다


def test_cache_is_bounded():
    assert selector._doc_tokens.cache_info().maxsize is not None
    selector._doc_tokens.cache_clear()
    cap = selector._doc_tokens.cache_info().maxsize
    for i in range(cap + 50):
        selector._doc_tokens(f"doc number {i}", "word")
    assert selector._doc_tokens.cache_info().currsize <= cap
