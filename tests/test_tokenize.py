import pytest

from kore.tokenize import estimate_tokens, tokenize


def test_latin_is_lowercased_and_split():
    assert tokenize("Hello, World 42") == ["hello", "world", "42"]


def test_hangul_word_gets_char_bigrams_in_ngram_mode():
    assert tokenize("목성은 크다") == ["목성은", "목성", "성은", "크다"]


def test_word_mode_has_no_bigrams():
    assert tokenize("목성은 크다", "word") == ["목성은", "크다"]


def test_particles_share_a_stem_token():
    assert "목성" in set(tokenize("목성은")) & set(tokenize("목성이"))


def test_mixed_script_is_split_at_script_boundary():
    assert tokenize("abc가나다") == ["abc", "가나다", "가나", "나다"]


def test_fullwidth_is_normalized():
    assert tokenize("ＡＢＣ") == ["abc"]


def test_empty_text():
    assert tokenize("") == []
    assert tokenize("  !?  ") == []


def test_invalid_mode():
    with pytest.raises(ValueError):
        tokenize("x", "bogus")


def test_tokenize_is_deterministic():
    text = "이진 탐색은 O(log n) 이다. binary search"
    assert tokenize(text) == tokenize(text)


def test_estimate_tokens_basics():
    assert estimate_tokens("") == 0
    assert estimate_tokens("   ") == 0
    assert estimate_tokens("hello world") == 3  # ceil(2 * 1.3)
    assert estimate_tokens("hello world foo") > estimate_tokens("hello world")
    assert estimate_tokens("안녕하세요") >= 1
