"""한국어·영어 혼용 문서를 CPU만으로 다루는 가벼운 토크나이저.

한국어는 조사가 붙은 어절('목성은', '목성이')이 서로 다른 토큰이 되는 문제가 있다.
세 글자 이상의 한글 어절에는 글자 2-gram을 함께 넣어 어간이 겹치게 한다.
형태소 분석기가 아니라서 정확도에는 한계가 있고, 개선은 실험으로 확인해야 한다.
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any

MODES = ("ngram", "word")

_TOKEN_RE = re.compile(r"[가-힣]+|[A-Za-z0-9]+|[^\W_]+")
_HANGUL_RE = re.compile(r"^[가-힣]+$")
_LATIN_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_HANGUL_CHAR_RE = re.compile(r"[가-힣]")
_PUNCT_RE = re.compile(r"[^\sA-Za-z0-9가-힣]")


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower()


def tokenize(text: str, mode: str = "ngram") -> list[str]:
    if mode not in MODES:
        raise ValueError(f"tokenizer mode는 {MODES} 중 하나여야 한다: {mode!r}")
    tokens: list[str] = []
    for word in _TOKEN_RE.findall(normalize(text)):
        tokens.append(word)
        if mode == "ngram" and len(word) >= 3 and _HANGUL_RE.match(word):
            tokens.extend(word[i : i + 2] for i in range(len(word) - 1))
    return tokens


def estimate_tokens(text: str) -> int:
    """LLM 토큰 수의 거친 추정치. 실제 토크나이저 값이 아니므로 모델을 정하면 교체한다."""
    if not text.strip():
        return 0
    words = len(_LATIN_WORD_RE.findall(text))
    hangul = len(_HANGUL_CHAR_RE.findall(text))
    punct = len(_PUNCT_RE.findall(text))
    return max(1, math.ceil(words * 1.3 + hangul * 0.7 + punct * 0.5))


def estimate_item_tokens(item: Any) -> int:
    """제목과 본문을 합친 추정 토큰 수. 선택기 예산과 평가 지표가 같은 값을 쓴다."""
    return estimate_tokens(f"{item.title} {item.text}")
