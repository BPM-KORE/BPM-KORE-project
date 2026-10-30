"""한국어·영어 혼용 문서를 CPU만으로 다루는 가벼운 토크나이저.

한국어는 조사가 붙은 어절('목성은', '목성이')이 서로 다른 토큰이 되는 문제가 있다.
모드:
- `word`: 어절만.
- `ngram`: 어절 + 세 글자 이상 한글 어절의 글자 2-gram.
- `ngram_stem`(기본): `ngram`에 더해 흔한 조사·어미를 뗀 어간을 한 번 더 넣는다('목성은' → '목성').
  KorQuAD 1.0 dev(문단 964개, 질문 5,774개)에서 `ngram`보다 Recall@1이 0.879 → 0.886,
  표현이 많이 바뀐 질문 2,754개에서 0.809 → 0.824였다(McNemar z = 3.2, 3.9). 우리 데이터에서의 효과는 아직 모른다.
형태소 분석기가 아니라서 정확도에는 한계가 있다.
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any

MODES = ("ngram_stem", "ngram", "word")
DEFAULT_MODE = "ngram_stem"

# 길이가 긴 것부터 시도한다. 떼고 남은 어간이 두 글자 이상일 때만 뗀다.
_SUFFIXES = sorted(
    (
        "은 는 이 가 을 를 의 에 에서 에게 께 한테 으로 로 와 과 도 만 까지 부터 보다 처럼 이나 나 이며 며 이고 고 "
        "에는 에서는 으로는 로는 과의 와의 에의 이라 라 이란 란 인 이다 였다 이었다 했다 하는 하였다 한 된 되는 되었다 "
        "들 들은 들이 들을"
    ).split(),
    key=len,
    reverse=True,
)


def strip_suffix(word: str) -> str | None:
    """한글 어절 끝의 흔한 조사·어미를 하나 떼어 낸 어간. 뗄 것이 없으면 None."""
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 2:
            return word[: -len(suf)]
    return None

_TOKEN_RE = re.compile(r"[가-힣]+|[A-Za-z0-9]+|[^\W_]+")
_HANGUL_RE = re.compile(r"^[가-힣]+$")
_LATIN_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_HANGUL_CHAR_RE = re.compile(r"[가-힣]")
_PUNCT_RE = re.compile(r"[^\sA-Za-z0-9가-힣]")


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower()


def tokenize(text: str, mode: str = DEFAULT_MODE) -> list[str]:
    if mode not in MODES:
        raise ValueError(f"tokenizer mode는 {MODES} 중 하나여야 한다: {mode!r}")
    tokens: list[str] = []
    for word in _TOKEN_RE.findall(normalize(text)):
        tokens.append(word)
        if mode == "word" or not _HANGUL_RE.match(word):
            continue
        if mode == "ngram_stem":
            stem = strip_suffix(word)
            if stem:
                tokens.append(stem)
        if len(word) >= 3:
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
