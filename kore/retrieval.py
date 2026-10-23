"""BM25 검색기. 외부 라이브러리 없이 CPU만으로 돈다."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from kore.schemas import KnowledgeItem
from kore.tokenize import tokenize


class BM25Retriever:
    """Okapi BM25. idf는 Lucene 방식(항상 양수): ln(1 + (N - df + 0.5) / (df + 0.5))."""

    def __init__(
        self,
        items: list[KnowledgeItem],
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer_mode: str = "ngram",
    ) -> None:
        ids = [i.id for i in items]
        if len(set(ids)) != len(ids):
            dup = sorted({x for x in ids if ids.count(x) > 1})
            raise ValueError(f"id가 중복이다: {dup[:5]}")
        self.k1 = k1
        self.b = b
        self.tokenizer_mode = tokenizer_mode
        self._items = list(items)
        self._by_id = {i.id: i for i in self._items}
        self._lengths: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for idx, item in enumerate(self._items):
            toks = tokenize(f"{item.title} {item.text}", tokenizer_mode)
            self._lengths.append(len(toks))
            for term, tf in Counter(toks).items():
                self._postings[term].append((idx, tf))
        n = len(self._items)
        self._avgdl = sum(self._lengths) / n if n else 0.0
        self._idf = {
            term: math.log(1.0 + (n - len(post) + 0.5) / (len(post) + 0.5))
            for term, post in self._postings.items()
        }

    @property
    def items(self) -> list[KnowledgeItem]:
        return list(self._items)

    def get(self, item_id: str) -> KnowledgeItem:
        return self._by_id[item_id]

    def _scores(self, question: str, domains: set[str] | None) -> dict[int, float]:
        scores: dict[int, float] = defaultdict(float)
        for term in dict.fromkeys(tokenize(question, self.tokenizer_mode)):
            posting = self._postings.get(term)
            if not posting:
                continue
            idf = self._idf[term]
            for idx, tf in posting:
                if domains is not None and self._items[idx].domain not in domains:
                    continue
                norm = self.k1 * (1.0 - self.b + self.b * self._lengths[idx] / self._avgdl)
                scores[idx] += idf * tf * (self.k1 + 1.0) / (tf + norm)
        return scores

    def retrieve(
        self, question: str, k: int = 10, domains: list[str] | None = None
    ) -> list[KnowledgeItem]:
        """점수가 0보다 큰 문서만 최대 k개 돌려준다. 동점이면 입력 순서가 앞선 문서가 먼저다."""
        if k < 1:
            raise ValueError("k는 1 이상이어야 한다")
        scores = self._scores(question, set(domains) if domains else None)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [
            self._items[idx].copy_with(score=s, retrieval_score=s, selection_score=0.0)
            for idx, s in ranked
            if s > 0.0
        ]

    def term_weights(self, question: str) -> dict[str, float]:
        """질문 토큰별 코퍼스 IDF. 코퍼스에 없는 토큰은 어떤 문서도 덮을 수 없으니 뺀다."""
        return {t: self._idf[t] for t in dict.fromkeys(tokenize(question, self.tokenizer_mode)) if t in self._idf}

    def domain_scores(self, question: str, top_n: int = 3) -> dict[str, float]:
        """분야 라우팅용. 분야별 상위 top_n개 문서 점수의 합을 최고값 1.0으로 맞춘 값."""
        per_domain: dict[str, list[float]] = defaultdict(list)
        for idx, s in self._scores(question, None).items():
            per_domain[self._items[idx].domain].append(s)
        totals = {d: sum(sorted(v, reverse=True)[:top_n]) for d, v in per_domain.items()}
        top = max(totals.values(), default=0.0)
        if top <= 0.0:
            return {}
        return dict(sorted(((d, t / top) for d, t in totals.items()), key=lambda kv: (-kv[1], kv[0])))
