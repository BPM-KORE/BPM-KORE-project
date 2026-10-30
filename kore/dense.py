"""의미 검색(임베딩)과 BM25를 섞는 하이브리드 검색기.

BM25는 질문과 문서가 같은 낱말을 쓸 때 강하고, 표현이 바뀌면 약하다. C가 질문을 쓸 때 근거 문장을
베끼지 않도록 했으므로 우리 데이터에서는 표현이 바뀐 질문이 많을 것이다. 임베딩 검색은 이 약점을
메우는 표준 방법이다. 다만 **우리 데이터에서 실제로 나아지는지는 아직 모른다.** `kore retrieval-eval`로
train·dev에서 BM25와 비교해서 확인한 뒤에 켠다.

- 모델 기본값: `intfloat/multilingual-e5-small` (약 0.1B, MIT, 384차원, 최대 512토큰, "query: "/"passage: " 접두사,
  평균 풀링 후 정규화 — 모델 카드에서 확인함). 문서 200개는 무료 Colab CPU에서도 부담이 작다.
- torch와 transformers는 `HFEncoder`를 만들 때만 불러온다. 핵심 코드는 계속 표준 라이브러리만 쓴다.
- 이 환경에는 torch가 없어서 `HFEncoder`는 실행해 보지 못했다. 섞는 계산은 가짜 인코더로 시험했다.
"""

from __future__ import annotations

import math
from typing import Protocol, Sequence

from kore.retrieval import BM25Retriever
from kore.schemas import KnowledgeItem

DEFAULT_DENSE_MODEL = "intfloat/multilingual-e5-small"


class Encoder(Protocol):
    name: str

    def encode(self, texts: Sequence[str], kind: str) -> list[list[float]]:
        """kind는 'query' 또는 'passage'. 길이 1로 정규화한 벡터를 돌려준다."""
        ...


def _normalize(vec: Sequence[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in vec))
    return [x / n for x in vec] if n > 0 else [0.0 for _ in vec]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class HFEncoder:
    """Hugging Face 문장 임베딩. e5 계열은 접두사를 붙이고 평균 풀링, 그 밖의 모델은 CLS 풀링을 쓴다."""

    def __init__(self, model: str = DEFAULT_DENSE_MODEL, device: str | None = None, batch_size: int = 32,
                 max_length: int = 512) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - torch가 없는 환경
            raise ImportError("하이브리드 검색에는 torch와 transformers가 필요하다: pip install -e \".[hf]\"") from exc
        self._torch = torch
        self.model_name = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).to(self.device).eval()
        self.batch_size = batch_size
        self.max_length = max_length
        self.is_e5 = "e5" in model.lower()
        self.name = f"dense:{model}"

    def encode(self, texts: Sequence[str], kind: str) -> list[list[float]]:  # pragma: no cover - GPU/torch 경로
        if kind not in ("query", "passage"):
            raise ValueError("kind는 query 또는 passage")
        torch = self._torch
        prefix = f"{kind}: " if self.is_e5 else ""
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [prefix + t for t in texts[start : start + self.batch_size]]
            enc = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                hidden = self.model(**enc).last_hidden_state
            if self.is_e5:
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            else:
                pooled = hidden[:, 0]
            pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)
            out.extend(pooled.cpu().tolist())
        return out


class HybridRetriever:
    """BM25 점수와 임베딩 코사인 유사도를 0~1로 맞춘 뒤 섞는다.

    점수 = alpha × (BM25 / 후보 풀의 최대 BM25) + (1 − alpha) × ((cos − 풀 최소) / (풀 최대 − 풀 최소)).
    후보 풀은 BM25 상위 pool개와 임베딩 상위 pool개의 합집합이다. alpha=1이면 BM25 순위와 같다.
    선택기의 `term_weights`와 분야 라우팅은 BM25 것을 그대로 쓴다.
    """

    def __init__(self, items: list[KnowledgeItem], bm25: BM25Retriever, encoder: Encoder,
                 alpha: float = 0.5, pool: int = 50) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha는 0~1")
        if pool < 1:
            raise ValueError("pool은 1 이상")
        self.bm25 = bm25
        self.encoder = encoder
        self.alpha = alpha
        self.pool = pool
        self._items = bm25.items
        vecs = encoder.encode([f"{i.title}. {i.text}" for i in self._items], "passage")
        if len(vecs) != len(self._items):
            raise ValueError("인코더가 돌려준 벡터 수가 문서 수와 다르다")
        self._vecs = [_normalize(v) for v in vecs]
        self.tokenizer_mode = bm25.tokenizer_mode

    @property
    def items(self) -> list[KnowledgeItem]:
        return list(self._items)

    def get(self, item_id: str) -> KnowledgeItem:
        return self.bm25.get(item_id)

    def term_weights(self, question: str) -> dict[str, float]:
        return self.bm25.term_weights(question)

    def domain_scores(self, question: str, top_n: int = 3) -> dict[str, float]:
        return self.bm25.domain_scores(question, top_n)

    def fused_scores(self, question: str, domains: list[str] | None = None) -> dict[int, float]:
        allowed = set(domains) if domains else None
        lex = self.bm25._scores(question, allowed)
        qv = _normalize(self.encoder.encode([question], "query")[0])
        sem = {
            idx: _dot(qv, v)
            for idx, v in enumerate(self._vecs)
            if allowed is None or self._items[idx].domain in allowed
        }
        lex_top = sorted(lex, key=lambda i: (-lex[i], i))[: self.pool]
        sem_top = sorted(sem, key=lambda i: (-sem[i], i))[: self.pool]
        pool = set(lex_top) | set(sem_top)
        if not pool:
            return {}
        lmax = max((lex.get(i, 0.0) for i in pool), default=0.0)
        smin = min(sem[i] for i in pool)
        smax = max(sem[i] for i in pool)
        out = {}
        for i in pool:
            l_norm = lex.get(i, 0.0) / lmax if lmax > 0 else 0.0
            s_norm = (sem[i] - smin) / (smax - smin) if smax > smin else 0.0
            out[i] = self.alpha * l_norm + (1.0 - self.alpha) * s_norm
        return out

    def retrieve(self, question: str, k: int = 10, domains: list[str] | None = None) -> list[KnowledgeItem]:
        if k < 1:
            raise ValueError("k는 1 이상이어야 한다")
        scores = self.fused_scores(question, domains)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [
            self._items[idx].copy_with(score=s, retrieval_score=s, selection_score=0.0)
            for idx, s in ranked
            if s > 0.0
        ]
