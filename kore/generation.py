"""답변 생성기 인터페이스.

생성 모델은 아직 정해지지 않았다(미확정 사항). 파이프라인은 Generator 프로토콜만 요구한다.
- ExtractiveGenerator: 근거에서 문장을 뽑는 결정적 기준선. 배관 점검과 데모용이고 언어모델이 아니다.
- HFGenerator: Hugging Face 모델 래퍼. 이 저장소의 테스트 환경에서는 실행해 보지 못했다(GPU·모델 필요).
"""

from __future__ import annotations

import re
from typing import Protocol

from kore.schemas import KnowledgeItem
from kore.tokenize import tokenize

_SENT_SPLIT = re.compile(r"(?<=[.!?。])\s+|\n+")


class Generator(Protocol):
    name: str

    def generate(self, question: str, evidence: list[KnowledgeItem]) -> str: ...


def build_prompt(question: str, evidence: list[KnowledgeItem]) -> str:
    """세 조건(no_retrieval, vanilla_rag, kore)이 같은 틀을 쓰도록 하나의 함수로 만든다."""
    if not evidence:
        return f"다음 질문에 짧게 답하라.\n\n질문: {question}\n답:"
    blocks = "\n\n".join(
        f"[근거 {n}] (id={e.id}, 출처={e.source_url})\n{e.text}" for n, e in enumerate(evidence, start=1)
    )
    return (
        "아래 근거만 사용해 질문에 짧게 답하라. 근거에 답이 없으면 모른다고 답하라.\n\n"
        f"{blocks}\n\n질문: {question}\n답:"
    )


class ExtractiveGenerator:
    name = "extractive-v0"

    def generate(self, question: str, evidence: list[KnowledgeItem]) -> str:
        if not evidence:
            return ""
        q_terms = set(tokenize(question, "word"))
        best = ("", -1.0)
        for item in evidence:
            for sent in _SENT_SPLIT.split(item.text):
                sent = sent.strip()
                if not sent:
                    continue
                overlap = len(q_terms & set(tokenize(sent, "word")))
                if overlap > best[1]:
                    best = (sent, float(overlap))
        return best[0]


class HFGenerator:
    """greedy 디코딩으로 고정한 Hugging Face 생성기. 같은 입력이면 같은 출력이 나오게 한다."""

    def __init__(self, model_name: str, max_new_tokens: int = 128) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - 선택 의존성
            raise ImportError("pip install -e .[hf] 로 transformers와 torch를 설치해야 한다") from exc
        self.name = f"hf:{model_name}"
        self.max_new_tokens = max_new_tokens
        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(model_name)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._device == "cuda":
            self._model = self._model.half()
        self._model.to(self._device)
        self._model.eval()

    def generate(self, question: str, evidence: list[KnowledgeItem]) -> str:  # pragma: no cover
        prompt = build_prompt(question, evidence)
        if getattr(self._tok, "chat_template", None):
            inputs = self._tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self._device)
            input_ids = inputs
        else:
            input_ids = self._tok(prompt, return_tensors="pt").input_ids.to(self._device)
        with self._torch.no_grad():
            out = self._model.generate(input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self._tok.decode(out[0][input_ids.shape[-1] :], skip_special_tokens=True).strip()
