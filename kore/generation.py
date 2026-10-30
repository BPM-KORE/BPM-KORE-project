"""답변 생성기와 답변 점수기.

파이프라인은 Generator 프로토콜(name, generate)만 요구한다. 점수기(AnswerScorer)는 기여도 라벨링에서만 쓴다.
- ExtractiveGenerator: 근거에서 문장을 뽑는 결정적 기준선. 배관 점검용이고 언어모델이 아니다.
- HFGenerator: Hugging Face 오픈 모델(무료 Colab T4 기준). greedy 디코딩. 이 저장소의 테스트 환경에는
  GPU와 torch가 없어서 모델을 실제로 올려 보지 못했다. 프롬프트 조립과 설정 검사까지만 테스트하고,
  모델 로딩·생성·점수는 notebooks/02_colab_gpu_check.ipynb가 Colab에서 확인한다.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Protocol

from kore.schemas import KnowledgeItem
from kore.tokenize import tokenize

_SENT_SPLIT = re.compile(r"(?<=[.!?。])\s+|\n+")

PROMPT_VERSION = "v1"  # build_prompt를 바꾸면 올린다. 생성기 이름에 들어가서 이전 결과와 섞이지 않는다.
DTYPES = ("float16", "bfloat16", "float32", "auto")


class Generator(Protocol):
    name: str

    def generate(self, question: str, evidence: list[KnowledgeItem]) -> str: ...


class AnswerScorer(Protocol):
    """정답 문장이 주어진 근거 아래에서 얼마나 그럴듯한지 점수를 낸다(높을수록 좋다)."""

    name: str

    def score(self, question: str, evidence: list[KnowledgeItem], answer: str) -> float: ...


class PromptTooLongError(ValueError):
    """프롬프트가 max_input_tokens를 넘었다. 조용히 잘라서 비교를 왜곡하지 않고 멈춘다."""


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


def render_prompt(
    tokenizer: Any,
    question: str,
    evidence: list[KnowledgeItem],
    system_prompt: str | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> str:
    """토크나이저의 채팅 템플릿으로 프롬프트 문자열을 만든다. 템플릿이 없으면 일반 문자열을 쓴다.

    apply_chat_template(tokenize=False)로 문자열을 먼저 받고 따로 토큰화한다.
    transformers 버전마다 반환 형식(텐서/딕셔너리)이 달라서 그 차이를 피하려는 선택이다.
    """
    prompt = build_prompt(question, evidence)
    if getattr(tokenizer, "chat_template", None):
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **(chat_template_kwargs or {})
        )
    return f"{system_prompt}\n\n{prompt}" if system_prompt else prompt


@dataclass(frozen=True)
class GeneratorSpec:
    """생성기 설정. configs/generator.json이 이 모양이다."""

    kind: str = "hf"  # none | extractive | hf
    model: str = "Qwen/Qwen3-4B-Instruct-2507"
    revision: str | None = None  # 재현하려면 모델 저장소의 커밋 해시를 적는다
    dtype: str = "float16"  # Colab T4는 bfloat16을 기본 지원하지 않는다
    max_new_tokens: int = 96
    max_input_tokens: int = 4096
    system_prompt: str | None = None
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    trust_remote_code: bool = False
    load_in_4bit: bool = False

    def validate(self) -> None:
        if self.kind not in ("none", "extractive", "hf"):
            raise ValueError(f"kind는 none/extractive/hf 중 하나여야 한다: {self.kind!r}")
        if self.dtype not in DTYPES:
            raise ValueError(f"dtype은 {DTYPES} 중 하나여야 한다: {self.dtype!r}")
        if self.max_new_tokens < 1 or self.max_input_tokens < 64:
            raise ValueError("max_new_tokens는 1 이상, max_input_tokens는 64 이상이어야 한다")
        if self.kind == "hf" and not self.model:
            raise ValueError("kind가 hf이면 model이 필요하다")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GeneratorSpec":
        """모르는 키는 오타일 수 있으니 오류를 낸다. '_'로 시작하는 키(메모)는 무시한다."""
        data = {k: v for k, v in data.items() if not k.startswith("_")}
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"생성기 설정에 모르는 키가 있다: {sorted(unknown)}")
        spec = cls(**data)
        spec.validate()
        return spec

    @classmethod
    def from_json(cls, path: str | Path) -> "GeneratorSpec":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def _short_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:6]


class HFGenerator:
    """greedy 디코딩으로 고정한 Hugging Face 생성기이자 답변 점수기.

    같은 입력이면 같은 출력이 나오게 한다(같은 GPU 종류, 같은 라이브러리 버전 기준).
    이름(name)에 모델·리비전·dtype·장치·프롬프트 버전이 들어가서, 이 중 하나가 바뀌면 이전 결과와 섞이지 않는다.
    """

    def __init__(self, spec: GeneratorSpec | str | None = None, max_new_tokens: int | None = None) -> None:
        if spec is None:
            spec = GeneratorSpec()
        elif isinstance(spec, str):  # 예전 호출 방식(모델 이름만)도 받는다
            spec = GeneratorSpec(model=spec)
        if max_new_tokens is not None:
            spec = GeneratorSpec.from_dict({**spec.to_dict(), "max_new_tokens": max_new_tokens})
        spec.validate()
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - 선택 의존성
            raise ImportError('pip install -e ".[hf]" 로 transformers와 torch를 설치해야 한다') from exc
        self.spec = spec
        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = spec.dtype
        if self._device == "cpu" and dtype in ("float16", "bfloat16"):
            dtype = "float32"  # CPU에서 반정밀도는 느리거나 지원되지 않는다
        self.effective_dtype = dtype
        sys_hash = _short_hash([spec.system_prompt, spec.chat_template_kwargs])
        self.name = (
            f"hf:{spec.model}@{spec.revision or 'main'}|{dtype}|{self._device}|new{spec.max_new_tokens}"
            f"|prompt-{PROMPT_VERSION}-{sys_hash}" + ("|4bit" if spec.load_in_4bit else "")
        )
        common: dict[str, Any] = {"trust_remote_code": spec.trust_remote_code}
        if spec.revision:
            common["revision"] = spec.revision
        self._tok = AutoTokenizer.from_pretrained(spec.model, **common)
        load_kwargs: dict[str, Any] = dict(common, low_cpu_mem_usage=True)
        if spec.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
                )
                load_kwargs["device_map"] = "auto"
            except ImportError as exc:  # pragma: no cover
                raise ImportError("4비트 로딩에는 bitsandbytes가 필요하다: pip install bitsandbytes") from exc
        torch_dtype = "auto" if dtype == "auto" else getattr(torch, dtype)
        try:  # 최신 transformers는 dtype=, 오래된 것은 torch_dtype=
            self._model = AutoModelForCausalLM.from_pretrained(spec.model, dtype=torch_dtype, **load_kwargs)
        except TypeError:  # pragma: no cover
            self._model = AutoModelForCausalLM.from_pretrained(spec.model, torch_dtype=torch_dtype, **load_kwargs)
        if not spec.load_in_4bit:
            self._model.to(self._device)
        self._model.eval()

    # ---------------------------------------------------------------- 생성

    def _prompt(self, question: str, evidence: list[KnowledgeItem]) -> str:
        return render_prompt(self._tok, question, evidence, self.spec.system_prompt, self.spec.chat_template_kwargs)

    def _pad_id(self) -> int | None:
        pad = self._tok.pad_token_id
        return pad if pad is not None else self._tok.eos_token_id

    def generate(self, question: str, evidence: list[KnowledgeItem]) -> str:  # pragma: no cover - 모델 필요
        enc = self._tok(self._prompt(question, evidence), return_tensors="pt", add_special_tokens=False)
        n_in = int(enc["input_ids"].shape[1])
        if n_in > self.spec.max_input_tokens:
            raise PromptTooLongError(
                f"프롬프트가 {n_in}토큰이라 max_input_tokens={self.spec.max_input_tokens}를 넘는다. "
                "token_budget을 줄이거나 문서를 짧게 나눈다(조건마다 다르게 잘리면 비교가 왜곡된다)."
            )
        enc = enc.to(self._device)
        with self._torch.inference_mode():
            out = self._model.generate(
                **enc, max_new_tokens=self.spec.max_new_tokens, do_sample=False, pad_token_id=self._pad_id()
            )
        return self._tok.decode(out[0][n_in:], skip_special_tokens=True).strip()

    # ---------------------------------------------------------------- 점수

    def answer_logprob(  # pragma: no cover - 모델 필요
        self, question: str, evidence: list[KnowledgeItem], answer: str
    ) -> dict[str, float]:
        """정답 문장의 토큰별 로그확률(프롬프트가 주어졌을 때). 정답 구간의 로짓만 계산해서 메모리를 아낀다."""
        if not answer.strip():
            raise ValueError("점수를 낼 정답 문장이 비었다")
        torch = self._torch
        p_ids = self._tok(self._prompt(question, evidence), add_special_tokens=False)["input_ids"]
        a_ids = self._tok(answer.strip(), add_special_tokens=False)["input_ids"]
        if len(p_ids) + len(a_ids) > self.spec.max_input_tokens:
            raise PromptTooLongError(
                f"프롬프트+정답이 {len(p_ids) + len(a_ids)}토큰이라 max_input_tokens를 넘는다"
            )
        ids = torch.tensor([p_ids + a_ids], device=self._device)
        with torch.inference_mode():
            hidden = self._model.base_model(input_ids=ids).last_hidden_state
            tail = hidden[:, len(p_ids) - 1 : -1, :]
            logits = self._model.get_output_embeddings()(tail).float()
            logp = torch.log_softmax(logits, dim=-1)
            target = ids[:, len(p_ids) :]
            token_lp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        total = float(token_lp.sum())
        return {"sum": total, "n_tokens": float(len(a_ids)), "mean": total / len(a_ids)}

    def score(self, question: str, evidence: list[KnowledgeItem], answer: str) -> float:  # pragma: no cover
        """AnswerScorer 규칙: 정답 토큰 평균 로그확률(높을수록 좋다)."""
        return self.answer_logprob(question, evidence, answer)["mean"]

    def check_scoring(  # pragma: no cover - 모델 필요
        self, question: str, evidence: list[KnowledgeItem], answer: str
    ) -> float:
        """answer_logprob의 지름길 계산이 전체 순전파와 같은 값을 내는지 본다. 평균 로그확률 차이의 절댓값을 돌려준다.

        모델 구조(로짓 스케일링 등) 때문에 지름길이 어긋나는 경우를 Colab 첫 실행에서 잡으려는 점검이다.
        """
        torch = self._torch
        p_ids = self._tok(self._prompt(question, evidence), add_special_tokens=False)["input_ids"]
        a_ids = self._tok(answer.strip(), add_special_tokens=False)["input_ids"]
        ids = torch.tensor([p_ids + a_ids], device=self._device)
        with torch.inference_mode():
            logits = self._model(input_ids=ids).logits[:, len(p_ids) - 1 : -1, :].float()
            full = torch.log_softmax(logits, dim=-1).gather(-1, ids[:, len(p_ids) :].unsqueeze(-1)).sum()
        return abs(float(full) / len(a_ids) - self.answer_logprob(question, evidence, answer)["mean"])


def load_generator(spec: GeneratorSpec | None) -> Generator | None:
    """설정에서 생성기를 만든다. none이면 None(생성 없이 검색 지표만 잰다)."""
    if spec is None or spec.kind == "none":
        return None
    if spec.kind == "extractive":
        return ExtractiveGenerator()
    return HFGenerator(spec)
