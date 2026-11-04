"""편집된 하네스 설정을 kore의 실제 실험 경로(run_experiment, 생성기 인터페이스)로 넘기는 어댑터.

배관 연결용이다. 성능을 재는 코드가 아니다.

연결한 것
- k, candidate_k, token_budget, 선택기 가중치: HarnessConfig.pipeline_config() -> PipelineConfig -> run_experiment(config=).
- query_rules: 질문 문자열을 바꿔서 run_experiment에 넘긴다(질문 id와 분할은 그대로).
- prompt_template: kore.generation.build_prompt는 고정 함수라서(kore/ 수정 금지) 템플릿을 받지 못한다.
  대신 Generator 프로토콜(name, generate)을 따르는 TemplateGenerator가 템플릿으로 프롬프트를 직접 만들어
  '프롬프트 문자열을 받는 모델'(PromptModel)에 넘긴다. 템플릿이 없으면 build_prompt와 글자 하나까지 같다.

실행하지 않은 것
- HFTemplateGenerator는 HFGenerator를 상속해 _prompt만 템플릿으로 바꾼 클래스다. import만 확인했다.
  GPU와 torch가 없는 환경이라 모델 로딩, 생성, 토큰 수 검사는 한 번도 실행하지 못했다. Colab GPU에서 처음 확인한다.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

from harness.components import HarnessConfig
from kore.generation import ExtractiveGenerator, Generator, HFGenerator, PromptTooLongError, build_prompt
from kore.pipeline import PipelineConfig, run_experiment
from kore.schemas import EvalQuestion, KnowledgeItem
from kore.tokenize import estimate_tokens

SETTING_PIPELINE_KEYS = ("k", "candidate_k", "token_budget", "tokenizer_mode", "retriever", "hybrid_alpha")
# 비교 후보 사이에서 같아야 하는 설정. 편집이 일부러 바꾸는 키는 check_config_parity의 vary로 허용한다.
PARITY_KEYS = ("k", "candidate_k", "token_budget", "tokenizer_mode", "retriever", "prompt_template", "query_rules")


class ParityError(ValueError):
    """비교 후보들이 같은 질문·k·토큰 예산 조건에서 돌지 않았다."""


class PromptModel(Protocol):
    """프롬프트 문자열 하나를 받아 답을 돌려주는 모델. 가짜 모델과 실제 모델이 같은 모양을 쓴다."""

    name: str

    def complete(self, prompt: str) -> str: ...


# ---------------------------------------------------------------- 설정 변환


def harness_config_from_settings(settings: dict[str, Any]) -> HarnessConfig:
    """편집 설정 JSON을 HarnessConfig로 바꾼다.

    두 모양을 받는다. (1) HarnessConfig.to_dict() 모양 {"pipeline": {...}, "prompt_template", "query_rules"}.
    (2) 평평한 모양 {"k", "candidate_k", "token_budget", "weights", "prompt_template", "query_rules"}.
    모르는 키는 오류를 낸다(조용히 무시하면 편집이 안 넘어간 것을 놓친다).
    """
    s = dict(settings)
    if "pipeline" in s:
        extra = set(s) - {"pipeline", "prompt_template", "query_rules"}
        if extra:
            raise ValueError(f"설정에 모르는 키가 있다: {sorted(extra)}")
        pipe = dict(s["pipeline"])
    else:
        top = {"prompt_template", "query_rules", "weights"}
        extra = set(s) - top - set(SETTING_PIPELINE_KEYS)
        if extra:
            raise ValueError(f"설정에 모르는 키가 있다: {sorted(extra)}")
        pipe = PipelineConfig().to_dict()
        pipe.update({k: s[k] for k in SETTING_PIPELINE_KEYS if k in s})
        if "weights" in s:
            pipe["weights"] = {**pipe["weights"], **s["weights"]}
    cfg = HarnessConfig(
        pipeline=pipe,
        prompt_template=s.get("prompt_template"),
        query_rules=list(s.get("query_rules") or []),
    )
    cfg.pipeline_config()  # PipelineConfig 검사(k <= candidate_k 등)
    if cfg.prompt_template is not None:
        render_prompt_text(cfg.prompt_template, "q", [KnowledgeItem(id="_", title="", text="", domain="", source_url="", license="")])  # 자리표시자 검사
    return cfg


# ---------------------------------------------------------------- 프롬프트


def _evidence_blocks(evidence: list[KnowledgeItem]) -> str:
    """build_prompt와 같은 근거 블록 형식."""
    return "\n\n".join(
        f"[근거 {n}] (id={e.id}, 출처={e.source_url})\n{e.text}" for n, e in enumerate(evidence, start=1)
    )


def render_prompt_text(template: str | None, question: str, evidence: list[KnowledgeItem]) -> str:
    """템플릿으로 프롬프트를 만든다. 템플릿이 없거나 근거가 없으면(no_retrieval) build_prompt를 그대로 쓴다."""
    if template is None or not evidence:
        return build_prompt(question, evidence)
    try:
        return template.format(question=question, evidence=_evidence_blocks(evidence))
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(f"프롬프트 템플릿에는 {{question}}, {{evidence}}만 쓸 수 있다: {exc}") from exc


def _template_tag(template: str | None) -> str:
    return "" if template is None else "|tpl-" + hashlib.sha256(template.encode("utf-8")).hexdigest()[:6]


# ---------------------------------------------------------------- 생성기


class FakePromptModel:
    """결정적 가짜 모델. 프롬프트가 한 글자라도 다르면 답에 붙는 꼬리표(프롬프트 해시)가 달라진다.

    답 본문은 ExtractiveGenerator처럼 근거 문장을 뽑는다. 언어모델이 아니고 점수는 연구 결과가 아니다.
    """

    name = "fake-prompt-model-v0"

    def __init__(self) -> None:
        self._ext = ExtractiveGenerator()

    def complete(self, prompt: str) -> str:
        tag = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
        body = self._ext.generate(self._question_of(prompt), self._evidence_of(prompt))
        return f"{body} [p:{tag}]".strip()

    @staticmethod
    def _question_of(prompt: str) -> str:
        return prompt

    @staticmethod
    def _evidence_of(prompt: str) -> list[KnowledgeItem]:
        # 프롬프트 전체를 근거 한 덩어리로 본다. 배관 시험용이라 충분하다.
        return [KnowledgeItem(id="_p", title="", text=prompt, domain="", source_url="", license="")]


class TemplateGenerator:
    """Generator 프로토콜(name, generate)을 따른다. 템플릿으로 프롬프트를 만들어 PromptModel에 넘긴다."""

    def __init__(self, model: PromptModel | None = None, template: str | None = None,
                 max_input_tokens: int | None = None) -> None:
        self.model = model or FakePromptModel()
        self.template = template
        self.max_input_tokens = max_input_tokens
        self.name = f"tpl:{self.model.name}{_template_tag(template)}"
        self.prompts: list[str] = []  # 실제로 모델에 들어간 프롬프트(검사용)

    def prompt_for(self, question: str, evidence: list[KnowledgeItem]) -> str:
        return render_prompt_text(self.template, question, evidence)

    def generate(self, question: str, evidence: list[KnowledgeItem]) -> str:
        prompt = self.prompt_for(question, evidence)
        if self.max_input_tokens is not None and estimate_tokens(prompt) > self.max_input_tokens:
            raise PromptTooLongError("프롬프트가 max_input_tokens를 넘는다(조용히 자르지 않고 멈춘다)")
        self.prompts.append(prompt)
        return self.model.complete(prompt)


class HFTemplateGenerator(HFGenerator):  # pragma: no cover - GPU 필요
    """HFGenerator의 프롬프트 조립만 템플릿으로 바꾼다. 이 환경(GPU 없음)에서는 실행하지 않았다.

    생성과 점수 경로(generate, answer_logprob)는 부모를 그대로 쓰고 _prompt를 거친다.
    첫 확인은 Colab GPU에서 한다. 이름(name)에 템플릿 해시를 붙여 결과가 섞이지 않게 한다.
    """

    def __init__(self, template: str | None = None, *args: Any, **kwargs: Any) -> None:
        self.template = template
        super().__init__(*args, **kwargs)
        self.name += _template_tag(template)

    def _prompt(self, question: str, evidence: list[KnowledgeItem]) -> str:
        if self.template is None or not evidence:
            return super()._prompt(question, evidence)
        text = render_prompt_text(self.template, question, evidence)
        sp = self.spec.system_prompt
        if getattr(self._tok, "chat_template", None):
            msgs = ([{"role": "system", "content": sp}] if sp else []) + [{"role": "user", "content": text}]
            return self._tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, **(self.spec.chat_template_kwargs or {})
            )
        return f"{sp}\n\n{text}" if sp else text


def make_generator(cfg: HarnessConfig, kind: str = "fake", model: PromptModel | None = None,
                   hf_spec: Any = None) -> Generator:
    """kind='fake'가 기본이다. kind='hf'는 GPU 환경에서만 부른다."""
    if kind == "fake":
        return TemplateGenerator(model, cfg.prompt_template)
    if kind == "hf":  # pragma: no cover
        return HFTemplateGenerator(cfg.prompt_template, hf_spec)
    raise ValueError(f"kind는 fake 또는 hf여야 한다: {kind!r}")


# ---------------------------------------------------------------- 실행 인자와 실행


def to_run_kwargs(cfg: HarnessConfig, questions: list[EvalQuestion], generator: Generator | None) -> dict[str, Any]:
    """run_experiment에 넘길 인자. query_rules는 여기서 질문 문자열에 적용한다."""
    qs = [replace(q, question=cfg.transform_query(q.question)) for q in questions]
    return {"questions": qs, "config": cfg.pipeline_config(), "generator": generator}


def run_adapted(items: list[KnowledgeItem], questions: list[EvalQuestion], cfg: HarnessConfig,
                out_dir: str | Path, generator: Generator | None = None, conditions: tuple[str, ...] = ("kore",),
                seed: int = 0) -> dict[str, Any]:
    """편집된 설정으로 run_experiment를 부른다. test 분할은 켜지 않는다."""
    kw = to_run_kwargs(cfg, questions, generator)
    return run_experiment(items, kw["questions"], out_dir, conditions=conditions, config=kw["config"],
                          generator=kw["generator"], seed=seed, include_test=False, resume=False)


# ---------------------------------------------------------------- 같은 조건 검사


def check_config_parity(configs: dict[str, HarnessConfig], vary: tuple[str, ...] = ()) -> list[str]:
    """후보 설정들이 PARITY_KEYS에서 같은지 본다. 편집이 일부러 바꾸는 키는 vary로 허용한다. 위반 목록을 돌려준다."""
    problems: list[str] = []
    views = {n: _parity_view(c) for n, c in configs.items()}
    names = list(views)
    for key in PARITY_KEYS:
        if key in vary:
            continue
        vals = {n: repr(views[n][key]) for n in names}
        if len(set(vals.values())) > 1:
            problems.append(f"{key}가 후보마다 다르다: {vals}")
    return problems


def _parity_view(cfg: HarnessConfig) -> dict[str, Any]:
    p = cfg.pipeline_config()
    return {"k": p.k, "candidate_k": p.candidate_k, "token_budget": p.token_budget,
            "tokenizer_mode": p.tokenizer_mode, "retriever": p.retriever,
            "prompt_template": cfg.prompt_template, "query_rules": cfg.query_rules}


def check_run_parity(rows_by_candidate: dict[str, list[dict[str, Any]]], configs: dict[str, HarnessConfig],
                     vary: tuple[str, ...] = (), same_candidates: bool = True) -> list[str]:
    """실행 로그(run_log.jsonl 행)가 같은 질문 집합을 쓰고 각 후보의 k와 토큰 예산을 지켰는지 본다.

    같은 질문과 같은 검색 조건이면 후보 문서 목록도 같아야 한다(query_rules를 vary로 허용하면 이 검사는 건너뛴다).
    """
    problems = check_config_parity(configs, vary)
    qsets = {n: {r["question_id"] for r in rows} for n, rows in rows_by_candidate.items()}
    if len({frozenset(s) for s in qsets.values()}) > 1:
        problems.append(f"후보마다 질문 집합이 다르다: { {n: sorted(s) for n, s in qsets.items()} }")
    for n, rows in rows_by_candidate.items():
        p = configs[n].pipeline_config()
        for r in rows:
            if len(r["selected_ids"]) > p.k:
                problems.append(f"{n}/{r['question_id']}: 선택 {len(r['selected_ids'])}개가 k={p.k}를 넘는다")
            if p.token_budget is not None and r["context_tokens"] > p.token_budget:
                problems.append(f"{n}/{r['question_id']}: 문맥 {r['context_tokens']}토큰이 예산 {p.token_budget}을 넘는다")
    if same_candidates and "query_rules" not in vary:
        by_q: dict[str, set[tuple[str, ...]]] = {}
        for rows in rows_by_candidate.values():
            for r in rows:
                by_q.setdefault(r["question_id"], set()).add(tuple(r["candidate_ids"]))
        for qid, sets in sorted(by_q.items()):
            if len(sets) > 1:
                problems.append(f"{qid}: 후보 문서 목록이 후보 설정마다 다르다")
    return problems


def assert_parity(problems: list[str]) -> None:
    if problems:
        raise ParityError("; ".join(problems))
