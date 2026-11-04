"""편집할 수 있는 하네스 구성요소와 편집 기록.

여기서 '구성요소'는 파일/JSON으로 표현되는 설정뿐이다. 임의 코드, 외부 호출, 파일 경로는 편집 대상이 아니다.
- selector_weights: 선택기 가중치(sim, coverage, diversity, redundancy, reliability)
- k: 질문마다 고르는 문서 수(비교 고정 조건을 바꾸므로 시험표에 표시한다)
- prompt_template: 프롬프트 틀({question}, {evidence}만 허용). 진짜 생성기에 연결하는 일은 아직 안 했다
- query_rules: 질의 변환 규칙(리터럴 문자열 연산만. 정규식과 코드는 허용하지 않는다)
후보 수(candidate_k), 토큰 예산, 검색기는 편집할 수 없다. 모든 시험이 같은 후보 풀을 쓰게 하려는 제한이다.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import string
from dataclasses import dataclass, field
from typing import Any

from kore.pipeline import PipelineConfig
from kore.selector import SelectorWeights

STATUSES = ("proposed", "testing", "awaiting_approval", "adopted", "rejected", "invalid")
STATUS_KO = {
    "proposed": "제안",
    "testing": "시험중",
    "awaiting_approval": "사람 승인 대기",
    "adopted": "채택",
    "rejected": "기각",
    "invalid": "무효",
}
COMPONENTS = ("selector_weights", "k", "prompt_template", "query_rules")
# 이 구성요소를 바꾸면 '같은 후보, 같은 k' 비교 조건 중 무엇이 달라지는지. 시험표에 그대로 적는다.
FIXED_CONDITIONS_CHANGED = {
    "selector_weights": [],
    "k": ["k"],
    "prompt_template": ["prompt"],
    "query_rules": ["candidates", "question_text"],
}
WEIGHT_RANGE = (0.0, 2.0)
K_RANGE = (1, 10)
MAX_TEMPLATE_CHARS = 2000
MAX_RULES = 10
MAX_TERMS = 8
MAX_TERM_CHARS = 30
QUERY_OPS = ("append_terms", "remove_phrases", "replace_literal")
EDIT_KEYS = {"id", "component", "change", "rationale", "confidence", "target_path", "proposer"}
ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,40}$")


class EditError(ValueError):
    """편집이 형식·범위 검사를 통과하지 못했다."""


def canonical_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _short_text(v: Any, name: str, max_chars: int) -> str:
    if not isinstance(v, str) or not v.strip():
        raise EditError(f"{name}은 비어 있지 않은 문자열이어야 한다")
    if len(v) > max_chars:
        raise EditError(f"{name}은 {max_chars}자 이하여야 한다")
    return v


# ---------------------------------------------------------------- 구성요소별 변경 내용 검사


def validate_change(component: str, change: Any) -> None:
    if component not in COMPONENTS:
        raise EditError(f"알 수 없는 구성요소: {component!r} (가능: {COMPONENTS})")
    if not isinstance(change, dict) or not change:
        raise EditError("change는 비어 있지 않은 객체여야 한다")
    if component == "selector_weights":
        names = set(SelectorWeights().as_dict())
        bad = set(change) - names
        if bad:
            raise EditError(f"selector_weights에 모르는 키가 있다: {sorted(bad)} (가능: {sorted(names)})")
        for key, val in change.items():
            if not _is_number(val) or not WEIGHT_RANGE[0] <= val <= WEIGHT_RANGE[1]:
                raise EditError(f"가중치 {key}는 {WEIGHT_RANGE[0]}~{WEIGHT_RANGE[1]} 사이의 숫자여야 한다: {val!r}")
    elif component == "k":
        if set(change) != {"k"}:
            raise EditError("k 편집은 {'k': 정수} 하나만 받는다")
        val = change["k"]
        if isinstance(val, bool) or not isinstance(val, int) or not K_RANGE[0] <= val <= K_RANGE[1]:
            raise EditError(f"k는 {K_RANGE[0]}~{K_RANGE[1]} 사이의 정수여야 한다: {val!r}")
    elif component == "prompt_template":
        if set(change) != {"template"}:
            raise EditError("prompt_template 편집은 {'template': 문자열} 하나만 받는다")
        tpl = _short_text(change["template"], "template", MAX_TEMPLATE_CHARS)
        try:
            fields = {name for _, name, _, _ in string.Formatter().parse(tpl) if name is not None}
        except ValueError as exc:
            raise EditError(f"template의 중괄호 형식이 잘못됐다: {exc}") from exc
        if fields != {"question", "evidence"}:
            raise EditError(f"template의 자리표시자는 {{question}}, {{evidence}} 둘 다 있어야 하고 다른 것은 없어야 한다: {sorted(fields)}")
    elif component == "query_rules":
        if set(change) != {"rules"}:
            raise EditError("query_rules 편집은 {'rules': [..]} 하나만 받는다")
        rules = change["rules"]
        if not isinstance(rules, list) or not rules or len(rules) > MAX_RULES:
            raise EditError(f"rules는 1~{MAX_RULES}개의 목록이어야 한다")
        for rule in rules:
            _validate_rule(rule)


def _validate_rule(rule: Any) -> None:
    if not isinstance(rule, dict) or rule.get("op") not in QUERY_OPS:
        raise EditError(f"규칙의 op는 {QUERY_OPS} 중 하나여야 한다: {rule!r}")
    op = rule["op"]
    if op == "append_terms":
        allowed, listkey = {"op", "terms"}, "terms"
    elif op == "remove_phrases":
        allowed, listkey = {"op", "phrases"}, "phrases"
    else:
        allowed, listkey = {"op", "find", "with"}, None
    if set(rule) - allowed or (allowed - set(rule)):
        raise EditError(f"{op} 규칙의 키는 정확히 {sorted(allowed)}여야 한다: {sorted(rule)}")
    if listkey:
        vals = rule[listkey]
        if not isinstance(vals, list) or not vals or len(vals) > MAX_TERMS:
            raise EditError(f"{listkey}는 1~{MAX_TERMS}개의 목록이어야 한다")
        for v in vals:
            _short_text(v, listkey, MAX_TERM_CHARS)
    else:
        _short_text(rule["find"], "find", MAX_TERM_CHARS)
        if not isinstance(rule["with"], str) or len(rule["with"]) > MAX_TERM_CHARS:
            raise EditError(f"with는 {MAX_TERM_CHARS}자 이하 문자열이어야 한다")


# ---------------------------------------------------------------- 하네스 설정


@dataclass
class HarnessConfig:
    """시험 대상 설정 전체. pipeline은 PipelineConfig.to_dict() 모양이다."""

    pipeline: dict[str, Any]
    prompt_template: str | None = None
    query_rules: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def default(cls) -> "HarnessConfig":
        return cls(pipeline=PipelineConfig().to_dict())

    @classmethod
    def from_pipeline_config(cls, cfg: PipelineConfig) -> "HarnessConfig":
        return cls(pipeline=cfg.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline": copy.deepcopy(self.pipeline),
            "prompt_template": self.prompt_template,
            "query_rules": copy.deepcopy(self.query_rules),
        }

    def hash(self) -> str:
        return canonical_hash(self.to_dict())

    def pipeline_config(self) -> PipelineConfig:
        return PipelineConfig.from_dict(copy.deepcopy(self.pipeline))

    def transform_query(self, question: str) -> str:
        text = question
        for rule in self.query_rules:
            op = rule["op"]
            if op == "append_terms":
                text = text + " " + " ".join(rule["terms"])
            elif op == "remove_phrases":
                for phrase in rule["phrases"]:
                    text = text.replace(phrase, " ")
            else:
                text = text.replace(rule["find"], rule["with"])
        return " ".join(text.split())


def apply_edit(cfg: HarnessConfig, edit: "Edit") -> HarnessConfig:
    """편집을 적용한 새 설정을 돌려준다. 원본은 바꾸지 않는다."""
    validate_change(edit.component, edit.change)
    new = HarnessConfig(**copy.deepcopy(cfg.__dict__))
    if edit.component == "selector_weights":
        new.pipeline["weights"] = {**new.pipeline["weights"], **edit.change}
    elif edit.component == "k":
        new.pipeline["k"] = edit.change["k"]
    elif edit.component == "prompt_template":
        new.prompt_template = edit.change["template"]
    else:
        new.query_rules = copy.deepcopy(edit.change["rules"])
    new.pipeline_config()  # PipelineConfig 검사(k ≤ candidate_k 등)를 여기서 한다
    return new


# ---------------------------------------------------------------- 편집 기록


@dataclass
class Edit:
    id: str
    component: str
    change: dict[str, Any]
    rationale: str  # 근거: 왜 이 편집이 도움이 되리라 보는가
    confidence: float | None = None  # 제안자가 적은 신뢰도(0~1). 검증 결과가 아니다
    status: str = "proposed"
    proposer: str = ""
    target_path: str | None = None  # 편집 적용 결과를 남길 실험 폴더 안 상대 경로
    invalid_reason: str = ""
    result: dict[str, Any] | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def validate(self) -> None:
        if not ID_RE.match(self.id or ""):
            raise EditError(f"id는 영문·숫자·_.-로 1~40자여야 한다: {self.id!r}")
        validate_change(self.component, self.change)
        if not isinstance(self.rationale, str) or len(self.rationale.strip()) < 5:
            raise EditError("rationale(근거)이 비어 있거나 너무 짧다")
        if self.confidence is not None and (not _is_number(self.confidence) or not 0.0 <= self.confidence <= 1.0):
            raise EditError(f"confidence는 0~1 숫자여야 한다: {self.confidence!r}")
        if self.status not in STATUSES:
            raise EditError(f"status는 {STATUSES} 중 하나여야 한다")

    def set_status(self, status: str, why: str = "") -> None:
        if status not in STATUSES:
            raise EditError(f"status는 {STATUSES} 중 하나여야 한다: {status!r}")
        self.history.append({"from": self.status, "to": status, "why": why})
        self.status = status

    def mark_invalid(self, reason: str) -> None:
        self.invalid_reason = reason
        self.set_status("invalid", reason)

    def change_key(self) -> str:
        return canonical_hash({"component": self.component, "change": self.change})

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "component": self.component, "change": self.change, "rationale": self.rationale,
            "confidence": self.confidence, "status": self.status, "proposer": self.proposer,
            "target_path": self.target_path, "invalid_reason": self.invalid_reason, "result": self.result,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Edit":
        """제안 파일의 편집 한 건을 읽는다. 키가 정확해야 하고, 제안은 늘 'proposed'로 시작한다."""
        if not isinstance(data, dict):
            raise EditError("편집은 객체여야 한다")
        extra = set(data) - EDIT_KEYS
        if extra:
            raise EditError(f"편집에 허용되지 않은 키가 있다: {sorted(extra)} (제안 파일은 status·result를 정할 수 없다)")
        missing = {"id", "component", "change", "rationale"} - set(data)
        if missing:
            raise EditError(f"편집에 필수 키가 없다: {sorted(missing)}")
        edit = cls(
            id=data["id"], component=data["component"], change=data["change"], rationale=data["rationale"],
            confidence=data.get("confidence"), proposer=str(data.get("proposer", "")), target_path=data.get("target_path"),
        )
        edit.validate()
        return edit

    @classmethod
    def from_record(cls, data: dict[str, Any]) -> "Edit":
        """체크포인트에 저장된 기록을 되살린다(검사 없이)."""
        return cls(**{k: data[k] for k in data})
