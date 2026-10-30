"""공통 데이터 계약. A·B·C가 공유하는 유일한 기준이다.

계약을 바꾸려면 docs/CONTRACT.md와 이 파일을 한 PR에서 같이 고치고 세 사람이 모두 확인한다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

CONTRACT_VERSION = "1.0"

SPLITS = ("train", "dev", "test")
DIFFICULTIES = ("easy", "medium", "hard")
LIST_SEP = "|"  # expected_source_ids, rubric 항목 구분자
ALT_SEP = "/"  # rubric 한 항목 안의 동의어 구분자

KNOWLEDGE_REQUIRED = ("id", "domain", "title", "text", "source_url", "license")
KNOWLEDGE_OPTIONAL = ("reliability", "lang")
QUESTION_REQUIRED = (
    "question_id",
    "question",
    "expected_source_ids",
    "domain",
    "difficulty",
    "split",
)
QUESTION_OPTIONAL = ("reference_answer", "rubric")
RELATION_REQUIRED = ("src_id", "dst_id", "relation")

DEFAULT_RELIABILITY = 0.5


@dataclass
class KnowledgeItem:
    """지식 한 조각.

    score는 계약 호환용으로 남겼다. 가장 최근 단계(검색 또는 선택)의 점수가 들어간다.
    단계별 점수는 retrieval_score와 selection_score에 따로 담는다.
    """

    id: str
    domain: str
    title: str
    text: str
    source_url: str
    license: str
    score: float = 0.0
    retrieval_score: float = 0.0
    selection_score: float = 0.0
    reliability: float = DEFAULT_RELIABILITY
    lang: str = ""

    def copy_with(self, **changes: Any) -> "KnowledgeItem":
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvalQuestion:
    """평가 질문 한 개. rubric은 '항목|항목' 형식이고 항목 안의 동의어는 '/'로 잇는다."""

    question_id: str
    question: str
    expected_source_ids: list[str]
    domain: str
    difficulty: str = "medium"
    split: str = "dev"
    reference_answer: str = ""
    rubric: str = ""

    def rubric_items(self) -> list[list[str]]:
        items: list[list[str]] = []
        for raw in self.rubric.split(LIST_SEP):
            alts = [a.strip() for a in raw.split(ALT_SEP) if a.strip()]
            if alts:
                items.append(alts)
        return items

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
