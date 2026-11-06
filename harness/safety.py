"""안전 장치. 이 파일은 금지 패턴을 정의하므로 소스 검사(scan_source_file)의 문자열 검사에서 빠진다.

- test 분할 질문은 읽지도 않는다(행 단계에서 버리고 객체로 만들지 않는다).
- 제안 안의 모든 문자열에서 외부 호출·유료 서비스·test 접근 시도를 찾아 거부한다.
- 편집 결과를 쓰는 경로는 실험 폴더 안으로 제한한다.
- 하네스 코드 자체에 외부 호출 import나 URL이 없는지 AST로 검사한다.
"""

from __future__ import annotations

import ast
import csv
import re
from pathlib import Path
from typing import Any, Iterable

from kore.schemas import LIST_SEP, EvalQuestion, KnowledgeItem

ALLOWED_SPLITS = ("train", "dev")
DEFAULT_SPLITS = ("dev",)


class SafetyError(ValueError):
    """안전 규칙 위반. 사람이 읽고 고쳐야 한다."""


class SplitViolation(SafetyError):
    """test 분할에 접근하려 했다."""


class PathEscape(SafetyError):
    """실험 폴더 밖에 쓰려 했다."""


# (이름, 정규식). 제안 파일의 키와 문자열 값 전부에 적용한다.
FORBIDDEN_TEXT: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("외부 URL", re.compile(r"(?i)\b(?:https?|ftp|wss?)://")),
    ("키·토큰", re.compile(r"(?i)api[_\- ]?key|secret[_\- ]?key|access[_\- ]?token|\bbearer\s")),
    ("유료·외부 서비스", re.compile(r"(?i)\b(vertex|openai|anthropic|gemini|bedrock|azure|cohere)\b|유료|결제")),
    ("네트워크 호출", re.compile(r"(?i)\b(requests|urllib|httpx|aiohttp|curl|wget|socket)\b")),
    ("test 분할 접근", re.compile(
        r"(?i)final[-_ ]?test|include[_\- ]?test|blind[_\-]?key|\bsplits?\W{0,6}test\b|\btest\s*(분할|split|질문|셋|set)")),
)
FORBIDDEN_IMPORTS = frozenset(
    {"requests", "urllib", "urllib3", "http", "socket", "ftplib", "smtplib", "httpx", "aiohttp", "openai", "anthropic",
     "google", "subprocess", "torch", "transformers"}
)
_URL_OR_KEY = re.compile(r"(?i)\b(?:https?|ftp|wss?)://|api[_\- ]?key")


def scan_text(text: str) -> list[str]:
    return [name for name, pat in FORBIDDEN_TEXT if pat.search(text)]


def scan_object(obj: Any, where: str = "$") -> list[str]:
    """JSON 객체의 키와 문자열 값을 모두 훑어 위반 목록('위치: 이름')을 돌려준다."""
    found: list[str] = []
    if isinstance(obj, str):
        found += [f"{where}: {name}" for name in scan_text(obj)]
    elif isinstance(obj, dict):
        for key, val in obj.items():
            found += [f"{where}.{key}(키): {name}" for name in scan_text(str(key))]
            found += scan_object(val, f"{where}.{key}")
    elif isinstance(obj, (list, tuple)):
        for i, val in enumerate(obj):
            found += scan_object(val, f"{where}[{i}]")
    return found


def ensure_safe_object(obj: Any, label: str = "제안") -> None:
    found = scan_object(obj)
    if found:
        raise SafetyError(f"{label}에 금지된 내용이 있다: " + "; ".join(found))


# ---------------------------------------------------------------- 경로


def resolve_inside(root: str | Path, rel: str | Path) -> Path:
    """root 안의 경로만 돌려준다. 절대 경로, '..', 심볼릭 링크 탈출은 PathEscape."""
    root_p = Path(root).resolve()
    rel_p = Path(rel)
    if rel_p.is_absolute() or any(part == ".." for part in rel_p.parts) or not str(rel).strip():
        raise PathEscape(f"실험 폴더 밖이거나 잘못된 경로다: {str(rel)!r}")
    target = (root_p / rel_p).resolve()
    if target != root_p and root_p not in target.parents:
        raise PathEscape(f"실험 폴더 밖으로 나간다: {str(rel)!r}")
    return target


# ---------------------------------------------------------------- 분할


def assert_no_test(questions: Iterable[EvalQuestion], allowed: tuple[str, ...] = DEFAULT_SPLITS) -> None:
    """질문 목록에 test 또는 허용되지 않은 분할이 있으면 SplitViolation."""
    check_splits(allowed)
    for q in questions:
        if q.split == "test" or q.split not in allowed:
            raise SplitViolation(f"질문 {q.question_id}의 분할 '{q.split}'은 허용되지 않는다(허용: {allowed}). test는 최종 평가에서 사람이 연다")


def check_splits(splits: Iterable[str]) -> tuple[str, ...]:
    splits = tuple(splits)
    if "test" in splits:
        raise SplitViolation("test 분할은 읽을 수 없다. train 또는 dev만 지정한다")
    bad = [s for s in splits if s not in ALLOWED_SPLITS]
    if bad or not splits:
        raise SplitViolation(f"허용되는 분할은 {ALLOWED_SPLITS}뿐이다: {splits}")
    return splits


def load_dev_questions(
    path: str | Path, items: list[KnowledgeItem], splits: tuple[str, ...] = DEFAULT_SPLITS
) -> list[EvalQuestion]:
    """질문 CSV에서 허용된 분할의 행만 객체로 만든다. test 행은 객체로 만들지 않고 건너뛴다.

    kore.io.load_questions는 파일 전체를 해석하므로 쓰지 않는다. 데이터 계약 검사는 `python -m kore validate`가 맡는다.
    """
    splits = check_splits(splits)
    known = {i.id for i in items}
    out: list[EvalQuestion] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8-sig", newline="") as fh:
        for n, row in enumerate(csv.DictReader(fh), start=2):
            split = (row.get("split") or "").strip()
            if split not in splits:
                continue
            qid = (row.get("question_id") or "").strip()
            expected = [x.strip() for x in (row.get("expected_source_ids") or "").split(LIST_SEP) if x.strip()]
            if not qid or not (row.get("question") or "").strip() or not expected:
                raise SafetyError(f"{path} {n}행: question_id, question, expected_source_ids가 필요하다")
            if qid in seen:
                raise SafetyError(f"{path} {n}행: question_id {qid!r} 중복")
            seen.add(qid)
            missing = [e for e in expected if e not in known]
            if missing:
                raise SafetyError(f"{path} {n}행: knowledge에 없는 근거 id {missing}")
            out.append(
                EvalQuestion(
                    question_id=qid, question=row["question"].strip(), expected_source_ids=expected,
                    domain=(row.get("domain") or "").strip(), difficulty=(row.get("difficulty") or "medium").strip(),
                    split=split, reference_answer=row.get("reference_answer") or "", rubric=row.get("rubric") or "",
                )
            )
    if not out:
        raise SafetyError(f"{path}에 분할 {splits}의 질문이 없다")
    return out


# ---------------------------------------------------------------- 소스 검사


def scan_source_file(path: str | Path, check_strings: bool = True) -> list[str]:
    """파이썬 파일에서 외부 호출 import와(옵션) URL·키 문자열을 찾는다."""
    path = Path(path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        mods: list[str] = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods = [node.module]
        for mod in mods:
            if mod.split(".")[0] in FORBIDDEN_IMPORTS:
                found.append(f"{path.name}:{node.lineno}: 금지된 import {mod}")
        if check_strings and isinstance(node, ast.Constant) and isinstance(node.value, str) and _URL_OR_KEY.search(node.value):
            found.append(f"{path.name}:{node.lineno}: URL 또는 키 문자열")
    return found
