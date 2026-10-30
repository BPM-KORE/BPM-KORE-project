"""knowledge / evaluation / relations 파일의 읽기, 쓰기, 검증.

B와 C는 파일을 올리기 전에 `python -m kore validate`로 이 검증기를 통과시킨다.
errors가 하나라도 있으면 통합하지 않고, warnings는 사람이 읽고 판단한다.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kore.schemas import (
    DEFAULT_RELIABILITY,
    DIFFICULTIES,
    KNOWLEDGE_OPTIONAL,
    KNOWLEDGE_REQUIRED,
    LIST_SEP,
    QUESTION_OPTIONAL,
    QUESTION_REQUIRED,
    SPLITS,
    EvalQuestion,
    KnowledgeItem,
)
from kore.splits import near_duplicate_pairs
from kore.tokenize import normalize

ID_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")
URL_PREFIXES = ("http://", "https://", "local:")
UNKNOWN_LICENSES = {"unknown", "?", "n/a", "na", "none", "tbd"}
MIN_TEXT_CHARS = 30
MAX_TEXT_CHARS = 6000
MIN_SPLIT_SIZE = 10


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


class DataError(ValueError):
    def __init__(self, report: ValidationReport, path: str = "") -> None:
        self.report = report
        head = f"{path}: " if path else ""
        super().__init__(head + f"데이터 오류 {len(report.errors)}건: " + "; ".join(report.errors[:5]))


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _read_rows(path: Path, report: ValidationReport) -> tuple[list[dict[str, str]], set[str]]:
    """파일을 행 목록으로 읽는다. 반환: (행, 존재하는 열 이름 집합)."""
    if not path.exists():
        report.error(f"{path}: 파일이 없다")
        return [], set()
    suffix = path.suffix.lower()
    rows: list[dict[str, str]] = []
    columns: set[str] = set()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            columns = {c.strip() for c in (reader.fieldnames or []) if c}
            for raw in reader:
                rows.append({_clean(k): _clean(v) for k, v in raw.items() if k is not None})
    elif suffix == ".jsonl":
        with path.open(encoding="utf-8-sig") as fh:
            for lineno, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    report.error(f"{path.name}:{lineno}: JSON 구문 오류 ({exc.msg})")
                    continue
                if not isinstance(obj, dict):
                    report.error(f"{path.name}:{lineno}: 객체가 아니다")
                    continue
                row: dict[str, str] = {}
                for k, v in obj.items():
                    if isinstance(v, list):
                        row[_clean(k)] = LIST_SEP.join(_clean(x) for x in v)
                    else:
                        row[_clean(k)] = _clean(v)
                rows.append(row)
                columns |= set(row)
    else:
        report.error(f"{path}: 지원하지 않는 확장자다(.csv 또는 .jsonl만 쓴다)")
    return rows, columns


def _lineno(idx: int, path: Path) -> str:
    # CSV는 1행이 헤더라서 데이터 첫 줄이 2행이다.
    return f"{path.name}:{idx + (2 if path.suffix.lower() == '.csv' else 1)}"


def read_knowledge(path: str | Path) -> tuple[list[KnowledgeItem], ValidationReport]:
    path = Path(path)
    report = ValidationReport()
    rows, columns = _read_rows(path, report)
    missing = [c for c in KNOWLEDGE_REQUIRED if c not in columns]
    if report.errors and not rows:  # 파일을 아예 못 읽은 경우만 여기서 멈춘다. 일부 줄만 깨졌으면 나머지는 계속 검사한다
        return [], report
    if missing:
        report.error(f"{path.name}: 필수 열이 없다: {', '.join(missing)}")
        return [], report
    unknown = columns - set(KNOWLEDGE_REQUIRED) - set(KNOWLEDGE_OPTIONAL)
    if unknown:
        report.warn(f"{path.name}: 계약에 없는 열은 무시한다: {', '.join(sorted(unknown))}")

    items: list[KnowledgeItem] = []
    seen_ids: dict[str, int] = {}
    seen_text: dict[str, str] = {}
    for idx, row in enumerate(rows):
        where = _lineno(idx, path)
        ok = True
        for col in KNOWLEDGE_REQUIRED:
            if not row.get(col):
                report.error(f"{where}: '{col}'이 비어 있다")
                ok = False
        item_id = row.get("id", "")
        if item_id:
            if not ID_RE.match(item_id):
                report.error(f"{where}: id '{item_id}'에는 영문·숫자·_ . - 만 쓸 수 있다")
                ok = False
            if item_id in seen_ids:
                report.error(f"{where}: id '{item_id}'가 {seen_ids[item_id]}번째 행과 중복이다")
                ok = False
            else:
                seen_ids[item_id] = idx + 1
        reliability = DEFAULT_RELIABILITY
        raw_rel = row.get("reliability", "")
        if raw_rel:
            try:
                reliability = float(raw_rel)
                if not 0.0 <= reliability <= 1.0:
                    raise ValueError
            except ValueError:
                report.error(f"{where}: reliability는 0~1 사이 숫자여야 한다: {raw_rel!r}")
                ok = False
        lang = row.get("lang", "")
        if lang and lang not in ("ko", "en"):
            report.warn(f"{where}: lang '{lang}'은 ko/en이 아니다")
        url = row.get("source_url", "")
        if url and not url.startswith(URL_PREFIXES):
            report.warn(f"{where}: source_url이 http(s)나 local:로 시작하지 않는다: {url!r}")
        if row.get("license", "").lower() in UNKNOWN_LICENSES:
            report.warn(f"{where}: license가 확인되지 않았다({row['license']!r}). 출처 검증 전에는 연구에 쓰지 않는다")
        text = row.get("text", "")
        if text:
            if len(text) < MIN_TEXT_CHARS:
                report.warn(f"{where}: text가 {len(text)}자로 매우 짧다")
            if len(text) > MAX_TEXT_CHARS:
                report.warn(f"{where}: text가 {len(text)}자로 길다. k개를 묶으면 컨텍스트 예산을 넘기기 쉽다")
            key = normalize(text)
            if key in seen_text and item_id and seen_text[key] != item_id:
                report.warn(f"{where}: text가 '{seen_text[key]}'와 완전히 같다(중복 문서)")
            else:
                seen_text.setdefault(key, item_id)
        if ok:
            items.append(
                KnowledgeItem(
                    id=item_id,
                    domain=row["domain"],
                    title=row["title"],
                    text=text,
                    source_url=url,
                    license=row["license"],
                    reliability=reliability,
                    lang=lang,
                )
            )
    if not rows:
        report.error(f"{path.name}: 데이터 행이 없다")
    return items, report


def load_knowledge(path: str | Path) -> list[KnowledgeItem]:
    items, report = read_knowledge(path)
    if not report.ok:
        raise DataError(report, str(path))
    return items


def read_questions(
    path: str | Path, items: list[KnowledgeItem] | None = None
) -> tuple[list[EvalQuestion], ValidationReport]:
    path = Path(path)
    report = ValidationReport()
    rows, columns = _read_rows(path, report)
    missing = [c for c in QUESTION_REQUIRED if c not in columns]
    if report.errors and not rows:
        return [], report
    if missing:
        report.error(f"{path.name}: 필수 열이 없다: {', '.join(missing)}")
        return [], report
    if "reference_answer" not in columns and "rubric" not in columns:
        report.error(f"{path.name}: reference_answer 또는 rubric 열이 하나는 있어야 한다")
        return [], report

    known_ids = {i.id: i for i in items} if items is not None else None
    questions: list[EvalQuestion] = []
    seen: dict[str, int] = {}
    for idx, row in enumerate(rows):
        where = _lineno(idx, path)
        ok = True
        for col in ("question_id", "question", "domain"):
            if not row.get(col):
                report.error(f"{where}: '{col}'이 비어 있다")
                ok = False
        qid = row.get("question_id", "")
        if qid:
            if not ID_RE.match(qid):
                report.error(f"{where}: question_id '{qid}'에는 영문·숫자·_ . - 만 쓸 수 있다")
                ok = False
            if qid in seen:
                report.error(f"{where}: question_id '{qid}'가 {seen[qid]}번째 행과 중복이다")
                ok = False
            else:
                seen[qid] = idx + 1
        split = row.get("split", "")
        if split not in SPLITS:
            report.error(f"{where}: split은 {SPLITS} 중 하나여야 한다: {split!r}")
            ok = False
        difficulty = row.get("difficulty", "")
        if difficulty not in DIFFICULTIES:
            report.error(f"{where}: difficulty는 {DIFFICULTIES} 중 하나여야 한다: {difficulty!r}")
            ok = False
        expected = [x.strip() for x in row.get("expected_source_ids", "").split(LIST_SEP) if x.strip()]
        if not expected:
            report.error(f"{where}: expected_source_ids가 비어 있다. 근거 문서 id를 '{LIST_SEP}'로 이어 적는다")
            ok = False
        elif known_ids is not None:
            for sid in expected:
                if sid not in known_ids:
                    report.error(f"{where}: expected_source_ids의 '{sid}'가 knowledge에 없다")
                    ok = False
                elif row.get("domain") and known_ids[sid].domain != row["domain"]:
                    report.warn(
                        f"{where}: 질문 분야 '{row['domain']}'와 근거 '{sid}'의 분야 '{known_ids[sid].domain}'가 다르다"
                    )
        reference = row.get("reference_answer", "")
        rubric = row.get("rubric", "")
        if not reference and not rubric:
            report.error(f"{where}: reference_answer와 rubric이 둘 다 비어 있다")
            ok = False
        question_text = normalize(row.get("question", ""))
        if rubric and question_text:
            for raw_item in rubric.split(LIST_SEP):
                for alt in raw_item.split("/"):
                    alt = alt.strip()
                    if len(alt) >= 2 and normalize(alt) in question_text:
                        report.warn(f"{where}: rubric 항목 '{alt}'이 질문 문장에 그대로 들어 있다(정답 누출 가능)")
        if ok:
            questions.append(
                EvalQuestion(
                    question_id=qid,
                    question=row["question"],
                    expected_source_ids=expected,
                    domain=row["domain"],
                    difficulty=difficulty,
                    split=split,
                    reference_answer=reference,
                    rubric=rubric,
                )
            )
    if not rows:
        report.error(f"{path.name}: 데이터 행이 없다")
        return questions, report

    for a, b, sim in near_duplicate_pairs(questions):
        report.warn(f"{a}와 {b}가 서로 다른 분할에 있는데 거의 같은 질문이다(유사도 {sim:.2f})")
    counts = {s: sum(1 for q in questions if q.split == s) for s in SPLITS}
    if counts["test"] == 0:
        report.warn("test 분할이 비어 있다. 최종 평가를 할 수 없다")
    for s, n in counts.items():
        if 0 < n < MIN_SPLIT_SIZE:
            report.warn(f"{s} 분할이 {n}개뿐이다. 이 크기로는 작은 차이를 구분하기 어렵다")
    return questions, report


def load_questions(path: str | Path, items: list[KnowledgeItem] | None = None) -> list[EvalQuestion]:
    questions, report = read_questions(path, items)
    if not report.ok:
        raise DataError(report, str(path))
    return questions


def write_questions(path: str | Path, questions: list[EvalQuestion]) -> None:
    columns = list(QUESTION_REQUIRED) + list(QUESTION_OPTIONAL)
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for q in questions:
            row = q.to_dict()
            row["expected_source_ids"] = LIST_SEP.join(q.expected_source_ids)
            writer.writerow({c: row.get(c, "") for c in columns})


def write_knowledge(path: str | Path, items: list[KnowledgeItem], append: bool = False) -> None:
    """지식 문서를 계약 형식 CSV로 쓴다.

    append=True이고 파일에 내용이 있으면 기존 헤더의 열 순서를 그대로 따라 뒤에 붙인다.
    기존 파일에 필수 열이 빠져 있으면 열이 어긋나지 않게 ValueError를 낸다.
    """
    target = Path(path)
    columns = list(KNOWLEDGE_REQUIRED) + list(KNOWLEDGE_OPTIONAL)
    has_content = append and target.exists() and target.stat().st_size > 0
    if has_content:
        with target.open(newline="", encoding="utf-8-sig") as fh:
            header = [h.strip() for h in next(csv.reader(fh), [])]
        missing = [c for c in KNOWLEDGE_REQUIRED if c not in header]
        if missing:
            raise ValueError(f"{target.name}에 필수 열이 없어서 이어 쓸 수 없다: {', '.join(missing)}")
        columns = header
        if not target.read_bytes().endswith(b"\n"):
            with target.open("a", encoding="utf-8") as fh:
                fh.write("\n")
    with target.open("a" if has_content else "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        if not has_content:
            writer.writeheader()
        for item in items:
            row = item.to_dict()
            writer.writerow({c: row.get(c, "") for c in columns})
