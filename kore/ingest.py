"""원문 텍스트를 계약 형식의 지식 문서로 잘라 주는 도구 (담당: B).

B가 하는 일은 원문을 직접 열어서 텍스트를 복사해 txt 파일에 붙여 넣는 것까지다.
그다음 자르기, id 붙이기, CSV 쓰기, 출처 기록 줄 만들기는 이 도구가 한다.
도구는 내용의 사실 여부와 라이선스를 판단하지 못한다. 출처 기록의 license_checked는 항상 'no'로 시작하고,
B가 원문 페이지에서 이용 조건을 직접 읽은 뒤 'yes'로 바꾼다.

자르는 방식(mode)
- article  : 법령. '제N조(제목)' 줄에서 한 문서로 자른다. 조문 단위가 질문과 근거를 잇기에 가장 알맞다.
- heading  : 문서. 마크다운 '#' 제목이나 위키 '== 제목 ==' 줄에서 자른다.
- paragraph: 제목이 없는 글. 빈 줄로 나누고 짧은 문단은 다음 문단에 붙인다.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from kore.schemas import KnowledgeItem

MODES = ("article", "heading", "paragraph")
SOURCE_LOG_COLUMNS = ["id", "source_url", "accessed_date", "license_checked", "verified_by", "note"]

_ARTICLE = re.compile(r"^\s*제\s*(\d+)\s*조(?:\s*의\s*(\d+))?(.*)$")
_PARTICLE_START = re.compile(r"^(에|의|와|과|를|을|은|는|이|가|도|로|으로|부터|까지|제|및|또는|에서|에게|인|한|된|하)")
_CHAPTER = re.compile(r"^\s*제\s*\d+\s*[편장절관](?:의\s*\d+)?(?:\s|$)")
_REVISION_LINE = re.compile(r"^\s*\[(?:전문개정|개정|신설|본조신설|본조삭제|제목개정|종전|시행일)[^\]]*\]\s*$")
_REVISION_INLINE = re.compile(r"<\s*(?:(?:전문개정|개정|신설|삭제|본조신설|제목개정|종전)|\d{4}\s*\.)[^>]*>")
_MD_HEADING = re.compile(r"^\s*#{1,6}\s+(.+?)\s*$")
_WIKI_HEADING = re.compile(r"^\s*=+\s*(.+?)\s*=+\s*$")
_SENT_END = re.compile(r"(?<=[.!?。다])\s+")


@dataclass
class Chunk:
    title: str
    text: str


@dataclass
class IngestResult:
    items: list[KnowledgeItem] = field(default_factory=list)
    dropped: list[tuple[str, str]] = field(default_factory=list)  # (제목, 이유)
    notes: list[str] = field(default_factory=list)


def clean_law_text(text: str) -> str:
    """개정 표시(<개정 …>, [전문개정 …])와 장·절 제목 줄을 지우고 공백을 정리한다."""
    lines = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if _REVISION_LINE.match(raw) or _CHAPTER.match(raw):
            continue
        line = _REVISION_INLINE.sub("", raw)
        lines.append(re.sub(r"[ \t 　]+", " ", line).strip())
    return "\n".join(lines)


def _tidy_article_title(raw: str) -> str:
    """'제 3 조 의 2 (제목)'을 '제3조의2(제목)'으로 맞춘다. 괄호 안 제목의 띄어쓰기는 그대로 둔다."""
    head, sep, tail = raw.partition("(")
    head = re.sub(r"\s+", "", head)
    return head + (sep + tail if sep else "")


def _split_articles(text: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    last: tuple[int, int] = (0, 0)
    cur_title, cur_lines = "", []

    def flush() -> None:
        if cur_title:
            chunks.append(Chunk(cur_title, "\n".join(cur_lines).strip()))

    for line in text.split("\n"):
        m = _ARTICLE.match(line)
        if m:
            num, sub, rest = int(m.group(1)), int(m.group(2) or 0), m.group(3)
            rest_ok = (not rest) or rest[:1].isspace() or rest.lstrip().startswith("(")
            looks_like_header = rest_ok and not _PARTICLE_START.match(rest.lstrip())
            # 줄 첫머리에 '제5조…'가 나오는 본문 줄을 조 제목으로 오인하지 않도록,
            # 조사·'제'로 이어지는 줄은 제외하고 번호가 앞보다 커야 새 조로 본다
            if looks_like_header and (num, sub) > last:
                flush()
                last = (num, sub)
                title_match = re.match(r"^\s*(제\s*\d+\s*조(?:\s*의\s*\d+)?(?:\s*\([^)]*\))?)", line)
                cur_title = _tidy_article_title(title_match.group(1)) if title_match else f"제{num}조"
                cur_lines = [line.strip()]
                continue
        if cur_title:
            cur_lines.append(line)
    flush()
    return chunks


def _split_headings(text: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    title, buf = "", []
    for line in text.split("\n"):
        m = _MD_HEADING.match(line) or _WIKI_HEADING.match(line)
        if m:
            if "".join(buf).strip() or title:
                chunks.append(Chunk(title, "\n".join(buf).strip()))
            title, buf = m.group(1).strip(), []
        else:
            buf.append(line)
    if "".join(buf).strip() or title:
        chunks.append(Chunk(title, "\n".join(buf).strip()))
    return chunks


def _split_paragraphs(text: str, min_chars: int) -> list[Chunk]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[Chunk] = []
    carry = ""
    for p in paras:
        merged = f"{carry}\n{p}".strip() if carry else p
        if len(merged) < min_chars:
            carry = merged
        else:
            chunks.append(Chunk("", merged))
            carry = ""
    if carry:
        chunks.append(Chunk("", carry))
    return chunks


def _cut_long(chunk: Chunk, max_chars: int) -> list[Chunk]:
    """max_chars를 넘는 문서를 줄 경계에서 나눈다(한 줄이 너무 길면 문장 경계에서)."""
    if len(chunk.text) <= max_chars:
        return [chunk]
    pieces: list[str] = []
    for line in chunk.text.split("\n"):
        pieces.extend(_SENT_END.split(line) if len(line) > max_chars else [line])
    parts: list[str] = []
    buf = ""
    for piece in pieces:
        candidate = f"{buf}\n{piece}" if buf else piece
        if len(candidate) > max_chars and buf:
            parts.append(buf)
            buf = piece
        else:
            buf = candidate
    if buf:
        parts.append(buf)
    if len(parts) == 1:
        return [chunk]
    return [Chunk(f"{chunk.title}({n}/{len(parts)})" if chunk.title else "", part) for n, part in enumerate(parts, 1)]


def split_text(text: str, mode: str, min_chars: int = 30, max_chars: int = 1500, law_clean: bool | None = None) -> list[Chunk]:
    if mode not in MODES:
        raise ValueError(f"mode는 {MODES} 중 하나여야 한다: {mode!r}")
    if law_clean is None:
        law_clean = mode == "article"
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = clean_law_text(text) if law_clean else text
    if mode == "article":
        chunks = _split_articles(text)
    elif mode == "heading":
        chunks = _split_headings(text)
    else:
        chunks = _split_paragraphs(text, min_chars)
    out: list[Chunk] = []
    for c in chunks:
        out.extend(_cut_long(c, max_chars))
    return out


def _next_index(existing_ids: set[str], prefix: str) -> int:
    pat = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    nums = [int(m.group(1)) for i in existing_ids if (m := pat.match(i))]
    return (max(nums) + 1) if nums else 1


def ingest_text(
    text: str,
    domain: str,
    id_prefix: str,
    source_url: str,
    license: str,
    mode: str = "paragraph",
    title_prefix: str = "",
    reliability: float = 0.5,
    lang: str = "ko",
    min_chars: int = 30,
    max_chars: int = 1500,
    existing_ids: set[str] | None = None,
) -> IngestResult:
    """텍스트를 문서 목록으로 만든다. 너무 짧거나 '삭제'뿐인 조각은 버리고 이유를 남긴다."""
    if not re.match(r"^[A-Za-z0-9_.\-]+$", id_prefix):
        raise ValueError("id_prefix에는 영문·숫자·_ . - 만 쓸 수 있다")
    if not 0.0 <= reliability <= 1.0:
        raise ValueError("reliability는 0~1 사이여야 한다")
    result = IngestResult()
    chunks = split_text(text, mode, min_chars=min_chars, max_chars=max_chars)
    if not chunks:
        result.notes.append("자를 내용이 없다. mode가 원문 모양과 맞는지 확인한다(법령은 article).")
        return result
    start = _next_index(existing_ids or set(), id_prefix)
    n = start
    for c in chunks:
        body = c.text.strip()
        label = f"{title_prefix} {c.title}".strip() or f"{id_prefix}_{n:03d}"
        if body.replace(" ", "").endswith("삭제") and len(body) < 40:
            result.dropped.append((label, "삭제된 조문"))
            continue
        if len(body) < min_chars:
            result.dropped.append((label, f"{len(body)}자로 너무 짧다"))
            continue
        result.items.append(
            KnowledgeItem(
                id=f"{id_prefix}_{n:03d}",
                domain=domain,
                title=label,
                text=body,
                source_url=source_url,
                license=license,
                reliability=reliability,
                lang=lang,
            )
        )
        n += 1
    if mode == "article" and len(result.items) == 0:
        result.notes.append("조문을 하나도 찾지 못했다. '제1조(목적)'처럼 줄 앞에서 시작하는지 확인한다.")
    return result


def append_source_log(path: str | Path, items: list[KnowledgeItem], accessed_date: str | None = None, note: str = "") -> int:
    """출처 기록에 문서마다 한 줄씩 붙인다. license_checked는 'no', verified_by는 비워 둔다(사람이 채운다)."""
    p = Path(path)
    new_file = not p.exists() or p.stat().st_size == 0
    existing: set[str] = set()
    if not new_file:
        with p.open(newline="", encoding="utf-8-sig") as fh:
            existing = {r["id"].strip() for r in csv.DictReader(fh) if r.get("id")}
        if not p.read_bytes().endswith(b"\n"):
            with p.open("a", encoding="utf-8") as fh:
                fh.write("\n")
    day = accessed_date or date.today().isoformat()
    written = 0
    with p.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SOURCE_LOG_COLUMNS)
        if new_file:
            writer.writeheader()
        for it in items:
            if it.id in existing:
                continue
            writer.writerow(
                {"id": it.id, "source_url": it.source_url, "accessed_date": day, "license_checked": "no", "verified_by": "", "note": note}
            )
            written += 1
    return written
