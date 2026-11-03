"""test 분할 질문의 지문(manifest) 만들기와 대조. 표준 라이브러리만 쓴다.

목적: test 질문·정답 내용은 저장소에 올리지 않고(비공개 Drive에만 둔다), 나중에 "그 질문이 그대로인가"만 확인할 수 있게
질문마다 SHA-256 해시와 글자 수만 저장소에 남긴다. 이 도구는 질문·정답 내용을 화면에도 파일에도 출력하지 않는다.

  python experiments/test_manifest/make_manifest.py make   --questions 비공개/evaluation.csv --out experiments/test_manifest/test_manifest.csv
  python experiments/test_manifest/make_manifest.py verify --questions 비공개/evaluation.csv --manifest experiments/test_manifest/test_manifest.csv
  python experiments/test_manifest/make_manifest.py scan   --questions 비공개/evaluation.csv --repo .      # 저장소에 test 내용이 새지 않았는지

해시 입력: 질문, 참조답, 루브릭, 근거ID(정렬해서 `|`로 이음) 네 값을 NFC 정규화 + 공백 정리한 뒤 U+001F(단위 구분자)로 잇는다.
루브릭과 근거ID 안에 `|`가 들어 있어서 `|`로 이으면 경계가 모호해지기 때문이다.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
import unicodedata
from pathlib import Path

MANIFEST_COLUMNS = ["question_id", "sha256", "question_chars", "reference_chars", "rubric_chars"]
SEP = "\x1f"
TEXT_SUFFIXES = {".md", ".txt", ".csv", ".json", ".jsonl", ".py", ".ipynb", ".yml", ".yaml", ".toml", ".cfg"}
MIN_SCAN_CHARS = 12  # 이보다 짧은 값은 우연히 겹치기 쉬워서 내용 검색에서 뺀다


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text or "")).strip()


def normalize_ids(raw: str) -> str:
    return "|".join(sorted(p for p in (normalize(x) for x in (raw or "").split("|")) if p))


def question_digest(row: dict[str, str]) -> str:
    parts = [normalize(row.get("question", "")), normalize(row.get("reference_answer", "")),
             normalize(row.get("rubric", "")), normalize_ids(row.get("expected_source_ids", ""))]
    return hashlib.sha256(SEP.join(parts).encode("utf-8")).hexdigest()


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def test_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [r for r in rows if normalize(r.get("split", "")) == "test"]


def build_manifest(rows: list[dict[str, str]]) -> list[dict[str, str | int]]:
    out = []
    for r in sorted(test_rows(rows), key=lambda r: r["question_id"]):
        out.append({
            "question_id": normalize(r["question_id"]),
            "sha256": question_digest(r),
            "question_chars": len(normalize(r.get("question", ""))),
            "reference_chars": len(normalize(r.get("reference_answer", ""))),
            "rubric_chars": len(normalize(r.get("rubric", ""))),
        })
    return out


def write_manifest(path: str | Path, manifest: list[dict[str, str | int]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(manifest)


def read_manifest(path: str | Path) -> dict[str, dict[str, str]]:
    return {r["question_id"]: r for r in read_rows(path)}


def verify(rows: list[dict[str, str]], manifest: dict[str, dict[str, str]]) -> dict[str, list[str]]:
    """id만 돌려준다. matched: 해시 일치, mismatched: 내용이 바뀜, missing: manifest에만 있음, extra: 원본의 test에만 있음."""
    current = {m["question_id"]: m for m in build_manifest(rows)}
    res: dict[str, list[str]] = {"matched": [], "mismatched": [], "missing": [], "extra": []}
    for qid, m in manifest.items():
        if qid not in current:
            res["missing"].append(qid)
        elif current[qid]["sha256"] == m["sha256"]:
            res["matched"].append(qid)
        else:
            res["mismatched"].append(qid)
    res["extra"] = [q for q in current if q not in manifest]
    return {k: sorted(v) for k, v in res.items()}


def check_manifest_clean(path: str | Path) -> list[str]:
    """manifest 파일에 해시·숫자 말고 다른 것이 들어 있으면 문제 목록을 돌려준다(내용은 인용하지 않는다)."""
    problems: list[str] = []
    rows = read_rows(path)
    with Path(path).open(encoding="utf-8-sig", newline="") as fh:
        header = next(csv.reader(fh), [])
    if header != MANIFEST_COLUMNS:
        problems.append("열 이름이 MANIFEST_COLUMNS와 다르다")
    for i, r in enumerate(rows, start=2):
        if not re.fullmatch(r"[0-9a-f]{64}", r.get("sha256") or ""):
            problems.append(f"{i}행: sha256 형식이 아니다")
        for c in ("question_chars", "reference_chars", "rubric_chars"):
            if not re.fullmatch(r"\d+", r.get(c) or ""):
                problems.append(f"{i}행: {c}가 숫자가 아니다")
        if not re.fullmatch(r"[A-Za-z0-9_.\-]+", r.get("question_id") or ""):
            problems.append(f"{i}행: question_id 형식이 아니다")
    return problems


def scan_repo_for_test_content(repo: str | Path, rows: list[dict[str, str]], skip_dirs: tuple[str, ...] = (".git", "__pycache__", ".pytest_cache")) -> list[tuple[str, str]]:
    """저장소의 텍스트 파일에 test 질문·참조답 문장이 그대로 들어 있는지 찾는다. (파일 경로, question_id)만 돌려준다."""
    needles: list[tuple[str, str]] = []
    for r in test_rows(rows):
        for col in ("question", "reference_answer"):
            v = normalize(r.get(col, ""))
            if len(v) >= MIN_SCAN_CHARS:
                needles.append((r["question_id"], v))
    hits: list[tuple[str, str]] = []
    root = Path(repo)
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in TEXT_SUFFIXES or any(d in p.parts for d in skip_dirs):
            continue
        try:
            text = normalize(p.read_text(encoding="utf-8-sig"))
        except (UnicodeDecodeError, OSError):
            continue
        for qid, v in needles:
            if v in text:
                hits.append((str(p.relative_to(root)), qid))
    return sorted(set(hits))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--questions", required=True)
    m.add_argument("--out", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--questions", required=True, help="비공개 Drive의 원본 CSV")
    v.add_argument("--manifest", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--questions", required=True)
    s.add_argument("--repo", default=".")
    a = ap.parse_args(argv)

    rows = read_rows(a.questions)
    if a.cmd == "make":
        man = build_manifest(rows)
        if not man:
            print("test 분할 질문이 없어서 manifest를 만들지 않았다")
            return 1
        write_manifest(a.out, man)
        print(f"test 질문 {len(man)}개의 지문을 {a.out}에 썼다(해시와 글자 수만 들어 있다)")
        return 0
    if a.cmd == "verify":
        res = verify(rows, read_manifest(a.manifest))
        for k, label in (("matched", "일치"), ("mismatched", "불일치(내용 바뀜)"), ("missing", "원본에 없음"), ("extra", "manifest에 없음")):
            print(f"{label} {len(res[k])}개: {', '.join(res[k]) or '-'}")
        return 0 if not (res["mismatched"] or res["missing"] or res["extra"]) else 1
    hits = scan_repo_for_test_content(a.repo, rows)
    for path, qid in hits:
        print(f"유출 의심: {path} ({qid})")
    print(f"검사 끝: 의심 {len(hits)}건")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
