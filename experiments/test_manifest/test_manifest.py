"""make_manifest 시험. 합성 데이터만 쓴다(실제 평가 데이터 사용 금지).

저장소 기본 testpaths는 tests/라서 이 파일은 `python -m pytest -q experiments/test_manifest`로 돌린다.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_manifest as mm  # noqa: E402

FIELDS = ["question_id", "question", "expected_source_ids", "domain", "difficulty", "split", "reference_answer", "rubric"]


def synth(n_test: int = 3) -> list[dict[str, str]]:
    rows = []
    for i in range(n_test):
        rows.append({"question_id": f"s{i:02d}", "question": f"합성 질문 번호 {i}번에 대한 가짜 문장입니다 zzz{i}",
                     "expected_source_ids": f"doc_b{i}|doc_a{i}", "domain": "fake", "difficulty": "easy", "split": "test",
                     "reference_answer": f"합성 참조답 {i} 가짜 정답 문장 qqq{i}", "rubric": f"가짜{i}/동의어|키워드{i}"})
    rows.append({"question_id": "d00", "question": "개발용 합성 질문 문장 abcdefg", "expected_source_ids": "doc_x", "domain": "fake",
                 "difficulty": "easy", "split": "dev", "reference_answer": "개발용 합성 참조답 문장 hijklmn", "rubric": "a|b"})
    return rows


def write(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return path


def test_manifest_has_only_test_rows_hashes_and_counts(tmp_path):
    q = write(tmp_path / "q.csv", synth())
    out = tmp_path / "m.csv"
    assert mm.main(["make", "--questions", str(q), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "d00" not in text and "합성" not in text and "가짜" not in text and "doc_" not in text
    assert mm.check_manifest_clean(out) == []
    assert len(mm.read_manifest(out)) == 3


def test_cli_never_prints_question_content(tmp_path, capsys):
    q = write(tmp_path / "q.csv", synth())
    out = tmp_path / "m.csv"
    mm.main(["make", "--questions", str(q), "--out", str(out)])
    mm.main(["verify", "--questions", str(q), "--manifest", str(out)])
    mm.main(["scan", "--questions", str(q), "--repo", str(tmp_path)])
    shown = capsys.readouterr().out
    for bad in ("합성", "가짜", "zzz", "qqq", "doc_", "키워드"):
        assert bad not in shown


def test_hash_ignores_whitespace_and_id_order_but_not_content():
    base = synth(1)[0]
    same = dict(base, question="  " + base["question"].replace(" ", "  ") + " ", expected_source_ids="doc_a0|doc_b0")
    assert mm.question_digest(base) == mm.question_digest(same)
    for col in ("question", "reference_answer", "rubric", "expected_source_ids"):
        assert mm.question_digest(dict(base, **{col: base[col] + "x"})) != mm.question_digest(base)


def test_verify_reports_match_mismatch_missing_extra(tmp_path):
    rows = synth(3)
    man = {m["question_id"]: {k: str(v) for k, v in m.items()} for m in mm.build_manifest(rows)}
    changed = [dict(r) for r in rows]
    changed[1]["reference_answer"] += " 수정"          # 불일치
    changed = [r for r in changed if r["question_id"] != "s02"]  # 원본에서 사라짐
    changed.append(dict(rows[0], question_id="s09"))   # 새 test 질문
    res = mm.verify(changed, man)
    assert res == {"matched": ["s00"], "mismatched": ["s01"], "missing": ["s02"], "extra": ["s09"]}
    q = write(tmp_path / "q.csv", rows)
    mf = tmp_path / "m.csv"
    mm.write_manifest(mf, mm.build_manifest(rows))
    assert mm.main(["verify", "--questions", str(q), "--manifest", str(mf)]) == 0


def test_check_manifest_clean_flags_leaked_text(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("question_id,sha256,question_chars,reference_chars,rubric_chars\ns00,합성 질문 원문,3,4,5\n", encoding="utf-8")
    assert mm.check_manifest_clean(bad)
    assert mm.check_manifest_clean(_write_text(tmp_path / "bad2.csv", "question_id,question\ns00,x\n"))


def _write_text(p: Path, s: str) -> Path:
    p.write_text(s, encoding="utf-8")
    return p


def test_repo_scan_finds_test_text_but_not_dev_or_manifest(tmp_path):
    rows = synth()
    repo = tmp_path / "repo"
    repo.mkdir()
    mm.write_manifest(repo / "manifest.csv", mm.build_manifest(rows))
    (repo / "dev_note.md").write_text(rows[-1]["question"], encoding="utf-8")   # dev 내용은 괜찮다
    assert mm.scan_repo_for_test_content(repo, rows) == []
    (repo / "leak.md").write_text("메모: " + rows[1]["reference_answer"], encoding="utf-8")
    assert mm.scan_repo_for_test_content(repo, rows) == [("leak.md", "s01")]


def test_this_repository_has_no_manifest_leak_of_synthetic_marker():
    """저장소 자체에 합성 시험 문장이 새지 않았는지(스캐너가 실제 저장소 트리를 읽을 수 있는지) 확인한다."""
    root = Path(__file__).resolve().parents[2]
    rows = synth()
    hits = [h for h in mm.scan_repo_for_test_content(root, rows) if not h[0].startswith("experiments/test_manifest/")]
    assert hits == []
