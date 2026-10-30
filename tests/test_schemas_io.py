import json

import pytest

from kore.io import (
    DataError, load_knowledge, load_questions, read_knowledge, read_questions, write_questions,
)
from kore.schemas import EvalQuestion
from tests.helpers import K_HEADER, Q_HEADER, k_row, q_row, write_csv


def test_sample_files_load_clean(items, questions):
    assert len(items) == 15
    assert len(questions) == 9
    assert {i.domain for i in items} == {"astronomy", "chemistry", "computing"}
    assert {q.split for q in questions} == {"train", "dev", "test"}
    assert all(0.0 <= i.reliability <= 1.0 for i in items)


def test_rubric_items_parse_alternatives():
    q = EvalQuestion("q", "?", ["a"], "d", rubric="alpha/beta|gamma| ")
    assert q.rubric_items() == [["alpha", "beta"], ["gamma"]]
    assert EvalQuestion("q", "?", ["a"], "d").rubric_items() == []


# ---------------------------------------------------------------- knowledge


def test_knowledge_missing_required_column(tmp_path):
    header = [c for c in K_HEADER if c != "license"]
    rows = [[v for c, v in zip(K_HEADER, k_row()) if c != "license"]]
    items, report = read_knowledge(write_csv(tmp_path / "k.csv", header, rows))
    assert not report.ok and "license" in report.errors[0]
    assert items == []


def test_knowledge_duplicate_id_and_bad_id(tmp_path):
    path = write_csv(tmp_path / "k.csv", K_HEADER, [k_row("a1"), k_row("a1"), k_row("bad id")])
    _, report = read_knowledge(path)
    assert any("중복" in e for e in report.errors)
    assert any("bad id" in e for e in report.errors)


def test_knowledge_empty_required_field(tmp_path):
    _, report = read_knowledge(write_csv(tmp_path / "k.csv", K_HEADER, [k_row(text="")]))
    assert any("'text'" in e for e in report.errors)


def test_knowledge_bad_reliability(tmp_path):
    _, report = read_knowledge(write_csv(tmp_path / "k.csv", K_HEADER, [k_row(reliability="1.5")]))
    assert any("reliability" in e for e in report.errors)
    _, report = read_knowledge(write_csv(tmp_path / "k2.csv", K_HEADER, [k_row(reliability="abc")]))
    assert any("reliability" in e for e in report.errors)


def test_knowledge_blank_reliability_defaults(tmp_path):
    items, report = read_knowledge(write_csv(tmp_path / "k.csv", K_HEADER, [k_row(reliability="")]))
    assert report.ok and items[0].reliability == 0.5


def test_knowledge_warnings_do_not_fail(tmp_path):
    rows = [
        k_row("a1", license="unknown"),
        k_row("a2", text="too short"),
        k_row("a3", source_url="not-a-url"),
        k_row("a4", lang="fr"),
    ]
    items, report = read_knowledge(write_csv(tmp_path / "k.csv", K_HEADER, rows))
    assert report.ok and len(items) == 4
    joined = " ".join(report.warnings)
    for needle in ("license", "짧다", "source_url", "lang"):
        assert needle in joined


def test_knowledge_duplicate_text_warns(tmp_path):
    rows = [k_row("a1"), k_row("a2")]
    _, report = read_knowledge(write_csv(tmp_path / "k.csv", K_HEADER, rows))
    assert report.ok and any("완전히 같다" in w for w in report.warnings)


def test_knowledge_reads_excel_bom_csv(tmp_path):
    path = write_csv(tmp_path / "k.csv", K_HEADER, [k_row()], encoding="utf-8-sig")
    assert len(load_knowledge(path)) == 1


def test_knowledge_extra_columns_are_ignored_with_warning(tmp_path):
    path = write_csv(tmp_path / "k.csv", K_HEADER + ["memo"], [k_row() + ["hello"]])
    items, report = read_knowledge(path)
    assert report.ok and len(items) == 1
    assert any("memo" in w for w in report.warnings)


def test_knowledge_jsonl_accepts_lists_and_reports_syntax_errors(tmp_path):
    path = tmp_path / "k.jsonl"
    good = dict(zip(K_HEADER, k_row()))
    path.write_text(json.dumps(good) + "\n{broken\n", encoding="utf-8")
    items, report = read_knowledge(path)
    assert len(items) == 1
    assert any("JSON" in e for e in report.errors)


def test_load_knowledge_raises_with_report(tmp_path):
    path = write_csv(tmp_path / "k.csv", K_HEADER, [k_row(text="")])
    with pytest.raises(DataError) as exc:
        load_knowledge(path)
    assert exc.value.report.errors


def test_unsupported_extension_and_missing_file(tmp_path):
    (tmp_path / "k.txt").write_text("x", encoding="utf-8")
    assert not read_knowledge(tmp_path / "k.txt")[1].ok
    assert not read_knowledge(tmp_path / "nope.csv")[1].ok


def test_empty_file_is_an_error(tmp_path):
    _, report = read_knowledge(write_csv(tmp_path / "k.csv", K_HEADER, []))
    assert any("행이 없다" in e for e in report.errors)


# ---------------------------------------------------------------- questions


def _corpus(tmp_path):
    return load_knowledge(write_csv(tmp_path / "k.csv", K_HEADER, [k_row("a1"), k_row("a2", domain="chemistry")]))


def test_questions_ok(tmp_path):
    items = _corpus(tmp_path)
    qs, report = read_questions(write_csv(tmp_path / "q.csv", Q_HEADER, [q_row()]), items)
    assert report.ok and qs[0].expected_source_ids == ["a1"]


def test_questions_unknown_source_id(tmp_path):
    items = _corpus(tmp_path)
    _, report = read_questions(write_csv(tmp_path / "q.csv", Q_HEADER, [q_row(expected_source_ids="zzz")]), items)
    assert any("zzz" in e for e in report.errors)


def test_questions_bad_split_difficulty_and_duplicates(tmp_path):
    rows = [q_row("q1", split="valid"), q_row("q2", difficulty="hardest"), q_row("q3"), q_row("q3")]
    _, report = read_questions(write_csv(tmp_path / "q.csv", Q_HEADER, rows))
    text = " ".join(report.errors)
    assert "split" in text and "difficulty" in text and "중복" in text


def test_questions_need_reference_or_rubric(tmp_path):
    _, report = read_questions(write_csv(tmp_path / "q.csv", Q_HEADER, [q_row(reference_answer="", rubric="")]))
    assert any("둘 다 비어" in e for e in report.errors)


def test_questions_without_answer_columns(tmp_path):
    header = [c for c in Q_HEADER if c not in ("reference_answer", "rubric")]
    rows = [[v for c, v in zip(Q_HEADER, q_row()) if c in header]]
    _, report = read_questions(write_csv(tmp_path / "q.csv", header, rows))
    assert any("reference_answer 또는 rubric" in e for e in report.errors)


def test_questions_empty_expected_ids(tmp_path):
    _, report = read_questions(write_csv(tmp_path / "q.csv", Q_HEADER, [q_row(expected_source_ids="")]))
    assert any("expected_source_ids" in e for e in report.errors)


def test_questions_domain_mismatch_warns(tmp_path):
    items = _corpus(tmp_path)
    rows = [q_row(expected_source_ids="a2")]  # a2는 chemistry, 질문은 astronomy
    _, report = read_questions(write_csv(tmp_path / "q.csv", Q_HEADER, rows), items)
    assert report.ok and any("분야" in w for w in report.warnings)


def test_questions_answer_leak_warns(tmp_path):
    rows = [q_row(question="Is the answer Jupiter or something else?", rubric="Jupiter")]
    _, report = read_questions(write_csv(tmp_path / "q.csv", Q_HEADER, rows))
    assert report.ok and any("누출" in w for w in report.warnings)


def test_questions_cross_split_near_duplicates_warn(tmp_path):
    rows = [
        q_row("q1", question="What is the tallest volcano in the Solar System?", split="train"),
        q_row("q2", question="What is the tallest volcano in the Solar System", split="test"),
    ]
    _, report = read_questions(write_csv(tmp_path / "q.csv", Q_HEADER, rows))
    assert any("거의 같은 질문" in w for w in report.warnings)


def test_small_or_missing_split_warns(tmp_path):
    rows = [q_row("q1", split="train")]
    _, report = read_questions(write_csv(tmp_path / "q.csv", Q_HEADER, rows))
    text = " ".join(report.warnings)
    assert "test 분할이 비어" in text and "train 분할이 1개뿐" in text


def test_load_questions_raises(tmp_path):
    with pytest.raises(DataError):
        load_questions(write_csv(tmp_path / "q.csv", Q_HEADER, [q_row(split="x")]))


def test_write_questions_roundtrip(tmp_path, questions):
    path = tmp_path / "out.csv"
    write_questions(path, questions)
    again = load_questions(path)
    assert [q.to_dict() for q in again] == [q.to_dict() for q in questions]
