import pytest

from kore.io import read_questions
from kore.knowledge_graph import RelationIndex, read_relations
from kore.memory import ExperienceLog
from kore.schemas import EvalQuestion
from kore.splits import assign_splits, near_duplicate_pairs
from kore.tuning import grid_search_weights
from tests.helpers import Q_HEADER, q_row, write_csv


def make_questions(n_per_domain=5, domains=("a", "b")):
    return [
        EvalQuestion(f"{d}{i}", f"question {d} {i}", ["x"], d, "easy", "train", rubric="r")
        for d in domains
        for i in range(n_per_domain)
    ]


# ---------------------------------------------------------------- 튜닝은 test를 건드리지 못한다


def test_grid_search_rejects_test_split(items, questions):
    with pytest.raises(ValueError, match="test"):
        grid_search_weights(items, questions, splits=("train", "test"))
    with pytest.raises(ValueError, match="test"):
        grid_search_weights(items, questions, splits=("test",))


def test_grid_search_runs_on_train_dev_and_warns_on_small_samples(items, questions):
    res = grid_search_weights(items, questions)
    assert res["n_questions"] == 7 and set(res["splits"]) == {"train", "dev"}
    assert 0.0 <= res["best"]["recall"] <= 1.0 and 0.0 <= res["baseline_similarity_recall"] <= 1.0
    assert res["table"] and "과적합" in res["warning"]
    assert res["best"]["recall"] == max(row["recall"] for row in res["table"])


def test_grid_search_ignores_test_questions_even_if_present(items, questions):
    only_train_dev = [q for q in questions if q.split != "test"]
    a = grid_search_weights(items, questions)
    b = grid_search_weights(items, only_train_dev)
    assert a["table"] == b["table"]


def test_grid_search_validates_inputs(items, questions):
    with pytest.raises(ValueError):
        grid_search_weights(items, questions, grid={"sim": [1.0]})
    with pytest.raises(ValueError):
        grid_search_weights(items, [q for q in questions if q.split == "test"])


# ---------------------------------------------------------------- 기억 오염 방지


def test_experience_log_rejects_test_split_and_records_gain(tmp_path):
    log = ExperienceLog(tmp_path / "exp.jsonl")
    with pytest.raises(ValueError, match="기억 오염"):
        log.append("q1", "test", ["a"], ["a"])
    log.append("q1", "train", ["a", "b"], ["a"], quality=0.9, baseline_quality=0.4)
    log.append("q2", "dev", ["a"], ["a"])
    recs = list(log.records())
    assert [r["question_id"] for r in recs] == ["q1", "q2"]
    assert recs[0]["gain"] == pytest.approx(0.5) and recs[1]["gain"] is None
    assert list(ExperienceLog(tmp_path / "none.jsonl").records()) == []


# ---------------------------------------------------------------- 분할


def test_assign_splits_is_deterministic_and_stratified():
    qs = make_questions()
    a = assign_splits(qs, seed=0)
    b = assign_splits(qs, seed=0)
    assert [q.split for q in a] == [q.split for q in b]
    for d in ("a", "b"):
        assert {q.split for q in a if q.domain == d} == {"train", "dev", "test"}
    assert [q.question_id for q in a] == [q.question_id for q in qs]  # 입력 순서 유지
    assert all(q.split == "train" for q in qs)  # 원본은 바뀌지 않는다


def test_assign_splits_depends_on_seed_and_not_on_input_order():
    qs = make_questions(n_per_domain=10)
    by0 = {q.question_id: q.split for q in assign_splits(qs, seed=0)}
    assert by0 == {q.question_id: q.split for q in assign_splits(list(reversed(qs)), seed=0)}
    assert by0 != {q.question_id: q.split for q in assign_splits(qs, seed=1)}


def test_assign_splits_ratios_and_tiny_domains():
    qs = make_questions(n_per_domain=10, domains=("a",))
    out = assign_splits(qs, seed=0)
    counts = {s: sum(q.split == s for q in out) for s in ("train", "dev", "test")}
    assert counts == {"train": 6, "dev": 2, "test": 2}
    tiny = assign_splits(make_questions(n_per_domain=2, domains=("z",)))
    assert all(q.split == "train" for q in tiny)
    with pytest.raises(ValueError):
        assign_splits(qs, ratios=(0.5, 0.2, 0.2))


def test_near_duplicates_only_flagged_across_splits():
    base = dict(expected_source_ids=["x"], domain="d", rubric="r")
    qs = [
        EvalQuestion("q1", "what is the tallest volcano", split="train", **base),
        EvalQuestion("q2", "what is the tallest volcano", split="train", **base),
        EvalQuestion("q3", "what is the tallest volcano", split="test", **base),
        EvalQuestion("q4", "completely different words here", split="test", **base),
    ]
    pairs = near_duplicate_pairs(qs)
    assert {(a, b) for a, b, _ in pairs} == {("q1", "q3"), ("q2", "q3")}


# ---------------------------------------------------------------- 관계 파일


def _rel_file(tmp_path, rows):
    return write_csv(tmp_path / "rel.csv", ["src_id", "dst_id", "relation", "evidence_url"], rows)


def test_relations_ok_and_neighbors(tmp_path, items):
    path = _rel_file(tmp_path, [["ast_001", "ast_005", "similar_to", "https://e.org"]])
    rels, report = read_relations(path, items)
    assert report.ok and len(rels) == 1
    idx = RelationIndex(rels)
    assert [r.dst_id for r in idx.neighbors("ast_001")] == ["ast_005"]
    assert idx.neighbors("zzz") == []


def test_relations_errors_and_warnings(tmp_path, items):
    path = _rel_file(
        tmp_path,
        [
            ["ast_001", "ast_001", "self", "https://e.org"],
            ["ast_001", "nope", "x", "https://e.org"],
            ["ast_001", "ast_002", "a", ""],
            ["ast_001", "ast_002", "a", ""],
            ["", "ast_002", "a", ""],
        ],
    )
    rels, report = read_relations(path, items)
    text = " ".join(report.errors)
    assert "자기 자신" in text and "nope" in text and "'src_id'" in text
    assert any("중복" in w for w in report.warnings) and any("evidence_url" in w for w in report.warnings)
    assert len(rels) == 2


def test_cli_validate_relations(tmp_path, capsys):
    from kore.cli import main

    good = _rel_file(tmp_path, [["ast_001", "ast_005", "similar_to", "https://e.org"]])
    assert main(["validate", "--knowledge", "data/sample_knowledge.csv", "--relations", str(good)]) == 0
    assert "[relations] 통과" in capsys.readouterr().out
    bad = tmp_path / "bad.csv"
    write_csv(bad, ["src_id", "dst_id", "relation", "evidence_url"], [["ast_001", "nope", "x", "https://e.org"]])
    assert main(["validate", "--knowledge", "data/sample_knowledge.csv", "--relations", str(bad)]) == 1


def test_relations_missing_columns(tmp_path):
    path = write_csv(tmp_path / "rel.csv", ["src_id", "dst_id"], [["a", "b"]])
    assert not read_relations(path)[1].ok


def test_jsonl_questions_accept_list_ids(tmp_path):
    import json

    path = tmp_path / "q.jsonl"
    row = dict(zip(Q_HEADER, q_row()))
    row["expected_source_ids"] = ["a1", "a2"]
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    qs, report = read_questions(path)
    assert report.ok and qs[0].expected_source_ids == ["a1", "a2"]
