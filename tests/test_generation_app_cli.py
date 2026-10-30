import csv

from app import build_retriever, demo_search
from kore.cli import main
from kore.generation import ExtractiveGenerator, build_prompt
from tests.helpers import K_HEADER, Q_HEADER, k_row, mk_item, q_row, write_csv

SAMPLE_K = "data/sample_knowledge.csv"
SAMPLE_Q = "data/sample_eval.csv"


# ---------------------------------------------------------------- 생성기


def test_build_prompt_with_and_without_evidence():
    ev = [mk_item("a1", "First fact."), mk_item("b2", "Second fact.")]
    with_ev = build_prompt("Why?", ev)
    assert "id=a1" in with_ev and "id=b2" in with_ev and "Why?" in with_ev
    assert with_ev.index("First fact.") < with_ev.index("Second fact.")
    without = build_prompt("Why?", [])
    assert "Why?" in without and "근거" not in without


def test_extractive_generator_returns_best_matching_sentence():
    ev = [mk_item("a", "Cats sleep a lot. Jupiter is the largest planet. Dogs bark.")]
    assert ExtractiveGenerator().generate("Which planet is largest?", ev) == "Jupiter is the largest planet."
    assert ExtractiveGenerator().generate("anything", []) == ""


# ---------------------------------------------------------------- 데모 함수


def test_demo_search_returns_table_rows():
    retriever = build_retriever()
    rows = demo_search(retriever, "Which planet is the largest in the Solar System?", k=2)
    assert len(rows) == 2 and rows[0][0] == "ast_001" and len(rows[0]) == 6
    assert demo_search(retriever, "   ") == []
    sim = demo_search(retriever, "pure water pH", k=1, strategy="similarity")
    assert sim[0][0] == "che_001"


# ---------------------------------------------------------------- CLI


def test_cli_validate_ok(capsys):
    assert main(["validate", "--knowledge", SAMPLE_K, "--questions", SAMPLE_Q]) == 0
    out = capsys.readouterr().out
    assert "[knowledge] 통과" in out and "[evaluation] 통과" in out


def test_cli_validate_fails_on_bad_data(tmp_path, capsys):
    bad = write_csv(tmp_path / "k.csv", K_HEADER, [k_row("a1"), k_row("a1")])
    assert main(["validate", "--knowledge", str(bad)]) == 1
    assert "중복" in capsys.readouterr().out


def test_cli_validate_fails_on_unknown_source_ids(tmp_path):
    k = write_csv(tmp_path / "k.csv", K_HEADER, [k_row("a1")])
    q = write_csv(tmp_path / "q.csv", Q_HEADER, [q_row(expected_source_ids="zzz")])
    assert main(["validate", "--knowledge", str(k), "--questions", str(q)]) == 1


def test_cli_run_writes_outputs_and_excludes_test_by_default(tmp_path, capsys):
    out = tmp_path / "run"
    assert main(["run", "--knowledge", SAMPLE_K, "--questions", SAMPLE_Q, "--out", str(out)]) == 0
    assert (out / "results.csv").exists() and (out / "env.json").exists()
    assert "21행" in capsys.readouterr().out


def test_cli_run_final_test_flag(tmp_path, capsys):
    out = tmp_path / "run"
    code = main(["run", "--knowledge", SAMPLE_K, "--questions", SAMPLE_Q, "--out", str(out), "--final-test"])
    assert code == 0
    out_text = capsys.readouterr().out
    assert "27행" in out_text and "test 분할을 포함" in out_text


def test_cli_run_refuses_invalid_data(tmp_path, capsys):
    k = write_csv(tmp_path / "k.csv", K_HEADER, [k_row("a1")])
    q = write_csv(tmp_path / "q.csv", Q_HEADER, [q_row(split="bogus")])
    code = main(["run", "--knowledge", str(k), "--questions", str(q), "--out", str(tmp_path / "o")])
    assert code == 1 and not (tmp_path / "o").exists()
    assert "시작하지 않는다" in capsys.readouterr().out


def test_cli_run_fresh_and_ablation(tmp_path):
    out = tmp_path / "run"
    args = ["run", "--knowledge", SAMPLE_K, "--questions", SAMPLE_Q, "--out", str(out),
            "--conditions", "kore", "kore_ablate_diversity"]
    assert main(args) == 0
    assert main(args + ["--fresh"]) == 0


def test_cli_search(capsys):
    code = main(["search", "What is the pH of pure water at 25 C?", "--knowledge", SAMPLE_K, "--strategy", "similarity", "--k", "1"])
    assert code == 0 and "che_001" in capsys.readouterr().out


def test_cli_split(tmp_path, capsys):
    k_rows = [k_row(f"a{i}", text=f"A reasonably long sentence number {i} for the document.") for i in range(12)]
    k = write_csv(tmp_path / "k.csv", K_HEADER, k_rows)
    q_rows = [q_row(f"q{i}", question=f"Unique question number {i} about topic {i * 7}?", expected_source_ids=f"a{i}") for i in range(12)]
    q = write_csv(tmp_path / "q.csv", Q_HEADER, q_rows)
    out = tmp_path / "split.csv"
    assert main(["split", "--questions", str(q), "--knowledge", str(k), "--out", str(out), "--seed", "3"]) == 0
    with out.open(encoding="utf-8") as fh:
        splits = [r["split"] for r in csv.DictReader(fh)]
    assert set(splits) == {"train", "dev", "test"} and len(splits) == 12
    assert main(["validate", "--knowledge", str(k), "--questions", str(out)]) == 0
