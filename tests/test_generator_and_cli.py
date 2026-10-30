import json
from pathlib import Path

import pytest

from kore.cli import build_parser, main
from kore.doctor import check_environment, format_environment
from kore.generation import (
    ExtractiveGenerator, GeneratorSpec, HFGenerator, PROMPT_VERSION, build_prompt, load_generator, render_prompt,
)
from kore.io import read_questions
from kore.pipeline import KOREPipeline, all_conditions
from tests.helpers import mk_item

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_K, SAMPLE_Q = "data/sample_knowledge.csv", "data/sample_eval.csv"


class FakeTokenizer:
    def __init__(self, chat_template="x"):
        self.chat_template = chat_template
        self.calls = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        self.calls.append((messages, tokenize, add_generation_prompt, kwargs))
        return "<s>" + "|".join(f"{m['role']}:{m['content']}" for m in messages) + "<gen>"


# ---------------------------------------------------------------- 생성기 설정


def test_generator_configs_in_repo_are_valid_and_do_not_need_paid_services():
    for name in ("generator", "generator_fallback_qwen25_3b", "generator_transfer_exaone"):
        spec = GeneratorSpec.from_json(ROOT / "configs" / f"{name}.json")
        assert spec.kind == "hf" and "/" in spec.model and spec.dtype == "float16"  # T4는 bfloat16을 기본 지원하지 않는다
    main_spec = GeneratorSpec.from_json(ROOT / "configs" / "generator.json")
    assert main_spec == GeneratorSpec()  # 코드 기본값과 설정 파일이 어긋나지 않는다
    text = "".join((ROOT / "configs" / f).read_text(encoding="utf-8") for f in ("generator.json",))
    assert "api_key" not in text.lower()


def test_generator_spec_rejects_typos_and_bad_values():
    with pytest.raises(ValueError, match="모르는 키"):
        GeneratorSpec.from_dict({"modle": "x"})
    with pytest.raises(ValueError):
        GeneratorSpec.from_dict({"dtype": "int3"})
    with pytest.raises(ValueError):
        GeneratorSpec.from_dict({"kind": "gpt"})
    with pytest.raises(ValueError):
        GeneratorSpec.from_dict({"max_new_tokens": 0})
    assert GeneratorSpec.from_dict({"_note": "메모는 무시한다", "model": "a/b"}).model == "a/b"


def test_load_generator_for_non_model_kinds():
    assert load_generator(None) is None
    assert load_generator(GeneratorSpec(kind="none")) is None
    assert isinstance(load_generator(GeneratorSpec(kind="extractive")), ExtractiveGenerator)


def test_hf_generator_fails_clearly_without_torch():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="transformers"):
            HFGenerator(GeneratorSpec())
    else:  # pragma: no cover - torch가 있는 환경에서는 모델을 내려받아야 해서 시험하지 않는다
        pytest.skip("torch가 설치되어 있어 ImportError 경로를 시험할 수 없다")


# ---------------------------------------------------------------- 프롬프트 조립


def test_render_prompt_uses_chat_template_with_system_and_kwargs():
    tok = FakeTokenizer()
    ev = [mk_item("a1", "근거 본문")]
    out = render_prompt(tok, "질문?", ev, system_prompt="시스템", chat_template_kwargs={"enable_thinking": False})
    messages, tokenize, add_gen, kwargs = tok.calls[0]
    assert tokenize is False and add_gen is True and kwargs == {"enable_thinking": False}
    assert messages[0] == {"role": "system", "content": "시스템"} and messages[1]["role"] == "user"
    assert build_prompt("질문?", ev) in out and out.endswith("<gen>")


def test_render_prompt_without_template_or_system_prompt():
    plain = FakeTokenizer(chat_template=None)
    assert render_prompt(plain, "q?", []) == build_prompt("q?", [])
    assert render_prompt(plain, "q?", [], system_prompt="S").startswith("S\n\n")
    tok = FakeTokenizer()
    render_prompt(tok, "q?", [])
    assert [m["role"] for m in tok.calls[0][0]] == ["user"]  # 시스템 프롬프트가 없으면 넣지 않는다


def test_conditions_see_identical_prompts():
    """세 조건이 같은 프롬프트 틀을 쓰는지(근거만 다르다) 확인한다. 공정한 비교의 전제다."""
    ev = [mk_item("a1", "근거 하나"), mk_item("a2", "근거 둘")]
    with_ev, none = build_prompt("Q", ev), build_prompt("Q", [])
    assert with_ev.endswith("질문: Q\n답:") and none.endswith("질문: Q\n답:")
    assert PROMPT_VERSION  # 프롬프트를 바꾸면 이 값을 올려서 이전 결과와 섞이지 않게 한다


# ---------------------------------------------------------------- 진단 조건 gold_context


def test_gold_context_gives_exactly_the_gold_documents(items):
    pipe = KOREPipeline(items)
    cands = pipe.candidates("Which planet is the largest in the Solar System?")
    got = pipe.select("q", cands, "gold_context", expected_ids=["ast_001", "nope"])
    assert [g.id for g in got] == ["ast_001"]  # 없는 id는 건너뛴다
    assert pipe.select("q", cands, "gold_context") == []
    assert "gold_context" in all_conditions()


# ---------------------------------------------------------------- CLI


def test_cli_run_with_extractive_generator_records_rubric(tmp_path):
    out = tmp_path / "r"
    assert main(["run", "--knowledge", SAMPLE_K, "--questions", SAMPLE_Q, "--out", str(out), "--generator", "extractive"]) == 0
    env = json.loads((out / "env.json").read_text(encoding="utf-8"))
    assert env["generator"] == "extractive-v0"
    rows = [json.loads(line) for line in (out / "run_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(r["rubric_score"] is not None for r in rows)


def test_cli_run_default_has_no_generator_and_config_file_is_used(tmp_path):
    out = tmp_path / "r0"
    assert main(["run", "--knowledge", SAMPLE_K, "--questions", SAMPLE_Q, "--out", str(out)]) == 0
    assert json.loads((out / "env.json").read_text(encoding="utf-8"))["generator"] == "none"
    cfg = tmp_path / "g.json"
    cfg.write_text(json.dumps({"kind": "extractive"}), encoding="utf-8")
    out2 = tmp_path / "r1"
    assert main(["run", "--knowledge", SAMPLE_K, "--questions", SAMPLE_Q, "--out", str(out2), "--generator-config", str(cfg)]) == 0
    assert json.loads((out2 / "env.json").read_text(encoding="utf-8"))["generator"] == "extractive-v0"


def test_cli_stats_modes(tmp_path, capsys):
    assert main(["stats", "--knowledge", SAMPLE_K, "--questions", SAMPLE_Q, "--stage", "week1"]) == 0
    assert "문서 수" in capsys.readouterr().out
    assert main(["stats", "--knowledge", SAMPLE_K, "--questions", SAMPLE_Q, "--stage", "final", "--strict"]) == 1  # 샘플은 규모가 작다
    capsys.readouterr()
    assert main(["stats", "--knowledge", SAMPLE_K, "--questions", SAMPLE_Q, "--stage", "week1", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stage"] == "week1" and payload["checks"]


def test_cli_stats_fails_on_invalid_data(tmp_path):
    bad = tmp_path / "k.csv"
    bad.write_text("id,domain\na,b\n", encoding="utf-8")
    assert main(["stats", "--knowledge", str(bad)]) == 1


def test_cli_split_defaults_to_the_documented_ratios(tmp_path):
    from tests.test_data_tools import synthetic

    items, qs = synthetic(n_docs=160, n_q=120)
    from kore.io import write_questions

    raw = tmp_path / "raw.csv"
    write_questions(raw, qs)
    out = tmp_path / "ev.csv"
    assert main(["split", "--questions", raw.as_posix(), "--out", out.as_posix()]) == 0
    got, _ = read_questions(out)
    counts = {s: sum(q.split == s for q in got) for s in ("train", "dev", "test")}
    assert counts == {"train": 60, "dev": 24, "test": 36}  # 120문항 × 0.5/0.2/0.3
    # 이미 있는 파일을 말없이 덮어쓰지 않는다(분할이 바뀌면 test 질문이 바뀐다)
    assert main(["split", "--questions", raw.as_posix(), "--out", out.as_posix()]) == 1
    assert main(["split", "--questions", raw.as_posix(), "--out", out.as_posix(), "--reshuffle", "--ratios", "0.6", "0.2", "0.2"]) == 0
    got2, _ = read_questions(out)
    assert sum(q.split == "test" for q in got2) == 24


def test_cli_split_keep_from_never_moves_existing_questions(tmp_path, capsys):
    from kore.io import write_questions
    from tests.test_data_tools import synthetic

    _, qs = synthetic(n_docs=160, n_q=120)
    first_raw, out = tmp_path / "first.csv", tmp_path / "ev.csv"
    write_questions(first_raw, qs[:60])
    assert main(["split", "--questions", first_raw.as_posix(), "--out", out.as_posix()]) == 0
    before = {q.question_id: q.split for q in read_questions(out)[0]}
    full_raw = tmp_path / "full.csv"
    write_questions(full_raw, qs)
    capsys.readouterr()
    assert main(["split", "--questions", full_raw.as_posix(), "--out", out.as_posix(), "--keep-from", out.as_posix()]) == 0
    assert "유지한 질문 60개" in capsys.readouterr().out
    after = {q.question_id: q.split for q in read_questions(out)[0]}
    assert len(after) == 120 and all(after[qid] == sp for qid, sp in before.items())  # 하나도 옮겨지지 않았다
    counts = {s: sum(v == s for v in after.values()) for s in ("train", "dev", "test")}
    assert counts["test"] >= 36 and counts["train"] + counts["dev"] + counts["test"] == 120
    assert main(["split", "--questions", full_raw.as_posix(), "--out", out.as_posix(), "--keep-from", "nope.csv"]) == 1


def test_cli_tune_runs_on_train_dev_and_reports_the_trial_count(tmp_path, capsys):
    assert main(["tune", "--knowledge", SAMPLE_K, "--questions", SAMPLE_Q, "--out", str(tmp_path / "t.json")]) == 0
    out = capsys.readouterr().out
    assert "유사도 기준선" in out and "시도한 조합 수" in out
    assert json.loads((tmp_path / "t.json").read_text(encoding="utf-8"))["splits"] == ["train", "dev"]
    with pytest.raises(SystemExit):  # argparse가 test를 막는다
        main(["tune", "--knowledge", SAMPLE_K, "--questions", SAMPLE_Q, "--splits", "test"])


def test_cli_label_dry_run_counts_calls_without_a_model(capsys):
    assert main(["label", "--knowledge", SAMPLE_K, "--questions", SAMPLE_Q, "--out", "unused", "--dry-run", "--sec-per-call", "0.3"]) == 0
    out = capsys.readouterr().out
    assert "질문당 93번" in out and "총" in out and "분" in out


def test_cli_label_without_scoring_model_explains_what_is_needed(tmp_path, capsys):
    assert main(["label", "--knowledge", SAMPLE_K, "--questions", SAMPLE_Q, "--out", str(tmp_path / "x")]) == 1
    assert "--generator hf" in capsys.readouterr().out


def test_cli_label_report_reads_dq_log(tmp_path, capsys):
    rows = [{"question_id": "q1", "domain": "d", "split": "train", "difficulty": "easy", "candidate_ids": ["a", "b", "c"],
             "expected_source_ids": ["a"], "scores": {"": -5.0, "a": -4.0, "b": -4.5, "c": -5.2, "a|b": -3.9, "a|c": -4.1, "b|c": -4.4}}]
    (tmp_path / "dq_log.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    assert main(["label-report", "--dir", str(tmp_path), "--k", "2", "--csv", str(tmp_path / "per_q.csv")]) == 0
    assert "예비 신호" in capsys.readouterr().out and (tmp_path / "per_q.csv").exists()
    assert main(["label-report", "--dir", str(tmp_path / "missing")]) == 1


def test_cli_doctor_runs_and_parser_lists_all_commands(capsys):
    assert main(["doctor"]) == 0
    assert "파이썬" in capsys.readouterr().out
    rows = check_environment("data")
    assert rows[0][1] == "파이썬" and "[OK]" in format_environment(rows)
    names = set(build_parser()._subparsers._group_actions[0].choices)
    assert {"validate", "stats", "ingest", "split", "run", "screen", "label", "label-report", "blind-export", "blind-score",
            "search", "doctor", "tune"} <= names
