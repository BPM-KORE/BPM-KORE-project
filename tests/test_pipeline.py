import json
from pathlib import Path

import pytest

from kore.generation import ExtractiveGenerator
from kore.pipeline import (
    BASE_CONDITIONS, KOREPipeline, PipelineConfig, all_conditions, run_experiment,
)
from kore.selector import SelectorWeights


def read_log(out: Path):
    return [json.loads(l) for l in (out / "run_log.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def read_csv_rows(path: Path):
    import csv

    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_default_run_excludes_test_split(tmp_path, items, questions):
    res = run_experiment(items, questions, tmp_path / "r")
    rows = read_log(tmp_path / "r")
    assert len(rows) == 7 * 3 and res["n_rows"] == 21
    assert {r["split"] for r in rows} == {"train", "dev"}
    assert {r["condition"] for r in rows} == set(BASE_CONDITIONS)


def test_include_test_adds_test_rows(tmp_path, items, questions):
    run_experiment(items, questions, tmp_path / "r", include_test=True)
    rows = read_log(tmp_path / "r")
    assert len(rows) == 9 * 3 and {r["split"] for r in rows} == {"train", "dev", "test"}
    env = json.loads((tmp_path / "r" / "env.json").read_text(encoding="utf-8"))
    assert env["include_test"] is True


def test_no_retrieval_has_empty_context_and_zero_recall(tmp_path, items, questions):
    run_experiment(items, questions, tmp_path / "r")
    for r in read_log(tmp_path / "r"):
        if r["condition"] == "no_retrieval":
            assert r["selected_ids"] == [] and r["recall"] == 0.0 and r["context_tokens"] == 0


def test_conditions_share_the_same_candidates(tmp_path, items, questions):
    run_experiment(items, questions, tmp_path / "r")
    by_q = {}
    for r in read_log(tmp_path / "r"):
        by_q.setdefault(r["question_id"], []).append(r)
    for rows in by_q.values():
        assert len({tuple(r["candidate_ids"]) for r in rows}) == 1
        assert len({r["candidate_recall"] for r in rows}) == 1
        for r in rows:  # 선택은 후보의 부분집합이다
            assert set(r["selected_ids"]) <= set(r["candidate_ids"])


def test_summary_and_comparison_files(tmp_path, items, questions):
    run_experiment(items, questions, tmp_path / "r")
    results = read_csv_rows(tmp_path / "r" / "results.csv")
    allrows = [r for r in results if r["split"] == "all"]
    assert {r["condition"] for r in allrows} == set(BASE_CONDITIONS)
    for r in allrows:
        assert int(r["n"]) == 7
        assert float(r["recall_ci_lo"]) <= float(r["recall"]) <= float(r["recall_ci_hi"])
    comps = read_csv_rows(tmp_path / "r" / "comparisons.csv")
    kinds = {(c["condition"], c["metric"], c["split"]) for c in comps}
    assert ("kore", "recall", "all") in kinds and ("no_retrieval", "recall", "all") in kinds
    assert all(c["baseline"] == "vanilla_rag" for c in comps)
    assert next(c for c in comps if c["condition"] == "kore" and c["split"] == "all")["n"] == "7"


def test_env_records_reproducibility_info(tmp_path, items, questions):
    run_experiment(items, questions, tmp_path / "r", seed=7)
    env = json.loads((tmp_path / "r" / "env.json").read_text(encoding="utf-8"))
    for key in ("seed", "python", "platform", "config_hash", "data_hash", "contract_version", "config", "conditions"):
        assert key in env
    assert env["seed"] == 7 and env["include_test"] is False and env["generator"] == "none"


def test_run_is_reproducible(tmp_path, items, questions):
    run_experiment(items, questions, tmp_path / "a", seed=1)
    run_experiment(items, questions, tmp_path / "b", seed=1)
    strip = lambda rows: [{k: v for k, v in r.items() if not k.endswith("_ms")} for r in rows]
    assert strip(read_log(tmp_path / "a")) == strip(read_log(tmp_path / "b"))
    ra = read_csv_rows(tmp_path / "a" / "results.csv")
    rb = read_csv_rows(tmp_path / "b" / "results.csv")
    cols = ["condition", "split", "n", "recall", "recall_ci_lo", "recall_ci_hi", "precision", "hit", "mrr"]
    assert [{c: r[c] for c in cols} for r in ra] == [{c: r[c] for c in cols} for r in rb]


def test_resume_does_not_duplicate_rows(tmp_path, items, questions):
    run_experiment(items, questions, tmp_path / "r")
    before = (tmp_path / "r" / "run_log.jsonl").read_text(encoding="utf-8")
    run_experiment(items, questions, tmp_path / "r")
    assert (tmp_path / "r" / "run_log.jsonl").read_text(encoding="utf-8") == before


def test_resume_adds_only_missing_conditions(tmp_path, items, questions):
    run_experiment(items, questions, tmp_path / "r", conditions=("vanilla_rag",))
    run_experiment(items, questions, tmp_path / "r", conditions=("vanilla_rag", "kore"))
    rows = read_log(tmp_path / "r")
    assert sum(r["condition"] == "vanilla_rag" for r in rows) == 7
    assert sum(r["condition"] == "kore" for r in rows) == 7


def test_resume_survives_a_truncated_last_line(tmp_path, items, questions):
    out = tmp_path / "r"
    run_experiment(items, questions, out, conditions=("vanilla_rag",))
    with (out / "run_log.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"question_id": "q0')  # 비정상 종료로 잘린 줄
    run_experiment(items, questions, out, conditions=("vanilla_rag", "kore"))
    valid = []
    for line in (out / "run_log.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            valid.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    assert sum(r["condition"] == "kore" for r in valid) == 7
    assert sum(r["condition"] == "vanilla_rag" for r in valid) == 7


def test_resume_refuses_changed_config_but_fresh_restarts(tmp_path, items, questions):
    run_experiment(items, questions, tmp_path / "r")
    with pytest.raises(ValueError, match="config_hash"):
        run_experiment(items, questions, tmp_path / "r", config=PipelineConfig(k=2))
    res = run_experiment(items, questions, tmp_path / "r", config=PipelineConfig(k=2), resume=False)
    assert res["n_rows"] == 21 and len(read_log(tmp_path / "r")) == 21


def test_resume_refuses_changed_data_seed_or_generator(tmp_path, items, questions):
    run_experiment(items, questions, tmp_path / "r")
    with pytest.raises(ValueError, match="data_hash"):
        run_experiment(items[:-1], questions, tmp_path / "r")
    with pytest.raises(ValueError, match="seed"):
        run_experiment(items, questions, tmp_path / "r", seed=99)
    with pytest.raises(ValueError, match="generator"):
        run_experiment(items, questions, tmp_path / "r", generator=ExtractiveGenerator())


def test_invalid_inputs(tmp_path, items, questions):
    with pytest.raises(ValueError, match="알 수 없는 조건"):
        run_experiment(items, questions, tmp_path / "r", conditions=("bogus",))
    only_test = [q for q in questions if q.split == "test"]
    with pytest.raises(ValueError, match="include_test"):
        run_experiment(items, only_test, tmp_path / "r2")
    with pytest.raises(ValueError):
        PipelineConfig(k=5, candidate_k=3).validate()


def test_ablation_conditions_run(tmp_path, items, questions):
    conds = ("kore", "kore_ablate_diversity", "kore_ablate_redundancy")
    run_experiment(items, questions, tmp_path / "r", conditions=conds)
    assert {r["condition"] for r in read_log(tmp_path / "r")} == set(conds)
    assert set(all_conditions()) >= set(conds)


def test_ablating_a_zero_weight_changes_nothing(items, questions):
    cfg = PipelineConfig(weights=SelectorWeights(diversity=0.0))
    pipe = KOREPipeline(items, cfg)
    for q in questions:
        cands = pipe.candidates(q.question)
        a = [i.id for i in pipe.select(q.question, cands, "kore")]
        b = [i.id for i in pipe.select(q.question, cands, "kore_ablate_diversity")]
        assert a == b


def test_unknown_condition_in_select(items):
    pipe = KOREPipeline(items)
    with pytest.raises(ValueError):
        pipe.select("q", [], "bogus")


def test_with_generator_rubric_scores_are_recorded(tmp_path, items, questions):
    run_experiment(items, questions, tmp_path / "r", generator=ExtractiveGenerator())
    rows = read_log(tmp_path / "r")
    assert all(r["answer"] is not None for r in rows)
    assert all(r["rubric_score"] is not None and 0.0 <= r["rubric_score"] <= 1.0 for r in rows)
    assert all(r["rubric_score"] == 0.0 for r in rows if r["condition"] == "no_retrieval")
    env = json.loads((tmp_path / "r" / "env.json").read_text(encoding="utf-8"))
    assert env["generator"] == "extractive-v0"
    results = read_csv_rows(tmp_path / "r" / "results.csv")
    assert all(r["rubric_score"] != "" for r in results)


def test_without_generator_there_is_no_rubric_column_value(tmp_path, items, questions):
    run_experiment(items, questions, tmp_path / "r")
    assert all(r["rubric_score"] == "" for r in read_csv_rows(tmp_path / "r" / "results.csv"))
    assert all(r["answer"] is None for r in read_log(tmp_path / "r"))


def test_token_budget_is_applied_to_both_selectors(items, questions):
    cfg = PipelineConfig(token_budget=60)
    pipe = KOREPipeline(items, cfg)
    for q in questions:
        cands = pipe.candidates(q.question)
        for cond in ("vanilla_rag", "kore"):
            from kore.tokenize import estimate_item_tokens

            assert sum(estimate_item_tokens(i) for i in pipe.select(q.question, cands, cond)) <= 60


def test_progress_callback(tmp_path, items, questions):
    calls = []
    run_experiment(items, questions, tmp_path / "r", progress=lambda n, total: calls.append((n, total)))
    assert calls[-1] == (7, 7) and len(calls) == 7
