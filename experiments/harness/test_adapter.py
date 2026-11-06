"""어댑터 시험. 실행: python -m pytest -q experiments/harness
배관 확인이다. 샘플 데이터 결과를 연구 결과로 인용하지 않는다. HFTemplateGenerator는 import만 확인한다(GPU 없음)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.harness.adapter import (
    HFTemplateGenerator, ParityError, TemplateGenerator, assert_parity, check_config_parity, check_run_parity,
    harness_config_from_settings, make_generator, render_prompt_text, run_adapted, to_run_kwargs,
)
from harness.components import Edit, HarnessConfig, apply_edit
from harness.safety import load_dev_questions
from kore.generation import PromptTooLongError, build_prompt
from kore.io import load_knowledge

DATA = Path(__file__).resolve().parents[2] / "data"
TPL2 = "질문 {question} 에 근거 {evidence} 로만 답하라."
DEFAULT_TPL = "아래 근거만 사용해 질문에 짧게 답하라. 근거에 답이 없으면 모른다고 답하라.\n\n{evidence}\n\n질문: {question}\n답:"


@pytest.fixture(scope="module")
def items():
    return load_knowledge(DATA / "sample_knowledge.csv")


@pytest.fixture(scope="module")
def qs(items):
    return load_dev_questions(DATA / "sample_eval.csv", items, ("train", "dev"))


def rows(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()]


def test_default_template_equals_build_prompt(items):
    assert render_prompt_text(DEFAULT_TPL, "무엇?", items[:2]) == build_prompt("무엇?", items[:2])
    assert render_prompt_text(None, "무엇?", items[:2]) == build_prompt("무엇?", items[:2])
    assert render_prompt_text(TPL2, "무엇?", []) == build_prompt("무엇?", [])  # 근거 없음은 틀과 무관


def test_braces_in_evidence_safe(items):
    ev = [replace(items[0], text="a {question} b {x}")]
    out = render_prompt_text(TPL2, "Q1", ev)
    assert "a {question} b {x}" in out and out.startswith("질문 Q1")


def test_settings_flat_and_nested(qs):
    cfg = harness_config_from_settings({"k": 2, "candidate_k": 6, "token_budget": 300, "prompt_template": TPL2})
    p = cfg.pipeline_config()
    assert (p.k, p.candidate_k, p.token_budget) == (2, 6, 300) and cfg.prompt_template == TPL2
    again = harness_config_from_settings(cfg.to_dict())
    assert again.hash() == cfg.hash()


def test_bad_settings_rejected():
    for bad in ({"kk": 1}, {"k": 5, "candidate_k": 2}, {"prompt_template": "{question} {nope}"},
                {"pipeline": {}, "k": 1}):
        with pytest.raises(ValueError):
            harness_config_from_settings(bad)


def test_query_rules_change_question_only(qs):
    cfg = harness_config_from_settings({"query_rules": [{"op": "append_terms", "terms": ["추가어"]}]})
    kw = to_run_kwargs(cfg, qs, None)
    assert all(q.question.endswith("추가어") for q in kw["questions"])
    assert [q.question_id for q in kw["questions"]] == [q.question_id for q in qs]
    assert [q.expected_source_ids for q in kw["questions"]] == [q.expected_source_ids for q in qs]


def test_prompt_edit_reaches_prompt_and_output(items, qs, tmp_path):
    base = HarnessConfig.default()
    edited = apply_edit(base, Edit(id="p1", component="prompt_template", change={"template": TPL2}, rationale="배관 시험"))
    g0, g1 = make_generator(base), make_generator(edited)
    assert g0.name != g1.name
    run_adapted(items, qs, base, tmp_path / "a", g0)
    run_adapted(items, qs, edited, tmp_path / "b", g1)
    assert len(g0.prompts) == len(g1.prompts) == len(qs) and g1.prompts
    assert all(p0 != p1 for p0, p1 in zip(g0.prompts, g1.prompts))
    assert all(p.startswith("질문 ") for p in g1.prompts)
    a0 = [r["answer"] for r in rows(tmp_path / "a" / "run_log.jsonl")]
    a1 = [r["answer"] for r in rows(tmp_path / "b" / "run_log.jsonl")]
    assert a0 != a1


def test_same_config_is_deterministic(items, qs, tmp_path):
    cfg = harness_config_from_settings({"prompt_template": TPL2})
    g0, g1 = make_generator(cfg), make_generator(cfg)
    run_adapted(items, qs, cfg, tmp_path / "a", g0)
    run_adapted(items, qs, cfg, tmp_path / "b", g1)
    assert g0.prompts == g1.prompts


def test_k_and_token_budget_reach_selection(items, qs, tmp_path):
    c1 = harness_config_from_settings({"k": 1})
    c3 = harness_config_from_settings({"k": 3})
    run_adapted(items, qs, c1, tmp_path / "k1")
    run_adapted(items, qs, c3, tmp_path / "k3")
    r1, r3 = rows(tmp_path / "k1" / "run_log.jsonl"), rows(tmp_path / "k3" / "run_log.jsonl")
    assert max(len(r["selected_ids"]) for r in r1) <= 1 and max(len(r["selected_ids"]) for r in r3) > 1
    configs = {"k1": c1, "k3": c3}
    logs = {"k1": r1, "k3": r3}
    assert check_run_parity(logs, configs, vary=("k",)) == []
    assert any("k가" in p for p in check_run_parity(logs, configs))  # k를 허용하지 않으면 걸린다
    tb = harness_config_from_settings({"k": 3, "token_budget": 60})
    run_adapted(items, qs, tb, tmp_path / "tb")
    rt = rows(tmp_path / "tb" / "run_log.jsonl")
    assert check_run_parity({"tb": rt}, {"tb": tb}) == []


def test_parity_detects_differences():
    a = harness_config_from_settings({"k": 3})
    b = harness_config_from_settings({"k": 3, "prompt_template": TPL2})
    assert check_config_parity({"a": a, "b": b}, vary=("prompt_template",)) == []
    assert any("prompt_template" in p for p in check_config_parity({"a": a, "b": b}))
    c = harness_config_from_settings({"k": 3, "token_budget": 100, "candidate_k": 8})
    probs = check_config_parity({"a": a, "c": c})
    assert any("token_budget" in p for p in probs) and any("candidate_k" in p for p in probs)
    with pytest.raises(ParityError):
        assert_parity(probs)


def test_run_parity_detects_log_problems():
    cfg = harness_config_from_settings({"k": 1})
    row = lambda q, cand, sel: {"question_id": q, "candidate_ids": cand, "selected_ids": sel, "context_tokens": 5}
    probs = check_run_parity({"a": [row("q1", ["x", "y"], ["x", "y"])], "b": [row("q2", ["x"], ["x"])]},
                             {"a": cfg, "b": cfg})
    assert any("질문 집합" in p for p in probs) and any("k=1" in p for p in probs)
    probs = check_run_parity({"a": [row("q1", ["x"], ["x"])], "b": [row("q1", ["z"], ["z"])]}, {"a": cfg, "b": cfg})
    assert any("후보 문서 목록" in p for p in probs)


def test_max_input_tokens_stops(items):
    g = TemplateGenerator(template=TPL2, max_input_tokens=5)
    with pytest.raises(PromptTooLongError):
        g.generate("질문", items[:2])


def test_hf_generator_importable_only():
    assert issubclass(HFTemplateGenerator, object) and "_prompt" in HFTemplateGenerator.__dict__  # 실행은 GPU에서만
