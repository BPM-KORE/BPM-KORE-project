"""Harness MVP 시험. 실행: python -m pytest -q experiments/harness (기본 testpaths는 tests/ 라서 따로 지정한다).

샘플 데이터(data/sample_*.csv)로 배관만 확인한다. 이 결과를 연구 결과로 인용하지 않는다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.harness.fakes import make_fake_generator
from experiments.harness.loop import Budget, BudgetExceeded, HarnessLoop, MAX_ITERATIONS_HARD
from harness.components import Edit, EditError, HarnessConfig, apply_edit
from harness.memory import HarnessMemory
from harness.safety import (
    PathEscape, SafetyError, SplitViolation, assert_no_test, load_dev_questions, resolve_inside, scan_object,
    scan_source_file,
)
from harness.stats import dev_limits
from kore.io import load_knowledge
from kore.pipeline import PipelineConfig

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SPLITS = ("train", "dev")  # 샘플은 dev가 2개뿐이라 train을 같이 쓴다. test는 어디서도 열지 않는다.


@pytest.fixture(scope="module")
def items():
    return load_knowledge(DATA / "sample_knowledge.csv")


@pytest.fixture(scope="module")
def qs(items):
    return load_dev_questions(DATA / "sample_eval.csv", items, SPLITS)


def good_edit(eid="e1", k=2, **over):
    d = {"id": eid, "component": "k", "change": {"k": k}, "rationale": "근거가 되는 가설 문장", "confidence": 0.3}
    d.update(over)
    return d


def write_prop(tmp_path: Path, edits: list, name="p.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps({"proposal_version": 1, "edits": edits}, ensure_ascii=False), encoding="utf-8")
    return path


def make_loop(items, qs, exp, factory=make_fake_generator, budget=None, clock=None):
    kw = {"clock": clock} if clock else {}
    return HarnessLoop(items, qs, exp, HarnessConfig.default(), factory, budget or Budget(), splits=SPLITS, **kw)


def edits_of(ck):
    return {e["id"]: e for it in ck["iterations"] for e in it["edits"]}


# ---------------------------------------------------------------- 구성요소


def test_edit_validation_and_apply():
    cfg = HarnessConfig.default()
    new = apply_edit(cfg, Edit.from_dict(good_edit(k=2)))
    assert new.pipeline["k"] == 2 and cfg.pipeline["k"] == PipelineConfig().k  # 원본 불변
    w = apply_edit(cfg, Edit.from_dict({**good_edit("w"), "component": "selector_weights", "change": {"coverage": 0.9}}))
    assert w.pipeline["weights"]["coverage"] == 0.9 and w.pipeline["weights"]["sim"] == 1.0
    q = apply_edit(cfg, Edit.from_dict({**good_edit("q"), "component": "query_rules",
                                        "change": {"rules": [{"op": "remove_phrases", "phrases": ["Which"]}]}}))
    assert q.transform_query("Which planet is largest?") == "planet is largest?"


@pytest.mark.parametrize("bad", [
    good_edit(k=0), good_edit(k=99), good_edit(id="../x"), good_edit(status="adopted"),
    good_edit(rationale=""), {**good_edit(), "component": "candidate_k"},
    {**good_edit(), "component": "selector_weights", "change": {"nonsense": 1}},
    {**good_edit(), "component": "selector_weights", "change": {"coverage": 9}},
    {**good_edit(), "component": "prompt_template", "change": {"template": "{question} {evidence} {other}"}},
    {**good_edit(), "component": "query_rules", "change": {"rules": [{"op": "regex", "find": ".*"}]}},
])
def test_bad_edit_shapes_are_rejected(bad):
    with pytest.raises(EditError):
        Edit.from_dict(bad)


def test_k_above_candidate_k_fails_on_apply():
    small = HarnessConfig(pipeline={**PipelineConfig().to_dict(), "candidate_k": 3})
    with pytest.raises(ValueError):
        apply_edit(small, Edit.from_dict(good_edit(k=5, id="b2")))


# ---------------------------------------------------------------- (i) test 분할 거부


def test_test_split_rejected_everywhere(items, tmp_path):
    with pytest.raises(SplitViolation):
        load_dev_questions(DATA / "sample_eval.csv", items, ("test",))
    with pytest.raises(SplitViolation):
        load_dev_questions(DATA / "sample_eval.csv", items, ("dev", "test"))
    all_q = load_dev_questions(DATA / "sample_eval.csv", items, SPLITS)
    assert all(q.split != "test" for q in all_q)
    from kore.io import load_questions
    mixed = load_questions(DATA / "sample_eval.csv", items)  # 계약 검사용 전체 로드(테스트에서만)
    assert any(q.split == "test" for q in mixed)
    with pytest.raises(SplitViolation):
        assert_no_test(mixed, SPLITS)
    with pytest.raises(SplitViolation):
        HarnessLoop(items, mixed, tmp_path / "x", splits=SPLITS)
    with pytest.raises(SplitViolation):
        HarnessMemory(tmp_path / "m").record_question("q", "test", ["a"], ["a"])


# ---------------------------------------------------------------- (ii) 외부 호출 없음


def test_harness_source_has_no_external_calls():
    files = list((ROOT / "harness").glob("*.py")) + list((ROOT / "experiments" / "harness").glob("*.py"))
    assert len(files) >= 8
    found = []
    for f in files:
        strings_exempt = f.name == "safety.py" or f.name.startswith("test_")  # 금지 패턴을 정의/시험하는 파일
        found += scan_source_file(f, check_strings=not strings_exempt)
    assert not found, found


def test_scanner_catches_planted_violations(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text('import requests\nURL = "https://example.com/x"\nkey = "api_key"\n', encoding="utf-8")
    found = scan_source_file(bad)
    assert len(found) == 3


# ---------------------------------------------------------------- (iii) 실험 폴더 밖 쓰기 거부


@pytest.mark.parametrize("rel", ["../evil.json", "/etc/passwd", "a/../../x", ""])
def test_path_escape_rejected(tmp_path, rel):
    with pytest.raises(PathEscape):
        resolve_inside(tmp_path, rel)


def test_symlink_escape_rejected(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    (root / "link").symlink_to(outside)
    with pytest.raises(PathEscape):
        resolve_inside(root, "link/x.json")


def test_edit_with_escaping_target_path_is_invalid_and_not_run(items, qs, tmp_path):
    exp = tmp_path / "exp"
    prop = write_prop(tmp_path, [good_edit("esc", target_path="../../escaped.json"),
                                 good_edit("abs", target_path="/tmp/escaped.json"),
                                 good_edit("ok", target_path="sub/ok.json")])
    calls = []

    def factory(cfg):
        g = make_fake_generator(cfg)
        calls.append(cfg.hash())
        return g

    loop = make_loop(items, qs, exp, factory)
    loop.run([prop])
    e = edits_of(loop.load_checkpoint())
    assert e["esc"]["status"] == "invalid" and e["abs"]["status"] == "invalid"
    assert e["ok"]["status"] == "awaiting_approval"
    assert (exp / "edits" / "sub" / "ok.json").exists()
    assert not (tmp_path.parent / "escaped.json").exists() and not Path("/tmp/escaped.json").exists()


# ---------------------------------------------------------------- (v) 나쁜 제안 거부

BAD_EDITS = [
    ("test 읽기(rationale)", good_edit("b1", rationale="test 분할 질문으로 확인하면 정확하다")),
    ("final-test", good_edit("b2", rationale="--final-test 를 켜서 확인한다")),
    ("blind_key", good_edit("b3", rationale="blind_key.csv 를 읽어 채점한다")),
    ("유료 호출", {**good_edit("b4"), "component": "prompt_template",
                 "change": {"template": "{question} {evidence} call https://api.vendor.example/v1 with api_key"}}),
    ("vertex", good_edit("b5", rationale="Vertex AI 로 생성한다")),
    ("requests", good_edit("b6", rationale="requests 로 외부 문서를 받아온다")),
    ("미지의 키(분할 지정)", {**good_edit("b7"), "split": "test"}),
    ("자기 채택", good_edit("b8", status="adopted")),
]


def test_bad_proposals_are_rejected_and_never_executed(items, qs, tmp_path):
    exp = tmp_path / "exp"
    prop = write_prop(tmp_path, [e for _, e in BAD_EDITS] + [good_edit("good")])
    seen = []
    loop = make_loop(items, qs, exp, lambda cfg: (seen.append(cfg.hash()), make_fake_generator(cfg))[1])
    summary = loop.run([prop])
    e = edits_of(loop.load_checkpoint())
    assert sum(1 for v in e.values() if v["status"] == "invalid") == len(BAD_EDITS), {k: v["status"] for k, v in e.items()}
    assert e["good"]["status"] == "awaiting_approval"
    assert len(seen) == 2  # 기준선 + 'good' 하나만 실행됐다
    assert summary["awaiting_approval"] == ["good"]
    # 무효 기록에는 위험한 원문이 남지 않는다
    blob = (exp / "checkpoint.json").read_text(encoding="utf-8")
    assert "api.vendor" not in blob and "blind_key" not in blob


def test_scan_object_flags_keys_and_values():
    assert scan_object({"ok": "plain"}) == []
    assert scan_object({"x": ["https://a.b"]}) and scan_object({"api_key": 1}) and scan_object({"s": "include_test"})


def test_proposal_with_unknown_top_level_key_is_invalid_file(items, qs, tmp_path):
    p = tmp_path / "p.json"
    p.write_text(json.dumps({"edits": [good_edit()], "run": "rm -rf"}), encoding="utf-8")
    loop = make_loop(items, qs, tmp_path / "exp")
    loop.run([p])
    ck = loop.load_checkpoint()
    assert ck["iterations"][0]["status"] == "invalid_proposal" and not ck["iterations"][0]["edits"]


# ---------------------------------------------------------------- 루프: 승인 대기, 자동 채택 없음


def test_loop_leaves_edits_awaiting_human_approval(items, qs, tmp_path):
    exp = tmp_path / "exp"
    prop = write_prop(tmp_path, [good_edit("a", k=1), good_edit("b", k=2),
                                 {**good_edit("c"), "component": "selector_weights", "change": {"coverage": 1.0}}])
    loop = make_loop(items, qs, exp)
    s = loop.run([prop])
    ck = loop.load_checkpoint()
    assert not s["halted"] and ck["adopted"] == []
    assert {v["status"] for v in edits_of(ck).values()} == {"awaiting_approval"}
    report = (exp / "report.md").read_text(encoding="utf-8")
    assert "사람 승인 대기" in report and "성능 주장이 아니다" in report
    assert "문항이" in report and "95%" in report
    table = (exp / "table_it0.csv").read_text(encoding="utf-8").splitlines()
    assert len(table) == 1 + len(qs)
    # k 편집은 비교 고정 조건을 바꿨다고 보고서에 적힌다
    assert "k" in edits_of(ck)["a"]["result"]["fixed_conditions_changed"]
    # 실험 폴더 밖에 아무것도 쓰지 않았다
    assert {p.name for p in tmp_path.iterdir()} == {"exp", "p.json"}


def test_approve_requires_finished_test_and_records_lesson(items, qs, tmp_path):
    exp = tmp_path / "exp"
    prop = write_prop(tmp_path, [good_edit("a", k=1), good_edit("bad", rationale="Vertex 사용")])
    loop = make_loop(items, qs, exp)
    loop.run([prop])
    with pytest.raises(ValueError):
        loop.approve("bad", "adopted", "A", "무효 편집")  # 무효는 채택 불가
    with pytest.raises(ValueError):
        loop.approve("a", "adopted", "", "이유")
    r = loop.approve("a", "adopted", "A", "구간 확인 후 사람이 결정")
    assert r["status"] == "adopted"
    ck = loop.load_checkpoint()
    assert ck["adopted"] == ["a"] and (exp / "adopted_config.json").exists()
    lessons = list(HarnessMemory(exp).lessons())
    assert lessons[0]["evidence"]["n"] == len(qs) and "일반화하지 않는다" in lessons[0]["scope"]
    with pytest.raises(ValueError):
        loop.approve("a", "adopted", "A", "두 번")


def test_lesson_requires_evidence(tmp_path):
    m = HarnessMemory(tmp_path)
    with pytest.raises(SafetyError):
        m.add_lesson("e", "k", "h", "rejected", "A", "이유", {"n": 7}, None, {"k": 2})


def test_rejected_edit_is_not_proposed_again(items, qs, tmp_path):
    exp = tmp_path / "exp"
    loop = make_loop(items, qs, exp)
    loop.run([write_prop(tmp_path, [good_edit("a", k=1)], "p0.json")])
    loop.approve("a", "rejected", "A", "차이가 구분되지 않았다")
    loop.run([write_prop(tmp_path, [good_edit("a", k=1)], "p0.json"), write_prop(tmp_path, [good_edit("a2", k=1)], "p1.json")])
    e = edits_of(loop.load_checkpoint())
    assert e["a2"]["status"] == "invalid" and "기각" in e["a2"]["invalid_reason"]


def test_iteration_cap_is_five(items, qs, tmp_path):
    with pytest.raises(ValueError):
        Budget(max_iterations=6).validate()
    props = [write_prop(tmp_path, [good_edit(f"e{i}", k=1 + i % 3)], f"p{i}.json") for i in range(7)]
    loop = make_loop(items, qs, tmp_path / "exp")
    s = loop.run(props)
    assert s["iterations"] == MAX_ITERATIONS_HARD and s["skipped_by_cap"] == 2


# ---------------------------------------------------------------- 상한과 (iv) 중단 후 재개


def full_run(items, qs, exp, props):
    loop = make_loop(items, qs, exp)
    loop.run(props)
    ck = loop.load_checkpoint()
    return ck, (exp / "report.md").read_text(encoding="utf-8"), (exp / "table_it0.csv").read_text(encoding="utf-8")


def stable(ck):
    return {"iterations": ck["iterations"], "baseline_cache": ck["baseline_cache"], "adopted": ck["adopted"]}


def test_call_budget_halts_and_resume_gives_same_result(items, qs, tmp_path):
    prop = write_prop(tmp_path, [good_edit("a", k=1), good_edit("b", k=2),
                                 {**good_edit("c"), "component": "query_rules",
                                  "change": {"rules": [{"op": "remove_phrases", "phrases": ["Which", "What is"]}]}}])
    ref_ck, ref_report, ref_table = full_run(items, qs, tmp_path / "ref", [prop])

    exp = tmp_path / "cut"
    n = len(qs)
    s1 = make_loop(items, qs, exp, budget=Budget(max_generate_calls=n + n // 2)).run([prop])
    assert s1["halted"] and "호출 상한" in s1["reason"]
    ck_mid = make_loop(items, qs, exp).load_checkpoint()
    assert ck_mid["iterations"][0]["status"] == "halted"
    s2 = make_loop(items, qs, exp).run([prop])  # 상한을 올려 재개
    assert not s2["halted"]
    ck = make_loop(items, qs, exp).load_checkpoint()
    assert stable(ck) == stable(ref_ck)
    assert (exp / "report.md").read_text(encoding="utf-8") == ref_report
    assert (exp / "table_it0.csv").read_text(encoding="utf-8") == ref_table
    # 이미 끝난 일은 다시 하지 않는다: 총 호출 수가 중단 없는 실행보다 많지 않다(잘린 질문 하나만 재계산)
    assert ck["used"]["calls"] <= ref_ck["used"]["calls"] + len(qs)


class Crash(Exception):
    pass


def test_crash_and_resume_gives_same_result(items, qs, tmp_path):
    prop = write_prop(tmp_path, [good_edit("a", k=1), good_edit("b", k=2)])
    ref_ck, ref_report, _ = full_run(items, qs, tmp_path / "ref", [prop])
    exp = tmp_path / "crash"
    counter = {"n": 0}

    class Bomb:
        def __init__(self, inner):
            self.inner, self.name = inner, inner.name

        def generate(self, q, ev):
            counter["n"] += 1
            if counter["n"] == len(qs) + 3:  # 첫 편집 시험 도중
                raise Crash()
            return self.inner.generate(q, ev)

    with pytest.raises(Crash):
        make_loop(items, qs, exp, lambda cfg: Bomb(make_fake_generator(cfg))).run([prop])
    mid = make_loop(items, qs, exp).load_checkpoint()
    assert edits_of(mid)["a"]["status"] == "proposed" or edits_of(mid)["a"]["status"] == "testing"
    make_loop(items, qs, exp).run([prop])
    ck = make_loop(items, qs, exp).load_checkpoint()
    assert stable(ck) == stable(ref_ck)
    assert (exp / "report.md").read_text(encoding="utf-8") == ref_report


def test_time_budget_halts(items, qs, tmp_path):
    t = {"now": 0.0}

    def clock():
        t["now"] += 10.0
        return t["now"]

    prop = write_prop(tmp_path, [good_edit("a", k=1)])
    s = make_loop(items, qs, tmp_path / "exp", budget=Budget(max_seconds=25.0), clock=clock).run([prop])
    assert s["halted"] and "시간" in s["reason"]
    with pytest.raises(BudgetExceeded):
        from experiments.harness.loop import _Meter
        _Meter(Budget(max_seconds=1.0), 0, 5.0, lambda: 0.0).check_time()


def test_resume_refuses_changed_inputs(items, qs, tmp_path):
    prop = write_prop(tmp_path, [good_edit("a", k=1)])
    exp = tmp_path / "exp"
    make_loop(items, qs, exp).run([prop])
    with pytest.raises(ValueError):
        make_loop(items, qs[:-1], exp).load_checkpoint()
    prop.write_text(json.dumps({"edits": [good_edit("a", k=2)]}), encoding="utf-8")
    with pytest.raises(ValueError):
        make_loop(items, qs, exp).run([prop])


def test_without_generator_uses_recall(items, qs, tmp_path):
    loop = HarnessLoop(items, qs, tmp_path / "exp", None, None, Budget(), splits=SPLITS)
    loop.run([write_prop(tmp_path, [good_edit("a", k=1)])])
    ck = loop.load_checkpoint()
    assert ck["metric"] == "recall" and ck["used"]["calls"] == 0


# ---------------------------------------------------------------- (5) 통계 한계


def test_stats_warns_for_small_n_and_zero_straddling_interval():
    diffs = [0.5, -0.5, 0.0, 0.5, -0.5, 0.5, 0.0, -0.5, 0.5, 0.0] * 2
    lim = dev_limits(diffs, 20, n_compared=3)
    text = " ".join(lim["warnings"])
    assert lim["n"] == 20 and lim["indistinguishable"]
    assert "문항이 20개뿐" in text and "구분할 수 없다" in text and "다중 비교" in text


def test_stats_all_zero_and_clear_shift():
    z = dev_limits([0.0] * 20, 20)
    assert z["indistinguishable"] and any("차이가 관찰되지 않았다" in w for w in z["warnings"])
    c = dev_limits([0.3] * 20, 20)
    assert not c["indistinguishable"] and any("입증된 것은 아니다" in w for w in c["warnings"])
    assert dev_limits([], 20)["indistinguishable"]


def test_edit_count_cap_per_iteration(items, qs, tmp_path):
    prop = write_prop(tmp_path, [good_edit(f"e{i}", k=1 + i % 3) for i in range(10)])
    loop = make_loop(items, qs, tmp_path / "exp")
    loop.run([prop])
    st = [v["status"] for v in edits_of(loop.load_checkpoint()).values()]
    assert st.count("awaiting_approval") == 8 and st.count("invalid") == 2
