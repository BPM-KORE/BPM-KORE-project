import csv
import json

import pytest

from kore.blind import (
    export_blind, parse_scores, read_csv_rows, score_blind, weighted_kappa, write_key, write_sheet,
)
from kore.cli import main
from kore.generation import ExtractiveGenerator
from kore.ingest import append_source_log, clean_law_text, ingest_text, split_text
from kore.io import read_knowledge, write_knowledge
from kore.pipeline import run_experiment
from kore.schemas import EvalQuestion, KnowledgeItem
from kore.screen import domain_gate, screen_rows, screen_run_dir
from kore.stats import copy_risk, data_health, format_health, read_source_log
from tests.helpers import K_HEADER, mk_item, write_csv

LAW = """제1장 총칙
제1조(목적) 이 법은 근로조건의 기준을 정함으로써 근로자의 기본적 생활을 보장한다. <개정 2007. 4. 11.>
제2조(정의) ① 이 법에서 사용하는 용어의 뜻은 다음과 같다. <개정 2008. 3. 21.>
  1. "근로자"란 직업의 종류와 관계없이 임금을 목적으로 사업이나 사업장에 근로를 제공하는 사람을 말한다.
제3조에 따른 기준은 별도로 정한다. 이 줄은 앞 조문에 이어지는 본문이다.
제3조 제1항에 따른 경우에도 같다. 띄어 쓴 참조도 새 조문이 아니다.
제3조(근로조건의 기준) 이 법에서 정하는 근로조건은 최저기준이므로 근로 관계 당사자는 이 기준을 이유로 근로조건을 낮출 수 없다.
제4조 삭제 <2007. 4. 11.>
[전문개정 2008. 3. 21.]
제5조의2(예시 조문) 이 조는 번호가 건너뛰어도 새 조문으로 인식되는지 확인하려고 만든 예시 조문이다.
"""


# ---------------------------------------------------------------- 입력 도구


def test_article_mode_splits_by_article_and_cleans_revision_marks():
    r = ingest_text(LAW, "law", "law_t", "https://www.law.go.kr/", "저작권법 제7조", mode="article", title_prefix="근로기준법")
    titles = [i.title for i in r.items]
    assert titles == ["근로기준법 제1조(목적)", "근로기준법 제2조(정의)", "근로기준법 제3조(근로조건의 기준)", "근로기준법 제5조의2(예시 조문)"]
    assert [i.id for i in r.items] == ["law_t_001", "law_t_002", "law_t_003", "law_t_004"]
    assert all("<개정" not in i.text and "[전문개정" not in i.text and "총칙" not in i.text for i in r.items)
    assert ("근로기준법 제4조", "삭제된 조문") in r.dropped


def test_article_continuation_line_is_not_a_new_article():
    r = ingest_text(LAW, "law", "law_t", "u", "l", mode="article")
    second = next(i for i in r.items if "제2조(정의)" in i.title)
    assert "제3조에 따른 기준은 별도로 정한다" in second.text  # 제2조 본문에 이어 붙는다
    assert "띄어 쓴 참조도 새 조문이 아니다" in second.text
    assert len(r.items) == 4


def test_ids_continue_after_existing_ones_and_prefix_is_validated():
    r = ingest_text(LAW, "law", "law_t", "u", "l", mode="article", existing_ids={"law_t_007", "other_001"})
    assert r.items[0].id == "law_t_008"
    with pytest.raises(ValueError):
        ingest_text(LAW, "law", "bad prefix", "u", "l", mode="article")
    with pytest.raises(ValueError):
        ingest_text(LAW, "law", "p", "u", "l", mode="article", reliability=1.5)


def test_heading_and_paragraph_modes():
    doc = "# 첫째\n첫째 항목의 본문으로 충분히 긴 문장을 적어 둔다 여기에.\n## 둘째\n둘째 항목의 본문도 충분히 긴 문장으로 적어 둔다 여기에."
    r = ingest_text(doc, "wiki", "w", "u", "CC BY-SA 4.0", mode="heading")
    assert [i.title for i in r.items] == ["첫째", "둘째"]
    para = "짧음\n\n이 문단은 충분히 길어서 따로 문서가 되어야 하는 문단이다 확실하게.\n\n또 하나의 충분히 긴 문단이 여기에 있다 이것도 따로 문서가 된다."
    chunks = split_text(para, "paragraph", min_chars=30)
    assert len(chunks) == 2 and chunks[0].text.startswith("짧음")  # 짧은 문단은 다음 문단에 붙는다


def test_long_chunks_are_cut_at_line_boundaries():
    body = "\n".join(f"{n}번째 줄이다 " + "가" * 40 for n in range(1, 31))
    chunks = split_text("제1조(긴 조문) " + body, "article", max_chars=300)
    assert len(chunks) > 1 and all(len(c.text) <= 340 for c in chunks)
    assert chunks[0].title.endswith("(1/%d)" % len(chunks))


def test_clean_law_text_only_removes_marks():
    cleaned = clean_law_text("제1장 총칙\n제1조(목적) 내용이다. <개정 2020. 1. 1.>\n[본조신설 2020. 1. 1.]")
    assert cleaned == "제1조(목적) 내용이다."


def test_write_knowledge_append_keeps_existing_header_order(tmp_path):
    path = tmp_path / "k.csv"
    write_csv(path, ["title", "id", "domain", "text", "source_url", "license"],
              [["T", "a1", "d", "충분히 긴 본문이다 이것은 서른 글자를 넘기기 위한 문장이다 정말로.", "https://e.org", "CC0"]])
    item = KnowledgeItem("a2", "d", "T2", "또 다른 충분히 긴 본문이다 이것도 서른 글자를 넘기는 문장이다.", "https://e.org", "CC0")
    write_knowledge(path, [item], append=True)
    items, report = read_knowledge(path)
    assert report.ok and [i.id for i in items] == ["a1", "a2"] and items[1].title == "T2"
    bad = tmp_path / "bad.csv"
    write_csv(bad, ["id", "title"], [["x", "y"]])
    with pytest.raises(ValueError, match="필수 열"):
        write_knowledge(bad, [item], append=True)


def test_source_log_starts_unchecked_and_does_not_duplicate(tmp_path):
    items = [mk_item("a1", "t"), mk_item("a2", "t")]
    log = tmp_path / "source_log.csv"
    assert append_source_log(log, items, accessed_date="2026-10-23") == 2
    assert append_source_log(log, items, accessed_date="2026-10-23") == 0
    rows = read_source_log(log)
    assert rows["a1"]["license_checked"] == "no" and rows["a1"]["verified_by"] == "" and rows["a1"]["accessed_date"] == "2026-10-23"


def test_cli_ingest_end_to_end_and_dry_run(tmp_path, capsys):
    src = tmp_path / "law.txt"
    src.write_text(LAW, encoding="utf-8")
    out = tmp_path / "knowledge.csv"
    base = ["ingest", "--input", str(src), "--domain", "law", "--id-prefix", "law_t", "--source-url", "https://www.law.go.kr/",
            "--license", "저작권법 제7조", "--mode", "article", "--out", str(out), "--title-prefix", "근로기준법"]
    assert main(base + ["--dry-run"]) == 0 and not out.exists()
    assert main(base) == 0 and out.exists()
    assert main(base + ["--append"]) == 0  # 두 번째는 id를 이어서 매긴다
    items, report = read_knowledge(out)
    assert report.ok and len(items) == 8 and items[-1].id == "law_t_008"
    log = read_source_log(tmp_path / "source_log.csv")
    assert len(log) == 8 and all(v["license_checked"] == "no" for v in log.values())
    assert "license_checked" in capsys.readouterr().out


def test_cli_ingest_fails_when_nothing_found(tmp_path):
    src = tmp_path / "x.txt"
    src.write_text("조문이 하나도 없는 글이다 그냥 문장이다.", encoding="utf-8")
    assert main(["ingest", "--input", str(src), "--domain", "d", "--id-prefix", "p", "--source-url", "u", "--license", "l",
                 "--mode", "article", "--out", str(tmp_path / "k.csv")]) == 1


# ---------------------------------------------------------------- 데이터 점검


def synthetic(n_docs=160, n_q=100):
    domains = ("law", "guide", "wiki")
    items = [
        KnowledgeItem(f"{domains[i % 3]}_{i:03d}", domains[i % 3], f"T{i}", f"고유단어{i}x 에 관한 설명이다 " + "내용 " * 40,
                      "https://e.org/%d" % i, "CC-BY-4.0", reliability=0.8, lang="ko")
        for i in range(n_docs)
    ]
    qs = []
    for j in range(n_q):
        gold = [items[j].id] + ([items[(j + 50) % n_docs].id] if j % 3 == 0 else [])
        split = "test" if j % 4 == 0 else ("dev" if j % 4 == 1 else "train")
        diff = ("easy", "medium", "hard")[j % 3]
        qs.append(EvalQuestion(f"q{j:03d}", f"고유단어{j}x 는 무엇인가 알려 줘 {j}번", gold, items[j].domain, diff, split,
                               reference_answer="설명이다", rubric="설명"))
    return items, qs


def test_data_health_passes_a_well_formed_final_dataset():
    items, qs = synthetic()
    log = {i.id: {"license_checked": "yes"} for i in items}
    rep = data_health(items, qs, stage="final", source_log=log)
    failed = [c for c in rep["checks"] if c["level"] == "필수" and not c["ok"]]
    assert rep["ready"], failed
    assert "필수 항목을 모두 통과" in format_health(rep)


def test_data_health_flags_each_required_problem():
    items, qs = synthetic()
    rep = data_health(items[:40], qs[:20], stage="final")
    assert not rep["ready"] and {"문서 수", "질문 수"} <= set(rep["required_failed"])
    items2, qs2 = synthetic()
    items2[0] = items2[0].copy_with(license="unknown")
    rep2 = data_health(items2, qs2, stage="final", source_log={i.id: {"license_checked": "no"} for i in items2})
    assert {"라이선스 미확인 문서", "출처 기록(license_checked=yes)"} <= set(rep2["required_failed"])
    rep3 = data_health(items2, [], stage="week1")
    assert not rep3["ready"]


def test_data_health_stage_thresholds_and_invalid_stage():
    items, qs = synthetic(n_docs=60, n_q=35)
    assert data_health(items, qs, stage="week1")["ready"]
    assert not data_health(items, qs, stage="week2")["ready"]
    with pytest.raises(ValueError):
        data_health(items, qs, stage="nope")


def test_copy_risk_detects_copied_sentences():
    item = KnowledgeItem("a", "d", "t", "목성은 태양계에서 가장 큰 행성이며 지름은 약 14만 킬로미터이다", "u", "l")
    copied = EvalQuestion("q1", "목성은 태양계에서 가장 큰 행성이며", ["a"], "d")
    rephrased = EvalQuestion("q2", "우리 태양계 안에서 덩치가 제일 큰 천체는 어느 것인가", ["a"], "d")
    assert copy_risk(copied, {"a": item}) > 0.9 and copy_risk(rephrased, {"a": item}) < 0.2


# ---------------------------------------------------------------- 닫힌북 필터와 게이트


def rows_for(qid, domain, nr, gc, recall=1.0):
    base = {"question_id": qid, "domain": domain, "split": "dev", "candidate_recall": recall}
    return [dict(base, condition="no_retrieval", rubric_score=nr), dict(base, condition="gold_context", rubric_score=gc)]


def test_screen_flags_and_domain_gate():
    rows = []
    for i in range(6):
        rows += rows_for(f"a{i}", "law", 0.0, 1.0)  # 좋은 분야
    for i in range(6):
        rows += rows_for(f"b{i}", "wiki", 1.0, 1.0)  # 모델이 이미 아는 분야
    for i in range(6):
        rows += rows_for(f"c{i}", "guide", 0.0, 0.0, recall=0.0)  # 근거를 줘도 못 맞히고 검색도 못 함
    rows += rows_for("z", "tiny", 0.0, 1.0)
    screened = screen_rows(rows)
    flags = {d["question_id"]: d["flags"] for d in screened}
    assert flags["a0"] == "" and "모델이_이미_안다" in flags["b0"]
    assert "정답근거를_줘도_못맞힘" in flags["c0"] and "후보에_정답근거_없음" in flags["c0"]
    gate = {g["domain"]: g for g in domain_gate(screened)}
    assert gate["law"]["verdict"] == "PASS" and gate["wiki"]["verdict"] == "FAIL" and gate["guide"]["verdict"] == "FAIL"
    assert gate["tiny"]["verdict"].startswith("판정 보류")
    assert any("이미 안다" in r for r in gate["wiki"]["reasons"])


def test_domain_gate_flags_no_headroom_when_vanilla_already_matches_gold():
    rows = []
    for i in range(6):
        r = rows_for(f"h{i}", "flat", 0.0, 1.0)
        r.append(dict(r[0], condition="vanilla_rag", rubric_score=1.0))  # 유사도 상위 k가 이미 정답 근거 수준
        rows += r
    for i in range(6):
        r = rows_for(f"g{i}", "room", 0.0, 1.0)
        r.append(dict(r[0], condition="vanilla_rag", rubric_score=0.5))
        rows += r
    gate = {g["domain"]: g for g in domain_gate(screen_rows(rows))}
    assert gate["flat"]["verdict"] == "FAIL" and gate["flat"]["headroom"] == pytest.approx(0.0)
    assert any("개선 여지" in r for r in gate["flat"]["reasons"])
    assert gate["room"]["verdict"] == "PASS" and gate["room"]["headroom"] == pytest.approx(0.5)
    # vanilla_rag를 안 돌렸으면 개선 여지는 판정에 쓰지 않는다
    assert {g["domain"]: g for g in domain_gate(screen_rows(rows_for("x0", "n", 0.0, 1.0) * 1))}["n"]["headroom"] is None


def test_screen_run_dir_errors_and_end_to_end(tmp_path, items, questions):
    with pytest.raises(ValueError, match="결과가 없다"):
        screen_run_dir(tmp_path)
    out = tmp_path / "run"
    run_experiment(items, questions, out, conditions=("vanilla_rag",), generator=ExtractiveGenerator())
    with pytest.raises(ValueError, match="no_retrieval"):
        screen_run_dir(out)
    out2 = tmp_path / "run2"
    run_experiment(items, questions, out2, conditions=("no_retrieval", "gold_context"), generator=ExtractiveGenerator())
    res = screen_run_dir(out2)
    assert res["screened"] and res["gate"]
    gold_rows = [r for r in map(json.loads, (out2 / "run_log.jsonl").read_text(encoding="utf-8").splitlines()) if r["condition"] == "gold_context"]
    assert all(r["recall"] == 1.0 for r in gold_rows)  # 정답 근거를 그대로 주므로 재현율이 1이다


def test_cli_screen_writes_csv(tmp_path, capsys):
    out = tmp_path / "r"
    assert main(["run", "--knowledge", "data/sample_knowledge.csv", "--questions", "data/sample_eval.csv", "--out", str(out),
                 "--conditions", "no_retrieval", "gold_context", "--generator", "extractive"]) == 0
    assert main(["screen", "--run-dir", str(out)]) == 0
    assert (out / "screen.csv").exists() and "분야 게이트" in capsys.readouterr().out


# ---------------------------------------------------------------- 블라인드 채점


def answers_log(n=8):
    rows = []
    for i in range(n):
        for cond, ans in (("vanilla_rag", f"답 {i}"), ("kore", f"답 {i}" if i % 2 == 0 else f"다른 답 {i}")):
            rows.append({"question_id": f"q{i}", "condition": cond, "answer": ans, "rubric_score": 0.5 if cond == "kore" else 1.0})
    return rows


def blind_questions(n=8, split="dev"):
    return [EvalQuestion(f"q{i}", f"질문 {i}", ["a"], "d", "easy", split, reference_answer="정답", rubric="정답") for i in range(n)]


def test_blind_export_hides_conditions_merges_identical_answers_and_is_deterministic():
    rows, qs = answers_log(), blind_questions()
    sheet, key = export_blind(rows, qs, ["vanilla_rag", "kore"], n=8, seed=1)
    assert len(sheet) == 12  # 8개 질문 중 4개는 두 조건의 답이 같아 한 줄로 합쳐진다
    flat = json.dumps(sheet, ensure_ascii=False)
    assert "vanilla_rag" not in flat and "kore" not in flat
    merged = [k for k in key if ";" in k["conditions"]]
    assert len(merged) == 4
    again = export_blind(rows, qs, ["vanilla_rag", "kore"], n=8, seed=1)
    assert again == (sheet, key)
    assert [s["item_id"] for s in sheet] == [f"b{i:03d}" for i in range(1, 13)]
    other = export_blind(rows, qs, ["vanilla_rag", "kore"], n=8, seed=2)[0]
    assert [s["answer"] for s in other] != [s["answer"] for s in sheet]


def test_blind_export_excludes_test_split_unless_final_and_needs_answers():
    rows = answers_log()
    with pytest.raises(ValueError, match="test"):
        export_blind(rows, blind_questions(split="test"), ["vanilla_rag", "kore"])
    sheet, _ = export_blind(rows, blind_questions(split="test"), ["vanilla_rag", "kore"], include_test=True)
    assert sheet
    with pytest.raises(ValueError, match="채점할 답변이 없다"):
        export_blind([], blind_questions(), ["kore"])
    no_answers = [dict(r, answer=None) for r in rows]
    with pytest.raises(ValueError):
        export_blind(no_answers, blind_questions(), ["kore"])


def test_parse_scores_rejects_blank_and_out_of_range():
    scores, problems = parse_scores([
        {"item_id": "b1", "score": "2"}, {"item_id": "b2", "score": ""}, {"item_id": "b3", "score": "3"},
        {"item_id": "b4", "score": "x"}, {"item_id": "b5", "score": "1.5"}, {"item_id": "b6", "score": "0"},
    ])
    assert scores == {"b1": 2, "b6": 0} and len(problems) == 4


def test_weighted_kappa_known_values():
    assert weighted_kappa([0, 1, 2, 2, 1, 0], [0, 1, 2, 2, 1, 0]) == pytest.approx(1.0)
    assert weighted_kappa([0, 0, 0, 0], [0, 0, 0, 0]) is None  # 한 가지 값뿐이면 정의되지 않는다
    k = weighted_kappa([0, 0, 2, 2], [2, 2, 0, 0])
    assert k < -0.9  # 정반대로 채점하면 -1에 가깝다
    assert -1.0 <= weighted_kappa([0, 1, 2, 1], [0, 2, 2, 1]) <= 1.0


def test_blind_round_trip_through_files_and_scoring(tmp_path):
    rows, qs = answers_log(), blind_questions()
    sheet, key = export_blind(rows, qs, ["vanilla_rag", "kore"], n=8, seed=0)
    write_sheet(tmp_path / "sheet.csv", sheet)
    write_key(tmp_path / "key.csv", key)
    filled = read_csv_rows(tmp_path / "sheet.csv")
    key_rows = read_csv_rows(tmp_path / "key.csv")
    kmap = {k["item_id"]: k for k in key_rows}
    for r in filled:  # 사람이 채점했다고 치고, '다른 답'은 0점, 나머지는 2점
        r["score"] = "0" if r["answer"].startswith("다른") else "2"
    second = [dict(r) for r in filled]
    second[0]["score"] = "1"
    res = score_blind([filled, second], key_rows)
    assert res["n_items"] == 12 and res["baseline"] == "vanilla_rag"
    assert res["by_condition"]["vanilla_rag"]["mean"] > res["by_condition"]["kore"]["mean"]
    assert res["paired"]["kore"]["mean_diff"] < 0
    assert 0.0 <= res["grader_agreement"]["exact_rate"] <= 1.0 and "auto_vs_human" in res
    assert all(kmap[i]["item_id"] == i for i in kmap)


def test_score_blind_validates_inputs():
    key = [{"item_id": "b001", "question_id": "q", "domain": "d", "conditions": "kore", "rubric_score": ""}]
    with pytest.raises(ValueError, match="열쇠에 없는"):
        score_blind([[{"item_id": "zzz", "score": "1"}]], key)
    with pytest.raises(ValueError, match="채점표는"):
        score_blind([], key)
    with pytest.raises(ValueError, match="점수를 매긴 항목이 없다"):
        score_blind([[{"item_id": "b001", "score": ""}]], key)


def test_cli_blind_round_trip(tmp_path, capsys):
    run = tmp_path / "run"
    assert main(["run", "--knowledge", "data/sample_knowledge.csv", "--questions", "data/sample_eval.csv", "--out", str(run),
                 "--conditions", "vanilla_rag", "kore", "--generator", "extractive"]) == 0
    out = tmp_path / "blind"
    assert main(["blind-export", "--knowledge", "data/sample_knowledge.csv", "--questions", "data/sample_eval.csv",
                 "--run-dir", str(run), "--out-dir", str(out), "--n", "5"]) == 0
    sheet = read_csv_rows(out / "blind_sheet.csv")
    with (out / "filled.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sheet[0]))
        w.writeheader()
        for r in sheet:
            r["score"] = "2"
            w.writerow(r)
    assert main(["blind-score", "--sheet", str(out / "filled.csv"), "--key", str(out / "blind_key.csv")]) == 0
    assert "조건별 사람 점수" in capsys.readouterr().out
