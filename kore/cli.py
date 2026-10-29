"""명령줄 도구.

validate, stats, ingest, split, run, screen, tune, label, label-report, blind-export, blind-score, search, doctor
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from datetime import date
from pathlib import Path

from kore.blind import (
    export_blind, format_blind_report, parse_scores, read_csv_rows, score_blind, write_key, write_sheet,
)
from kore.doctor import check_environment, format_environment
from kore.generation import GeneratorSpec, load_generator
from kore.ingest import MODES, append_source_log, ingest_text
from kore.io import ValidationReport, read_knowledge, read_questions, write_knowledge, write_questions
from kore.knowledge_graph import read_relations
from kore.labeling import analyze_labels, count_calls, format_report, run_labeling
from kore.pipeline import BASE_CONDITIONS, PipelineConfig, _read_log, all_conditions, load_config, run_experiment
from kore.pipeline import build_retriever
from kore.retrieval_eval import compare_retrievers, format_comparison
from kore.screen import format_gate, screen_run_dir, write_screen_csv
from kore.selector import select_knowledge
from kore.splits import assign_splits
from kore.stats import STAGES, data_health, format_health, read_source_log
from kore.tuning import grid_search_weights

MAX_SHOWN = 40


def _print_report(label: str, report: ValidationReport) -> None:
    status = "통과" if report.ok else "실패"
    print(f"[{label}] {status}: 오류 {len(report.errors)}건, 경고 {len(report.warnings)}건")
    for msg in report.errors[:MAX_SHOWN]:
        print(f"  오류  {msg}")
    for msg in report.warnings[:MAX_SHOWN]:
        print(f"  경고  {msg}")
    hidden = len(report.errors) + len(report.warnings) - 2 * MAX_SHOWN
    if len(report.errors) > MAX_SHOWN or len(report.warnings) > MAX_SHOWN:
        print(f"  ... 일부 생략({max(hidden, 0)}건 이상)")


def _load_both(knowledge: str, questions: str | None):
    items, k_report = read_knowledge(knowledge)
    qs, q_report = read_questions(questions, items) if questions else ([], ValidationReport())
    ok = k_report.ok and q_report.ok
    if not ok:
        _print_report("knowledge", k_report)
        if questions:
            _print_report("evaluation", q_report)
        print("데이터 검증에 실패했다. 먼저 `python -m kore validate`로 오류를 고친다.")
    return items, qs, ok


def _generator_from_args(args: argparse.Namespace):
    """--generator-config와 --generator/--model에서 생성기를 만든다. 아무것도 안 주면 생성하지 않는다(none)."""
    if args.generator_config:
        spec = GeneratorSpec.from_json(args.generator_config)
    else:
        spec = GeneratorSpec(kind="none")
    if args.generator:
        spec = replace(spec, kind=args.generator)
    if args.model:
        spec = replace(spec, model=args.model)
    spec.validate()
    return load_generator(spec), spec


# ---------------------------------------------------------------- 명령


def cmd_validate(args: argparse.Namespace) -> int:
    items, k_report = read_knowledge(args.knowledge)
    _print_report("knowledge", k_report)
    code = 0 if k_report.ok else 1
    if items:
        domains = Counter(i.domain for i in items)
        print(f"  문서 {len(items)}개, 분야 {dict(domains)}")
    if args.questions:
        questions, q_report = read_questions(args.questions, items if items else None)
        _print_report("evaluation", q_report)
        code = code or (0 if q_report.ok else 1)
        if questions:
            by_split = Counter(q.split for q in questions)
            print(f"  질문 {len(questions)}개, 분할 {dict(by_split)}")
    if args.relations:
        relations, r_report = read_relations(args.relations, items if items else None)
        _print_report("relations", r_report)
        code = code or (0 if r_report.ok else 1)
        if relations:
            print(f"  관계 {len(relations)}개")
    return code


def cmd_stats(args: argparse.Namespace) -> int:
    items, questions, ok = _load_both(args.knowledge, args.questions)
    if not ok:
        return 1
    log_path = args.source_log or str(Path(args.knowledge).with_name("source_log.csv"))
    source_log = read_source_log(log_path) if Path(log_path).exists() else None
    report = data_health(items, questions, stage=args.stage, source_log=source_log)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(format_health(report))
        if source_log is None:
            print(f"\n(출처 기록 {log_path}가 없어서 출처 확인 항목은 건너뛰었다)")
    return 1 if (args.strict and not report["ready"]) else 0


def cmd_ingest(args: argparse.Namespace) -> int:
    text = Path(args.input).read_text(encoding="utf-8-sig")
    out = Path(args.out)
    existing: set[str] = set()
    if args.append and out.exists() and out.stat().st_size > 0:
        old, old_report = read_knowledge(out)
        if not old_report.ok:
            _print_report("기존 knowledge", old_report)
            print("기존 파일에 오류가 있어 이어 쓰지 않는다. 먼저 고친다.")
            return 1
        existing = {i.id for i in old}
    result = ingest_text(
        text, domain=args.domain, id_prefix=args.id_prefix, source_url=args.source_url, license=args.license,
        mode=args.mode, title_prefix=args.title_prefix, reliability=args.reliability, lang=args.lang,
        min_chars=args.min_chars, max_chars=args.max_chars, existing_ids=existing,
    )
    print(f"문서 {len(result.items)}개 생성, {len(result.dropped)}개 제외")
    for label, why in result.dropped[:20]:
        print(f"  제외  {label}: {why}")
    for note in result.notes:
        print(f"  참고  {note}")
    for it in result.items[:3]:
        print(f"  예시  {it.id} | {it.title} | {it.text[:50].replace(chr(10), ' ')}...")
    if not result.items:
        print("만들 문서가 없어서 파일을 쓰지 않았다")
        return 1
    if args.dry_run:
        print("dry-run: 파일을 쓰지 않았다")
        return 0
    write_knowledge(out, result.items, append=args.append)
    log_path = args.source_log or str(out.with_name("source_log.csv"))
    n_log = append_source_log(log_path, result.items, accessed_date=args.accessed_date, note=args.note)
    print(f"{out}에 썼다. 출처 기록 {log_path}에 {n_log}줄을 더했다 (license_checked=no).")
    print("다음: 원문 페이지에서 이용 조건을 직접 읽고, 확인한 줄의 license_checked를 yes로, verified_by를 채운다.")
    items, report = read_knowledge(out)
    _print_report("knowledge", report)
    return 0 if report.ok else 1


def cmd_split(args: argparse.Namespace) -> int:
    items, k_report = read_knowledge(args.knowledge) if args.knowledge else ([], ValidationReport())
    questions, q_report = read_questions(args.questions, items or None)
    if not q_report.ok:
        _print_report("evaluation", q_report)
        return 1
    fixed = None
    if args.keep_from:
        if not Path(args.keep_from).exists():
            print(f"--keep-from 파일이 없다: {args.keep_from}")
            return 1
        prev, prev_report = read_questions(args.keep_from)
        if not prev_report.ok:
            _print_report("keep-from", prev_report)
            return 1
        fixed = {q.question_id: q.split for q in prev}
    elif Path(args.out).exists() and not args.reshuffle:
        print(f"{args.out}이(가) 이미 있다. 질문을 더 만들어서 다시 나누는 거라면 이전 분할을 유지하도록\n"
              f"  --keep-from {args.out}\n"
              "을 붙인다. 처음부터 다시 섞는 거라면(아직 튜닝이나 라벨링을 안 했을 때만) --reshuffle을 붙인다.")
        return 1
    new = assign_splits(questions, seed=args.seed, ratios=tuple(args.ratios), fixed=fixed)
    write_questions(args.out, new)
    kept = sum(q.question_id in fixed for q in new) if fixed else 0
    print(f"분할 {dict(Counter(q.split for q in new))} -> {args.out} (seed={args.seed}, 비율={tuple(args.ratios)})")
    if fixed:
        print(f"이전 분할을 유지한 질문 {kept}개, 새로 배정한 질문 {len(new) - kept}개")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    items, questions, ok = _load_both(args.knowledge, args.questions)
    if not ok:
        print("데이터 검증에 실패해서 실험을 시작하지 않는다.")
        return 1
    config, run_level = (load_config(args.config) if args.config else (PipelineConfig(), {}))
    # 명령줄 값은 설정 파일보다 우선한다. 값을 주지 않은 항목만 파일(없으면 기본값)을 따른다.
    overrides = {k: v for k, v in (("k", args.k), ("candidate_k", args.candidate_k), ("token_budget", args.token_budget),
                                   ("retriever", args.retriever)) if v is not None}
    config = replace(config, **overrides)
    conditions = tuple(args.conditions or run_level.get("conditions") or BASE_CONDITIONS)
    seed = args.seed if args.seed is not None else run_level.get("seed", 0)
    generator, spec = _generator_from_args(args)
    if generator is not None:
        print(f"생성기: {generator.name}")
    result = run_experiment(
        items, questions, args.out, conditions=conditions, config=config, generator=generator,
        seed=seed, include_test=args.final_test, resume=not args.fresh,
        progress=lambda n, total: print(f"  {n}/{total}", end="\r", flush=True) if n % 5 == 0 or n == total else None,
    )
    print(f"완료: {result['n_rows']}행 -> {args.out}      ")
    for row in result["results"]:
        if row["split"] == "all":
            rub = "-" if row.get("rubric_score") is None else f"{row['rubric_score']:.3f}"
            print(f"  {row['condition']:<24} n={row['n']:<4} recall={row['recall']:.3f} hit={row['hit']:.3f} rubric={rub} tokens={row['context_tokens']:.0f}")
    if args.final_test:
        print("주의: test 분할을 포함했다. 이 결과를 보고 알고리즘을 고치지 않는다.")
    return 0


def cmd_screen(args: argparse.Namespace) -> int:
    res = screen_run_dir(args.run_dir, known=args.known, gold_low=args.gold_low)
    out = Path(args.out) if args.out else Path(args.run_dir) / "screen.csv"
    write_screen_csv(out, res["screened"])
    print(format_gate(res["gate"]))
    flagged = [d for d in res["screened"] if d["flags"]]
    print(f"\n표시된 질문 {len(flagged)}개 / {len(res['screened'])}개 -> {out}")
    print("모델이_이미_안다: 질문을 바꾸거나 빼고, 정답근거를_줘도_못맞힘: 질문·루브릭을 고치고, 후보에_정답근거_없음: 검색(토크나이저, 질문 표현)부터 본다.")
    return 0


def cmd_label(args: argparse.Namespace) -> int:
    items, questions, ok = _load_both(args.knowledge, args.questions)
    if not ok:
        return 1
    config, _ = (load_config(args.config) if args.config else (PipelineConfig(), {}))
    splits = tuple(args.splits)
    scope = [q for q in questions if q.split in splits and q.reference_answer.strip()]
    per_q = count_calls(args.max_cands, args.max_subset)
    print(f"대상 질문 {len(scope)}개, 질문당 {per_q}번, 총 {per_q * len(scope)}번 점수 계산")
    missing_ref = [q.question_id for q in questions if q.split in splits and not q.reference_answer.strip()]
    if missing_ref:
        print(f"reference_answer가 없어 제외되는 질문 {len(missing_ref)}개: {missing_ref[:5]}")
    if args.sec_per_call:
        total = per_q * len(scope) * args.sec_per_call
        print(f"호출당 {args.sec_per_call}초라면 약 {total / 60:.0f}분")
    if args.dry_run:
        return 0
    generator, spec = _generator_from_args(args)
    if generator is None or not hasattr(generator, "score"):
        print("점수를 내려면 로그확률을 계산하는 모델이 필요하다: --generator hf (또는 --generator-config configs/generator.json)")
        return 1
    result = run_labeling(
        items, questions, args.out, generator, config=config, splits=splits, max_cands=args.max_cands,
        max_subset=args.max_subset, resume=not args.fresh, limit=args.limit,
        progress=lambda n, total: print(f"  {n}/{total}", end="\r", flush=True),
    )
    print(f"라벨링 {result['n_labeled']}개 질문 -> {args.out}/dq_log.jsonl      ")
    return 0


def cmd_label_report(args: argparse.Namespace) -> int:
    rows = _read_log(Path(args.dir) / "dq_log.jsonl")
    if not rows:
        print(f"{args.dir}/dq_log.jsonl에 라벨이 없다")
        return 1
    analysis = analyze_labels(rows, k=args.k, seed=args.seed)
    print(format_report(analysis))
    if args.csv:
        import csv as _csv

        per_q = analysis["per_question"]
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as fh:
            writer = _csv.DictWriter(fh, fieldnames=list(per_q[0]))
            writer.writeheader()
            writer.writerows(per_q)
        print(f"\n질문별 값 -> {args.csv}")
    return 0


def cmd_tune(args: argparse.Namespace) -> int:
    items, questions, ok = _load_both(args.knowledge, args.questions)
    if not ok:
        return 1
    config, _ = (load_config(args.config) if args.config else (PipelineConfig(), {}))
    res = grid_search_weights(items, questions, config=config, splits=tuple(args.splits))
    ranked = sorted(res["table"], key=lambda row: -row["recall"])
    print(f"질문 {res['n_questions']}개({','.join(res['splits'])}), 유사도 기준선 Recall {res['baseline_similarity_recall']:.3f}")
    print("상위 조합:")
    for row in ranked[: args.top]:
        w = row["weights"]
        print(f"  recall {row['recall']:.3f}  coverage={w['coverage']} diversity={w['diversity']} redundancy={w['redundancy']} reliability={w['reliability']}")
    if res["warning"]:
        print("주의:", res["warning"])
    print("목적함수가 근거 문서 Recall이라서 답변 품질을 직접 최적화한 값이 아니다. 고른 조합과 시도한 조합 수(%d)를 보고서에 함께 적는다." % len(res["table"]))
    if args.out:
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"전체 표 -> {args.out}")
    return 0


def cmd_blind_export(args: argparse.Namespace) -> int:
    items, questions, ok = _load_both(args.knowledge, args.questions)
    if not ok:
        return 1
    rows = _read_log(Path(args.run_dir) / "run_log.jsonl")
    sheet, key = export_blind(
        rows, questions, list(args.conditions), n=args.n, seed=args.seed, include_test=args.final_test
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_sheet(out / "blind_sheet.csv", sheet)
    write_key(out / "blind_key.csv", key)
    print(f"채점할 답변 {len(sheet)}개 -> {out}/blind_sheet.csv")
    print(f"열쇠 -> {out}/blind_key.csv (채점이 끝나기 전에는 열지 않는다)")
    print("채점자는 blind_sheet.csv의 score 열에 0(틀림) 1(일부 맞음) 2(정답)만 적는다. 채점자 2명이 각자 복사본으로 따로 채점한다.")
    return 0


def cmd_blind_score(args: argparse.Namespace) -> int:
    sheets = [read_csv_rows(p) for p in args.sheet]
    key = read_csv_rows(args.key)
    for p, s in zip(args.sheet, sheets):
        _, problems = parse_scores(s)
        if problems:
            print(f"{p}: 채점 문제 {len(problems)}건")
            for line in problems[:10]:
                print(f"  {line}")
    res = score_blind(sheets, key, seed=args.seed, baseline=args.baseline)
    print(format_blind_report(res))
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    items, report = read_knowledge(args.knowledge)
    if not report.ok:
        _print_report("knowledge", report)
        return 1
    config, _ = (load_config(args.config) if args.config else (PipelineConfig(), {}))
    if args.retriever:
        config = replace(config, retriever=args.retriever)
    retriever = build_retriever(items, config)
    cands = retriever.retrieve(args.question, k=args.candidate_k)
    picked = select_knowledge(args.question, cands, k=args.k, strategy=args.strategy,
                              tokenizer_mode=config.tokenizer_mode, term_weights=retriever.term_weights(args.question))
    print(f"후보 {len(cands)}개 중 {args.strategy}로 {len(picked)}개 선택")
    for p in picked:
        print(f"  {p.id:<12} [{p.domain}] {p.selection_score:.3f}  {p.title}")
    return 0


def cmd_retrieval_eval(args: argparse.Namespace) -> int:
    items, questions, ok = _load_both(args.knowledge, args.questions)
    if not ok:
        return 1
    config, _ = (load_config(args.config) if args.config else (PipelineConfig(), {}))
    try:
        res = compare_retrievers(items, questions, args.compare, base=config, splits=tuple(args.splits))
    except (ValueError, ImportError) as exc:
        print(f"비교하지 못했다: {exc}")
        return 1
    print(format_comparison(res))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        slim = {k: v for k, v in res.items() if k != "per_question"}
        Path(args.out).write_text(json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"요약 -> {args.out}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    rows = check_environment(args.data_dir)
    print(format_environment(rows))
    return 1 if any(s == "실패" for s, _, _ in rows) else 0


# ---------------------------------------------------------------- 인자


def _add_generator_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--generator", choices=["none", "extractive", "hf"], help="생성기 종류(기본: 설정 파일의 kind, 파일이 없으면 none)")
    p.add_argument("--generator-config", help="생성기 JSON 설정(예: configs/generator.json)")
    p.add_argument("--model", help="설정 파일의 모델 이름을 덮어쓴다(예: Qwen/Qwen2.5-3B-Instruct)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kore", description="KORE 연구 프로토타입 도구")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="knowledge/evaluation 파일 계약 검증")
    v.add_argument("--knowledge", required=True)
    v.add_argument("--questions")
    v.add_argument("--relations", help="relations.csv(선택)")
    v.set_defaults(func=cmd_validate)

    st = sub.add_parser("stats", help="데이터 건강검진(규모, 균형, 후보 재현율, 베끼기, 출처 확인)")
    st.add_argument("--knowledge", required=True)
    st.add_argument("--questions")
    st.add_argument("--stage", choices=sorted(STAGES), default="final", help="목표 단계(week1/week2/final)")
    st.add_argument("--source-log", help="출처 기록 CSV(기본: knowledge와 같은 폴더의 source_log.csv)")
    st.add_argument("--strict", action="store_true", help="필수 항목이 하나라도 실패하면 종료 코드 1")
    st.add_argument("--json", action="store_true")
    st.set_defaults(func=cmd_stats)

    ing = sub.add_parser("ingest", help="원문 텍스트를 문서로 잘라 knowledge.csv에 더한다(B)")
    ing.add_argument("--input", required=True, help="원문을 복사해 붙인 txt 파일")
    ing.add_argument("--domain", required=True)
    ing.add_argument("--id-prefix", required=True, help="예: law_labor → law_labor_001, law_labor_002 ...")
    ing.add_argument("--source-url", required=True)
    ing.add_argument("--license", required=True, help="확인한 이용 조건을 적는다. 모르면 unknown")
    ing.add_argument("--mode", choices=MODES, default="paragraph")
    ing.add_argument("--title-prefix", default="", help="제목 앞에 붙일 이름(예: 근로기준법)")
    ing.add_argument("--reliability", type=float, default=0.5)
    ing.add_argument("--lang", default="ko")
    ing.add_argument("--min-chars", type=int, default=30)
    ing.add_argument("--max-chars", type=int, default=1500)
    ing.add_argument("--out", default="data/knowledge.csv")
    ing.add_argument("--append", action="store_true", help="이미 있는 파일 뒤에 더한다(id는 이어서 매긴다)")
    ing.add_argument("--source-log", help="출처 기록 CSV(기본: out과 같은 폴더의 source_log.csv)")
    ing.add_argument("--accessed-date", default=date.today().isoformat())
    ing.add_argument("--note", default="")
    ing.add_argument("--dry-run", action="store_true", help="파일을 쓰지 않고 어떻게 잘리는지만 본다")
    ing.set_defaults(func=cmd_ingest)

    r = sub.add_parser("run", help="no_retrieval/vanilla_rag/kore 비교 실험")
    r.add_argument("--knowledge", required=True)
    r.add_argument("--questions", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--config", help="JSON 설정 파일(예: configs/default.json). 아래 옵션이 있으면 그 값이 우선한다")
    r.add_argument("--retriever", choices=["bm25", "hybrid"], help="후보 검색기(기본: 설정 파일, 없으면 bm25)")
    r.add_argument("--k", type=int, default=None)
    r.add_argument("--candidate-k", type=int, default=None)
    r.add_argument("--token-budget", type=int, default=None)
    r.add_argument("--seed", type=int, default=None)
    r.add_argument("--conditions", nargs="*", choices=all_conditions())
    r.add_argument("--final-test", action="store_true", help="test 분할을 포함한다(최종 평가에서만)")
    r.add_argument("--fresh", action="store_true", help="이어서 하지 않고 out 폴더를 새로 시작한다")
    _add_generator_args(r)
    r.set_defaults(func=cmd_run)

    sc = sub.add_parser("screen", help="닫힌북 필터와 분야 게이트(run 결과 폴더가 필요)")
    sc.add_argument("--run-dir", required=True)
    sc.add_argument("--known", type=float, default=0.67, help="no_retrieval 점수가 이 이상이면 '이미 안다'")
    sc.add_argument("--gold-low", type=float, default=0.5, help="gold_context 점수가 이 미만이면 문제 질문")
    sc.add_argument("--out", help="표시된 질문 목록 CSV(기본: run-dir/screen.csv)")
    sc.set_defaults(func=cmd_screen)

    lb = sub.add_parser("label", help="문서 기여도(ΔQ) 라벨링: 부분집합별 정답 로그확률")
    lb.add_argument("--knowledge", required=True)
    lb.add_argument("--questions", required=True)
    lb.add_argument("--out", required=True)
    lb.add_argument("--config", help="파이프라인 설정(후보 검색에 쓴다)")
    lb.add_argument("--splits", nargs="+", choices=["train", "dev"], default=["train", "dev"])
    lb.add_argument("--max-cands", type=int, default=8)
    lb.add_argument("--max-subset", type=int, default=3)
    lb.add_argument("--limit", type=int, default=None, help="앞에서부터 이 개수만(시험용)")
    lb.add_argument("--sec-per-call", type=float, default=None, help="호출당 초(스모크 테스트로 잰 값)를 주면 걸리는 시간을 어림한다")
    lb.add_argument("--dry-run", action="store_true", help="모델을 올리지 않고 계산 횟수만 센다")
    lb.add_argument("--fresh", action="store_true")
    _add_generator_args(lb)
    lb.set_defaults(func=cmd_label)

    lr = sub.add_parser("label-report", help="라벨로 H1·H2 예비 지표를 계산한다")
    lr.add_argument("--dir", required=True, help="label 결과 폴더")
    lr.add_argument("--k", type=int, default=3)
    lr.add_argument("--seed", type=int, default=0)
    lr.add_argument("--csv", help="질문별 지표를 저장할 CSV")
    lr.set_defaults(func=cmd_label_report)

    tu = sub.add_parser("tune", help="선택기 가중치 격자 탐색(train/dev만, test는 거부)")
    tu.add_argument("--knowledge", required=True)
    tu.add_argument("--questions", required=True)
    tu.add_argument("--config", help="파이프라인 설정")
    tu.add_argument("--splits", nargs="+", choices=["train", "dev"], default=["train", "dev"])
    tu.add_argument("--top", type=int, default=5)
    tu.add_argument("--out", help="전체 표를 저장할 JSON")
    tu.set_defaults(func=cmd_tune)

    be = sub.add_parser("blind-export", help="블라인드 채점표와 열쇠를 만든다(C)")
    be.add_argument("--knowledge", required=True)
    be.add_argument("--questions", required=True)
    be.add_argument("--run-dir", required=True)
    be.add_argument("--conditions", nargs="+", default=["vanilla_rag", "kore"])
    be.add_argument("--n", type=int, default=30)
    be.add_argument("--seed", type=int, default=0)
    be.add_argument("--final-test", action="store_true", help="test 분할 질문도 뽑는다(최종 평가에서만)")
    be.add_argument("--out-dir", required=True)
    be.set_defaults(func=cmd_blind_export)

    bs = sub.add_parser("blind-score", help="채워진 채점표로 사람 점수를 계산한다(C)")
    bs.add_argument("--sheet", nargs="+", required=True, help="채점표 1개 또는 2개(채점자 2명)")
    bs.add_argument("--key", required=True)
    bs.add_argument("--baseline", default=None)
    bs.add_argument("--seed", type=int, default=0)
    bs.set_defaults(func=cmd_blind_score)

    s = sub.add_parser("search", help="질문 하나로 검색·선택 결과 확인")
    s.add_argument("question")
    s.add_argument("--knowledge", required=True)
    s.add_argument("--k", type=int, default=3)
    s.add_argument("--candidate-k", type=int, default=10)
    s.add_argument("--strategy", choices=["similarity", "kore"], default="kore")
    s.add_argument("--config", help="파이프라인 설정")
    s.add_argument("--retriever", choices=["bm25", "hybrid"], help="설정의 검색기를 덮어쓴다(hybrid는 torch 필요)")
    s.set_defaults(func=cmd_search)

    re_ = sub.add_parser("retrieval-eval", help="검색기 설정 비교(train·dev): 정답 근거가 후보에 들어오는가")
    re_.add_argument("--knowledge", required=True)
    re_.add_argument("--questions", required=True)
    re_.add_argument("--config", help="기준 파이프라인 설정")
    re_.add_argument("--compare", nargs="+", default=["bm25:ngram", "bm25:ngram_stem"],
                     help="검색기[:토크나이저[:alpha]]. 첫 번째가 기준. 예: bm25:ngram_stem hybrid:ngram_stem:0.5")
    re_.add_argument("--splits", nargs="+", choices=["train", "dev"], default=["train", "dev"])
    re_.add_argument("--out", help="요약 JSON")
    re_.set_defaults(func=cmd_retrieval_eval)

    sp = sub.add_parser("split", help="질문을 분야별로 층화해서 train/dev/test 배정")
    sp.add_argument("--questions", required=True)
    sp.add_argument("--knowledge")
    sp.add_argument("--out", required=True)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--ratios", type=float, nargs=3, default=[0.5, 0.2, 0.3], metavar=("TRAIN", "DEV", "TEST"),
                    help="분할 비율(기본 0.5 0.2 0.3 — docs/DECISIONS.md 참고)")
    sp.add_argument("--keep-from", metavar="PREV_CSV",
                    help="이전에 나눈 파일. 거기 있는 질문은 분할을 옮기지 않고 새 질문만 배정한다")
    sp.add_argument("--reshuffle", action="store_true",
                    help="out 파일이 이미 있어도 처음부터 다시 섞는다(튜닝·라벨링 전에만)")
    sp.set_defaults(func=cmd_split)

    d = sub.add_parser("doctor", help="환경 점검(파이썬, torch, GPU, 디스크, 데이터 파일)")
    d.add_argument("--data-dir", default="data")
    d.set_defaults(func=cmd_doctor)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
