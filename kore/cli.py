"""명령줄 도구: validate, run, search, split."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace

from kore.io import ValidationReport, read_knowledge, read_questions, write_questions
from kore.knowledge_graph import read_relations
from kore.pipeline import BASE_CONDITIONS, PipelineConfig, all_conditions, load_config, run_experiment
from kore.retrieval import BM25Retriever
from kore.selector import select_knowledge
from kore.splits import assign_splits

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


def cmd_run(args: argparse.Namespace) -> int:
    items, k_report = read_knowledge(args.knowledge)
    questions, q_report = read_questions(args.questions, items)
    if not (k_report.ok and q_report.ok):
        _print_report("knowledge", k_report)
        _print_report("evaluation", q_report)
        print("데이터 검증에 실패해서 실험을 시작하지 않는다.")
        return 1
    config, run_level = (load_config(args.config) if args.config else (PipelineConfig(), {}))
    # 명령줄 값은 설정 파일보다 우선한다. 값을 주지 않은 항목만 파일(없으면 기본값)을 따른다.
    overrides = {k: v for k, v in (("k", args.k), ("candidate_k", args.candidate_k), ("token_budget", args.token_budget)) if v is not None}
    config = replace(config, **overrides)
    conditions = tuple(args.conditions or run_level.get("conditions") or BASE_CONDITIONS)
    seed = args.seed if args.seed is not None else run_level.get("seed", 0)
    result = run_experiment(
        items, questions, args.out, conditions=conditions, config=config,
        seed=seed, include_test=args.final_test, resume=not args.fresh,
    )
    print(f"완료: {result['n_rows']}행 -> {args.out}")
    for row in result["results"]:
        if row["split"] == "all":
            print(f"  {row['condition']:<24} n={row['n']:<4} recall={row['recall']:.3f} hit={row['hit']:.3f} tokens={row['context_tokens']:.0f}")
    if args.final_test:
        print("주의: test 분할을 포함했다. 이 결과를 보고 알고리즘을 고치지 않는다.")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    items, report = read_knowledge(args.knowledge)
    if not report.ok:
        _print_report("knowledge", report)
        return 1
    retriever = BM25Retriever(items)
    cands = retriever.retrieve(args.question, k=args.candidate_k)
    picked = select_knowledge(args.question, cands, k=args.k, strategy=args.strategy)
    print(f"후보 {len(cands)}개 중 {args.strategy}로 {len(picked)}개 선택")
    for p in picked:
        print(f"  {p.id:<12} [{p.domain}] {p.selection_score:.3f}  {p.title}")
    return 0


def cmd_split(args: argparse.Namespace) -> int:
    items, k_report = read_knowledge(args.knowledge) if args.knowledge else ([], ValidationReport())
    questions, q_report = read_questions(args.questions, items or None)
    if not q_report.ok:
        _print_report("evaluation", q_report)
        return 1
    new = assign_splits(questions, seed=args.seed)
    write_questions(args.out, new)
    print(f"분할 {dict(Counter(q.split for q in new))} -> {args.out} (seed={args.seed})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kore", description="KORE 연구 프로토타입 도구")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="knowledge/evaluation 파일 계약 검증")
    v.add_argument("--knowledge", required=True)
    v.add_argument("--questions")
    v.add_argument("--relations", help="relations.csv(선택)")
    v.set_defaults(func=cmd_validate)

    r = sub.add_parser("run", help="no_retrieval/vanilla_rag/kore 비교 실험")
    r.add_argument("--knowledge", required=True)
    r.add_argument("--questions", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--config", help="JSON 설정 파일(예: configs/default.json). 아래 옵션이 있으면 그 값이 우선한다")
    r.add_argument("--k", type=int, default=None)
    r.add_argument("--candidate-k", type=int, default=None)
    r.add_argument("--token-budget", type=int, default=None)
    r.add_argument("--seed", type=int, default=None)
    r.add_argument("--conditions", nargs="*", choices=all_conditions())
    r.add_argument("--final-test", action="store_true", help="test 분할을 포함한다(최종 평가에서만)")
    r.add_argument("--fresh", action="store_true", help="이어서 하지 않고 out 폴더를 새로 시작한다")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("search", help="질문 하나로 검색·선택 결과 확인")
    s.add_argument("question")
    s.add_argument("--knowledge", required=True)
    s.add_argument("--k", type=int, default=3)
    s.add_argument("--candidate-k", type=int, default=10)
    s.add_argument("--strategy", choices=["similarity", "kore"], default="kore")
    s.set_defaults(func=cmd_search)

    sp = sub.add_parser("split", help="질문을 분야별로 층화해서 train/dev/test 배정")
    sp.add_argument("--questions", required=True)
    sp.add_argument("--knowledge")
    sp.add_argument("--out", required=True)
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=cmd_split)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
