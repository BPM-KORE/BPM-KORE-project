"""검색 → 선택 → (생성) → 평가 → 기록 파이프라인과 재개 가능한 실험 러너."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from kore import __version__
from kore.evaluation import bootstrap_ci, evaluate, paired_bootstrap_diff, recall_at_k
from kore.generation import Generator
from kore.retrieval import BM25Retriever
from kore.schemas import CONTRACT_VERSION, EvalQuestion, KnowledgeItem
from kore.selector import FEATURES, SelectorWeights, select_knowledge

BASE_CONDITIONS = ("no_retrieval", "vanilla_rag", "kore")
ABLATION_PREFIX = "kore_ablate_"
BASELINE = "vanilla_rag"


def all_conditions() -> tuple[str, ...]:
    return BASE_CONDITIONS + tuple(ABLATION_PREFIX + f for f in FEATURES)


@dataclass
class PipelineConfig:
    candidate_k: int = 10
    k: int = 3
    token_budget: int | None = None
    tokenizer_mode: str = "ngram"
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    weights: SelectorWeights = field(default_factory=SelectorWeights)

    def validate(self) -> None:
        if self.k < 1 or self.candidate_k < 1:
            raise ValueError("k와 candidate_k는 1 이상이어야 한다")
        if self.k > self.candidate_k:
            raise ValueError("k는 candidate_k보다 클 수 없다")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        """설정 사전에서 만든다. 모르는 키는 오타일 수 있으니 조용히 넘기지 않고 오류를 낸다."""
        data = dict(data)
        weights_raw = data.pop("weights", {}) or {}
        allowed = {"candidate_k", "k", "token_budget", "tokenizer_mode", "bm25_k1", "bm25_b"}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"설정에 모르는 키가 있다: {sorted(unknown)}")
        bad_w = set(weights_raw) - set(SelectorWeights().as_dict())
        if bad_w:
            raise ValueError(f"weights에 모르는 키가 있다: {sorted(bad_w)}")
        cfg = cls(weights=SelectorWeights(**weights_raw), **data)
        cfg.validate()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_k": self.candidate_k,
            "k": self.k,
            "token_budget": self.token_budget,
            "tokenizer_mode": self.tokenizer_mode,
            "bm25_k1": self.bm25_k1,
            "bm25_b": self.bm25_b,
            "weights": self.weights.as_dict(),
        }


RUN_LEVEL_KEYS = ("seed", "conditions")


def load_config(path: str | Path) -> tuple[PipelineConfig, dict[str, Any]]:
    """JSON 설정 파일을 읽어 (파이프라인 설정, 실행 수준 값{seed, conditions})을 돌려준다."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    run_level = {k: raw.pop(k) for k in RUN_LEVEL_KEYS if k in raw}
    if "conditions" in run_level:
        bad = [c for c in run_level["conditions"] if c not in all_conditions()]
        if bad:
            raise ValueError(f"설정의 conditions에 알 수 없는 조건이 있다: {bad}")
    return PipelineConfig.from_dict(raw), run_level


class KOREPipeline:
    def __init__(
        self,
        items: list[KnowledgeItem],
        config: PipelineConfig | None = None,
        generator: Generator | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.config.validate()
        self.generator = generator
        self.retriever = BM25Retriever(
            items,
            k1=self.config.bm25_k1,
            b=self.config.bm25_b,
            tokenizer_mode=self.config.tokenizer_mode,
        )

    def candidates(self, question: str) -> list[KnowledgeItem]:
        return self.retriever.retrieve(question, k=self.config.candidate_k)

    def select(self, question: str, candidates: list[KnowledgeItem], condition: str) -> list[KnowledgeItem]:
        cfg = self.config
        if condition == "no_retrieval":
            return []
        if condition == "vanilla_rag":
            return select_knowledge(
                question, candidates, k=cfg.k, strategy="similarity",
                token_budget=cfg.token_budget, tokenizer_mode=cfg.tokenizer_mode,
            )
        term_weights = self.retriever.term_weights(question)
        weights = cfg.weights
        if condition.startswith(ABLATION_PREFIX):
            weights = weights.ablate(condition[len(ABLATION_PREFIX):])
        elif condition != "kore":
            raise ValueError(f"알 수 없는 조건: {condition!r} (가능: {all_conditions()})")
        return select_knowledge(
            question, candidates, k=cfg.k, strategy="kore", weights=weights,
            token_budget=cfg.token_budget, tokenizer_mode=cfg.tokenizer_mode,
            term_weights=term_weights,
        )

    def run_question(self, q: EvalQuestion, conditions: tuple[str, ...] | list[str]) -> list[dict[str, Any]]:
        """한 질문을 여러 조건으로 돌린다. 후보 목록은 모든 조건이 같은 것을 공유한다."""
        t0 = time.perf_counter()
        cands = self.candidates(q.question)
        retrieval_ms = (time.perf_counter() - t0) * 1000
        cand_ids = [c.id for c in cands]
        reference = {"expected_source_ids": q.expected_source_ids, "rubric_items": q.rubric_items()}
        records: list[dict[str, Any]] = []
        for cond in conditions:
            t1 = time.perf_counter()
            selected = self.select(q.question, cands, cond)
            select_ms = (time.perf_counter() - t1) * 1000
            answer: str | None = None
            generate_ms = 0.0
            if self.generator is not None:
                t2 = time.perf_counter()
                answer = self.generator.generate(q.question, selected)
                generate_ms = (time.perf_counter() - t2) * 1000
            metrics = evaluate(q.question, selected, reference, answer)
            records.append(
                {
                    "question_id": q.question_id,
                    "condition": cond,
                    "domain": q.domain,
                    "split": q.split,
                    "difficulty": q.difficulty,
                    "candidate_ids": cand_ids,
                    "selected_ids": [s.id for s in selected],
                    "expected_source_ids": q.expected_source_ids,
                    "candidate_recall": recall_at_k(cand_ids, q.expected_source_ids),
                    **metrics,
                    "answer": answer,
                    "retrieval_ms": round(retrieval_ms, 3),
                    "select_ms": round(select_ms, 3),
                    "generate_ms": round(generate_ms, 3),
                }
            )
        return records


# ---------------------------------------------------------------- 실험 러너


def _canonical_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _read_log(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # 비정상 종료로 잘린 마지막 줄은 버리고 다시 계산한다
    return rows


def _mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def summarize(rows: list[dict[str, Any]], seed: int = 0) -> list[dict[str, Any]]:
    """조건×분할별 평균과 recall·rubric의 95% 부트스트랩 구간."""
    out: list[dict[str, Any]] = []
    conditions = sorted({r["condition"] for r in rows}, key=lambda c: all_conditions().index(c) if c in all_conditions() else 99)
    splits = sorted({r["split"] for r in rows})
    for cond in conditions:
        for split in splits + ["all"]:
            sel = [r for r in rows if r["condition"] == cond and (split == "all" or r["split"] == split)]
            if not sel:
                continue
            rec_m, rec_lo, rec_hi = bootstrap_ci([r["recall"] for r in sel], seed=seed)
            rub_m, rub_lo, rub_hi = bootstrap_ci([r["rubric_score"] for r in sel], seed=seed)
            out.append(
                {
                    "condition": cond,
                    "split": split,
                    "n": len(sel),
                    "recall": rec_m,
                    "recall_ci_lo": rec_lo,
                    "recall_ci_hi": rec_hi,
                    "precision": _mean([r["precision"] for r in sel]),
                    "hit": _mean([r["hit"] for r in sel]),
                    "mrr": _mean([r["mrr"] for r in sel]),
                    "candidate_recall": _mean([r["candidate_recall"] for r in sel]),
                    "context_tokens": _mean([r["context_tokens"] for r in sel]),
                    "rubric_score": rub_m,
                    "rubric_ci_lo": rub_lo,
                    "rubric_ci_hi": rub_hi,
                    "retrieval_ms": _mean([r["retrieval_ms"] for r in sel]),
                    "select_ms": _mean([r["select_ms"] for r in sel]),
                    "generate_ms": _mean([r["generate_ms"] for r in sel]),
                }
            )
    return out


def compare_to_baseline(rows: list[dict[str, Any]], baseline: str = BASELINE, seed: int = 0) -> list[dict[str, Any]]:
    """각 조건을 기준선과 같은 질문끼리 짝지어 비교한다."""
    out: list[dict[str, Any]] = []
    base_rows = {(r["question_id"]): r for r in rows if r["condition"] == baseline}
    if not base_rows:
        return out
    conditions = sorted({r["condition"] for r in rows} - {baseline})
    splits = sorted({r["split"] for r in rows})
    for cond in conditions:
        cond_rows = {r["question_id"]: r for r in rows if r["condition"] == cond}
        for split in splits + ["all"]:
            qids = sorted(
                qid for qid in cond_rows
                if qid in base_rows and (split == "all" or cond_rows[qid]["split"] == split)
            )
            if not qids:
                continue
            for metric in ("recall", "rubric_score"):
                diff = paired_bootstrap_diff(
                    [cond_rows[q][metric] for q in qids], [base_rows[q][metric] for q in qids], seed=seed
                )
                if diff["n"] == 0:
                    continue
                out.append({"condition": cond, "baseline": baseline, "metric": metric, "split": split, **diff})
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {c: ("" if row.get(c) is None else (round(row[c], 4) if isinstance(row[c], float) else row[c])) for c in columns}
            )


SUMMARY_COLUMNS = [
    "condition", "split", "n", "recall", "recall_ci_lo", "recall_ci_hi", "precision", "hit", "mrr",
    "candidate_recall", "context_tokens", "rubric_score", "rubric_ci_lo", "rubric_ci_hi",
    "retrieval_ms", "select_ms", "generate_ms",
]
COMPARISON_COLUMNS = ["condition", "baseline", "metric", "split", "n", "mean_diff", "ci_lo", "ci_hi"]


def run_experiment(
    items: list[KnowledgeItem],
    questions: list[EvalQuestion],
    out_dir: str | Path,
    conditions: tuple[str, ...] | list[str] = BASE_CONDITIONS,
    config: PipelineConfig | None = None,
    generator: Generator | None = None,
    seed: int = 0,
    include_test: bool = False,
    resume: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """실험을 돌리고 out_dir에 run_log.jsonl, results.csv, comparisons.csv, env.json을 남긴다.

    - test 분할은 include_test=True일 때만 돌린다. 최종 평가에서 한 번만 켠다.
    - resume=True이면 이미 끝난 (질문, 조건) 쌍은 건너뛴다. Colab 연결이 끊겨도 이어서 돌릴 수 있다.
    - 설정·데이터·생성기·시드가 이전 실행과 다르면 결과가 섞이지 않게 ValueError를 낸다.
    """
    config = config or PipelineConfig()
    config.validate()
    conditions = tuple(conditions)
    for c in conditions:
        if c not in all_conditions():
            raise ValueError(f"알 수 없는 조건: {c!r} (가능: {all_conditions()})")
    scope = [q for q in questions if include_test or q.split != "test"]
    if not scope:
        raise ValueError("돌릴 질문이 없다(test 분할만 있다면 include_test=True가 필요하다)")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path, env_path = out / "run_log.jsonl", out / "env.json"
    generator_name = generator.name if generator is not None else "none"
    fingerprint = {
        "config_hash": _canonical_hash(config.to_dict()),
        "data_hash": _canonical_hash(
            {"items": [i.to_dict() for i in items], "questions": [q.to_dict() for q in questions]}
        ),
        "generator": generator_name,
        "seed": seed,
    }

    if resume and env_path.exists():
        prev = json.loads(env_path.read_text(encoding="utf-8"))
        diff = [k for k, v in fingerprint.items() if prev.get(k) != v]
        if diff:
            raise ValueError(
                f"이전 실행과 {', '.join(diff)}가 다르다. 결과가 섞이지 않게 새 out_dir를 쓰거나 resume=False로 다시 시작한다"
            )
    if not resume:
        for p in (log_path, env_path):
            if p.exists():
                p.unlink()

    done = {(r["question_id"], r["condition"]) for r in _read_log(log_path)}
    pipeline = KOREPipeline(items, config=config, generator=generator)
    if log_path.exists() and log_path.stat().st_size > 0 and not log_path.read_bytes().endswith(b"\n"):
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write("\n")  # 잘린 마지막 줄을 새 줄과 분리한다

    started = datetime.now(timezone.utc).isoformat()
    with log_path.open("a", encoding="utf-8") as fh:
        for n, q in enumerate(scope, start=1):
            need = [c for c in conditions if (q.question_id, c) not in done]
            if need:
                for rec in pipeline.run_question(q, need):
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
            if progress:
                progress(n, len(scope))

    in_scope = {q.question_id for q in scope}
    rows = [
        r for r in _read_log(log_path)
        if r["condition"] in conditions and r["question_id"] in in_scope
    ]
    results = summarize(rows, seed=seed)
    comparisons = compare_to_baseline(rows, seed=seed)
    _write_csv(out / "results.csv", results, SUMMARY_COLUMNS)
    _write_csv(out / "comparisons.csv", comparisons, COMPARISON_COLUMNS)
    env = {
        **fingerprint,
        "kore_version": __version__,
        "contract_version": CONTRACT_VERSION,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "config": config.to_dict(),
        "conditions": list(conditions),
        "include_test": include_test,
        "n_questions": len(scope),
        "n_rows": len(rows),
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    env_path.write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"out_dir": str(out), "n_rows": len(rows), "results": results, "comparisons": comparisons, "env": env}
