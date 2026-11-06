"""Harness² 구조(제안 → 시험 → 조합)의 MVP 루프. 공식 구현은 쓰지 않는다.

한 바퀴(iteration) = 제안 파일 하나를 읽고(Propose), 편집마다 같은 dev 질문으로 시험하고(Probe),
문항별 차이표와 통계 한계를 만든다(Compose). 채택은 자동으로 하지 않는다. 시험을 마친 편집은 '사람 승인 대기'에서 멈추고,
`approve`로 사람이 결정해야 채택·기각된다. 채택된 편집은 다음 바퀴의 기준 설정에만 반영되고 kore/ 와 configs/ 는 건드리지 않는다.

상한: 반복 최대 5회, 시간(초), 생성기 호출 수. 매 편집을 마칠 때마다 checkpoint.json을 저장해서 중단 후 재개할 수 있다.
검증된 개선은 아직 없다. 이 루프의 숫자를 성능 주장으로 쓰지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from harness.components import (
    COMPONENTS, FIXED_CONDITIONS_CHANGED, STATUS_KO, Edit, EditError, HarnessConfig, apply_edit, canonical_hash,
)
from harness.memory import HarnessMemory
from harness.safety import (
    SafetyError, assert_no_test, check_splits, ensure_safe_object, load_dev_questions, resolve_inside,
)
from harness.stats import dev_limits, format_limits
from kore.pipeline import run_experiment
from kore.schemas import EvalQuestion, KnowledgeItem

MAX_ITERATIONS_HARD = 5
MAX_EDITS_PER_ITERATION = 8
MAX_PROPOSAL_BYTES = 1_000_000
PROPOSAL_KEYS = {"proposal_version", "proposer", "note", "edits"}
CHECKPOINT_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[2]

GeneratorFactory = Callable[[HarnessConfig], Any]


class BudgetExceeded(RuntimeError):
    """시간 또는 생성기 호출 수 상한에 닿았다. 체크포인트는 남아 있고 상한을 올려 재개할 수 있다."""


@dataclass
class Budget:
    max_iterations: int = MAX_ITERATIONS_HARD
    max_seconds: float = 900.0
    max_generate_calls: int = 400

    def validate(self) -> None:
        if not 1 <= self.max_iterations <= MAX_ITERATIONS_HARD:
            raise ValueError(f"max_iterations는 1~{MAX_ITERATIONS_HARD}이다(고정 상한): {self.max_iterations}")
        if self.max_seconds <= 0 or self.max_generate_calls < 0:
            raise ValueError("max_seconds는 양수, max_generate_calls는 0 이상이어야 한다")


class _Meter:
    def __init__(self, budget: Budget, used_calls: int, used_seconds: float, clock: Callable[[], float]) -> None:
        self.budget, self.calls, self.base_seconds, self.clock = budget, used_calls, used_seconds, clock
        self.t0 = clock()

    def seconds(self) -> float:
        return self.base_seconds + (self.clock() - self.t0)

    def check_time(self) -> None:
        if self.seconds() > self.budget.max_seconds:
            raise BudgetExceeded(f"시간 상한 {self.budget.max_seconds}초를 넘었다")

    def charge_call(self) -> None:
        self.check_time()
        if self.calls >= self.budget.max_generate_calls:
            raise BudgetExceeded(f"생성기 호출 상한 {self.budget.max_generate_calls}회에 닿았다")
        self.calls += 1


class _CountingGenerator:
    def __init__(self, inner: Any, meter: _Meter) -> None:
        self.inner, self.meter, self.name = inner, meter, inner.name

    def generate(self, question: str, evidence: list[KnowledgeItem]) -> str:
        self.meter.charge_call()
        return self.inner.generate(question, evidence)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # 비정상 종료로 잘린 줄
    return rows


class HarnessLoop:
    def __init__(
        self,
        items: list[KnowledgeItem],
        questions: list[EvalQuestion],
        exp_dir: str | Path,
        base_config: HarnessConfig | None = None,
        generator_factory: GeneratorFactory | None = None,
        budget: Budget | None = None,
        seed: int = 0,
        splits: tuple[str, ...] = ("dev",),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.splits = check_splits(splits)
        assert_no_test(questions, self.splits)
        self.items, self.questions = items, list(questions)
        if not self.questions:
            raise ValueError("시험할 질문이 없다")
        self.exp_dir = Path(exp_dir).resolve()
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.base_config = base_config or HarnessConfig.default()
        self.base_config.pipeline_config()
        self.factory = generator_factory
        self.budget = budget or Budget()
        self.budget.validate()
        self.seed, self.clock = seed, clock
        self.metric = "rubric_score" if generator_factory is not None else "recall"
        self.memory = HarnessMemory(self.exp_dir)
        self.data_hash = canonical_hash(
            {"items": [i.to_dict() for i in items], "questions": [q.to_dict() for q in self.questions]}
        )

    # ------------------------------------------------------------ 체크포인트

    @property
    def checkpoint_path(self) -> Path:
        return resolve_inside(self.exp_dir, "checkpoint.json")

    def _fresh_checkpoint(self) -> dict[str, Any]:
        return {
            "version": CHECKPOINT_VERSION, "seed": self.seed, "metric": self.metric, "splits": list(self.splits),
            "data_hash": self.data_hash, "base_config": self.base_config.to_dict(),
            "base_config_hash": self.base_config.hash(),
            "used": {"calls": 0, "seconds": 0.0}, "adopted": [], "baseline_cache": {}, "iterations": [],
        }

    def load_checkpoint(self) -> dict[str, Any]:
        path = self.checkpoint_path
        if not path.exists():
            return self._fresh_checkpoint()
        ck = json.loads(path.read_text(encoding="utf-8"))
        for key, now in (("version", CHECKPOINT_VERSION), ("seed", self.seed), ("metric", self.metric),
                         ("data_hash", self.data_hash), ("base_config_hash", self.base_config.hash())):
            if ck.get(key) != now:
                raise ValueError(f"체크포인트의 {key}가 지금 설정과 다르다. 새 exp_dir를 쓴다(결과가 섞이면 안 된다)")
        return ck

    def _save(self, ck: dict[str, Any], meter: _Meter | None = None) -> None:
        if meter is not None:
            ck["used"] = {"calls": meter.calls, "seconds": round(meter.seconds(), 3)}
        path = self.checkpoint_path
        tmp = resolve_inside(self.exp_dir, "checkpoint.json.tmp")
        tmp.write_text(json.dumps(ck, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)

    # ------------------------------------------------------------ 제안(Propose)

    def _edits_so_far(self, ck: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {e["id"]: e for it in ck["iterations"] for e in it["edits"]}

    def parse_proposal(self, raw: bytes, ck: dict[str, Any]) -> list[Edit]:
        """제안 파일을 편집 목록으로 바꾼다. 위반 편집은 버리지 않고 '무효'로 남겨서 이유를 기록한다."""
        if len(raw) > MAX_PROPOSAL_BYTES:
            raise SafetyError("제안 파일이 너무 크다")
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SafetyError(f"제안 파일이 올바른 JSON이 아니다: {exc}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("edits"), list):
            raise SafetyError("제안 파일은 {'edits': [...]} 형식이어야 한다")
        extra = set(data) - PROPOSAL_KEYS
        if extra:
            raise SafetyError(f"제안 파일에 허용되지 않은 최상위 키가 있다: {sorted(extra)}")
        known = set(self._edits_so_far(ck))
        out: list[Edit] = []
        for n, item in enumerate(data["edits"]):
            if sum(1 for e in out if e.status != "invalid") >= MAX_EDITS_PER_ITERATION:
                edit = self._placeholder(item, n, known | {e.id for e in out})
                edit.mark_invalid(f"한 바퀴에 시험할 수 있는 편집은 {MAX_EDITS_PER_ITERATION}개까지다")
                out.append(edit)
                continue
            try:
                ensure_safe_object(item, f"편집 #{n}")
                edit = Edit.from_dict(item)
                if edit.id in known or any(e.id == edit.id for e in out):
                    raise EditError(f"id {edit.id!r}가 이미 쓰였다")
                if edit.target_path is not None:
                    resolve_inside(self.exp_dir / "edits", edit.target_path)
                if self.memory.was_rejected(edit.change_key()):
                    raise EditError("과거에 사람이 기각한 것과 같은 편집이다(교훈 참고)")
            except (SafetyError, EditError) as exc:
                edit = self._placeholder(item, n, known | {e.id for e in out})
                edit.mark_invalid(str(exc))
            out.append(edit)
        return out

    @staticmethod
    def _placeholder(item: Any, n: int, taken: set[str]) -> Edit:
        """검사에 실패한 편집을 '무효'로 남기기 위한 빈 기록. 내용은 보관하지 않고 해시만 남긴다."""
        raw_id = item.get("id") if isinstance(item, dict) else None
        ok = isinstance(raw_id, str) and 0 < len(raw_id) <= 40 and all(c.isalnum() or c in "_.-" for c in raw_id)
        eid = raw_id if ok and raw_id not in taken else f"invalid_{n}_{len(taken)}"
        comp = item.get("component") if isinstance(item, dict) else None
        return Edit(id=eid, component=comp if comp in COMPONENTS else "unknown", change={}, rationale="",
                    result={"raw_hash": canonical_hash(item)})

    # ------------------------------------------------------------ 시험(Probe)

    def _probe(self, subdir: str, cfg: HarnessConfig, meter: _Meter) -> dict[str, dict[str, Any]]:
        out_dir = resolve_inside(self.exp_dir, f"runs/{subdir}")
        qs = [replace(q, question=cfg.transform_query(q.question)) for q in self.questions]
        gen = None
        if self.factory is not None:
            gen = _CountingGenerator(self.factory(cfg), meter)
        run_experiment(
            self.items, qs, out_dir, conditions=("kore",), config=cfg.pipeline_config(), generator=gen,
            seed=self.seed, include_test=False, resume=True,
        )
        scope = {q.question_id for q in self.questions}
        per_q: dict[str, dict[str, Any]] = {}
        for r in _read_jsonl(out_dir / "run_log.jsonl"):
            if r["condition"] == "kore" and r["question_id"] in scope:
                per_q[r["question_id"]] = {
                    "score": r.get(self.metric), "recall": r.get("recall"), "split": r["split"],
                    "candidate_ids": r["candidate_ids"], "selected_ids": r["selected_ids"],
                }
        return per_q

    def _current_base(self, ck: dict[str, Any], edits_by_id: dict[str, dict[str, Any]]) -> HarnessConfig:
        cfg = self.base_config
        for eid in ck["adopted"]:
            cfg = apply_edit(cfg, Edit.from_record(edits_by_id[eid]))
        return cfg

    # ------------------------------------------------------------ 한 바퀴

    def _run_iteration(self, ck: dict[str, Any], idx: int, proposal: Path, meter: _Meter) -> None:
        raw = proposal.read_bytes()
        phash = canonical_hash(raw.decode("utf-8", errors="replace"))
        if idx < len(ck["iterations"]):
            it = ck["iterations"][idx]
            if it["proposal_hash"] != phash:
                raise ValueError(f"바퀴 {idx}의 제안 파일이 처음 읽은 것과 다르다. 새 exp_dir를 쓴다")
            it["status"], it["halt_reason"] = "running", ""  # 중단 뒤 재개
        else:
            it = {"index": idx, "proposal_file": proposal.name, "proposal_hash": phash, "status": "running",
                  "halt_reason": "", "base_hash": "", "edits": [], "summary": {}}
            try:
                edits = self.parse_proposal(raw, ck)
            except SafetyError as exc:
                it["status"], it["halt_reason"] = "invalid_proposal", str(exc)
                ck["iterations"].append(it)
                self._save(ck, meter)
                return
            it["edits"] = [e.to_dict() for e in edits]
            ck["iterations"].append(it)
            self._save(ck, meter)

        base = self._current_base(ck, self._edits_so_far(ck))
        it["base_hash"] = base.hash()
        if it["base_hash"] not in ck["baseline_cache"]:
            ck["baseline_cache"][it["base_hash"]] = self._probe(f"baseline_{it['base_hash']}", base, meter)
            self._save(ck, meter)

        for rec in it["edits"]:
            if rec["status"] not in ("proposed", "testing"):
                continue
            meter.check_time()
            edit = Edit.from_record(rec)
            if edit.status == "proposed":  # 재개할 때는 '시험중' 기록을 되풀이하지 않는다
                edit.set_status("testing", "시험 시작")
                rec.update(edit.to_dict())
                self._save(ck, meter)
            try:
                cfg = apply_edit(base, edit)
            except (EditError, ValueError) as exc:
                edit.mark_invalid(f"적용 실패: {exc}")
                rec.update(edit.to_dict())
                self._save(ck, meter)
                continue
            target = resolve_inside(self.exp_dir / "edits", edit.target_path or f"{edit.id}.json")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            per_q = self._probe(f"it{idx}_{edit.id}", cfg, meter)
            base_q = ck["baseline_cache"][it["base_hash"]]
            changed = sum(1 for q, v in per_q.items() if v["candidate_ids"] != base_q.get(q, {}).get("candidate_ids"))
            edit.result = {
                "config_hash": cfg.hash(), "per_question": per_q, "n_candidates_changed": changed,
                "fixed_conditions_changed": FIXED_CONDITIONS_CHANGED[edit.component], "experience_logged": False,
            }
            edit.set_status("awaiting_approval", "시험 완료. 사람 승인 대기")
            rec.update(edit.to_dict())
            self._save(ck, meter)

        self._compose(ck, it)
        it["status"] = "done"
        self._save(ck, meter)

    # ------------------------------------------------------------ 조합(Compose)

    def _compose(self, ck: dict[str, Any], it: dict[str, Any]) -> None:
        base_q = ck["baseline_cache"][it["base_hash"]]
        tested = [e for e in it["edits"] if e["result"] and "per_question" in e["result"]]
        qids = sorted(base_q)
        summary: dict[str, Any] = {}
        for e in tested:
            pq = e["result"]["per_question"]
            diffs = [
                None if pq.get(q, {}).get("score") is None or base_q[q]["score"] is None
                else pq[q]["score"] - base_q[q]["score"]
                for q in qids
            ]
            lim = dev_limits(diffs, len(qids), n_compared=len(tested), seed=self.seed)
            summary[e["id"]] = lim
            if not e["result"]["experience_logged"]:
                for q in qids:
                    if q in pq:
                        self.memory.record_question(
                            q, pq[q]["split"], pq[q]["candidate_ids"], pq[q]["selected_ids"],
                            quality=pq[q]["score"], baseline_quality=base_q[q]["score"],
                        )
                e["result"]["experience_logged"] = True
        it["summary"] = summary
        rows = []
        for q in qids:
            row: dict[str, Any] = {"question_id": q, "baseline": base_q[q]["score"]}
            for e in tested:
                s = e["result"]["per_question"].get(q, {}).get("score")
                row[e["id"]] = s
                row[e["id"] + "__diff"] = None if s is None or base_q[q]["score"] is None else round(s - base_q[q]["score"], 6)
            rows.append(row)
        cols = ["question_id", "baseline"] + [c for e in tested for c in (e["id"], e["id"] + "__diff")]
        path = resolve_inside(self.exp_dir, f"table_it{it['index']}.csv")
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: ("" if r.get(c) is None else r[c]) for c in cols})

    # ------------------------------------------------------------ 보고서

    def render_report(self, ck: dict[str, Any]) -> str:
        def f(v: Any) -> str:
            return "-" if v is None else (f"{v:+.3f}" if isinstance(v, float) else str(v))

        lines = [
            "# Harness 루프 시험 보고서 (자동 생성)",
            "",
            "- 이 숫자는 성능 주장이 아니다. 검증된 개선은 아직 없다. 채택은 사람이 `approve`로만 한다.",
            f"- 지표: `{ck['metric']}`(기준: 편집 없는 설정), 질문 분할: {ck['splits']}, 시드 {ck['seed']}",
            "- 생성기가 가짜(추출 기준선)이면 점수는 배관 점검용이다. 연구 결과로 인용하지 않는다.",
        ]
        for it in ck["iterations"]:
            lines += ["", f"## 바퀴 {it['index']} — {it['proposal_file']} ({it['status']})"]
            if it["halt_reason"]:
                lines.append(f"- 중단·거부 사유: {it['halt_reason']}")
            if not it["edits"]:
                continue
            lines += ["", "| 편집 | 구성요소 | 상태 | 바뀐 비교 조건 | 평균 차이 | 95% 구간 | 판정 |", "|---|---|---|---|---|---|---|"]
            for e in it["edits"]:
                s = it["summary"].get(e["id"])
                fc = ", ".join(e["result"]["fixed_conditions_changed"]) if e["result"] and "fixed_conditions_changed" in e["result"] else "-"
                if s:
                    verdict = "구분 불가" if s["indistinguishable"] else "구간이 0을 벗어남(입증 아님)"
                    lines.append(f"| {e['id']} | {e['component']} | {STATUS_KO[e['status']]} | {fc or '없음'} | {f(s['mean_diff'])} | [{f(s['ci_lo'])}, {f(s['ci_hi'])}] | {verdict} |")
                else:
                    lines.append(f"| {e['id']} | {e['component']} | {STATUS_KO[e['status']]} | - | - | - | {e['invalid_reason'] or '-'} |")
            for e in it["edits"]:
                s = it["summary"].get(e["id"])
                if s:
                    lines += ["", format_limits(e["id"], s), f"  - 제안 근거: {e['rationale']} (제안자 신뢰도 {f(e['confidence'])}, 검증된 값이 아님)"]
                    if e["result"]["n_candidates_changed"]:
                        lines.append(f"  - 후보 목록이 바뀐 문항 {e['result']['n_candidates_changed']}개. '같은 후보' 조건이 이 편집에서는 지켜지지 않았다")
            waiting = [e["id"] for e in it["edits"] if e["status"] == "awaiting_approval"]
            if waiting:
                lines += ["", f"사람 승인 대기: {', '.join(waiting)}. 문항별 표는 `table_it{it['index']}.csv`."]
        return "\n".join(lines) + "\n"

    def write_report(self, ck: dict[str, Any]) -> Path:
        path = resolve_inside(self.exp_dir, "report.md")
        path.write_text(self.render_report(ck), encoding="utf-8")
        return path

    # ------------------------------------------------------------ 실행

    def run(self, proposals: list[str | Path]) -> dict[str, Any]:
        ck = self.load_checkpoint()
        meter = _Meter(self.budget, ck["used"]["calls"], ck["used"]["seconds"], self.clock)
        halted, reason = False, ""
        skipped = max(0, len(proposals) - self.budget.max_iterations)
        for idx, proposal in enumerate(list(proposals)[: self.budget.max_iterations]):
            if idx < len(ck["iterations"]) and ck["iterations"][idx]["status"] in ("done", "invalid_proposal"):
                if ck["iterations"][idx]["proposal_hash"] != canonical_hash(Path(proposal).read_bytes().decode("utf-8", errors="replace")):
                    raise ValueError(f"바퀴 {idx}의 제안 파일이 처음 읽은 것과 다르다. 새 exp_dir를 쓴다")
                continue
            try:
                self._run_iteration(ck, idx, Path(proposal), meter)
            except BudgetExceeded as exc:
                halted, reason = True, str(exc)
                if idx < len(ck["iterations"]):
                    ck["iterations"][idx]["status"] = "halted"
                    ck["iterations"][idx]["halt_reason"] = reason
                self._save(ck, meter)
                break
        report = self.write_report(ck)
        waiting = [e["id"] for it in ck["iterations"] for e in it["edits"] if e["status"] == "awaiting_approval"]
        return {"halted": halted, "reason": reason, "iterations": len(ck["iterations"]), "skipped_by_cap": skipped,
                "awaiting_approval": waiting, "report": str(report), "used": ck["used"]}

    # ------------------------------------------------------------ 사람 승인

    def approve(self, edit_id: str, decision: str, approver: str, reason: str) -> dict[str, Any]:
        """사람이 결정한다. 시험을 마친(사람 승인 대기) 편집만 채택·기각할 수 있다."""
        if decision not in ("adopted", "rejected"):
            raise ValueError("decision은 adopted 또는 rejected여야 한다")
        if not approver.strip() or not reason.strip():
            raise ValueError("승인자 이름과 이유가 필요하다")
        ck = self.load_checkpoint()
        for it in ck["iterations"]:
            for rec in it["edits"]:
                if rec["id"] != edit_id:
                    continue
                if rec["status"] != "awaiting_approval":
                    raise ValueError(f"{edit_id}의 상태가 '{STATUS_KO[rec['status']]}'라서 결정할 수 없다(시험을 마친 편집만 가능)")
                s = it["summary"][edit_id]
                edit = Edit.from_record(rec)
                self.memory.add_lesson(
                    edit_id, edit.component, edit.change_key(), decision, approver, reason,
                    {"n": s["n"], "mean_diff": s["mean_diff"], "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"],
                     "indistinguishable": s["indistinguishable"], "metric": ck["metric"]},
                    edit.confidence, edit.change,
                )
                edit.set_status(decision, f"{approver}: {reason}")
                rec.update(edit.to_dict())
                if decision == "adopted":
                    ck["adopted"].append(edit_id)
                    cfg = self._current_base(ck, self._edits_so_far(ck))
                    resolve_inside(self.exp_dir, "adopted_config.json").write_text(
                        json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
                self._save(ck)
                self.write_report(ck)
                return {"edit_id": edit_id, "status": decision, "indistinguishable": s["indistinguishable"]}
        raise ValueError(f"편집 {edit_id!r}를 찾지 못했다")


# ---------------------------------------------------------------- CLI


def _build_loop(args: argparse.Namespace) -> HarnessLoop:
    from kore.io import load_knowledge
    from kore.pipeline import load_config
    from experiments.harness.fakes import make_fake_generator

    exp_dir = resolve_inside(REPO_ROOT / "experiments" / "harness", args.exp_dir)  # 실험 폴더 밖은 거부
    items = load_knowledge(args.knowledge)
    splits = tuple(args.splits)
    questions = load_dev_questions(args.questions, items, splits)
    base = HarnessConfig.from_pipeline_config(load_config(args.base_config)[0])
    budget = Budget(args.max_iterations, args.max_seconds, args.max_calls)
    return HarnessLoop(items, questions, exp_dir, base, make_fake_generator, budget, args.seed, splits)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Harness² 구조 MVP 루프(가짜 생성기, 네트워크 없음)")
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--knowledge", required=True)
    common.add_argument("--questions", required=True)
    common.add_argument("--exp-dir", default="runs/latest", help="experiments/harness 아래 상대 경로")
    common.add_argument("--base-config", default=str(REPO_ROOT / "configs" / "default.json"))
    common.add_argument("--splits", nargs="+", default=["dev"])
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS_HARD)
    common.add_argument("--max-seconds", type=float, default=900.0)
    common.add_argument("--max-calls", type=int, default=400)
    r = sub.add_parser("run", parents=[common])
    r.add_argument("--proposals", nargs="+", required=True)
    a = sub.add_parser("approve", parents=[common])
    a.add_argument("--edit-id", required=True)
    a.add_argument("--decision", choices=["adopted", "rejected"], required=True)
    a.add_argument("--approver", required=True)
    a.add_argument("--reason", required=True)
    args = p.parse_args(argv)
    loop = _build_loop(args)
    if args.cmd == "run":
        print(json.dumps(loop.run(args.proposals), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(loop.approve(args.edit_id, args.decision, args.approver, args.reason), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
