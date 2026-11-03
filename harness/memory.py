"""하네스 경험 저장. kore.memory.ExperienceLog를 그대로 재사용하고 교훈 기록을 덧붙인다.

- 문항별 경험: ExperienceLog(train, dev만 허용). 이 모듈이 먼저 test를 거부해서 SplitViolation을 낸다.
- 교훈: 사람이 편집을 채택·기각할 때만 쓴다. 근거(문항 수, 구간, 평균 차이)가 없는 교훈은 거부한다.
  교훈은 '이 dev 세트에서 이렇게 관찰했다'는 기록이지 일반 법칙이 아니다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from harness.safety import ALLOWED_SPLITS, SafetyError, SplitViolation, ensure_safe_object, resolve_inside
from kore.memory import ExperienceLog

REQUIRED_EVIDENCE = ("n", "mean_diff", "ci_lo", "ci_hi")
LESSON_DECISIONS = ("adopted", "rejected")


class HarnessMemory:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.experience = ExperienceLog(resolve_inside(self.root, "memory/experience.jsonl"))
        self.lessons_path = resolve_inside(self.root, "memory/lessons.jsonl")

    def record_question(
        self, question_id: str, split: str, candidate_ids: list[str], selected_ids: list[str],
        quality: float | None = None, baseline_quality: float | None = None,
    ) -> None:
        if split not in ALLOWED_SPLITS:
            raise SplitViolation(f"split '{split}'의 경험은 기록하지 않는다(허용: {ALLOWED_SPLITS})")
        self.experience.append(question_id, split, candidate_ids, selected_ids, quality, baseline_quality)

    def add_lesson(
        self, edit_id: str, component: str, change_key: str, decision: str, decided_by: str, reason: str,
        evidence: dict[str, Any], confidence: float | None, change: dict[str, Any],
    ) -> dict[str, Any]:
        if decision not in LESSON_DECISIONS:
            raise SafetyError(f"교훈의 decision은 {LESSON_DECISIONS} 중 하나여야 한다")
        if not decided_by.strip() or not reason.strip():
            raise SafetyError("교훈에는 결정한 사람(decided_by)과 이유(reason)가 필요하다")
        missing = [k for k in REQUIRED_EVIDENCE if k not in evidence or evidence[k] is None]
        if missing:
            raise SafetyError(f"근거 없는 교훈은 저장하지 않는다. 빠진 값: {missing}")
        record = {
            "edit_id": edit_id, "component": component, "change_key": change_key, "change": change,
            "decision": decision, "decided_by": decided_by, "reason": reason, "evidence": evidence,
            "proposer_confidence": confidence,
            "scope": "이 dev 세트에서 관찰한 결과. 일반화하지 않는다",
        }
        ensure_safe_object(record, "교훈")
        self.lessons_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lessons_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def lessons(self, component: str | None = None) -> Iterator[dict[str, Any]]:
        if not self.lessons_path.exists():
            return
        for line in self.lessons_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if component is None or rec["component"] == component:
                    yield rec

    def was_rejected(self, change_key: str) -> bool:
        return any(r["change_key"] == change_key and r["decision"] == "rejected" for r in self.lessons())
