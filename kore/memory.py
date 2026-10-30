"""검색 경험 기록. H3(과거 경험으로 학습한 선택기)를 검증하려면 필요한 학습 데이터다.

지금은 기록만 한다. 이 기록으로 선택 정책을 갱신하는 단계는 후속 작업이다.
기억 오염을 막으려고 test 분할 질문의 경험은 기록 자체를 거부한다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

ALLOWED_SPLITS = ("train", "dev")


class ExperienceLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(
        self,
        question_id: str,
        split: str,
        candidate_ids: list[str],
        selected_ids: list[str],
        quality: float | None = None,
        baseline_quality: float | None = None,
    ) -> None:
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"split '{split}'의 경험은 기록하지 않는다(허용: {ALLOWED_SPLITS}). 기억 오염을 막기 위해서다")
        record: dict[str, Any] = {
            "question_id": question_id,
            "split": split,
            "candidate_ids": candidate_ids,
            "selected_ids": selected_ids,
            "quality": quality,
            "baseline_quality": baseline_quality,
            "gain": None if quality is None or baseline_quality is None else quality - baseline_quality,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def records(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
