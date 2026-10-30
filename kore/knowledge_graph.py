"""문서 사이의 관계(relations.csv). B가 만들고 A가 읽는다.

MVP에서는 관계를 검색이나 선택에 쓰지 않는다. 형식을 먼저 고정하고 검증하는 데까지만 구현했다.
관계가 실제로 답변 품질에 도움이 되는지는 별도 실험으로 확인해야 한다.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from kore.io import ValidationReport, _read_rows
from kore.schemas import RELATION_REQUIRED, KnowledgeItem


@dataclass(frozen=True)
class Relation:
    src_id: str
    dst_id: str
    relation: str
    evidence_url: str = ""


class RelationIndex:
    def __init__(self, relations: list[Relation]) -> None:
        self._out: dict[str, list[Relation]] = defaultdict(list)
        for r in relations:
            self._out[r.src_id].append(r)

    def neighbors(self, item_id: str) -> list[Relation]:
        return list(self._out.get(item_id, []))


def read_relations(
    path: str | Path, items: list[KnowledgeItem] | None = None
) -> tuple[list[Relation], ValidationReport]:
    path = Path(path)
    report = ValidationReport()
    rows, columns = _read_rows(path, report)
    if report.errors and not rows:
        return [], report
    missing = [c for c in RELATION_REQUIRED if c not in columns]
    if missing:
        report.error(f"{path.name}: 필수 열이 없다: {', '.join(missing)}")
        return [], report
    known = {i.id for i in items} if items is not None else None
    relations: list[Relation] = []
    seen: set[tuple[str, str, str]] = set()
    for idx, row in enumerate(rows, start=2):
        where = f"{path.name}:{idx}"
        ok = True
        for col in RELATION_REQUIRED:
            if not row.get(col):
                report.error(f"{where}: '{col}'이 비어 있다")
                ok = False
        if not ok:
            continue
        if row["src_id"] == row["dst_id"]:
            report.error(f"{where}: 자기 자신을 가리키는 관계다")
            continue
        if known is not None:
            for col in ("src_id", "dst_id"):
                if row[col] not in known:
                    report.error(f"{where}: {col} '{row[col]}'가 knowledge에 없다")
                    ok = False
        key = (row["src_id"], row["dst_id"], row["relation"])
        if key in seen:
            report.warn(f"{where}: 같은 관계가 중복이다")
        seen.add(key)
        if not row.get("evidence_url"):
            report.warn(f"{where}: evidence_url이 없다. 관계는 근거를 붙여 검증한다")
        if ok:
            relations.append(Relation(row["src_id"], row["dst_id"], row["relation"], row.get("evidence_url", "")))
    return relations, report
