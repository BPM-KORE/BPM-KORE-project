from __future__ import annotations

import csv
from pathlib import Path

from kore.schemas import KnowledgeItem

K_HEADER = ["id", "domain", "title", "text", "source_url", "license", "reliability", "lang"]
Q_HEADER = [
    "question_id", "question", "expected_source_ids", "domain", "difficulty", "split",
    "reference_answer", "rubric",
]


def mk_item(
    item_id: str,
    text: str,
    score: float = 1.0,
    domain: str = "d",
    reliability: float = 0.5,
    title: str = "",
) -> KnowledgeItem:
    return KnowledgeItem(
        id=item_id, domain=domain, title=title, text=text, source_url="local:t", license="t",
        score=score, retrieval_score=score, reliability=reliability,
    )


def write_csv(path: Path, header: list[str], rows: list[list[str]], encoding: str = "utf-8") -> Path:
    with path.open("w", newline="", encoding=encoding) as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def k_row(item_id: str = "a1", **over: str) -> list[str]:
    base = {
        "id": item_id, "domain": "astronomy", "title": "T",
        "text": "A reasonably long sentence about the topic that passes the length check.",
        "source_url": "https://example.org/x", "license": "CC-BY-4.0", "reliability": "0.7", "lang": "en",
    }
    base.update(over)
    return [base[c] for c in K_HEADER]


def q_row(qid: str = "q1", **over: str) -> list[str]:
    base = {
        "question_id": qid, "question": "What is the topic of the first document?",
        "expected_source_ids": "a1", "domain": "astronomy", "difficulty": "easy", "split": "train",
        "reference_answer": "The topic.", "rubric": "topic",
    }
    base.update(over)
    return [base[c] for c in Q_HEADER]
