from __future__ import annotations

from pathlib import Path

import pytest

from kore.io import load_knowledge, load_questions

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


@pytest.fixture(scope="session")
def items():
    return load_knowledge(DATA / "sample_knowledge.csv")


@pytest.fixture(scope="session")
def questions(items):
    return load_questions(DATA / "sample_eval.csv", items)
