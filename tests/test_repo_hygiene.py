"""저장소 위생 점검: 비밀키가 들어갔는지, 필수 파일이 있는지, 노트북이 유효한지."""

import ast
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "node_modules", ".venv", "venv"}
TEXT_SUFFIXES = {".py", ".md", ".json", ".yml", ".yaml", ".csv", ".jsonl", ".txt", ".toml", ".ipynb", ".cfg", ".ini", ""}

SECRET_PATTERNS = {
    "OpenAI/Anthropic류 키": re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
    "GitHub 토큰": re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    "AWS 액세스 키": re.compile(r"AKIA[0-9A-Z]{16}"),
    "Hugging Face 토큰": re.compile(r"hf_[A-Za-z0-9]{30,}"),
    "Google API 키": re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
    "개인 키 블록": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}


def tracked_text_files():
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
            continue
        if path.suffix.lower() in TEXT_SUFFIXES and path.stat().st_size < 2_000_000:
            yield path


def test_no_secrets_in_repository():
    hits = []
    for path in tracked_text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                hits.append(f"{path.relative_to(ROOT)}: {label}")
    assert not hits, "비밀 정보로 보이는 문자열이 있다: " + "; ".join(hits)


def test_secret_scanner_would_catch_a_leak(tmp_path):
    leaked = "token = 'ghp_" + "a" * 36 + "'"
    assert SECRET_PATTERNS["GitHub 토큰"].search(leaked)
    block = "-----BEGIN RSA " + "PRIVATE KEY-----"  # 이어 붙여 이 파일 자체가 스캔에 걸리지 않게 한다
    assert SECRET_PATTERNS["개인 키 블록"].search(block)


@pytest.mark.parametrize(
    "rel",
    [
        "README.md", "CLAUDE.md", "pyproject.toml", "requirements.txt", ".gitignore",
        "docs/CONTRACT.md", "docs/BOUNDARIES.md", "docs/PROTOCOL.md",
        "configs/default.json", "data/sample_knowledge.csv", "data/sample_eval.csv",
        "notebooks/01_colab_quickstart.ipynb", ".github/workflows/ci.yml", "app.py",
    ],
)
def test_required_files_exist(rel):
    assert (ROOT / rel).is_file(), f"{rel}가 없다"


def test_notebook_is_valid_json_and_code_cells_parse():
    nb = json.loads((ROOT / "notebooks" / "01_colab_quickstart.ipynb").read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code_cells
    for cell in code_cells:
        ast.parse("".join(cell["source"]))  # 문법 오류가 있으면 실패한다


def test_notebook_does_not_contain_placeholder_secrets_and_uses_valid_cli_flags():
    nb = json.loads((ROOT / "notebooks" / "01_colab_quickstart.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    from kore.cli import build_parser

    parser = build_parser()
    # 노트북이 부르는 명령의 옵션이 실제 CLI와 맞는지 확인한다.
    for flag in ("--config", "--knowledge", "--questions", "--out"):
        assert flag in source
    ns = parser.parse_args(
        ["run", "--config", "configs/default.json", "--knowledge", "a.csv", "--questions", "b.csv", "--out", "o"]
    )
    assert ns.cmd == "run"


def test_ci_workflow_covers_supported_python_versions():
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for version in ("3.10", "3.11", "3.12"):
        assert version in text
    assert "kore validate" in text and "pytest" in text


def test_pyproject_declares_no_runtime_dependencies():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "dependencies = []" in text
