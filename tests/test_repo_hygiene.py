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
        "notebooks/01_colab_quickstart.ipynb", "notebooks/02_colab_gpu_run.ipynb", ".github/workflows/ci.yml", "app.py",
        "configs/generator.json", "docs/DECISIONS.md", "docs/TASKS.md", "docs/START.md", "configs/hybrid.json",
            "notebooks/03_colab_data_tools.ipynb",
    ],
)
def test_required_files_exist(rel):
    assert (ROOT / rel).is_file(), f"{rel}가 없다"


NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_valid_json_and_code_cells_parse(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code_cells
    for cell in code_cells:
        ast.parse("".join(cell["source"]))  # 문법 오류가 있으면 실패한다
        assert not cell.get("outputs"), "실행 결과(출력)를 노트북에 남기지 않는다"


def _string_constants(source: str) -> list[str]:
    return [n.value for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_cli_flags_exist_in_the_real_cli(path):
    """노트북이 부르는 `kore <명령> --옵션`이 실제 CLI에 있는지 확인한다(오타나 옛 옵션이 Colab에서 터지는 것을 막는다)."""
    from kore.cli import build_parser

    sub = build_parser()._subparsers._group_actions[0].choices
    nb = json.loads(path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        consts = _string_constants("".join(cell["source"]))
        commands = set()
        for c in consts:
            commands |= set(re.findall(r"-m kore ([a-z][a-z\-]*)", c))
        for cmd in commands:
            assert cmd in sub, f"{path.name}: 없는 명령 {cmd}"
        if not commands:
            continue
        allowed = set()
        for cmd in commands:
            allowed |= {o for a in sub[cmd]._actions for o in a.option_strings}
        used = set()
        for c in consts:
            if "git " in c or "pip install" in c:  # git·pip 옵션은 kore와 무관하다
                continue
            used |= set(re.findall(r"(?<![\w-])(--[a-z][a-z\-]*)", c))
        unknown = {u for u in used if u not in allowed}
        assert not unknown, f"{path.name}: CLI에 없는 옵션 {sorted(unknown)} (명령 {sorted(commands)})"


def test_notebooks_do_not_embed_secrets_and_use_colab_secrets_for_tokens():
    for path in NOTEBOOKS:
        text = path.read_text(encoding="utf-8")
        for label, pattern in SECRET_PATTERNS.items():
            assert not pattern.search(text), f"{path.name}: {label}"
    gpu = json.loads((ROOT / "notebooks" / "02_colab_gpu_run.ipynb").read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in gpu["cells"] if c["cell_type"] == "code")
    assert "userdata" in code and "GITHUB_TOKEN" in code
    # 실행 코드가 test 분할을 켜는 옵션을 넘기지 않는다(주석과 설명 글은 제외하고 문자열 상수만 본다)
    assert not any("--final-test" in c for c in _string_constants(code))


def test_ci_workflow_covers_supported_python_versions():
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for version in ("3.10", "3.11", "3.12"):
        assert version in text
    assert "kore validate" in text and "pytest" in text


def test_pyproject_declares_no_runtime_dependencies():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "dependencies = []" in text
