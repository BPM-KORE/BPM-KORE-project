import json
from pathlib import Path

import pytest

from kore.cli import main
from kore.pipeline import PipelineConfig, load_config

ROOT = Path(__file__).resolve().parent.parent


def test_default_config_file_matches_code_defaults():
    # configs/default.json과 코드 기본값이 어긋나면 '파일에 적힌 설정'이 거짓말이 된다.
    cfg, run_level = load_config(ROOT / "configs" / "default.json")
    assert cfg.to_dict() == PipelineConfig().to_dict()
    assert run_level == {"seed": 0, "conditions": ["no_retrieval", "vanilla_rag", "kore"]}


def test_from_dict_rejects_unknown_keys():
    with pytest.raises(ValueError, match="모르는 키"):
        PipelineConfig.from_dict({"candidate_k": 10, "kk": 3})
    with pytest.raises(ValueError, match="weights"):
        PipelineConfig.from_dict({"weights": {"diversty": 0.1}})
    with pytest.raises(ValueError):
        PipelineConfig.from_dict({"k": 5, "candidate_k": 3})


def test_load_config_rejects_unknown_condition(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"conditions": ["kore", "bogus"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="bogus"):
        load_config(p)


def test_cli_run_uses_config_and_flags_override(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"k": 2, "seed": 5, "conditions": ["vanilla_rag"]}), encoding="utf-8")
    out = tmp_path / "run"
    base = ["run", "--knowledge", "data/sample_knowledge.csv", "--questions", "data/sample_eval.csv"]
    assert main(base + ["--out", str(out), "--config", str(cfg)]) == 0
    env = json.loads((out / "env.json").read_text(encoding="utf-8"))
    assert env["config"]["k"] == 2 and env["seed"] == 5 and env["conditions"] == ["vanilla_rag"]
    out2 = tmp_path / "run2"
    assert main(base + ["--out", str(out2), "--config", str(cfg), "--k", "1", "--seed", "9"]) == 0
    env2 = json.loads((out2 / "env.json").read_text(encoding="utf-8"))
    assert env2["config"]["k"] == 1 and env2["seed"] == 9
