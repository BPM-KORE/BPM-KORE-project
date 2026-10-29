"""환경 점검. Colab, 노트북, 어디서든 '내 환경이 준비됐는가'를 한 번에 본다."""

from __future__ import annotations

import importlib
import platform
import shutil
import sys
from pathlib import Path


def _version(module: str) -> str | None:
    try:
        mod = importlib.import_module(module)
    except Exception:  # 설치되지 않았거나 불러오다 실패한 경우 모두 '없음'으로 본다
        return None
    return str(getattr(mod, "__version__", "?"))


def check_environment(data_dir: str | Path = "data") -> list[tuple[str, str, str]]:
    """(상태, 항목, 설명) 목록. 상태는 OK / 주의 / 실패."""
    out: list[tuple[str, str, str]] = []
    py = sys.version_info
    out.append(("OK" if py >= (3, 10) else "실패", "파이썬", f"{platform.python_version()} (3.10 이상 필요)"))

    torch_v, tf_v = _version("torch"), _version("transformers")
    out.append(("OK" if torch_v else "주의", "torch", torch_v or "없음 — 생성·기여도 라벨링에만 필요하다. pip install -e \".[hf]\""))
    out.append(("OK" if tf_v else "주의", "transformers", tf_v or "없음 — 생성·기여도 라벨링에만 필요하다"))
    if torch_v:
        try:
            import torch

            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                out.append(("OK", "GPU", f"{name}, {mem:.1f}GB"))
                bf16 = torch.cuda.is_bf16_supported()
                out.append(("OK", "dtype", "bfloat16 지원" if bf16 else "bfloat16 미지원 → float16을 쓴다(configs/generator.json 기본값)"))
            else:
                out.append(("주의", "GPU", "없음 — 검색·선택 실험은 CPU로 되지만 소형 모델 생성은 매우 느리다"))
        except Exception as exc:  # pragma: no cover - 드라이버 문제 등
            out.append(("주의", "GPU", f"확인하지 못했다: {exc}"))
    elif shutil.which("nvidia-smi"):
        out.append(("주의", "GPU", "nvidia-smi는 있는데 torch가 없다"))

    free_gb = shutil.disk_usage(".").free / 1024**3
    out.append(("OK" if free_gb >= 15 else "주의", "디스크", f"여유 {free_gb:.1f}GB (4B 모델 내려받기에 10GB 안팎 필요)"))

    d = Path(data_dir)
    for name in ("knowledge.csv", "evaluation.csv", "source_log.csv"):
        out.append(("OK" if (d / name).exists() else "주의", f"data/{name}", "있음" if (d / name).exists() else "아직 없음"))
    return out


def format_environment(rows: list[tuple[str, str, str]]) -> str:
    return "\n".join(f"[{s}] {name}: {desc}" for s, name, desc in rows)
