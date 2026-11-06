# ============================================================================
# KORE 기준선 4조건 실행 셀 (Colab 단일 셀: 통째로 복사해서 붙여넣고 실행한다)
#
# 주의: 이 코드는 GPU가 없는 환경에서 만들었다. Colab(T4)에서 실제로 돌려 보지 못했다.
#       로컬에서는 CPU의 extractive 생성기로 명령 조립과 재개만 확인했다(README.md 참고).
#       HFGenerator의 모델 로딩과 생성은 첫 실행에서 확인해야 한다.
#
# 하는 일: Drive 마운트 -> 저장소 clone/pull -> pip install -e . -> GPU 점검 ->
#          `python -m kore run`(no_retrieval, vanilla_rag, kore, gold_context; dev 질문만) ->
#          요약 블록 1개 출력. 세션이 끊기면 같은 셀을 다시 실행하면 이어서 돌아간다.
# 넣지 않는 것: 비밀키, 토큰. 공개 저장소만 쓴다. test 분할은 돌리지 않는다(--final-test 없음).
# ============================================================================
from __future__ import annotations

import csv
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ----------------------------- 바꿔 쓰는 값 -----------------------------
REPO_URL = "https://github.com/BPM-KORE/BPM-KORE-project.git"  # 공개 저장소
BRANCH = "main"                      # PR #3이 합쳐지기 전이면 그 브랜치 이름으로 바꾼다
REPO_DIR = os.environ.get("KORE_REPO_DIR", "/content/BPM-KORE-project")
DRIVE_ROOT = os.environ.get("KORE_DRIVE_ROOT", "/content/drive/MyDrive")
RESULTS_DIRNAME = "KORE_results"
KNOWLEDGE = "team/provisional/law/knowledge.csv"
QUESTIONS = os.environ.get("KORE_QUESTIONS", "experiments/eval_normalization/evaluation_questions_20_v2.csv")
CONFIG = "configs/default.json"
GENERATOR_CONFIG = "configs/generator.json"
GENERATOR_KIND = os.environ.get("KORE_GENERATOR", "hf")   # 로컬 시험에서만 extractive/none으로 바꾼다
CONDITIONS = ["no_retrieval", "vanilla_rag", "kore", "gold_context"]
RUN_DATE = ""                        # 비워 두면 이어 돌릴 폴더가 있으면 그것을, 없으면 오늘 날짜를 쓴다
SKIP_SETUP = os.environ.get("KORE_SKIP_SETUP") == "1"     # 로컬 시험용(마운트, clone, pip 건너뜀)
# ------------------------------------------------------------------------


def sh(cmd: list[str], cwd: str | None = None, log: Path | None = None) -> int:
    if log is None:
        return subprocess.run(cmd, cwd=cwd).returncode
    with log.open("a", encoding="utf-8") as fh:
        fh.write("\n$ " + " ".join(cmd) + "\n")
        fh.flush()
        return subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT).returncode


def out(cmd: list[str], cwd: str | None = None) -> str:
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def main() -> None:
    # 1) Drive 마운트
    if not SKIP_SETUP:
        try:
            from google.colab import drive  # type: ignore
            if not os.path.isdir("/content/drive/MyDrive"):
                drive.mount("/content/drive")
        except ImportError:
            print("Colab이 아니다: Drive를 마운트하지 않는다. DRIVE_ROOT =", DRIVE_ROOT)
    if not os.path.isdir(DRIVE_ROOT):
        sys.exit(f"결과를 둘 폴더가 없다: {DRIVE_ROOT} (Drive 마운트를 확인한다)")

    # 2) 저장소 clone/pull, 설치
    if not SKIP_SETUP:
        if not Path(REPO_DIR, ".git").is_dir():
            if sh(["git", "clone", "--branch", BRANCH, REPO_URL, REPO_DIR]) != 0:
                sys.exit("git clone 실패: REPO_URL, BRANCH를 확인한다")
        else:
            if sh(["git", "fetch", "origin", BRANCH], cwd=REPO_DIR) != 0 or \
               sh(["git", "checkout", BRANCH], cwd=REPO_DIR) != 0 or \
               sh(["git", "pull", "--ff-only", "origin", BRANCH], cwd=REPO_DIR) != 0:
                sys.exit("git pull 실패: 저장소 폴더에 고친 파일이 있는지 본다")
        if sh([sys.executable, "-m", "pip", "install", "-q", "-e", "."], cwd=REPO_DIR) != 0:
            sys.exit("pip install -e . 실패")
    commit = out(["git", "rev-parse", "HEAD"], cwd=REPO_DIR)

    # 3) GPU 점검(hf 생성기일 때만 필요)
    gpu = "n/a"
    if GENERATOR_KIND == "hf":
        try:
            import torch  # type: ignore
            gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
        except ImportError:
            gpu = ""
        if not gpu:
            sys.exit("GPU를 찾지 못했다. 런타임 > 런타임 유형 변경에서 T4 GPU를 고르고 다시 실행한다.")

    # 4) 출력 폴더: Drive/KORE_results/<날짜>_<모델>_<revision 앞 7자리>
    spec = json.loads(Path(REPO_DIR, GENERATOR_CONFIG).read_text(encoding="utf-8")) if GENERATOR_KIND == "hf" else {}
    model = (spec.get("model") or GENERATOR_KIND).split("/")[-1]
    rev7 = (spec.get("revision") or "norev")[:7]
    root = Path(DRIVE_ROOT, RESULTS_DIRNAME)
    root.mkdir(parents=True, exist_ok=True)
    suffix = f"_{model}_{rev7}"
    if RUN_DATE:
        run_dir = root / f"{RUN_DATE}{suffix}"
    else:
        # 자정을 넘겨 이어 돌릴 때 폴더가 갈라지지 않게, 같은 모델·revision의 폴더 중 아직 끝나지 않은 최신 것을 찾는다
        open_runs = sorted(p for p in root.glob(f"*{suffix}") if (p / "run_log.jsonl").exists() and not (p / "SUMMARY_DONE").exists())
        run_dir = open_runs[-1] if open_runs else root / f"{_dt.date.today().isoformat()}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 5) kore run (같은 명령이라 끊겨도 이어진다: 끝난 (질문, 조건)은 건너뛴다)
    cmd = [sys.executable, "-m", "kore", "run", "--config", CONFIG, "--knowledge", KNOWLEDGE,
           "--questions", QUESTIONS, "--out", str(run_dir), "--conditions", *CONDITIONS]
    if GENERATOR_KIND == "hf":
        cmd += ["--generator", "hf", "--generator-config", GENERATOR_CONFIG]
    else:
        cmd += ["--generator", GENERATOR_KIND]
    log = run_dir / "run_stdout.log"   # 진행 출력은 파일에만 남긴다(토큰 아끼기)
    rc = sh(cmd, cwd=REPO_DIR, log=log)
    if rc != 0:
        print(f"kore run 실패(종료 코드 {rc}). 마지막 줄:")
        print("\n".join(log.read_text(encoding="utf-8").splitlines()[-8:]))
        print(f"같은 셀을 다시 실행하면 이어서 돈다. 전체 로그: {log}")
        return

    # 6) 요약 블록 1개
    env = json.loads((run_dir / "env.json").read_text(encoding="utf-8"))
    with (run_dir / "results.csv").open(encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["split"] == "all"]

    def f(v: str, nd: int = 3) -> str:
        return "-" if v in ("", None) else f"{float(v):.{nd}f}"

    gen_name = env.get("generator", "none")
    dtype = gen_name.split("|")[1] if gen_name.startswith("hf:") and gen_name.count("|") >= 2 else spec.get("dtype", "n/a")
    lines = [
        "=== KORE 기준선 요약 ===",
        f"폴더: {run_dir}",
        f"커밋: {commit[:12] or env.get('git_commit')}  브랜치: {BRANCH}  질문 {env['n_questions']}개(test 제외: {not env['include_test']})",
        f"생성기: {gen_name}",
        f"모델: {spec.get('model', GENERATOR_KIND)}  revision: {spec.get('revision', '-')}  dtype: {dtype}  GPU: {gpu}",
        f"설정: k={env['config']['k']} candidate_k={env['config']['candidate_k']} token_budget={env['config']['token_budget']} "
        f"tokenizer={env['config']['tokenizer_mode']} seed={env['seed']}",
        f"{'조건':<14}{'n':>3} {'rubric':>7} {'recall':>7} {'hit':>6} {'ctx_tok':>8} {'생성ms/호출':>12}",
    ]
    for r in rows:
        lines.append(f"{r['condition']:<14}{r['n']:>3} {f(r['rubric_score']):>7} {f(r['recall']):>7} {f(r['hit']):>6} "
                     f"{f(r['context_tokens'], 0):>8} {f(r['generate_ms'], 0):>12}")
    lines.append("rubric은 키워드 일치(의미 평가의 대용치). 20문항이라 차이를 결론으로 쓰지 않는다. 샘플 아닌 dev 질문이지만 연구 결과 주장 전 별도 검토가 필요하다.")
    print("\n".join(lines))
    (run_dir / "SUMMARY_DONE").write_text(_dt.datetime.now().isoformat(), encoding="utf-8")
    (run_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


main()
