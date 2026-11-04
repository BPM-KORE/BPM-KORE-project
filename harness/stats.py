"""작은 dev 세트(수십 문항 이하)의 통계적 한계를 보고서에 자동으로 적는다.

부트스트랩 구간은 kore.evaluation.bootstrap_ci를 그대로 쓴다. 구간이 0을 포함하면 '구분할 수 없다'고 쓴다.
구간이 0을 포함하지 않아도 '개선이 입증됐다'고 쓰지 않는다. 20문항 구간은 좁게 나오기 쉽고 후보를 여러 개 시험했다면 최고 후보는 과대평가된다.
"""

from __future__ import annotations

from typing import Any

from kore.evaluation import approx_mde, bootstrap_ci

SMALL_SAMPLE = 30
N_BOOT = 2000


def dev_limits(
    diffs: list[float | None], n_questions: int, n_compared: int = 1, seed: int = 0
) -> dict[str, Any]:
    """문항별 점수 차이(편집 - 기준)로 구간과 경고를 만든다. None인 문항은 빼고 센다."""
    vals = [d for d in diffs if d is not None]
    n = len(vals)
    mean, lo, hi = bootstrap_ci(vals, n_boot=N_BOOT, seed=seed)
    changed = sum(1 for d in vals if abs(d) > 1e-12)
    mde = None
    if n >= 2:
        m = sum(vals) / n
        sd = (sum((d - m) ** 2 for d in vals) / (n - 1)) ** 0.5
        mde = approx_mde(n, sd) if sd > 0 else None
    warnings: list[str] = []
    if n < SMALL_SAMPLE:
        warnings.append(f"문항이 {n}개뿐이다(전체 {n_questions}개 중 점수가 있는 쌍). {SMALL_SAMPLE}개 미만이라 작은 차이를 구분하기 어렵다")
    if n == 0:
        warnings.append("비교할 수 있는 문항이 없다. 구간을 계산하지 못했다")
        indistinguishable = True
    elif lo is not None and hi is not None and lo <= 0.0 <= hi:
        indistinguishable = True
        warnings.append(f"95% 부트스트랩 구간 [{lo:+.3f}, {hi:+.3f}]에 0이 들어 있다. 이 구간이면 차이를 구분할 수 없다")
    else:
        indistinguishable = False
        warnings.append(
            f"구간 [{lo:+.3f}, {hi:+.3f}]이 0을 포함하지 않는다. 그래도 개선이 입증된 것은 아니다. "
            "문항이 적어 구간이 실제보다 좁을 수 있고, 같은 dev로 편집을 고르면 과적합한다"
        )
    if n and changed == 0:
        warnings.append("모든 문항에서 점수가 같았다. 구간 폭 0은 '정밀하다'가 아니라 '차이가 관찰되지 않았다'는 뜻이다")
    elif n and changed < max(3, n // 4):
        warnings.append(f"점수가 달라진 문항이 {changed}/{n}개뿐이다. 소수 문항이 평균을 좌우한다")
    if n_compared > 1:
        warnings.append(f"같은 dev 문항으로 편집 {n_compared}개를 비교했다. 그중 최고 후보를 고르면 차이가 과대평가된다(다중 비교)")
    if mde is not None and mean is not None and abs(mean) < mde:
        warnings.append(f"관찰한 평균 차이 {mean:+.3f}가 이 표본의 검출 가능 최소 차이 약 {mde:.3f}보다 작다(근사값)")
    return {
        "n": n, "n_questions": n_questions, "n_changed": changed, "mean_diff": mean, "ci_lo": lo, "ci_hi": hi,
        "mde_approx": mde, "indistinguishable": indistinguishable, "warnings": warnings,
    }


def format_limits(label: str, lim: dict[str, Any]) -> str:
    """보고서에 들어갈 마크다운 한 덩어리."""
    def f(v: float | None) -> str:
        return "-" if v is None else f"{v:+.3f}"

    head = (
        f"- 평균 차이 {f(lim['mean_diff'])}, 95% 구간 [{f(lim['ci_lo'])}, {f(lim['ci_hi'])}], "
        f"문항 {lim['n']}개(점수가 달라진 문항 {lim['n_changed']}개)"
    )
    verdict = "구분 불가" if lim["indistinguishable"] else "구간이 0을 벗어남(입증 아님)"
    lines = [f"**{label}** 판정: {verdict}", head] + [f"  - 경고: {w}" for w in lim["warnings"]]
    return "\n".join(lines)
