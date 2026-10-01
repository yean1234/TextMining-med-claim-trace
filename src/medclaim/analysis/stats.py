"""묶음(같은 논문) 단위 의존성을 고려한 통계.

같은 논문을 보도한 기사들은 서로 독립이 아니다(Sumner 는 GEE 사용). 묶음 20~25개 규모에서는
GEE 추정이 불안정할 수 있어, 해석이 쉬운 '묶음 단위 부트스트랩' 신뢰구간을 기본으로 쓰고
GEE 는 보조로 보고한다.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def cluster_bootstrap(df: pd.DataFrame, col: str, cluster: str = "bundle_id", n_boot: int = 2000,
                      seed: int = 0) -> tuple[float, float, float, int]:
    """(평균, 2.5%, 97.5%, n). 묶음을 복원추출해 평균의 신뢰구간을 구한다."""
    d = df[[cluster, col]].dropna()
    n = len(d)
    if n == 0:
        return (np.nan, np.nan, np.nan, 0)
    est = float(d[col].mean())
    groups = [g[col].to_numpy(dtype=float) for _, g in d.groupby(cluster)]
    if len(groups) < 2:
        return (est, np.nan, np.nan, n)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(groups), len(groups))
        vals = np.concatenate([groups[i] for i in idx])
        stats.append(vals.mean())
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return (est, float(lo), float(hi), n)


def cluster_bootstrap_diff(df: pd.DataFrame, col: str, group: str, cluster: str = "bundle_id",
                           n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """group(0/1)별 col 평균의 차이(1 − 0)와 묶음 부트스트랩 95% CI."""
    d = df[[cluster, col, group]].dropna()
    if d[group].nunique() < 2:
        return (np.nan, np.nan, np.nan)
    est = float(d.loc[d[group] == 1, col].mean() - d.loc[d[group] == 0, col].mean())
    clusters = [g for _, g in d.groupby(cluster)]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        sample = pd.concat([clusters[i] for i in rng.integers(0, len(clusters), len(clusters))])
        a, b = sample.loc[sample[group] == 1, col], sample.loc[sample[group] == 0, col]
        if len(a) and len(b):
            diffs.append(a.mean() - b.mean())
    if len(diffs) < n_boot * 0.5:
        return (est, np.nan, np.nan)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return (est, float(lo), float(hi))


def gee_logit(df: pd.DataFrame, outcome: str, predictor: str, cluster: str = "bundle_id",
              min_n: int = 30, min_clusters: int = 8) -> tuple[pd.DataFrame | None, str]:
    """이항 GEE(교환 가능 상관). 조건이 안 맞으면 (None, 이유)."""
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except ImportError:  # pragma: no cover
        return None, "statsmodels 미설치"
    d = df[[outcome, predictor, cluster]].dropna().copy()
    if len(d) < min_n:
        return None, f"표본 부족 (n={len(d)} < {min_n})"
    if d[cluster].nunique() < min_clusters:
        return None, f"묶음 수 부족 ({d[cluster].nunique()} < {min_clusters})"
    if d[outcome].nunique() < 2:
        return None, "결과 변수가 한 값뿐"
    counts = d[predictor].value_counts()
    rare = counts[counts < 5].index
    if len(rare):
        d[predictor] = d[predictor].where(~d[predictor].isin(rare), "other")
    if d[predictor].nunique() < 2:
        return None, "예측 변수 범주가 하나뿐"
    rates = d.groupby(predictor)[outcome].mean()
    separated = rates[(rates == 0) | (rates == 1)]
    if len(separated):
        return None, ("완전 분리: " + ", ".join(f"{k}={v:.0%}" for k, v in separated.items())
                      + " → 오즈비를 추정할 수 없음. 위 비율 표(부트스트랩 CI)를 보세요.")
    ref = d[predictor].value_counts().index[0]
    formula = f"{outcome} ~ C({predictor}, Treatment(reference='{ref}'))"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = smf.gee(formula, groups=cluster, data=d, family=sm.families.Binomial(),
                            cov_struct=sm.cov_struct.Exchangeable())
            res = model.fit()
        except Exception as e:  # noqa: BLE001 — 작은 표본에서 수렴 실패가 흔함
            return None, f"GEE 적합 실패: {type(e).__name__}: {e}"
    ci = res.conf_int()
    table = pd.DataFrame({
        "term": res.params.index,
        "odds_ratio": np.exp(res.params.values).round(3),
        "ci_low": np.exp(ci[0].values).round(3),
        "ci_high": np.exp(ci[1].values).round(3),
        "p_value": res.pvalues.values.round(4),
    })
    return table, f"GEE binomial, exchangeable, reference={ref}, n={len(d)}, 묶음={d[cluster].nunique()}"
