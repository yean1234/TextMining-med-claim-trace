"""RQ1 (인과 표현 강화)·RQ2 (조건 누락)·제목-본문 불일치 비교표."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .dataset import condition_retention
from .stats import cluster_bootstrap, gee_logit

RATE_COLS = {
    "exceeds_title": "제목이 논문보다 강함",
    "exceeds_lead": "리드가 논문보다 강함",
    "causal_inflation_title": "상관→인과 강화(관찰연구, 제목)",
    "direction_mismatch": "방향 불일치(증가/감소/무관)",
    "strong_title": "제목 강도 ≥5 (can/단정적 인과)",
    "title_body_gap": "강한 제목 + 본문엔 주의 문장",
}


def fmt_ci(est: float, lo: float, hi: float, pct: bool = True) -> str:
    if est is None or np.isnan(est):
        return "–"
    k = 100 if pct else 1
    core = f"{est * k:.1f}{'%' if pct else ''}"
    if np.isnan(lo):
        return core
    return f"{core} [{lo * k:.1f}, {hi * k:.1f}]"


def overall_rates(df: pd.DataFrame, n_boot: int = 2000) -> pd.DataFrame:
    rows = []
    for col, label in RATE_COLS.items():
        est, lo, hi, n = cluster_bootstrap(df, col, n_boot=n_boot)
        rows.append({"지표": label, "변수": col, "n(판단 가능)": n, "비율 [95% 묶음 부트스트랩 CI]": fmt_ci(est, lo, hi),
                     "_est": est})
    return pd.DataFrame(rows)


def by_outlet_type(df: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    rows = []
    for otype, g in df.groupby("outlet_type"):
        row = {"outlet_type": otype, "기사 수": len(g), "묶음 수": g["bundle_id"].nunique(),
               "제목 강도 평균": round(g["title_strength"].mean(), 2),
               "Δ(제목−논문) 평균": round(g["delta_title"].mean(), 2)}
        for col in ("exceeds_title", "title_body_gap", "strong_title"):
            est, lo, hi, n = cluster_bootstrap(g, col, n_boot=n_boot)
            row[RATE_COLS[col]] = fmt_ci(est, lo, hi)
        row["선정 표현(제목) 평균"] = round(g["n_sensational_title"].mean(), 2) if "n_sensational_title" in g else ""
        row["의문형 제목 %"] = round(g["auto_title_question"].mean() * 100, 1) if "auto_title_question" in g else ""
        rows.append(row)
    return pd.DataFrame(rows).sort_values("기사 수", ascending=False)


def bundle_ladder(df: pd.DataFrame) -> pd.DataFrame:
    """묶음별: 논문 강도 vs 기사 제목 강도 분포 (사례2처럼 '제목 분산이 큰 묶음' 찾기)."""
    rows = []
    for bid, g in df.groupby("bundle_id"):
        ts = g["title_strength"].dropna()
        rows.append({
            "bundle_id": bid, "설계": g["design"].iloc[0], "논문 강도": g["paper_strength"].iloc[0],
            "보도자료 강도": g["press_release_strength"].iloc[0] if "press_release_strength" in g else np.nan,
            "기사 수": len(g), "제목 강도 min": ts.min(), "제목 강도 max": ts.max(),
            "제목 강도 평균": round(ts.mean(), 2), "제목 강도 SD": round(ts.std(ddof=0), 2),
            "논문보다 강한 제목": int((g["exceeds_title"] == 1).sum()),
            "매체 유형": ", ".join(sorted(g["outlet_type"].unique())),
        })
    return pd.DataFrame(rows)


def title_vs_lead(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["title_strength", "lead_strength"])
    if d.empty:
        return pd.DataFrame()
    return pd.DataFrame([{
        "n": len(d),
        "제목 > 리드 %": round((d["title_minus_lead"] > 0).mean() * 100, 1),
        "제목 = 리드 %": round((d["title_minus_lead"] == 0).mean() * 100, 1),
        "제목 < 리드 %": round((d["title_minus_lead"] < 0).mean() * 100, 1),
        "평균 (제목−리드)": round(d["title_minus_lead"].mean(), 2),
    }])


def press_release_conditional(df: pd.DataFrame) -> pd.DataFrame | None:
    """Sumner 식 질문: 보도자료가 논문보다 강할 때 기사 제목도 강한가? (보도자료 강도가 코딩된 묶음만)"""
    d = df.dropna(subset=["press_release_strength", "paper_strength", "exceeds_title"])
    if d["bundle_id"].nunique() < 3:
        return None
    d = d.assign(pr_exceeds=(d["press_release_strength"] > d["paper_strength"]).astype(int))
    rows = []
    for v, g in d.groupby("pr_exceeds"):
        est, lo, hi, n = cluster_bootstrap(g, "exceeds_title")
        rows.append({"보도자료가 논문보다 강함": "예" if v else "아니오", "기사 수": n,
                     "제목이 논문보다 강한 비율": fmt_ci(est, lo, hi)})
    return pd.DataFrame(rows)


def condition_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    long = condition_retention(df)
    if long.empty:
        return long, pd.DataFrame()
    summary = long.groupby("condition").agg(
        n=("article_id", "size"),
        제목에_남음=("in_title", "mean"), 리드에_남음=("in_lead", "mean"), 본문에_남음=("in_body", "mean"),
        제목에서만_빠짐=("dropped_in_title_kept_in_body", "mean"), 어디에도_없음=("absent_everywhere", "mean"),
    ).reset_index()
    for c in summary.columns[2:]:
        summary[c] = (summary[c] * 100).round(1)
    return long, summary


def gee_outlet(df: pd.DataFrame) -> tuple[pd.DataFrame | None, str]:
    return gee_logit(df, "exceeds_title", "outlet_type")
