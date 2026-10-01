"""RQ3: 조건이 생략되거나 단정적인 제목이 붙은 기사에서 댓글의 불안·불신 표현이 더 많이 관찰되는가?

측정하는 것은 '댓글에 드러난 표현'이다. 독자 전체의 심리나 기사가 감정을 유발했다는 인과는
측정하지 않는다. 기사마다 댓글 수가 크게 달라서 기사 단위 비율(기사당 가중치 동일)로 비교한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from ..nlp.lexicon import Lexicon
from ..paths import Workspace
from ..utils import read_jsonl
from .stats import cluster_bootstrap_diff

REACTIONS = ["anxiety", "distrust", "reassurance"]


def code_comments(ws: Workspace, lex: Lexicon) -> pd.DataFrame:
    rows = read_jsonl(ws.comments_jsonl)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for r in REACTIONS:
        df[r] = df["text"].map(lambda t, r=r: int(lex.any(t, "comment_reactions", r)))
    return df


def per_article(comments: pd.DataFrame) -> pd.DataFrame:
    if comments.empty:
        return pd.DataFrame()
    agg = comments.groupby("article_id").agg(n_comments=("text", "size"),
                                             **{f"share_{r}": (r, "mean") for r in REACTIONS})
    return agg.reset_index()


def compare_groups(articles: pd.DataFrame, per_art: pd.DataFrame, min_comments: int = 5) -> pd.DataFrame:
    """제목 특성(강한 제목/논문보다 강한 제목/조건 생략 오해) 있음 vs 없음 → 댓글 반응 비율 차이."""
    if per_art.empty:
        return pd.DataFrame()
    d = articles.merge(per_art, on="article_id", how="inner")
    d = d[d["n_comments"] >= min_comments].copy()
    if "title_omission_misleading" in d.columns:
        tom = d["title_omission_misleading"].astype(str).str.upper()
        d["omission_misleading"] = np.where(tom.isin(["Y", "N"]), tom.eq("Y").astype(float), np.nan)
    rows = []
    for group, label in (("strong_title", "제목 강도 ≥5"), ("exceeds_title", "제목이 논문보다 강함"),
                         ("omission_misleading", "제목 조건 생략이 오해 유발")):
        if group not in d.columns:
            continue
        g = d.dropna(subset=[group])
        if g[group].nunique() < 2:
            continue
        for r in REACTIONS:
            col = f"share_{r}"
            a, b = g.loc[g[group] == 1, col], g.loc[g[group] == 0, col]
            diff, lo, hi = cluster_bootstrap_diff(g, col, group)
            p = mannwhitneyu(a, b).pvalue if len(a) >= 3 and len(b) >= 3 else np.nan
            rows.append({"비교": label, "반응": r, "기사 수(있음/없음)": f"{len(a)}/{len(b)}",
                         "있음 평균 %": round(a.mean() * 100, 1), "없음 평균 %": round(b.mean() * 100, 1),
                         "차이 %p [95% CI]": "–" if np.isnan(diff) else
                         f"{diff * 100:.1f} [{lo * 100:.1f}, {hi * 100:.1f}]" if not np.isnan(lo)
                         else f"{diff * 100:.1f}",
                         "Mann-Whitney p": round(p, 4) if not np.isnan(p) else "–"})
    return pd.DataFrame(rows)
