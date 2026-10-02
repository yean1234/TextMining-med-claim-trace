"""분석용 데이터셋: 기사 특징 + 기사 코드(사람/자동) + 논문 코드 → 파생 변수.

source
  final   annotation/article_coding_final.csv (두 코더 조정 후) — 본 분석
  <코더명> 한 코더의 시트 (예비 분석)
  auto    규칙 기반 자동 코드만 — 파이프라인 점검용. 사전(lexicon)으로 과장을 판정하고 다시 같은
          사전 특징을 비교하면 순환 논리가 되므로 결론에 쓰면 안 된다.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..coding.features import CONDITION_CATEGORIES
from ..coding.precode import ARTICLE_HUMAN_COLS
from ..paths import Workspace
from ..utils import log, read_csv

OBSERVATIONAL = {"observational", "pharmacovigilance", "case_report"}
DIR_MAP = {"increase": "inc", "decrease": "dec", "null": "null", "mixed": "unclear", "": ""}


def load_codes(ws: Workspace, source: str, features: pd.DataFrame) -> pd.DataFrame:
    if source == "auto":
        return pd.DataFrame({
            "article_id": features["article_id"],
            "title_strength": features["auto_title_strength"],
            "lead_strength": features["auto_lead_strength"],
            "direction": features["auto_direction"].astype(str).map(lambda d: DIR_MAP.get(d, d)),
            "advice": features["auto_advice"],
            "caveat_in_body": features["auto_caveat"].astype(str).map({"1": "Y", "0": "N"}),
        })
    path = ws.final_coding_csv if source == "final" else ws.coding_csv(source)
    if not path.exists():
        raise FileNotFoundError(f"{path} 가 없습니다. (--source auto 로 파이프라인만 점검할 수 있음)")
    codes = read_csv(path)
    cols = ["article_id"] + [c for c in ARTICLE_HUMAN_COLS if c in codes.columns]
    codes = codes[cols]
    coded = codes["title_strength"].astype(str).str.strip() != ""
    if (~coded).any():
        log.warning("%s: 제목 강도가 비어 있는 기사 %d건은 분석에서 빠집니다.", path.name, int((~coded).sum()))
    return codes[coded]


def load_papers(ws: Workspace, allow_auto: bool) -> pd.DataFrame:
    papers = read_csv(ws.paper_coding_csv)

    def pick(human: str, auto: str) -> pd.Series:
        h = papers[human].astype(str).str.strip() if human in papers.columns else pd.Series("", index=papers.index)
        a = papers[auto].astype(str).str.strip() if auto in papers.columns else pd.Series("", index=papers.index)
        return h.where(h != "", a if allow_auto else "")

    out = pd.DataFrame({"bundle_id": papers["bundle_id"]})
    out["paper_strength"] = pick("paper_strength", "strength_auto")
    out["paper_direction"] = pick("paper_direction", "direction_auto")
    out["design"] = pick("design", "design_auto")
    out["key_conditions"] = papers.get("key_conditions", pd.Series("", index=papers.index))
    out["press_release_strength"] = papers.get("press_release_strength", pd.Series("", index=papers.index))
    out["paper_checked_by"] = papers.get("checked_by", pd.Series("", index=papers.index))
    out["paper_code_source"] = np.where(
        papers.get("paper_strength", pd.Series("", index=papers.index)).astype(str).str.strip() != "",
        "coded", "auto" if allow_auto else "missing")
    return out


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.replace("", np.nan), errors="coerce")


def _flag(cond: pd.Series, valid: pd.Series) -> pd.Series:
    """참/거짓을 1.0/0.0, 판단 불가는 NaN."""
    return pd.Series(np.where(valid, cond.astype(float), np.nan), index=cond.index)


def build_dataset(ws: Workspace, source: str = "final", paper_auto_fallback: bool = False) -> pd.DataFrame:
    """paper_auto_fallback=True 면 사람이 아직 안 채운 논문 강도·설계를 자동 제안값으로 대신한다(예비 분석용)."""
    feats = read_csv(ws.features_csv) if ws.features_csv.exists() else pd.DataFrame()
    if feats.empty:
        raise RuntimeError("기사 특징(article_features.csv)이 비어 있습니다. scrape → precode 를 먼저 실행하세요.")
    codes = load_codes(ws, source, feats)
    papers = load_papers(ws, allow_auto=(source == "auto" or paper_auto_fallback))
    df = feats.merge(codes, on="article_id", how="inner").merge(papers, on="bundle_id", how="left").copy()
    for c in ["title_strength", "lead_strength", "paper_strength", "advice", "press_release_strength",
              "auto_title_strength", "auto_lead_strength", "auto_body_max_strength", "body_len",
              "n_sensational_title", "n_hedges_title", "n_boosters_title", "n_hedges_body",
              "n_boosters_body", "n_sensational_body", "auto_title_question"]:
        if c in df.columns:
            df[c] = _num(df[c])

    df = df.copy()   # 열이 많아 조각난 프레임을 정리
    has_t = df["title_strength"].notna() & df["paper_strength"].notna()
    has_l = df["lead_strength"].notna() & df["paper_strength"].notna()
    df["delta_title"] = df["title_strength"] - df["paper_strength"]
    df["delta_lead"] = df["lead_strength"] - df["paper_strength"]
    df["exceeds_title"] = _flag(df["delta_title"] > 0, has_t)
    df["exceeds_lead"] = _flag(df["delta_lead"] > 0, has_l)
    obs = df["design"].isin(OBSERVATIONAL)
    df["causal_inflation_title"] = _flag((df["paper_strength"] <= 3) & (df["title_strength"] >= 4),
                                         has_t & obs & (df["paper_strength"] <= 3))
    d_ok = df["direction"].isin(["inc", "dec", "null"]) & df["paper_direction"].isin(["inc", "dec", "null"])
    df["direction_mismatch"] = _flag(df["direction"] != df["paper_direction"], d_ok)
    df["strong_title"] = _flag(df["title_strength"] >= 5, df["title_strength"].notna())
    cav = df["caveat_in_body"].astype(str).str.upper()
    df["title_body_gap"] = _flag((df["title_strength"] >= 5) & cav.eq("Y"),
                                 df["title_strength"].notna() & cav.isin(["Y", "N"]))
    df["title_minus_lead"] = df["title_strength"] - df["lead_strength"]
    df["code_source"] = source
    return df


def parse_key_conditions(value: str) -> list[str]:
    parts = [p.strip().lower() for p in re.split(r"[;,/\s]+", str(value or "")) if p.strip()]
    return [p for p in parts if p in CONDITION_CATEGORIES]


def condition_retention(df: pd.DataFrame) -> pd.DataFrame:
    """논문의 '핵심 조건' 범주가 기사 제목/리드/본문에 남아 있는지 (기사 × 조건 단위)."""
    rows = []
    for _, r in df.iterrows():
        for cat in parse_key_conditions(r.get("key_conditions", "")):
            def has(sec: str) -> int:
                return int(str(r.get(f"auto_cond_{sec}_{cat}", "0")) == "1")
            rows.append({"article_id": r["article_id"], "bundle_id": r["bundle_id"],
                         "outlet_type": r["outlet_type"], "condition": cat,
                         "in_title": has("title"), "in_lead": has("lead"), "in_body": has("body")})
    out = pd.DataFrame(rows, columns=["article_id", "bundle_id", "outlet_type", "condition",
                                      "in_title", "in_lead", "in_body"])
    if len(out):
        out["dropped_in_title_kept_in_body"] = ((out["in_title"] == 0) & (out["in_body"] == 1)).astype(int)
        out["absent_everywhere"] = ((out["in_title"] == 0) & (out["in_lead"] == 0) &
                                    (out["in_body"] == 0)).astype(int)
    return out
