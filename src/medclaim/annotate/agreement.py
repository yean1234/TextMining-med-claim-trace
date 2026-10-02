"""코더 간 일치도, 자동 코딩의 타당도, 불일치 목록, 최종 코드 확정.

Sumner et al.(2014): 보도자료·논문 27%, 뉴스 21%를 두 명이 중복 코딩, 일치율 91%, κ=0.88.
우리도 최소 20~30%를 두 명이 독립 코딩하고 κ 를 보고한 뒤 불일치를 토론으로 조정한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from ..coding.precode import ARTICLE_HUMAN_COLS
from ..paths import Workspace
from ..utils import log, merge_preserving, read_csv, write_csv

FIELDS = {
    "title_strength": "ordinal",
    "lead_strength": "ordinal",
    "direction": "nominal",
    "advice": "ordinal",
    "caveat_in_body": "nominal",
    "title_omission_misleading": "nominal",
    "animal_generalization": "ordinal",
    "source_error": "nominal",
}
AUTO_EQUIV = {  # 사람 코드 열 → features 의 자동 열
    "title_strength": "auto_title_strength",
    "lead_strength": "auto_lead_strength",
    "direction": "auto_direction",
    "advice": "auto_advice",
    "caveat_in_body": "auto_caveat",
}
DIR_MAP = {"increase": "inc", "decrease": "dec", "null": "null", "mixed": "unclear", "": ""}


def _clean(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper().replace({"NAN": ""})


def kappa_row(a: pd.Series, b: pd.Series, field: str, kind: str) -> dict:
    a, b = _clean(a), _clean(b)
    mask = (a != "") & (b != "") & (a != "NA") & (b != "NA")
    a, b = a[mask], b[mask]
    n = int(mask.sum())
    row = {"field": field, "n": n, "pct_agree": np.nan, "kappa": np.nan, "weighted_kappa": np.nan}
    if n == 0:
        return row
    row["pct_agree"] = round(float((a == b).mean()) * 100, 1)
    if len(set(a) | set(b)) > 1:
        row["kappa"] = round(float(cohen_kappa_score(a, b)), 3)
        if kind == "ordinal":
            try:
                row["weighted_kappa"] = round(float(cohen_kappa_score(a.astype(int), b.astype(int),
                                                                      weights="linear")), 3)
            except ValueError:
                pass
    return row


def compare_coders(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = df1.merge(df2, on="article_id", suffixes=("_c1", "_c2"))
    rows, dis = [], []
    for field, kind in FIELDS.items():
        if f"{field}_c1" not in m.columns:
            continue
        rows.append(kappa_row(m[f"{field}_c1"], m[f"{field}_c2"], field, kind))
        a, b = _clean(m[f"{field}_c1"]), _clean(m[f"{field}_c2"])
        for i in m.index[(a != "") & (b != "") & (a != b)]:
            dis.append({"article_id": m.at[i, "article_id"], "bundle_id": m.at[i, "bundle_id_c1"],
                        "title": m.at[i, "title_c1"], "field": field,
                        "coder1": m.at[i, f"{field}_c1"], "coder2": m.at[i, f"{field}_c2"]})
    return pd.DataFrame(rows), pd.DataFrame(dis, columns=["article_id", "bundle_id", "title", "field",
                                                          "coder1", "coder2"])


def auto_vs_human(features: pd.DataFrame, human: pd.DataFrame, coder: str) -> pd.DataFrame:
    """규칙 기반 사전 코딩이 사람 판정과 얼마나 맞는지 (사전 자체의 타당도)."""
    f = features.copy()
    if "auto_direction" in f.columns:
        f["auto_direction"] = f["auto_direction"].astype(str).map(lambda d: DIR_MAP.get(d, d))
    if "auto_caveat" in f.columns:
        f["auto_caveat"] = f["auto_caveat"].astype(str).map({"1": "Y", "0": "N"}).fillna("")
    m = human.merge(f, on="article_id", suffixes=("", "_f"))
    rows = []
    for field, auto_col in AUTO_EQUIV.items():
        if field in m.columns and auto_col in m.columns:
            r = kappa_row(m[auto_col], m[field], field, FIELDS[field])
            r["comparison"] = f"auto vs {coder}"
            rows.append(r)
    return pd.DataFrame(rows)


def run_agreement(ws: Workspace, coders: list[str]) -> pd.DataFrame:
    tables = []
    sheets = {c: read_csv(ws.coding_csv(c)) for c in coders if ws.coding_csv(c).exists()}
    if len(sheets) >= 2:
        c1, c2 = list(sheets)[:2]
        table, dis = compare_coders(sheets[c1], sheets[c2])
        table["comparison"] = f"{c1} vs {c2}"
        tables.append(table)
        dis = merge_preserving(dis, ws.disagreements_csv, ["article_id", "field"], ["resolved", "resolution_notes"])
        write_csv(dis, ws.disagreements_csv)
        log.info("불일치 %d건 → %s (resolved 열에 합의값을 적으세요)", len(dis), ws.disagreements_csv)
    if ws.features_csv.exists():
        feats = read_csv(ws.features_csv)
        for c, sheet in sheets.items():
            tables.append(auto_vs_human(feats, sheet, c))
    out = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    if not out.empty:
        out = out[["comparison", "field", "n", "pct_agree", "kappa", "weighted_kappa"]]
        ws.tables_dir.mkdir(parents=True, exist_ok=True)
        write_csv(out, ws.tables_dir / "agreement.csv")
    return out


def finalize(ws: Workspace, coders: list[str]) -> pd.DataFrame:
    """최종 코드: 두 코더 일치 → 그 값 / 불일치 → disagreements.csv 의 resolved / 한 명만 → 그 값(표시)."""
    sheets = [read_csv(ws.coding_csv(c)) for c in coders if ws.coding_csv(c).exists()]
    if not sheets:
        raise FileNotFoundError("코딩 시트가 없습니다. precode → annotate 를 먼저 하세요.")
    base = sheets[0].copy()
    resolved = {}
    if ws.disagreements_csv.exists():
        d = read_csv(ws.disagreements_csv)
        if "resolved" in d.columns:
            for _, r in d.iterrows():
                if str(r["resolved"]).strip():
                    resolved[(r["article_id"], r["field"])] = str(r["resolved"]).strip()
    others = [s.set_index("article_id") for s in sheets[1:]]
    status_col = []
    for i, row in base.iterrows():
        aid = row["article_id"]
        statuses = set()
        for field in ARTICLE_HUMAN_COLS:
            if field == "coder_notes":
                continue
            values = [str(row.get(field, "")).strip()]
            values += [str(o.at[aid, field]).strip() for o in others if aid in o.index and field in o.columns]
            filled = [v for v in values if v]
            if (aid, field) in resolved:
                base.at[i, field] = resolved[(aid, field)]
                statuses.add("adjudicated")
            elif len(filled) >= 2 and len({v.upper() for v in filled}) == 1:
                base.at[i, field] = filled[0]
                statuses.add("agreed")
            elif len(filled) == 1:
                base.at[i, field] = filled[0]
                statuses.add("single_coded")
            elif len(filled) >= 2:
                base.at[i, field] = ""
                statuses.add("UNRESOLVED")
        status_col.append(",".join(sorted(statuses)))
    base["final_status"] = status_col
    base["coders"] = "+".join(c for c in coders if ws.coding_csv(c).exists())
    keep = ["article_id", "bundle_id", "outlet", "outlet_type", "published", "title"] + \
        ARTICLE_HUMAN_COLS + ["final_status", "coders"]
    final = base[[c for c in keep if c in base.columns]]
    write_csv(final, ws.final_coding_csv)
    n_unres = int(final["final_status"].str.contains("UNRESOLVED").sum())
    log.info("최종 코드 %d건 → %s (미해결 %d건)", len(final), ws.final_coding_csv, n_unres)
    return final
