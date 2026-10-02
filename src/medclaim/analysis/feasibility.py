"""파일럿용 '이 주제가 실행 가능한가' 점검표.

각 지표에 (값, 기준, 판정)을 붙인다. 기준은 팀이 정할 수 있는 잠정값이다 — 숫자 자체보다
'어디가 병목인지'를 보는 용도다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..bundles import Bundle
from ..paths import Workspace
from ..utils import read_csv, read_jsonl


def _verdict(ok: bool | None) -> str:
    return "–" if ok is None else ("✅ 충족" if ok else "⚠️ 미달")


def feasibility_table(ws: Workspace, bundles: list[Bundle], df: pd.DataFrame, cond_long: pd.DataFrame,
                      per_art: pd.DataFrame, agreement: pd.DataFrame | None) -> pd.DataFrame:
    rows = []

    def add(name: str, value: str, criterion: str, ok: bool | None, meaning: str) -> None:
        rows.append({"점검 항목": name, "값": value, "기준(잠정)": criterion, "판정": _verdict(ok), "의미": meaning})

    # 1) 묶음 확보
    n_main = {b.bundle_id: 0 for b in bundles}
    if len(df):
        for bid, n in df.groupby("bundle_id").size().items():
            n_main[bid] = int(n)
    ok_b = sum(1 for n in n_main.values() if n >= 3)
    add("기사 3건 이상 묶음", f"{ok_b}/{len(bundles)}개 ({', '.join(f'{k[:3]}={v}' for k, v in n_main.items())})",
        "묶음의 70% 이상", ok_b >= 0.7 * len(bundles),
        "20~25개 묶음을 채울 수 있는지. 미달 묶음은 검색 보강(네이버 API·BigKinds) 또는 교체")

    # 2) 검색 후보 중 실제 보도 비율
    if ws.candidates_csv.exists():
        cand = read_csv(ws.candidates_csv)
        scraped = {a["article_id"] for a in read_jsonl(ws.articles_jsonl) if a.get("status") == "ok"}
        judged = cand[cand["candidate_id"].isin(scraped)]
        if len(judged):
            inc = judged["include"].astype(str).str.upper().eq("Y")
            lt = judged.get("link_type", pd.Series("", index=judged.index)).astype(str)
            add("후보 기사 중 '그 논문 보도' 비율", f"{inc.mean():.0%} ({inc.sum()}/{len(judged)}; "
                f"배경 인용 {lt.eq('background').sum()}, 다른 논문·기타 {lt.eq('unrelated').sum()})",
                "참고 지표", None,
                "낮을수록 사람(또는 LLM)의 연결 판정이 필수. 키워드 검색만으로 묶음을 만들면 다른 논문 기사가 섞임")

    # 3) 비교할 변량이 있는가
    d = df.dropna(subset=["delta_title"]) if len(df) else df
    if len(d):
        diff = (d["delta_title"] != 0).mean()
        add("제목 강도 ≠ 논문 강도 비율", f"{diff:.0%} (n={len(d)})", "30% 이상", diff >= 0.3,
            "0%면 '제목이 바뀌는가'를 물을 변량이 없음")
        sd = d.groupby("bundle_id")["title_strength"].std(ddof=0).mean()
        add("묶음 내 제목 강도 표준편차(평균)", f"{sd:.2f}", "0.5 이상", sd >= 0.5,
            "같은 논문을 매체마다 다르게 쓰는지(매체 간 비교 가능성)")
        up = (d["delta_title"] > 0).mean()
        add("제목이 논문보다 강한 비율", f"{up:.0%}", "참고 지표", None, "RQ1의 핵심 수치")

    # 4) 조건 누락
    if len(cond_long):
        drop = cond_long["dropped_in_title_kept_in_body"].mean()
        add("핵심 조건이 본문엔 있고 제목엔 없는 비율", f"{drop:.0%} (기사×조건 {len(cond_long)}쌍)", "참고 지표", None,
            "RQ2. 자동 탐지값이라 사람 판정(title_omission_misleading)과 함께 볼 것")
    if "title_omission_misleading" in df.columns and len(df):
        tom = df["title_omission_misleading"].astype(str).str.upper()
        valid = tom.isin(["Y", "N"])
        if valid.any():
            add("제목 조건 생략이 오해를 부른다(사람 판정)", f"{tom[valid].eq('Y').mean():.0%} (n={valid.sum()})",
                "참고 지표", None, "RQ2를 사람 판단으로 본 값")

    # 5) 자동 코딩 신뢰도
    if agreement is not None and len(agreement):
        a = agreement[agreement["comparison"].astype(str).str.startswith("auto")]
        t = a[a["field"] == "title_strength"]
        if len(t):
            k = pd.to_numeric(t["weighted_kappa"].iloc[0], errors="coerce")
            add("자동 사전 vs 코더 (제목 강도, 가중 κ)", f"{k:.2f}" if not np.isnan(k) else "–", "0.6 이상",
                None if np.isnan(k) else bool(k >= 0.6),
                "충족하면 규모를 키울 때 자동 사전이 1차 분류를 대신할 수 있음")
        weak = a[pd.to_numeric(a["kappa"], errors="coerce") < 0.4]["field"].tolist()
        if weak:
            add("자동 사전이 약한 항목", ", ".join(weak), "κ<0.4", False, "이 항목은 사람/LLM 코딩이 필요")

    # 6) 출처 오류
    if "source_error" in df.columns and len(df):
        se = df["source_error"].astype(str).str.upper()
        if se.isin(["Y", "N"]).any():
            add("출처 표기 오류(저널·데이터·주체)", f"{se.eq('Y').mean():.0%} (n={se.isin(['Y', 'N']).sum()})",
                "참고 지표", None, "새 코딩 항목으로 쓸 만한지")

    # 7) 댓글
    n_with = int((per_art["n_comments"] >= 5).sum()) if len(per_art) else 0
    add("댓글 5개 이상 기사", f"{n_with}/{len(df)}건" if len(per_art) else "미수집",
        "50% 이상", (n_with >= 0.5 * len(df)) if len(per_art) else None,
        "RQ3 가능 여부. 의약전문지 기사는 대부분 댓글이 없음")
    return pd.DataFrame(rows)
