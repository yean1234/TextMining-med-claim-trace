"""자동 사전 코딩 → 사람 코딩 시트 생성.

만드는 파일
  annotation/paper_coding.csv            논문 1편당 1행. 자동 제안 + 사람 입력 열
  data/processed/article_features.csv    기사별 자동 특징 전체 (분석용)
  annotation/article_coding_<coder>.csv  코더별 코딩 시트. 두 번째 코더는 기본 '블라인드'
                                         (자동 제안값을 숨겨 앵커링 편향을 줄임)
이미 사람이 입력한 칸은 다시 실행해도 지워지지 않는다(utils.merge_preserving).
"""
from __future__ import annotations

import random
from collections import defaultdict

import pandas as pd

from ..bundles import Bundle, bundle_map
from ..collect.outlets import OutletResolver
from ..nlp.lexicon import Lexicon
from ..paths import Workspace
from ..utils import load_yaml, log, merge_preserving, read_csv, read_jsonl, write_csv
from .features import CONDITION_CATEGORIES, JournalMatcher, extract_features
from .strength import KoreanClaimCoder

PAPER_HUMAN_COLS = ["design", "paper_strength", "paper_direction", "key_conditions",
                    "key_conditions_detail", "main_finding_ko", "paper_caveat",
                    "press_release_url", "press_release_strength", "checked_by", "paper_notes"]
ARTICLE_HUMAN_COLS = ["title_strength", "lead_strength", "direction", "advice", "caveat_in_body",
                      "title_omission_misleading", "animal_generalization", "source_error",
                      "coder_notes"]
# 사람 입력값 검증 규칙 (annotate / agreement 에서 사용)
ALLOWED = {
    "title_strength": [str(i) for i in range(7)],
    "lead_strength": [str(i) for i in range(7)],
    "paper_strength": [str(i) for i in range(7)],
    "press_release_strength": [str(i) for i in range(7)],
    "direction": ["inc", "dec", "null", "unclear"],
    "paper_direction": ["inc", "dec", "null", "unclear"],
    "advice": ["0", "1", "2", "3"],
    "caveat_in_body": ["Y", "N"],
    "title_omission_misleading": ["Y", "N", "NA"],
    "animal_generalization": ["0", "1", "2", "NA"],
    "source_error": ["Y", "N"],
    "design": ["observational", "rct", "animal", "in_vitro", "meta", "review", "case_report",
               "pharmacovigilance", "other"],
}
DIRECTION_TO_CODE = {"increase": "inc", "decrease": "dec", "null": "null", "mixed": "unclear", "": ""}


# ------------------------------------------------------------------ 논문 시트
def build_paper_sheet(ws: Workspace, bundles: list[Bundle]) -> pd.DataFrame:
    if ws.papers_csv.exists():
        papers = read_csv(ws.papers_csv)
    else:
        log.warning("papers.csv 가 없어 bundles.yaml 정보만으로 논문 시트를 만듭니다 (fetch-papers 권장).")
        papers = pd.DataFrame([{"bundle_id": b.bundle_id, "doi": b.paper.get("doi", ""),
                                "title": b.paper.get("title", ""), "journal": b.journal,
                                "published": b.paper.get("published", "")} for b in bundles])
    bmap = bundle_map(bundles)
    papers["label"] = papers["bundle_id"].map(lambda x: bmap[x].label if x in bmap else "")
    papers["bundle_notes"] = papers["bundle_id"].map(lambda x: bmap[x].notes if x in bmap else "")
    if "direction_auto" in papers.columns:
        papers["direction_auto"] = papers["direction_auto"].map(lambda d: DIRECTION_TO_CODE.get(str(d), str(d)))
    keep = ["bundle_id", "label", "doi", "title", "journal", "published", "pmid", "design_auto",
            "design_cues", "conclusion_en", "conclusion_source", "strength_auto", "strength_auto_cues",
            "direction_auto", "bundle_notes", "fetch_status"]
    sheet = papers[[c for c in keep if c in papers.columns]].copy()
    sheet = merge_preserving(sheet, ws.paper_coding_csv, "bundle_id", PAPER_HUMAN_COLS)
    write_csv(sheet, ws.paper_coding_csv)
    log.info("논문 코딩 시트 → %s", ws.paper_coding_csv)
    return sheet


# ------------------------------------------------------------------ 기사 선택
def usable_articles(ws: Workspace) -> list[dict]:
    """본문 수집 성공 + 사람이 본문 점검에서 N 으로 표시하지 않은 기사."""
    arts = [a for a in read_jsonl(ws.articles_jsonl) if a.get("status") == "ok" and a.get("body")]
    if ws.text_check_csv.exists():
        chk = read_csv(ws.text_check_csv)
        if "text_ok" in chk.columns:
            bad = set(chk.loc[chk["text_ok"].str.upper().eq("N"), "article_id"])
            arts = [a for a in arts if a["article_id"] not in bad]
    if ws.candidates_csv.exists():
        cand = read_csv(ws.candidates_csv)
        excluded = set(cand.loc[cand.get("include", pd.Series("", index=cand.index)).str.upper().eq("N"),
                                "candidate_id"])
        arts = [a for a in arts if a["article_id"] not in excluded]
    return arts


def select_articles(articles: list[dict], max_per_bundle: int, seed: int = 42) -> pd.DataFrame:
    """묶음당 최대 N건. 넘치면 매체 유형이 고르게 섞이도록 돌아가며 뽑는다(층화).

    노션 메모: '10건짜리 묶음 1개보다 4건짜리 묶음 20개가 낫다' → 한 논문이 결과를 지배하지 않게.
    """
    rng = random.Random(seed)
    rows = []
    by_bundle: dict[str, list[dict]] = defaultdict(list)
    for a in articles:
        by_bundle[a["bundle_id"]].append(a)
    for bid, arts in by_bundle.items():
        if max_per_bundle <= 0 or len(arts) <= max_per_bundle:
            rows += [{"article_id": a["article_id"], "bundle_id": bid, "selected": "Y",
                      "reason": "all"} for a in arts]
            continue
        by_type: dict[str, list[dict]] = defaultdict(list)
        for a in sorted(arts, key=lambda x: x["article_id"]):
            by_type[a.get("outlet_type", "unknown")].append(a)
        for lst in by_type.values():
            rng.shuffle(lst)
        order = sorted(by_type)
        rng.shuffle(order)
        picked: list[dict] = []
        while len(picked) < max_per_bundle:
            for t in order:
                if by_type[t] and len(picked) < max_per_bundle:
                    picked.append(by_type[t].pop())
        ids = {a["article_id"] for a in picked}
        rows += [{"article_id": a["article_id"], "bundle_id": bid,
                  "selected": "Y" if a["article_id"] in ids else "N",
                  "reason": f"stratified_sample(max={max_per_bundle},seed={seed})"} for a in arts]
    return pd.DataFrame(rows, columns=["article_id", "bundle_id", "selected", "reason"])


# ------------------------------------------------------------------ 기사 특징·시트
def build_article_features(ws: Workspace, bundles: list[Bundle], max_per_bundle: int = 5,
                           seed: int = 42) -> pd.DataFrame:
    lex = Lexicon.load(ws.config_file("lexicon_ko.yaml"))
    coder = KoreanClaimCoder(lex)
    journals = JournalMatcher(load_yaml(ws.config_file("journals.yaml")))
    bmap = bundle_map(bundles)
    arts = usable_articles(ws)
    resolver = OutletResolver(ws.config_file("outlets.yaml"))
    for a in arts:   # 매체 분류는 항상 최신 outlets.yaml 기준
        name, otype = resolver.resolve(a.get("url", ""), a.get("outlet", ""))
        if otype != "unknown":
            a["outlet"], a["outlet_type"] = name, otype
    sel = select_articles(arts, max_per_bundle, seed)
    write_csv(sel, ws.selection_csv)
    chosen = set(sel.loc[sel["selected"] == "Y", "article_id"])
    rows = []
    for a in arts:
        if a["article_id"] not in chosen:
            continue
        if a["bundle_id"] not in bmap:
            log.warning("bundles.yaml 에 없는 묶음 %s — 건너뜀", a["bundle_id"])
            continue
        rows.append(extract_features(a, bmap[a["bundle_id"]], lex, coder, journals))
    df = pd.DataFrame(rows)
    write_csv(df, ws.features_csv)
    log.info("기사 특징 %d건 → %s (선택 %d / 사용 가능 %d)", len(df), ws.features_csv, len(chosen), len(arts))
    return df


def _cats(row: pd.Series, section: str) -> str:
    return ", ".join(c for c in CONDITION_CATEGORIES if str(row.get(f"auto_cond_{section}_{c}", "0")) == "1")


def build_coding_sheet(ws: Workspace, features: pd.DataFrame, coder: str, blind: bool) -> pd.DataFrame:
    papers = read_csv(ws.paper_coding_csv) if ws.paper_coding_csv.exists() else pd.DataFrame()
    pinfo = {}
    for _, p in papers.iterrows():
        summary = p.get("main_finding_ko") or p.get("conclusion_en") or ""
        pinfo[p["bundle_id"]] = (summary, p.get("key_conditions", ""), p.get("design") or p.get("design_auto", ""))
    rows = []
    for _, f in features.iterrows():
        summary, keyc, design = pinfo.get(f["bundle_id"], ("", "", ""))
        row = {"article_id": f["article_id"], "bundle_id": f["bundle_id"], "outlet": f["outlet"],
               "outlet_type": f["outlet_type"], "published": f["published"], "url": f["url"],
               "paper_design": design, "paper_finding": summary, "paper_key_conditions": keyc,
               "title": f["title"], "lead": f["lead"]}
        if not blind:
            row.update({
                "auto_title_strength": f["auto_title_strength"], "auto_title_cues": f["auto_title_cues"],
                "auto_lead_strength": f["auto_lead_strength"], "auto_lead_cues": f["auto_lead_cues"],
                "auto_direction": DIRECTION_TO_CODE.get(str(f["auto_direction"]), f["auto_direction"]),
                "auto_advice": f["auto_advice"], "auto_advice_cues": f["auto_advice_cues"],
                "auto_caveat": "Y" if str(f["auto_caveat"]) == "1" else "N",
                "auto_caveat_sentences": f["auto_caveat_sentences"],
                "auto_title_question": f["auto_title_question"],
                "auto_title_conditions": _cats(f, "title"), "auto_body_conditions": _cats(f, "body"),
                "journals_mentioned": f["journals_mentioned"], "non_journal_sources": f["non_journal_sources"],
            })
        rows.append(row)
    sheet = pd.DataFrame(rows)
    path = ws.coding_csv(coder)
    sheet = merge_preserving(sheet, path, "article_id", ARTICLE_HUMAN_COLS)
    write_csv(sheet, path)
    log.info("코딩 시트(%s%s) → %s", coder, ", 블라인드" if blind else "", path)
    return sheet


def precode(ws: Workspace, bundles: list[Bundle], coders: list[str], blind_coders: list[str],
            max_per_bundle: int = 5, seed: int = 42) -> None:
    ws.ensure_dirs()
    build_paper_sheet(ws, bundles)
    feats = build_article_features(ws, bundles, max_per_bundle, seed)
    if feats.empty:
        log.warning("코딩할 기사가 없습니다 (scrape 결과·include 확인).")
        return
    for c in coders:
        build_coding_sheet(ws, feats, c, blind=c in blind_coders)
