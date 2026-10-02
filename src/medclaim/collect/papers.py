"""원논문 메타데이터·초록 수집 (Crossref + Europe PMC).

- Crossref: 제목·저널·저자·출판일 (DOI 또는 제목 검색)
- Europe PMC: 초록(구조화 초록이면 Conclusions 구간 분리), PMID, 출판 유형
결과는 data/raw/papers/{bundle_id}.json 과 data/interim/papers.csv 에 저장한다.

네트워크가 안 되면 bundles.yaml 의 정보만으로 행을 만들고 fetch_status 에 오류를 적는다.
"""
from __future__ import annotations

import difflib
import html
import json
import re

import pandas as pd
import requests

from ..bundles import Bundle
from ..nlp.lexicon import Lexicon
from ..paths import Workspace
from ..utils import RateLimiter, log, write_csv

CROSSREF_WORKS = "https://api.crossref.org/works"
EUROPEPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
CONCLUSION_HEADS = re.compile(r"^(conclusions?(\s+and\s+relevance)?|interpretation|implications?|summary)$", re.I)


# ------------------------------------------------------------------ Crossref
def _crossref_record(item: dict) -> dict:
    authors = item.get("author") or []
    date_parts = (item.get("published-online") or item.get("published-print")
                  or item.get("published") or item.get("issued") or {}).get("date-parts", [[None]])[0]
    published = "-".join(f"{p:02d}" if i else str(p) for i, p in enumerate(date_parts) if p)
    return {
        "doi": (item.get("DOI") or "").lower(),
        "cr_title": " ".join(item.get("title") or []),
        "cr_journal": " ".join(item.get("container-title") or []),
        "cr_first_author": (authors[0].get("family") or "") if authors else "",
        "cr_n_authors": len(authors),
        "cr_published": published,
        "cr_type": item.get("type", ""),
        "cr_abstract": item.get("abstract", ""),
    }


def fetch_crossref_by_doi(session: requests.Session, doi: str, mailto: str = "") -> dict | None:
    params = {"mailto": mailto} if mailto else None
    resp = session.get(f"{CROSSREF_WORKS}/{doi}", params=params, timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return _crossref_record(resp.json()["message"])


def search_crossref(session: requests.Session, title: str, mailto: str = "",
                    min_similarity: float = 0.85) -> dict | None:
    params = {"query.bibliographic": title, "rows": 5}
    if mailto:
        params["mailto"] = mailto
    resp = session.get(CROSSREF_WORKS, params=params, timeout=30)
    resp.raise_for_status()
    best, best_sim = None, 0.0
    for item in resp.json()["message"].get("items", []):
        cand = " ".join(item.get("title") or [])
        sim = difflib.SequenceMatcher(None, cand.lower(), title.lower()).ratio()
        if sim > best_sim:
            best, best_sim = item, sim
    if best is None or best_sim < min_similarity:
        return None
    rec = _crossref_record(best)
    rec["cr_title_similarity"] = round(best_sim, 3)
    return rec


# ------------------------------------------------------------------ Europe PMC
def fetch_europepmc(session: requests.Session, doi: str = "", title: str = "") -> dict | None:
    if doi:
        query = f'DOI:"{doi}"'
    elif title:
        query = f'TITLE:"{title}"'
    else:
        return None
    resp = session.get(EUROPEPMC_SEARCH, params={"query": query, "resultType": "core", "format": "json",
                                                 "pageSize": 1}, timeout=30)
    resp.raise_for_status()
    results = resp.json().get("resultList", {}).get("result", [])
    if not results:
        return None
    r = results[0]
    return {
        "pmid": r.get("pmid", ""),
        "pmcid": r.get("pmcid", ""),
        "epmc_title": r.get("title", ""),
        "epmc_journal": (r.get("journalInfo") or {}).get("journal", {}).get("title", ""),
        "epmc_pub_types": "; ".join((r.get("pubTypeList") or {}).get("pubType", [])),
        "epmc_abstract": r.get("abstractText", ""),
        "epmc_first_pub_date": r.get("firstPublicationDate", ""),
    }


# ------------------------------------------------------------------ 초록 파싱
def strip_tags(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def parse_structured_abstract(raw: str) -> dict[str, str]:
    """Europe PMC(<h4>제목</h4>), Crossref JATS(<jats:title>), 'CONCLUSIONS:' 인라인 형식을 처리한다.
    구조화되지 않은 초록은 {'_full': 전체} 만 돌려준다."""
    raw = raw or ""
    sections: dict[str, str] = {}
    # 1) JATS
    jats = re.findall(r"<jats:sec[^>]*>\s*<jats:title>(.*?)</jats:title>(.*?)</jats:sec>", raw, flags=re.S)
    if jats:
        for head, body in jats:
            sections[strip_tags(head).lower()] = strip_tags(body)
    # 2) <h4> 또는 <b> 머리말
    if not sections:
        parts = re.split(r"<(?:h\d|b|strong)>(.*?)</(?:h\d|b|strong)>", raw, flags=re.S)
        if len(parts) > 2:
            for i in range(1, len(parts) - 1, 2):
                sections[strip_tags(parts[i]).rstrip(":").lower()] = strip_tags(parts[i + 1])
    # 3) 'CONCLUSIONS AND RELEVANCE:' 같은 대문자 인라인 머리말
    if not sections:
        plain = strip_tags(raw)
        pieces = re.split(r"\b([A-Z][A-Z ,&]{3,40}):\s", plain)
        if len(pieces) > 2:
            for i in range(1, len(pieces) - 1, 2):
                sections[pieces[i].strip().lower()] = pieces[i + 1].strip()
    sections["_full"] = strip_tags(raw)
    return sections


def pick_conclusion(sections: dict[str, str]) -> tuple[str, str]:
    """(결론 텍스트, 출처). 구조화 초록이면 Conclusions 구간, 아니면 마지막 두 문장."""
    for head, text in sections.items():
        if head != "_full" and CONCLUSION_HEADS.match(head.strip()):
            return text, f"section:{head}"
    full = sections.get("_full", "")
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z])", full)
    return " ".join(sents[-2:]).strip(), "last_two_sentences"


def guess_design(lex_en: Lexicon, title: str, abstract: str, pub_types: str = "") -> tuple[str, str]:
    text = f"{title} {abstract}"
    if re.search(r"review|commentary|editorial", pub_types or "", re.I) and not re.search(r"meta", text, re.I):
        return "review", f"pubType:{pub_types}"
    for design in lex_en.subkeys("design"):
        cues = lex_en.find(text, "design", design)
        if design == "rct" and cues and re.search(r"emulat", text, re.I):
            continue  # target trial emulation 은 관찰연구
        if cues:
            return design, "; ".join(c.text for c in cues[:3])
    return "unknown", ""


# ------------------------------------------------------------------ 실행
def fetch_papers(ws: Workspace, bundles: list[Bundle], session: requests.Session,
                 mailto: str = "", force: bool = False, delay: float = 1.0) -> pd.DataFrame:
    from ..coding.strength import EnglishClaimCoder

    ws.ensure_dirs()
    lex_en = Lexicon.load(ws.config_file("lexicon_en.yaml"))
    coder = EnglishClaimCoder(lex_en)
    limiter = RateLimiter(delay)
    rows = []
    for b in bundles:
        cache = ws.paper_raw_dir / f"{b.bundle_id}.json"
        record: dict = {}
        status = []
        if cache.exists() and not force:
            record = json.loads(cache.read_text(encoding="utf-8"))
            status.append("cache")
        else:
            doi = b.paper.get("doi", "").strip()
            try:
                limiter.wait()
                cr = fetch_crossref_by_doi(session, doi, mailto) if doi else \
                    search_crossref(session, b.paper.get("title", ""), mailto)
                if cr:
                    record.update(cr)
                    doi = doi or cr.get("doi", "")
                    status.append("crossref_ok" if b.paper.get("doi") else
                                  f"crossref_title_match({cr.get('cr_title_similarity')})")
                else:
                    status.append("crossref_not_found")
            except requests.RequestException as e:
                status.append(f"crossref_error:{type(e).__name__}")
            try:
                limiter.wait()
                ep = fetch_europepmc(session, doi=doi, title="" if doi else b.paper.get("title", ""))
                if ep:
                    record.update(ep)
                    status.append("europepmc_ok")
                else:
                    status.append("europepmc_not_found")
            except requests.RequestException as e:
                status.append(f"europepmc_error:{type(e).__name__}")
            if any(s.endswith("_ok") or s.startswith("crossref_title") for s in status):
                cache.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

        abstract_raw = record.get("epmc_abstract") or record.get("cr_abstract") or ""
        sections = parse_structured_abstract(abstract_raw)
        conclusion, conc_src = pick_conclusion(sections) if abstract_raw else ("", "")
        title = record.get("cr_title") or record.get("epmc_title") or b.paper.get("title", "")
        design, design_cues = guess_design(lex_en, title, sections.get("_full", ""),
                                           record.get("epmc_pub_types", ""))
        main, strongest = coder.code_conclusion(conclusion) if conclusion else (None, None)
        rows.append({
            "bundle_id": b.bundle_id,
            "doi": (b.paper.get("doi") or record.get("doi", "")).lower(),
            "title": title,
            "journal": record.get("cr_journal") or record.get("epmc_journal") or b.journal,
            "journal_key": b.journal,
            "first_author": record.get("cr_first_author") or b.paper.get("first_author", ""),
            "published": record.get("cr_published") or record.get("epmc_first_pub_date")
            or b.paper.get("published", ""),
            "pmid": record.get("pmid", ""),
            "pub_types": record.get("epmc_pub_types", ""),
            "abstract": sections.get("_full", ""),
            "conclusion_en": conclusion,
            "conclusion_source": conc_src,
            "design_auto": design,
            "design_cues": design_cues,
            "strength_auto": main.level if main else "",
            "strength_auto_cues": main.cue_str() if main else "",
            "strength_auto_max": strongest.level if strongest else "",
            "direction_auto": main.direction if main else "",
            "fetch_status": ", ".join(status) or "skipped",
        })
        log.info("논문 %s: %s", b.bundle_id, rows[-1]["fetch_status"])
    df = pd.DataFrame(rows)
    if ws.papers_csv.exists():   # --bundle 로 일부만 돌려도 다른 묶음 행은 보존
        from ..utils import read_csv
        old = read_csv(ws.papers_csv)
        if "bundle_id" in old.columns:
            df = pd.concat([old[~old["bundle_id"].isin(df["bundle_id"])], df], ignore_index=True)
            df = df.sort_values("bundle_id").reset_index(drop=True)
    write_csv(df, ws.papers_csv)
    return df
