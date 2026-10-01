"""네이버 검색 API(뉴스)로 묶음별 기사 후보 수집 → annotation/article_candidates.csv

API 문서: https://developers.naver.com/docs/serviceapi/search/news/news.md
  - display 최대 100, start 최대 1000 → 질의 하나로 최대 1,000건
  - 날짜 범위 필터가 없다 → 받은 뒤 pubDate 로 거른다 (오래된 사건은 누락 가능, LIMITATIONS 참고)
"""
from __future__ import annotations

import html
import re
from datetime import datetime
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse, urlunparse

import pandas as pd
import requests

from ..bundles import Bundle
from ..coding.features import JournalMatcher
from ..coding.relevance import score_candidate
from ..paths import Workspace
from ..utils import RateLimiter, load_yaml, log, merge_preserving, stable_id, write_csv
from .outlets import OutletResolver

NAVER_NEWS_API = "https://openapi.naver.com/v1/search/news.json"

# article_candidates.csv 에서 사람이 채우는 열
CANDIDATE_HUMAN_COLS = ["include", "link_type", "paper_match_evidence", "candidate_notes"]
CANDIDATE_COLS = ["candidate_id", "bundle_id", "source", "query", "pub_date", "outlet", "outlet_type",
                  "title", "description", "url", "naver_url", "relevance_score", "auto_include",
                  "relevance_reasons"]


def clean_api_text(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"</?b>", "", text or ""))).strip()


def parse_pubdate(value: str) -> datetime | None:
    try:
        return parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None


def normalize_url(url: str) -> str:
    """중복 제거용 URL 정규화 (쿼리스트링 중 추적용 파라미터 제거)."""
    if not url:
        return ""
    p = urlparse(url.strip())
    query = "&".join(q for q in p.query.split("&") if q and not q.startswith(("utm_", "fbclid", "from=")))
    return urlunparse((p.scheme or "https", p.netloc.lower(), p.path.rstrip("/"), "", query, ""))


def search_news(session: requests.Session, query: str, client_id: str, client_secret: str,
                max_results: int = 300, sort: str = "sim", limiter: RateLimiter | None = None) -> list[dict]:
    headers = {"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret}
    items: list[dict] = []
    start = 1
    while start <= min(max_results, 1000):
        display = min(100, max_results - len(items))
        if display <= 0:
            break
        if limiter:
            limiter.wait()
        resp = session.get(NAVER_NEWS_API, headers=headers, timeout=20,
                           params={"query": query, "display": display, "start": start, "sort": sort})
        if resp.status_code == 401:
            raise RuntimeError("네이버 API 인증 실패: NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 확인")
        resp.raise_for_status()
        data = resp.json()
        batch = data.get("items", [])
        items.extend(batch)
        if len(batch) < display or start + display > int(data.get("total", 0)):
            break
        start += display
    return items


def items_to_candidates(bundle: Bundle, query: str, items: list[dict], resolver: OutletResolver,
                        journals: JournalMatcher, source: str = "naver_api") -> list[dict]:
    rows = []
    for it in items:
        title = clean_api_text(it.get("title", ""))
        desc = clean_api_text(it.get("description", ""))
        original = it.get("originallink") or ""
        naver = it.get("link") or ""
        if naver and "naver.com" not in naver:
            naver = ""
        url = original or naver
        pub = parse_pubdate(it.get("pubDate", ""))
        outlet, otype = resolver.resolve(original or naver)
        score, auto, reasons = score_candidate(bundle, title, desc, pub, journals)
        rows.append({
            "candidate_id": stable_id(bundle.bundle_id, normalize_url(url) or title),
            "bundle_id": bundle.bundle_id,
            "source": source,
            "query": query,
            "pub_date": pub.strftime("%Y-%m-%d %H:%M") if pub else "",
            "outlet": outlet,
            "outlet_type": otype,
            "title": title,
            "description": desc,
            "url": url,
            "naver_url": naver,
            "relevance_score": score,
            "auto_include": auto,
            "relevance_reasons": "; ".join(reasons),
        })
    return rows


def in_window(row: dict, bundle: Bundle) -> bool:
    if not row["pub_date"]:
        return True
    d = datetime.strptime(row["pub_date"][:10], "%Y-%m-%d").date()
    if bundle.date_from and d < bundle.date_from:
        return False
    if bundle.date_to and d > bundle.date_to:
        return False
    return True


def known_article_rows(bundle: Bundle, resolver: OutletResolver, journals: JournalMatcher) -> list[dict]:
    """bundles.yaml 의 known_articles(사람이 미리 찾은 기사)를 후보로 넣는다."""
    rows = []
    for k in bundle.known_articles:
        title = str(k.get("title_hint") or "")
        url = str(k.get("url") or "")
        outlet, otype = resolver.resolve(url, str(k.get("outlet") or ""))
        score, _, reasons = score_candidate(bundle, title, "", None, journals)
        rows.append({
            "candidate_id": stable_id(bundle.bundle_id, normalize_url(url) or f"{outlet}|{title}"),
            "bundle_id": bundle.bundle_id, "source": "known(bundles.yaml)", "query": "",
            "pub_date": str(k.get("date") or ""), "outlet": outlet, "outlet_type": otype,
            "title": title, "description": "(사람이 찾은 기사 — URL·전체 제목 확인 필요)" if not url else "",
            "url": url, "naver_url": url if "naver.com" in url else "",
            "relevance_score": score, "auto_include": "?", "relevance_reasons": "; ".join(reasons),
        })
    return rows


def dedupe(rows: list[dict]) -> list[dict]:
    """같은 묶음 안에서 URL 또는 (매체, 제목)이 같은 후보를 하나로 합친다(질의는 이어 붙임)."""
    seen: dict[tuple, dict] = {}
    for r in rows:
        key_url = (r["bundle_id"], normalize_url(r["url"])) if r["url"] else None
        key_title = (r["bundle_id"], r["outlet"], re.sub(r"\W+", "", r["title"]))
        hit = seen.get(key_url) if key_url else None
        hit = hit or seen.get(key_title)
        if hit:
            if r["query"] and r["query"] not in hit["query"]:
                hit["query"] = f"{hit['query']} | {r['query']}".strip(" |")
            continue
        if key_url:
            seen[key_url] = r
        seen[key_title] = r
    unique = {id(r): r for r in seen.values()}
    return list(unique.values())


def collect_candidates(ws: Workspace, bundles: list[Bundle], session: requests.Session,
                       client_id: str, client_secret: str, max_per_query: int = 300,
                       sort: str = "sim", delay: float = 0.2) -> pd.DataFrame:
    resolver = OutletResolver(ws.config_file("outlets.yaml"))
    journals = JournalMatcher(load_yaml(ws.config_file("journals.yaml")))
    limiter = RateLimiter(delay)
    rows: list[dict] = []
    for b in bundles:
        rows.extend(known_article_rows(b, resolver, journals))
        for q in b.queries:
            items = search_news(session, q, client_id, client_secret, max_per_query, sort, limiter)
            cands = [r for r in items_to_candidates(b, q, items, resolver, journals) if in_window(r, b)]
            log.info("[%s] '%s' → %d건 중 기간 내 %d건", b.bundle_id, q, len(items), len(cands))
            rows.extend(cands)
    return save_candidates(ws, rows)


def save_candidates(ws: Workspace, rows: list[dict]) -> pd.DataFrame:
    """기존 article_candidates.csv 의 사람 입력(include 등)을 보존하며 저장."""
    df = pd.DataFrame(dedupe(rows), columns=CANDIDATE_COLS)
    if ws.candidates_csv.exists():
        from ..utils import read_csv
        old = read_csv(ws.candidates_csv)
        # 이전 실행·수동 추가 행 중 이번 결과에 없는 것도 유지한다 (manual 행 포함)
        old_only = old[~old["candidate_id"].isin(df["candidate_id"])]
        old_only = old_only[[c for c in CANDIDATE_COLS if c in old_only.columns]]
        df = pd.concat([df, old_only], ignore_index=True)
    df = merge_preserving(df, ws.candidates_csv, "candidate_id", CANDIDATE_HUMAN_COLS)
    if "_stale" in df.columns:
        df = df.drop(columns="_stale")
    df["relevance_score"] = pd.to_numeric(df["relevance_score"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values(["bundle_id", "relevance_score"], ascending=[True, False])
    write_csv(df, ws.candidates_csv)
    log.info("후보 %d건 → %s", len(df), ws.candidates_csv)
    return df
