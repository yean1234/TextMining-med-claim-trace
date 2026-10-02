"""네이버 검색 API(뉴스)로 묶음별 기사 후보 수집 → annotation/article_candidates.csv

두 가지 발급 경로를 모두 지원한다 (환경변수 NAVER_API_BACKEND):
  - hub (기본) : NAVER API HUB (네이버 클라우드 콘솔에서 발급)
                 https://naverapihub.apigw.ntruss.com/search/v1/news
                 헤더 X-NCP-APIGW-API-KEY-ID / X-NCP-APIGW-API-KEY
  - openapi    : 기존 네이버 개발자센터(developers.naver.com) 키
                 https://openapi.naver.com/v1/search/news.json
                 헤더 X-Naver-Client-Id / X-Naver-Client-Secret
공통: display 최대 100, start 최대 1000 → 질의 하나로 최대 1,000건. 날짜 범위 필터가 없어서
받은 뒤 pubDate 로 거른다(오래된 기사는 누락 가능, LIMITATIONS 참고).

호출 한도: 검색 API 는 일 25,000건이 상한이다. CallBudget 이 하루 호출 수를 파일에 기록하고
NAVER_DAILY_CALL_BUDGET(기본 1,000)에 닿으면 요청을 보내기 전에 멈춘다. 자동 재시도도 꺼서
기록되지 않은 숨은 호출이 생기지 않게 한다.
"""
from __future__ import annotations

import html
import json
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import pandas as pd
import requests

from ..bundles import Bundle
from ..coding.features import JournalMatcher
from ..coding.relevance import score_candidate
from ..paths import Workspace
from ..utils import RateLimiter, get_session, load_yaml, log, merge_preserving, stable_id, write_csv
from .outlets import OutletResolver

BACKENDS = {
    "hub": ("https://naverapihub.apigw.ntruss.com/search/v1/news",
            "X-NCP-APIGW-API-KEY-ID", "X-NCP-APIGW-API-KEY"),
    "openapi": ("https://openapi.naver.com/v1/search/news.json",
                "X-Naver-Client-Id", "X-Naver-Client-Secret"),
}
DEFAULT_DAILY_BUDGET = 1000


class BudgetExceeded(RuntimeError):
    pass


class CallBudget:
    """하루 API 호출 수를 파일(JSON)에 누적 기록하고 상한에 닿으면 막는다."""

    def __init__(self, path: Path | None, daily_limit: int = DEFAULT_DAILY_BUDGET) -> None:
        self.path = path
        self.daily_limit = daily_limit
        self.counts: dict[str, int] = {}
        if path and path.exists():
            try:
                self.counts = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                self.counts = {}

    @property
    def today(self) -> str:
        return date.today().isoformat()

    @property
    def used_today(self) -> int:
        return int(self.counts.get(self.today, 0))

    @property
    def remaining(self) -> int:
        return max(0, self.daily_limit - self.used_today)

    def take(self) -> None:
        if self.used_today >= self.daily_limit:
            raise BudgetExceeded(
                f"오늘 네이버 API 호출 {self.used_today}건 — 상한 {self.daily_limit}건(NAVER_DAILY_CALL_BUDGET)에 "
                "도달해 멈췄습니다. 지금까지 받은 결과는 저장됩니다.")
        self.counts[self.today] = self.used_today + 1
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.counts, indent=1), encoding="utf-8")


@dataclass
class NaverClient:
    client_id: str
    client_secret: str
    backend: str = "hub"
    budget: CallBudget = field(default_factory=lambda: CallBudget(None))
    delay: float = 0.2
    session: requests.Session = field(default_factory=lambda: get_session(retries=0))

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"NAVER_API_BACKEND 는 {list(BACKENDS)} 중 하나여야 합니다 (현재 {self.backend!r})")
        self.limiter = RateLimiter(self.delay)

    def get(self, params: dict) -> dict:
        url, id_header, secret_header = BACKENDS[self.backend]
        self.budget.take()
        self.limiter.wait()
        resp = self.session.get(url, params=params, timeout=20,
                                headers={id_header: self.client_id, secret_header: self.client_secret})
        if resp.status_code in (401, 403):
            raise RuntimeError(f"네이버 API 인증 실패({resp.status_code}, backend={self.backend}): 키 값과 "
                               "NAVER_API_BACKEND(hub=API HUB 키 / openapi=개발자센터 키)를 확인하세요. "
                               f"응답: {resp.text[:200]}")
        if resp.status_code == 429:
            raise BudgetExceeded(f"네이버 API 한도 초과 응답(429): {resp.text[:200]}")
        resp.raise_for_status()
        return resp.json()


def estimate_calls(n_queries: int, max_per_query: int) -> int:
    return n_queries * math.ceil(min(max_per_query, 1000) / 100)


def _norm_item(it: dict) -> dict:
    """API HUB 와 개발자센터 응답의 필드 이름 차이(originallink/original_link, pubDate/pub_date)를 맞춘다."""
    return {
        "title": it.get("title", ""),
        "description": it.get("description", ""),
        "originallink": it.get("originallink") or it.get("original_link") or it.get("originalLink") or "",
        "link": it.get("link", ""),
        "pubDate": it.get("pubDate") or it.get("pub_date") or "",
    }

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


def search_news(client: NaverClient, query: str, max_results: int = 300, sort: str = "sim") -> list[dict]:
    items: list[dict] = []
    start = 1
    while start <= min(max_results, 1000):
        display = min(100, max_results - len(items))
        if display <= 0:
            break
        data = client.get({"query": query, "display": display, "start": start, "sort": sort})
        batch = data.get("items")
        if batch is None:   # 응답이 한 단계 감싸져 오는 경우 대비
            batch = (data.get("result") or data.get("data") or {}).get("items", [])
        items.extend(_norm_item(it) for it in batch)
        total = int(data.get("total") or (data.get("result") or {}).get("total") or 0)
        if len(batch) < display or start + display > total:
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


def collect_candidates(ws: Workspace, bundles: list[Bundle], client: NaverClient, max_per_query: int = 300,
                       sort: str = "sim") -> pd.DataFrame:
    resolver = OutletResolver(ws.config_file("outlets.yaml"))
    journals = JournalMatcher(load_yaml(ws.config_file("journals.yaml")))
    rows: list[dict] = []
    try:
        for b in bundles:
            rows.extend(known_article_rows(b, resolver, journals))
            for q in b.queries:
                items = search_news(client, q, max_per_query, sort)
                cands = [r for r in items_to_candidates(b, q, items, resolver, journals) if in_window(r, b)]
                log.info("[%s] '%s' → %d건 중 기간 내 %d건 (오늘 API %d건 사용)", b.bundle_id, q, len(items),
                         len(cands), client.budget.used_today)
                rows.extend(cands)
    except BudgetExceeded as e:
        log.warning("%s", e)
    return save_candidates(ws, rows)


def seed_candidates(ws: Workspace, bundles: list[Bundle]) -> pd.DataFrame:
    """API 없이 bundles.yaml 의 known_articles(사람·웹검색으로 찾은 기사 URL)만 후보로 넣는다."""
    resolver = OutletResolver(ws.config_file("outlets.yaml"))
    journals = JournalMatcher(load_yaml(ws.config_file("journals.yaml")))
    rows = [r for b in bundles for r in known_article_rows(b, resolver, journals)]
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
