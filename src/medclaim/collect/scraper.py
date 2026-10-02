"""기사 본문 수집 (include=Y 후보만).

순서
  1) annotation/manual_texts/{article_id}.txt 가 있으면 그것을 쓴다(사람이 붙여넣은 원문 — 최우선)
  2) 언론사 원문 URL(originallink) → 범용 추출기
  3) 실패하면 네이버 뉴스 URL → 네이버 전용 추출기
robots.txt 를 기본으로 지킨다. 막힌 사이트·추출 실패 기사는 article_text_check.csv 에
'manual_needed'로 표시되며, 사람이 manual_texts 에 원문을 붙여넣으면 다음 실행에 반영된다.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from urllib import robotparser
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

from ..paths import Workspace
from ..utils import (RateLimiter, USER_AGENT, decode_html, log, merge_preserving, read_csv,
                     read_jsonl, write_csv, write_jsonl)
from .outlets import OutletResolver, is_naver_news

GENERIC_BODY_SELECTORS = [
    "div.article_view section[dmcf-sid]", "div.article_view",   # 다음(v.daum.net) 기사 페이지
    "#article-view-content-div",      # 많은 지역·전문지 CMS(엔디소프트)
    "div[itemprop='articleBody']",
    "#articleBody", "#article_body", "#articeBody", "#articleText", "#article-body",
    "#news_body_area", "#newsEndContents", "#textBody", "#news_content", "#CmAdContent",
    ".article_body", ".article-body", ".news_body", ".view_con", ".article_txt", ".news_text",
    "#newsct_article", "article",
]
REMOVE_SELECTORS = ["script", "style", "noscript", "iframe", "figure", "figcaption", "table",
                    ".photo", ".caption", ".img_desc", ".end_photo_org", ".ab_photo", ".vod_area",
                    ".reporter_area", ".byline", ".copyright", ".relation_news", ".article_relation",
                    ".ad", "[class*='banner']", "[id*='banner']"]
BOILERPLATE_LINE = re.compile(
    r"무단\s*전재|재배포\s*금지|저작권자|Copyright|ⓒ|©|기사제보|구독\s*신청|^\s*관련\s*기사|"
    r"[\w.+-]+@[\w-]+\.[\w.]+|^\s*[▶☞■◆]")
CAPTION_LINE = re.compile(r"(사진|그래픽|이미지|자료)\s*[=:]|게티이미지|이미지투데이|셔터스톡|클립아트코리아")
DATE_TEXT = re.compile(r"(입력|등록|승인|기사입력|게재)\s*[:]?\s*(20\d{2})[.\-/년 ]\s*(\d{1,2})[.\-/월 ]\s*(\d{1,2})")

TEXT_CHECK_HUMAN_COLS = ["text_ok", "text_fix_notes"]


# ------------------------------------------------------------------ 파싱
def clean_body(text: str) -> str:
    lines = []
    for line in re.split(r"\n+", text or ""):
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        if BOILERPLATE_LINE.search(line) and len(line) < 120:
            continue
        if CAPTION_LINE.search(line) and len(line) < 60:
            continue
        lines.append(line)
    return "\n".join(lines)


def _text_with_breaks(node) -> str:
    for br in node.find_all("br"):
        br.replace_with("\n")
    for p in node.find_all(["p", "div"]):
        p.insert_after("\n")
    return node.get_text()


def _meta(soup: BeautifulSoup, *names: str) -> str:
    for n in names:
        tag = soup.find("meta", attrs={"property": n}) or soup.find("meta", attrs={"name": n})
        if tag and tag.get("content"):
            return tag["content"].strip()
    return ""


def parse_naver_article(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    title_el = soup.select_one("#title_area") or soup.select_one(".media_end_head_headline") \
        or soup.select_one("h2.end_tit")
    title = title_el.get_text(" ", strip=True) if title_el else _meta(soup, "og:title")
    body_el = soup.select_one("#dic_area") or soup.select_one("#newsct_article") \
        or soup.select_one("#articeBody")
    summary = ""
    if body_el:
        summ = body_el.select_one(".media_end_summary")
        if summ:
            summary = summ.get_text(" ", strip=True)
            summ.decompose()
        for sel in REMOVE_SELECTORS:
            for t in body_el.select(sel):
                t.decompose()
    body = clean_body(_text_with_breaks(body_el)) if body_el else ""
    date_el = soup.select_one("[data-date-time]")
    published = date_el["data-date-time"] if date_el else ""
    logo = soup.select_one(".media_end_head_top_logo img[alt]") or soup.select_one(".press_logo img[alt]")
    outlet = logo["alt"].strip() if logo else _meta(soup, "twitter:creator", "og:article:author").split("|")[0]
    return {"title": clean_title(title), "subtitle": summary, "body": body, "published": published,
            "outlet_name": clean_outlet_name(outlet), "extractor": "naver"}


def clean_title(title: str, outlet: str = "") -> str:
    """og:title 의 이중 인코딩(&amp;#39;), 앞머리 '[매체명]', 뒤꼬리 '- 매체명' 을 정리한다."""
    import html as _html
    t = _html.unescape(_html.unescape(title or "")).strip()
    t = re.sub(r"^\[[^\]]{2,15}\]\s*", "", t)
    return _strip_site_suffix(t, outlet)


def clean_outlet_name(name: str) -> str:
    """'Daum | 연합뉴스' → '연합뉴스', '디지털투데이 (DigitalToday)' → '디지털투데이'."""
    name = (name or "").strip()
    if "|" in name:
        parts = [x.strip() for x in name.split("|") if x.strip()]
        name = next((x for x in parts if x.lower() not in ("daum", "다음", "네이버", "naver", "nate", "네이트")),
                    parts[-1] if parts else "")
    return re.sub(r"\s*\([A-Za-z ]+\)$", "", name).strip()


def _strip_site_suffix(title: str, outlet: str) -> str:
    for sep in (" | ", " - ", " : ", " :: ", " < "):
        if sep in title:
            head, tail = title.rsplit(sep, 1)
            if len(tail) <= 20 and (not outlet or outlet.replace(" ", "") in tail.replace(" ", "")
                                    or len(tail) <= 8):
                return head.strip()
    return title.strip()


FUSION_RE = re.compile(r"Fusion\.globalContent=(\{.*?\});Fusion\.", re.S)


def parse_fusion_body(html: str) -> str:
    """조선일보 등 Arc(Fusion) CMS: 본문이 페이지 안 JSON(Fusion.globalContent)의 content_elements 에 있다."""
    m = FUSION_RE.search(html)
    if not m:
        return ""
    try:
        content = json.loads(m.group(1))
    except ValueError:
        return ""
    paras = [BeautifulSoup(e.get("content", ""), "lxml").get_text(" ", strip=True)
             for e in content.get("content_elements", []) if e.get("type") == "text"]
    return clean_body("\n".join(p for p in paras if p))


def parse_generic_article(html: str, outlet: str = "") -> dict:
    soup = BeautifulSoup(html, "lxml")
    title = _meta(soup, "og:title", "twitter:title")
    if not title:
        h1 = soup.find("h1")
        title = h1.get_text(" ", strip=True) if h1 else (soup.title.get_text(strip=True) if soup.title else "")
    title = clean_title(title, outlet)

    best_text, best_sel = parse_fusion_body(html), "fusion-json"
    for sel in ([] if len(best_text) >= 300 else GENERIC_BODY_SELECTORS):
        for el in soup.select(sel):
            for rm in REMOVE_SELECTORS:
                for t in el.select(rm):
                    t.decompose()
            text = clean_body(_text_with_breaks(el))
            if len(text) > len(best_text):
                best_text, best_sel = text, sel
        if len(best_text) >= 300:
            break
    if len(best_text) < 200:   # 마지막 수단: <p> 를 가장 많이 가진 부모
        parents: dict = {}
        for p in soup.find_all("p"):
            parents.setdefault(p.parent, []).append(p.get_text(" ", strip=True))
        if parents:
            parent, texts = max(parents.items(), key=lambda kv: sum(len(t) for t in kv[1]))
            cand = clean_body("\n".join(texts))
            if len(cand) > len(best_text):
                best_text, best_sel = cand, "p-cluster"

    published = _meta(soup, "article:published_time", "og:regDate", "pubdate", "date")
    if not published:
        t = soup.find("time", attrs={"datetime": True})
        published = t["datetime"] if t else ""
    if not published:
        m = DATE_TEXT.search(soup.get_text(" "))
        if m:
            published = f"{m.group(2)}-{int(m.group(3)):02d}-{int(m.group(4)):02d}"
    m14 = re.fullmatch(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", published or "")
    if m14:   # 다음 og:regDate 형식 20240107101538
        published = f"{m14[1]}-{m14[2]}-{m14[3]}T{m14[4]}:{m14[5]}"
    return {"title": title, "subtitle": _meta(soup, "og:description")[:200], "body": best_text,
            "published": published, "outlet_name": clean_outlet_name(_meta(soup, "og:site_name")),
            "extractor": f"generic:{best_sel}"}


def read_manual_text(ws: Workspace, article_id: str) -> dict | None:
    """manual_texts/{article_id}.txt — 첫 줄 = 제목, 나머지 = 본문."""
    path = ws.manual_text_dir / f"{article_id}.txt"
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return None
    return {"title": lines[0].strip(), "subtitle": "", "body": clean_body("\n".join(lines[1:])),
            "published": "", "outlet_name": "", "extractor": "manual"}


# ------------------------------------------------------------------ robots.txt
# 이 도구는 AI 에이전트(Claude 등)가 대신 돌리는 경우가 있다. 그래서 AI 크롤러를 따로 막아 둔 사이트
# (예: 'User-agent: ClaudeBot / Disallow: /')도 수집하지 않는다 → 필요하면 사람이 manual_texts 에 붙여넣는다.
AI_AGENTS = ("ClaudeBot", "anthropic-ai", "Claude-Web", "Claude-User")


class RobotsCache:
    def __init__(self, session: requests.Session, respect: bool = True) -> None:
        self.session = session
        self.respect = respect
        self.cache: dict[str, robotparser.RobotFileParser | None] = {}

    def allowed(self, url: str) -> bool:
        return not self.blocked_by(url)

    def blocked_by(self, url: str) -> str:
        """url 을 막은 user-agent 이름. 허용이면 ''."""
        if not self.respect:
            return ""
        p = urlparse(url)
        base = f"{p.scheme}://{p.netloc}"
        if base not in self.cache:
            rp = robotparser.RobotFileParser()
            try:
                resp = self.session.get(base + "/robots.txt", timeout=10)
                if resp.status_code >= 400:
                    rp = None   # robots.txt 없음 → 허용으로 간주
                else:
                    rp.parse(resp.text.splitlines())
            except requests.RequestException:
                rp = None
            self.cache[base] = rp
        rp = self.cache[base]
        if rp is None:
            return ""
        for agent in (USER_AGENT, *AI_AGENTS):
            if not rp.can_fetch(agent, url):
                return agent
        return ""


# ------------------------------------------------------------------ 실행
def selected_candidates(ws: Workspace, use_auto: bool = False) -> pd.DataFrame:
    if not ws.candidates_csv.exists():
        raise FileNotFoundError(f"{ws.candidates_csv} 가 없습니다. search-news 또는 import-bigkinds 를 먼저 실행하세요.")
    df = read_csv(ws.candidates_csv)
    for col in ("include", "auto_include", "link_type"):
        if col not in df.columns:
            df[col] = ""
    inc = df["include"].str.strip().str.upper()
    mask = inc.eq("Y")
    if use_auto:
        mask = mask | (inc.eq("") & df["auto_include"].str.upper().eq("Y"))
    if "link_type" in df.columns:
        mask &= ~df["link_type"].str.strip().str.lower().isin(["background", "unrelated"])
    return df[mask]


def scrape_articles(ws: Workspace, session: requests.Session, use_auto: bool = False,
                    delay: float = 2.0, respect_robots: bool = True, force: bool = False) -> pd.DataFrame:
    ws.ensure_dirs()
    resolver = OutletResolver(ws.config_file("outlets.yaml"))
    robots = RobotsCache(session, respect_robots)
    limiter = RateLimiter(delay)
    existing = {a["article_id"]: a for a in read_jsonl(ws.articles_jsonl)}
    cands = selected_candidates(ws, use_auto)
    if cands.empty:
        log.warning("include=Y 인 후보가 없습니다. article_candidates.csv 의 include 열을 먼저 채우세요.")
    out: dict[str, dict] = {}
    for _, c in cands.iterrows():
        aid = c["candidate_id"]
        parsed, status, used_url = read_manual_text(ws, aid), "", ""
        if parsed is None and aid in existing and not force and existing[aid].get("status") == "ok":
            out[aid] = existing[aid]
            continue
        if parsed:
            status = "ok"
        else:
            for url, kind in ((c.get("url", ""), "generic"), (c.get("naver_url", ""), "naver")):
                if not url:
                    continue
                blocked = robots.blocked_by(url)
                if blocked:
                    ai = "" if blocked == USER_AGENT else f"(AI 크롤러 차단: {blocked})"
                    status = f"robots_disallowed:{urlparse(url).netloc}{ai}"
                    continue
                try:
                    limiter.wait()
                    resp = session.get(url, timeout=20)
                    resp.raise_for_status()
                except requests.RequestException as e:
                    status = f"http_error:{type(e).__name__}"
                    continue
                html = decode_html(resp)
                (ws.html_dir / f"{aid}.html").write_text(html, encoding="utf-8")
                parsed = parse_naver_article(html) if (kind == "naver" or is_naver_news(resp.url)) \
                    else parse_generic_article(html, c.get("outlet", ""))
                used_url = resp.url
                if len(parsed["body"]) >= 200:
                    status = "ok"
                    break
                status = "short_body"
        parsed = parsed or {"title": c["title"], "subtitle": "", "body": "", "published": "",
                            "outlet_name": "", "extractor": "none"}
        outlet, otype = resolver.resolve(c.get("url", ""), parsed.get("outlet_name") or c.get("outlet", ""))
        if otype == "unknown" and c.get("outlet_type") not in ("", "unknown"):
            outlet, otype = c["outlet"], c["outlet_type"]
        out[aid] = {
            "article_id": aid, "bundle_id": c["bundle_id"], "url": c.get("url", ""),
            "naver_url": c.get("naver_url", ""), "fetched_url": used_url,
            "outlet": outlet, "outlet_type": otype,
            "title": parsed["title"] or c["title"], "subtitle": parsed.get("subtitle", ""),
            "body": parsed["body"], "published": parsed["published"] or c.get("pub_date", ""),
            "extractor": parsed["extractor"], "status": status or "no_url",
            "scraped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        log.info("[%s] %s %s (%d자)", status or "no_url", outlet, out[aid]["title"][:40], len(parsed["body"]))
    # 이번에 선택되지 않은 기존 기사도 보존 (include 를 바꿨다면 precode 단계에서 걸러짐)
    for aid, art in existing.items():
        out.setdefault(aid, art)
    write_jsonl(out.values(), ws.articles_jsonl)
    return write_text_check(ws, list(out.values()))


def write_text_check(ws: Workspace, articles: list[dict]) -> pd.DataFrame:
    rows = []
    for a in articles:
        n = len(a.get("body", ""))
        flag = "ok"
        if a.get("status") != "ok":
            flag = "manual_needed"
        elif n < 400:
            flag = "check_short"
        rows.append({"article_id": a["article_id"], "bundle_id": a["bundle_id"], "outlet": a["outlet"],
                     "title": a["title"], "body_len": n, "extractor": a.get("extractor", ""),
                     "status": a.get("status", ""), "flag": flag,
                     "body_head": a.get("body", "")[:150].replace("\n", " "),
                     "body_tail": a.get("body", "")[-120:].replace("\n", " "),
                     "manual_text_file": f"annotation/manual_texts/{a['article_id']}.txt"})
    df = pd.DataFrame(rows)
    df = merge_preserving(df, ws.text_check_csv, "article_id", TEXT_CHECK_HUMAN_COLS)
    write_csv(df, ws.text_check_csv)
    n_bad = int((df["flag"] != "ok").sum()) if len(df) else 0
    log.info("본문 점검표 → %s (확인 필요 %d건)", ws.text_check_csv, n_bad)
    return df
