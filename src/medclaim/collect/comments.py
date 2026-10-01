"""네이버 뉴스 댓글 수집 (RQ3용, 선택 단계).

주의
  - 네이버가 공개한 API가 아니라 웹페이지가 내부적으로 쓰는 엔드포인트다. 예고 없이 바뀔 수 있고,
    이용약관상 자동 수집이 제한될 수 있으므로 CLI 에서 --i-accept-naver-terms 를 줘야만 실행된다.
  - 수업 과제 범위의 소량·저속 수집을 전제로 한다(기사당 최대 N개, 요청 간 지연).
  - 개인정보 보호: 작성자 ID·닉네임·프로필은 저장하지 않는다. 댓글 본문·시각·공감수만 남긴다.
  - 네이버에 '인링크'로 송고된 기사만 댓글이 있다. 의약전문지 기사는 대부분 댓글이 없다.
"""
from __future__ import annotations

import json
import re

import requests

from ..paths import Workspace
from ..utils import RateLimiter, log, read_jsonl, stable_id, write_jsonl
from .outlets import naver_ids

COMMENT_API = "https://apis.naver.com/commentBox/cbox/web_naver_list_jsonp.json"


def parse_jsonp(text: str) -> dict:
    m = re.search(r"^[^(]*\((.*)\)\s*;?\s*$", text.strip(), flags=re.S)
    return json.loads(m.group(1) if m else text)


def fetch_comments(session: requests.Session, oid: str, aid: str, max_comments: int = 200,
                   sort: str = "FAVORITE", limiter: RateLimiter | None = None) -> list[dict]:
    headers = {"Referer": f"https://n.news.naver.com/mnews/article/comment/{oid}/{aid}"}
    out: list[dict] = []
    page = 1
    while len(out) < max_comments:
        params = {"ticket": "news", "templateId": "default_society", "pool": "cbox5", "_callback": "cb",
                  "lang": "ko", "country": "KR", "objectId": f"news{oid},{aid}", "pageSize": 100,
                  "indexSize": 10, "listType": "OBJECT", "pageType": "more", "page": page, "sort": sort}
        if limiter:
            limiter.wait()
        resp = session.get(COMMENT_API, params=params, headers=headers, timeout=20)
        resp.raise_for_status()
        data = parse_jsonp(resp.text)
        result = data.get("result") or {}
        batch = result.get("commentList") or []
        for c in batch:
            text = (c.get("contents") or "").strip()
            if not text or c.get("deleted"):
                continue
            out.append({
                "comment_id": stable_id(oid, aid, c.get("commentNo", ""), text),
                "text": text,
                "reg_time": c.get("regTime", ""),
                "likes": c.get("sympathyCount", 0),
                "dislikes": c.get("antipathyCount", 0),
            })
        page_model = result.get("pageModel") or {}
        last_page = page_model.get("lastPage") or page_model.get("totalPages") or page
        if not batch or page >= int(last_page):
            break
        page += 1
    return out[:max_comments]


def collect_comments(ws: Workspace, session: requests.Session, max_per_article: int = 200,
                     sort: str = "FAVORITE", delay: float = 2.0) -> int:
    limiter = RateLimiter(delay)
    articles = read_jsonl(ws.articles_jsonl)
    existing = read_jsonl(ws.comments_jsonl)
    done = {c["article_id"] for c in existing}
    rows = list(existing)
    for a in articles:
        if a["article_id"] in done:
            continue
        ids = naver_ids(a.get("naver_url") or "") or naver_ids(a.get("fetched_url") or "")
        if not ids:
            continue
        try:
            comments = fetch_comments(session, ids[0], ids[1], max_per_article, sort, limiter)
        except (requests.RequestException, ValueError) as e:
            log.warning("댓글 수집 실패 %s: %s", a["article_id"], e)
            continue
        for c in comments:
            c.update({"article_id": a["article_id"], "bundle_id": a["bundle_id"], "sort": sort})
        rows.extend(comments)
        log.info("[%s] 댓글 %d개", a["article_id"], len(comments))
    write_jsonl(rows, ws.comments_jsonl)
    return len(rows)
