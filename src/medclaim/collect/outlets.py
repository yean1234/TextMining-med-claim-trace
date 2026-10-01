"""URL·언론사명 → 매체명/매체 유형."""
from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

from ..utils import load_yaml

NAVER_ARTICLE_RE = re.compile(r"/article/(?:comment/)?(\d{3})/(\d{6,12})")


def naver_ids(url: str) -> tuple[str, str] | None:
    """네이버 뉴스 URL → (oid, aid). 아니면 None."""
    if not url:
        return None
    parsed = urlparse(url)
    if "naver.com" not in parsed.netloc:
        return None
    m = NAVER_ARTICLE_RE.search(parsed.path)
    if m:
        return m.group(1), m.group(2)
    qs = parse_qs(parsed.query)
    if "oid" in qs and "aid" in qs:
        return qs["oid"][0], qs["aid"][0]
    return None


def is_naver_news(url: str) -> bool:
    return naver_ids(url) is not None


class OutletResolver:
    def __init__(self, path) -> None:
        data = load_yaml(path)
        self.type_labels: dict[str, str] = data.get("outlet_types", {})
        self.by_domain: list[tuple[str, dict]] = []
        self.by_oid: dict[str, dict] = {}
        self.by_name: dict[str, dict] = {}
        for o in data.get("outlets", []):
            for d in o.get("domains", []):
                self.by_domain.append((d.lower(), o))
            if o.get("naver_oid"):
                self.by_oid[str(o["naver_oid"]).zfill(3)] = o
            self.by_name[_norm_name(o["name"])] = o
        # 'health.chosun.com'이 'chosun.com'보다 먼저 맞도록 긴 도메인부터
        self.by_domain.sort(key=lambda x: len(x[0]), reverse=True)

    def resolve(self, url: str = "", name: str = "") -> tuple[str, str]:
        """(매체명, 매체 유형). 모르면 (name 또는 도메인, 'unknown')."""
        if name and _norm_name(name) in self.by_name:
            o = self.by_name[_norm_name(name)]
            return o["name"], o["type"]
        host = urlparse(url or "").netloc.lower()
        for domain, o in self.by_domain:
            if host == domain or host.endswith("." + domain):
                return o["name"], o["type"]
        ids = naver_ids(url or "")
        if ids and ids[0] in self.by_oid:
            o = self.by_oid[ids[0]]
            return o["name"], o["type"]
        return (name or host or ""), "unknown"


def _norm_name(name: str) -> str:
    return re.sub(r"\s+", "", str(name or "")).lower()
