"""새 묶음 후보 찾기: '어떤 약물-부작용 연구가 한국어로 여러 번 보도됐나'를 센다.

출력 annotation/bundle_candidates.csv
  drug, outcome, journal, month, n_articles, n_outlets, example_titles, example_urls
→ n_articles ≥ 3 인 줄부터 원논문을 확인하고 bundles.yaml 에 추가한다.
(같은 약물·결과·달이라도 서로 다른 논문일 수 있다 — 반드시 기사 몇 개를 열어 확인)
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd
import requests

from ..coding.features import JournalMatcher
from ..paths import Workspace
from ..utils import load_yaml, log, write_csv
from .naver import (BudgetExceeded, NaverClient, clean_api_text, estimate_calls, normalize_url, parse_pubdate,
                    search_news)
from .outlets import OutletResolver


def discovery_queries(ws: Workspace) -> list[tuple[str, str]]:
    cfg = load_yaml(ws.config_file("discovery.yaml"))
    return [(drug, tpl.format(drug=alias)) for drug, aliases in (cfg.get("drugs") or {}).items()
            for alias in aliases for tpl in cfg.get("query_templates", [])]


def estimate_discovery_calls(ws: Workspace, max_per_query: int | None = None) -> int:
    cfg = load_yaml(ws.config_file("discovery.yaml"))
    return estimate_calls(len(discovery_queries(ws)), max_per_query or int(cfg.get("max_per_query", 100)))


def discover_bundles(ws: Workspace, client: NaverClient, max_per_query: int | None = None) -> pd.DataFrame:
    cfg = load_yaml(ws.config_file("discovery.yaml"))
    journals = JournalMatcher(load_yaml(ws.config_file("journals.yaml")))
    resolver = OutletResolver(ws.config_file("outlets.yaml"))
    max_per_query = max_per_query or int(cfg.get("max_per_query", 100))
    outcomes = [str(o) for o in cfg.get("outcomes", [])]
    groups: dict[tuple, dict] = defaultdict(lambda: {"urls": set(), "outlets": set(), "titles": []})
    for drug, query in discovery_queries(ws):
        try:
            items = search_news(client, query, max_per_query, "sim")
        except BudgetExceeded as e:
            log.warning("%s", e)
            break
        except requests.RequestException as e:
            log.warning("검색 실패 '%s': %s", query, e)
            continue
        for it in items:
            title = clean_api_text(it.get("title", ""))
            desc = clean_api_text(it.get("description", ""))
            text = f"{title} {desc}"
            found_outcomes = [o for o in outcomes if o in text]
            found_journals, _ = journals.find(text)
            if not found_outcomes or not found_journals:
                continue   # 결과(부작용)와 저널이 둘 다 언급된 '연구 보도'만 센다
            pub = parse_pubdate(it.get("pubDate", ""))
            month = pub.strftime("%Y-%m") if pub else ""
            url = it.get("originallink") or it.get("link") or ""
            outlet, _ = resolver.resolve(url)
            for o in found_outcomes[:2]:
                for j in found_journals[:2]:
                    g = groups[(drug, o, j, month)]
                    if normalize_url(url) in g["urls"]:
                        continue
                    g["urls"].add(normalize_url(url))
                    g["outlets"].add(outlet)
                    if len(g["titles"]) < 4:
                        g["titles"].append(f"[{outlet}] {title}")
    rows = [{"drug": d, "outcome": o, "journal": j, "month": m, "n_articles": len(g["urls"]),
             "n_outlets": len(g["outlets"]), "example_titles": " || ".join(g["titles"]),
             "example_urls": " ".join(list(g["urls"])[:4]), "picked": "", "paper_doi": "", "notes": ""}
            for (d, o, j, m), g in groups.items()]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["n_outlets", "n_articles"], ascending=False)
    write_csv(df, ws.discovery_csv)
    log.info("묶음 후보 %d개 → %s (n_articles≥3: %d개)", len(df), ws.discovery_csv,
             int((df["n_articles"] >= 3).sum()) if len(df) else 0)
    return df
