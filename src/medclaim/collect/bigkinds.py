"""BigKinds(빅카인즈, bigkinds.or.kr) 뉴스 검색 결과 엑셀 → 기사 후보로 가져오기.

네이버 API는 날짜 필터가 없어 오래된 기사를 놓치기 쉽다. BigKinds 는 기간·언론사를 지정해
검색하고 결과를 엑셀로 내려받을 수 있으므로 보조 수집 경로로 쓴다.
(내려받은 엑셀의 '본문'은 앞부분 일부만 들어 있으므로 본문은 scrape 단계에서 다시 받는다.)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..bundles import Bundle
from ..coding.features import JournalMatcher
from ..coding.relevance import score_candidate
from ..paths import Workspace
from ..utils import load_yaml, log, read_csv, stable_id
from .naver import normalize_url, save_candidates
from .outlets import OutletResolver

# BigKinds 엑셀 열 이름 → 내부 이름 (버전에 따라 조금씩 달라서 후보를 여러 개 둔다)
COLUMN_ALIASES = {
    "pub_date": ["일자", "날짜", "date"],
    "outlet": ["언론사", "매체", "media"],
    "title": ["제목", "title"],
    "description": ["본문", "내용", "content"],
    "url": ["URL", "url", "링크"],
}


def _pick(df: pd.DataFrame, names: list[str]) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    return None


def import_bigkinds(ws: Workspace, path: Path, bundle: Bundle) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xls"):
        raw = pd.read_excel(path, dtype=str).fillna("")
    else:
        raw = read_csv(path)
    cols = {k: _pick(raw, v) for k, v in COLUMN_ALIASES.items()}
    if not cols["title"]:
        raise ValueError(f"제목 열을 찾지 못했습니다. 열 목록: {list(raw.columns)}")
    resolver = OutletResolver(ws.config_file("outlets.yaml"))
    journals = JournalMatcher(load_yaml(ws.config_file("journals.yaml")))
    rows = []
    for _, r in raw.iterrows():
        get = lambda k: str(r[cols[k]]).strip() if cols[k] else ""  # noqa: E731
        date_raw = get("pub_date")
        pub = pd.to_datetime(date_raw, errors="coerce", format="%Y%m%d" if date_raw.isdigit() else None)
        outlet, otype = resolver.resolve(get("url"), get("outlet"))
        title, desc = get("title"), get("description")[:300]
        score, auto, reasons = score_candidate(bundle, title, desc, None if pd.isna(pub) else pub.date(),
                                               journals)
        rows.append({
            "candidate_id": stable_id(bundle.bundle_id, normalize_url(get("url")) or f"{outlet}|{title}"),
            "bundle_id": bundle.bundle_id, "source": "bigkinds", "query": path.name,
            "pub_date": "" if pd.isna(pub) else pub.strftime("%Y-%m-%d"),
            "outlet": outlet, "outlet_type": otype, "title": title, "description": desc,
            "url": get("url"), "naver_url": "", "relevance_score": score, "auto_include": auto,
            "relevance_reasons": "; ".join(reasons),
        })
    log.info("BigKinds %s → %d건 (%s)", path.name, len(rows), bundle.bundle_id)
    existing = read_csv(ws.candidates_csv).to_dict("records") if ws.candidates_csv.exists() else []
    return save_candidates(ws, existing + rows)
