"""bundles.yaml 읽기·검증."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from .utils import load_yaml

BUNDLE_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


@dataclass
class Bundle:
    bundle_id: str
    label: str
    drug_terms: list[str]
    outcome_terms: list[str]
    paper: dict
    queries: list[str]
    date_from: date | None
    date_to: date | None
    known_articles: list[dict] = field(default_factory=list)
    notes: str = ""

    @property
    def journal(self) -> str:
        return str(self.paper.get("journal") or "")


def _to_date(value) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def load_bundles(path: Path) -> list[Bundle]:
    data = load_yaml(path)
    raw = data.get("bundles") or []
    bundles: list[Bundle] = []
    errors: list[str] = []
    seen: set[str] = set()
    for i, b in enumerate(raw):
        bid = str(b.get("bundle_id", "")).strip()
        where = f"bundles[{i}] ({bid or '?'})"
        if not bid or not BUNDLE_ID_RE.match(bid):
            errors.append(f"{where}: bundle_id 는 영문/숫자/_/- 만 쓰세요")
        if bid in seen:
            errors.append(f"{where}: bundle_id 중복")
        seen.add(bid)
        paper = b.get("paper") or {}
        if not (paper.get("doi") or paper.get("title")):
            errors.append(f"{where}: paper.doi 또는 paper.title 중 하나는 있어야 합니다")
        search = b.get("search") or {}
        queries = [q for q in (search.get("queries") or []) if str(q).strip()]
        if not queries:
            errors.append(f"{where}: search.queries 가 비어 있습니다")
        drug_terms = [str(t) for t in (b.get("drug_terms") or [])]
        outcome_terms = [str(t) for t in (b.get("outcome_terms") or [])]
        if not drug_terms or not outcome_terms:
            errors.append(f"{where}: drug_terms / outcome_terms 가 필요합니다")
        try:
            d_from, d_to = _to_date(search.get("date_from")), _to_date(search.get("date_to"))
        except ValueError as e:
            errors.append(f"{where}: 날짜 형식 오류 ({e})")
            d_from = d_to = None
        if d_from and d_to and d_from > d_to:
            errors.append(f"{where}: date_from 이 date_to 보다 늦습니다")
        bundles.append(Bundle(
            bundle_id=bid,
            label=str(b.get("label") or bid),
            drug_terms=drug_terms,
            outcome_terms=outcome_terms,
            paper={k: ("" if v is None else str(v)) for k, v in paper.items()},
            queries=[str(q) for q in queries],
            date_from=d_from,
            date_to=d_to,
            known_articles=list(b.get("known_articles") or []),
            notes=str(b.get("notes") or "").strip(),
        ))
    if errors:
        raise ValueError("bundles.yaml 검증 실패:\n  - " + "\n  - ".join(errors))
    return bundles


def bundle_map(bundles: list[Bundle]) -> dict[str, Bundle]:
    return {b.bundle_id: b for b in bundles}
