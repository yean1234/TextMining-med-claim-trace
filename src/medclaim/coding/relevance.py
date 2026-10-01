"""검색으로 찾은 기사 후보가 '이 묶음의 논문을 보도한 기사'인지 점수 매기기.

점수는 사람이 include 여부를 판단할 때 정렬·우선순위를 정하는 용도다.
같은 약물·같은 부작용을 다룬 '다른 논문' 기사(예: 사례2 타이레놀 A/B 논문, 사례5 반박 코멘터리)는
점수만으로 가려낼 수 없으므로 사람이 link_type 을 반드시 확인해야 한다.
"""
from __future__ import annotations

from datetime import date, datetime

from ..bundles import Bundle
from .features import JournalMatcher

RESEARCH_WORDS = ["연구", "논문", "학술지", "저널", "게재", "연구팀", "연구진", "분석 결과", "발표"]


def _contains_any(text: str, terms) -> list[str]:
    low = (text or "").lower()
    return [t for t in terms if t and t.lower() in low]


def _paper_date(bundle: Bundle) -> date | None:
    value = bundle.paper.get("published") or ""
    try:
        return date.fromisoformat(str(value)[:10]) if value else None
    except ValueError:
        return None


def score_candidate(bundle: Bundle, title: str, description: str, pub_date: datetime | date | None,
                    journals: JournalMatcher) -> tuple[int, str, list[str]]:
    """(점수, 자동 제안 Y/N/?, 근거 목록)"""
    reasons: list[str] = []
    score = 0
    drug_t = _contains_any(title, bundle.drug_terms)
    drug_d = _contains_any(description, bundle.drug_terms)
    out_t = _contains_any(title, bundle.outcome_terms)
    out_d = _contains_any(description, bundle.outcome_terms)
    if drug_t:
        score += 2
        reasons.append(f"제목에 약물({drug_t[0]})")
    elif drug_d:
        score += 1
        reasons.append(f"요약에 약물({drug_d[0]})")
    if out_t:
        score += 2
        reasons.append(f"제목에 결과({out_t[0]})")
    elif out_d:
        score += 1
        reasons.append(f"요약에 결과({out_d[0]})")

    found, _ = journals.find(f"{title} {description}")
    if bundle.journal and bundle.journal in found:
        score += 3
        reasons.append(f"저널 언급({bundle.journal})")
    elif found:
        score -= 1
        reasons.append(f"다른 저널 언급({', '.join(found)})")

    author = bundle.paper.get("first_author", "")
    if author and author.lower() in f"{title} {description}".lower():
        score += 2
        reasons.append(f"저자 언급({author})")

    if _contains_any(f"{title} {description}", RESEARCH_WORDS):
        score += 1
        reasons.append("연구 보도 표현")

    p_date = _paper_date(bundle)
    if pub_date is not None:
        d = pub_date.date() if isinstance(pub_date, datetime) else pub_date
        if bundle.date_from and d < bundle.date_from:
            score -= 3
            reasons.append("논문 공개 이전 기사")
        elif bundle.date_to and d > bundle.date_to:
            score -= 1
            reasons.append("수집 기간 이후")
        elif p_date and 0 <= (d - p_date).days <= 14:
            score += 1
            reasons.append("논문 공개 2주 이내")

    has_drug = bool(drug_t or drug_d)
    has_outcome = bool(out_t or out_d)
    if not (has_drug and has_outcome):
        suggestion = "N"
    elif score >= 7:
        suggestion = "Y"
    else:
        suggestion = "?"
    return score, suggestion, reasons
