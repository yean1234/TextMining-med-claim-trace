"""기사 1건 → 자동 특징(사전 코딩 값) 추출.

추출 항목
  - 제목/리드/나머지 본문의 인과 표현 강도(0~6)·방향·단서
  - 적용 조건 범주(대상·동물·표본수·용량·기간·설계·효과크기·한계)의 구간별 언급 여부
  - 불확실성(hedge)·확실성 강화(booster)·선정성 표현 수
  - 행동 권고 수준(0~3), 한계/주의(caveat) 문장
  - 저널명 언급과 묶음 논문 저널과의 일치 여부
"""
from __future__ import annotations

import re

from ..bundles import Bundle
from ..nlp.lexicon import Lexicon, mask_terms
from ..nlp.segment import is_quoted, normalize_text, segment_article
from .strength import DRUG_MASK, KoreanClaimCoder

CONDITION_CATEGORIES = ["population", "animal", "sample_size", "dose", "duration", "design",
                        "effect_size", "limitation"]


class JournalMatcher:
    def __init__(self, journals_cfg: dict) -> None:
        cfg = dict(journals_cfg.get("journals") or {})
        self.not_journal = [str(a) for a in cfg.pop("_not_journal", [])]
        self.aliases: list[tuple[str, str]] = []
        for key, aliases in cfg.items():
            for a in aliases or []:
                self.aliases.append((str(a), str(key)))
        # 긴 별칭부터 찾아서 '네이처 메디슨'이 '네이처'로 잡히지 않게 한다.
        self.aliases.sort(key=lambda x: len(x[0]), reverse=True)

    @staticmethod
    def _pattern(alias: str) -> re.Pattern:
        if re.fullmatch(r"[A-Za-z .]+", alias):
            return re.compile(r"(?<![A-Za-z])" + re.escape(alias) + r"(?![A-Za-z])")
        return re.compile(re.escape(alias))

    def find(self, text: str) -> tuple[list[str], list[str]]:
        """(언급된 저널 키 목록, 저널처럼 인용된 비저널 이름 목록)"""
        text = text or ""
        non_journal = []
        for name in sorted(self.not_journal, key=len, reverse=True):
            if self._pattern(name).search(text):
                non_journal.append(name)
                text = self._pattern(name).sub(" ", text)
        found: list[str] = []
        for alias, key in self.aliases:
            pat = self._pattern(alias)
            if pat.search(text):
                if key not in found:
                    found.append(key)
                text = pat.sub(" ", text)
        return found, non_journal

    def aliases_for(self, journal: str) -> list[str]:
        return [a for a, k in self.aliases if k == journal]


def condition_hits(lex: Lexicon, text: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for cat in lex.subkeys("conditions"):
        cues = lex.find(text, "conditions", cat)
        out[cat] = [c.text for c in cues]
    return out


def advice_level(lex: Lexicon, text: str) -> tuple[int, list[str]]:
    for level in (3, 2, 1):
        cues = lex.find(text, "advice", level)
        if cues:
            return level, [c.text for c in cues]
    return 0, []


def count(lex: Lexicon, text: str, key: str) -> tuple[int, list[str]]:
    cues = lex.find(text, key)
    return len(cues), [c.text for c in cues]


def extract_features(article: dict, bundle: Bundle | None, lex: Lexicon,
                     coder: KoreanClaimCoder, journals: JournalMatcher) -> dict:
    title = normalize_text(article.get("title", ""))
    body = article.get("body", "") or ""
    seg = segment_article(title, body)
    drug_terms = bundle.drug_terms if bundle else []
    outcome_terms = bundle.outcome_terms if bundle else []
    kw = {"drug_terms": drug_terms, "outcome_terms": outcome_terms}

    # 제목은 기사 전체의 주제를 말하므로 약물명이 없어도(예: '연관성 있어도 원인 아냐') 코딩한다.
    t_code = coder.code_unit([title], require_target=False, **kw)
    l_code = coder.code_unit(seg["lead_sentences"], **kw)
    r_code = coder.code_unit(seg["rest_sentences"], **kw)

    feats: dict = {
        "article_id": article.get("article_id", ""),
        "bundle_id": article.get("bundle_id", ""),
        "outlet": article.get("outlet", ""),
        "outlet_type": article.get("outlet_type", "unknown"),
        "published": article.get("published", ""),
        "url": article.get("url", ""),
        "title": title,
        "lead": seg["lead"],
        "body_len": len(body),
        "n_sentences": len(seg["sentences"]),
        "auto_title_strength": t_code.level,
        "auto_title_label": t_code.label,
        "auto_title_cues": t_code.cue_str(),
        "auto_title_direction": t_code.direction,
        "auto_title_question": int(t_code.question),
        "auto_title_quoted": int(is_quoted(title)),
        "auto_lead_strength": l_code.level,
        "auto_lead_label": l_code.label,
        "auto_lead_cues": l_code.cue_str(),
        "auto_lead_direction": l_code.direction,
        "auto_body_max_strength": r_code.level,
        "auto_body_max_cues": r_code.cue_str(),
        "auto_body_max_sentence": r_code.sentence[:200],
    }
    # 방향: 제목 → 리드 순으로 처음 결정되는 값
    feats["auto_direction"] = t_code.direction or l_code.direction or r_code.direction

    sections = {"title": title, "lead": seg["lead"], "body": " ".join(seg["sentences"])}
    for sec, text in sections.items():
        masked = mask_terms(text, drug_terms, DRUG_MASK)
        conds = condition_hits(lex, masked)
        for cat in CONDITION_CATEGORIES:
            hits = conds.get(cat, [])
            feats[f"auto_cond_{sec}_{cat}"] = int(bool(hits))
            feats[f"auto_cond_{sec}_{cat}_cues"] = "; ".join(dict.fromkeys(hits))[:150]
        for key in ("hedges", "boosters", "sensational"):
            n, cues = count(lex, masked, key)
            feats[f"n_{key}_{sec}"] = n
            if sec == "title":
                feats[f"{key}_title_cues"] = "; ".join(cues)

    # 한계·주의 문장 (본문 전체에서)
    caveats = [s for s in seg["sentences"] if lex.find(s, "conditions", "limitation")]
    feats["auto_caveat"] = int(bool(caveats))
    feats["auto_caveat_sentences"] = " || ".join(c[:160] for c in caveats[:3])

    adv, adv_cues = advice_level(lex, " ".join([title] + seg["sentences"]))
    feats["auto_advice"] = adv
    feats["auto_advice_cues"] = "; ".join(dict.fromkeys(adv_cues))

    full = " ".join([title, body])
    found, non_journal = journals.find(full)
    feats["journals_mentioned"] = "; ".join(found)
    feats["non_journal_sources"] = "; ".join(non_journal)
    target = bundle.journal if bundle else ""
    feats["auto_journal_match"] = (1 if target in found else 0) if (found or non_journal) else ""
    feats["n_attribution"] = len(lex.find(full, "attribution"))
    return feats
