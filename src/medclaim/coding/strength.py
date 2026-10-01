"""인과 표현 강도 자동 사전 코딩 (Sumner et al., 2014의 7범주).

    0 진술 없음 | 1 관계 없음 명시 | 2 상관(연관) | 3 모호
    4 조건부 인과(might/may) | 5 인과 가능(can) | 6 단정적 인과

이 모듈의 결과는 사람 코더에게 보여줄 '제안값'이다. 규칙 기반이라 부정의 범위,
인용문 속 주장, 다른 연구를 언급한 문장 등을 구분하지 못한다(LIMITATIONS.md 참고).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from ..nlp.lexicon import Lexicon, mask_terms
from ..nlp.segment import normalize_text

LEVELS = {
    0: "no_statement",
    1: "no_relationship",
    2: "correlational",
    3: "ambiguous",
    4: "conditional_causal",
    5: "can_cause",
    6: "causal",
}
LEVELS_KO = {
    0: "진술 없음",
    1: "관계 없음",
    2: "상관(연관)",
    3: "모호",
    4: "조건부 인과",
    5: "인과 가능",
    6: "단정적 인과",
}
DRUG_MASK = "〈약〉"
WINDOW_STOP = re.compile(r"[,.…·;!?'\"“”‘’()\[\]<>〈〉/|]")
EN_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


def split_english_sentences(text: str) -> list[str]:
    text = normalize_text(text).replace("\n", " ")
    return [s.strip() for s in EN_SENT_SPLIT.split(text) if s.strip()]


@dataclass
class ClaimCode:
    level: int = 0
    cues: list[str] = field(default_factory=list)
    negated: bool = False
    question: bool = False
    direction: str = ""          # increase / decrease / null / mixed / ""
    sentence: str = ""

    @property
    def label(self) -> str:
        return LEVELS[self.level]

    def cue_str(self) -> str:
        return "; ".join(dict.fromkeys(self.cues))


def _resolve(hits: list[tuple[int, str]]) -> tuple[int, list[str], bool]:
    """여러 단서 중 '가장 강한 긍정 주장'을 고른다. 긍정 주장이 없을 때만 1(관계 없음)."""
    positive = [h for h in hits if h[0] >= 2]
    if positive:
        best = max(h[0] for h in positive)
        return best, [f"{lvl}:{txt}" for lvl, txt in positive if lvl == best], False
    neg = [h for h in hits if h[0] == 1]
    if neg:
        return 1, [f"1:{txt}" for _, txt in neg], True
    return 0, [], False


class KoreanClaimCoder:
    def __init__(self, lexicon: Lexicon) -> None:
        self.lex = lexicon
        self.window = int(lexicon.data.get("window", 8))
        self.negation = lexicon.regex("negation_after")
        self.may = lexicon.regex("modal_after", "may")
        self.can = lexicon.regex("modal_after", "can")

    # -- 내부 도우미 -------------------------------------------------------
    def _after(self, text: str, end: int) -> str:
        window = text[end:end + self.window]
        stop = WINDOW_STOP.search(window)
        return window[:stop.start()] if stop else window

    def _mentions_target(self, text: str, outcome_terms: Iterable[str]) -> bool:
        if DRUG_MASK in text:
            return True
        low = text.lower()
        if any(t and t.lower() in low for t in outcome_terms):
            return True
        return self.lex.any(text, "exposure")

    def _direction(self, text: str, level: int) -> str:
        if level == 1:
            return "null"
        if level == 0:
            return ""
        up = self.lex.any(text, "direction", "increase")
        down = self.lex.any(text, "direction", "decrease")
        if up and down:
            return "mixed"
        return "increase" if up else "decrease" if down else ""

    # -- 공개 API ----------------------------------------------------------
    def code_sentence(self, sentence: str, drug_terms: Iterable[str] = (),
                      outcome_terms: Iterable[str] = (), require_target: bool = True) -> ClaimCode:
        raw = normalize_text(sentence)
        s = mask_terms(raw, drug_terms, DRUG_MASK)
        question = "?" in s
        if require_target and not self._mentions_target(s, outcome_terms):
            return ClaimCode(0, question=question, sentence=raw)

        hits: list[tuple[int, str]] = []
        for cue in self.lex.find(s, "claim_strength", "no_relationship"):
            hits.append((1, cue.text))
        for cue in self.lex.find(s, "claim_strength", "causal"):
            after = self._after(s, cue.end)
            if self.negation.search(after):
                hits.append((1, cue.text + "+부정"))
            elif self.may.search(after):
                hits.append((4, cue.text + "+가능성"))
            elif self.can.search(after):
                hits.append((5, cue.text + "+수 있"))
            else:
                hits.append((6, cue.text))
        for key, base in (("conditional", 4), ("ambiguous", 3), ("correlational", 2)):
            for cue in self.lex.find(s, "claim_strength", key):
                after = self._after(s, cue.end)
                if base in (2, 3) and self.negation.search(after):
                    hits.append((1, cue.text + "+부정"))
                else:
                    hits.append((base, cue.text))

        if not any(h[0] >= 2 for h in hits) and self.lex.has("claim_strength", "ambiguous_weak"):
            for cue in self.lex.find(s, "claim_strength", "ambiguous_weak"):
                if self.negation.search(self._after(s, cue.end)):
                    hits.append((1, cue.text + "+부정"))
                else:
                    hits.append((3, cue.text + "(약)"))

        level, cues, negated = _resolve(hits)
        return ClaimCode(level, cues, negated, question, self._direction(s, level), raw)

    def code_unit(self, sentences: Iterable[str], **kw) -> ClaimCode:
        """여러 문장(예: 리드 2문장) 중 가장 강한 주장을 단위의 값으로 쓴다."""
        codes = [self.code_sentence(s, **kw) for s in sentences if s and s.strip()]
        if not codes:
            return ClaimCode(0)
        best = max(codes, key=lambda c: c.level)
        best.question = any(c.question for c in codes)
        return best


class EnglishClaimCoder:
    """영어 논문 초록 결론부용. 영어는 조동사·부정어가 동사 앞에 오므로 앞쪽 창을 본다."""

    def __init__(self, lexicon: Lexicon) -> None:
        self.lex = lexicon
        self.window = int(lexicon.data.get("window_before", 30))
        self.negation = lexicon.regex("negation_before")
        self.may = lexicon.regex("modal_before", "may")
        self.can = lexicon.regex("modal_before", "can")

    def _before(self, text: str, start: int) -> str:
        return text[max(0, start - self.window):start]

    def code_sentence(self, sentence: str) -> ClaimCode:
        s = normalize_text(sentence)
        hits: list[tuple[int, str]] = []
        for cue in self.lex.find(s, "claim_strength", "no_relationship"):
            hits.append((1, cue.text))
        for cue in self.lex.find(s, "claim_strength", "causal"):
            before = self._before(s, cue.start)
            if self.negation.search(before):
                hits.append((1, "not+" + cue.text))
            elif self.may.search(before):
                hits.append((4, "may+" + cue.text))
            elif self.can.search(before):
                hits.append((5, "can+" + cue.text))
            else:
                hits.append((6, cue.text))
        for key, base in (("ambiguous", 3), ("correlational", 2)):
            for cue in self.lex.find(s, "claim_strength", key):
                if self.negation.search(self._before(s, cue.start)):
                    hits.append((1, "not+" + cue.text))
                else:
                    hits.append((base, cue.text))
        level, cues, negated = _resolve(hits)
        if level == 1:
            direction = "null"
        elif level == 0:
            direction = ""
        else:
            up = self.lex.any(s, "direction", "increase")
            down = self.lex.any(s, "direction", "decrease")
            direction = "mixed" if (up and down) else "increase" if up else "decrease" if down else ""
        return ClaimCode(level, cues, negated, "?" in s, direction, s)

    def code_conclusion(self, text: str) -> tuple[ClaimCode, ClaimCode]:
        """(주 주장, 최강 주장). 주 주장 = 결론부에서 처음 나오는 주장 문장 (Sumner 방식)."""
        codes = [self.code_sentence(s) for s in split_english_sentences(text)]
        claim_codes = [c for c in codes if c.level > 0]
        if not claim_codes:
            return ClaimCode(0), ClaimCode(0)
        return claim_codes[0], max(claim_codes, key=lambda c: c.level)
