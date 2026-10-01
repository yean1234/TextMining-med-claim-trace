"""Kiwi 형태소 분석 기반 토큰화.

노션 정리 내용대로 '명사만' 뽑지 않는다. 부정(않다·없다·아니다·못)과 가능(수 있다)처럼
인과 강도·불확실성을 나타내는 형태소를 남겨야 이 연구의 핵심 정보가 보존된다.
"""
from __future__ import annotations

import re
from typing import Iterable

from .segment import normalize_text

CONTENT_TAGS = {"NNG", "NNP", "SL", "SH", "XR", "MAG"}
PREDICATE_TAGS = {"VV", "VA", "VX", "VCN"}
KEEP_FUNCTIONAL = {("NNB", "수")}
FALLBACK_TOKEN = re.compile(r"[가-힣]+|[A-Za-z][A-Za-z0-9\-]+|\d+(?:\.\d+)?")


class Tokenizer:
    def __init__(self, user_words: Iterable[str] = (), stopwords: Iterable[str] = ()) -> None:
        self.stopwords = {w.strip().lower() for w in stopwords if w and w.strip()}
        self.kiwi = None
        try:
            from kiwipiepy import Kiwi
            self.kiwi = Kiwi()
            for w in {w.strip() for w in user_words if w and w.strip()}:
                if " " not in w and len(w) >= 2:
                    self.kiwi.add_user_word(w, "NNP", 0.0)
        except ImportError:  # pragma: no cover
            self.kiwi = None

    def __call__(self, text: str) -> list[str]:
        return self.tokens(text)

    def tokens(self, text: str) -> list[str]:
        text = normalize_text(text)
        if not text:
            return []
        if self.kiwi is None:
            toks = [t.lower() for t in FALLBACK_TOKEN.findall(text)]
        else:
            toks = []
            for t in self.kiwi.tokenize(text):
                tag = t.tag.split("-")[0]
                if tag in CONTENT_TAGS:
                    toks.append(t.form.lower())
                elif tag in PREDICATE_TAGS:
                    toks.append(t.form + "다")
                elif (tag, t.form) in KEEP_FUNCTIONAL:
                    toks.append(t.form)
        return [t for t in toks if t not in self.stopwords and not (len(t) == 1 and t.isascii())]
