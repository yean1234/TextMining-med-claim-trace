"""문장 분리와 기사 구간(제목 / 리드 / 나머지 본문) 나누기.

Sumner et al.(2014)은 기사의 인과 주장을 '제목 + 첫 두 문장'에서 평가했다.
여기서도 리드(lead) = 본문 첫 두 문장(바이라인·사진 설명 제외)으로 둔다.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

NOISE_PATTERNS = [
    re.compile(r"^\(?[^()]{0,25}=\s*[^()]{0,15}\)?\s*$"),          # (서울=연합뉴스)
    re.compile(r"^\[?[^\]]{0,30}기자\]?$"),                          # [홍길동 기자]
    re.compile(r"(사진|그래픽|이미지|자료)\s*=\s*\S+"),               # 사진=게티이미지
    re.compile(r"게티이미지|이미지투데이|클립아트코리아|셔터스톡"),
    re.compile(r"무단\s*전재|재배포\s*금지|저작권자|Copyright|ⓒ|©"),
    re.compile(r"^[▶☞■◆●▲△※]"),
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),                          # 이메일
]
LEADING_BYLINE = re.compile(
    r"^\s*(\([^)]{1,30}=[^)]{1,20}\)\s*)?(\[[^\]]{1,40}\]\s*)?([가-힣]{2,4}\s*(기자|특파원|객원기자)\s*=\s*)?"
)
QUOTE_CHARS = "\"'“”‘’「」『』"


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("​", "").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


@lru_cache(maxsize=1)
def get_kiwi():
    try:
        from kiwipiepy import Kiwi
    except ImportError:  # pragma: no cover - kiwipiepy 미설치 환경
        return None
    return Kiwi()


def split_sentences(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    kiwi = get_kiwi()
    out: list[str] = []
    for para in [p.strip() for p in text.split("\n") if p.strip()]:
        if kiwi is not None:
            out.extend(s.text.strip() for s in kiwi.split_into_sents(para))
        else:
            out.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+(?=\S)", para))
    return [s for s in out if s]


def is_noise_sentence(sentence: str) -> bool:
    s = sentence.strip()
    if len(s) < 8:
        return True
    if len(s) < 60 and any(p.search(s) for p in NOISE_PATTERNS):
        return True
    return False


def strip_leading_byline(sentence: str) -> str:
    return LEADING_BYLINE.sub("", sentence, count=1).strip()


def segment_article(title: str, body: str, n_lead: int = 2) -> dict:
    sentences = [strip_leading_byline(s) for s in split_sentences(body)]
    sentences = [s for s in sentences if s and not is_noise_sentence(s)]
    lead = sentences[:n_lead]
    rest = sentences[n_lead:]
    return {
        "title": normalize_text(title),
        "lead_sentences": lead,
        "rest_sentences": rest,
        "sentences": sentences,
        "lead": " ".join(lead),
        "rest": " ".join(rest),
    }


def is_quoted(text: str) -> bool:
    """제목이 따옴표 인용으로 이뤄졌는지 (주장의 출처를 취재원에게 돌리는 형식)."""
    t = normalize_text(text)
    return sum(t.count(q) for q in QUOTE_CHARS) >= 2
