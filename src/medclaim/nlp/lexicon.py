"""YAML 사전 → 정규식 컴파일·매칭."""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from ..utils import load_yaml


def compile_entry(entry: str) -> re.Pattern:
    entry = str(entry)
    if entry.startswith("re:"):
        return re.compile(entry[3:], re.IGNORECASE)
    return re.compile(re.escape(entry), re.IGNORECASE)


@dataclass(frozen=True)
class Cue:
    category: str
    text: str
    start: int
    end: int


class Lexicon:
    def __init__(self, data: dict) -> None:
        self.data = data
        self._compiled: dict[tuple, list[re.Pattern]] = {}

    @classmethod
    def load(cls, path: Path) -> "Lexicon":
        return cls(load_yaml(path))

    def section(self, *keys) -> object:
        node: object = self.data
        for k in keys:
            if not isinstance(node, dict):
                raise KeyError(keys)
            if k not in node and isinstance(k, str) and k.isdigit() and int(k) in node:
                k = int(k)
            node = node[k]
        return node

    def has(self, *keys) -> bool:
        try:
            self.section(*keys)
            return True
        except KeyError:
            return False

    def patterns(self, *keys) -> list[re.Pattern]:
        if keys not in self._compiled:
            entries = self.section(*keys)
            if isinstance(entries, str):
                entries = [entries]
            self._compiled[keys] = [compile_entry(e) for e in (entries or [])]
        return self._compiled[keys]

    def regex(self, *keys) -> re.Pattern:
        """단일 정규식 문자열 항목(예: negation_after)."""
        if keys not in self._compiled:
            self._compiled[keys] = [re.compile(str(self.section(*keys)), re.IGNORECASE)]
        return self._compiled[keys][0]

    def find(self, text: str, *keys, category: str | None = None) -> list[Cue]:
        cat = category or ".".join(str(k) for k in keys)
        found: list[Cue] = []
        for pat in self.patterns(*keys):
            for m in pat.finditer(text or ""):
                if m.group().strip():
                    found.append(Cue(cat, m.group(), m.start(), m.end()))
        return _dedupe_overlaps(found)

    def any(self, text: str, *keys) -> bool:
        return any(p.search(text or "") for p in self.patterns(*keys))

    def subkeys(self, *keys) -> list:
        node = self.section(*keys)
        return list(node.keys()) if isinstance(node, dict) else []


def _dedupe_overlaps(cues: list[Cue]) -> list[Cue]:
    """같은 위치를 여러 패턴이 잡으면 가장 긴 것 하나만 남긴다."""
    cues = sorted(cues, key=lambda c: (c.start, -(c.end - c.start)))
    out: list[Cue] = []
    last_end = -1
    for c in cues:
        if c.start >= last_end:
            out.append(c)
            last_end = c.end
    return out


def mask_terms(text: str, terms: Iterable[str], mask: str) -> str:
    """약물명 등을 하나의 기호로 바꾼다. '위산분비억제제'의 '억제'가 방향 단서로 잡히는 일 등을 막는다."""
    out = text or ""
    for term in sorted({t for t in terms if t}, key=len, reverse=True):
        out = re.sub(re.escape(term), mask, out, flags=re.IGNORECASE)
    return out


@lru_cache(maxsize=8)
def load_lexicon_cached(path_str: str) -> Lexicon:
    return Lexicon.load(Path(path_str))
