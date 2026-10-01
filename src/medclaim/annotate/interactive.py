"""터미널 대화형 코딩 도구.

엑셀/구글시트에서 CSV 를 직접 채워도 되지만, 이 도구는
  - 논문 결론·핵심 조건과 기사 제목·리드·본문을 한 화면에 보여주고
  - 입력값을 코드북 허용값으로 검증하고
  - 한 기사마다 바로 저장해서 중간에 꺼도 이어서 할 수 있다.
블라인드 코더(자동 제안 열이 없는 시트)는 Enter 로 제안을 수락할 수 없다.
"""
from __future__ import annotations

import textwrap

import pandas as pd

from ..coding.precode import ALLOWED, ARTICLE_HUMAN_COLS
from ..coding.strength import LEVELS_KO
from ..paths import Workspace
from ..utils import read_csv, read_jsonl, write_csv

PROMPTS = {
    "title_strength": ("제목 인과 강도 0~6", "auto_title_strength"),
    "lead_strength": ("리드(첫 두 문장) 인과 강도 0~6", "auto_lead_strength"),
    "direction": ("방향 inc/dec/null/unclear", "auto_direction"),
    "advice": ("행동 권고 0 없음/1 암묵/2 의료진 대상/3 독자 대상", "auto_advice"),
    "caveat_in_body": ("본문에 한계·인과 유보·주의 문장이 있나 Y/N", "auto_caveat"),
    "title_omission_misleading": ("제목의 조건 생략이 오해를 부르나 Y/N/NA", None),
    "animal_generalization": ("(동물연구만) 0 동물 명시/1 인간 암시/2 인간 명시/NA", None),
    "source_error": ("출처(저널명 등) 표기가 틀렸나 Y/N", None),
    "coder_notes": ("메모(자유)", None),
}


def _wrap(text: str, width: int = 100, limit: int | None = None) -> str:
    text = (text or "").replace("\n", " ")
    if limit and len(text) > limit:
        text = text[:limit] + " …"
    return textwrap.fill(text, width=width, subsequent_indent="    ")


def _ask(field: str, suggestion: str, blind: bool) -> str | None:
    """None = 저장 후 종료, '__skip__' = 이 기사 건너뛰기."""
    label, _ = PROMPTS[field]
    allowed = ALLOWED.get(field)
    hint = f" [Enter=제안 {suggestion}]" if (suggestion and not blind) else ""
    while True:
        ans = input(f"  {label}{hint} > ").strip()
        if ans.lower() == "q":
            return None
        if ans.lower() == "s":
            return "__skip__"
        if ans == "" and suggestion and not blind:
            ans = suggestion
        if allowed is None:
            return ans
        for value in allowed:
            if ans.lower() == value.lower():
                return value
        print(f"    허용값: {', '.join(allowed)}  (s=이 기사 건너뛰기, q=저장 후 종료)")


def run_interactive(ws: Workspace, coder: str, show_body_chars: int = 1500) -> None:
    path = ws.coding_csv(coder)
    sheet = read_csv(path)
    blind = "auto_title_strength" not in sheet.columns
    bodies = {a["article_id"]: a.get("body", "") for a in read_jsonl(ws.articles_jsonl)}
    todo = sheet.index[sheet["title_strength"].astype(str).str.strip() == ""].tolist()
    print(f"\n[{coder}] 남은 기사 {len(todo)} / {len(sheet)}  (블라인드={blind})")
    print("강도 척도: " + ", ".join(f"{k}={v}" for k, v in LEVELS_KO.items()))
    print("q = 저장 후 종료, s = 이 기사 건너뛰기\n")
    for n, idx in enumerate(todo, 1):
        row = sheet.loc[idx]
        print("=" * 100)
        print(f"({n}/{len(todo)}) {row['article_id']}  |  {row['bundle_id']}  |  {row['outlet']} "
              f"({row['outlet_type']})  {row['published']}")
        print(f"[논문 결론] {_wrap(row.get('paper_finding', ''), limit=400)}")
        print(f"[논문 핵심 조건] {row.get('paper_key_conditions', '') or '(paper_coding.csv 에 아직 없음)'}")
        print(f"[제목] {row['title']}")
        print(f"[리드] {_wrap(row['lead'])}")
        if show_body_chars:
            print(f"[본문] {_wrap(bodies.get(row['article_id'], ''), limit=show_body_chars)}")
        if not blind:
            print(f"  (자동) 제목 {row.get('auto_title_strength')} [{row.get('auto_title_cues')}] / "
                  f"리드 {row.get('auto_lead_strength')} [{row.get('auto_lead_cues')}] / "
                  f"조건(제목) {row.get('auto_title_conditions')} / 조건(본문) {row.get('auto_body_conditions')}")
            if row.get("auto_caveat_sentences"):
                print(f"  (자동) 주의 문장: {_wrap(row['auto_caveat_sentences'], limit=300)}")
        answers: dict[str, str] = {}
        for field in ARTICLE_HUMAN_COLS:
            auto_col = PROMPTS[field][1]
            suggestion = str(row.get(auto_col, "")) if auto_col and not blind else ""
            ans = _ask(field, suggestion, blind)
            if ans is None:
                write_csv(sheet, path)
                print(f"저장했습니다 → {path}")
                return
            if ans == "__skip__":
                answers = {}
                break
            answers[field] = ans
        for k, v in answers.items():
            sheet.at[idx, k] = v
        write_csv(sheet, path)
    print(f"모든 기사를 코딩했습니다 → {path}")


def coding_progress(ws: Workspace, coder: str) -> tuple[int, int]:
    path = ws.coding_csv(coder)
    if not path.exists():
        return 0, 0
    sheet = read_csv(path)
    done = int((sheet["title_strength"].astype(str).str.strip() != "").sum())
    return done, len(sheet)


def coding_frame(ws: Workspace, coder: str) -> pd.DataFrame:
    return read_csv(ws.coding_csv(coder))
