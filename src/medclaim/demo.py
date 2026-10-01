"""합성(가짜) 데이터로 전체 파이프라인을 오프라인에서 끝까지 돌려 보는 데모.

- 약물·결과·매체·기사·댓글이 모두 지어낸 것이다(실제 약물/언론사와 무관). 결과 수치는 의미가 없다.
- 네트워크 수집 단계(논문/네이버/스크래핑/댓글)는 건너뛰고, 그 결과물 파일을 직접 만든다.
- '사람 코더' 2명도 자동 코드에 잡음을 섞어 흉내 낸다(일치도·불일치 조정 단계를 보여주기 위함).
"""
from __future__ import annotations

import random
import shutil
from pathlib import Path

import pandas as pd
import yaml

from .annotate.agreement import finalize, run_agreement
from .bundles import load_bundles
from .coding.precode import precode
from .paths import DEFAULT_CONFIG_DIR, Workspace
from .utils import log, read_csv, stable_id, write_csv, write_jsonl

FAKE_OUTLETS = [("예시통신", "wire"), ("가상일보", "national_daily"), ("모의경제", "economic_daily"),
                ("샘플메디칼", "medical_trade"), ("데모헬스", "health_consumer"), ("테스트인사이드", "online")]
# 매체 유형별 '제목을 세게 뽑는 경향' (합성 데이터 생성용 가정일 뿐)
TITLE_BOOST = {"wire": 0, "national_daily": 1, "economic_daily": 1, "medical_trade": 0,
               "health_consumer": 2, "online": 3}

BUNDLES = [
    dict(id="D01_fictaprazole_memory", drug="가상프라졸", drug_en="Fictaprazole", outcome="기억력 저하",
         outcome_en="memory decline", design="observational", strength=2, direction="inc",
         key="duration;population", pop="60세 이상 성인", n="3만2천 명", dur="5년 이상 누적 복용",
         design_ko="코호트 연구", effect="1.3"),
    dict(id="D02_examplumab_vision", drug="예시맙", drug_en="Examplumab", outcome="시력 이상",
         outcome_en="visual disturbance", design="observational", strength=2, direction="inc",
         key="design;population", pop="당뇨 환자", n="1,200명", dur="2년간", design_ko="단일기관 후향 연구",
         effect="2.1"),
    dict(id="D03_samplin_sleep", drug="샘플린", drug_en="Samplin", outcome="수면장애", outcome_en="insomnia",
         design="rct", strength=6, direction="inc", key="dose", pop="성인", n="800명", dur="12주간",
         design_ko="무작위 임상시험", effect="1.8", dose="고용량(20mg)"),
    dict(id="D04_testophen_mood", drug="테스트펜", drug_en="Testophen", outcome="우울감", outcome_en="low mood",
         design="case_report", strength=3, direction="inc", key="design;sample_size", pop="환자", n="12명",
         dur="", design_ko="사례 보고", effect=""),
    dict(id="D05_mockstatin_muscle", drug="모의스타틴", drug_en="Mockstatin", outcome="근육통",
         outcome_en="myalgia", design="observational", strength=1, direction="null", key="population",
         pop="65세 이상 노인", n="9만 명", dur="3년간", design_ko="형제 비교 코호트", effect=""),
    dict(id="D06_demotin_lifespan", drug="데모틴", drug_en="Demotin", outcome="수명 단축",
         outcome_en="shortened lifespan", design="animal", strength=6, direction="inc", key="animal",
         pop="생쥐", n="120마리", dur="18개월", design_ko="동물 실험", effect=""),
    dict(id="D07_hypothesol_rash", drug="가정솔", drug_en="Hypothesol", outcome="피부 발진",
         outcome_en="skin rash", design="observational", strength=2, direction="dec", key="population;duration",
         pop="청소년", n="4만 명", dur="1년 이상", design_ko="건강보험 빅데이터 분석", effect="0.7"),
    dict(id="D08_specimenic_headache", drug="견본산", drug_en="Specimenic acid", outcome="두통",
         outcome_en="headache", design="meta", strength=2, direction="inc", key="design", pop="성인",
         n="15만 명", dur="", design_ko="메타분석", effect="1.2"),
]

TITLE_TEMPLATES = {
    1: ["{drug}, {outcome_gwa} 연관성 없어", "\"{drug} 먹어도 {outcome} 걱정 없다\"…{outcome} 괴담 깨져"],
    2: ["{drug} 복용자, {outcome} 위험과 연관성 관찰", "{drug} 장기 복용, {outcome_gwa} 관련 있어"],
    3: ["{drug} 복용시 {outcome} 위험 1.4배", "{drug_gwa} {outcome}, 연결고리 찾았다"],
    4: ["{drug} 복용, {outcome} 위험 가능성", "{drug} {outcome} 우려…연구 결과 나와"],
    5: ["{drug}, {outcome} 유발할 수 있다", "{drug} 먹으면 {outcome} 위험 높아질 수 있어"],
    6: ["{drug}, {outcome} 위험 높인다", "{drug} 먹으면 {outcome} 생긴다", "충격! {outcome} 부르는 {drug}"],
}
DEC_TITLES = {6: ["{drug}, {outcome} 위험 낮춘다"], 5: ["{drug}, {outcome} 줄일 수 있다"]}
COMMENTS = {
    "anxiety": ["{drug} 먹고 있는데 무섭네요", "걱정되네 끊어야 하나", "우리 엄마도 먹는데 어떡하죠"],
    "distrust": ["기레기 또 과장하네", "제약사 광고 아님?", "이런 기사 못 믿겠다"],
    "reassurance": ["다행이네요", "괜찮다니 안심", "상관없다니 다행"],
    "neutral": ["기사 잘 봤습니다", "의사랑 상담해봐야겠네", "논문 원문 링크는 없나요", "다른 연구도 궁금하네"],
}


def _josa(word: str, with_batchim: str, without: str) -> str:
    """'가상프라졸'+과 / '예시맙'+과 / '샘플린'+과 / '데모틴'+과 … 받침 유무로 조사를 고른다."""
    last = word[-1]
    if "가" <= last <= "힣":
        return word + (with_batchim if (ord(last) - 0xAC00) % 28 else without)
    return word + without


def _fill(template: str, b: dict) -> str:
    return template.format(drug=b["drug"], outcome=b["outcome"],
                           drug_gwa=_josa(b["drug"], "과", "와"), outcome_gwa=_josa(b["outcome"], "과", "와"),
                           drug_eul=_josa(b["drug"], "을", "를"), drug_eun=_josa(b["drug"], "은", "는"),
                           outcome_ga=_josa(b["outcome"], "이", "가"), outcome_eun=_josa(b["outcome"], "은", "는"))


def _pick_level(true_level: int, outlet_type: str, rng: random.Random) -> int:
    boost = TITLE_BOOST[outlet_type] + rng.choice([-1, 0, 0, 1])
    level = true_level + max(0, boost)
    if true_level == 1 and rng.random() < 0.6:
        level = 1
    return max(1, min(6, level))


def _article(b: dict, outlet: str, otype: str, rng: random.Random) -> tuple[dict, int]:
    level = _pick_level(b["strength"], otype, rng)
    pool = DEC_TITLES.get(level, TITLE_TEMPLATES[level]) if b["direction"] == "dec" else TITLE_TEMPLATES[level]
    title = _fill(rng.choice(pool), b)
    if otype in ("online", "health_consumer") and rng.random() < 0.3 and not title.endswith("?"):
        title += "?"
    keep_cond = rng.random() < (0.85 if otype in ("medical_trade", "wire") else 0.5)
    dur = f" {b['dur']}" if b["dur"] else ""
    lead1 = (f"[합성 예시] {b['design_ko']}에서 {b['pop']} {_josa(b['n'], '을', '를')}{dur} 분석한 결과, "
             if keep_cond else "[합성 예시] 최근 연구에서 ")
    if b["direction"] == "null":
        lead1 += _fill("{drug} 복용과 {outcome} 사이에 연관성이 없는 것으로 나타났다.", b)
    elif b["direction"] == "dec":
        lead1 += f"{b['drug']} 복용군의 {b['outcome']} 위험이 낮은 것으로 나타났다."
    else:
        eff = f" {b['effect']}배" if b["effect"] else ""
        lead1 += f"{b['drug']} 복용군의 {b['outcome']} 위험이{eff} 높은 것으로 나타났다."
    if b.get("dose") and keep_cond:
        lead1 += f" 효과는 {b['dose']}에서 뚜렷했다."
    lead2 = "연구 결과는 국제학술지 '가상의학저널'에 게재됐다."
    body = [lead1, lead2,
            _fill("연구팀은 {drug_eul} 복용한 집단과 복용하지 않은 집단을 비교했다.", b),
            _fill("{outcome_eun} 고령층에서 흔히 나타나는 증상으로 알려져 있다.", b)]
    if rng.random() < (0.8 if otype in ("medical_trade", "wire") else 0.4):
        body.append("연구팀은 \"인과관계를 단정할 수 없으며 추가 연구가 필요하다\"고 밝혔다.")
    if rng.random() < 0.5:
        body.append("전문가들은 임의로 복용을 중단하지 말고 의사와 상담할 것을 권고했다.")
    body.append("[이 기사는 파이프라인 시연용으로 생성된 가짜 기사입니다]")
    return {"title": title, "body": "\n".join(body)}, level


def _comments(b: dict, level: int, otype: str, rng: random.Random) -> list[str]:
    if otype == "medical_trade":
        return []
    n = rng.randint(4, 25)
    p_anx = 0.15 + 0.06 * level
    p_dis = 0.10 + 0.03 * level
    p_rea = 0.25 if b["direction"] == "null" else 0.08
    out = []
    for _ in range(n):
        r = rng.random()
        kind = "anxiety" if r < p_anx else "distrust" if r < p_anx + p_dis else \
            "reassurance" if r < p_anx + p_dis + p_rea else "neutral"
        out.append(rng.choice(COMMENTS[kind]).format(drug=b["drug"]))
    return out


def _paper_conclusion(b: dict) -> str:
    d, o = b["drug_en"], b["outcome_en"]
    return {
        ("observational", "inc"): f"Cumulative use of {d} was associated with a higher risk of {o}.",
        ("observational", "dec"): f"Use of {d} was associated with a lower risk of {o}.",
        ("observational", "null"): f"{d} use was not associated with {o} in sibling-controlled analyses.",
        ("rct", "inc"): f"High-dose {d} increased the risk of {o} compared with placebo.",
        ("case_report", "inc"): f"In reported cases, {o} was linked to {d}.",
        ("animal", "inc"): f"{d} caused {o} in mice.",
        ("meta", "inc"): f"{d} was associated with a modestly higher risk of {o}.",
    }[(b["design"], b["direction"])]


def build_demo(out_dir: Path, seed: int = 7) -> Workspace:
    rng = random.Random(seed)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "config").mkdir(parents=True)
    (out_dir / "SYNTHETIC").write_text("이 작업 공간의 데이터는 모두 합성(가짜) 데이터입니다.\n", encoding="utf-8")
    for name in ("lexicon_ko.yaml", "lexicon_en.yaml", "journals.yaml", "outlets.yaml", "discovery.yaml"):
        shutil.copy(DEFAULT_CONFIG_DIR / name, out_dir / "config" / name)
    # 합성 매체·저널을 설정에 추가
    outlets = yaml.safe_load((out_dir / "config" / "outlets.yaml").read_text(encoding="utf-8"))
    outlets["outlets"] += [{"name": n, "type": t, "domains": [f"{stable_id(n, length=6)}.example"]}
                           for n, t in FAKE_OUTLETS]
    (out_dir / "config" / "outlets.yaml").write_text(yaml.safe_dump(outlets, allow_unicode=True), encoding="utf-8")
    journals = yaml.safe_load((out_dir / "config" / "journals.yaml").read_text(encoding="utf-8"))
    journals["journals"]["Journal of Imaginary Medicine"] = ["Journal of Imaginary Medicine", "가상의학저널"]
    (out_dir / "config" / "journals.yaml").write_text(yaml.safe_dump(journals, allow_unicode=True), encoding="utf-8")

    bundles_yaml = {"bundles": [{
        "bundle_id": b["id"], "label": f"[합성] {b['drug']} ↔ {b['outcome']}",
        "drug_terms": [b["drug"]], "outcome_terms": [b["outcome"]],
        "paper": {"doi": f"10.0000/demo.{b['id'][:3].lower()}", "title": f"[SYNTHETIC] {b['drug_en']} and {b['outcome_en']}",
                  "journal": "Journal of Imaginary Medicine", "first_author": "Example",
                  "published": "2026-01-15"},
        "search": {"queries": [f"{b['drug']} {b['outcome']}"], "date_from": "2026-01-15", "date_to": "2026-03-31"},
        "known_articles": [], "notes": "합성 데이터"} for b in BUNDLES]}
    (out_dir / "config" / "bundles.yaml").write_text(yaml.safe_dump(bundles_yaml, allow_unicode=True), encoding="utf-8")
    ws = Workspace(root=out_dir, config_dir=out_dir / "config")
    ws.ensure_dirs()

    # 1) fetch-papers 결과 흉내
    from .coding.strength import EnglishClaimCoder
    from .nlp.lexicon import Lexicon
    coder = EnglishClaimCoder(Lexicon.load(ws.config_file("lexicon_en.yaml")))
    papers = []
    for b in BUNDLES:
        concl = _paper_conclusion(b)
        main, _ = coder.code_conclusion(concl)
        papers.append({"bundle_id": b["id"], "doi": f"10.0000/demo.{b['id'][:3].lower()}",
                       "title": f"[SYNTHETIC] {b['drug_en']} and {b['outcome_en']}",
                       "journal": "Journal of Imaginary Medicine", "published": "2026-01-15",
                       "design_auto": b["design"], "conclusion_en": concl, "conclusion_source": "synthetic",
                       "strength_auto": main.level, "strength_auto_cues": main.cue_str(),
                       "direction_auto": main.direction, "fetch_status": "synthetic"})
    write_csv(pd.DataFrame(papers), ws.papers_csv)

    # 2) search-news → 사람 include 판정 → scrape 결과 흉내
    candidates, articles, comments = [], [], []
    for b in BUNDLES:
        n_articles = rng.randint(3, 6)       # 6건 묶음은 precode 의 '묶음당 최대 5건' 선택을 보여준다
        outlets = rng.sample(FAKE_OUTLETS, k=min(n_articles, len(FAKE_OUTLETS)))
        for outlet, otype in outlets:
            art, level = _article(b, outlet, otype, rng)
            aid = stable_id(b["id"], outlet)
            url = f"https://{stable_id(outlet, length=6)}.example/news/{aid}"
            candidates.append({"candidate_id": aid, "bundle_id": b["id"], "source": "synthetic", "query": "",
                               "pub_date": "2026-01-16", "outlet": outlet, "outlet_type": otype,
                               "title": art["title"], "description": "", "url": url, "naver_url": "",
                               "relevance_score": 9, "auto_include": "Y", "relevance_reasons": "synthetic",
                               "include": "Y", "link_type": "main", "paper_match_evidence": "합성",
                               "candidate_notes": ""})
            articles.append({"article_id": aid, "bundle_id": b["id"], "url": url, "naver_url": "",
                             "fetched_url": url, "outlet": outlet, "outlet_type": otype, "title": art["title"],
                             "subtitle": "", "body": art["body"], "published": "2026-01-16",
                             "extractor": "synthetic", "status": "ok", "scraped_at": ""})
            for i, text in enumerate(_comments(b, level, otype, rng)):
                comments.append({"comment_id": stable_id(aid, i, text), "article_id": aid, "bundle_id": b["id"],
                                 "text": text, "reg_time": "", "likes": rng.randint(0, 50), "dislikes": 0,
                                 "sort": "synthetic"})
    write_csv(pd.DataFrame(candidates), ws.candidates_csv)
    write_jsonl(articles, ws.articles_jsonl)
    write_jsonl(comments, ws.comments_jsonl)

    # 3) precode (coder1 = 제안값 보이는 시트, coder2 = 블라인드)
    bundles = load_bundles(ws.bundles_path)
    precode(ws, bundles, coders=["coder1", "coder2"], blind_coders=["coder2"], max_per_bundle=5)

    # 4) 사람 코딩 흉내: 논문 시트
    truth = {b["id"]: b for b in BUNDLES}
    ps = read_csv(ws.paper_coding_csv)
    for i, r in ps.iterrows():
        b = truth[r["bundle_id"]]
        ps.at[i, "design"] = b["design"]
        ps.at[i, "paper_strength"] = str(b["strength"])
        ps.at[i, "paper_direction"] = b["direction"]
        ps.at[i, "key_conditions"] = b["key"]
        ps.at[i, "main_finding_ko"] = f"[합성] {b['drug']}–{b['outcome']} ({b['design_ko']})"
        ps.at[i, "checked_by"] = "demo-bot"
    write_csv(ps, ws.paper_coding_csv)
    # 5) 사람 코딩 흉내: 기사 시트 (자동값 + 잡음, coder2 는 40%만 이중 코딩)
    feats = read_csv(ws.features_csv)
    auto = feats.set_index("article_id")
    dir_map = {"increase": "inc", "decrease": "dec", "null": "null", "mixed": "unclear", "": "unclear"}

    def noisy(v: str, p: float) -> str:
        x = int(v)
        if rng.random() < p:
            x = max(0, min(6, x + rng.choice([-1, 1])))
        return str(x)

    for coder_name, p_noise, share in (("coder1", 0.15, 1.0), ("coder2", 0.15, 0.4)):
        sheet = read_csv(ws.coding_csv(coder_name))
        for i, r in sheet.iterrows():
            if rng.random() > share:
                continue
            a = auto.loc[r["article_id"]]
            sheet.at[i, "title_strength"] = noisy(a["auto_title_strength"], p_noise)
            sheet.at[i, "lead_strength"] = noisy(a["auto_lead_strength"], p_noise)
            sheet.at[i, "direction"] = dir_map.get(a["auto_direction"], "unclear")
            sheet.at[i, "advice"] = a["auto_advice"]
            sheet.at[i, "caveat_in_body"] = "Y" if a["auto_caveat"] == "1" else "N"
            sheet.at[i, "title_omission_misleading"] = "Y" if int(a["auto_title_strength"]) >= 5 and \
                rng.random() < 0.6 else "N"
            sheet.at[i, "animal_generalization"] = "NA"
            sheet.at[i, "source_error"] = "N"
        write_csv(sheet, ws.coding_csv(coder_name))
    # 6) 일치도 → 불일치 조정(흉내: coder1 값 채택) → 최종
    run_agreement(ws, ["coder1", "coder2"])
    dis = read_csv(ws.disagreements_csv)
    if len(dis):
        dis["resolved"] = dis["coder1"]
        dis["resolution_notes"] = "demo: coder1 값 채택"
        write_csv(dis, ws.disagreements_csv)
    finalize(ws, ["coder1", "coder2"])
    log.info("합성 데모 작업 공간 → %s", out_dir)
    return ws
