from medclaim.bundles import Bundle
from medclaim.coding.features import advice_level, condition_hits, extract_features
from medclaim.coding.relevance import score_candidate
from medclaim.nlp.segment import segment_article, split_sentences


def _bundle():
    return Bundle(bundle_id="B01", label="PPI-치매", drug_terms=["PPI", "양성자펌프억제제"],
                  outcome_terms=["치매"], paper={"journal": "Neurology", "first_author": "Northuis",
                                                 "published": "2023-08-09"},
                  queries=["PPI 치매"], date_from=None, date_to=None)


def test_conditions(lex_ko):
    hits = condition_hits(lex_ko, "45세 이상 성인 5,712명을 누적 4.4년 이상 추적한 코호트 연구에서 HR 1.33")
    assert hits["population"] and hits["sample_size"] and hits["duration"]
    assert hits["design"] and hits["effect_size"]
    assert not hits["animal"]


def test_year_is_not_duration(lex_ko):
    hits = condition_hits(lex_ko, "2023년 8월 10일 발표된 연구")
    assert not hits["duration"]


def test_animal(lex_ko):
    assert condition_hits(lex_ko, "생쥐 실험에서 수명이 줄었다")["animal"]


def test_limitation(lex_ko):
    hits = condition_hits(lex_ko, "연구팀은 \"인과관계는 더 많은 연구가 필요하다\"며 PPI를 끊어야 한다는 의미는 아니라고 했다")
    assert hits["limitation"]


def test_advice_levels(lex_ko):
    assert advice_level(lex_ko, "임의로 중단하지 말고 의사와 상담하세요")[0] == 3
    assert advice_level(lex_ko, "처방 시 주의가 필요하다")[0] == 2
    assert advice_level(lex_ko, "장기 복용에 주의가 필요하다")[0] == 1
    assert advice_level(lex_ko, "연구 결과가 발표됐다")[0] == 0


def test_journal_matcher_longest_first(journals):
    found, non = journals.find("네이처 메디슨에 실린 논문")
    assert found == ["Nature Medicine"]
    found, non = journals.find("국제학술지 Psychiatrist.com에 게재")
    assert non == ["Psychiatrist.com"] and found == []


def test_segment_skips_byline_and_caption():
    body = "(서울=연합뉴스) 홍길동 기자 = PPI를 오래 먹으면 치매 위험이 높았다.\n사진=게티이미지\n두 번째 문장이다.\n세 번째 문장이다."
    seg = segment_article("제목", body)
    assert seg["lead_sentences"][0].startswith("PPI를")
    assert all("게티이미지" not in s for s in seg["sentences"])


def test_split_sentences():
    assert len(split_sentences("첫 문장이다. 두 번째 문장이다.")) == 2


def test_extract_features(lex_ko, ko, journals):
    article = {"article_id": "a1", "bundle_id": "B01", "outlet": "메디칼타임즈", "outlet_type": "medical_trade",
               "title": "PPI제제 장기 복용시 치매 위험…1.3배 발병률 상승",
               "body": "미국 연구팀이 45세 이상 5,712명을 분석한 결과 PPI를 4.4년 이상 복용한 사람의 치매 위험이 "
                       "1.33배 높았다. 연구는 뉴롤로지에 게재됐다. 연구팀은 \"인과관계는 더 많은 연구가 필요하다\"고 "
                       "밝혔다. PPI를 끊어야 한다는 의미는 아니다."}
    f = extract_features(article, _bundle(), lex_ko, ko, journals)
    assert f["auto_title_strength"] == 3
    assert f["auto_lead_strength"] == 2
    assert f["auto_cond_title_duration"] == 1          # '장기'
    assert f["auto_cond_title_population"] == 0
    assert f["auto_cond_lead_population"] == 1
    assert f["auto_caveat"] == 1
    assert f["journals_mentioned"] == "Neurology" and f["auto_journal_match"] == 1


def test_relevance(journals):
    b = _bundle()
    score, auto, reasons = score_candidate(b, "PPI 장기 복용, 치매 위험", "뉴롤로지 게재 연구팀", None, journals)
    assert auto == "Y" and score >= 7
    _, auto2, _ = score_candidate(b, "PPI 부작용 총정리", "속쓰림", None, journals)
    assert auto2 == "N"
