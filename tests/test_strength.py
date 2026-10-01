"""인과 강도 사전 코딩 규칙 테스트.

제목 예시는 노션 사례 정리(사례 1~5)의 실제 기사 제목 조각이다. 기대값은 규칙이 '의도대로'
동작하는지 보는 것이지, 사람 코딩의 정답이라는 뜻은 아니다(최종 판정은 codebook 기준 사람).
"""
import pytest

TITLE_CASES = [
    ("PPI제제 장기 복용시 치매 위험…1.3배 발병률 상승", 3, "increase", False),
    ('트럼프가 틀렸다… "임신 중 타이레놀 복용, 아이 자폐와 연관성 없어"', 1, "null", False),
    ("타이레놀 괴담?…", 1, "null", True),
    ("연관성 있어도 원인 아냐", 2, "", False),
    ("'시력 손실' 부작용…안전성 검토 필요", 3, "", False),
    ("실명 유발 안질환 위험 높여", 6, "increase", False),
    ("자칫하단 '실명위험'", 3, "", False),
    ('"심각한 정신 질환 유발?"', 6, "increase", True),
    ("우울증 논란 탈모약…'인과관계 회의적'", 1, "null", False),
]


@pytest.mark.parametrize("title,level,direction,question", TITLE_CASES)
def test_screenshot_titles(ko, title, level, direction, question):
    r = ko.code_unit([title], require_target=False)
    assert r.level == level, (title, r.cues)
    assert r.direction == direction
    assert r.question == question


@pytest.mark.parametrize("sentence,level", [
    ("PPI를 장기 복용하면 치매를 유발할 가능성이 있다", 4),       # 조건부 인과
    ("위고비 복용 시 실명 위험이 높아질 수 있다", 5),              # can
    ("위산분비억제제 장기 복용, 치매 위험 키운다", 6),              # 단정적 인과
    ("PPI 복용자의 치매 위험이 1.33배 높은 것으로 나타났다", 2),   # 통계적 서술 = 상관
    ("타이레놀이 자폐 위험을 높이지 않는다", 1),                   # 부정
    ("타이레놀은 자폐를 유발하지 않는다", 1),
    ("세마글루타이드 복용자의 자살 생각 위험이 49~73% 낮았다", 2),
])
def test_sentence_levels(ko, sentence, level):
    r = ko.code_sentence(sentence, drug_terms=["PPI", "위고비", "위산분비억제제", "타이레놀", "세마글루타이드"],
                         outcome_terms=["치매", "실명", "자폐", "자살"])
    assert r.level == level, (sentence, r.cues)


def test_drug_name_is_masked_before_direction(ko):
    # '위산분비억제제'의 '억제'가 감소 방향 단서로 잡히면 안 된다
    r = ko.code_sentence("위산분비억제제 장기 복용, 치매 위험 키운다", drug_terms=["위산분비억제제"],
                         outcome_terms=["치매"])
    assert r.direction == "increase"


def test_sentence_without_target_is_not_coded(ko):
    r = ko.code_sentence("치매 환자는 고령화로 늘고 있다", drug_terms=["PPI"], outcome_terms=[])
    assert r.level == 0


def test_unit_takes_strongest(ko):
    r = ko.code_unit(["PPI와 치매의 연관성이 관찰됐다.", "PPI가 치매를 유발한다."],
                     drug_terms=["PPI"], outcome_terms=["치매"])
    assert r.level == 6


@pytest.mark.parametrize("text,main_level", [
    ("Acetaminophen use during pregnancy was not associated with children's risk of autism, ADHD, or "
     "intellectual disability in sibling control analysis. This suggests that associations observed in "
     "other models may have been attributable to familial confounding.", 1),
    ("Cumulative PPI use for more than 4.4 years was associated with a higher risk of dementia.", 2),
    ("Current evidence shows that finasteride use can cause depression and suicidality.", 5),
    ("These findings suggest semaglutide may increase the risk of NAION.", 4),
    ("Our findings do not support higher risks of suicidal ideation with semaglutide.", 1),
])
def test_english_conclusions(en, text, main_level):
    main, strongest = en.code_conclusion(text)
    assert main.level == main_level
    assert strongest.level >= main.level
