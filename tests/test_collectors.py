"""수집기 파싱 테스트 (네트워크 없이 가짜 응답·가짜 HTML 사용).

실제 API/사이트 구조가 바뀌면 이 테스트는 통과해도 실제 수집은 실패할 수 있다 — LIMITATIONS.md 참고.
"""
import json
from datetime import date

from medclaim.bundles import Bundle
from medclaim.collect.comments import parse_jsonp
from medclaim.collect.naver import clean_api_text, dedupe, items_to_candidates, search_news
from medclaim.collect.outlets import OutletResolver, naver_ids
from medclaim.collect.papers import guess_design, parse_structured_abstract, pick_conclusion
from medclaim.collect.scraper import parse_generic_article, parse_naver_article
from medclaim.paths import DEFAULT_CONFIG_DIR

from .conftest import FIXTURES


class FakeResp:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeSession:
    def __init__(self, pages):
        self.pages, self.calls, self.headers = pages, [], []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append(params)
        self.headers.append(headers)
        return FakeResp(self.pages[len(self.calls) - 1])


def _bundle():
    return Bundle(bundle_id="B01", label="", drug_terms=["PPI"], outcome_terms=["치매"],
                  paper={"journal": "Neurology"}, queries=["PPI 치매"],
                  date_from=date(2023, 8, 9), date_to=date(2023, 10, 31))


def test_naver_search_pagination_hub_fields():
    from medclaim.collect.naver import NaverClient
    # API HUB 는 original_link / pub_date 처럼 필드명이 다를 수 있다 → 정규화 확인
    page1 = {"total": 150, "items": [{"title": f"<b>PPI</b> 치매 {i}", "original_link": f"https://www.yna.co.kr/{i}",
                                      "link": f"https://n.news.naver.com/mnews/article/001/{i:010d}",
                                      "description": "연구", "pub_date": "Thu, 10 Aug 2023 09:00:00 +0900"}
                                     for i in range(100)]}
    page2 = {"total": 150, "items": page1["items"][:50]}
    s = FakeSession([page1, page2])
    client = NaverClient("id", "secret", backend="hub", session=s, delay=0)
    items = search_news(client, "PPI 치매", max_results=300)
    assert len(items) == 150
    assert s.calls[1]["start"] == 101
    assert items[0]["originallink"] == "https://www.yna.co.kr/0" and items[0]["pubDate"].startswith("Thu")
    assert s.headers[0]["X-NCP-APIGW-API-KEY-ID"] == "id"
    assert client.budget.used_today == 2


def test_call_budget_stops_before_limit(tmp_path):
    import pytest

    from medclaim.collect.naver import BudgetExceeded, CallBudget, NaverClient
    budget = CallBudget(tmp_path / "usage.json", daily_limit=2)
    page = {"total": 1000, "items": [{"title": "x"}] * 100}
    s = FakeSession([page] * 5)
    client = NaverClient("id", "secret", budget=budget, session=s, delay=0)
    with pytest.raises(BudgetExceeded):
        search_news(client, "q", max_results=500)
    assert len(s.calls) == 2                                  # 상한에서 요청 전에 멈춤
    assert CallBudget(tmp_path / "usage.json", 2).used_today == 2   # 파일에 기록돼 다음 실행에도 유지


def test_items_to_candidates_and_dedupe():
    resolver = OutletResolver(DEFAULT_CONFIG_DIR / "outlets.yaml")
    from medclaim.coding.features import JournalMatcher
    from medclaim.utils import load_yaml
    journals = JournalMatcher(load_yaml(DEFAULT_CONFIG_DIR / "journals.yaml"))
    items = [{"title": "<b>PPI</b> 장기 복용, 치매 위험 &quot;1.3배&quot;", "originallink": "https://www.medicaltimes.com/a?utm_source=x",
              "link": "https://n.news.naver.com/mnews/article/001/0000000001", "description": "뉴롤로지 게재",
              "pubDate": "Thu, 10 Aug 2023 09:00:00 +0900"}] * 2
    rows = items_to_candidates(_bundle(), "PPI 치매", items, resolver, journals)
    assert rows[0]["title"] == 'PPI 장기 복용, 치매 위험 "1.3배"'
    assert rows[0]["outlet"] == "메디칼타임즈" and rows[0]["outlet_type"] == "medical_trade"
    assert rows[0]["auto_include"] == "Y"
    assert len(dedupe(rows)) == 1


def test_clean_api_text():
    assert clean_api_text("<b>위고비</b> &amp; 실명") == "위고비 & 실명"


def test_outlet_resolver_longest_domain():
    r = OutletResolver(DEFAULT_CONFIG_DIR / "outlets.yaml")
    assert r.resolve("https://health.chosun.com/x") == ("헬스조선", "health_consumer")
    assert r.resolve("https://www.chosun.com/x") == ("조선일보", "national_daily")
    assert r.resolve("https://n.news.naver.com/mnews/article/469/0000000001") == ("한국일보", "national_daily")
    assert naver_ids("https://n.news.naver.com/mnews/article/001/0014123456?sid=103") == ("001", "0014123456")


def test_parse_naver_article():
    html = (FIXTURES / "naver_article.html").read_text(encoding="utf-8")
    a = parse_naver_article(html)
    assert a["title"] == "가상약, 가상 부작용 위험 높여"   # 앞머리 [태그] 제거
    assert a["outlet_name"] == "예시일보"
    assert a["published"].startswith("2026-01-16")
    assert "1.5배 높은 것으로" in a["body"]
    assert "게티이미지" not in a["body"] and "무단 전재" not in a["body"] and "@" not in a["body"]
    assert "장기 복용자 위험" in a["subtitle"]


def test_parse_generic_article():
    html = (FIXTURES / "generic_article.html").read_text(encoding="utf-8")
    a = parse_generic_article(html, outlet="예시메디칼")
    assert a["title"] == "가상약 장기 복용시 가상 부작용 위험 1.5배"
    assert a["extractor"] == "generic:#article-view-content-div"
    assert "코호트 연구" in a["body"] and "임의로 복용을 중단" in a["body"]
    assert "이미지투데이" not in a["body"] and "무단전재" not in a["body"] and "메뉴" not in a["body"]
    assert a["published"].startswith("2026-01-16")


def test_structured_abstract_epmc():
    raw = ("<h4>Importance</h4>Some text.<h4>Design, Setting, and Participants</h4>Cohort of 5712."
           "<h4>Conclusions and Relevance</h4>Cumulative PPI use was associated with higher dementia risk.")
    sec = parse_structured_abstract(raw)
    text, src = pick_conclusion(sec)
    assert text.startswith("Cumulative PPI use") and src.startswith("section:conclusions")


def test_structured_abstract_jats():
    raw = ("<jats:sec><jats:title>Background</jats:title><jats:p>x</jats:p></jats:sec>"
           "<jats:sec><jats:title>Conclusions</jats:title><jats:p>No association was found.</jats:p></jats:sec>")
    text, _ = pick_conclusion(parse_structured_abstract(raw))
    assert text == "No association was found."


def test_unstructured_abstract_last_sentences():
    text, src = pick_conclusion(parse_structured_abstract("First. Second sentence. Third sentence here."))
    assert src == "last_two_sentences" and text == "Second sentence. Third sentence here."


def test_guess_design(lex_en):
    assert guess_design(lex_en, "", "A retrospective matched cohort study of 16,827 patients")[0] == "observational"
    assert guess_design(lex_en, "", "A target trial emulation using randomized-like design")[0] == "observational"
    assert guess_design(lex_en, "", "a randomized placebo-controlled trial")[0] == "rct"
    assert guess_design(lex_en, "", "lifespan of mice")[0] == "animal"
    assert guess_design(lex_en, "Analytical Review", "adverse event reporting", "Review")[0] == "review"


def test_parse_jsonp():
    data = {"result": {"commentList": [{"contents": "무섭네요"}]}}
    assert parse_jsonp("cb(" + json.dumps(data) + ");") == data


def test_candidates_rerun_keeps_human_include_and_bigkinds(tmp_path):
    import pandas as pd

    from medclaim.collect.bigkinds import import_bigkinds
    from medclaim.collect.naver import save_candidates
    from medclaim.paths import Workspace
    from medclaim.utils import read_csv, write_csv

    ws = Workspace(root=tmp_path)
    ws.ensure_dirs()
    row = {"candidate_id": "c1", "bundle_id": "B01", "source": "naver_api", "query": "q", "pub_date": "2023-08-10",
           "outlet": "메디칼타임즈", "outlet_type": "medical_trade", "title": "PPI 치매", "description": "",
           "url": "https://www.medicaltimes.com/a", "naver_url": "", "relevance_score": 7, "auto_include": "Y",
           "relevance_reasons": ""}
    save_candidates(ws, [row])
    df = read_csv(ws.candidates_csv)
    df.loc[df["candidate_id"] == "c1", ["include", "link_type"]] = ["Y", "main"]
    write_csv(df, ws.candidates_csv)
    save_candidates(ws, [dict(row, relevance_score=8)])          # 재실행
    df = read_csv(ws.candidates_csv).set_index("candidate_id")
    assert df.at["c1", "include"] == "Y" and df.at["c1", "link_type"] == "main"
    assert df.at["c1", "relevance_score"] == "8"

    bk = tmp_path / "bigkinds.csv"
    pd.DataFrame({"일자": ["20230811"], "언론사": ["연합뉴스"], "제목": ["PPI 장기 복용 치매 위험"],
                  "본문": ["뉴롤로지 게재"], "URL": ["https://www.yna.co.kr/view/1"]}).to_csv(bk, index=False)
    out = import_bigkinds(ws, bk, _bundle()).set_index("source")
    assert out.at["bigkinds", "outlet"] == "연합뉴스" and out.at["bigkinds", "pub_date"] == "2023-08-11"
    assert read_csv(ws.candidates_csv).set_index("candidate_id").at["c1", "include"] == "Y"


def test_clean_title_and_outlet():
    from medclaim.collect.scraper import clean_outlet_name, clean_title
    assert clean_title("[메디칼타임즈] PPI제제 장기 복용시 치매 위험") == "PPI제제 장기 복용시 치매 위험"
    assert clean_title("위고비·삭센다 치료 중 &amp;#39;시력 손실&amp;#39; 부작용") == "위고비·삭센다 치료 중 '시력 손실' 부작용"
    assert clean_outlet_name("Daum | 연합뉴스") == "연합뉴스"
    assert clean_outlet_name("디지털투데이 (DigitalToday)") == "디지털투데이"
