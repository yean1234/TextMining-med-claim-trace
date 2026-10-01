# TextMining-med-claim-trace

**약물 부작용 연구의 온라인 전달 과정에서 나타나는 불확실성 표현과 적용 조건의 변화: 논문·기사 제목·본문·댓글 비교**
→ "어떤 기사를 신뢰해서 봐야 하는가?"

텍스트마이닝 팀 프로젝트(아이디어 2)의 실험 코드입니다. 논문 1편과 그 논문을 보도한 한국어 기사 3~5건을
한 **묶음(bundle)** 으로 묶고, 논문 → 기사 제목 → 리드 → 본문 → 댓글로 내려가면서
**인과 표현 강도**와 **적용 조건(대상·용량·기간·설계·한계)** 이 어떻게 바뀌는지 측정합니다.

| 연구 질문 | 측정 | 비교 결과물 |
|---|---|---|
| **RQ1** 논문의 "관련성이 관찰됐다"가 기사 제목에서 "원인이다"로 바뀌는가? | Sumner et al.(2014) 인과 표현 7범주(0~6)를 논문 초록 결론 / 기사 제목 / 리드에 코딩 → 차이(Δ) | 제목>논문 비율, 매체 유형별 비교, 묶음별 강도 사다리 그림, GEE |
| **RQ2** 본문에는 있는 연구 대상·용량·한계가 제목에는 빠지는가? | 논문별 '핵심 조건' 범주가 제목/리드/본문에 남았는지 + 사람의 '오해 유발' 판정 | 조건 유지율 히트맵, 제목에서만 빠진 비율 |
| **RQ3** 조건이 생략되거나 단정적인 제목이 붙은 기사에서 댓글의 불안·불신 표현이 더 많은가? | 댓글 반응 사전(불안·불신·안심) → 기사 단위 비율 | 강한 제목 vs 아닌 제목의 댓글 반응 차이(묶음 부트스트랩 CI) |

> ⚠️ **먼저 [LIMITATIONS.md](LIMITATIONS.md)를 읽어 주세요.** 이 코드의 수집기는 개발 환경에서 외부 네트워크가 막혀
> 실제 사이트로는 실행해 보지 못했고(가짜 응답·가짜 HTML로만 테스트), 자동 코딩은 '사람 코딩 보조용 제안값'입니다.

---

## 1. 데이터 구조

```
묶음 1 = 논문 1편 + 그 논문을 보도한 한국어 기사 3~5건
목표   = 묶음 20~25개 (기사 약 60~125건)
방향   = 논문을 먼저 정하고 → 그 논문을 보도한 기사를 찾아 내려간다
```

`config/bundles.yaml` 에 노션 사례 정리의 6개 묶음이 미리 들어 있습니다.

| bundle_id | 논문 | 비고 |
|---|---|---|
| B01_PPI_dementia | Northuis 2023, *Neurology* | 제목-본문 불일치, '4.4년 초과' 조건 누락 |
| B02_APAP_autism_Ahlqvist2024 | Ahlqvist 2024, *JAMA* | 형제 대조 시 연관성 소멸 |
| B03_APAP_autism_Wan2026 | Wan·Wong 2026, *JAMA Intern Med* | 제목 프레임 분산이 가장 큼 (DOI 미확인 → 제목 검색) |
| B04_semaglutide_NAION | Hathaway 2024, *JAMA Ophthalmol* | 단일기관 조건이 사라지는지 |
| B05_semaglutide_suicidality | Wang 2024, *Nature Medicine* | 위험 '축소' 방향 |
| B06_finasteride_depression | Brezis 2025, *J Clin Psychiatry* | '유발?' vs '인과관계 회의적', 출처 표기 오류 |

사례 2(타이레놀)는 논문이 2편이라 '묶음당 논문 1편' 원칙에 따라 B02/B03으로 나눴습니다.
나머지 14~19개 묶음은 `medclaim discover`로 후보를 찾아 추가합니다.

## 2. 파이프라인 (★ = 사람이 검증·입력하는 단계)

```
 (1) 데이터셋 만들기                              (2) 방법론 적용                         (3) 비교 결과
 ─────────────────                               ─────────────                          ──────────
 discover ──★ bundles.yaml 에 묶음 추가            precode                                 analyze
 fetch-papers (Crossref·Europe PMC)               ├ 문장분리·제목/리드/본문 분할(Kiwi)     ├ RQ1 표·그림·GEE
 search-news (네이버 API) / import-bigkinds        ├ 인과 강도 0~6 사전 코딩(규칙)          ├ RQ2 조건 유지율
   └★ article_candidates.csv: include, link_type   ├ 조건·hedge·booster·권고·출처 탐지      ├ RQ3 댓글 반응
 scrape (원문 → 실패 시 네이버 → 수동 붙여넣기)     ├ 묶음당 최대 5건 층화 선택              ├ 텍스트마이닝(로그오즈·TF-IDF·PMI·LDA)
   └★ article_text_check.csv / manual_texts/       └ 코딩 시트 생성(코더2는 블라인드)       ├ NB 분류(GroupKFold)
 collect-comments (선택)                           ★ paper_coding.csv (논문 강도·핵심 조건)  └ outputs/report.md
                                                  ★ annotate (코더1, 코더2)
                                                  agreement → ★ disagreements.csv → finalize
```

## 3. 설치 · 빠른 시작

```bash
pip install -r requirements.txt        # 또는 pip install -e .
python -m medclaim demo                # 합성(가짜) 데이터로 전체 흐름 시연 → runs/demo/outputs/report.md
python -m pytest -q                    # 테스트
```

`demo`는 네트워크 없이 **지어낸 약물·매체·기사·댓글**로 precode → 가상 코더 2명 → 일치도 → 최종 확정 → 분석·리포트까지
돌립니다. 결과 형식을 미리 보는 용도이며 수치에는 의미가 없습니다.

## 4. 실제 실행 순서

```bash
cp .env.example .env    # NAVER_CLIENT_ID / NAVER_CLIENT_SECRET / CROSSREF_MAILTO 입력
```

| 단계 | 명령 | 사람이 할 일 | 산출물 |
|---|---|---|---|
| 0 | `python -m medclaim status` | 진행 상황 확인 (묶음별 후보·include·본문·코딩 수) | – |
| 1 | `python -m medclaim discover` | `annotation/bundle_candidates.csv`에서 기사 3건 이상 붙은 (약물, 결과, 저널) 조합을 골라 원논문을 확인하고 `config/bundles.yaml`에 추가 | bundle_candidates.csv |
| 2 | `python -m medclaim fetch-papers` | `fetch_status` 확인. B03처럼 DOI 없이 제목으로 찾은 논문은 맞는 논문인지 확인 | data/interim/papers.csv |
| 3 | `python -m medclaim search-news` | **`annotation/article_candidates.csv`**: `relevance_score` 높은 순으로 보면서 `include`(Y/N), `link_type`(main/background/unrelated), `paper_match_evidence`(저널명·연구팀 언급 등) 입력. 같은 약물·결과를 다룬 **다른 논문** 기사(예: B02↔B03, B06 반박 코멘터리)를 반드시 가려낼 것 | article_candidates.csv |
| 3' | `python -m medclaim import-bigkinds 파일.xlsx --bundle B01_PPI_dementia` | (선택) 네이버 API는 날짜 필터가 없어 오래된 기사를 놓칠 수 있음 → BigKinds에서 기간 지정 검색 후 엑셀을 추가 | article_candidates.csv |
| 4 | `python -m medclaim scrape` | **`annotation/article_text_check.csv`**: `flag`가 ok가 아닌 기사는 `annotation/manual_texts/<article_id>.txt`에 원문을 붙여넣고(첫 줄 제목) 다시 `scrape`. 본문이 잘못 잡혔으면 `text_ok=N` | data/interim/articles.jsonl |
| 5 | `python -m medclaim collect-comments --i-accept-naver-terms` | (선택, RQ3) 비공식 엔드포인트 — 주의사항 확인 | data/interim/comments.jsonl |
| 6 | `python -m medclaim precode` | **`annotation/paper_coding.csv`**: 논문별 `design`, `paper_strength`(초록 결론의 주 주장 0~6), `paper_direction`, `key_conditions`(예: `duration;population`), `main_finding_ko` 입력 → 입력 후 `precode`를 한 번 더 돌리면 코딩 시트에 반영 | paper_coding.csv, article_coding_coder1/2.csv |
| 7 | `python -m medclaim annotate --coder coder1` (코더2도) | [코드북](docs/codebook.md)대로 코딩. 엑셀에서 CSV를 직접 채워도 됨. 코더2 시트는 자동 제안이 가려진 **블라인드** 시트 | article_coding_coder*.csv |
| 8 | `python -m medclaim agreement` | κ 확인 → **`annotation/disagreements.csv`**의 `resolved`에 토론 후 합의값 입력 | outputs/tables/agreement.csv |
| 9 | `python -m medclaim finalize` | `final_status`에 UNRESOLVED가 없는지 확인 | article_coding_final.csv |
| 10 | `python -m medclaim analyze` | 리포트 해석 (정량 + 정성 해석) | outputs/report.md, tables/, figures/ |

- 사람 입력이 끝나기 전 예비 분석: `analyze --source coder1 --paper-auto-fallback` 또는 `analyze --source auto`
  (auto 리포트에는 순환 논리 경고가 붙습니다).
- 어느 단계든 다시 실행해도 **사람이 이미 입력한 칸은 지워지지 않습니다**(자동 열만 갱신, 사라진 행은 `_stale=Y`로 남김).
- 묶음당 기사가 5건을 넘으면 `precode`가 매체 유형이 섞이도록 5건을 층화 추출합니다(`--max-per-bundle`).
  노션 메모의 "10건짜리 묶음 1개보다 4건짜리 묶음 20개가 낫다"를 반영한 것입니다.

## 5. 방법론 — 수업 내용과의 연결

| 수업 | 이 코드에서 | 위치 |
|---|---|---|
| 2주차 수집·크롤링·API | Crossref/Europe PMC API, 네이버 검색 API, 언론사 페이지 스크래핑(robots.txt 준수), BigKinds 가져오기 | `collect/` |
| 3주차 전처리·토큰화 | NFKC 정규화, 바이라인·사진설명·저작권 문구 제거, Kiwi 문장 분리·형태소 분석. **명사만 뽑지 않고** 부정(않다·없다·아니다)·가능(수 있다)을 보존 | `nlp/` |
| 5주차 BoW·TF-IDF | 매체 유형별 TF-IDF 상위어, 제목 vs 본문 로그오즈(Monroe et al. 2008) | `analysis/textmining.py` |
| 9~10주차 감성분석·분류 | 댓글 반응 사전(불안·불신·안심), Multinomial NB(묶음 단위 GroupKFold) | `analysis/comments.py`, `classify.py` |
| 12주차 동시출현·PMI | 약물 언급 문장의 공기어 PMI | `analysis/textmining.py` |
| 13주차 LDA | 댓글 토픽 (댓글 100개 이상일 때) | `analysis/textmining.py` |
| 내용분석(선행연구) | Sumner 7범주 인과 강도, 행동 권고 4범주, 동물→인간 일반화, 이중 코딩·κ, GEE | `coding/`, `annotate/`, `analysis/stats.py` |

**Text Mining = 정량 분석 + 정성 해석.** 사전 기반 자동 코딩은 '검토할 패턴을 찾는 역할'이고, 과장 여부는
원논문과의 비교로 사람이 판정합니다. 과장을 '단정어가 들어간 글'로 정의하고 '과장된 글에 단정어가 많다'고 결론
내리면 순환 논리가 되므로, 분석은 **사람이 확정한 코드(`--source final`)** 로 합니다.

## 6. 폴더 구조

```
config/        bundles.yaml(묶음) · lexicon_ko/en.yaml(사전) · journals.yaml · outlets.yaml · discovery.yaml
src/medclaim/  collect/(수집) nlp/(전처리) coding/(사전 코딩) annotate/(사람 코딩·κ) analysis/(분석·리포트) demo.py cli.py
annotation/    ★ 사람이 채우는 CSV (git으로 팀 공유)
data/          raw·interim·processed — 기사 원문·댓글이라 git에 올리지 않음(.gitignore)
outputs/       분석 결과 (git 제외, 필요하면 리포트만 따로 공유)
docs/          codebook.md(코딩 지침)
tests/         pytest (네트워크 없이 동작)
```

## 7. 참고문헌

- Sumner P, et al. (2014). The association between exaggeration in health related science news and academic press releases. *BMJ*, 349:g7015.
- Propfe & Seifert (2025 온라인/2026). Misrepresentation of semaglutide in social media. *Naunyn-Schmiedeberg's Arch Pharmacol*.
- Al Khaja et al. (2018). Drug information, misinformation, and disinformation on social media: a content analysis study. *J Public Health Policy*.
- Yeung et al. (2025). Online Information About Side Effects and Safety Concerns of Semaglutide: Mixed Methods Study of YouTube Videos. PMID 40198905.
- Ko et al. (2026). Comparative analysis of social issues toward medical abortion using mifepristone in South Korea and the United States: Topic modeling and sentiment analysis. *PLOS ONE*.
- Monroe BL, Colaresi MP, Quinn KM. (2008). Fightin' words. *Political Analysis*, 16(4).
