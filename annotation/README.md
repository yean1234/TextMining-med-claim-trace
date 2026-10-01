# annotation/ — 사람이 검증·입력하는 파일

이 폴더의 CSV 는 git 으로 팀원끼리 공유합니다(기사 원문은 없음). 엑셀에서 열 때는 UTF-8(BOM) 이라 한글이 그대로 보입니다.
명령을 다시 실행해도 아래 '사람 입력 열'은 지워지지 않습니다.

| 파일 | 만드는 명령 | 사람 입력 열 | 할 일 |
|---|---|---|---|
| `bundle_candidates.csv` | `discover` | picked, paper_doi, notes | 기사 3건↑ 조합의 원논문 확인 → `config/bundles.yaml` 에 추가 |
| `article_candidates.csv` | `search-news`, `import-bigkinds` | include, link_type, paper_match_evidence, candidate_notes | 이 논문을 보도한 기사인지 판정 (docs/codebook.md J) |
| `article_text_check.csv` | `scrape` | text_ok, text_fix_notes | 본문이 제대로 수집됐는지 확인. 실패 기사는 `manual_texts/<article_id>.txt` 에 원문 붙여넣기 |
| `paper_coding.csv` | `precode` | design, paper_strength, paper_direction, key_conditions, key_conditions_detail, main_finding_ko, paper_caveat, press_release_url, press_release_strength, checked_by, paper_notes | 논문 코딩 (codebook I) |
| `article_coding_<코더>.csv` | `precode` | title_strength, lead_strength, direction, advice, caveat_in_body, title_omission_misleading, animal_generalization, source_error, coder_notes | 기사 코딩 (codebook B~H). `annotate --coder <코더>` 로 해도 됨 |
| `disagreements.csv` | `agreement` | resolved, resolution_notes | 두 코더가 다른 값 → 토론 후 합의값 |
| `article_coding_final.csv` | `finalize` | – | 분석에 쓰이는 최종 코드 |
