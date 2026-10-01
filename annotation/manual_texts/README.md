# manual_texts/

스크래핑에 실패한 기사(유료·로그인·robots.txt 차단·본문 추출 실패)의 원문을 사람이 붙여넣는 곳입니다.

- 파일명: `<article_id>.txt` (article_id 는 `article_text_check.csv` 의 값)
- 첫 줄: 기사 제목 / 둘째 줄부터: 본문
- 다시 `python -m medclaim scrape` 를 실행하면 이 파일이 최우선으로 쓰입니다.
- 저작권 때문에 이 폴더의 .txt 는 git 에 올라가지 않습니다(.gitignore).
