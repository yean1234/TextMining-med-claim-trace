"""명령행 진입점:  python -m medclaim <명령>  (또는 pip install -e . 후 medclaim <명령>)

전체 흐름 (★ = 사람이 검증·입력하는 단계)
  status            묶음별 진행 상황
  discover          (선택) 새 묶음 후보 찾기 → ★ bundles.yaml 에 추가
  fetch-papers      논문 메타데이터·초록
  search-news       네이버 뉴스 후보 → ★ article_candidates.csv 의 include/link_type
  import-bigkinds   (선택) BigKinds 엑셀을 후보에 추가
  scrape            본문 수집 → ★ article_text_check.csv (실패 기사는 manual_texts 에 붙여넣기)
  collect-comments  (선택) 네이버 댓글
  precode           자동 사전 코딩 + 코딩 시트 생성 → ★ paper_coding.csv, article_coding_<코더>.csv
  annotate          ★ 대화형 코딩 도구
  agreement         코더 간 κ, 자동 vs 사람 → ★ disagreements.csv 의 resolved
  finalize          최종 코드 확정
  analyze           표·그림·리포트
  demo              합성 데이터로 전체 흐름 시연 (네트워크 불필요)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .paths import DEFAULT_CONFIG_DIR, REPO_ROOT, Workspace
from .utils import get_session, load_dotenv, log, read_csv, read_jsonl, setup_logging


def _ws(args) -> Workspace:
    return Workspace(root=Path(args.root), config_dir=Path(args.config_dir),
                     bundles_path=Path(args.bundles) if args.bundles else None)


def _bundles(ws: Workspace, only: str | None = None):
    from .bundles import load_bundles
    bundles = load_bundles(ws.bundles_path)
    if only:
        wanted = set(only.split(","))
        bundles = [b for b in bundles if b.bundle_id in wanted]
        if not bundles:
            sys.exit(f"--bundle {only} 에 해당하는 묶음이 없습니다.")
    return bundles


def _naver_keys() -> tuple[str, str]:
    cid, secret = os.environ.get("NAVER_CLIENT_ID", ""), os.environ.get("NAVER_CLIENT_SECRET", "")
    if not cid or not secret:
        sys.exit("NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 가 필요합니다 (.env.example 참고).")
    return cid, secret


# ------------------------------------------------------------------ 명령
def cmd_status(args) -> None:
    ws = _ws(args)
    bundles = _bundles(ws)
    cand = read_csv(ws.candidates_csv) if ws.candidates_csv.exists() else None
    arts = read_jsonl(ws.articles_jsonl)
    feats = read_csv(ws.features_csv) if ws.features_csv.exists() else None
    print(f"묶음 {len(bundles)}개 (목표 20~25)\n")
    print(f"{'bundle_id':38} {'후보':>5} {'include':>8} {'본문OK':>7} {'코딩대상':>8}  상태")
    for b in bundles:
        n_c = n_inc = 0
        if cand is not None:
            c = cand[cand["bundle_id"] == b.bundle_id]
            n_c = len(c)
            n_inc = int(c["include"].str.upper().eq("Y").sum()) if "include" in c else 0
        n_ok = sum(1 for a in arts if a["bundle_id"] == b.bundle_id and a.get("status") == "ok")
        n_f = int((feats["bundle_id"] == b.bundle_id).sum()) if feats is not None and len(feats) else 0
        flag = "OK" if n_ok >= 3 else ("기사 부족(<3)" if n_inc or n_ok else "수집 전")
        print(f"{b.bundle_id:38} {n_c:>5} {n_inc:>8} {n_ok:>7} {n_f:>8}  {flag}")
    from .annotate.interactive import coding_progress
    for coder in args.coders:
        done, total = coding_progress(ws, coder)
        if total:
            print(f"\n코딩 진행 [{coder}] {done}/{total}")


def cmd_discover(args) -> None:
    from .collect.discover import discover_bundles
    cid, secret = _naver_keys()
    discover_bundles(_ws(args), get_session(), cid, secret, args.max_per_query)


def cmd_fetch_papers(args) -> None:
    from .collect.papers import fetch_papers
    ws = _ws(args)
    fetch_papers(ws, _bundles(ws, args.bundle), get_session(),
                 mailto=args.mailto or os.environ.get("CROSSREF_MAILTO", ""), force=args.force)


def cmd_search_news(args) -> None:
    from .collect.naver import collect_candidates
    ws = _ws(args)
    cid, secret = _naver_keys()
    ws.ensure_dirs()
    collect_candidates(ws, _bundles(ws, args.bundle), get_session(), cid, secret, args.max_per_query, args.sort)


def cmd_import_bigkinds(args) -> None:
    from .collect.bigkinds import import_bigkinds
    ws = _ws(args)
    ws.ensure_dirs()
    import_bigkinds(ws, Path(args.file), _bundles(ws, args.bundle)[0])


def cmd_scrape(args) -> None:
    from .collect.scraper import scrape_articles
    scrape_articles(_ws(args), get_session(), use_auto=args.use_auto, delay=args.delay,
                    respect_robots=not args.ignore_robots, force=args.force)


def cmd_collect_comments(args) -> None:
    if not args.i_accept_naver_terms:
        sys.exit("댓글 수집은 비공식 엔드포인트를 씁니다. collect/comments.py 상단 주의사항을 읽고 "
                 "--i-accept-naver-terms 를 붙여 다시 실행하세요.")
    from .collect.comments import collect_comments
    n = collect_comments(_ws(args), get_session(), args.max_per_article, args.sort, args.delay)
    log.info("댓글 총 %d개", n)


def cmd_precode(args) -> None:
    from .coding.precode import precode
    ws = _ws(args)
    precode(ws, _bundles(ws), coders=args.coders, blind_coders=args.blind, max_per_bundle=args.max_per_bundle,
            seed=args.seed)


def cmd_annotate(args) -> None:
    from .annotate.interactive import run_interactive
    run_interactive(_ws(args), args.coder, args.show_body_chars)


def cmd_agreement(args) -> None:
    from .annotate.agreement import run_agreement
    from .analysis.report import df_to_md
    table = run_agreement(_ws(args), args.coders)
    print(df_to_md(table))


def cmd_finalize(args) -> None:
    from .annotate.agreement import finalize
    finalize(_ws(args), args.coders)


def cmd_analyze(args) -> None:
    from .analysis.report import run_analysis
    ws = _ws(args)
    run_analysis(ws, source=args.source, paper_auto_fallback=args.paper_auto_fallback, n_boot=args.n_boot)
    print(f"\n리포트: {ws.report_md}")


def cmd_demo(args) -> None:
    from .analysis.report import run_analysis
    from .demo import build_demo
    ws = build_demo(Path(args.out), seed=args.seed)
    run_analysis(ws, source="final", n_boot=500)
    print(f"\n합성 데모 리포트: {ws.report_md}")


# ------------------------------------------------------------------ 파서
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="medclaim", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(REPO_ROOT), help="작업 공간 루트 (기본: 레포 루트)")
    p.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR), help="사전·매체·저널 설정 폴더")
    p.add_argument("--bundles", default=None, help="bundles.yaml 경로 (기본: <config-dir>/bundles.yaml)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("status", help="묶음별 진행 상황")
    s.add_argument("--coders", nargs="+", default=["coder1", "coder2"])
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("discover", help="새 묶음 후보 찾기 (네이버 API)")
    s.add_argument("--max-per-query", type=int, default=None)
    s.set_defaults(func=cmd_discover)

    s = sub.add_parser("fetch-papers", help="논문 메타데이터·초록 (Crossref, Europe PMC)")
    s.add_argument("--bundle", help="쉼표로 구분한 bundle_id (기본: 전체)")
    s.add_argument("--mailto", default="")
    s.add_argument("--force", action="store_true", help="캐시 무시")
    s.set_defaults(func=cmd_fetch_papers)

    s = sub.add_parser("search-news", help="네이버 뉴스 검색 → 기사 후보")
    s.add_argument("--bundle")
    s.add_argument("--max-per-query", type=int, default=300)
    s.add_argument("--sort", choices=["sim", "date"], default="sim")
    s.set_defaults(func=cmd_search_news)

    s = sub.add_parser("import-bigkinds", help="BigKinds 엑셀 → 기사 후보")
    s.add_argument("file")
    s.add_argument("--bundle", required=True)
    s.set_defaults(func=cmd_import_bigkinds)

    s = sub.add_parser("scrape", help="include=Y 기사 본문 수집")
    s.add_argument("--use-auto", action="store_true", help="include 가 빈 칸이면 auto_include=Y 를 사용 (예비용)")
    s.add_argument("--delay", type=float, default=2.0)
    s.add_argument("--ignore-robots", action="store_true", help="robots.txt 무시 (권장하지 않음)")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_scrape)

    s = sub.add_parser("collect-comments", help="(선택) 네이버 댓글 수집")
    s.add_argument("--i-accept-naver-terms", action="store_true")
    s.add_argument("--max-per-article", type=int, default=200)
    s.add_argument("--sort", choices=["FAVORITE", "NEW"], default="FAVORITE")
    s.add_argument("--delay", type=float, default=2.0)
    s.set_defaults(func=cmd_collect_comments)

    s = sub.add_parser("precode", help="자동 사전 코딩 + 코딩 시트 생성")
    s.add_argument("--coders", nargs="+", default=["coder1", "coder2"])
    s.add_argument("--blind", nargs="*", default=["coder2"], help="자동 제안을 숨길 코더")
    s.add_argument("--max-per-bundle", type=int, default=5, help="묶음당 코딩할 최대 기사 수 (0=전부)")
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=cmd_precode)

    s = sub.add_parser("annotate", help="대화형 코딩")
    s.add_argument("--coder", required=True)
    s.add_argument("--show-body-chars", type=int, default=1500)
    s.set_defaults(func=cmd_annotate)

    s = sub.add_parser("agreement", help="코더 간 일치도 + 불일치 목록")
    s.add_argument("--coders", nargs="+", default=["coder1", "coder2"])
    s.set_defaults(func=cmd_agreement)

    s = sub.add_parser("finalize", help="최종 코드 확정")
    s.add_argument("--coders", nargs="+", default=["coder1", "coder2"])
    s.set_defaults(func=cmd_finalize)

    s = sub.add_parser("analyze", help="분석·그림·리포트")
    s.add_argument("--source", default="final", help="final | auto | <코더명>")
    s.add_argument("--paper-auto-fallback", action="store_true",
                   help="사람이 안 채운 논문 강도·설계를 자동 제안값으로 대체 (예비 분석)")
    s.add_argument("--n-boot", type=int, default=2000)
    s.set_defaults(func=cmd_analyze)

    s = sub.add_parser("demo", help="합성 데이터 데모 (네트워크 불필요)")
    s.add_argument("--out", default=str(REPO_ROOT / "runs" / "demo"))
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    load_dotenv(Path(args.root) / ".env")
    try:
        args.func(args)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        if args.verbose:
            raise
        sys.exit(f"오류: {e}")


if __name__ == "__main__":
    main()
