"""합성 데이터로 precode → 코딩 흉내 → 일치도 → 최종 → 분석·리포트까지 끝까지 돈다."""
from medclaim.analysis.report import run_analysis
from medclaim.bundles import load_bundles
from medclaim.demo import build_demo
from medclaim.paths import DEFAULT_CONFIG_DIR
from medclaim.utils import read_csv


def test_real_bundles_yaml_is_valid():
    bundles = load_bundles(DEFAULT_CONFIG_DIR / "bundles.yaml")
    assert len(bundles) >= 6
    assert all(b.drug_terms and b.outcome_terms and b.queries for b in bundles)


def test_demo_end_to_end(tmp_path):
    ws = build_demo(tmp_path / "demo", seed=3)
    report = run_analysis(ws, source="final", n_boot=100)
    assert "합성(가짜) 데이터" in report
    assert (ws.figures_dir / "strength_ladder.png").exists()
    assert (ws.tables_dir / "rq1_overall.csv").exists()
    assert (ws.tables_dir / "agreement.csv").exists()
    final = read_csv(ws.final_coding_csv)
    assert (final["title_strength"] != "").all()
    # 자동 코드만으로도 돌아가야 한다 (경고 문구 포함)
    auto_report = run_analysis(ws, source="auto", n_boot=50)
    assert "자동(규칙 기반) 코드" in auto_report
