"""분석 실행 → outputs/tables/*.csv, outputs/figures/*.png, outputs/report.md"""
from __future__ import annotations

from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..bundles import load_bundles  # noqa: E402
from ..coding.strength import LEVELS  # noqa: E402
from ..nlp.lexicon import Lexicon  # noqa: E402
from ..nlp.tokenize import Tokenizer  # noqa: E402
from ..paths import Workspace  # noqa: E402
from ..utils import log, read_csv, read_jsonl, write_csv  # noqa: E402
from . import classify, comments, compare, textmining  # noqa: E402
from .dataset import build_dataset  # noqa: E402

OUTLET_COLORS = {"wire": "#4C78A8", "national_daily": "#F58518", "economic_daily": "#54A24B",
                 "broadcast": "#B279A2", "medical_trade": "#E45756", "health_consumer": "#72B7B2",
                 "online": "#9D755D", "unknown": "#BAB0AC"}


# ------------------------------------------------------------------ 도우미
def df_to_md(df: pd.DataFrame | None, max_rows: int = 40) -> str:
    if df is None or df.empty:
        return "_(결과 없음)_\n"
    d = df.head(max_rows).copy()
    d = d[[c for c in d.columns if not str(c).startswith("_")]]
    cells = d.astype(object).where(d.notna(), "").astype(str).map(lambda x: x.replace("|", "\\|").replace("\n", " "))
    lines = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    lines += ["| " + " | ".join(row) + " |" for row in cells.to_numpy().tolist()]
    more = f"\n_(상위 {max_rows}행만 표시, 전체 {len(df)}행)_" if len(df) > max_rows else ""
    return "\n".join(lines) + more + "\n"


def setup_font() -> bool:
    """한글 폰트가 있으면 쓰고, 없으면 그래프 라벨을 영문으로만 쓴다."""
    from matplotlib import font_manager
    for f in font_manager.fontManager.ttflist:
        if any(k in f.name for k in ("NanumGothic", "Noto Sans CJK", "Noto Sans KR", "Malgun", "AppleGothic")):
            plt.rcParams["font.family"] = f.name
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


# ------------------------------------------------------------------ 그림
def fig_strength_ladder(df: pd.DataFrame, path) -> None:
    bundles = sorted(df["bundle_id"].unique())
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(bundles) + 1.6))
    rng = np.random.default_rng(0)
    for i, bid in enumerate(bundles):
        g = df[df["bundle_id"] == bid]
        ps = g["paper_strength"].iloc[0]
        if not np.isnan(ps):
            ax.scatter(ps, i, marker="D", s=90, color="black", zorder=3, label="paper" if i == 0 else None)
        pr = g["press_release_strength"].iloc[0] if "press_release_strength" in g else np.nan
        if not np.isnan(pr):
            ax.scatter(pr, i, marker="s", s=60, facecolor="none", edgecolor="black", zorder=3,
                       label="press release" if i == 0 else None)
        for _, r in g.iterrows():
            if np.isnan(r["title_strength"]):
                continue
            ax.scatter(r["title_strength"] + rng.uniform(-0.15, 0.15), i + rng.uniform(-0.18, 0.18),
                       color=OUTLET_COLORS.get(r["outlet_type"], "#888"), s=40, alpha=0.85, zorder=2)
    for t, c in OUTLET_COLORS.items():
        if t in set(df["outlet_type"]):
            ax.scatter([], [], color=c, label=f"title: {t}")
    ax.set_yticks(range(len(bundles)))
    ax.set_yticklabels(bundles, fontsize=8)
    ax.set_xticks(range(7))
    ax.set_xticklabels([f"{k}\n{v}" for k, v in LEVELS.items()], fontsize=7)
    ax.set_xlim(-0.5, 6.5)
    ax.set_xlabel("causal claim strength (Sumner et al. 2014)")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title("Paper vs. news headline claim strength per bundle")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_outlet_rates(table: pd.DataFrame, df: pd.DataFrame, path) -> None:
    from .stats import cluster_bootstrap
    rows = []
    for otype, g in df.groupby("outlet_type"):
        est, lo, hi, n = cluster_bootstrap(g, "exceeds_title", n_boot=1000)
        if n:
            rows.append((otype, est, lo, hi, n))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for i, (otype, est, lo, hi, n) in enumerate(rows):
        ax.bar(i, est * 100, color=OUTLET_COLORS.get(otype, "#888"))
        if not np.isnan(lo):
            ax.errorbar(i, est * 100, yerr=[[est * 100 - lo * 100], [hi * 100 - est * 100]], color="black",
                        capsize=4)
        ax.text(i, 2, f"n={n}", ha="center", fontsize=8, color="white")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r[0] for r in rows], rotation=20, fontsize=8)
    ax.set_ylabel("% headlines stronger than paper")
    ax.set_ylim(0, 100)
    ax.set_title("Headline exceeds paper claim, by outlet type (95% cluster-bootstrap CI)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_condition_heatmap(summary: pd.DataFrame, path) -> None:
    if summary.empty:
        return
    cols = ["제목에_남음", "리드에_남음", "본문에_남음"]
    mat = summary[cols].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(5.5, 0.5 * len(summary) + 1.5))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["title", "lead", "body"])
    ax.set_yticks(range(len(summary)))
    ax.set_yticklabels([f"{c} (n={n})" for c, n in zip(summary["condition"], summary["n"])], fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.0f}%", ha="center", va="center",
                    color="white" if mat[i, j] > 60 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="% of articles mentioning the paper's key condition")
    ax.set_title("Key-condition retention by article section")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_comment_reactions(articles: pd.DataFrame, per_art: pd.DataFrame, path) -> None:
    if per_art.empty:
        return
    d = articles.merge(per_art, on="article_id").dropna(subset=["strong_title"])
    if d["strong_title"].nunique() < 2:
        return
    fig, ax = plt.subplots(figsize=(6, 3.2))
    width = 0.35
    for k, (val, label) in enumerate(((0, "title strength < 5"), (1, "title strength >= 5"))):
        g = d[d["strong_title"] == val]
        means = [g[f"share_{r}"].mean() * 100 for r in comments.REACTIONS]
        ax.bar(np.arange(3) + (k - 0.5) * width, means, width, label=f"{label} (n={len(g)})")
    ax.set_xticks(range(3))
    ax.set_xticklabels(comments.REACTIONS)
    ax.set_ylabel("% of comments (article-level mean)")
    ax.legend(fontsize=8)
    ax.set_title("Comment reactions by headline strength")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ 메인
def run_analysis(ws: Workspace, source: str = "final", paper_auto_fallback: bool = False,
                 n_boot: int = 2000) -> str:
    ws.ensure_dirs()
    setup_font()
    df = build_dataset(ws, source, paper_auto_fallback)
    if df.empty:
        raise RuntimeError("분석할 기사가 없습니다 (코딩 시트·features 확인).")
    write_csv(df, ws.tables_dir / "analysis_dataset.csv")
    bundles = {b.bundle_id: b for b in load_bundles(ws.bundles_path)}
    lex = Lexicon.load(ws.config_file("lexicon_ko.yaml"))
    user_words = [t for b in bundles.values() for t in b.drug_terms + b.outcome_terms]
    tok = Tokenizer(user_words, lex.data.get("stopwords", []))
    articles = {a["article_id"]: a for a in read_jsonl(ws.articles_jsonl)}
    bodies = {k: v.get("body", "") for k, v in articles.items()}
    synthetic = (ws.root / "SYNTHETIC").exists()

    # ---- RQ1
    overall = compare.overall_rates(df, n_boot)
    outlet = compare.by_outlet_type(df, max(500, n_boot // 2))
    ladder = compare.bundle_ladder(df)
    tvl = compare.title_vs_lead(df)
    prc = compare.press_release_conditional(df)
    gee_table, gee_msg = compare.gee_outlet(df)
    for name, t in (("rq1_overall", overall), ("rq1_by_outlet_type", outlet), ("rq1_bundle_ladder", ladder),
                    ("rq1_title_vs_lead", tvl), ("rq1_press_release", prc), ("rq1_gee", gee_table)):
        if t is not None and not t.empty:
            write_csv(t, ws.tables_dir / f"{name}.csv")
    fig_strength_ladder(df, ws.figures_dir / "strength_ladder.png")
    fig_outlet_rates(outlet, df, ws.figures_dir / "outlet_exceeds.png")

    # ---- RQ2
    cond_long, cond_sum = compare.condition_tables(df)
    if not cond_long.empty:
        write_csv(cond_long, ws.tables_dir / "rq2_condition_long.csv")
        write_csv(cond_sum, ws.tables_dir / "rq2_condition_summary.csv")
        fig_condition_heatmap(cond_sum, ws.figures_dir / "condition_retention.png")

    # ---- RQ3
    cdf = comments.code_comments(ws, lex)
    per_art = comments.per_article(cdf)
    rq3 = comments.compare_groups(df, per_art)
    if not per_art.empty:
        write_csv(per_art, ws.tables_dir / "rq3_comments_per_article.csv")
        write_csv(rq3, ws.tables_dir / "rq3_comment_reactions.csv")
        fig_comment_reactions(df, per_art, ws.figures_dir / "comment_reactions.png")

    # ---- 텍스트마이닝
    title_toks = [t for x in df["title"] for t in tok.tokens(x)]
    body_toks = [t for aid in df["article_id"] for t in tok.tokens(bodies.get(aid, ""))]
    lo_title_body = textmining.log_odds_dirichlet(title_toks, body_toks)
    ex = df.dropna(subset=["exceeds_title"])
    lo_exceeds = textmining.log_odds_dirichlet(
        [t for x in ex.loc[ex["exceeds_title"] == 1, "title"] for t in tok.tokens(x)],
        [t for x in ex.loc[ex["exceeds_title"] == 0, "title"] for t in tok.tokens(x)], min_count=2)
    tfidf_outlet = textmining.tfidf_by_group((df["title"] + " " + df["lead"]).tolist(),
                                             df["outlet_type"].tolist(), tok)
    pmi = textmining.drug_pmi(df, bodies, {k: b.drug_terms for k, b in bundles.items()}, tok)
    lda = textmining.lda_topics(cdf["text"].tolist(), tok) if not cdf.empty else pd.DataFrame()
    for name, t in (("tm_logodds_title_vs_body", lo_title_body), ("tm_logodds_exceeds", lo_exceeds),
                    ("tm_tfidf_outlet", tfidf_outlet), ("tm_drug_pmi", pmi), ("tm_comment_lda", lda)):
        if t is not None and not t.empty:
            write_csv(t, ws.tables_dir / f"{name}.csv")

    # ---- 분류
    nb_table, nb_msg = classify.nb_groupkfold(df, tok)

    # ---- 리포트
    agreement = read_csv(ws.tables_dir / "agreement.csv") if (ws.tables_dir / "agreement.csv").exists() else None
    n_b = df["bundle_id"].nunique()
    per_bundle = df.groupby("bundle_id").size()
    paper_src = df.drop_duplicates("bundle_id")["paper_code_source"].value_counts().to_dict()
    md: list[str] = []
    md.append("# 결과 리포트 — 약물 부작용 연구의 온라인 전달 과정\n")
    md.append(f"_생성: {datetime.now():%Y-%m-%d %H:%M} · 기사 코드 출처: `{source}` · 논문 코드 출처: {paper_src}_\n")
    if synthetic:
        md.append("> ⚠️ **합성(가짜) 데이터 데모입니다.** 약물·매체·기사·댓글이 모두 지어낸 것이며, 아래 수치는 "
                  "파이프라인이 돌아가는지 보여줄 뿐 아무 의미가 없습니다.\n")
    if source == "auto":
        md.append("> ⚠️ **자동(규칙 기반) 코드로 만든 리포트입니다.** 사람 검증 전 값이며, 특히 텍스트마이닝 절은 "
                  "같은 사전으로 판정·설명하는 순환 구조라 결론에 쓰면 안 됩니다.\n")
    if paper_auto_fallback:
        md.append("> ⚠️ 일부 논문 강도·설계가 자동 제안값으로 대체되었습니다 (`--paper-auto-fallback`).\n")

    md.append("## 0. 데이터 현황\n")
    md.append(f"- 묶음 {n_b}개 (목표 20~25) · 기사 {len(df)}건 · 묶음당 기사 {per_bundle.min()}~{per_bundle.max()}건 "
              f"(중앙값 {per_bundle.median():.0f}, 목표 3~5)")
    small = per_bundle[per_bundle < 3]
    if len(small):
        md.append(f"- ⚠️ 기사 3건 미만 묶음: {', '.join(small.index)} → 매체 간 비교에서 제외 고려")
    md.append(f"- 매체 유형: {df['outlet_type'].value_counts().to_dict()}")
    if "final_status" in df.columns:
        md.append(f"- 최종 코드 상태: {df['final_status'].value_counts().to_dict()}")
    md.append("")

    md.append("## 1. 코딩 신뢰도\n")
    md.append("코더 간 일치도(Cohen's κ, 순서형은 선형 가중 κ)와 자동 사전 코딩 vs 사람 일치도.\n")
    md.append(df_to_md(agreement))

    md.append("## 2. RQ1 — 논문의 '연관'이 기사 제목에서 '원인'으로 강화되는가\n")
    md.append("척도: " + ", ".join(f"{k}={v}" for k, v in LEVELS.items()) + "\n")
    md.append("### 2-1. 전체 비율\n" + df_to_md(overall))
    md.append("### 2-2. 매체 유형별\n" + df_to_md(outlet))
    md.append("![ladder](figures/strength_ladder.png)\n\n![outlet](figures/outlet_exceeds.png)\n")
    md.append("### 2-3. 묶음별 (논문 강도 vs 제목 강도 분포)\n" + df_to_md(ladder))
    md.append("### 2-4. 제목 vs 리드\n" + df_to_md(tvl))
    if prc is not None:
        md.append("### 2-5. 보도자료가 논문보다 강할 때 (Sumner 식 비교)\n" + df_to_md(prc))
    md.append(f"### 2-6. GEE (제목>논문 ~ 매체 유형)\n_{gee_msg}_\n\n" + df_to_md(gee_table))

    md.append("## 3. RQ2 — 연구 대상·용량·기간·한계가 제목에서 빠지는가\n")
    md.append("논문별 핵심 조건(paper_coding.csv 의 key_conditions)이 기사 구간에 언급됐는지(자동 탐지)의 비율(%).\n")
    md.append(df_to_md(cond_sum))
    if not cond_sum.empty:
        md.append("![conditions](figures/condition_retention.png)\n")
    md.append("‘언급 여부’만 본 것이다. 4.4년 → ‘장기’처럼 뭉뚱그린 경우는 언급으로 잡히므로, 오해 유발 여부는 "
              "사람 코드(title_omission_misleading)로 따로 본다.\n")

    md.append("## 4. RQ3 — 단정적·조건 생략 제목 기사에서 댓글의 불안·불신이 더 많은가\n")
    if per_art.empty:
        md.append("_(댓글 데이터 없음 — `medclaim collect-comments` 필요)_\n")
    else:
        md.append(f"댓글 {len(cdf)}개 / 기사 {per_art.shape[0]}건 (기사당 5개 이상만 비교).\n")
        md.append(df_to_md(rq3))
        md.append("![comments](figures/comment_reactions.png)\n")
        md.append("연관만 보여준다(기사→감정 인과 아님). 댓글은 네이버 인링크 기사에만 있어 일반지에 편중된다.\n")

    md.append("## 5. 텍스트마이닝\n")
    md.append("### 5-1. 제목 특유 표현 (로그오즈 z, A=제목 / B=본문)\n" + df_to_md(lo_title_body))
    md.append("### 5-2. 논문보다 강한 제목(A) vs 아닌 제목(B)\n" + df_to_md(lo_exceeds))
    md.append("### 5-3. 매체 유형별 TF-IDF 상위어 (제목+리드)\n" + df_to_md(tfidf_outlet))
    md.append("### 5-4. 약물 언급 문장의 공기어 (PMI)\n" + df_to_md(pmi))
    md.append("### 5-5. 댓글 LDA 토픽\n" + df_to_md(lda))

    md.append(f"## 6. (탐색) Naive Bayes 분류\n_{nb_msg}_\n\n" + df_to_md(nb_table))

    md.append("## 7. 해석 시 주의\n")
    md.append("- 같은 논문의 기사들은 독립이 아니다 → 묶음 단위 부트스트랩 CI를 같이 본다.\n"
              "- '논문보다 강하다' = 원논문 대비 상대 판정이다. 원논문 자체의 과장은 잡지 못한다.\n"
              "- 조건 미언급 ≠ 오해 유발. 제목에 모든 조건을 넣을 수는 없다.\n"
              "- 자세한 한계는 LIMITATIONS.md 참고.\n")
    text = "\n".join(md)
    ws.report_md.write_text(text, encoding="utf-8")
    log.info("리포트 → %s", ws.report_md)
    return text

