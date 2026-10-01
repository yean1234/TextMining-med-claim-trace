"""작업 공간(workspace) 경로 모음.

모든 단계는 파일로 주고받는다. 사람이 손대는 파일은 annotation/ 에, 자동 산출물은
data/ 와 outputs/ 에 둔다. --root 로 작업 공간을 바꾸면(예: 데모) 같은 코드가 다른 폴더에서 돈다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = REPO_ROOT / "config"


@dataclass
class Workspace:
    root: Path = REPO_ROOT
    config_dir: Path = DEFAULT_CONFIG_DIR
    bundles_path: Path | None = None

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        self.config_dir = Path(self.config_dir).resolve()
        if self.bundles_path is None:
            self.bundles_path = self.config_dir / "bundles.yaml"
        self.bundles_path = Path(self.bundles_path).resolve()

    # ---- 디렉터리 ----
    @property
    def raw_dir(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.root / "data" / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def annotation_dir(self) -> Path:
        return self.root / "annotation"

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs"

    @property
    def html_dir(self) -> Path:
        return self.raw_dir / "html"

    @property
    def paper_raw_dir(self) -> Path:
        return self.raw_dir / "papers"

    @property
    def manual_text_dir(self) -> Path:
        return self.annotation_dir / "manual_texts"

    @property
    def figures_dir(self) -> Path:
        return self.outputs_dir / "figures"

    @property
    def tables_dir(self) -> Path:
        return self.outputs_dir / "tables"

    # ---- 자동 산출 파일 ----
    @property
    def papers_csv(self) -> Path:
        return self.interim_dir / "papers.csv"

    @property
    def articles_jsonl(self) -> Path:
        return self.interim_dir / "articles.jsonl"

    @property
    def comments_jsonl(self) -> Path:
        return self.interim_dir / "comments.jsonl"

    @property
    def selection_csv(self) -> Path:
        return self.interim_dir / "selection.csv"

    @property
    def features_csv(self) -> Path:
        return self.processed_dir / "article_features.csv"

    @property
    def report_md(self) -> Path:
        return self.outputs_dir / "report.md"

    # ---- 사람이 검증/코딩하는 파일 ----
    @property
    def discovery_csv(self) -> Path:
        return self.annotation_dir / "bundle_candidates.csv"

    @property
    def candidates_csv(self) -> Path:
        return self.annotation_dir / "article_candidates.csv"

    @property
    def text_check_csv(self) -> Path:
        return self.annotation_dir / "article_text_check.csv"

    @property
    def paper_coding_csv(self) -> Path:
        return self.annotation_dir / "paper_coding.csv"

    def coding_csv(self, coder: str) -> Path:
        return self.annotation_dir / f"article_coding_{coder}.csv"

    @property
    def disagreements_csv(self) -> Path:
        return self.annotation_dir / "disagreements.csv"

    @property
    def final_coding_csv(self) -> Path:
        return self.annotation_dir / "article_coding_final.csv"

    def config_file(self, name: str) -> Path:
        return self.config_dir / name

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.interim_dir, self.processed_dir, self.annotation_dir,
                  self.outputs_dir, self.html_dir, self.paper_raw_dir, self.manual_text_dir,
                  self.figures_dir, self.tables_dir):
            d.mkdir(parents=True, exist_ok=True)
