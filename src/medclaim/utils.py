"""입출력·HTTP·로깅 공통 함수."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd
import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger("medclaim")

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36 medclaim-course-project/0.1"
)


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    for noisy in ("matplotlib", "urllib3", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def load_dotenv(path: Path) -> None:
    """python-dotenv 없이 .env 를 읽어 환경변수로 넣는다(이미 있는 값은 덮어쓰지 않음)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# ---------------------------------------------------------------- HTTP
def get_session(user_agent: str = USER_AGENT) -> requests.Session:
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1.5, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=("GET",))
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": user_agent, "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8"})
    return session


class RateLimiter:
    """같은 서버에 너무 빨리 요청하지 않도록 최소 간격을 둔다."""

    def __init__(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self._last = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last = time.monotonic()


def decode_html(resp: requests.Response) -> str:
    """한국 언론사 중 EUC-KR 페이지가 있어 인코딩을 신중히 고른다."""
    ctype = resp.headers.get("Content-Type", "").lower()
    if "charset=" in ctype:
        return resp.text
    head = resp.content[:4096].decode("ascii", errors="ignore").lower()
    for enc in ("euc-kr", "cp949", "ks_c_5601"):
        if enc in head:
            return resp.content.decode("cp949", errors="replace")
    if "utf-8" in head:
        return resp.content.decode("utf-8", errors="replace")
    resp.encoding = resp.apparent_encoding or "utf-8"
    return resp.text


# ---------------------------------------------------------------- 파일
def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_csv(path: Path) -> pd.DataFrame:
    """모든 열을 문자열로 읽는다(사람이 엑셀에서 고친 값이 숫자/NaN 으로 바뀌지 않도록)."""
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    return df.astype(object)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """엑셀에서 한글이 깨지지 않도록 utf-8-sig(BOM)로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_id(*parts: str, length: int = 12) -> str:
    h = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return h[:length]


def merge_preserving(new: pd.DataFrame, path: Path, key: str | list[str],
                     human_cols: list[str]) -> pd.DataFrame:
    """자동 산출 시트를 다시 만들 때 사람이 이미 입력한 열(human_cols)을 보존한다.

    - 기존 파일에 있던 행의 human_cols 값은 그대로 유지
    - 새로 생긴 행은 human_cols 를 빈칸으로 추가
    - 기존 파일에만 있고 새 결과에 없는 행은 지우지 않고 `_stale=Y` 로 표시해 남긴다
    """
    keys = [key] if isinstance(key, str) else list(key)
    new = new.copy().astype(object)
    for col in human_cols:
        if col not in new.columns:
            new[col] = ""
    if not path.exists():
        return new
    old = read_csv(path)
    if not all(k in old.columns for k in keys):
        return new
    old_keys = old[keys].astype(str).agg("||".join, axis=1)
    new_keys = new[keys].astype(str).agg("||".join, axis=1)
    old_map = old.assign(_k=old_keys).drop_duplicates("_k").set_index("_k")
    for col in human_cols:
        if col not in old_map.columns:
            continue
        vals = new_keys.map(old_map[col])
        new[col] = [v if isinstance(v, str) else cur for v, cur in zip(vals, new[col])]
    stale = old[~old_keys.isin(set(new_keys))].copy()
    if len(stale):
        stale["_stale"] = "Y"
        new = pd.concat([new, stale], ignore_index=True, sort=False).fillna("")
    return new
