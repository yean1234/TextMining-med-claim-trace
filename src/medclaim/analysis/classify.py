"""(탐색적) Naive Bayes: 제목+리드 텍스트로 '논문보다 강한 제목' 여부를 예측할 수 있나.

같은 논문의 기사가 학습/평가에 갈라져 들어가면 모델이 그 연구 표현을 외워 성능이 부풀려진다
→ GroupKFold(묶음 단위)로 나눈다. 기사 60~125건 규모에서는 성능 추정의 불확실성이 크다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..nlp.tokenize import Tokenizer


def nb_groupkfold(df: pd.DataFrame, tok: Tokenizer, target: str = "exceeds_title",
                  min_n: int = 20) -> tuple[pd.DataFrame | None, str]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import GroupKFold
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import make_pipeline

    d = df.dropna(subset=[target]).copy()
    if len(d) < min_n:
        return None, f"표본 부족 (n={len(d)} < {min_n})"
    y = d[target].astype(int).to_numpy()
    if len(set(y)) < 2:
        return None, "한 클래스뿐"
    groups = d["bundle_id"].to_numpy()
    n_splits = min(5, len(set(groups)))
    if n_splits < 2:
        return None, "묶음이 2개 미만"
    X = (d["title"].fillna("") + " " + d["lead"].fillna("")).to_numpy()
    preds = np.zeros_like(y)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        if len(set(y[tr])) < 2:
            preds[te] = y[tr][0]
            continue
        model = make_pipeline(TfidfVectorizer(tokenizer=tok.tokens, lowercase=False, token_pattern=None),
                              MultinomialNB(alpha=1.0))
        model.fit(X[tr], y[tr])
        preds[te] = model.predict(X[te])
    majority = np.full_like(y, np.bincount(y).argmax())
    table = pd.DataFrame([
        {"model": "MultinomialNB (TF-IDF, Kiwi 토큰)", "accuracy": accuracy_score(y, preds),
         "macro_F1": f1_score(y, preds, average="macro")},
        {"model": "다수 클래스 기준선", "accuracy": accuracy_score(y, majority),
         "macro_F1": f1_score(y, majority, average="macro")},
    ]).round(3)
    return table, f"GroupKFold(n_splits={n_splits}, 묶음 단위), n={len(d)}, 양성 비율={y.mean():.2f}"
