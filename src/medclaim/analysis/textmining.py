"""수업 방법론 적용: 토큰화 → 빈도·TF-IDF → 로그오즈(제목 특유 표현) → 공기어 PMI → LDA(댓글).

주의(순환 논리): 과장 여부는 원논문 대비 사람 판정으로 먼저 정하고, 언어 특징은 그 뒤에 비교한다.
--source auto 로 돌리면 '사전으로 판정한 과장'을 '같은 사전 단어'로 설명하는 셈이 되므로 해석 금지.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd

from ..coding.strength import DRUG_MASK
from ..nlp.lexicon import mask_terms
from ..nlp.segment import split_sentences
from ..nlp.tokenize import Tokenizer


def log_odds_dirichlet(tokens_a: list[str], tokens_b: list[str], prior_scale: float = 0.1,
                       min_count: int = 3, top_k: int = 15) -> pd.DataFrame:
    """Monroe, Colaresi & Quinn(2008) 정보적 디리클레 사전확률 로그오즈. z>0 이면 A 쪽 특유어."""
    ca, cb = Counter(tokens_a), Counter(tokens_b)
    na, nb = sum(ca.values()), sum(cb.values())
    if na == 0 or nb == 0:
        return pd.DataFrame()
    pooled = ca + cb
    total = sum(pooled.values())
    alpha0 = prior_scale * total
    rows = []
    for w, c in pooled.items():
        if c < min_count:
            continue
        aw = prior_scale * c
        ya, yb = ca[w], cb[w]
        delta = (math.log((ya + aw) / (na + alpha0 - ya - aw))
                 - math.log((yb + aw) / (nb + alpha0 - yb - aw)))
        var = 1.0 / (ya + aw) + 1.0 / (yb + aw)
        rows.append({"token": w, "count_a": ya, "count_b": yb, "z": delta / math.sqrt(var)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    top = pd.concat([df.nlargest(top_k, "z").assign(side="A"), df.nsmallest(top_k, "z").assign(side="B")])
    top["z"] = top["z"].round(2)
    return top.reset_index(drop=True)


def tfidf_by_group(texts: list[str], groups: list[str], tok: Tokenizer, top_k: int = 10) -> pd.DataFrame:
    from sklearn.feature_extraction.text import TfidfVectorizer
    if len(texts) < 3:
        return pd.DataFrame()
    vec = TfidfVectorizer(tokenizer=tok.tokens, lowercase=False, token_pattern=None, min_df=2)
    try:
        X = vec.fit_transform(texts)
    except ValueError:
        return pd.DataFrame()
    vocab = np.array(vec.get_feature_names_out())
    rows = []
    g = np.array(groups)
    for grp in sorted(set(groups)):
        mean = np.asarray(X[g == grp].mean(axis=0)).ravel()
        idx = mean.argsort()[::-1][:top_k]
        rows.append({"group": grp, "n_docs": int((g == grp).sum()),
                     "top_terms": ", ".join(f"{vocab[i]}({mean[i]:.3f})" for i in idx if mean[i] > 0)})
    return pd.DataFrame(rows)


def drug_pmi(articles: pd.DataFrame, bodies: dict[str, str], drug_terms_by_bundle: dict[str, list[str]],
             tok: Tokenizer, min_count: int = 3, top_k: int = 25) -> pd.DataFrame:
    """문장 단위 공기: 약물이 언급된 문장에 특히 자주 함께 나오는 단어 (PMI, log2)."""
    n_sent = 0
    n_drug = 0
    tok_count: Counter = Counter()
    joint: Counter = Counter()
    for _, a in articles.iterrows():
        text = f"{a['title']}\n{bodies.get(a['article_id'], '')}"
        terms = drug_terms_by_bundle.get(a["bundle_id"], [])
        for s in split_sentences(text):
            masked = mask_terms(s, terms, f" {DRUG_MASK} ")
            has_drug = DRUG_MASK in masked
            toks = set(tok.tokens(masked.replace(DRUG_MASK, " "))) - {t.lower() for t in terms}
            n_sent += 1
            n_drug += has_drug
            for t in toks:
                tok_count[t] += 1
                if has_drug:
                    joint[t] += 1
    if n_sent == 0 or n_drug == 0:
        return pd.DataFrame()
    p_drug = n_drug / n_sent
    rows = []
    for t, c in joint.items():
        if c < min_count:
            continue
        pmi = math.log2((c / n_sent) / ((tok_count[t] / n_sent) * p_drug))
        rows.append({"token": t, "co_count": c, "token_count": tok_count[t], "pmi": round(pmi, 3)})
    df = pd.DataFrame(rows)
    return df.sort_values(["pmi", "co_count"], ascending=False).head(top_k) if len(df) else df


def lda_topics(texts: list[str], tok: Tokenizer, n_topics: int = 5, top_k: int = 10,
               min_docs: int = 100, seed: int = 0) -> pd.DataFrame:
    """댓글 토픽 (13주차 LDA). 문서 수가 적으면 건너뛴다."""
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer
    if len(texts) < min_docs:
        return pd.DataFrame()
    vec = CountVectorizer(tokenizer=tok.tokens, lowercase=False, token_pattern=None, min_df=3, max_df=0.5)
    try:
        X = vec.fit_transform(texts)
    except ValueError:
        return pd.DataFrame()
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=seed, learning_method="batch")
    doc_topic = lda.fit_transform(X)
    vocab = vec.get_feature_names_out()
    rows = []
    for k, comp in enumerate(lda.components_):
        top = comp.argsort()[::-1][:top_k]
        rep = int(doc_topic[:, k].argmax())
        rows.append({"topic": k, "share": round(float(doc_topic[:, k].mean()), 3),
                     "top_terms": ", ".join(vocab[i] for i in top),
                     "representative": texts[rep][:80]})
    return pd.DataFrame(rows)
