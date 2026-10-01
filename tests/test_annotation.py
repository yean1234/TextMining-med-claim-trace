import pandas as pd

from medclaim.annotate.agreement import compare_coders, finalize, kappa_row
from medclaim.coding.precode import select_articles
from medclaim.paths import Workspace
from medclaim.utils import merge_preserving, read_csv, write_csv


def test_merge_preserving_keeps_human_input(tmp_path):
    path = tmp_path / "sheet.csv"
    old = pd.DataFrame({"id": ["a", "b"], "auto": ["1", "2"], "human": ["Y", ""]})
    write_csv(old, path)
    new = pd.DataFrame({"id": ["a", "c"], "auto": ["9", "3"]})
    merged = merge_preserving(new, path, "id", ["human"])
    m = merged.set_index("id")
    assert m.at["a", "human"] == "Y" and m.at["a", "auto"] == "9"   # 사람 값 유지, 자동 값 갱신
    assert m.at["c", "human"] == ""
    assert m.at["b", "_stale"] == "Y"                                # 사라진 행은 지우지 않고 표시


def test_kappa_row():
    r = kappa_row(pd.Series(["1", "2", "3", "3"]), pd.Series(["1", "2", "3", "2"]), "x", "ordinal")
    assert r["n"] == 4 and r["pct_agree"] == 75.0 and r["weighted_kappa"] > r["kappa"]


def test_compare_and_finalize(tmp_path):
    ws = Workspace(root=tmp_path)
    ws.ensure_dirs()
    base = {"bundle_id": ["B1", "B1"], "title": ["t1", "t2"]}
    c1 = pd.DataFrame({"article_id": ["a1", "a2"], **base, "title_strength": ["6", "2"], "direction": ["inc", "inc"]})
    c2 = pd.DataFrame({"article_id": ["a1", "a2"], **base, "title_strength": ["5", "2"], "direction": ["inc", ""]})
    write_csv(c1, ws.coding_csv("coder1"))
    write_csv(c2, ws.coding_csv("coder2"))
    table, dis = compare_coders(c1, c2)
    assert list(dis["field"]) == ["title_strength"]
    dis["resolved"] = "6"
    write_csv(dis, ws.disagreements_csv)
    final = finalize(ws, ["coder1", "coder2"]).set_index("article_id")
    assert final.at["a1", "title_strength"] == "6" and "adjudicated" in final.at["a1", "final_status"]
    assert final.at["a2", "direction"] == "inc" and "single_coded" in final.at["a2", "final_status"]
    assert read_csv(ws.final_coding_csv).shape[0] == 2


def test_select_articles_caps_and_stratifies():
    arts = [{"article_id": f"a{i}", "bundle_id": "B1", "outlet_type": t}
            for i, t in enumerate(["wire"] * 4 + ["online"] * 4 + ["medical_trade"] * 2)]
    arts += [{"article_id": "z1", "bundle_id": "B2", "outlet_type": "wire"}]
    sel = select_articles(arts, max_per_bundle=5, seed=1)
    chosen = sel[(sel["bundle_id"] == "B1") & (sel["selected"] == "Y")]["article_id"]
    types = {a["outlet_type"] for a in arts if a["article_id"] in set(chosen)}
    assert len(chosen) == 5 and types == {"wire", "online", "medical_trade"}
    assert sel.set_index("article_id").at["z1", "selected"] == "Y"
