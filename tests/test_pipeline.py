"""Small, fast unit tests on synthetic data (no download needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shopsegments.data import clean_transactions  # noqa: E402
from shopsegments.recommender import (  # noqa: E402
    InteractionData, ItemCFRecommender, PopularityRecommender, build_holdout, ranking_metrics,
)
from shopsegments.rfm import build_rfm  # noqa: E402
from shopsegments.segmentation import SegmentModel  # noqa: E402


@pytest.fixture
def raw():
    rows = [
        ("1", "10001", "MUG RED", 2, "2011-01-01 10:00", 1.0, 1, "UK"),
        ("1", "10002", "MUG BLUE", 1, "2011-01-01 10:00", 2.0, 1, "UK"),
        ("C2", "10001", "MUG RED", -1, "2011-01-02 10:00", 1.0, 1, "UK"),   # cancellation
        ("3", "POST", "POSTAGE", 1, "2011-01-03 10:00", 18.0, 2, "UK"),      # fee
        ("4", "10002", "MUG BLUE", 1, "2011-01-04 10:00", 2.0, None, "UK"),  # no customer
        ("5", "10001", "MUG RED", 0, "2011-01-05 10:00", 1.0, 3, "UK"),      # qty 0
        ("6", "10003A", "PLATE", 3, "2011-02-01 10:00", 1.5, 2, "FR"),
        ("6", "10003A", "PLATE", 3, "2011-02-01 10:00", 1.5, 2, "FR"),       # duplicate
    ]
    df = pd.DataFrame(rows, columns=["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate",
                                     "UnitPrice", "CustomerID", "Country"])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["CustomerID"] = df["CustomerID"].astype("Int64")
    return df


def test_cleaning_rules(raw):
    tx, rep = clean_transactions(raw, return_report=True)
    assert len(tx) == 3
    assert not tx["InvoiceNo"].str.startswith("C").any()
    assert set(tx["StockCode"]) == {"10001", "10002", "10003A"}
    assert rep["raw_rows"] == 8 and rep["after_duplicates"] == 3
    assert np.isclose(tx["Revenue"].sum(), 2 * 1 + 1 * 2 + 3 * 1.5)


def test_rfm(raw):
    tx = clean_transactions(raw)
    rfm = build_rfm(tx, pd.Timestamp("2011-02-02"))
    assert rfm.loc[1, "Frequency"] == 1 and rfm.loc[1, "Monetary"] == 4.0
    assert rfm.loc[2, "Recency"] == 0 and rfm.loc[1, "Recency"] == 31


def synthetic_rfm(n=500, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "Recency": rng.integers(0, 365, n),
        "Frequency": rng.integers(1, 40, n),
        "Monetary": rng.lognormal(6, 1.2, n),
    })


def test_segment_names_and_predict():
    rfm = synthetic_rfm()
    model = SegmentModel.fit(rfm, k=5)
    assert set(model.names.values()) == {"Champions", "Loyal", "New / Promising", "At Risk", "Lost"}
    seg = model.predict(rfm)
    med = rfm.assign(s=seg).groupby("s").median()
    assert med.loc["Champions", "Monetary"] > med.loc["Lost", "Monetary"]
    assert med.loc["Lost", "Recency"] > med.loc["New / Promising", "Recency"]


def test_metrics_perfect_and_empty():
    truth = {1: {"a", "b"}}
    m = ranking_metrics({1: ["a", "b"] + ["x"] * 8}, truth, k=10)
    assert m["Recall@10"] == 1.0 and m["Precision@10"] == 0.2 and m["NDCG@10"] == pytest.approx(1.0)
    assert ranking_metrics({1: ["x"] * 10}, truth, k=10)["HitRate@10"] == 0.0


def _toy_tx():
    rows = []
    for u in range(20):
        items = ["A", "B"] if u % 2 == 0 else ["C", "D"]
        for i, it in enumerate(items):
            rows.append((f"{u}-{i}", it, it, 1, pd.Timestamp("2011-01-01") + pd.Timedelta(days=i * 40), 1.0, u, "UK"))
    return pd.DataFrame(rows, columns=["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate",
                                       "UnitPrice", "CustomerID", "Country"])


def test_itemcf_learns_cooccurrence_and_filters_seen():
    tx = _toy_tx()
    train, truth, _ = build_holdout(tx, "2011-02-01")  # first item in train, second in test
    data = InteractionData(train)
    rec = ItemCFRecommender(n_neighbours=5).fit(data).recommend([0, 1], k=1)
    # with only first items in train there is no co-occurrence; just make sure seen items are filtered
    assert rec[0][0] != "A" and rec[1][0] != "C"
    full = InteractionData(tx)
    cf = ItemCFRecommender(n_neighbours=5).fit(full)
    assert cf.sim[full.item_index["A"], full.item_index["B"]] > 0
    assert cf.sim[full.item_index["A"], full.item_index["C"]] == 0


def test_popularity_cold_start():
    data = InteractionData(_toy_tx())
    pop = PopularityRecommender().fit(data)
    assert len(pop.recommend_new(3)) == 3
    assert pop.recommend([999], k=2)[0] == pop.recommend_new(2)  # unknown user -> popular items
