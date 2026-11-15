"""Bonus: market basket analysis (association rules) and segment movement over time."""
from __future__ import annotations

import pandas as pd

from .rfm import build_rfm


def association_rules_table(tx: pd.DataFrame, top_items: int = 400, min_support: float = 0.01,
                            min_confidence: float = 0.3, min_lift: float = 3.0) -> pd.DataFrame:
    """FP-Growth on invoice baskets (restricted to the most common products for speed)."""
    from mlxtend.frequent_patterns import association_rules, fpgrowth

    common = tx["StockCode"].value_counts().index[:top_items]
    t = tx[tx["StockCode"].isin(common)]
    basket = pd.crosstab(t["InvoiceNo"], t["Description"]).gt(0)
    freq = fpgrowth(basket, min_support=min_support, use_colnames=True, max_len=3)
    rules = association_rules(freq, metric="confidence", min_threshold=min_confidence)
    rules = rules[rules["lift"] >= min_lift].copy()
    rules["if_bought"] = rules["antecedents"].apply(lambda s: " + ".join(sorted(s)))
    rules["then_likely"] = rules["consequents"].apply(lambda s: " + ".join(sorted(s)))
    rules = rules[["if_bought", "then_likely", "support", "confidence", "lift"]]
    return rules.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)


def segment_movement(tx: pd.DataFrame, model, before: str, after: str | None = None) -> pd.DataFrame:
    """Segment of every customer at two snapshot dates, using the same fitted segment model.

    Customers who had no purchases by `before` are labelled 'Not yet a customer'.
    """
    after_ts = pd.Timestamp(after) if after else tx["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    rfm_b = build_rfm(tx, pd.Timestamp(before))
    rfm_a = build_rfm(tx, after_ts)
    seg_b = model.predict(rfm_b).rename("from")
    seg_a = model.predict(rfm_a).rename("to")
    moves = seg_a.to_frame().join(seg_b, how="left")
    moves["from"] = moves["from"].fillna("Not yet a customer")
    return moves.groupby(["from", "to"]).size().rename("customers").reset_index()
