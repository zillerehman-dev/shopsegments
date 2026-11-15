"""RFM (Recency, Frequency, Monetary) feature building."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_rfm(tx: pd.DataFrame, snapshot_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per customer.

    Recency   = days since last purchase (relative to snapshot_date)
    Frequency = number of distinct invoices
    Monetary  = total spend
    Plus a few descriptive extras used in the profile cards.
    """
    if snapshot_date is None:
        snapshot_date = tx["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    tx = tx[tx["InvoiceDate"] < snapshot_date]

    g = tx.groupby("CustomerID")
    rfm = pd.DataFrame(
        {
            "Recency": (snapshot_date - g["InvoiceDate"].max()).dt.days,
            "Frequency": g["InvoiceNo"].nunique(),
            "Monetary": g["Revenue"].sum(),
            "FirstPurchase": g["InvoiceDate"].min(),
            "LastPurchase": g["InvoiceDate"].max(),
            "Country": g["Country"].agg(lambda s: s.mode().iloc[0]),
            "UniqueProducts": g["StockCode"].nunique(),
        }
    )
    rfm["AvgOrderValue"] = rfm["Monetary"] / rfm["Frequency"]
    rfm["TenureDays"] = (snapshot_date - rfm["FirstPurchase"]).dt.days
    return rfm


def rfm_scores(rfm: pd.DataFrame) -> pd.DataFrame:
    """Classic 1-5 quintile scores (5 = best). Used as a rule-based reference."""
    out = pd.DataFrame(index=rfm.index)
    # rank(method='first') avoids duplicate bin edges when many customers tie.
    out["R"] = pd.qcut(rfm["Recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    out["F"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    out["M"] = pd.qcut(rfm["Monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    out["RFM_Score"] = out["R"].astype(str) + out["F"].astype(str) + out["M"].astype(str)
    return out


def log_features(rfm: pd.DataFrame, cols=("Recency", "Frequency", "Monetary")) -> pd.DataFrame:
    """RFM is heavily right-skewed; log1p makes it usable for distance-based clustering."""
    return np.log1p(rfm[list(cols)].clip(lower=0))
