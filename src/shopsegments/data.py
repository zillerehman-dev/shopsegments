"""Loading and cleaning the Online Retail transactions."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import RAW_CSV

# Real products have a 5-digit stock code, sometimes followed by a letter variant
# (e.g. 85123A). Everything else is postage, fees, manual adjustments, gift vouchers...
PRODUCT_CODE_PATTERN = r"^\d{5}[A-Za-z]{0,2}$"


def load_raw(path: str | Path = RAW_CSV) -> pd.DataFrame:
    """Read the raw CSV with sensible dtypes and parsed dates."""
    df = pd.read_csv(
        path,
        dtype={"Invoice": str, "InvoiceNo": str, "StockCode": str, "Description": str, "Country": str},
        encoding="utf-8",
    )
    # Accept both "Online Retail" and "Online Retail II" column names.
    df = df.rename(columns={"Invoice": "InvoiceNo", "Price": "UnitPrice", "Customer ID": "CustomerID"})
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="mixed")
    df = df[df["InvoiceDate"] >= "2010-12-01"].reset_index(drop=True)  # Retail II starts Dec 2009
    df["CustomerID"] = df["CustomerID"].astype("Int64")
    return df


def clean_transactions(df: pd.DataFrame, return_report: bool = False):
    """Apply the cleaning rules and return one row per purchased line.

    Rules (each one is logged in the report so the notebook can show them):
      1. drop rows without a CustomerID (can't segment anonymous buyers)
      2. drop cancellations (InvoiceNo starting with 'C')
      3. drop non-positive quantity / price
      4. keep only real product stock codes (no POST, DOT, M, BANK CHARGES ...)
      5. drop exact duplicate rows
    """
    report = {"raw_rows": len(df)}
    out = df.copy()

    out = out[out["CustomerID"].notna()]
    report["after_missing_customer"] = len(out)

    out = out[~out["InvoiceNo"].str.startswith("C")]
    report["after_cancellations"] = len(out)

    out = out[(out["Quantity"] > 0) & (out["UnitPrice"] > 0)]
    report["after_non_positive"] = len(out)

    out = out[out["StockCode"].str.match(PRODUCT_CODE_PATTERN)]
    report["after_non_product_codes"] = len(out)

    out = out.drop_duplicates()
    report["after_duplicates"] = len(out)

    out["StockCode"] = out["StockCode"].str.upper()
    out["CustomerID"] = out["CustomerID"].astype(int)
    out["Revenue"] = out["Quantity"] * out["UnitPrice"]

    # One canonical description per product (the most frequent one).
    desc = (
        out.dropna(subset=["Description"])
        .groupby("StockCode")["Description"]
        .agg(lambda s: s.str.strip().value_counts().index[0])
    )
    out["Description"] = out["StockCode"].map(desc).fillna(out["StockCode"])
    out = out.reset_index(drop=True)

    return (out, report) if return_report else out


def product_catalog(tx: pd.DataFrame) -> pd.DataFrame:
    """Per-product summary used by the app (name, price, popularity)."""
    cat = (
        tx.groupby("StockCode")
        .agg(
            Description=("Description", "first"),
            median_price=("UnitPrice", "median"),
            n_customers=("CustomerID", "nunique"),
            n_orders=("InvoiceNo", "nunique"),
            units=("Quantity", "sum"),
            revenue=("Revenue", "sum"),
        )
        .sort_values("n_customers", ascending=False)
        .reset_index()
    )
    return cat
