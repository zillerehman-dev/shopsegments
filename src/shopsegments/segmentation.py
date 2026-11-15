"""Clustering customers on RFM and turning clusters into business segments."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .config import RANDOM_STATE, RFM_FEATURES

# What the marketing team should do with each segment.
SEGMENT_PLAYBOOK = {
    "Champions": {
        "emoji": "🏆",
        "color": "#2E7D32",
        "description": "Bought recently, buy often and spend the most.",
        "action": "Reward them: VIP / loyalty perks, early access to new ranges, ask for reviews and referrals. "
        "Don't waste discounts here.",
    },
    "Loyal": {
        "emoji": "💙",
        "color": "#1565C0",
        "description": "Regular, recent buyers with solid spend — the next Champions.",
        "action": "Upsell and cross-sell with 'also bought' bundles, tiered loyalty points, "
        "free-shipping thresholds just above their usual basket.",
    },
    "New / Promising": {
        "emoji": "🌱",
        "color": "#00897B",
        "description": "Bought recently but only once or twice, small baskets so far.",
        "action": "Onboarding flow: welcome series, best-seller recommendations, "
        "a time-limited second-order incentive to build the habit.",
    },
    "At Risk": {
        "emoji": "⚠️",
        "color": "#EF6C00",
        "description": "Used to buy fairly regularly but haven't been back for a couple of months.",
        "action": "Win-back campaign now: personalised 'we miss you' email with their favourite "
        "categories and a moderate discount. Highest ROI for reactivation spend.",
    },
    "Lost": {
        "emoji": "💤",
        "color": "#757575",
        "description": "One-off, low-spend buyers who haven't purchased in ~6+ months.",
        "action": "Low-cost channels only: occasional newsletter or big seasonal sale. "
        "Don't spend paid-media budget here.",
    },
}
SEGMENT_ORDER = list(SEGMENT_PLAYBOOK)


def make_preprocessor() -> Pipeline:
    """log1p -> standardise. Fitted on the RFM table only (no target, so no leakage concern)."""
    return Pipeline(
        [
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )


def k_selection_table(X: np.ndarray, k_range=range(2, 11)) -> pd.DataFrame:
    """Inertia (elbow), silhouette, Calinski-Harabasz and Davies-Bouldin for each k."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        rows.append(
            {
                "k": k,
                "inertia": km.inertia_,
                "silhouette": silhouette_score(X, km.labels_),
                "calinski_harabasz": calinski_harabasz_score(X, km.labels_),
                "davies_bouldin": davies_bouldin_score(X, km.labels_),
            }
        )
    return pd.DataFrame(rows)


def compare_algorithms(X: np.ndarray, k: int) -> pd.DataFrame:
    """K-Means vs Gaussian Mixture vs DBSCAN on the same scaled features."""
    out = []
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
    out.append(("K-Means", k, silhouette_score(X, km.labels_), 0.0))

    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=RANDOM_STATE).fit(X)
    gl = gmm.predict(X)
    out.append(("Gaussian Mixture", len(np.unique(gl)), silhouette_score(X, gl), 0.0))

    for eps in (0.3, 0.5, 0.7):
        db = DBSCAN(eps=eps, min_samples=15).fit(X)
        lab = db.labels_
        n_clusters = len(set(lab)) - (1 if -1 in lab else 0)
        noise = float((lab == -1).mean())
        mask = lab != -1
        sil = silhouette_score(X[mask], lab[mask]) if n_clusters > 1 else np.nan
        out.append((f"DBSCAN eps={eps}", n_clusters, sil, noise))
    return pd.DataFrame(out, columns=["model", "n_clusters", "silhouette", "noise_share"])


def name_clusters(rfm: pd.DataFrame, labels: np.ndarray) -> dict[int, str]:
    """Map raw cluster ids to business names using each cluster's median R/F/M.

    Logic (for k=5):
      * value = rank of median Frequency + rank of median Monetary
      * highest value          -> Champions
      * second highest value   -> Loyal
      * of the remaining three: the least recent -> Lost,
        the most recent -> New / Promising, the middle one -> At Risk
    For other k we fall back to generic ordered names.
    """
    prof = rfm.assign(cluster=labels).groupby("cluster")[RFM_FEATURES].median()
    value = prof["Frequency"].rank() + prof["Monetary"].rank()
    order = value.sort_values(ascending=False).index.tolist()
    k = len(prof)
    if k != 5:
        return {c: f"Segment {i + 1}" for i, c in enumerate(order)}

    names = {order[0]: "Champions", order[1]: "Loyal"}
    rest = prof.loc[order[2:]].sort_values("Recency")
    names[rest.index[0]] = "New / Promising"
    names[rest.index[1]] = "At Risk"
    names[rest.index[2]] = "Lost"
    return names


@dataclass
class SegmentModel:
    """Fitted preprocessor + KMeans + cluster-id -> name mapping."""

    preprocessor: Pipeline
    kmeans: KMeans
    names: dict[int, str]

    @classmethod
    def fit(cls, rfm: pd.DataFrame, k: int = 5) -> "SegmentModel":
        pre = make_preprocessor()
        X = pre.fit_transform(rfm[RFM_FEATURES])
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(X)
        return cls(pre, km, name_clusters(rfm, km.labels_))

    def transform(self, rfm: pd.DataFrame) -> np.ndarray:
        return self.preprocessor.transform(rfm[RFM_FEATURES])

    def predict(self, rfm: pd.DataFrame) -> pd.Series:
        ids = self.kmeans.predict(self.transform(rfm))
        return pd.Series([self.names[i] for i in ids], index=rfm.index, name="Segment")


def segment_profiles(rfm: pd.DataFrame, tx: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Profile card data: size, spend, frequency, recency, revenue share, top products, action."""
    total_rev = rfm["Monetary"].sum()
    prof = rfm.groupby("Segment").agg(
        customers=("Monetary", "size"),
        avg_spend=("Monetary", "mean"),
        median_spend=("Monetary", "median"),
        avg_orders=("Frequency", "mean"),
        avg_recency_days=("Recency", "mean"),
        avg_order_value=("AvgOrderValue", "mean"),
        revenue=("Monetary", "sum"),
    )
    prof["share_customers"] = prof["customers"] / prof["customers"].sum()
    prof["share_revenue"] = prof["revenue"] / total_rev

    seg_tx = tx.merge(rfm[["Segment"]], left_on="CustomerID", right_index=True)
    top = (
        seg_tx.groupby(["Segment", "Description"])["CustomerID"]
        .nunique()
        .rename("buyers")
        .reset_index()
        .sort_values(["Segment", "buyers"], ascending=[True, False])
        .groupby("Segment")
        .head(top_n)
        .groupby("Segment")["Description"]
        .apply(list)
    )
    prof["top_products"] = top
    prof["action"] = [SEGMENT_PLAYBOOK.get(s, {}).get("action", "") for s in prof.index]
    prof["description"] = [SEGMENT_PLAYBOOK.get(s, {}).get("description", "") for s in prof.index]
    order = [s for s in SEGMENT_ORDER if s in prof.index]
    return prof.loc[order]
