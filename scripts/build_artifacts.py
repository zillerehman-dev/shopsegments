"""End-to-end pipeline: raw CSV -> cleaned data -> segments -> recommenders -> app artifacts.

Usage:
    python scripts/build_artifacts.py            # full run (~2-3 min on a laptop)
    python scripts/build_artifacts.py --skip-tuning

Everything the Streamlit app needs is written to ./artifacts so the deployed app
never has to touch the raw data or train anything.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shopsegments.basket import association_rules_table, segment_movement  # noqa: E402
from shopsegments.config import ARTIFACTS, DATA_DIR, RANDOM_STATE, RECO_SPLIT_DATE, RFM_FEATURES, TOP_K  # noqa: E402
from shopsegments.data import clean_transactions, load_raw, product_catalog  # noqa: E402
from shopsegments.recommender import (  # noqa: E402
    ALSRecommender,
    HybridRecommender,
    InteractionData,
    ItemCFRecommender,
    PopularityRecommender,
    also_bought_table,
    build_holdout,
    evaluate_models,
    explain_recommendations,
    ranking_metrics,
)
from shopsegments.rfm import build_rfm, rfm_scores  # noqa: E402
from shopsegments.segmentation import SegmentModel, compare_algorithms, k_selection_table, segment_profiles  # noqa: E402

K_CLUSTERS = 5
VALIDATION_SPLIT = "2011-06-09"  # tuning happens inside the training period only


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tune_recommenders(train_tx: pd.DataFrame) -> dict:
    """Pick hyper-parameters on a validation window carved out of the training period,
    so the final test window is only used once."""
    fit_tx, val_truth, _ = build_holdout(train_tx, VALIDATION_SPLIT)
    log(f"validation users: {len(val_truth)}")

    cf_grid = [ItemCFRecommender(n) for n in (10, 20, 30, 50)]
    cf_res = evaluate_models(cf_grid, fit_tx, val_truth)
    cf_res["n_neighbours"] = [m.n_neighbours for m in cf_grid]
    best_cf = int(cf_res.sort_values("Precision@10", ascending=False).iloc[0]["n_neighbours"])

    als_grid = [ALSRecommender(f, r, a) for f in (32, 64, 128) for r in (0.05, 0.5) for a in (1, 2, 5, 20)]
    als_res = evaluate_models(als_grid, fit_tx, val_truth)
    als_res["params"] = [json.dumps(m.params) for m in als_grid]
    best_als = json.loads(als_res.sort_values("Precision@10", ascending=False).iloc[0]["params"])

    hy_grid = [
        HybridRecommender(ItemCFRecommender(best_cf), ALSRecommender(**best_als), w_itemcf=w) for w in (0.3, 0.5, 0.7, 0.85)
    ]
    hy_res = evaluate_models(hy_grid, fit_tx, val_truth)
    hy_res["w_itemcf"] = [m.w for m in hy_grid]
    best_w = float(hy_res.sort_values("Precision@10", ascending=False).iloc[0]["w_itemcf"])

    tuning = pd.concat(
        [
            cf_res.assign(params=cf_res.pop("n_neighbours").map(lambda n: f"n_neighbours={n}")),
            als_res,
            hy_res.assign(params=hy_res.pop("w_itemcf").map(lambda w: f"w_itemcf={w}")),
        ],
        ignore_index=True,
    )
    tuning.to_csv(ARTIFACTS / "reco_tuning_validation.csv", index=False)
    return {"itemcf_neighbours": best_cf, "als": best_als, "hybrid_w_itemcf": best_w}


def main(skip_tuning: bool = False) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)

    # 1. Data --------------------------------------------------------------
    log("loading + cleaning transactions")
    tx, clean_report = clean_transactions(load_raw(), return_report=True)
    tx.to_parquet(DATA_DIR / "processed" / "transactions.parquet", index=False)
    catalog = product_catalog(tx)
    catalog.to_parquet(ARTIFACTS / "catalog.parquet", index=False)

    # 2. RFM + segmentation ---------------------------------------------------
    log("RFM + clustering")
    rfm = build_rfm(tx)
    seg_model = SegmentModel.fit(rfm, k=K_CLUSTERS)
    X = seg_model.transform(rfm)
    rfm["Segment"] = seg_model.predict(rfm)
    rfm = rfm.join(rfm_scores(rfm))

    k_table = k_selection_table(X)
    k_table.to_csv(ARTIFACTS / "k_selection.csv", index=False)
    algo_table = compare_algorithms(X, K_CLUSTERS)
    algo_table.to_csv(ARTIFACTS / "clustering_comparison.csv", index=False)

    log("UMAP + PCA projection")
    emb = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=RANDOM_STATE).fit_transform(X)
    rfm["umap_x"], rfm["umap_y"] = emb[:, 0], emb[:, 1]
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(X)
    p = pca.transform(X)
    rfm["pca_x"], rfm["pca_y"] = p[:, 0], p[:, 1]

    profiles = segment_profiles(rfm, tx)
    profiles.to_json(ARTIFACTS / "segment_profiles.json", orient="index", indent=2)
    joblib.dump(seg_model, ARTIFACTS / "segment_model.joblib")
    # Plain-JSON copy so the deployed app can score "what-if" customers without scikit-learn/pickles.
    scaler = seg_model.preprocessor.named_steps["scale"]
    (ARTIFACTS / "segment_model.json").write_text(json.dumps({
        "features": RFM_FEATURES,
        "transform": "log1p then standardise",
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "centroids": seg_model.kmeans.cluster_centers_.tolist(),
        "names": [seg_model.names[i] for i in range(K_CLUSTERS)],
    }, indent=2))

    # 3. Recommender evaluation ---------------------------------------------
    train_tx = tx[tx["InvoiceDate"] < pd.Timestamp(RECO_SPLIT_DATE)]
    if skip_tuning:
        best = {"itemcf_neighbours": 20, "als": {"factors": 64, "regularization": 0.5, "alpha": 2.0, "iterations": 20},
                "hybrid_w_itemcf": 0.7}
    else:
        log("tuning recommenders on validation window")
        best = tune_recommenders(train_tx)
    log(f"best params: {best}")

    log("final hold-out evaluation (test window used once)")
    train, truth, cold_truth = build_holdout(tx, RECO_SPLIT_DATE)
    models = [
        PopularityRecommender(),
        ItemCFRecommender(best["itemcf_neighbours"]),
        ALSRecommender(**best["als"]),
        HybridRecommender(ItemCFRecommender(best["itemcf_neighbours"]), ALSRecommender(**best["als"]),
                          w_itemcf=best["hybrid_w_itemcf"]),
    ]
    test_metrics = evaluate_models(models, train, truth)

    # Cold start: customers whose first purchase is after the split only get popularity recs.
    data_train = InteractionData(train)
    pop = PopularityRecommender().fit(data_train)
    pop_list = pop.recommend_new(TOP_K)
    cold_metrics = ranking_metrics({u: pop_list for u in cold_truth}, cold_truth, TOP_K, len(data_train.items))
    rng = np.random.default_rng(RANDOM_STATE)
    rand_metrics = ranking_metrics(
        {u: list(rng.choice(data_train.items, TOP_K, replace=False)) for u in cold_truth}, cold_truth, TOP_K
    )

    # Segment-level breakdown for the best model
    best_model = models[int(test_metrics["Precision@10"].idxmax())]
    users = list(truth)
    recs_best = dict(zip(users, best_model.recommend(users, TOP_K)))
    by_seg = []
    for seg, g in rfm.loc[rfm.index.intersection(users)].groupby("Segment"):
        t = {u: truth[u] for u in g.index}
        by_seg.append({"Segment": seg, **ranking_metrics(recs_best, t, TOP_K)})

    metrics = {
        "split_date": RECO_SPLIT_DATE,
        "best_params": best,
        "best_model": best_model.name,
        "test": test_metrics.to_dict(orient="records"),
        "cold_start": {"popularity": cold_metrics, "random": rand_metrics},
        "by_segment": by_seg,
        "clean_report": clean_report,
        "n_customers": int(len(rfm)),
        "n_products": int(tx["StockCode"].nunique()),
        "n_invoices": int(tx["InvoiceNo"].nunique()),
        "date_range": [str(tx["InvoiceDate"].min().date()), str(tx["InvoiceDate"].max().date())],
        "silhouette_k5": float(k_table.set_index("k").loc[K_CLUSTERS, "silhouette"]),
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    log("\n" + test_metrics.round(4).to_string())

    # 4. Production recommender trained on ALL data -------------------------
    log("training production recommender on full history")
    full = InteractionData(tx)
    prod = HybridRecommender(ItemCFRecommender(best["itemcf_neighbours"]), ALSRecommender(**best["als"]),
                             w_itemcf=best["hybrid_w_itemcf"]).fit(full)
    all_users = list(full.users)
    all_recs = prod.recommend(all_users, TOP_K)
    # "because you bought ..." uses the un-pruned cosine similarity
    explainer = ItemCFRecommender(n_neighbours=0).fit(full)
    rows = []
    for u, recs in zip(all_users, all_recs):
        because = explain_recommendations(full, explainer.sim, u, recs)
        for r, (it, b) in enumerate(zip(recs, because), 1):
            rows.append((u, r, it, b))
    pd.DataFrame(rows, columns=["CustomerID", "rank", "StockCode", "because_StockCode"]).to_parquet(
        ARTIFACTS / "customer_recs.parquet", index=False
    )

    # cold start lists: overall + last-90-days trending + per country
    recent = tx[tx["InvoiceDate"] >= tx["InvoiceDate"].max() - pd.Timedelta(days=90)]
    popular = {
        "all_time": catalog["StockCode"].head(50).tolist(),
        "trending_90d": recent.groupby("StockCode")["CustomerID"].nunique().sort_values(ascending=False).head(50).index.tolist(),
        "by_country": {
            c: g.groupby("StockCode")["CustomerID"].nunique().sort_values(ascending=False).head(20).index.tolist()
            for c, g in tx.groupby("Country") if g["CustomerID"].nunique() >= 5
        },
    }
    (ARTIFACTS / "popular.json").write_text(json.dumps(popular, indent=2))

    log("also-bought table")
    also_bought_table(full).to_parquet(ARTIFACTS / "also_bought.parquet", index=False)

    # purchase history (top 15 products per customer) for the app
    hist = (
        tx.groupby(["CustomerID", "StockCode"])
        .agg(orders=("InvoiceNo", "nunique"), units=("Quantity", "sum"), spend=("Revenue", "sum"),
             last=("InvoiceDate", "max"))
        .reset_index()
        .sort_values(["CustomerID", "orders", "spend"], ascending=[True, False, False])
        .groupby("CustomerID")
        .head(15)
    )
    hist.to_parquet(ARTIFACTS / "customer_history.parquet", index=False)

    # 5. Bonus: basket rules + segment movement ------------------------------
    log("market basket analysis")
    association_rules_table(tx).head(300).to_parquet(ARTIFACTS / "basket_rules.parquet", index=False)
    log("segment movement")
    segment_movement(tx, seg_model, before="2011-06-09").to_parquet(ARTIFACTS / "segment_movement.parquet", index=False)

    # monthly revenue by segment (for the overview chart)
    monthly = (
        tx.merge(rfm[["Segment"]], left_on="CustomerID", right_index=True)
        .assign(month=lambda d: d["InvoiceDate"].dt.to_period("M").dt.to_timestamp())
        .groupby(["month", "Segment"])["Revenue"].sum().reset_index()
    )
    monthly.to_parquet(ARTIFACTS / "monthly_revenue.parquet", index=False)

    keep = RFM_FEATURES + ["AvgOrderValue", "UniqueProducts", "TenureDays", "Country", "FirstPurchase",
                           "LastPurchase", "Segment", "R", "F", "M", "RFM_Score", "umap_x", "umap_y", "pca_x", "pca_y"]
    rfm[keep].reset_index().to_parquet(ARTIFACTS / "customers.parquet", index=False)
    log("done ✔")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-tuning", action="store_true")
    main(**vars(ap.parse_args()))
