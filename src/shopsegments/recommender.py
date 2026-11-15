"""Product recommenders: popularity (cold start), item-based CF and ALS matrix factorisation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from .config import RANDOM_STATE, TOP_K


# --------------------------------------------------------------------------- #
# Interaction matrix
# --------------------------------------------------------------------------- #
class InteractionData:
    """Customer x product matrix (value = number of invoices containing the product)."""

    def __init__(self, tx: pd.DataFrame, min_item_buyers: int = 1):
        counts = tx.groupby(["CustomerID", "StockCode"])["InvoiceNo"].nunique().rename("n").reset_index()
        if min_item_buyers > 1:
            buyers = counts.groupby("StockCode")["CustomerID"].transform("size")
            counts = counts[buyers >= min_item_buyers]
        self.users = np.sort(counts["CustomerID"].unique())
        self.items = np.sort(counts["StockCode"].unique())
        self.user_index = {u: i for i, u in enumerate(self.users)}
        self.item_index = {it: i for i, it in enumerate(self.items)}
        rows = counts["CustomerID"].map(self.user_index).to_numpy()
        cols = counts["StockCode"].map(self.item_index).to_numpy()
        self.counts = sp.csr_matrix(
            (counts["n"].to_numpy(dtype=np.float32), (rows, cols)),
            shape=(len(self.users), len(self.items)),
        )
        self.binary = self.counts.copy()
        self.binary.data[:] = 1.0

    def seen(self, uidx: int) -> np.ndarray:
        return self.binary[uidx].indices


def _top_k_excluding(scores: np.ndarray, exclude: np.ndarray, k: int) -> np.ndarray:
    scores = scores.astype(np.float64, copy=True)
    scores[exclude] = -np.inf
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx[np.isfinite(scores[idx])]


# --------------------------------------------------------------------------- #
# Models — all share fit(data) / recommend(user_ids, k) -> list[list[StockCode]]
# --------------------------------------------------------------------------- #
class PopularityRecommender:
    """Most-bought products (by number of distinct buyers). Used for cold start."""

    name = "Popularity"

    def fit(self, data: InteractionData):
        self.data = data
        self.scores = np.asarray(data.binary.sum(axis=0)).ravel()
        self.ranking = np.argsort(-self.scores)
        return self

    def recommend_new(self, k: int = TOP_K, exclude_items=()) -> list[str]:
        excl = set(exclude_items)
        return [self.data.items[i] for i in self.ranking if self.data.items[i] not in excl][:k]

    def recommend(self, user_ids, k: int = TOP_K) -> list[list[str]]:
        out = []
        for u in user_ids:
            ui = self.data.user_index.get(u)
            seen = self.data.seen(ui) if ui is not None else np.array([], dtype=int)
            out.append([self.data.items[i] for i in _top_k_excluding(self.scores, seen, k)])
        return out


class ItemCFRecommender:
    """Item-based collaborative filtering with cosine similarity on the binary matrix.

    score(user, item) = sum over items j the user bought of sim(j, item)
    """

    name = "Item-based CF"

    def __init__(self, n_neighbours: int = 50):
        self.n_neighbours = n_neighbours

    def fit(self, data: InteractionData):
        self.data = data
        item_vecs = normalize(data.binary.T.tocsr(), norm="l2", axis=1)
        sim = (item_vecs @ item_vecs.T).toarray().astype(np.float32)
        np.fill_diagonal(sim, 0.0)
        # keep only the strongest neighbours per item (reduces noise)
        if self.n_neighbours and self.n_neighbours < sim.shape[0]:
            thresh = -np.partition(-sim, self.n_neighbours - 1, axis=1)[:, self.n_neighbours - 1 : self.n_neighbours]
            sim[sim < thresh] = 0.0
        self.sim = sim
        self.popularity = np.asarray(data.binary.sum(axis=0)).ravel()
        return self

    def user_scores(self, uidx: int) -> np.ndarray:
        return self.data.binary[uidx] @ self.sim  # (1, n_items) dense via sparse@dense

    def recommend(self, user_ids, k: int = TOP_K) -> list[list[str]]:
        out = []
        for u in user_ids:
            ui = self.data.user_index.get(u)
            if ui is None:
                scores, seen = self.popularity.astype(float), np.array([], dtype=int)
            else:
                scores = np.asarray(self.user_scores(ui)).ravel()
                # tiny popularity tie-breaker so users with few signals still get full lists
                scores = scores + 1e-6 * self.popularity / self.popularity.max()
                seen = self.data.seen(ui)
            out.append([self.data.items[i] for i in _top_k_excluding(scores, seen, k)])
        return out


class ALSRecommender:
    """Implicit-feedback matrix factorisation (Hu, Koren & Volinsky) via the `implicit` library."""

    name = "ALS (implicit)"

    def __init__(self, factors: int = 64, regularization: float = 0.05, alpha: float = 20.0, iterations: int = 20):
        self.params = dict(factors=factors, regularization=regularization, alpha=alpha, iterations=iterations)

    def fit(self, data: InteractionData):
        from implicit.cpu.als import AlternatingLeastSquares

        self.data = data
        # confidence grows with how many separate orders contained the item (log-damped)
        weighted = data.counts.copy()
        weighted.data = np.log1p(weighted.data)
        self.user_items = weighted.tocsr()
        self.model = AlternatingLeastSquares(random_state=RANDOM_STATE, num_threads=0, **self.params)
        self.model.fit(self.user_items, show_progress=False)
        self.popularity = np.asarray(data.binary.sum(axis=0)).ravel()
        return self

    def recommend(self, user_ids, k: int = TOP_K) -> list[list[str]]:
        known = [u for u in user_ids if u in self.data.user_index]
        recs = {}
        if known:
            uidx = np.array([self.data.user_index[u] for u in known])
            ids, _ = self.model.recommend(uidx, self.user_items[uidx], N=k, filter_already_liked_items=True)
            for u, row in zip(known, ids):
                recs[u] = [self.data.items[i] for i in row if i >= 0]
        pop = [self.data.items[i] for i in np.argsort(-self.popularity)[:k]]
        return [recs.get(u, pop) for u in user_ids]


class HybridRecommender:
    """Reciprocal-rank fusion of item-based CF and ALS candidate lists."""

    name = "Hybrid (ItemCF + ALS)"

    def __init__(self, itemcf: ItemCFRecommender, als: ALSRecommender, w_itemcf: float = 0.5,
                 n_candidates: int = 50, rrf_k: int = 20):
        self.itemcf, self.als = itemcf, als
        self.w, self.n_candidates, self.rrf_k = w_itemcf, n_candidates, rrf_k

    def fit(self, data: InteractionData):
        self.data = data
        self.itemcf.fit(data)
        self.als.fit(data)
        return self

    def recommend(self, user_ids, k: int = TOP_K) -> list[list[str]]:
        a = self.itemcf.recommend(user_ids, self.n_candidates)
        b = self.als.recommend(user_ids, self.n_candidates)
        out = []
        for la, lb in zip(a, b):
            score: dict[str, float] = {}
            for r, it in enumerate(la):
                score[it] = score.get(it, 0) + self.w / (self.rrf_k + r)
            for r, it in enumerate(lb):
                score[it] = score.get(it, 0) + (1 - self.w) / (self.rrf_k + r)
            out.append(sorted(score, key=score.get, reverse=True)[:k])
        return out


# --------------------------------------------------------------------------- #
# Offline evaluation
# --------------------------------------------------------------------------- #
def build_holdout(tx: pd.DataFrame, split_date: str):
    """Temporal split. Ground truth = products a returning customer bought after the split
    that they had NOT bought before (we evaluate discovery of new products)."""
    split = pd.Timestamp(split_date)
    train = tx[tx["InvoiceDate"] < split]
    test = tx[tx["InvoiceDate"] >= split]
    train_items = set(train["StockCode"].unique())
    seen = train.groupby("CustomerID")["StockCode"].apply(set)
    bought_after = test.groupby("CustomerID")["StockCode"].apply(set)

    returning, cold = {}, {}
    for u, items in bought_after.items():
        items = items & train_items  # can't recommend products that didn't exist yet
        if u in seen.index:
            new_items = items - seen[u]
            if new_items:
                returning[u] = new_items
        elif items:
            cold[u] = items
    return train, returning, cold


def ranking_metrics(recs: dict, truth: dict, k: int = TOP_K, n_catalog: int | None = None) -> dict:
    precisions, recalls, hits, ndcgs = [], [], [], []
    rec_items = set()
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    for u, rel in truth.items():
        r = recs[u][:k]
        rec_items.update(r)
        hit_vec = np.array([1.0 if i in rel else 0.0 for i in r] + [0.0] * (k - len(r)))
        n_hit = hit_vec.sum()
        precisions.append(n_hit / k)
        recalls.append(n_hit / len(rel))
        hits.append(float(n_hit > 0))
        idcg = discounts[: min(len(rel), k)].sum()
        ndcgs.append((hit_vec * discounts).sum() / idcg)
    out = {
        f"Precision@{k}": float(np.mean(precisions)),
        f"Recall@{k}": float(np.mean(recalls)),
        f"HitRate@{k}": float(np.mean(hits)),
        f"NDCG@{k}": float(np.mean(ndcgs)),
        "users_evaluated": len(truth),
    }
    if n_catalog:
        out["Coverage"] = len(rec_items) / n_catalog
    return out


def evaluate_models(models, train_tx: pd.DataFrame, truth: dict, k: int = TOP_K) -> pd.DataFrame:
    data = InteractionData(train_tx)
    users = list(truth)
    rows = []
    for m in models:
        m.fit(data)
        recs = dict(zip(users, m.recommend(users, k)))
        rows.append({"model": m.name, **ranking_metrics(recs, truth, k, n_catalog=len(data.items))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Helpers for the app
# --------------------------------------------------------------------------- #
def also_bought_table(data: InteractionData, top_n: int = 10, min_co_buyers: int = 5) -> pd.DataFrame:
    """'Customers who bought this also bought' — cosine similarity between product buyer sets,
    keeping only pairs with at least `min_co_buyers` shared customers."""
    B = data.binary.T.tocsr()  # items x users
    co = (B @ B.T).toarray().astype(np.float32)  # shared buyers
    n = np.diag(co).copy()
    sim = co / np.sqrt(np.outer(n, n) + 1e-9)
    np.fill_diagonal(sim, 0)
    sim[co < min_co_buyers] = 0
    rows = []
    for i in range(sim.shape[0]):
        nz = np.flatnonzero(sim[i])
        if len(nz) == 0:
            continue
        top = nz[np.argsort(-sim[i, nz])[:top_n]]
        for rank, j in enumerate(top, 1):
            rows.append((data.items[i], rank, data.items[j], float(sim[i, j]), int(co[i, j]), co[i, j] / n[i]))
    return pd.DataFrame(rows, columns=["StockCode", "rank", "rec_StockCode", "similarity", "co_buyers", "share_of_buyers"])


def explain_recommendations(data: InteractionData, item_sim: np.ndarray, user_id, rec_items) -> list[str | None]:
    """For each recommended item, the product the customer already bought that is most similar to it."""
    ui = data.user_index.get(user_id)
    if ui is None:
        return [None] * len(rec_items)
    seen = data.seen(ui)
    out = []
    for it in rec_items:
        j = data.item_index[it]
        s = item_sim[seen, j]
        out.append(data.items[seen[int(np.argmax(s))]] if len(s) and s.max() > 0 else None)
    return out
