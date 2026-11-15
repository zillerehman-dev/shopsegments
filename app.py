"""ShopSegments — Customer Segmentation & Product Recommender dashboard.

Run locally:  streamlit run app.py
All data comes from ./artifacts (built by scripts/build_artifacts.py), so the app is light and fast.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ART = Path(__file__).parent / "artifacts"

st.set_page_config(page_title="ShopSegments", page_icon="🛍️", layout="wide")

SEGMENTS = {
    "Champions": ("🏆", "#2E7D32"),
    "Loyal": ("💙", "#1565C0"),
    "New / Promising": ("🌱", "#00897B"),
    "At Risk": ("⚠️", "#EF6C00"),
    "Lost": ("💤", "#9E9E9E"),
}
SEG_ORDER = list(SEGMENTS)
SEG_COLORS = {k: v[1] for k, v in SEGMENTS.items()}


# --------------------------------------------------------------------------- #
# Data loading (cached)
# --------------------------------------------------------------------------- #
@st.cache_data
def load():
    d = {
        "customers": pd.read_parquet(ART / "customers.parquet"),
        "catalog": pd.read_parquet(ART / "catalog.parquet"),
        "recs": pd.read_parquet(ART / "customer_recs.parquet"),
        "history": pd.read_parquet(ART / "customer_history.parquet"),
        "also": pd.read_parquet(ART / "also_bought.parquet"),
        "rules": pd.read_parquet(ART / "basket_rules.parquet"),
        "movement": pd.read_parquet(ART / "segment_movement.parquet"),
        "monthly": pd.read_parquet(ART / "monthly_revenue.parquet"),
        "k_sel": pd.read_csv(ART / "k_selection.csv"),
        "algos": pd.read_csv(ART / "clustering_comparison.csv"),
        "profiles": pd.read_json(ART / "segment_profiles.json", orient="index"),
        "metrics": json.loads((ART / "metrics.json").read_text()),
        "popular": json.loads((ART / "popular.json").read_text()),
        "model": json.loads((ART / "segment_model.json").read_text()),
    }
    d["names"] = d["catalog"].set_index("StockCode")["Description"].to_dict()
    d["prices"] = d["catalog"].set_index("StockCode")["median_price"].to_dict()
    return d


D = load()
cust = D["customers"]
names, prices = D["names"], D["prices"]


def product_name(code) -> str:
    return names.get(code, str(code)).title() if code is not None else "—"


def money(x: float) -> str:
    return f"£{x:,.0f}" if abs(x) >= 100 else f"£{x:,.2f}"


def predict_segment(recency: float, frequency: float, monetary: float) -> tuple[str, np.ndarray]:
    """Same maths as the trained pipeline: log1p -> standardise -> nearest K-Means centroid."""
    m = D["model"]
    x = (np.log1p([max(recency, 0), max(frequency, 0), max(monetary, 0)]) - np.array(m["mean"])) / np.array(m["scale"])
    dist = np.linalg.norm(np.array(m["centroids"]) - x, axis=1)
    return m["names"][int(np.argmin(dist))], dist


def product_table(codes, extra: dict | None = None) -> pd.DataFrame:
    df = pd.DataFrame({"#": range(1, len(codes) + 1), "Product": [product_name(c) for c in codes],
                       "Typical price": [money(prices.get(c, np.nan)) if c in prices else "—" for c in codes],
                       "Code": codes})
    for k, v in (extra or {}).items():
        df[k] = v
    return df


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
m = D["metrics"]
st.title("🛍️ ShopSegments")
st.caption(
    f"Customer segmentation & product recommendations for a UK online gift retailer · "
    f"{m['n_customers']:,} customers · {m['n_products']:,} products · {m['n_invoices']:,} orders · "
    f"{m['date_range'][0]} → {m['date_range'][1]}"
)

tabs = st.tabs([
    "📊 Overview", "🗺️ Segment map", "🧾 Segment profiles", "🎯 Recommend for a customer",
    "🛒 Also bought", "🧺 Basket rules", "🔀 Segment movement", "🧮 Classify a customer", "📐 Method & metrics",
])

# --------------------------------------------------------------------------- #
# 1. Overview
# --------------------------------------------------------------------------- #
with tabs[0]:
    prof = D["profiles"]
    c = st.columns(4)
    c[0].metric("Customers", f"{len(cust):,}")
    c[1].metric("Total revenue", money(cust["Monetary"].sum()))
    c[2].metric("Champions' share of revenue", f"{prof.loc['Champions', 'share_revenue']:.0%}",
                help="Share of all revenue coming from the Champions segment")
    c[3].metric("Customers at risk", f"{prof.loc['At Risk', 'customers']:,}",
                help="Used to buy regularly, but haven't ordered for ~3 months")

    st.markdown("#### Who are our customers — and who brings in the money?")
    share = pd.concat([
        prof["share_customers"].rename("Share of customers"),
        prof["share_revenue"].rename("Share of revenue"),
    ], axis=1).reset_index(names="Segment").melt("Segment", var_name="Measure", value_name="Share")
    fig = px.bar(share, x="Share", y="Measure", color="Segment", orientation="h",
                 color_discrete_map=SEG_COLORS, category_orders={"Segment": SEG_ORDER}, text_auto=".0%")
    fig.update_layout(height=260, xaxis_tickformat=".0%", legend_title=None, margin=dict(l=0, r=0, t=10, b=0),
                      yaxis_title=None)
    st.plotly_chart(fig, width="stretch")
    st.info(
        f"**Takeaway:** {prof.loc['Champions', 'share_customers']:.0%} of customers (Champions) generate "
        f"{prof.loc['Champions', 'share_revenue']:.0%} of revenue, while {prof.loc['Lost', 'share_customers']:.0%} "
        f"(Lost) generate only {prof.loc['Lost', 'share_revenue']:.0%}. Sending everyone the same promotion wastes "
        f"budget on the second group and under-serves the first."
    )

    st.markdown("#### Monthly revenue by (current) segment")
    mon = D["monthly"]
    fig = px.area(mon, x="month", y="Revenue", color="Segment", color_discrete_map=SEG_COLORS,
                  category_orders={"Segment": SEG_ORDER})
    fig.update_layout(height=340, legend_title=None, margin=dict(l=0, r=0, t=10, b=0), yaxis_tickprefix="£")
    st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------- #
# 2. Segment map
# --------------------------------------------------------------------------- #
with tabs[1]:
    st.markdown("#### Every dot is a customer — customers close together behave alike")
    l, r = st.columns([1, 3])
    with l:
        proj = st.radio("Projection", ["UMAP", "PCA"], horizontal=True,
                        help="UMAP keeps local neighbourhoods; PCA is a linear view of the same data")
        countries = ["All countries"] + cust["Country"].value_counts().index.tolist()
        country = st.selectbox("Country", countries)
        segs = st.multiselect("Segments", SEG_ORDER, default=SEG_ORDER)
        st.caption("Clusters were found on log-scaled Recency, Frequency and Monetary value. "
                   "Hover over a dot to see the customer.")
    view = cust[cust["Segment"].isin(segs)]
    if country != "All countries":
        view = view[view["Country"] == country]
    x, y = ("umap_x", "umap_y") if proj == "UMAP" else ("pca_x", "pca_y")
    fig = px.scatter(
        view, x=x, y=y, color="Segment", color_discrete_map=SEG_COLORS, category_orders={"Segment": SEG_ORDER},
        hover_name="CustomerID",
        hover_data={"Recency": True, "Frequency": True, "Monetary": ":,.0f", "Country": True, x: False, y: False},
        opacity=0.75,
    )
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(height=600, legend_title=None, xaxis_title=None, yaxis_title=None,
                      xaxis_showticklabels=False, yaxis_showticklabels=False, margin=dict(l=0, r=0, t=10, b=0))
    with r:
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Showing {len(view):,} customers.")

    st.markdown("#### What separates the segments (median values)")
    med = cust.groupby("Segment")[["Recency", "Frequency", "Monetary"]].median().loc[SEG_ORDER]
    med.columns = ["Days since last order", "Number of orders", "Total spend (£)"]
    st.dataframe(med.style.format("{:,.0f}"), width="stretch")

# --------------------------------------------------------------------------- #
# 3. Segment profiles
# --------------------------------------------------------------------------- #
with tabs[2]:
    st.markdown("#### Segment profile cards")
    prof = D["profiles"]
    for i in range(0, len(SEG_ORDER), 2):
        cols = st.columns(2)
        for col, seg in zip(cols, SEG_ORDER[i:i + 2]):
            p = prof.loc[seg]
            emoji, color = SEGMENTS[seg]
            with col.container(border=True):
                st.markdown(f"<h3 style='color:{color};margin-bottom:0'>{emoji} {seg}</h3>", unsafe_allow_html=True)
                st.caption(p["description"])
                a, b, c_, d_ = st.columns(4)
                a.metric("Customers", f"{p['customers']:,}", help=f"{p['share_customers']:.0%} of all customers")
                b.metric("Avg. spend", money(p["avg_spend"]))
                c_.metric("Avg. orders", f"{p['avg_orders']:.1f}")
                d_.metric("Days since order", f"{p['avg_recency_days']:.0f}")
                st.markdown(f"**{p['share_customers']:.0%} of customers → {p['share_revenue']:.0%} of revenue** · **Avg. order value:** {money(p['avg_order_value'])}")
                st.markdown("**Top products:** " + ", ".join(t.title() for t in p["top_products"]))
                st.success(f"**Suggested action:** {p['action']}")

    st.markdown("#### Export a segment for a campaign")
    seg_pick = st.selectbox("Segment", SEG_ORDER, key="export_seg")
    out = cust[cust["Segment"] == seg_pick][["CustomerID", "Country", "Recency", "Frequency", "Monetary",
                                             "AvgOrderValue", "RFM_Score"]].sort_values("Monetary", ascending=False)
    st.dataframe(out.head(200), width="stretch", hide_index=True,
                 column_config={"Monetary": st.column_config.NumberColumn("Total spend (£)", format="%.0f"),
                                "AvgOrderValue": st.column_config.NumberColumn("Avg order (£)", format="%.0f"),
                                "Recency": "Days since order", "Frequency": "Orders"})
    st.download_button(f"⬇️ Download all {len(out):,} '{seg_pick}' customers (CSV)", out.to_csv(index=False),
                       file_name=f"segment_{seg_pick.replace(' / ', '_').lower()}.csv", mime="text/csv")

# --------------------------------------------------------------------------- #
# 4. Personalised recommendations
# --------------------------------------------------------------------------- #
with tabs[3]:
    mode = st.radio("Customer type", ["Existing customer", "New customer (no purchase history)"], horizontal=True)
    if mode.startswith("Existing"):
        l, r = st.columns([1, 2])
        with l:
            seg_filter = st.selectbox("Pick from segment", ["Any"] + SEG_ORDER)
            pool = cust if seg_filter == "Any" else cust[cust["Segment"] == seg_filter]
            pool = pool.sort_values("Monetary", ascending=False)
            cid = st.selectbox("Customer ID (sorted by spend)", pool["CustomerID"].tolist())
            row = cust.set_index("CustomerID").loc[cid]
            emoji, color = SEGMENTS[row["Segment"]]
            with st.container(border=True):
                st.markdown(f"<h4 style='color:{color};margin:0'>{emoji} {row['Segment']}</h4>", unsafe_allow_html=True)
                st.write(f"📍 {row['Country']}")
                st.write(f"🧾 {row['Frequency']} orders · {money(row['Monetary'])} total")
                st.write(f"🕒 Last order {row['Recency']} day{'s' if row['Recency'] != 1 else ''} ago")
                st.caption(D["profiles"].loc[row["Segment"], "action"])
        with r:
            recs = D["recs"][D["recs"]["CustomerID"] == cid].sort_values("rank")
            st.markdown("#### 🎯 Top 10 products to show this customer")
            tbl = product_table(recs["StockCode"].tolist(),
                                {"Because they bought": [product_name(b) for b in recs["because_StockCode"]]})
            st.dataframe(tbl.drop(columns="Code"), width="stretch", hide_index=True)
            st.caption("Already-purchased products are excluded. Model: hybrid of item-based collaborative "
                       "filtering and ALS matrix factorisation.")
            with st.expander("What this customer has bought most often"):
                h = D["history"][D["history"]["CustomerID"] == cid]
                st.dataframe(pd.DataFrame({
                    "Product": [product_name(c) for c in h["StockCode"]],
                    "Orders": h["orders"].values, "Units": h["units"].values,
                    "Spend": [money(s) for s in h["spend"]],
                    "Last bought": pd.to_datetime(h["last"]).dt.date.values,
                }), width="stretch", hide_index=True)
    else:
        st.markdown("New customers have no history yet, so we fall back to **popularity** (the *cold-start* strategy).")
        pop = D["popular"]
        choice = st.radio("Popularity list", ["Trending (last 90 days)", "All-time best sellers", "Best sellers in a country"],
                          horizontal=True)
        if choice.startswith("Trending"):
            codes = pop["trending_90d"][:10]
        elif choice.startswith("All-time"):
            codes = pop["all_time"][:10]
        else:
            cty = st.selectbox("Country", sorted(pop["by_country"]), index=sorted(pop["by_country"]).index("United Kingdom"))
            codes = pop["by_country"][cty][:10]
        st.dataframe(product_table(codes).drop(columns="Code"), width="stretch", hide_index=True)
        cs = m["cold_start"]
        st.caption(f"Offline check on {cs['popularity']['users_evaluated']} brand-new customers: popularity gets "
                   f"Precision@10 = {cs['popularity']['Precision@10']:.3f} vs {cs['random']['Precision@10']:.3f} "
                   f"for random products ({cs['popularity']['Precision@10'] / cs['random']['Precision@10']:.0f}× better).")

# --------------------------------------------------------------------------- #
# 5. Also bought
# --------------------------------------------------------------------------- #
with tabs[4]:
    st.markdown("#### 🛒 Customers who bought this also bought…")
    also = D["also"]
    options = D["catalog"][D["catalog"]["StockCode"].isin(also["StockCode"].unique())]
    labels = {r.StockCode: f"{r.Description.title()} ({r.StockCode})" for r in options.itertuples()}
    code = st.selectbox("Choose a product (sorted by popularity)", list(labels), format_func=labels.get)
    sel = D["catalog"].set_index("StockCode").loc[code]
    st.caption(f"{sel['n_customers']:,} customers bought this · typical price {money(sel['median_price'])}")
    a = also[also["StockCode"] == code].sort_values("rank")
    t = product_table(a["rec_StockCode"].tolist(), {
        "Bought by % of its buyers": [f"{s:.0%}" for s in a["share_of_buyers"]],
        "Shared customers": a["co_buyers"].values,
        "Similarity": a["similarity"].round(2).values,
    }).drop(columns="Code")
    st.dataframe(t, width="stretch", hide_index=True)
    st.caption("Similarity = cosine similarity between the sets of customers who bought each product "
               "(only pairs with ≥5 shared customers are shown).")

# --------------------------------------------------------------------------- #
# 6. Basket rules (bonus)
# --------------------------------------------------------------------------- #
with tabs[5]:
    st.markdown("#### 🧺 Market basket analysis — products bought in the *same order*")
    st.caption("FP-Growth association rules on invoices. **Confidence** = how often the right side is in the basket "
               "when the left side is. **Lift** = how many times more likely than chance.")
    rules = D["rules"]
    q = st.text_input("Search a product (e.g. 'teacup', 'lunch bag', 'christmas')")
    min_conf = st.slider("Minimum confidence", 0.3, 1.0, 0.5, 0.05)
    v = rules[rules["confidence"] >= min_conf]
    if q:
        mask = v["if_bought"].str.contains(q, case=False) | v["then_likely"].str.contains(q, case=False)
        v = v[mask]
    st.dataframe(
        v.assign(**{c: v[c].str.title() for c in ["if_bought", "then_likely"]}).rename(
            columns={"if_bought": "If basket has…", "then_likely": "…it likely also has", "support": "Support",
                     "confidence": "Confidence", "lift": "Lift"}),
        width="stretch", hide_index=True,
        column_config={"Support": st.column_config.NumberColumn(format="%.3f"),
                       "Confidence": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
                       "Lift": st.column_config.NumberColumn(format="%.1f")},
    )
    st.caption(f"{len(v)} rules shown. Use them for bundles, 'complete the set' prompts and shelf/page placement.")

# --------------------------------------------------------------------------- #
# 7. Segment movement (bonus)
# --------------------------------------------------------------------------- #
with tabs[6]:
    st.markdown("#### 🔀 How customers moved between segments (9 Jun 2011 → 9 Dec 2011)")
    mv = D["movement"]
    src_nodes = ["Not yet a customer"] + SEG_ORDER
    nodes = [f"{s} (Jun)" for s in src_nodes] + [f"{s} (Dec)" for s in SEG_ORDER]
    idx = {n: i for i, n in enumerate(nodes)}
    node_colors = ["#BDBDBD"] + [SEG_COLORS[s] for s in SEG_ORDER] + [SEG_COLORS[s] for s in SEG_ORDER]

    def rgba(hex_, a=0.35):
        h = hex_.lstrip("#")
        return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"

    fig = go.Figure(go.Sankey(
        node=dict(label=nodes, color=node_colors, pad=18, thickness=16),
        link=dict(source=[idx[f"{f} (Jun)"] for f in mv["from"]], target=[idx[f"{t} (Dec)"] for t in mv["to"]],
                  value=mv["customers"], color=[rgba(SEG_COLORS.get(t, "#BDBDBD")) for t in mv["to"]]),
    ))
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=10), font_size=13)
    st.plotly_chart(fig, width="stretch")
    pivot = mv.pivot_table(index="from", columns="to", values="customers", fill_value=0)
    pivot = pivot.reindex(index=[s for s in src_nodes if s in pivot.index], columns=SEG_ORDER, fill_value=0)
    st.dataframe(pivot, width="stretch")
    lost_to = mv[(mv["from"] == "Loyal") & (mv["to"] == "At Risk")]["customers"].sum()
    up = mv[(mv["from"] == "Loyal") & (mv["to"] == "Champions")]["customers"].sum()
    st.caption(f"Each customer is scored at both dates with the same fitted model. E.g. of June's Loyal customers, "
               f"{up} became Champions while {lost_to} slipped to At Risk — that slip is what a win-back flow should catch.")

# --------------------------------------------------------------------------- #
# 8. Classify a customer (what-if)
# --------------------------------------------------------------------------- #
with tabs[7]:
    st.markdown("#### 🧮 Which segment would this customer fall into?")
    st.caption("Enter a customer's behaviour — useful for scoring customers from another system or for 'what if they "
               "placed one more order?' questions.")
    c1, c2, c3 = st.columns(3)
    rec = c1.number_input("Days since last order", 0, 1000, 30)
    freq = c2.number_input("Number of orders", 1, 500, 3)
    mon = c3.number_input("Total spend (£)", 1.0, 300000.0, 800.0, step=50.0)
    seg, dist = predict_segment(rec, freq, mon)
    emoji, color = SEGMENTS[seg]
    st.markdown(f"<h2 style='color:{color}'>{emoji} {seg}</h2>", unsafe_allow_html=True)
    st.write(D["profiles"].loc[seg, "description"])
    st.success(f"**Suggested action:** {D['profiles'].loc[seg, 'action']}")
    conf = pd.DataFrame({"Segment": D["model"]["names"], "Distance to segment centre": dist}).sort_values(
        "Distance to segment centre")
    st.dataframe(conf, hide_index=True, width="stretch",
                 column_config={"Distance to segment centre": st.column_config.NumberColumn(format="%.2f")})

# --------------------------------------------------------------------------- #
# 9. Method & metrics
# --------------------------------------------------------------------------- #
with tabs[8]:
    st.markdown("#### How many segments? (elbow + silhouette + business sense)")
    ks = D["k_sel"]
    c1, c2 = st.columns(2)
    f1 = px.line(ks, x="k", y="inertia", markers=True, title="Elbow: within-cluster inertia")
    f1.add_vline(x=5, line_dash="dash", line_color="#EF6C00")
    f2 = px.line(ks, x="k", y="silhouette", markers=True, title="Silhouette score (higher = better separated)")
    f2.add_vline(x=5, line_dash="dash", line_color="#EF6C00")
    for f in (f1, f2):
        f.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0))
    c1.plotly_chart(f1, width="stretch")
    c2.plotly_chart(f2, width="stretch")
    st.markdown(
        f"- **k = 2** has the best silhouette but only splits *active* vs *inactive* — not actionable.\n"
        f"- The elbow flattens around **k = 4–6**; silhouette stays almost flat from 3 to 6 (0.34 → 0.31).\n"
        f"- At **k = 4**, active *Loyal* buyers and lapsing *At Risk* buyers sit in one cluster (median 52 days since last "
        f"order). **k = 5** (silhouette {m['silhouette_k5']:.3f}, only 0.02 lower) splits them — and they need opposite "
        f"campaigns (upsell vs. win-back). k = 6+ just splits groups the marketing team would treat the same way."
    )
    st.markdown("##### Clustering algorithms compared (same scaled RFM features)")
    st.dataframe(D["algos"].round(3), hide_index=True, width="stretch")
    st.caption("DBSCAN finds one dense blob plus noise (RFM has no density gaps); GMM gives overlapping clusters. "
               "K-Means gives compact, explainable groups.")

    st.markdown("#### Recommender — offline evaluation")
    st.markdown(
        f"Temporal hold-out: trained on orders before **{m['split_date']}**, tested on what returning customers "
        f"bought afterwards that they had **never bought before**. Hyper-parameters were tuned on an earlier "
        f"validation window; the test window was used once."
    )
    tm = pd.DataFrame(m["test"]).set_index("model")
    st.dataframe(tm.style.format({c: "{:.3f}" for c in tm.columns if c != "users_evaluated"})
                 .highlight_max(subset=["Precision@10", "Recall@10", "NDCG@10"], color="#c8e6c9"),
                 width="stretch")
    best = tm["Precision@10"].max()
    st.caption(f"Best model beats the popularity baseline by {best / tm.loc['Popularity', 'Precision@10']:.1f}× on "
               f"Precision@10. Coverage = share of the catalogue that appears in anyone's top 10.")
    st.markdown("##### Precision@10 by segment (best model)")
    bs = pd.DataFrame(m["by_segment"]).set_index("Segment").reindex(SEG_ORDER).dropna(how="all")
    fig = px.bar(bs.reset_index(), x="Segment", y="Precision@10", color="Segment", color_discrete_map=SEG_COLORS,
                 text_auto=".3f", hover_data=["users_evaluated"])
    fig.update_layout(height=320, showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")

st.divider()
st.caption("Data: Online Retail dataset (UCI Machine Learning Repository, Chen et al.). "
           "Built with scikit-learn, UMAP, implicit (ALS), mlxtend and Streamlit.")
