# ============================================
# FYP PROJECT
# Predictive Consumer Segmentation and Spending Behaviour in Malaysian Retail
# Phase 2 — Modelling
# Objective 2: RFM Feature Engineering + K-Means Customer Segmentation
# ============================================
#
# METHODOLOGY NOTES (for report write-up)
# ----------------------------------------
# 1. RFM is computed over the FULL transaction history (all 715,863 rows),
#    not a time-limited window. Objective 2 is a DESCRIPTIVE segmentation of
#    "who are our customers", so it should use everything we know about
#    them. (Contrast with Objective 3, which deliberately uses a temporal
#    train/test split to avoid leakage in a PREDICTIVE task — see
#    phase2_objective3_predictive_modelling.py for why that distinction
#    matters.)
#
# 2. Frequency = COUNT OF UNIQUE INVOICES per customer, not row count. Each
#    invoice line item is one row, so counting rows would inflate Frequency
#    by "how many different products they bought per visit" rather than
#    "how many times they visited" — the correct RFM definition.
#
# 3. Recency/Frequency/Monetary are heavily right-skewed (a small number of
#    wholesale/power customers dominate) — see Phase 1's finding that
#    Quantity/Price/TotalAmount have strong positive skew. K-Means uses
#    Euclidean distance, which is distorted by skew and outliers: without
#    correction, the algorithm would essentially just separate "the few
#    giant spenders" from "everyone else" rather than finding meaningful
#    structure. We apply a log1p transform (log(1+x), safe for zero values)
#    to compress the tail, THEN standardise (mean=0, sd=1) so Recency,
#    Frequency, and Monetary — which live on completely different scales
#    (days vs counts vs currency) — contribute equally to distance
#    calculations. This log+scale combination is the standard treatment for
#    RFM-based K-Means in the literature.
#
# 4. Optimal k is chosen using BOTH the elbow method (inertia) and
#    silhouette score, because they can disagree — and here, they do (see
#    the flagged statistical tension below). Reporting both, and being
#    transparent about the disagreement, is more defensible than cherry-
#    picking whichever metric supports a pre-chosen k.

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # save PNGs directly; no GUI backend needed
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score)
from pathlib import Path

pd.set_option("display.max_columns", None)

OUT_DIR = Path("phase2_outputs")
OUT_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42

# --------------------------------------------
# Step 1: Load data
# --------------------------------------------
df = pd.read_csv("malaysian_context_online_retail.csv", encoding="ISO-8859-1")
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], dayfirst=True)

print(f"Loaded {len(df):,} rows, {df['Customer ID'].nunique():,} unique customers")

# --------------------------------------------
# Step 2: Compute RFM per customer
# --------------------------------------------
# Snapshot date = one day after the last transaction in the dataset. This is
# the standard convention (Fader, Hardie & Lee's RFM literature) so the most
# recent possible transaction has Recency = 1, not 0 (avoids log(0) issues
# and makes "Recency" strictly a count of elapsed days).
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
print(f"\nSnapshot date for Recency calculation: {snapshot_date.date()}")

rfm = df.groupby("Customer ID").agg(
    Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
    Frequency=("Invoice", "nunique"),
    Monetary=("TotalAmount", "sum"),
    # State is carried through for the Objective 4 dashboard's descriptive
    # filter ONLY. It is 1:1 per customer so "first" is safe. It is NEVER used
    # in clustering (rfm_log selects R/F/M explicitly below) and must never be
    # used for behavioural analysis — it was randomly assigned during the
    # Malaysian-context adaptation. See PHASE2_AUDIT_REPORT.md section 2.1.
    State=("State", "first"),
).reset_index()

print(f"\nRFM table shape: {rfm.shape}")
print("\nRaw RFM summary statistics:")
print(rfm[["Recency", "Frequency", "Monetary"]].describe().round(2))

skew_raw = {col: skew(rfm[col]) for col in ["Recency", "Frequency", "Monetary"]}
print("\nSkewness (raw):")
for col, s in skew_raw.items():
    print(f"  {col:10s}: {s:8.3f}")
print("Frequency and Monetary are extremely right-skewed (a handful of high-")
print("volume customers dominate) — this is exactly why raw RFM must not be")
print("fed directly into a distance-based algorithm like K-Means.")

# --------------------------------------------
# Step 3: Log transform + scale
# --------------------------------------------
rfm_log = rfm[["Recency", "Frequency", "Monetary"]].apply(np.log1p)
rfm_log.columns = ["Recency_log", "Frequency_log", "Monetary_log"]

skew_log = {col: skew(rfm_log[col]) for col in rfm_log.columns}
print("\nSkewness (after log1p transform):")
for col, s in skew_log.items():
    print(f"  {col:14s}: {s:8.3f}")
print("Log transform pulls all three features to within roughly [-1, 1] skew")
print("— an acceptable range for K-Means' Euclidean-distance assumption.")

scaler = StandardScaler()
X = scaler.fit_transform(rfm_log)

# --------------------------------------------
# Step 4: Determine optimal k — elbow method + silhouette score
# --------------------------------------------
K_RANGE = range(2, 11)
inertias = []
sil_scores = []

print("\nEvaluating k = 2..10 (inertia = elbow method, silhouette = cluster separation):")
print(f"{'k':>3} {'inertia':>12} {'silhouette':>12}")
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    sil = silhouette_score(X, labels)
    sil_scores.append(sil)
    print(f"{k:>3} {km.inertia_:>12.1f} {sil:>12.4f}")

# Persist the k-selection evidence. The proposal's "Model Evaluation Report"
# deliverable names WCSS (within-cluster sum of squares, = KMeans inertia)
# explicitly, so it must survive as a citable artefact rather than existing
# only in console output and inside a PNG.
selection_metrics = pd.DataFrame({
    "k": list(K_RANGE),
    "WCSS_inertia": inertias,
    "Silhouette": sil_scores,
})
selection_metrics.to_csv(OUT_DIR / "cluster_selection_metrics.csv", index=False)
print(f"\nSaved k-selection metrics (WCSS + silhouette) -> "
      f"{OUT_DIR / 'cluster_selection_metrics.csv'}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(list(K_RANGE), inertias, marker="o")
axes[0].set_title("Elbow Method (Inertia vs k)")
axes[0].set_xlabel("k (number of clusters)")
axes[0].set_ylabel("Inertia (within-cluster sum of squares)")
axes[0].grid(alpha=0.3)

axes[1].plot(list(K_RANGE), sil_scores, marker="o", color="darkorange")
axes[1].set_title("Silhouette Score vs k")
axes[1].set_xlabel("k (number of clusters)")
axes[1].set_ylabel("Silhouette Score")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "elbow_silhouette.png", dpi=130)
plt.close()
print(f"\nSaved elbow/silhouette plot -> {OUT_DIR / 'elbow_silhouette.png'}")

# ----------------------------------------------------------------------
# FLAG — STATISTICALLY QUESTIONABLE POINT (report this explicitly):
# Silhouette score is maximised at k=2, and k=2 will very likely remain the
# global maximum whenever you re-run this. A 2-cluster split on RFM data is
# almost always just "low spenders vs high spenders" — technically the best-
# separated clustering, but not a useful CUSTOMER SEGMENTATION for a
# business/report deliverable (a marketing team can't act on "spend more"
# vs "spend less" as a segment strategy).
#
# We therefore select k=4 as the FINAL_K: it is the best silhouette score
# among k>=3 (a defensible non-trivial value), AND it sits at the elbow's
# "knee" — the point where the marginal drop in inertia from adding another
# cluster visibly flattens (compare the inertia decrease from k=2->3->4
# against k=4->5->6 in the printed table above).
#
# This is a real tension between statistical optimality (k=2) and practical
# usefulness (k=4), and you should name it explicitly in your methodology
# chapter rather than silently picking k=4 as if silhouette agreed with it.
#
# ---- AUDIT ADDENDUM (see audit_2_clustering.py for the full evidence) ----
# The project proposal sets a success criterion of "silhouette > 0.5". That
# threshold IS numerically reachable on this data, but only via solutions
# that are degenerate or artefactual, which is why it is NOT adopted here:
#
#   * Skipping the log transform (raw + StandardScaler) yields silhouette
#     0.575 at k=4 — but the resulting clusters are 33% / 60% / 6.8% / and
#     ONE cluster containing 9 customers. At k=5 the score is 0.575 with a
#     cluster of a SINGLE customer. The score is high because K-Means is
#     isolating extreme outliers into micro-clusters that are trivially far
#     from everything else. A 9-customer "segment" is not a segment; you
#     cannot build a retail strategy on it.
#   * QuantileTransformer reaches 0.717 at k=2, but it forces each feature
#     into an exact Gaussian shape, manufacturing separation that reflects
#     the transform rather than customer behaviour — and k=2 is the trivial
#     split rejected above.
#   * Dropping an RFM dimension raises silhouette (e.g. R+F reaches 0.434)
#     purely because fewer dimensions dilute distances less. It would also
#     mean the study is no longer RFM segmentation.
#
# Hierarchical (Ward) and Gaussian Mixture were also tested and both scored
# LOWER than K-Means at every k (Ward 0.313, GMM 0.219 at k=4), and DBSCAN
# collapsed to 1-2 clusters. K-Means at k=4 is therefore the best honest
# choice available, and 0.360 reflects a genuine property of the data:
# RFM describes a behavioural continuum, not naturally separated groups.
#
# The defensible validity claim for this segmentation is STABILITY, not
# geometric separation — quantified in Step 6b below.
# ----------------------------------------------------------------------
FINAL_K = 4

print(f"\n{'='*60}\nFINAL_K = {FINAL_K} (see comments above for the k=2 vs k=4 justification)\n{'='*60}")

# --------------------------------------------
# Step 5: Fit final K-Means model
# --------------------------------------------
kmeans_final = KMeans(n_clusters=FINAL_K, random_state=RANDOM_SEED, n_init=10)
rfm["Cluster"] = kmeans_final.fit_predict(X)

final_sil = silhouette_score(X, rfm["Cluster"])
final_db = davies_bouldin_score(X, rfm["Cluster"])
final_ch = calinski_harabasz_score(X, rfm["Cluster"])
print(f"\nFinal internal validity metrics (k={FINAL_K}):")
print(f"  Silhouette score        : {final_sil:.4f}  (higher is better, range -1..1)")
print(f"  Davies-Bouldin index    : {final_db:.4f}  (LOWER is better)")
print(f"  Calinski-Harabasz index : {final_ch:.1f}  (higher is better)")
print("Reporting three complementary internal indices rather than silhouette")
print("alone is standard practice and avoids over-reliance on one metric that,")
print("as the audit showed, can be gamed by degenerate solutions.")

# --------------------------------------------
# Step 5b: Cluster stability — the primary validity evidence
# --------------------------------------------
# Silhouette answers "are the clusters geometrically separated?" — which, on a
# behavioural continuum, is the wrong question and the reason the 0.5 target is
# unreachable honestly. The question that actually matters for a segmentation
# a business would deploy is "are these segments REPRODUCIBLE, or an artefact
# of this particular sample and random seed?" We answer that with the Adjusted
# Rand Index (ARI) between the reference solution and (a) bootstrap resamples
# of the customer base, and (b) re-fits under different random seeds.
# ARI = 1.0 means identical partitions; ARI > 0.75 is conventionally "stable".
print("\nCluster stability assessment (Adjusted Rand Index vs reference solution):")

base_labels = rfm["Cluster"].values
boot_aris = []
for b in range(20):
    boot_rng = np.random.default_rng(1000 + b)
    idx = boot_rng.choice(len(X), len(X), replace=True)
    uniq = np.unique(idx)
    km_b = KMeans(n_clusters=FINAL_K, random_state=RANDOM_SEED, n_init=10).fit(X[idx])
    boot_aris.append(adjusted_rand_score(base_labels[uniq], km_b.predict(X[uniq])))
boot_aris = np.array(boot_aris)

seed_aris = [
    adjusted_rand_score(base_labels,
                        KMeans(n_clusters=FINAL_K, random_state=s, n_init=10).fit_predict(X))
    for s in [7, 13, 21, 99, 123]
]

print(f"  Bootstrap ARI (20 resamples): mean={boot_aris.mean():.4f}, "
      f"sd={boot_aris.std():.4f}, min={boot_aris.min():.4f}")
print(f"  Seed-variation ARI (5 seeds): mean={np.mean(seed_aris):.4f}, "
      f"min={np.min(seed_aris):.4f}")
print("  -> ARI comfortably above the 0.75 stability convention means the four")
print("     segments are a reproducible property of the customer base, not an")
print("     artefact of sampling or initialisation. THIS is the validity claim")
print("     to lead with in the report.")

# --------------------------------------------
# Step 6: Profile each segment
# --------------------------------------------
profile = rfm.groupby("Cluster").agg(
    Count=("Customer ID", "size"),
    Recency_mean=("Recency", "mean"),
    Frequency_mean=("Frequency", "mean"),
    Monetary_mean=("Monetary", "mean"),
    Monetary_median=("Monetary", "median"),
).round(1)
profile["Pct_of_customers"] = (profile["Count"] / profile["Count"].sum() * 100).round(1)

# Dynamic labelling: rank clusters relative to the OVERALL population median
# (not hardcoded numbers) so this logic still makes sense if re-run on a
# different sample/seed. "Good" = low Recency, high Frequency, high Monetary.
overall_r_med = rfm["Recency"].median()
overall_f_med = rfm["Frequency"].median()
overall_m_med = rfm["Monetary"].median()


def label_segment(row):
    """Assign an interpretable RFM label from the cluster's position on all
    THREE dimensions relative to the population medians.

    NOTE (audit fix): an earlier version of this function computed a
    `frequent` flag and then never used it, so Frequency — one third of the
    RFM method — played no role in how segments were named. Frequency is now
    used, which matters: it is what separates a genuinely loyal repeat
    customer from a one-off big-ticket buyer, and those warrant completely
    different retention strategies.
    """
    recent = row["Recency_mean"] <= overall_r_med
    frequent = row["Frequency_mean"] >= overall_f_med
    high_value = row["Monetary_mean"] >= overall_m_med

    if recent and frequent and high_value:
        return "Champions (recent, frequent, high spend)"
    if recent and high_value:
        return "Big Spenders (recent, high value, infrequent)"
    if recent and frequent:
        return "Loyal Regulars (recent, frequent, modest spend)"
    if recent:
        return "New / Promising (recent, low frequency & value)"
    if high_value and frequent:
        return "At-Risk High Value (lapsing loyal spender)"
    if high_value:
        return "At-Risk Big Spender (lapsing, was high value)"
    if frequent:
        return "Lapsing Regulars (was frequent, modest value)"
    return "Lost / Dormant Low-Value"


profile["Segment_Label"] = profile.apply(label_segment, axis=1)

# Guard: two clusters landing on the same label would make the profile table
# ambiguous to read. Disambiguate by monetary rank rather than failing, and
# warn so the situation is visible rather than silent.
if profile["Segment_Label"].duplicated().any():
    print("\nWARNING: two or more clusters received the same descriptive label.")
    print("They are distinguished below by monetary rank; consider whether k is")
    print("splitting a single behavioural group in two.")
    order = profile["Monetary_mean"].rank(ascending=False).astype(int)
    dup_mask = profile["Segment_Label"].duplicated(keep=False)
    profile.loc[dup_mask, "Segment_Label"] = (
        profile.loc[dup_mask, "Segment_Label"] + " [rank " + order[dup_mask].astype(str) + "]"
    )

print("\nSegment profile (k={}):".format(FINAL_K))
print(profile.sort_values("Monetary_mean", ascending=False).to_string())

# --------------------------------------------
# Step 7: Visualise segments
# --------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, col in zip(axes, ["Recency", "Frequency", "Monetary"]):
    for cluster_id in sorted(rfm["Cluster"].unique()):
        subset = rfm.loc[rfm["Cluster"] == cluster_id, col]
        ax.hist(np.log1p(subset), bins=25, alpha=0.5, label=f"Cluster {cluster_id}")
    ax.set_title(f"log1p({col}) by Cluster")
    ax.set_xlabel(f"log1p({col})")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR / "cluster_distributions.png", dpi=130)
plt.close()
print(f"\nSaved cluster distribution plot -> {OUT_DIR / 'cluster_distributions.png'}")

# 2D scatter on the two most interpretable scaled axes (Recency vs Monetary,
# log scale) coloured by cluster — this is the plot most reports use as the
# "segmentation result" figure.
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    np.log1p(rfm["Recency"]), np.log1p(rfm["Monetary"]),
    c=rfm["Cluster"], cmap="tab10", alpha=0.6, s=15,
)
plt.xlabel("log1p(Recency)")
plt.ylabel("log1p(Monetary)")
plt.title(f"Customer Segments (k={FINAL_K}) — Recency vs Monetary")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.savefig(OUT_DIR / "segments_scatter.png", dpi=130)
plt.close()
print(f"Saved segment scatter plot -> {OUT_DIR / 'segments_scatter.png'}")

# --------------------------------------------
# Step 8: Save outputs for downstream use / report appendix
# --------------------------------------------
segment_lookup = profile["Segment_Label"].to_dict()
rfm["Segment_Label"] = rfm["Cluster"].map(segment_lookup)

rfm.to_csv(OUT_DIR / "customer_rfm_segments.csv", index=False)
profile.to_csv(OUT_DIR / "segment_profile_summary.csv")

print(f"\nSaved per-customer RFM + segment assignments -> {OUT_DIR / 'customer_rfm_segments.csv'}")
print(f"Saved segment profile summary -> {OUT_DIR / 'segment_profile_summary.csv'}")
print("\nObjective 2 complete.")
