# ============================================
# FYP PROJECT — Phase 2 AUDIT (2 of 3)
# Is silhouette 0.360 a genuine ceiling, or is there a legitimate improvement?
# ============================================
#
# The proposal's success criterion is "silhouette > 0.5". Rather than tuning
# until that number appears, this script asks the prior question FIRST:
#   does this data actually contain separated clusters at all?
# If it does not, then no amount of algorithm-swapping produces a HONEST 0.5,
# and the correct academic response is to report the ceiling and explain it.

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                   PowerTransformer, QuantileTransformer)
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score)

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

df = pd.read_csv("malaysian_context_online_retail.csv", encoding="ISO-8859-1")
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], dayfirst=True)

snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)
rfm = df.groupby("Customer ID").agg(
    Recency=("InvoiceDate", lambda x: (snapshot - x.max()).days),
    Frequency=("Invoice", "nunique"),
    Monetary=("TotalAmount", "sum"),
).reset_index()

R, F, M = rfm["Recency"], rfm["Frequency"], rfm["Monetary"]
raw = rfm[["Recency", "Frequency", "Monetary"]].values

# --------------------------------------------------------------------
print("=" * 70)
print("B1. CLUSTER TENDENCY — DOES STRUCTURE EXIST AT ALL? (Hopkins statistic)")
print("=" * 70)
print("Hopkins compares nearest-neighbour distances in the real data against")
print("those in uniformly random data over the same bounding box.")
print("  H ~ 0.50  -> data is indistinguishable from uniform noise (no clusters)")
print("  H ~ 0.75+ -> meaningful clustering tendency")
print("  H ~ 0.99  -> highly separated, well-defined clusters\n")


def hopkins_statistic(X, sample_frac=0.10, seed=RANDOM_SEED):
    """Hopkins statistic. Values near 1 indicate strong clustering tendency,
    values near 0.5 indicate the data is essentially uniformly distributed."""
    local_rng = np.random.default_rng(seed)
    n, d = X.shape
    m = int(sample_frac * n)

    # m real points sampled without replacement
    idx = local_rng.choice(n, m, replace=False)
    real_pts = X[idx]

    # m synthetic points uniform over the data's bounding box
    mins, maxs = X.min(axis=0), X.max(axis=0)
    uniform_pts = local_rng.uniform(mins, maxs, size=(m, d))

    # w = real->nearest real neighbour (exclude self via 2nd smallest)
    d_real = cdist(real_pts, X)
    w = np.sort(d_real, axis=1)[:, 1]

    # u = synthetic->nearest real neighbour
    d_unif = cdist(uniform_pts, X)
    u = np.sort(d_unif, axis=1)[:, 0]

    return u.sum() / (u.sum() + w.sum())


# Evaluate tendency on the SAME representation the model uses (log + standardise)
log_std = StandardScaler().fit_transform(np.log1p(raw))
h_log = np.mean([hopkins_statistic(log_std, seed=s) for s in range(5)])
h_raw = np.mean([hopkins_statistic(StandardScaler().fit_transform(raw), seed=s) for s in range(5)])
print(f"Hopkins (log1p + standardised, i.e. the modelling representation): {h_log:.4f}")
print(f"Hopkins (standardised only, no log):                              {h_raw:.4f}")
print("\nINTERPRETATION -- READ CAREFULLY, THIS IS EASY TO MISREAD:")
print("Hopkins is HIGH (~0.95), which means the data is strongly NON-UNIFORM.")
print("It does NOT mean the data contains well-separated clusters. Hopkins is")
print("known to be inflated for skewed, heavily-concentrated data: because RFM")
print("values pile up in one corner of the bounding box, uniformly-sampled")
print("comparison points land far from any real customer, driving H toward 1")
print("regardless of whether distinct groups exist.")
print("")
print("So Hopkins alone CANNOT justify either a high or a low silhouette. The")
print("evidence that the structure is a CONTINUUM rather than separated groups")
print("comes from sections B4 (all algorithms, including density-based DBSCAN,")
print("plateau at similar modest silhouette values) and the degeneracy check on")
print("the high-silhouette solutions. Do not cite Hopkins as proof of a ceiling.")

# --------------------------------------------------------------------
print("\n" + "=" * 70)
print("B2. SCALING / TRANSFORM SWEEP (does a different preprocessing help?)")
print("=" * 70)

transforms = {
    "Raw + StandardScaler":        StandardScaler().fit_transform(raw),
    "log1p + StandardScaler (current)": log_std,
    "log1p + MinMaxScaler":        MinMaxScaler().fit_transform(np.log1p(raw)),
    "log1p + RobustScaler":        RobustScaler().fit_transform(np.log1p(raw)),
    "PowerTransformer (Yeo-Johnson)": PowerTransformer().fit_transform(raw),
    "QuantileTransformer (normal)": QuantileTransformer(output_distribution="normal",
                                                        random_state=RANDOM_SEED).fit_transform(raw),
}

print(f"{'Transform':<36} {'k=3':>8} {'k=4':>8} {'k=5':>8} {'best k':>8}")
for tname, Xt in transforms.items():
    scores = {}
    for k in range(2, 8):
        labels = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10).fit_predict(Xt)
        scores[k] = silhouette_score(Xt, labels)
    best_k = max(scores, key=scores.get)
    print(f"{tname:<36} {scores[3]:>8.4f} {scores[4]:>8.4f} {scores[5]:>8.4f} "
          f"{f'k={best_k} ({scores[best_k]:.3f})':>8}")

print("\nNOTE ON QuantileTransformer: it forces each feature to an exact Gaussian")
print("shape, which can inflate silhouette by manufacturing separation that is an")
print("artefact of the transform, not a property of customers. If it 'wins' here,")
print("that is a warning sign, not a result -- flag rather than adopt.")

# --------------------------------------------------------------------
print("\n" + "=" * 70)
print("B3. FEATURE-SUBSET SWEEP (is one RFM dimension adding only noise?)")
print("=" * 70)
subsets = {
    "R + F + M (current)": ["Recency", "Frequency", "Monetary"],
    "R + M":               ["Recency", "Monetary"],
    "R + F":               ["Recency", "Frequency"],
    "F + M":               ["Frequency", "Monetary"],
}
print(f"{'Feature subset':<24} {'k=3':>8} {'k=4':>8} {'k=5':>8}")
for sname, cols in subsets.items():
    Xs = StandardScaler().fit_transform(np.log1p(rfm[cols].values))
    row = []
    for k in [3, 4, 5]:
        labels = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10).fit_predict(Xs)
        row.append(silhouette_score(Xs, labels))
    print(f"{sname:<24} {row[0]:>8.4f} {row[1]:>8.4f} {row[2]:>8.4f}")

print("\nCAUTION: dropping a dimension almost always RAISES silhouette (fewer")
print("dimensions = less distance dilution). That is a geometric artefact, not")
print("evidence of a better segmentation. Dropping Frequency or Monetary from an")
print("RFM study to chase a metric would also break the method's definition --")
print("it would no longer be RFM segmentation.")

# --------------------------------------------------------------------
print("\n" + "=" * 70)
print("B4. ALGORITHM COMPARISON (K-Means vs Hierarchical vs GMM vs DBSCAN)")
print("=" * 70)

X = log_std
print(f"{'Algorithm':<34} {'k':>4} {'Silhouette':>11} {'Davies-Bouldin':>15} {'Calinski-H':>12}")

for k in [3, 4, 5]:
    lab = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10).fit_predict(X)
    print(f"{'K-Means':<34} {k:>4} {silhouette_score(X, lab):>11.4f} "
          f"{davies_bouldin_score(X, lab):>15.4f} {calinski_harabasz_score(X, lab):>12.1f}")

for k in [3, 4, 5]:
    lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    print(f"{'Hierarchical (Ward)':<34} {k:>4} {silhouette_score(X, lab):>11.4f} "
          f"{davies_bouldin_score(X, lab):>15.4f} {calinski_harabasz_score(X, lab):>12.1f}")

for k in [3, 4, 5]:
    lab = GaussianMixture(n_components=k, random_state=RANDOM_SEED).fit_predict(X)
    print(f"{'Gaussian Mixture':<34} {k:>4} {silhouette_score(X, lab):>11.4f} "
          f"{davies_bouldin_score(X, lab):>15.4f} {calinski_harabasz_score(X, lab):>12.1f}")

for eps in [0.3, 0.5, 0.7, 1.0]:
    lab = DBSCAN(eps=eps, min_samples=10).fit_predict(X)
    n_clusters = len(set(lab)) - (1 if -1 in lab else 0)
    n_noise = int((lab == -1).sum())
    if n_clusters >= 2:
        mask = lab != -1
        sil = silhouette_score(X[mask], lab[mask]) if mask.sum() > n_clusters else np.nan
        print(f"{f'DBSCAN (eps={eps})':<34} {n_clusters:>4} {sil:>11.4f} "
              f"{'(noise: ' + str(n_noise) + ')':>15}")
    else:
        print(f"{f'DBSCAN (eps={eps})':<34} {n_clusters:>4} {'n/a':>11} "
              f"{'(noise: ' + str(n_noise) + ')':>15}")

print("\nDBSCAN NOTE: DBSCAN is designed for density-separated clusters with noise.")
print("On a continuum it typically collapses to one giant cluster plus outliers --")
print("which, if observed above, is ITSELF strong evidence supporting the B1")
print("finding that there are no naturally separated groups to be found.")

# --------------------------------------------------------------------
print("\n" + "=" * 70)
print("B5. PCA-THEN-CLUSTER (a commonly-used silhouette inflator)")
print("=" * 70)
for n_comp in [2, 3]:
    Xp = PCA(n_components=n_comp, random_state=RANDOM_SEED).fit_transform(X)
    var = PCA(n_components=n_comp, random_state=RANDOM_SEED).fit(X).explained_variance_ratio_.sum()
    lab = KMeans(n_clusters=4, random_state=RANDOM_SEED, n_init=10).fit_predict(Xp)
    print(f"PCA({n_comp} comps, {var*100:.1f}% variance) + KMeans k=4: "
          f"silhouette = {silhouette_score(Xp, lab):.4f}")
print("\nIf PCA raises silhouette, understand WHY before using it: silhouette is")
print("computed in the REDUCED space, so you are scoring a different (easier)")
print("geometry than the one your segments are defined on. Reporting that number")
print("as if it were comparable to the 3-D result would be misleading.")

# --------------------------------------------------------------------
print("\n" + "=" * 70)
print("B6. CLUSTER STABILITY — a more meaningful validity check than silhouette")
print("=" * 70)
print("For a business segmentation, the question that actually matters is not")
print("'are the clusters geometrically separated?' but 'are they REPRODUCIBLE?'")
print("We bootstrap-resample customers, re-cluster, and measure agreement with")
print("the full-data solution via Adjusted Rand Index (1.0 = identical).\n")

base_labels = KMeans(n_clusters=4, random_state=RANDOM_SEED, n_init=10).fit_predict(X)
aris = []
for b in range(20):
    boot_rng = np.random.default_rng(1000 + b)
    idx = boot_rng.choice(len(X), len(X), replace=True)
    uniq = np.unique(idx)
    km_b = KMeans(n_clusters=4, random_state=RANDOM_SEED, n_init=10).fit(X[idx])
    aris.append(adjusted_rand_score(base_labels[uniq], km_b.predict(X[uniq])))
aris = np.array(aris)
print(f"Bootstrap ARI over 20 resamples: mean={aris.mean():.4f}, "
      f"sd={aris.std():.4f}, min={aris.min():.4f}")

seed_aris = []
for s in [7, 13, 21, 99, 123]:
    lab_s = KMeans(n_clusters=4, random_state=s, n_init=10).fit_predict(X)
    seed_aris.append(adjusted_rand_score(base_labels, lab_s))
print(f"ARI across 5 different random seeds: {np.round(seed_aris, 4)}")
print("\nARI > 0.75 indicates a stable, reproducible segmentation. This is a")
print("legitimate and defensible validity claim even when silhouette is modest,")
print("and is arguably the stronger result to lead with in your report.")

print("\nAudit 2 complete.")
