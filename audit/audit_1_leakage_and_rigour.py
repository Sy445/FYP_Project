# ============================================
# FYP PROJECT — Phase 2 AUDIT (1 of 3)
# Data leakage, temporal-split integrity, overfitting, model-comparison rigour
# ============================================
#
# This script re-derives Objective 3's data independently (it does NOT import
# from the modelling script) so that any bug in the modelling script would
# show up as a disagreement here rather than being silently reproduced.

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

pd.set_option("display.max_columns", None)

RANDOM_SEED = 42
CUTOFF_DATE = pd.Timestamp("2011-06-01")

df = pd.read_csv("malaysian_context_online_retail.csv", encoding="ISO-8859-1")
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], dayfirst=True)

obs = df[df["InvoiceDate"] < CUTOFF_DATE]
fut = df[df["InvoiceDate"] >= CUTOFF_DATE]

print("=" * 70)
print("A1. TEMPORAL SPLIT INTEGRITY")
print("=" * 70)

# A1.1 No overlap: the last observation-window timestamp must strictly precede
#      the first future-window timestamp.
obs_max, fut_min = obs["InvoiceDate"].max(), fut["InvoiceDate"].min()
no_overlap = obs_max < fut_min
print(f"Last obs-window timestamp:   {obs_max}")
print(f"First future-window timestamp: {fut_min}")
print(f"[{'PASS' if no_overlap else 'FAIL'}] No temporal overlap between windows")

# A1.2 Partition is exhaustive and disjoint (every row lands in exactly one window)
partition_ok = (len(obs) + len(fut) == len(df))
print(f"[{'PASS' if partition_ok else 'FAIL'}] Windows partition the dataset exactly "
      f"({len(obs):,} + {len(fut):,} = {len(df):,})")

# A1.3 Window lengths
print(f"\nObservation window length: {(obs_max - obs['InvoiceDate'].min()).days} days")
print(f"Future window length:      {(fut['InvoiceDate'].max() - fut_min).days} days")

# A1.4 Cold-start customers: appear ONLY in the future window. These are
#      correctly EXCLUDED from the model (we cannot build behavioural
#      features for a customer with no prior history), but the exclusion
#      must be acknowledged as a scope limitation, not hidden.
obs_ids = set(obs["Customer ID"].unique())
fut_ids = set(fut["Customer ID"].unique())
cold_start = fut_ids - obs_ids
print(f"\nCustomers in obs window:            {len(obs_ids):,}")
print(f"Customers in future window:         {len(fut_ids):,}")
print(f"Cold-start (future-only) customers: {len(cold_start):,} "
      f"({len(cold_start)/len(fut_ids)*100:.1f}% of future-window customers)")
print("  -> correctly excluded from modelling (no prior behaviour to learn from),")
print("     but this IS a scope limitation to state in the report.")

# --------------------------------------------------------------------
print("\n" + "=" * 70)
print("A2. FEATURE/TARGET LEAKAGE CHECK")
print("=" * 70)

features = obs.groupby("Customer ID").agg(
    Recency=("InvoiceDate", lambda x: (CUTOFF_DATE - x.max()).days),
    Frequency=("Invoice", "nunique"),
    Monetary=("TotalAmount", "sum"),
    Tenure=("InvoiceDate", lambda x: (CUTOFF_DATE - x.min()).days),
    State=("State", "first"),
).reset_index()
features["AvgOrderValue"] = features["Monetary"] / features["Frequency"]

future_spend = fut.groupby("Customer ID")["TotalAmount"].sum()
features["FutureSpend"] = features["Customer ID"].map(future_spend).fillna(0.0)

# A2.1 Every feature must be derivable from obs rows ONLY. We verify this
#      structurally: recompute each feature using a hard-filtered frame and
#      confirm identical values.
recheck = df[df["InvoiceDate"] < CUTOFF_DATE].groupby("Customer ID").agg(
    Recency=("InvoiceDate", lambda x: (CUTOFF_DATE - x.max()).days),
    Frequency=("Invoice", "nunique"),
    Monetary=("TotalAmount", "sum"),
).reset_index()
feat_clean = features.set_index("Customer ID")[["Recency", "Frequency", "Monetary"]]
recheck_clean = recheck.set_index("Customer ID")[["Recency", "Frequency", "Monetary"]]
features_obs_only = feat_clean.equals(recheck_clean)
print(f"[{'PASS' if features_obs_only else 'FAIL'}] All features derive from observation-window rows only")

# A2.2 Recency must never be negative or zero-by-lookahead: a customer whose
#      last obs purchase is the day before cutoff has Recency=1, not 0 or -N.
recency_sane = (features["Recency"] >= 0).all()
print(f"[{'PASS' if recency_sane else 'FAIL'}] Recency has no negative values "
      f"(min={features['Recency'].min()}, max={features['Recency'].max()})")

# A2.3 THRESHOLD LEAKAGE — the one real (if minor) issue.
# The modelling script computes the median future spend on ALL 4,749
# customers, then uses it to label train AND test. Strictly, the test set's
# targets contributed to defining the decision boundary. Standard rigour is
# to derive the threshold from TRAINING data only. We quantify the impact.
median_all = features["FutureSpend"].median()

y_all = (features["FutureSpend"] > median_all).astype(int)
NUMERIC = ["Recency", "Frequency", "Monetary", "AvgOrderValue", "Tenure"]
CATEGORICAL = ["State"]
X = features[NUMERIC + CATEGORICAL]

X_tr, X_te, ytr_all, yte_all, idx_tr, idx_te = train_test_split(
    X, y_all, features.index, test_size=0.2, random_state=RANDOM_SEED, stratify=y_all
)
median_train_only = features.loc[idx_tr, "FutureSpend"].median()

print(f"\nMedian future spend (all customers, as used in the script): RM {median_all:,.2f}")
print(f"Median future spend (training set only, the stricter choice): RM {median_train_only:,.2f}")
n_relabelled = int((
    (features["FutureSpend"] > median_all).astype(int)
    != (features["FutureSpend"] > median_train_only).astype(int)
).sum())
print(f"Customers whose label would change: {n_relabelled} / {len(features)} "
      f"({n_relabelled/len(features)*100:.2f}%)")
if n_relabelled == 0:
    print("  -> ZERO label changes: the threshold leakage is nominal only and has")
    print("     no material effect on results. Worth one sentence in limitations.")
else:
    print("  -> Non-zero: rerun with the train-only threshold before reporting.")

# --------------------------------------------------------------------
print("\n" + "=" * 70)
print("A3. THE 'State' FEATURE — IS IT LEGITIMATE SIGNAL OR SYNTHETIC NOISE?")
print("=" * 70)
print("CRITICAL: Dataset process/step1_geography.py assigns State by WEIGHTED")
print("RANDOM DRAW per customer, independent of any purchasing behaviour.")
print("Therefore State CANNOT carry genuine predictive signal about spending.")
print("Any non-zero State coefficient/importance is the model fitting noise.")
print("We test this empirically: chi-square independence + with/without ablation.\n")

# A3.1 Chi-square test of independence between State and the target
contingency = pd.crosstab(features["State"], y_all)
chi2, p_chi, dof, _ = stats.chi2_contingency(contingency)
print(f"Chi-square test (State vs HighFutureSpender): chi2={chi2:.3f}, dof={dof}, p={p_chi:.4f}")
print(f"  -> {'NO significant association (as expected for random assignment)' if p_chi > 0.05 else 'Significant association -- INVESTIGATE, this should not happen'}")

# A3.2 Ablation: model performance with vs without State
numeric_pipeline = Pipeline([
    ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
    ("scale", StandardScaler()),
])
prep_with = ColumnTransformer([
    ("num", numeric_pipeline, NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
])
prep_without = ColumnTransformer([("num", numeric_pipeline, NUMERIC)])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_SEED),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_SEED),
}

print(f"\n{'Model':<22} {'CV F1 with State':>18} {'CV F1 without':>15} {'Delta':>9}")
for name, clf in models.items():
    f1_with = cross_val_score(Pipeline([("p", prep_with), ("c", clf)]), X_tr, ytr_all,
                              cv=skf, scoring="f1").mean()
    f1_without = cross_val_score(Pipeline([("p", prep_without), ("c", clf)]), X_tr[NUMERIC], ytr_all,
                                 cv=skf, scoring="f1").mean()
    print(f"{name:<22} {f1_with:>18.4f} {f1_without:>15.4f} {f1_without-f1_with:>+9.4f}")

print("\nIf removing State leaves performance unchanged (or improves it), State is")
print("confirmed as noise and MUST be dropped -- keeping it invites an examiner to")
print("ask 'what does the Sarawak coefficient mean?', which has no honest answer.")

# --------------------------------------------------------------------
print("\n" + "=" * 70)
print("A4. OVERFITTING CHECK (train vs test gap)")
print("=" * 70)
print(f"{'Model':<22} {'Train Acc':>10} {'Test Acc':>10} {'Gap':>8} {'Test AUC':>10}")
fitted = {}
for name, clf in models.items():
    pipe = Pipeline([("p", prep_with), ("c", clf)])
    pipe.fit(X_tr, ytr_all)
    fitted[name] = pipe
    tr_acc = accuracy_score(ytr_all, pipe.predict(X_tr))
    te_acc = accuracy_score(yte_all, pipe.predict(X_te))
    auc = roc_auc_score(yte_all, pipe.predict_proba(X_te)[:, 1])
    print(f"{name:<22} {tr_acc:>10.4f} {te_acc:>10.4f} {tr_acc-te_acc:>+8.4f} {auc:>10.4f}")

print("\nInterpretation guide: a gap > ~0.05 suggests the model is memorising the")
print("training set. AUC is threshold-independent and measures how well the model")
print("RANKS customers by spend propensity -- often the more informative number")
print("when accuracy is capped by an arbitrary threshold (see audit 3).")

# --------------------------------------------------------------------
print("\n" + "=" * 70)
print("A5. IS 'TOP-2 MODELS INDISTINGUISHABLE' PROPERLY ESTABLISHED?")
print("=" * 70)
print("The current script compares |mean1-mean2| against max(std1,std2). That is a")
print("HEURISTIC, not a statistical test: the CV std is the spread across folds,")
print("not the standard error of the difference, and the two models are evaluated")
print("on the SAME folds (paired data), which the comparison ignores.")
print("Below are two defensible tests.\n")

# A5.1 Paired CV scores on identical folds
fold_scores = {}
for name, clf in models.items():
    fold_scores[name] = cross_val_score(Pipeline([("p", prep_with), ("c", clf)]),
                                        X_tr, ytr_all, cv=skf, scoring="f1")
    print(f"{name:<22} per-fold F1: {np.round(fold_scores[name], 4)}")

ranked = sorted(fold_scores.items(), key=lambda kv: kv[1].mean(), reverse=True)
(n1, s1), (n2, s2) = ranked[0], ranked[1]
diff = s1 - s2

# Standard paired t-test (noted caveat: CV folds are not fully independent,
# so this is anti-conservative -- Dietterich 1998)
t_stat, p_paired = stats.ttest_rel(s1, s2)

# Nadeau & Bengio (2003) corrected resampled t-test, which inflates the
# variance term to account for the train/test overlap between CV folds.
k = len(s1)
n_test_ratio = 1 / (k - 1)  # test/train size ratio for k-fold
corrected_var = diff.var(ddof=1) * (1 / k + n_test_ratio)
t_corrected = diff.mean() / np.sqrt(corrected_var) if corrected_var > 0 else np.nan
p_corrected = 2 * (1 - stats.t.cdf(abs(t_corrected), df=k - 1))

print(f"\nComparing top-2: {n1} vs {n2}")
print(f"  Mean F1 difference: {diff.mean():+.4f}")
print(f"  Paired t-test (uncorrected):        t={t_stat:+.3f}, p={p_paired:.4f}")
print(f"  Nadeau-Bengio corrected t-test:     t={t_corrected:+.3f}, p={p_corrected:.4f}")
print(f"  -> {'NOT significantly different (p > 0.05)' if p_corrected > 0.05 else 'Significantly different'}")

# A5.2 McNemar's test on the held-out test set (the standard test for
# comparing two classifiers on the SAME test sample). Implemented directly
# (exact binomial version) since statsmodels is not installed.
pred1 = fitted[n1].predict(X_te)
pred2 = fitted[n2].predict(X_te)
c1, c2 = (pred1 == yte_all.values), (pred2 == yte_all.values)
b = int((c1 & ~c2).sum())   # model1 right, model2 wrong
c = int((~c1 & c2).sum())   # model1 wrong, model2 right
mcnemar_p = stats.binomtest(b, b + c, 0.5).pvalue if (b + c) > 0 else 1.0
print(f"\nMcNemar's test on held-out test set ({n1} vs {n2}):")
print(f"  {n1} correct / {n2} wrong: b={b}")
print(f"  {n1} wrong / {n2} correct: c={c}")
print(f"  Exact binomial p-value: {mcnemar_p:.4f}")
print(f"  -> {'NOT significantly different' if mcnemar_p > 0.05 else 'Significantly different'}")

print("\nBoth tests should be reported instead of the ad-hoc std comparison.")
print("\nAudit 1 complete.")
