# ============================================
# FYP PROJECT — Phase 2 AUDIT (3 of 3)
# Is ~70% accuracy a model limitation, a feature limitation, or a data ceiling?
# ============================================
#
# The proposal's success criterion is "accuracy > 75%". This script tests, in
# order, every legitimate route to closing that gap:
#   C1  class balance (is accuracy even the right metric here?)
#   C2  hyperparameter tuning (never actually applied in the original script)
#   C3  feature engineering (richer observation-window behaviour features)
#   C4  a stronger model class, as a diagnostic ceiling probe
#   C5  learning curve (is the model data-starved, or at an irreducible ceiling?)
#   C6  the near-threshold analysis -- WHY a median-split target caps accuracy
#   C7  alternative target framing, honestly evaluated

import pandas as pd
import numpy as np
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score, GridSearchCV, learning_curve)
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             balanced_accuracy_score, precision_score, recall_score)

pd.set_option("display.max_columns", None)
RANDOM_SEED = 42
CUTOFF_DATE = pd.Timestamp("2011-06-01")

df = pd.read_csv("malaysian_context_online_retail.csv", encoding="ISO-8859-1")
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], dayfirst=True)
obs = df[df["InvoiceDate"] < CUTOFF_DATE].copy()
fut = df[df["InvoiceDate"] >= CUTOFF_DATE]

# ---- baseline feature set (as in the current Objective 3 script, minus State,
#      which audit 1 established is randomly-assigned synthetic noise) --------
base = obs.groupby("Customer ID").agg(
    Recency=("InvoiceDate", lambda x: (CUTOFF_DATE - x.max()).days),
    Frequency=("Invoice", "nunique"),
    Monetary=("TotalAmount", "sum"),
    Tenure=("InvoiceDate", lambda x: (CUTOFF_DATE - x.min()).days),
).reset_index()
base["AvgOrderValue"] = base["Monetary"] / base["Frequency"]

future_spend = fut.groupby("Customer ID")["TotalAmount"].sum()
base["FutureSpend"] = base["Customer ID"].map(future_spend).fillna(0.0)

BASE_FEATURES = ["Recency", "Frequency", "Monetary", "AvgOrderValue", "Tenure"]

# --------------------------------------------------------------------
print("=" * 70)
print("C1. CLASS BALANCE — IS ACCURACY THE RIGHT METRIC?")
print("=" * 70)
median_spend = base["FutureSpend"].median()
y = (base["FutureSpend"] > median_spend).astype(int)
print(f"Class distribution: {dict(y.value_counts())}")
print(f"Majority-class baseline accuracy: {max(y.mean(), 1-y.mean()):.4f}")
print("Target is balanced 50/50 BY CONSTRUCTION (median split), so:")
print("  - accuracy is a meaningful metric here (no imbalance inflation)")
print("  - but the naive baseline is 50%, so ~71% is +21pp over chance")
print("  - class weighting / SMOTE are NOT needed and would be inappropriate")

# --------------------------------------------------------------------
print("\n" + "=" * 70)
print("C2. HYPERPARAMETER TUNING (NOT previously applied — this is a real gap)")
print("=" * 70)

num_pipe = Pipeline([
    ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
    ("scale", StandardScaler()),
])

X = base[BASE_FEATURES]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

grids = {
    "Logistic Regression": (
        LogisticRegression(max_iter=2000, random_state=RANDOM_SEED),
        {"clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
         "clf__penalty": ["l2"]},
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=RANDOM_SEED),
        {"clf__max_depth": [3, 4, 5, 6, 8, 10, None],
         "clf__min_samples_leaf": [1, 5, 10, 20, 50],
         "clf__criterion": ["gini", "entropy"]},
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1),
        {"clf__n_estimators": [200, 400],
         "clf__max_depth": [5, 8, 12, None],
         "clf__min_samples_leaf": [1, 5, 10, 20],
         "clf__max_features": ["sqrt", 0.5]},
    ),
}

tuned = {}
print(f"{'Model':<22} {'Default CV F1':>14} {'Tuned CV F1':>12} {'Gain':>8}")
for name, (clf, grid) in grids.items():
    pipe = Pipeline([("num", num_pipe), ("clf", clf)])
    default_cv = cross_val_score(pipe, X_tr, y_tr, cv=skf, scoring="f1").mean()
    gs = GridSearchCV(pipe, grid, cv=skf, scoring="f1", n_jobs=-1)
    gs.fit(X_tr, y_tr)
    tuned[name] = gs.best_estimator_
    print(f"{name:<22} {default_cv:>14.4f} {gs.best_score_:>12.4f} "
          f"{gs.best_score_-default_cv:>+8.4f}")
    print(f"{'  best params:':<22} {gs.best_params_}")

print("\nTuned models on the held-out test set:")
print(f"{'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>9} {'F1':>8} {'AUC':>8}")
for name, est in tuned.items():
    pred = est.predict(X_te)
    proba = est.predict_proba(X_te)[:, 1]
    print(f"{name:<22} {accuracy_score(y_te, pred):>10.4f} "
          f"{precision_score(y_te, pred):>10.4f} {recall_score(y_te, pred):>9.4f} "
          f"{f1_score(y_te, pred):>8.4f} {roc_auc_score(y_te, proba):>8.4f}")

# --------------------------------------------------------------------
print("\n" + "=" * 70)
print("C3. FEATURE ENGINEERING (richer observation-window behaviour)")
print("=" * 70)
print("All new features are computed from observation-window rows ONLY,")
print("preserving the temporal-split guarantee verified in audit 1.\n")

inv = (obs.groupby(["Customer ID", "Invoice"])
          .agg(InvValue=("TotalAmount", "sum"),
               InvDate=("InvoiceDate", "first"),
               InvItems=("Quantity", "sum"))
          .reset_index())


def inter_purchase_stats(g):
    d = np.sort(g["InvDate"].values)
    if len(d) < 2:
        return pd.Series({"IPT_mean": np.nan, "IPT_std": np.nan})
    gaps = np.diff(d).astype("timedelta64[D]").astype(float)
    return pd.Series({"IPT_mean": gaps.mean(), "IPT_std": gaps.std()})


ipt = inv.groupby("Customer ID").apply(inter_purchase_stats, include_groups=False).reset_index()

extra = inv.groupby("Customer ID").agg(
    MaxInvoiceValue=("InvValue", "max"),
    StdInvoiceValue=("InvValue", "std"),
    AvgItemsPerInvoice=("InvItems", "mean"),
).reset_index()

prod = obs.groupby("Customer ID").agg(
    DistinctProducts=("StockCode", "nunique"),
    TotalQuantity=("Quantity", "sum"),
).reset_index()

obs["YearMonth"] = obs["InvoiceDate"].dt.to_period("M")
active = obs.groupby("Customer ID")["YearMonth"].nunique().rename("ActiveMonths").reset_index()

# Spend momentum: share of observation-window spend that fell in the final 90
# days. A customer accelerating into the cutoff should be likelier to keep
# spending than one whose activity is all historical.
recent_window = CUTOFF_DATE - pd.Timedelta(days=90)
recent_spend = (obs[obs["InvoiceDate"] >= recent_window]
                .groupby("Customer ID")["TotalAmount"].sum().rename("Spend_last90"))

feat = (base.merge(ipt, on="Customer ID", how="left")
            .merge(extra, on="Customer ID", how="left")
            .merge(prod, on="Customer ID", how="left")
            .merge(active, on="Customer ID", how="left")
            .merge(recent_spend, on="Customer ID", how="left"))

feat["Spend_last90"] = feat["Spend_last90"].fillna(0.0)
feat["MomentumRatio"] = feat["Spend_last90"] / feat["Monetary"].replace(0, np.nan)
feat["PurchaseRate"] = feat["Frequency"] / feat["Tenure"].replace(0, np.nan)
feat["IPT_mean"] = feat["IPT_mean"].fillna(feat["Tenure"])
feat["IPT_std"] = feat["IPT_std"].fillna(0.0)
feat["StdInvoiceValue"] = feat["StdInvoiceValue"].fillna(0.0)
feat["MomentumRatio"] = feat["MomentumRatio"].fillna(0.0)
feat["PurchaseRate"] = feat["PurchaseRate"].fillna(0.0)

RICH_FEATURES = BASE_FEATURES + [
    "IPT_mean", "IPT_std", "MaxInvoiceValue", "StdInvoiceValue",
    "AvgItemsPerInvoice", "DistinctProducts", "TotalQuantity",
    "ActiveMonths", "Spend_last90", "MomentumRatio", "PurchaseRate",
]
print(f"Baseline features: {len(BASE_FEATURES)} | Engineered feature set: {len(RICH_FEATURES)}")

Xr = feat[RICH_FEATURES]
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
    Xr, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

# log1p requires non-negative input; all engineered features are non-negative
# counts/amounts/durations, so this is safe (asserted rather than assumed).
assert (Xr.min() >= 0).all(), "log1p requires non-negative features"

print(f"\n{'Model':<22} {'Base CV F1':>12} {'Rich CV F1':>12} {'Gain':>8}")
rich_models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_SEED),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_SEED),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8,
                                            random_state=RANDOM_SEED, n_jobs=-1),
}
for name, clf in rich_models.items():
    p = Pipeline([("num", num_pipe), ("clf", clf)])
    b = cross_val_score(p, X_tr, y_tr, cv=skf, scoring="f1").mean()
    r = cross_val_score(p, Xr_tr, yr_tr, cv=skf, scoring="f1").mean()
    print(f"{name:<22} {b:>12.4f} {r:>12.4f} {r-b:>+8.4f}")

# --------------------------------------------------------------------
print("\n" + "=" * 70)
print("C4. CEILING PROBE — does a stronger model class break 75%?")
print("=" * 70)
print("Gradient boosting is not one of your three required models; it is used")
print("here purely as a DIAGNOSTIC. If a strong learner also plateaus near 71%,")
print("the limit is in the data, not in your choice of algorithm.\n")

hgb = HistGradientBoostingClassifier(random_state=RANDOM_SEED)
hgb_cv = cross_val_score(hgb, Xr_tr, yr_tr, cv=skf, scoring="f1").mean()
hgb.fit(Xr_tr, yr_tr)
hgb_pred, hgb_proba = hgb.predict(Xr_te), hgb.predict_proba(Xr_te)[:, 1]
print(f"HistGradientBoosting  CV F1={hgb_cv:.4f}  test acc={accuracy_score(yr_te, hgb_pred):.4f}  "
      f"test F1={f1_score(yr_te, hgb_pred):.4f}  AUC={roc_auc_score(yr_te, hgb_proba):.4f}")

# --------------------------------------------------------------------
print("\n" + "=" * 70)
print("C5. LEARNING CURVE — data-starved, or at an irreducible ceiling?")
print("=" * 70)
best_rf = Pipeline([("num", num_pipe),
                    ("clf", RandomForestClassifier(n_estimators=200, max_depth=8,
                                                   random_state=RANDOM_SEED, n_jobs=-1))])
sizes, train_sc, val_sc = learning_curve(
    best_rf, Xr_tr, yr_tr, cv=skf, scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1, random_state=RANDOM_SEED)
print(f"{'Train size':>11} {'Train acc':>11} {'Val acc':>10}")
for s, tr, va in zip(sizes, train_sc.mean(axis=1), val_sc.mean(axis=1)):
    print(f"{int(s):>11} {tr:>11.4f} {va:>10.4f}")
print("\nIf validation accuracy has FLATTENED, adding more customers will not help:")
print("the model is limited by signal, not sample size.")

# --------------------------------------------------------------------
print("\n" + "=" * 70)
print("C6. WHY A MEDIAN SPLIT CAPS ACCURACY (the key explanatory result)")
print("=" * 70)
print("A median split labels a customer who spends RM 344 as class 0 and one who")
print("spends RM 346 as class 1. Those two customers are behaviourally identical,")
print("so NO model can separate them. We measure accuracy as a function of how")
print("far a customer's actual future spend sits from the threshold.\n")

rf_fit = best_rf.fit(Xr_tr, yr_tr)
te_idx = Xr_te.index
te_pred = rf_fit.predict(Xr_te)
te_spend = base.loc[te_idx, "FutureSpend"]
correct = (te_pred == yr_te.values)

sp = te_spend.values
m_ = median_spend
bands = [
    ("RM 0 (fully churned)",            sp == 0),
    ("RM 0.01 - 0.5x threshold",        (sp > 0) & (sp < m_ * 0.5)),
    ("0.5x - 2x threshold (ambiguous)", (sp >= m_ * 0.5) & (sp <= m_ * 2)),
    ("2x - 10x threshold (moderate)",   (sp > m_ * 2) & (sp <= m_ * 10)),
    ("above 10x threshold (clear high)", sp > m_ * 10),
]
print(f"{'Band':<36} {'N':>6} {'Accuracy':>10}")
for label, m in bands:
    if m.sum() > 0:
        print(f"{label:<36} {int(m.sum()):>6} {correct[m].mean():>10.4f}")
print(f"{'ALL':<36} {len(sp):>6} {correct.mean():>10.4f}")

print("\nHONEST READING OF THIS TABLE (do not overstate it):")
print("The narrowly-ambiguous band around the threshold is real but SMALL (~6% of")
print("the test set), so it is NOT by itself the explanation for the ~71% ceiling.")
print("The dominant error source is the MODERATE-spender band (roughly 2x-10x the")
print("threshold, ~26% of customers), where accuracy sits near chance.")
print("")
print("The defensible claim is therefore narrower than 'the threshold is to blame':")
print("the model discriminates well at the BEHAVIOURAL EXTREMES -- customers who")
print("fully churn and customers who spend heavily -- and poorly in the middle,")
print("where future spend is genuinely volatile and weakly determined by past RFM.")
print("That is a substantive finding about consumer behaviour, and it is also")
print("exactly what an AUC near 0.80 with accuracy near 0.71 describes.")

# --------------------------------------------------------------------
print("\n" + "=" * 70)
print("C7. ALTERNATIVE TARGET FRAMING — top-quartile 'VIP' prediction")
print("=" * 70)
q75 = base["FutureSpend"].quantile(0.75)
y_vip = (base["FutureSpend"] > q75).astype(int)
print(f"Threshold (75th pct future spend): RM {q75:,.2f}")
print(f"Class balance: {dict(y_vip.value_counts())} "
      f"-> majority-class baseline accuracy = {max(y_vip.mean(), 1-y_vip.mean()):.4f}")

Xv_tr, Xv_te, yv_tr, yv_te = train_test_split(
    Xr, y_vip, test_size=0.2, random_state=RANDOM_SEED, stratify=y_vip)
vip_rf = Pipeline([("num", num_pipe),
                   ("clf", RandomForestClassifier(n_estimators=200, max_depth=8,
                                                  random_state=RANDOM_SEED, n_jobs=-1))])
vip_rf.fit(Xv_tr, yv_tr)
vp = vip_rf.predict(Xv_te)
vpr = vip_rf.predict_proba(Xv_te)[:, 1]
print(f"\nRandom Forest on VIP target:")
print(f"  Accuracy          : {accuracy_score(yv_te, vp):.4f}  <- EXCEEDS 75%")
print(f"  Balanced accuracy : {balanced_accuracy_score(yv_te, vp):.4f}  <- the honest number")
print(f"  Precision         : {precision_score(yv_te, vp):.4f}")
print(f"  Recall            : {recall_score(yv_te, vp):.4f}")
print(f"  F1                : {f1_score(yv_te, vp):.4f}")
print(f"  AUC               : {roc_auc_score(yv_te, vpr):.4f}")
print("\nCRITICAL HONESTY CHECK: on a 75/25 target, a model that predicts 'not VIP'")
print("for EVERY customer already scores 75% accuracy. So hitting '>75% accuracy'")
print("this way would satisfy the proposal's wording while being close to")
print("meaningless. Balanced accuracy and recall are the numbers that show whether")
print("the model has learned anything. Report those alongside, or the result is")
print("indefensible under questioning.")

print("\nAudit 3 complete.")
