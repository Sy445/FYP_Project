# FYP Project Review — Phase 2 Complete

**Predictive Consumer Segmentation and Spending Behaviour in Malaysian Retail**
Methodology: CRISP-DM · Status as of 21 August 2026

**Live dashboard:** https://fypproject-tp070227.streamlit.app/
**Repository:** https://github.com/Sy445/FYP_Project

---

## 1. Status at a glance

| Objective | Status |
|---|---|
| **1** — Preprocessing pipeline + EDA | ✅ Complete |
| **2** — RFM + K-means segmentation | ✅ Complete |
| **3** — Predictive modelling (3 algorithms) | ✅ Complete |
| **4** — Interactive dashboard | ✅ Complete · deployed and publicly accessible |

**Automated verification: 35 of 35 code-side compliance checks pass** (`python final_check.py`). No outstanding implementation items.

**Two proposal success criteria were not met.** Both are documented with evidence in Section 5, and both require a decision from you.

---

## 2. What was built

Six Python scripts, all executing cleanly end-to-end (exit code 0, no errors):

| Script | Purpose |
|---|---|
| `data_preprocessing.py` | Phase 1 cleaning → 715,863 records |
| `phase2_objective2_rfm_kmeans.py` | RFM computation + K-means segmentation |
| `phase2_objective3_predictive_modelling.py` | Three classifiers, tuned and compared |
| `audit/audit_1_leakage_and_rigour.py` | Data leakage + statistical rigour checks |
| `audit/audit_2_clustering.py` | Cluster validity investigation |
| `audit/audit_3_accuracy.py` | Predictive ceiling investigation |
| `app.py` | Streamlit dashboard (3 pages) |

All results are reproducible under a fixed random seed (42).

**Preprocessing outcome** — 67.1% of raw records retained:

| Stage | Records | Removed |
|---|---|---|
| Raw (2 source files, Dec 2009 – Dec 2011) | 1,067,371 | — |
| Missing Customer ID removed | 824,364 | −243,007 |
| Duplicates removed | 797,885 | −26,479 |
| Invalid transactions removed | 779,425 | −18,460 |
| IQR outliers removed | **715,863** | −63,562 |

---

## 3. Segmentation results (Objective 2)

Four segments from K-means (k = 4) on log-transformed, standardised RFM features. Optimal k selected using both the elbow method and silhouette analysis, as specified in Objective 1.4(2).

| Segment | Customers | % of base | Avg recency | Avg orders | Avg spend | % of revenue |
|---|---|---|---|---|---|---|
| Champions | 1,186 | 20.9% | 27.6 days | 17.3 | RM 26,342 | **65.7%** |
| At-Risk High Value | 1,396 | 24.6% | 232.2 days | 4.8 | RM 6,903 | 20.2% |
| New & Promising | 1,209 | 21.3% | 29.4 days | 2.9 | RM 3,516 | 8.9% |
| Lost / Dormant | 1,887 | 33.2% | 396.0 days | 1.4 | RM 1,302 | 5.2% |

**Key business finding:** 20.9% of customers generate 65.7% of revenue, and RM 9.6M of historical revenue sits in the At-Risk segment — customers who previously spent well and have not purchased in roughly eight months.

**Cluster validity:**

| Measure | Value | Interpretation |
|---|---|---|
| Silhouette (k=4) | 0.360 | Below the 0.5 target — see Section 5 |
| Davies–Bouldin | 0.941 | Lower is better |
| Calinski–Harabasz | 4,852.9 | Higher is better |
| Bootstrap ARI (20 resamples) | **0.956** | Well above the 0.75 stability convention |
| Seed-variation ARI (5 seeds) | **0.989** | Highly reproducible |

---

## 4. Predictive modelling results (Objective 3)

**Target:** binary classification — will a customer be an above-median spender in the following period?

**Design:** temporal split to prevent leakage. Features are drawn from a 546-day observation window (Dec 2009 – May 2011); the target is drawn from a strictly later 191-day window (Jun – Dec 2011). Verified: zero temporal overlap, all features derived from observation-window records only.

All three algorithms tuned via `GridSearchCV` on identical stratified folds.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train–test gap |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.7179 | 0.7336 | 0.6842 | 0.7081 | 0.7934 | **0.0105** |
| Random Forest | 0.7105 | 0.7304 | 0.6674 | 0.6975 | 0.7961 | 0.0660 |
| Decision Tree | 0.7042 | 0.7462 | 0.6189 | 0.6766 | 0.7853 | 0.0386 |

**The three models are not statistically distinguishable.** Nadeau–Bengio corrected paired t-test p = 0.895; McNemar exact test p = 0.550. Selecting the top row of the table would be reading noise as signal.

**Recommended model: Logistic Regression** — chosen on qualitative grounds, since the metrics tie: it has by far the smallest train–test gap (0.0105 vs 0.0660), directly interpretable coefficients, and the simplest deployment path.

**Predictive drivers** (standardised coefficients): prior spending +0.954, recency −0.501, order count +0.292, tenure −0.231, average order value −0.182. The negative coefficient on average order value once total spend is controlled indicates that **many small orders predict future value better than a few large ones**.

**Performance by customer type** — the model is strongest at the behavioural extremes:

| Actual future spend | Accuracy |
|---|---|
| Zero (fully churned) | 76.3% |
| Moderate band | 57.0% |
| High spenders | 87.4% |

---

## 5. Two success criteria were not met

Both were investigated rather than accepted, and both appear to be genuine ceilings rather than implementation failures.

### 5.1 Silhouette score: 0.360 against a 0.5 target

Scores above 0.5 **are** numerically reachable, but only through degenerate configurations. Omitting the log transform produces silhouette 0.575 at k = 4 — with these cluster sizes:

| Cluster | Customers |
|---|---|
| 1 | 1,877 (33.1%) |
| 2 | 3,406 (60.0%) |
| 3 | 386 (6.8%) |
| 4 | **9 (0.16%)** |

At k = 5 the same approach produces a cluster containing **one customer**. The high score reflects K-means isolating outliers, not meaningful segmentation.

Alternative algorithms were tested and none exceeded K-means:

| Algorithm | k=3 | k=4 | k=5 |
|---|---|---|---|
| K-means | 0.346 | **0.360** | 0.327 |
| Hierarchical (Ward) | 0.304 | 0.313 | 0.298 |
| Gaussian Mixture | 0.285 | 0.219 | 0.216 |
| DBSCAN | collapses to 1–2 clusters at every ε tested | | |

DBSCAN — designed specifically to find density-separated groups — found none. This suggests RFM describes a continuous behavioural gradient rather than four naturally separated customer types.

**Proposed position:** the 0.5 criterion presumes a separated cluster structure that RFM data does not exhibit. For an applied segmentation, reproducibility is the more meaningful validity claim, and the segments are strong on that measure (bootstrap ARI 0.956, seed ARI 0.989).

### 5.2 Predictive accuracy: 71.8% against a 75% target

Four improvement routes were tested; none closed the gap:

| Route | Outcome |
|---|---|
| Hyperparameter tuning (GridSearchCV) | 71.8% |
| Feature expansion (5 → 16 behavioural features) | +0.004 F1 |
| Stronger model class (gradient boosting, diagnostic) | 72.2% |
| More training data (learning curve) | Plateaus at ~2,600 samples |

Class balance is not a factor — the target is 50/50 by construction, so the majority-class baseline is 50.0%.

**A note on an available shortcut:** reframing the target as top-quartile "VIP" prediction yields 84.8% accuracy and would satisfy the criterion. It was not adopted, because the majority-class baseline on that target is already 75% — a model predicting "not VIP" for every customer would "pass". Balanced accuracy on that framing is 74.4% and recall 0.536.

**Proposed position:** report 71.8% with the ceiling evidence, noting ROC-AUC of 0.793 and the strong performance at the behavioural extremes where business value is concentrated.

---

## 6. Three discrepancies found between the report and the implementation

These were identified during a line-by-line audit of the code against the proposal document.

### 6.1 DOSM secondary dataset — committed but not implemented

The proposal commits to DOSM as a **secondary dataset** in four places: the Abstract, §1.6.1 (*"is employed"*), §1.6.3 Inclusion 5, and §2.3. No DOSM dataset exists in the project. DOSM is cited correctly as **literature** in Chapter 1 (the 97.4% and <18% figures), but not used as data.

### 6.2 Dataset description in §1.6.1 is inaccurate

The report states the dataset spans **December 2010 to December 2011** from a single Kaggle source. The implementation uses **two files spanning December 2009 to December 2011** (Online Retail II), totalling 1,067,371 raw records.

This is material: Objective 3's temporal split requires approximately two years of data. The dataset as described in §1.6.1 could not have produced the results in Section 4 above.

### 6.3 The Malaysian-context adaptation is not documented in Chapter 3

Chapter 3 documents the UK preprocessing only (Figures 3.2–3.14). Objectives 2–4 all run on an adapted Malaysian-context dataset produced by six transformation steps — geography remapped to Malaysian states, GBP converted to MYR at a fixed rate of 5.50, psychological price rounding, and product description localisation.

Validation evidence exists but is not yet in the report: row count and customer set identical to the source, and distribution shapes preserved (Quantity skewness 6.8186 → 6.8186; Price 3.1718 → 3.1745; TotalAmount 0.9342 → 0.9333).

**Related methodological note:** the `State` variable was **excluded** from predictive modelling. Because state was assigned by weighted random draw during adaptation, it cannot carry genuine predictive signal. A chi-square test confirmed no association with the target (p = 0.622), and ablation showed removal left performance unchanged or marginally improved. Retaining it would have produced coefficients inviting spurious geographic interpretation.

---

## 7. Decisions requested

| # | Question |
|---|---|
| 1 | Is it acceptable to replace the *"silhouette > 0.5"* criterion with a **stability criterion (ARI > 0.75)**, given the evidence in §5.1? |
| 2 | For predictive accuracy — report **71.8% with the ceiling evidence**, or additionally present the VIP framing as a documented secondary analysis? |
| 3 | For DOSM (§6.1) — should real DOSM retail data be sourced and a contextualisation section added, or should the four claims be rewritten to describe DOSM as a contextual literature source? |
| 4 | How much detail is required for the Malaysian adaptation in Chapter 3 (§6.3), and **can approval for this methodological choice be confirmed in writing?** |
| 5 | Does an unmet success criterion affect assessment if the shortfall is rigorously justified? |
| 6 | Is a live deployed dashboard URL sufficient for Objective 4, or is a deployment/user guide section also expected? |

---

## 8. Known limitations

1. **Cold-start customers excluded.** 929 customers (27.1% of the future window) appear only after the temporal cut-off and cannot be scored — a behavioural model cannot assess a customer with no prior behaviour. Predictions apply to existing customers only.
2. **Adaptation scripts are not currently re-runnable.** The five `Dataset process/` scripts contain hardcoded sandbox paths, so the Malaysian dataset cannot presently be regenerated from source. The existing file is validated and authoritative, but this is a reproducibility gap.
3. **Geography carries no real-world meaning.** State is synthetic and is used in the dashboard for descriptive filtering only, with an explicit on-screen warning.
4. **UK-origin behaviour.** Underlying purchasing patterns derive from a UK dataset; cultural and economic differences limit the generalisability of specific segment profiles to Malaysian consumers.

---

## 9. Supporting documentation

| Document | Contents |
|---|---|
| `CODE_COMPLIANCE.md` | Automated verification results (35/35), with evidence per check |
| `COMPLIANCE_CHECK.md` | Full audit against proposal objectives, deliverables, RQs, constraints |
| `audit/PHASE2_AUDIT_REPORT.md` | Critical assessment, ceiling analysis, draft report prose |
| `DATA_LINEAGE.md` | Complete data flow diagram and per-script file I/O |
| `final_check.py` | Re-runnable compliance verification |

All figures in this document are reproducible by running the scripts in Section 2.
