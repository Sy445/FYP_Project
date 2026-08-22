# Code Compliance Report

**FYP:** Predictive Consumer Segmentation and Spending Behaviour in Malaysian Retail
**Verified:** 21 August 2026
**Method:** automated — every check executes or inspects a real file
**Reproduce with:** `python final_check.py`

> **Scope: code only.** This report covers the implementation. Document-side items — the DOSM commitment, the dataset description in §1.6.1, and documenting the Malaysian adaptation in Chapter 3 — are edits to your written report and cannot be verified by code. Those are in `COMPLIANCE_CHECK.md`.

---

## Result

# 35 / 35 code-side checks passed

**All code-side requirements are satisfied.** The dashboard is deployed and publicly accessible at https://fypproject-tp070227.streamlit.app/

| Area | Checks | Result |
|---|---|---|
| Obj 1 / Del 1 — Preprocessing pipeline + EDA | 7 | ✅ 7/7 |
| Obj 2 / Del 2 — RFM + K-means segmentation | 6 | ✅ 6/6 |
| Obj 3 / Del 3 — Predictive modelling suite | 4 | ✅ 4/4 |
| Del 4 — Model Evaluation Report | 4 | ✅ 4/4 |
| Obj 4 / Del 5 / RQ4 — Dashboard | 8 | ✅ 8/8 |
| Constraints (§1.6.3) | 5 | ✅ 5/5 |
| Success-criteria disclosure | 1 | ✅ 1/1 |
| **Total** | **35** | **✅ 35 pass · 0 outstanding** |

---

## Script execution — all six run clean

Verified end-to-end, exit code 0, no errors on stderr:

| Script | Result |
|---|---|
| `phase2_objective2_rfm_kmeans.py` | exit 0, stderr clean |
| `phase2_objective3_predictive_modelling.py` | exit 0, stderr clean |
| `audit/audit_1_leakage_and_rigour.py` | exit 0, stderr clean |
| `audit/audit_2_clustering.py` | exit 0, stderr clean |
| `audit/audit_3_accuracy.py` | exit 0, stderr clean |
| `audit/validate_malaysian_dataset.py` | exit 0, stderr clean |

⚠️ The four `audit/` scripts use relative paths and must be run **from the project root** (`python audit/audit_1_leakage_and_rigour.py`), not from inside `audit/`.

---

## Objective 1 / Deliverable 1 — Preprocessing pipeline + EDA ✅ 7/7

Deliverable 1 names six required elements. All six verified present in `data_preprocessing.py`:

| Element | Implementation | Status |
|---|---|---|
| Missing value handling | `dropna` | ✅ |
| Duplicate removal | `drop_duplicates` | ✅ |
| Data type standardisation | `to_datetime` / `astype` | ✅ |
| Invalid transaction filtering | `Quantity > 0`, `Price > 0` | ✅ |
| TotalAmount feature engineering | `Quantity * Price` | ✅ |
| IQR outlier detection | `quantile(.25)` / `quantile(.75)` | ✅ |

Objective 1 additionally requires *"identifying key customer purchasing patterns through exploratory data analysis"* — satisfied by `describe()` on Quantity/Price/TotalAmount plus three distribution histograms (report Figures 3.12–3.14). ✅

**Pipeline reproduces exactly.** Re-executed independently:

| Stage | Rows | Removed |
|---|---|---|
| Raw combined (2 files) | 1,067,371 | — |
| After removing missing Customer ID | 824,364 | −243,007 |
| After removing duplicates | 797,885 | −26,479 |
| After removing invalid dates | 797,885 | −0 |
| After removing invalid transactions | 779,425 | −18,460 |
| After IQR outlier removal | **715,863** | −63,562 |
| **Retained** | **67.1%** | |

---

## Objective 2 / Deliverable 2 — RFM + K-means ✅ 6/6

```
[PASS] RFM computed per customer          5,678 customers, R/F/M columns present
[PASS] Cluster assignment present         4 clusters over 5,678 customers
[PASS] Interpretable behavioural profiles Champions, At-Risk High Value,
                                          New / Promising, Lost / Dormant
[PASS] Elbow method (WCSS) recorded       k=2..10, WCSS 8278 -> 2410
[PASS] Silhouette scores recorded         k=2..10
[PASS] No unassigned customers            5,678/5,678 labelled
```

Objective 2 names **both** validation methods — "silhouette scores and the elbow method". Both are present and now persisted to `phase2_outputs/cluster_selection_metrics.csv`:

| k | WCSS (inertia) | Silhouette |
|---|---|---|
| 2 | 8277.72 | 0.4361 |
| 3 | 6139.10 | 0.3457 |
| **4** | **4777.01** | **0.3598** |
| 5 | 4068.34 | 0.3267 |
| … | … | … |
| 10 | 2410.19 | 0.2927 |

Validation exceeds the requirement — also computed: Davies–Bouldin 0.941, Calinski–Harabasz 4852.9, bootstrap ARI 0.956 (min 0.851), seed ARI 0.989, plus a four-algorithm comparison (K-means / Ward / GMM / DBSCAN).

**Resulting segments:**

| Segment | Customers | Share | Avg recency | Avg orders | Avg value |
|---|---|---|---|---|---|
| Champions | 1,186 | 20.9% | 27.6 d | 17.3 | RM 26,342 |
| At-Risk High Value | 1,396 | 24.6% | 232.2 d | 4.8 | RM 6,903 |
| New & Promising | 1,209 | 21.3% | 29.4 d | 2.9 | RM 3,516 |
| Lost / Dormant | 1,887 | 33.2% | 396.0 d | 1.4 | RM 1,302 |

---

## Objective 3 / Deliverable 3 — Predictive modelling ✅ 4/4

```
[PASS] All three required algorithms      Decision Tree, Logistic Regression,
                                          Random Forest
[PASS] All four required metrics          Accuracy, Precision, Recall, F1
[PASS] Algorithmic justification          recommends Logistic Regression;
                                          statistically indistinguishable = True
[PASS] Feature importance, all 3 models    15 rows covering 3 models
```

**Measured performance** (`phase2_outputs/model_comparison.csv`):

| Model | Accuracy | Precision | Recall | F1 | AUC | Overfit gap |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.7179 | 0.7336 | 0.6842 | 0.7081 | 0.7934 | **0.0105** |
| Random Forest | 0.7105 | 0.7304 | 0.6674 | 0.6975 | 0.7961 | 0.0660 |
| Decision Tree | 0.7042 | 0.7462 | 0.6189 | 0.6766 | 0.7853 | 0.0386 |

All three tuned via `GridSearchCV` on identical stratified folds. The justification is substantive rather than a bare ranking: the three are **statistically indistinguishable** (Nadeau–Bengio corrected paired t-test p = 0.895; McNemar exact p = 0.550), so selection rests on the generalisation gap (0.0105 vs 0.0660), interpretability, and deployment simplicity.

---

## Deliverable 4 — Model Evaluation Report ✅ 4/4

Deliverable 4 requires critical assessment of **both** segmentation quality (silhouette, WCSS) **and** predictive performance, including limitation diagnostics.

```
[PASS] Segmentation quality persisted     cluster_selection_metrics.csv
[PASS] Predictive performance persisted   model_comparison.csv
[PASS] Limitation diagnostics documented  PHASE2_AUDIT_REPORT.md — 20,805 chars
[PASS] Diagnostics reproducible           3 executable audit scripts
```

⚠️ **Gap found and fixed during this audit.** WCSS was named explicitly in your deliverable but existed only as console output and pixels inside `elbow_silhouette.png` — not citable, and lost when the terminal closed. `phase2_objective2_rfm_kmeans.py` now writes it to `cluster_selection_metrics.csv`.

Limitation analysis is diagnostic rather than headline-only: silhouette-ceiling analysis with degenerate-solution evidence, accuracy-ceiling analysis across four improvement routes, accuracy decomposed by spend band, cold-start exclusion, and removal of a spurious feature.

---

## Objective 4 / Deliverable 5 / RQ4 — Dashboard ✅ 8/8

```
[PASS] Reads only phase2_outputs          no bare-path read_csv; 6 OUT_DIR refs
[PASS] Segmentation distributions         revenue concentration + RFM panels
[PASS] Segment-specific spending          per-segment value metrics
[PASS] Predictive model results           model comparison chart
[PASS] Strategic recommendations          SEGMENT_CONTENT action cards
[PASS] Colour not sole encoding            table views alongside charts
[PASS] Executes, all 3 pages, 0 exceptions AppTest across all pages
[PASS] DEPLOYED (public URL)               github.com/Sy445/FYP_Project ->
                                          fypproject-tp070227.streamlit.app
```

Deliverable 5 names four required content types — all four present. The dashboard also holds the "pure presentation layer" property: **zero** bare-path `read_csv` calls, so it cannot silently disagree with your report by recomputing anything.

Accessibility verified: table views accompany every chart, so colour never carries meaning alone. Chart colours were validated by computation (`validate_palette.py`) rather than by eye — a four-colour segment scatter was rejected because the yellow/orange pair measured ΔE 13.7 against a 15 floor, and replaced with faceted small multiples.

### ✅ Deployment confirmed

```
git remote -v  ->  origin  https://github.com/Sy445/FYP_Project.git
live URL       ->  https://fypproject-tp070227.streamlit.app/  (HTTP 200, serving)
```

Objective 4's verb is *"to design and **deploy**"*, and RQ4 asks how outputs can be *"visualised **and deployed**"*. Both are now satisfied: the dashboard is designed, functional, and publicly accessible.

The repository excludes the four large source CSVs via `.gitignore` — none are needed at runtime, since the app reads roughly 400 KB from `phase2_outputs/`.

---

## Constraints (§1.6.3) ✅ 5/5

```
[PASS] No real-time / streaming / scheduled processing
       scanned 14 .py files; matches: none
[PASS] No production-grade deployment infrastructure
       no Docker/CI/orchestration files present
[PASS] Lightweight free-tier dependency footprint
       3 runtime deps: streamlit, pandas, plotly
[PASS] UK origin + synthetic geography disclosed in the UI
       'About this data' panel
[PASS] No claim of perfect accuracy; limitations shown
       limitations panel on the prediction page
```

Constraint 4 (*"does not incorporate real-time streaming data or live production deployment"*) verified by scanning every Python file for `APScheduler`, `schedule.every`, `croniter`, `KafkaConsumer`, `websocket`, `st_autorefresh`, `while True`, and `time.sleep`. **Zero matches.** The system is strictly batch.

Constraint 2 (UK generalisability) is disclosed **inside the dashboard UI**, not only in the report — the "About this data" panel states the dataset was adapted from a UK retail dataset and that between-state differences carry no real-world meaning.

---

## Success criteria — measured values

| Criterion | Target | Measured | Status |
|---|---|---|---|
| Silhouette score | > 0.50 | **0.3598** | ❌ Not met |
| Predictive accuracy | > 0.75 | **0.7179** | ❌ Not met |

These are reported as measured. **They are not counted as code failures**, because both were established as genuine ceilings rather than implementation shortfalls:

- **Silhouette** — exceeding 0.5 is only reachable via degenerate solutions (dropping the log transform reaches 0.575 but yields a 9-customer cluster; at k=5, a 1-customer cluster). Ward, GMM and DBSCAN all scored at or below K-means.
- **Accuracy** — tuning → 71.8%; expanding 5 → 16 features → +0.004 F1; gradient boosting → 72.2%; learning curve plateaus at ~2,600 samples.

```
[PASS] Shortfalls disclosed in the dashboard UI (not glossed over)
       silhouette shortfall stated in the 'About this data' panel
```

⚠️ **Gap found and fixed during this audit.** The dashboard previously showed the *favourable* validity statistic (bootstrap ARI 0.956, "highly reproducible") while never mentioning silhouette. The "About this data" panel now states the 0.360 value, that it falls below the 0.5 target, and why — in non-technical language.

---

## A note on the verification script itself

On its first run, `final_check.py` reported a **false FAIL** on the real-time constraint. The cause was a bug in the checker, not in the project code: the script scans every `.py` file for streaming keywords, and it was scanning **itself** — matching its own search pattern.

Verified: every match came from `final_check.py`; zero from anywhere else. Fixed by excluding the file from its own scan. The constraint was never actually violated.

Recorded here because it is worth knowing when you re-run the check, and because a verification tool that cannot be trusted is worse than none.

---

## Fixed during this audit

| Fix | Requirement | Verified |
|---|---|---|
| WCSS persisted to `cluster_selection_metrics.csv` | Deliverable 4 names WCSS explicitly | Obj 2 re-run, exit 0 |
| Dashboard discloses the silhouette shortfall | Success-criteria honesty | AppTest, no exceptions |
| Checker excluded from its own keyword scan | Verification integrity | false positive removed |

---

## Summary

The implementation satisfies every code-side requirement in the proposal. All six scripts execute cleanly, all artefacts regenerate deterministically under `RANDOM_SEED = 42`, the dashboard runs without exceptions across all pages and filter combinations, and no stated exclusion is violated.

The two unmet success criteria are documented honestly in the code, the audit report, and the dashboard UI, with evidence that each represents a genuine ceiling rather than an implementation failure.

**No remaining implementation actions.** Verified 35/35 on 21 August 2026.
