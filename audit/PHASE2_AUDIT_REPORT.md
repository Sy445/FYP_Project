# Phase 2 Modelling — Critical Audit Report

**Project:** Predictive Consumer Segmentation and Spending Behaviour in Malaysian Retail
**Audit date:** 20 August 2026
**Scope:** `validate_malaysian_dataset.py`, `phase2_objective2_rfm_kmeans.py`, `phase2_objective3_predictive_modelling.py`, and `phase2_outputs/`
**Audit scripts (reproducible evidence):** `audit_1_leakage_and_rigour.py`, `audit_2_clustering.py`, `audit_3_accuracy.py`

---

## 1. Verdict

**The modelling work is methodologically sound. No data leakage was found. Nothing needs to be rebuilt before Objective 4.**

Four defects were found and fixed. One was serious enough that it would have put a false claim in your report. Two success criteria in your proposal are not met, and after testing every legitimate route to closing them, **neither gap can be closed honestly**. Section 4 and 5 explain why, and Section 6 gives you draft text to defend that position.

| Area | Status |
|---|---|
| Temporal split integrity | Clean — verified, no leakage |
| Feature/target separation | Clean — verified |
| Dataset validation | Clean — all checks pass |
| `State` feature | **Was a defect — now fixed** |
| Hyperparameter tuning | **Was missing — now added** |
| Model-comparison statistics | **Was a heuristic — now a real test** |
| Segment labelling logic | **Had dead code — now fixed** |
| Silhouette ≥ 0.5 criterion | **Not met — and not honestly reachable** |
| Accuracy ≥ 75% criterion | **Not met — 71.8%; ceiling is in the data** |

---

## 2. Defects found and fixed

### 2.1 SERIOUS — the `State` feature was synthetic noise being modelled as signal

`Dataset process/step1_geography.py` assigns `State` by **weighted random draw per customer**, independent of any purchasing behaviour. It therefore *cannot* carry predictive signal about spending. But v1 of Objective 3 included it as a one-hot feature, and the Logistic Regression assigned it large coefficients:

```
State_Sarawak     -0.5423
State_Kedah        0.5030
State_Labuan      -0.4760
State_Putrajaya    0.4204
```

These sat **above `Frequency` (0.3066)** in the ranked coefficient table. Had that table gone into your report, it would have supported statements like *"customers in Sarawak are significantly less likely to be high spenders"* — a finding drawn entirely from a random number generator, and one an examiner could dismantle in a single question.

**Evidence it is noise:**
- Chi-square test of independence vs the target: χ²(15) = 12.75, **p = 0.622** (no association)
- Ablation — removing `State` changed cross-validated F1 by: Logistic Regression **+0.0021**, Decision Tree **+0.0040**, Random Forest **−0.0001** (i.e. removing it was neutral-to-beneficial)

**Fix:** `State` removed from Objective 3. Do not re-add it. This is worth a sentence in your limitations chapter — it is a genuine and interesting constraint of a synthetically-localised dataset, and stating it proactively is far stronger than being asked about it.

> **Note for Objective 4:** any dashboard chart that breaks spending down by state is showing you the shape of the random draw in `MALAYSIA_GEO_WEIGHTS`, not Malaysian consumer behaviour. Use state for *descriptive* customer-count visuals only, clearly labelled as synthetic geography.

### 2.2 Hyperparameter tuning had never been applied

v1 used hand-picked hyperparameters (`max_depth=6`, `n_estimators=200`). Comparing model *families* on hand-picked settings isn't defensible — a badly-configured tree loses for reasons unrelated to the algorithm. `GridSearchCV` (5-fold stratified, F1-scored) is now applied to all three on identical folds.

| Model | CV F1 before | CV F1 tuned | Gain |
|---|---|---|---|
| Logistic Regression | 0.7175 | 0.7175 | +0.0000 |
| Decision Tree | 0.6533 | 0.7188 | **+0.0655** |
| Random Forest | 0.7000 | 0.7200 | +0.0200 |

The Decision Tree gained substantially — v1 was under-selling it. This does **not** close the accuracy gap (Section 5), but the comparison is now fair.

### 2.3 The "statistically indistinguishable" claim was not actually tested

v1 compared `|mean₁ − mean₂|` against `max(std₁, std₂)`. That is a heuristic, not a test: the CV standard deviation is the spread *across folds*, not the standard error of the difference, and it ignores that both models were scored on the **same folds** (paired data). Two proper tests now run:

- **Nadeau–Bengio corrected resampled paired t-test** — corrects the naive paired t-test, which is anti-conservative because CV training sets overlap between folds (Dietterich, 1998)
- **McNemar's exact test** on the held-out test set — the standard test for two classifiers on one sample

Result: **t = +0.140, p = 0.895** and **McNemar p = 0.550**. The original conclusion was right, but it now rests on evidence rather than a rule of thumb.

### 2.4 Dead code silently dropped Frequency from segment naming

In `label_segment()`, a `frequent` flag was computed and then **never used** — so segments were named on Recency and Monetary only. One third of the RFM method was doing no work in the interpretation layer. This matters substantively: Frequency is what separates a loyal repeat customer from a one-off big-ticket buyer, and those need different retention strategies. Fixed; the labels now use all three dimensions, with a collision guard.

---

## 3. Methodological soundness — leakage and rigour (clean)

All checks pass. Evidence in `audit_1_leakage_and_rigour.py`.

| Check | Result |
|---|---|
| Temporal overlap | **None.** Last observation row `2011-05-31 15:41`, first future row `2011-06-01 07:37` |
| Windows partition the data | 485,876 + 229,987 = 715,863 exactly |
| Features derive from observation window only | Verified by independent re-derivation |
| Recency lookahead | None — min 0, max 546, no negatives |
| Threshold leakage | Train-only median = full median = **RM 345**; **zero** labels change |
| Window lengths | 546 days observation / 191 days future |

Two things are now **stricter than they were**, so the criticism cannot be raised at all:

- The median threshold is now derived from the **training set only**, computed *after* the split.
- The temporal partition is enforced by `assert` statements rather than assumed.

**Scope limitation to state in your report:** 929 customers (27.1% of future-window customers) appear *only* after the cutoff. These "cold-start" customers are necessarily excluded — a behavioural model cannot score a customer with no prior behaviour. Your model predicts spend for **existing** customers, not new acquisitions.

**Overfitting after tuning:**

| Model | Train acc | Test acc | Gap |
|---|---|---|---|
| Logistic Regression | 0.7283 | 0.7179 | **+0.0105** |
| Decision Tree | 0.7428 | 0.7042 | +0.0386 |
| Random Forest | 0.7765 | 0.7105 | +0.0660 |

Random Forest still exceeds the ~0.05 rule of thumb even after tuning. Logistic Regression generalises best by a wide margin — which drives the recommendation in Section 7.

---

## 4. The silhouette gap: 0.360 vs the 0.5 criterion

**Honest answer: 0.5 is reachable numerically, but only via solutions that are degenerate or artefactual. It is not honestly reachable, and the criterion itself was mis-specified.**

### 4.1 Be aware of how 0.360 reads

Under the Kaufman & Rousseeuw (1990) interpretation bands — the ones most examiners will know — 0.26–0.50 is described as *"weak structure, could be artificial."* **Do not paper over this.** Your defence is not that 0.360 is secretly good; it is that **silhouette is the wrong criterion for this problem**, and you have the evidence to prove it.

### 4.2 What "hitting 0.5" would actually require

| Route | Silhouette | Why it is not legitimate |
|---|---|---|
| Drop the log transform (raw + StandardScaler), k=4 | **0.575** | Clusters are 33% / 60% / 6.8% / **9 customers**. The score is high because K-Means isolates extreme outliers into micro-clusters. |
| Same, k=5 | **0.575** | Contains a cluster of **one single customer**. |
| QuantileTransformer, k=2 | **0.717** | Forces each feature to an exact Gaussian, manufacturing separation from the transform, not from customers. Also k=2 = the trivial "high vs low spend" split. |
| Drop a dimension (R+F only) | 0.434 | Fewer dimensions dilute distances less — a geometric artefact. Also no longer *RFM* segmentation. |

A 9-customer "segment" is not a segment. You cannot build a Malaysian retail strategy on it, and it would not survive one question in a viva.

### 4.3 Evidence that ~0.36 is the genuine ceiling

Every algorithm family plateaus in the same range on the honest representation:

| Algorithm | k=3 | k=4 | k=5 |
|---|---|---|---|
| **K-Means** | 0.3457 | **0.3598** | 0.3267 |
| Hierarchical (Ward) | 0.3038 | 0.3126 | 0.2980 |
| Gaussian Mixture | 0.2851 | 0.2189 | 0.2158 |
| DBSCAN | collapses to 1–2 clusters at every ε tested | | |

K-Means at k=4 is the **best** honest option available, not a lazy default. DBSCAN — which is specifically designed to find density-separated groups — finding none is direct evidence that no such groups exist.

> **One caution on your own evidence:** the Hopkins statistic came out at 0.95, which *looks* like "strong clustering tendency." **Do not cite it as support.** Hopkins is inflated for skewed, heavily-concentrated data like RFM: because values pile into one corner of the bounding box, uniform comparison points land far from any real customer, driving H toward 1 regardless of whether distinct groups exist. It does not distinguish "separated clusters" from "one dense concentrated blob." Citing it would hand an examiner an easy correction.

### 4.4 What to report instead: stability

For an applied segmentation, the question that matters is not *"are the clusters geometrically separated?"* but *"are they reproducible?"* Adjusted Rand Index against the reference solution:

- **Bootstrap (20 resamples): mean ARI = 0.956**, sd 0.031, min 0.851
- **Across 5 random seeds: mean ARI = 0.989**, min 0.985

Conventional stability threshold is ARI > 0.75. Your segments clear it comfortably. **This is the validity claim to lead with.** Also now reported: Davies–Bouldin 0.941 and Calinski–Harabasz 4852.9, so you are not resting on a single metric.

---

## 5. The accuracy gap: 71.8% vs the 75% criterion

**Honest answer: ~72% is a genuine data ceiling. Every legitimate route was tested. None closes a 3.2-point gap.**

| Route tested | Result |
|---|---|
| Class imbalance | **Not the issue.** Target is 50/50 by construction; baseline is 50.0%. SMOTE/class weights would be inappropriate. |
| Hyperparameter tuning (GridSearchCV) | Best tuned test accuracy **71.8%** |
| Feature engineering (5 → 16 features) | **+0.004 CV F1 at best**; went *down* for the Decision Tree |
| Stronger model class (gradient boosting, diagnostic only) | **72.2%**, AUC 0.781 — no better |
| More training data (learning curve) | Validation accuracy **plateaus at ~2,600 samples** (0.7341 → 0.7326) |

The engineered feature set included inter-purchase-time mean/sd, max and sd of invoice value, average items per invoice, distinct products, total quantity, active months, last-90-day spend, spend momentum ratio, and purchase rate. **Eleven additional behavioural features bought essentially nothing.** That null result is itself a reportable finding: the limit is the intrinsic predictability of individual consumer behaviour, not insufficient feature extraction.

### 5.1 Where the model actually succeeds and fails

This is the most useful table in the whole audit:

| Actual future spend | N | Accuracy |
|---|---|---|
| RM 0 (fully churned) | 448 | **0.7634** |
| RM 0.01 – 172 | 9 | 0.6667 |
| RM 172 – 690 (ambiguous band) | 58 | 0.4828 |
| RM 690 – 3,450 (moderate) | 244 | **0.5697** |
| Above RM 3,450 (clear high) | 191 | **0.8743** |
| **All** | 950 | 0.7179 |

The model is strong at the **behavioural extremes** — 76% on customers who fully churn, 87% on heavy spenders — and near chance in the **moderate middle**, where future spend is genuinely volatile and only weakly determined by past RFM.

Note the honest nuance: the narrow ambiguous band around the threshold is real but **only ~6% of the test set**, so "the median threshold is to blame" is *not* a sufficient explanation. The dominant error source is the moderate-spender band (~26% of customers). The correct claim is about *where behaviour is predictable*, not about an artefact of thresholding.

This is exactly what **ROC-AUC 0.79** alongside accuracy 0.72 describes — and AUC is the more informative headline, because it is threshold-independent and measures how well the model *ranks* customers by spend propensity. AUC in the 0.7–0.8 range is conventionally described as acceptable/fair discrimination.

### 5.2 The route that *would* hit 75% — and why I recommend against it

Reframing the target as top-quartile "VIP" prediction:

```
Accuracy          : 0.8484   <- exceeds 75%
Balanced accuracy : 0.7441   <- the honest number
Recall            : 0.5359   <- misses nearly half of actual VIPs
AUC               : 0.8773
```

**A model that predicts "not VIP" for every single customer already scores 75.0% accuracy on this target.** Hitting the proposal's wording this way would be close to meaningless. If you do adopt the VIP framing — and it is a legitimate business question, arguably more useful than a median split — you **must** report balanced accuracy and recall alongside, or the result is indefensible.

My recommendation: **keep the median-split target, report ~72% honestly with the ceiling evidence above.** Optionally present the VIP framing as a secondary analysis with full honest metrics. Reporting a criterion as unmet, with rigorous evidence for why, is stronger work than reporting a number you cannot defend.

---

## 6. Draft justification text for your report

Adapt freely — this is scaffolding, not final prose.

> **On segmentation validity.** The segmentation produced a silhouette score of 0.360 at k=4, below the 0.5 threshold specified in the project proposal. Investigation established that this reflects a property of RFM data rather than a deficiency in the implementation. RFM variables describe a continuous behavioural gradient rather than naturally separated groups, and no clustering algorithm tested — K-Means, Ward hierarchical, Gaussian mixture, or density-based DBSCAN — exceeded a silhouette of 0.36 on the validated representation; DBSCAN failed to identify any density-separated groups at any parameterisation. Configurations that did exceed 0.5 were found to be degenerate: omitting the log transform yielded 0.575 but produced a cluster containing nine customers, and at k=5 a cluster containing one. Such solutions optimise the metric by isolating outliers and have no interpretive or operational value.
>
> The proposal's 0.5 criterion was therefore mis-specified: it presumes a naturally-separated cluster structure that RFM data does not exhibit. Following established practice for applied segmentation, cluster validity is instead evidenced through reproducibility. Adjusted Rand Index against bootstrap resamples averaged 0.956 (min 0.851) and across random initialisations averaged 0.989, all substantially above the conventional 0.75 stability threshold. Davies–Bouldin (0.941) and Calinski–Harabasz (4,852.9) indices are reported alongside silhouette to avoid dependence on a single measure. The four resulting segments are stable, interpretable, and operationally distinct.

> **On predictive performance.** The best model achieved 71.8% accuracy against a 75% criterion, with ROC-AUC of 0.793. Multiple routes to improvement were tested and none closed the gap: hyperparameter optimisation via grid search yielded 71.8%; expanding the feature set from five to sixteen behavioural variables changed cross-validated F1 by at most 0.004; a gradient-boosting model included as a diagnostic reached 72.2%; and learning-curve analysis showed validation accuracy plateauing at approximately 2,600 training instances, indicating the model is limited by available signal rather than sample size.
>
> Disaggregating accuracy by realised future spend clarifies the result. The model classifies correctly for 76.3% of fully-churned customers and 87.4% of high-spending customers, but achieves only 57.0% in the moderate-spend band. Predictive power is therefore concentrated at the behavioural extremes, where it also carries the greatest operational value — identifying likely churners and likely high-value customers — while mid-range spending behaviour proves substantially less determined by historical RFM. This pattern is consistent with the observed combination of moderate accuracy and acceptable AUC, and reflects a documented characteristic of individual-level consumer behaviour prediction rather than a modelling deficiency.

> **On geographic variables.** The `State` variable was excluded from predictive modelling. Because state was assigned by weighted random draw during the Malaysian-context adaptation, independent of transaction behaviour, it cannot carry genuine predictive signal. A chi-square test confirmed no association with the target (p = 0.622), and ablation showed removal left performance unchanged or marginally improved. Retaining it would have produced coefficients inviting spurious geographic interpretation.

⚠️ **Citations you must source yourself.** I have deliberately not fabricated reference details. Claims needing a citation: the silhouette interpretation bands (Kaufman & Rousseeuw, 1990, *Finding Groups in Data*); the CV paired t-test critique (Dietterich, 1998); the corrected resampled t-test (Nadeau & Bengio, 2003); AUC interpretation bands (commonly attributed to Hosmer & Lemeshow, *Applied Logistic Regression*); and any claim about *typical* silhouette/accuracy values in published RFM studies — **find real papers in your field for that last one; do not cite a range on my say-so.**

---

## 7. Recommended model

**Logistic Regression.**

The three models are **not statistically distinguishable** (corrected paired t-test p = 0.895; McNemar p = 0.550), so the choice rests on qualitative grounds — and on those, Logistic Regression wins clearly:

| Criterion | Logistic Regression |
|---|---|
| Test accuracy / F1 | Highest of the three (0.7179 / 0.7081) |
| Overfitting gap | **0.0105** vs Random Forest's 0.0660 |
| Interpretability | Direct signed coefficients on standardised features |
| Dashboard deployment | Simplest — a linear scoring function |

Its coefficients are also behaviourally coherent, which is a useful sanity check: `Monetary +0.954`, `Recency −0.501`, `Frequency +0.292`. More days since last purchase lowers high-spend probability; higher past spend raises it.

---

## 8. Final checklist — Objective 4 readiness

Full clean re-run from a cleared `phase2_outputs/`, all three scripts, exit code 0, no errors on stderr:

| Item | Status |
|---|---|
| `validate_malaysian_dataset.py` | Exit 0 — all 6 validation checks PASS |
| `phase2_objective2_rfm_kmeans.py` | Exit 0 |
| `phase2_objective3_predictive_modelling.py` | Exit 0 |
| Row count 715,863 / no nulls / CustomerID consistent | Verified |
| Skew preserved (Q 6.82, P 3.17, TA 0.933) | Verified |
| `customer_rfm_segments.csv` | 5,678 customers + cluster + label |
| `segment_profile_summary.csv` | 4 segments profiled |
| `model_comparison.csv` | 3 models × 11 metrics |
| `elbow_silhouette.png`, `cluster_distributions.png`, `segments_scatter.png` | Present |
| 3 × confusion matrix PNGs | Present |
| Reproducibility | `RANDOM_SEED = 42` throughout |

**Both objectives are complete and traceable. You are clear to build Objective 4.**

Two things to carry into the dashboard:
1. Use **Logistic Regression** as the scoring model.
2. Do **not** build state-level *behavioural* analytics — geography is synthetic. Descriptive customer counts by state are fine if labelled as such.

---

## 9. What to raise with your supervisor

Both unmet criteria are worth raising **before** submission, not defending afterwards:

1. **The silhouette criterion was mis-specified.** Propose replacing "silhouette > 0.5" with a stability-based criterion (ARI > 0.75, which you exceed at 0.956). You have the evidence that 0.5 is unreachable without degenerate clusters.
2. **The accuracy criterion is ~3 points above the data ceiling.** Propose either reporting 71.8% with AUC 0.793 and the ceiling evidence, or adding the VIP framing as a documented secondary analysis with balanced accuracy reported.

Supervisors generally respond well to a student who identifies an over-optimistic criterion, proves it empirically, and proposes a rigorous alternative. That is a stronger position than quietly missing a target — and considerably stronger than hitting one with a nine-customer cluster.
