# FYP Compliance Check

**Checked:** 21 August 2026 · Every verdict was established by reading and executing the actual files, not by accepting a claim that something was done.
**Revision 2** — now checked against the proposal text you supplied (Chapters 1–2), which resolved Section 1.5 and surfaced three discrepancies not visible from the paraphrase.

---

## What I could and could not check

You supplied Chapters 1 and 2. That resolved the Research Questions and gave me the **exact wording** of Objectives (§1.4), Deliverables (§1.6.2) and Constraints (§1.6.3) — all now verified against code.

**Still outside my view: Chapter 3 (Methodology).** Your message was truncated partway through §2.4. The success criteria you asked about — *silhouette > 0.5* and *accuracy > 75%* — live in Chapter 3 (Phase 1, Business Understanding). I am taking those two thresholds **on your word**; I have not seen them stated. Notably, §1.4 Objective 2 requires evaluation "through silhouette scores and the elbow method" but sets **no numeric threshold**, so the 0.5 figure appears only in Chapter 3.

---

## Summary

| Item | Verdict |
|---|---|
| **Obj 1** — Preprocess + EDA | ✅ **PASS** |
| **Obj 2** — RFM + K-means, silhouette & elbow | ✅ **PASS** |
| **Obj 3** — 3 algorithms, 4 metrics | ✅ **PASS** |
| **Obj 4** — Dashboard **deployed** | ⚠️ **PARTIAL — action required** |
| **Del 1** — Preprocessing pipeline | ✅ **PASS** |
| **Del 2** — Segmentation model | ✅ **PASS** |
| **Del 3** — Predictive suite | ✅ **PASS** |
| **Del 4** — Model Evaluation Report | ✅ **PASS** *(WCSS gap fixed)* |
| **Del 5** — Dashboard content | ✅ **PASS** |
| **RQ1** — Data quality challenges | ✅ **PASS** |
| **RQ2** — RFM + K-means effectiveness | ✅ **PASS** |
| **RQ3** — Which algorithm is best | ✅ **PASS** |
| **RQ4** — Visualised **and deployed** | ⚠️ **PARTIAL** |
| Criterion — Silhouette > 0.5 | ❌ **NOT MET (0.360)** — honestly documented |
| Criterion — Accuracy > 75% | ❌ **NOT MET (71.8%)** — honestly documented |
| Constraint — No real-time processing | ✅ **PASS** |
| Constraint — No production deployment | ✅ **PASS** |
| Constraint — UK generalisability stated | ✅ **PASS** |
| **§1.6.3 Inclusion 5** — DOSM contextualisation | ❌ **FAIL — not implemented** |
| **§1.6.1** — Dataset description accuracy | ❌ **FAIL — report is factually wrong** |
| **Malaysian adaptation** — documented in proposal | ❌ **ABSENT — major gap** |

**Three blocking items**, detailed below.

---

## 🔴 New findings from the proposal text

These were invisible from your paraphrase and are the most important part of this revision.

### N1. The DOSM secondary dataset was never implemented — ❌ FAIL

Your proposal commits to DOSM as a **dataset** in four separate places:

| Location | Wording |
|---|---|
| Abstract | *"supplemented by aggregated retail statistics from the Department of Statistics Malaysia (DOSM)"* |
| §1.6.1 | *"A secondary dataset comprising aggregated retail statistics from DOSM **is employed** to contextualise findings"* |
| §1.6.3 Inclusions #5 | *"Contextualisation of findings using Malaysian retail sector statistics from DOSM"* |
| §2.3 | *"By supplementing the primary Kaggle dataset with Malaysian retail statistics from DOSM, this investigation enhances the contextual relevance of its findings"* |

**Verified reality:** no DOSM file exists anywhere in the project. The only occurrences are two *code comments* in `Dataset process/step1_geography.py`:

```
line 18:  are the next-largest urban/retail hubs (DOSM population estimates).
line 54:  (blend of DOSM population share + urban e-commerce activity skew)
```

That is DOSM used as an **unsourced justification for synthetic weights**, not as a secondary dataset contextualising findings.

Note the distinction: you *do* cite DOSM correctly as **literature** in Chapter 1 (the 97.4% establishments figure, the <18% analytics-adoption figure). That is legitimate and unaffected. The failure is specifically the promised **secondary dataset** and the Inclusion-list commitment.

**Two honest routes:**
- **(a) Implement it.** Download a DOSM retail-trade indicator series, save it in the project, and add a short contextualisation section comparing your segment revenue concentration against national retail patterns. Genuinely strengthens Objective 1 and RQ4.
- **(b) Rewrite the commitment.** Amend the Abstract, §1.6.1, §1.6.3 and §2.3 to state that DOSM is used as a **contextual literature source**, not a secondary dataset. Remove Inclusion #5 or restate it.

Either is defensible. Leaving all four claims standing while no DOSM data exists is not — an examiner checking your Inclusions list against your artefacts will find this immediately.

### N2. §1.6.1's dataset description is factually wrong — ❌ FAIL

> §1.6.1: *"the Online Retail dataset obtained from Kaggle… comprises transactional records from a UK-based online retail company spanning the period **from December 2010 to December 2011**."*

**Verified reality:**

| | Proposal says | Code actually uses |
|---|---|---|
| Files | One Kaggle dataset | **Two** (`online_retail.csv` + `online_retail_.csv`) |
| Period | Dec 2010 → Dec 2011 | **Dec 2009 → Dec 2011** |
| Raw rows | — | 1,067,371 → cleaned to 715,863 |

`data_preprocessing.py` lines 18–19 read **both** files and concatenate them. This is the Online Retail **II** dataset (two annual sheets), not the single-year Online Retail dataset your text describes.

This matters beyond pedantry: **Objective 3's temporal split depends on it.** The 546-day observation window plus 191-day future window requires ~2 years. On a single Dec 2010–Dec 2011 year, the design in your code would be impossible. Your report currently describes a dataset that could not have produced your results.

**Fix:** update §1.6.1 to state two source files, the Dec 2009 – Dec 2011 span, and 1,067,371 raw records. Also worth adding: the two files overlap for nine days (2010-12-01 → 2010-12-09), producing 34,335 exact duplicate rows removed by `drop_duplicates()`.

### N3. The Malaysian-context adaptation is absent from the proposal — ❌ MAJOR GAP

Your proposal describes UK Kaggle data, justified by structural isomorphism with Malaysian POS systems, contextualised by DOSM statistics. It **never mentions** creating a synthetic Malaysian dataset.

But Objectives 2, 3 and 4 all run on `malaysian_context_online_retail.csv`, produced by five `Dataset process/` scripts that remap geography to Malaysian states, convert GBP→MYR at 5.50, apply psychological price rounding, and localise product descriptions.

You noted this had supervisor approval — that is not in question. The issue is that **Chapter 3 as written documents only the UK preprocessing** (Figures 3.2–3.14 and the §3.4 narrative all describe `data_preprocessing.py`), so a reader cannot trace how you got from the cleaned UK file to the dataset the modelling actually used.

**Fix:** add a §3.4.5 (or similar) documenting the adaptation — the six transformation steps, the fixed FX rate, customer-level state assignment, and critically the **validation evidence** that shape was preserved (skew: Quantity 6.82→6.82, Price 3.17→3.17, TotalAmount 0.934→0.933; row count and Customer ID set identical). That evidence already exists in `audit/validate_malaysian_dataset.py`; it just is not in your report.

Also update §1.6.1 and Constraint #2, which currently describe UK data as the analysis input.

---

## Research questions (§1.5) — now verifiable

### RQ1 — *"What data quality challenges are present… and how can systematic preprocessing enhance analytical reliability?"* — ✅ PASS

Fully answerable with quantified evidence. Re-executed the pipeline independently; it reproduces to exactly 715,863:

| Stage | Rows | Removed |
|---|---|---|
| Raw combined | 1,067,371 | — |
| After removing missing Customer ID | 824,364 | −243,007 |
| After removing duplicates | 797,885 | −26,479 |
| After removing invalid dates | 797,885 | −0 |
| After removing invalid transactions | 779,425 | −18,460 |
| After IQR outlier removal | **715,863** | −63,562 |
| **Retained** | **67.1%** | |

The audit work strengthens this answer with challenges your report has not yet mentioned: 34,335 cross-file duplicates from the nine-day overlap, and a date-format inconsistency between the cleaned (ISO) and Malaysian (D/M/YYYY) files where mis-parsing corrupts silently. Both are excellent RQ1 material.

### RQ2 — *"How effectively can RFM + K-means identify meaningful and distinct customer segments?"* — ✅ PASS

Answerable, and the honest answer is genuinely interesting because it splits the question:

- **Meaningful: yes.** Four interpretable segments, highly reproducible (bootstrap ARI 0.956, seed ARI 0.989), with sharply different revenue behaviour — Champions are 20.9% of customers and 65.7% of revenue.
- **Distinct: no.** Silhouette 0.360. Ward, GMM and DBSCAN all scored at or below K-means; solutions exceeding 0.5 are degenerate (a 9-customer cluster at k=4; a 1-customer cluster at k=5).

The word "distinct" in your RQ is precisely what silhouette measures, so this is a direct, evidence-backed answer rather than a miss.

### RQ3 — *"Which algorithm delivers the most accurate and reliable predictions?"* — ✅ PASS

Answerable, and your RQ's phrasing — *accurate **and reliable*** — is well served, because the two come apart:

| Model | Accuracy | F1 | AUC | Overfit gap |
|---|---|---|---|---|
| **Logistic Regression** | 0.7179 | 0.7081 | 0.7934 | **0.0105** |
| Random Forest | 0.7105 | 0.6975 | 0.7961 | 0.0660 |
| Decision Tree | 0.7042 | 0.6766 | 0.7853 | 0.0386 |

On **accuracy** the three are statistically indistinguishable (Nadeau–Bengio p = 0.895; McNemar p = 0.550). On **reliability** Logistic Regression wins decisively — a 0.0105 train–test gap versus Random Forest's 0.0660. Answering "they tie on accuracy; Logistic Regression wins on reliability" is a stronger response than naming a winner from a table.

### RQ4 — *"How can analytical outputs be effectively visualised **and deployed** to support managerial decision-making…?"* — ⚠️ PARTIAL

**Visualised: fully answered.** Three-page dashboard, plain language, verified across a 10-case matrix with zero exceptions.

**Deployed: not answered.** The RQ contains the word "deployed", and nothing is deployed. Same root cause as Objective 4.

---

## Objectives (§1.4)

**1. Preprocess + "identifying key customer purchasing patterns through exploratory data analysis" — ✅ PASS.** Pipeline verified (funnel above). The EDA clause is satisfied: `describe()` on Quantity/Price/TotalAmount plus three distribution histograms (lines 121, 154–179), matching report Figures 3.12–3.14.

**2. RFM + K-means, validity "through silhouette scores and the elbow method" — ✅ PASS.** Both required methods present, and exceeded. Now persisted in `cluster_selection_metrics.csv`:

```
k,WCSS_inertia,Silhouette
2,8277.72,0.4361
3,6139.10,0.3457
4,4777.01,0.3598
...
```
Plus Davies–Bouldin 0.941, Calinski–Harabasz 4852.9, bootstrap ARI 0.956, and a four-algorithm comparison.

**3. Three algorithms with "accuracy, precision, recall, and F1-score" — ✅ PASS.** All three implemented, `GridSearchCV`-tuned on identical folds, all four required metrics in `model_comparison.csv` (plus AUC and overfit gap).

**4. "To design and **deploy** an interactive web-based dashboard" — ⚠️ PARTIAL.** Your proposal's verb is confirmed as *deploy*. Verified:

```
git rev-parse --is-inside-work-tree  ->  NOT a git repository
git remote -v                        ->  no remote
```

No repo, no remote, no URL. Designed and functional: yes. Deployed: no.

---

## Deliverables (§1.6.2)

**1. Preprocessing pipeline — ✅ PASS.** All six named elements verified in `data_preprocessing.py`: missing values (`dropna`), duplicates (`drop_duplicates`), type standardisation (`to_datetime`/`astype`), invalid transactions (`Quantity>0 & Price>0`), TotalAmount engineering, IQR outliers.

**2. Segmentation model — ✅ PASS.**

| Segment | Customers | Share | Avg recency | Avg orders | Avg value |
|---|---|---|---|---|---|
| Champions | 1,186 | 20.9% | 27.6 d | 17.3 | RM 26,342 |
| At-Risk High Value | 1,396 | 24.6% | 232.2 d | 4.8 | RM 6,903 |
| New & Promising | 1,209 | 21.3% | 29.4 d | 2.9 | RM 3,516 |
| Lost / Dormant | 1,887 | 33.2% | 396.0 d | 1.4 | RM 1,302 |

**3. Predictive suite — ✅ PASS.** See RQ3.

**4. Model Evaluation Report — ✅ PASS** *(gap found and fixed)*. You named **within-cluster sum of squares** explicitly; WCSS existed only in console output and inside a PNG. Now written to `phase2_outputs/cluster_selection_metrics.csv`. Limitation diagnostics are substantive: silhouette-ceiling analysis, accuracy-ceiling across four improvement routes, accuracy decomposed by spend band, cold-start exclusion, and removal of a spurious feature.

**5. Dashboard content — ✅ PASS.** All four required content types present (segmentation distributions; segment-specific spending; predictive results; strategic recommendations), in non-technical language.

---

## Success criteria

**Silhouette > 0.5 — ❌ NOT MET (0.360).** Documentation honest: code comments, `audit_2_clustering.py`, and `PHASE2_AUDIT_REPORT.md` §4 with draft prose.

⚠️ **Gap found and fixed:** the dashboard showed the *favourable* statistic (ARI 0.956, "highly reproducible") and never mentioned silhouette — the exact asymmetry you asked me to catch. The "About this data" panel now states 0.360, that it falls below the 0.5 target, and why.

**Accuracy > 75% — ❌ NOT MET (71.8%).** Established as a data ceiling: tuning → 71.8%; 5→16 features → +0.004 F1; gradient boosting → 72.2%; learning curve plateaus at ~2,600 samples. Honestly framed throughout, including in the dashboard.

**Functional non-technical dashboard — ✅ PASS on functionality.** Filters demonstrably drive figures (Champions-only → 1,186 customers / RM 31.2M / 65.7% of revenue). Accessible to you locally; not to anyone else until deployed.

---

## Constraints (§1.6.3)

**Constraint 4 — no real-time / live deployment — ✅ PASS.** Searched all Python for `schedule`, `cron`, `kafka`, `stream`, `websocket`, `while True`, `time.sleep`, `st_autorefresh`, `APScheduler`: no matches. Strictly batch.

**No production-grade deployment — ✅ PASS.** No Docker, CI/CD, database, or auth. `requirements.txt` is three libraries. Streamlit Community Cloud free tier sits inside your boundary.

**Constraint 2 — UK generalisability — ✅ PASS**, and disclosed inside the dashboard UI, not only the report. ⚠️ But see N3: the constraint's wording describes UK data as the analysis input, which is no longer accurate.

**Exclusion 4 — no claims of 100% accuracy — ✅ PASS, strongly.** Every metric ships with limitations; the dashboard leads its prediction page with a ceiling explanation.

---

## Action list

### 🔴 Blocking

**1. Deploy the dashboard** (Obj 4 + RQ4 — both use the word "deploy").
```bash
git init
printf 'online_retail.csv\nonline_retail_.csv\ncleaned_combined_online_retail.csv\nmalaysian_context_online_retail.csv\n' > .gitignore
git add app.py requirements.txt .streamlit/ phase2_outputs/ *.py *.md
git commit -m "FYP dashboard"
```
Push to a **public** repo → [share.streamlit.io](https://share.streamlit.io) → point at `app.py`. Put the URL in your report. The `.gitignore` matters: none of those CSVs are needed (the app reads ~400 KB from `phase2_outputs/`) and two exceed GitHub's 50 MB warning.

**2. Resolve the DOSM commitment** (N1) — implement it or rewrite the four claims.

**3. Correct §1.6.1's dataset description** (N2) — two files, Dec 2009 – Dec 2011, 1,067,371 raw rows.

**4. Document the Malaysian adaptation in Chapter 3** (N3) — the six steps plus the preservation evidence that already exists.

### 🟡 Recommended

**5. Agree the two unmet criteria with your supervisor before submission.** Propose replacing "silhouette > 0.5" with a stability criterion (ARI > 0.75; you achieve 0.956). `PHASE2_AUDIT_REPORT.md` §9 has the framing.

**6. Close the adaptation reproducibility gap** — the five `Dataset process/` scripts hardcode sandbox paths, so the Malaysian dataset cannot currently be regenerated. Fix or state as a limitation. `DATA_LINEAGE.md` §6.3.

**7. Rename the raw files** — `online_retail_.csv` is the *earlier* period, so alphabetical order reverses chronological.

**8. Send me Chapter 3** if you want the success criteria and CRISP-DM phase mapping (Table 3.1) verified rather than assumed.

### ✅ Fixed during this check

- WCSS persisted to `cluster_selection_metrics.csv` (Deliverable 4)
- Dashboard now discloses the silhouette shortfall alongside the favourable stability figure

Both verified: Objective 2 exits 0, dashboard raises no exceptions.
