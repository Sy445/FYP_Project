# Supervisor Meeting — Prep Sheet

**Your position is strong.** Phase 2 is complete, audited, and the dashboard works. You have two unmet criteria and three report discrepancies — all of which you found yourself, before he did. That is the single most important framing point below.

---

## 🌙 Do tonight (~20 minutes)

### 1. Deploy the dashboard — highest value thing you can do

"Here's the live URL" and "it runs on my laptop" are two completely different meetings. It also closes your last code gap so you can say *35/35 checks passing*.

```bash
cd "…/FYP_Project"
git init
printf 'online_retail.csv\nonline_retail_.csv\ncleaned_combined_online_retail.csv\nmalaysian_context_online_retail.csv\n' > .gitignore
git add app.py requirements.txt .streamlit/ phase2_outputs/ *.py *.md
git commit -m "FYP dashboard"
```
Push to a **public** GitHub repo → [share.streamlit.io](https://share.streamlit.io) → point at `app.py`.

Then re-run `python final_check.py` and screenshot the 35/35.

⚠️ **Test the live URL on your phone** before the meeting. Streamlit Cloud sleeps idle apps — open it 10 minutes before you meet so it's warm, not showing a loading spinner while he watches.

### 2. Have these open in tabs, in this order

1. The **live dashboard**
2. `CODE_COMPLIANCE.md` — the 34/35 (or 35/35) result
3. `PHASE2_AUDIT_REPORT.md` **§4** — the degenerate-cluster table
4. `phase2_outputs/model_comparison.csv`
5. A terminal in the project folder

### 3. Print or write down the three decisions you need

Do not leave without answers to these. They're at the bottom of this sheet.

---

## 🎯 The meeting — suggested flow (~25 min)

### Open with the agenda, not the work (30 seconds)

> "Phase 2 is complete — segmentation, predictive modelling, and the dashboard are all built and deployed. I ran a full audit against the proposal and found three things I need your decision on. Can I show you the working system first, then walk through those?"

This does three things: signals completion, signals self-audit, and tells him where the meeting is going. Supervisors relax when they know there's a structure.

### 1 · Demo the dashboard (5 min) — earn attention first

Show, don't describe. Suggested path:

| Step | What to say |
|---|---|
| **Overview page** | "5,678 customers, RM 47.6M revenue." |
| Point at the headline | **"21% of customers generate 66% of revenue."** |
| Point at the At-Risk banner | **"RM 9.6M sits in customers who used to spend well and have gone quiet."** |
| **Filter to Champions** | "Filters drive everything — this is 65.7% of revenue." |
| **Segments page** | Scroll to an action card, open "Recommended action" |
| Read one line aloud | "Week 2 — single-use voucher, 15% max, 14-day expiry. The deadline matters more than the discount." |
| **Prediction page** | "Three models. The recommendation isn't the top row of the table — I'll explain why." |

**Lead with the RM 9.6M number.** It's the most concrete thing in the whole project and it lands immediately with anyone.

### 2 · Results (5 min)

**Segmentation** — four segments, and the honest headline:

> "The segments are highly stable — bootstrap ARI 0.956 — but not geometrically well-separated. Silhouette is 0.360, and I want to talk about that."

**Prediction** — the finding that shows judgment:

> "All three models land at 70–72%. I tested whether the differences are real — they aren't. Corrected paired t-test p = 0.90, McNemar p = 0.55. So I didn't pick the top row. I recommended Logistic Regression because its train–test gap is 0.0105 versus Random Forest's 0.0660, and because its coefficients are explainable."

That paragraph is the strongest 20 seconds in your project. It shows you understand that a metric ranking isn't automatically a result.

### 3 · The two unmet criteria (8 min) — the real conversation

**Do not bury this. Lead into it deliberately.** How you frame it decides whether it reads as a failure or as research maturity.

> "I set two success criteria in my proposal: silhouette above 0.5 and accuracy above 75%. I hit neither. I spent significant effort establishing *why*, and I believe both are ceilings rather than execution failures. Can I show you the evidence and propose an alternative?"

**On silhouette — show the degenerate-cluster table.** This is your single most persuasive artifact:

> "I *can* hit 0.575 — by dropping the log transform. But look at the cluster sizes: 33%, 60%, 6.8%, and **nine customers**. At k=5 one cluster contains a single customer. That's K-means isolating outliers, not segmentation. I also tested Ward, Gaussian Mixture and DBSCAN — all scored at or below K-means, and DBSCAN found no density-separated groups at any setting. RFM describes a continuum, not four natural types."

Then propose the alternative:

> "I'd argue the 0.5 criterion was mis-specified — it assumes separated clusters exist in RFM data. For an applied segmentation the question that matters is reproducibility, and there the segments are strong: bootstrap ARI 0.956, seed ARI 0.989, both well above the 0.75 convention. Could I replace the silhouette criterion with a stability criterion?"

**On accuracy:**

> "71.8% against a 75% target. I tested every route: hyperparameter tuning, expanding from 5 to 16 behavioural features, a gradient-boosting model as a ceiling probe, and a learning curve. Tuning gave 71.8%. Eleven extra features moved F1 by 0.004. Gradient boosting reached 72.2%. The learning curve flattens at about 2,600 samples, so more data wouldn't help either."

Then the nuance that shows depth:

> "Disaggregating by actual spend is more revealing than the headline: 76% accuracy on customers who fully churn, 87% on heavy spenders, but 57% in the moderate middle. The model is strong exactly where the business value is — identifying churners and high-value customers. AUC is 0.79."

And be straight about the tempting shortcut:

> "I could hit 75% by reframing the target as top-quartile VIP prediction — that gives 84.8%. But the majority-class baseline on that target is already 75%, so predicting 'not VIP' for everyone would 'pass'. Balanced accuracy is 74.4% and recall 0.54. I didn't want to claim a number I can't defend."

**Volunteering that last point is worth more than the 3 points of accuracy you'd have gained.**

### 4 · Three report discrepancies (5 min) — decisions needed

Present these as things you caught:

> "I audited my code against the proposal line by line and found three inconsistencies in the report I need to fix."

**(a) DOSM.** *"I committed to DOSM as a secondary dataset in four places — the abstract, §1.6.1, the Inclusions list, and §2.3. I've cited DOSM correctly as literature in Chapter 1, but there's no DOSM dataset in the project. Should I source real DOSM retail data and add a contextualisation section, or rewrite those four claims to describe DOSM as a contextual literature source?"*

**(b) Dataset description.** *"§1.6.1 says the dataset spans December 2010 to December 2011, but I actually used two files spanning December 2009 to December 2011 — Online Retail II. My Objective 3 temporal split needs two years, so the dataset as described couldn't have produced my results. I'll correct §1.6.1."*

**(c) The Malaysian adaptation.** *"Chapter 3 documents the UK preprocessing only, but Objectives 2–4 run on the Malaysian-context dataset. I have the validation evidence — skewness preserved to three decimals, identical row count and customer set — but it isn't in the report. How much detail do you want, and should it be a new subsection in Chapter 3?"*

### 5 · Close (2 min)

> "Can I confirm the actions: [read them back]. Is there anything else you want before the next milestone?"

---

## ❓ The questions you must get answered

Write the answers down in the meeting.

| # | Question | Why it matters |
|---|---|---|
| **1** | Can I replace *"silhouette > 0.5"* with a stability criterion (**ARI > 0.75**, achieved 0.956)? | Changes how Chapter 4 is written |
| **2** | For accuracy — report **71.8% honestly with the ceiling evidence**, or add the VIP framing as a documented secondary analysis? | Changes your results chapter |
| **3** | **DOSM** — source real data, or rewrite the four claims? | Only real "missing deliverable" |
| **4** | How much detail for the **Malaysian adaptation** in Chapter 3, and does it need supervisor sign-off in writing? | You said it was approved verbally — get it on record |
| **5** | Does an **unmet criterion cost marks** if the shortfall is rigorously justified? | Tells you where to spend remaining effort |
| **6** | Is a **live deployed URL** sufficient for Objective 4, or do you want a deployment/user guide section too? | Scope of remaining work |

**Question 4 matters more than it looks.** The synthetic Malaysian dataset is the most unusual methodological choice in your project and it isn't in your proposal. Get the approval documented — an email confirmation after the meeting is enough. If a second marker or external examiner reviews this, verbal approval is worth nothing.

---

## 🛡️ Likely hard questions — have these ready

**"Why is your accuracy only 71%?"**
> "Because individual spending is genuinely hard to predict. I tested four improvement routes and none moved it. The learning curve plateaus, so it's a signal limit, not a sample-size limit. AUC is 0.79, and the model reaches 87% on heavy spenders and 76% on churners — it's strong where the business value is."

**"Isn't 0.36 weak clustering?"**
> "Yes — under Kaufman and Rousseeuw's bands that's the 'weak structure' range, and I'm not going to spin it. My argument is that silhouette is the wrong criterion here, and I have four algorithms and a degeneracy test as evidence. The segments are stable at ARI 0.956, which is what matters for an applied segmentation."

*(Naming the unflattering band yourself is disarming. It's much stronger than being told.)*

**"Why did you use a synthetic Malaysian dataset?"**
> "There's no publicly available Malaysian customer-level retail transaction data, and PDPA restricts obtaining real data. You approved adapting the UK dataset. I preserved every distribution — Quantity skew is byte-identical, Price 3.1718 to 3.1745 — and I validated it. I also removed the State variable from the predictive model, because it was randomly assigned and can't carry real signal. Chi-square confirmed p = 0.62."

*That last sentence is the strongest thing you can say about the synthetic data — you removed a feature that would have flattered your model.*

**"Where's the DOSM data?"**
> "That's one of the three gaps I found. It's question 3 on my list."

**"Can I see it running?"**
> Have the URL open already.

**"Is this reproducible?"**
> "The Phase 2 pipeline is — fixed seed, deterministic, and `final_check.py` verifies it. One honest gap: the adaptation scripts have hardcoded sandbox paths, so the Malaysian dataset can't currently be regenerated from source. I'd like to fix that or state it as a limitation."

---

## The one thing to remember

You found every one of these problems yourself, with evidence, before your supervisor did.

That's not damage control — it's the difference between a student who ran some models and a student who **understands what their results mean and where they break down**. Two rigorously-justified unmet criteria demonstrate more than two criteria hit by quietly reframing the problem.

Lead with that confidence. Don't apologise for 0.360 or 71.8% — explain them.
