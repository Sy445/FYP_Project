"""
Step 1 — Geography Adaptation: UK -> Malaysia
================================================
Online Retail (UK) -> Malaysian-context synthetic adaptation
FYP: Customer Segmentation & Spending-Behaviour Prediction (Malaysian Retail)

LOGIC
-----
The original dataset is 90.8% United Kingdom, with a long tail of ~30 other
countries. To preserve that same "dominant-market-with-a-long-tail" shape
(important, because RFM / segmentation models are sensitive to how
geographically concentrated the customer base is), we replace Country with
Malaysian states/federal territories using a weighted random assignment
that mirrors two real-world facts about Malaysia:

  1. POPULATION CONCENTRATION: ~55% of Malaysians live in or do business
     through the Klang Valley (Selangor + Kuala Lumpur), Penang, and Johor
     are the next-largest urban/retail hubs (DOSM population estimates).
  2. E-COMMERCE / RETAIL ACTIVITY CONCENTRATION: online retail spending in
     Malaysia is even MORE concentrated in urban states than raw population,
     because e-commerce penetration, courier coverage, and disposable income
     are all higher in Klang Valley, Penang, and Johor Bahru than in East
     Malaysia or rural Peninsular states.

We therefore blend population share with a retail-activity skew, producing
weights where Selangor + Kuala Lumpur + Penang + Johor account for ~77% of
records — deliberately mirroring the ~90.8% dominance of the single leading
region (UK) in the original dataset, while still leaving a realistic tail
for every other state.

CRITICAL DESIGN DECISION — assign at the CUSTOMER level, not the row level
----------------------------------------------------------------------
A given Customer ID must always map to the SAME state. If we assigned state
randomly per transaction row, a single customer could appear to be buying
from Penang on Monday and Sabah on Tuesday, which:
  - breaks the geographic component of customer segmentation entirely
  - has no real-world analogue for a retail dataset (customers don't
    teleport between states between invoices)
  - would silently corrupt any state-level or region-level aggregation
So the mapping is built once per unique Customer ID and then joined back
onto all 715,863 rows for that customer. This does NOT touch Quantity,
Price, InvoiceDate, or TotalAmount, so RFM structure and the positive-skew
distributions of Quantity/Price/TotalAmount are completely unaffected by
this step (see Section 4 flag below).
"""

import pandas as pd
import numpy as np

RANDOM_SEED = 42  # fixed for reproducibility - report this in your methodology chapter

# ---------------------------------------------------------------------------
# 1. Malaysian states/federal territories with weights
#    (blend of DOSM population share + urban e-commerce activity skew)
# ---------------------------------------------------------------------------
MALAYSIA_GEO_WEIGHTS = {
    "Selangor":            0.34,   # Klang Valley core - highest population + retail/e-com hub
    "Kuala Lumpur":        0.20,   # Federal capital, highest income & online spend per capita
    "Johor":               0.11,   # 2nd largest economy, JB cross-border retail activity
    "Penang":              0.09,   # major urban/tech hub, historically strong retail base
    "Perak":               0.05,
    "Sabah":               0.04,
    "Sarawak":             0.04,
    "Negeri Sembilan":     0.03,
    "Melaka":              0.03,
    "Kedah":               0.02,
    "Pahang":              0.02,
    "Kelantan":            0.01,
    "Terengganu":          0.01,
    "Perlis":              0.005,
    "Putrajaya":           0.003,
    "Labuan":              0.002,
}

# sanity check - weights must sum to 1.0
assert abs(sum(MALAYSIA_GEO_WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

states = list(MALAYSIA_GEO_WEIGHTS.keys())
probs = list(MALAYSIA_GEO_WEIGHTS.values())


def assign_malaysian_geography(df: pd.DataFrame, customer_col: str = "Customer ID",
                                seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Replace Country with a Malaysian state, assigned once per unique customer
    and joined back onto every transaction row for that customer.

    Rows with a missing/NaN customer ID (guest-style transactions) are
    assigned a state independently per row, since there's no persistent
    identity to keep consistent for them.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()

    known_customers = df.loc[df[customer_col].notna(), customer_col].unique()
    customer_state_map = pd.Series(
        rng.choice(states, size=len(known_customers), p=probs),
        index=known_customers,
    )

    df["State"] = df[customer_col].map(customer_state_map)

    # guest / null-customer rows: assign independently per row
    missing_mask = df["State"].isna()
    if missing_mask.any():
        df.loc[missing_mask, "State"] = rng.choice(
            states, size=missing_mask.sum(), p=probs
        )

    # replace Country column in place, drop helper if you prefer a single column
    df["Country"] = "Malaysia"          # kept for schema compatibility / filtering
    df.rename(columns={"State": "State"}, inplace=True)

    return df


if __name__ == "__main__":
    df = pd.read_csv("/mnt/user-data/uploads/cleaned_combined_online_retail.csv")
    df = assign_malaysian_geography(df)

    print(f"Rows: {len(df):,}")
    print(f"Unique customers: {df['Customer ID'].nunique():,}")
    print("\nState distribution (row-level %):")
    print((df["State"].value_counts(normalize=True) * 100).round(2))

    # consistency check: every customer maps to exactly one state
    n_states_per_customer = df.groupby("Customer ID")["State"].nunique()
    print(f"\nCustomers with >1 state (should be 0): {(n_states_per_customer > 1).sum()}")

    df.to_csv("/home/claude/step1_geography_output.csv", index=False)
    print("\nSaved -> /home/claude/step1_geography_output.csv")
