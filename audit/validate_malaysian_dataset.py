# ============================================
# FYP PROJECT — Phase 2 (Modelling), Pre-Step
# Validation: malaysian_context_online_retail.csv vs Phase 1 cleaned dataset
# ============================================
#
# PURPOSE
# -------
# Before any RFM/K-means or predictive modelling work, confirm that the
# UK -> Malaysia adaptation (geography remap + GBP->MYR conversion +
# description localization) did NOT corrupt the structural properties that
# Phase 1 preprocessing validated:
#   1. Row count preserved (715,863)
#   2. No new nulls introduced in any column
#   3. Customer ID -> State mapping is 1:1 consistent (every customer has
#      exactly one State across all their transactions)
#   4. Quantity/Price/TotalAmount skewness pattern preserved (strong positive
#      skew in Quantity/Price, moderate in TotalAmount)
#
# This script is read-only: it does not modify or re-save the dataset.

import pandas as pd
import numpy as np
from scipy.stats import skew

pd.set_option("display.max_columns", None)

EXPECTED_ROWS = 715_863

# --------------------------------------------
# Step 1: Load both datasets
# --------------------------------------------
df_original = pd.read_csv("cleaned_combined_online_retail.csv", encoding="ISO-8859-1")
df_my = pd.read_csv("malaysian_context_online_retail.csv", encoding="ISO-8859-1")

print("=" * 60)
print("STEP 1: SHAPE CHECK")
print("=" * 60)
print(f"Phase 1 cleaned dataset shape:      {df_original.shape}")
print(f"Malaysian-context dataset shape:    {df_my.shape}")

rows_match = df_my.shape[0] == EXPECTED_ROWS
print(f"\n[{'PASS' if rows_match else 'FAIL'}] Row count == {EXPECTED_ROWS:,}  "
      f"(actual: {df_my.shape[0]:,})")

cols_match = df_original.shape[0] == df_my.shape[0]
print(f"[{'PASS' if cols_match else 'FAIL'}] Row count matches Phase 1 cleaned dataset "
      f"({df_original.shape[0]:,})")

print(f"\nMalaysian dataset columns: {list(df_my.columns)}")

# --------------------------------------------
# Step 2: Null check
# --------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: NULL VALUE CHECK")
print("=" * 60)
null_counts = df_my.isnull().sum()
print(null_counts)

n_nulls_total = null_counts.sum()
print(f"\n[{'PASS' if n_nulls_total == 0 else 'FAIL'}] Total nulls introduced: {n_nulls_total}")

# --------------------------------------------
# Step 3: Customer ID consistency
# --------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: CUSTOMER ID <-> STATE MAPPING CONSISTENCY")
print("=" * 60)

# 3a. Same set of unique Customer IDs as the Phase 1 dataset (no IDs
#     dropped, duplicated-with-a-twist, or invented during adaptation)
orig_ids = set(df_original["Customer ID"].astype(str).str.replace(r"\.0$", "", regex=True))
my_ids = set(df_my["Customer ID"].astype(str).str.replace(r"\.0$", "", regex=True))

ids_match = orig_ids == my_ids
print(f"Unique Customer IDs — Phase 1: {len(orig_ids):,} | Malaysian: {len(my_ids):,}")
print(f"[{'PASS' if ids_match else 'FAIL'}] Identical Customer ID set before/after adaptation")
if not ids_match:
    print(f"  IDs only in Phase 1:   {len(orig_ids - my_ids)}")
    print(f"  IDs only in Malaysian: {len(my_ids - orig_ids)}")

# 3b. Each Customer ID maps to exactly one State (geography assigned at
#     customer level, not row level — see Dataset process/step1_geography.py)
states_per_customer = df_my.groupby("Customer ID")["State"].nunique()
n_multi_state = (states_per_customer > 1).sum()
print(f"\n[{'PASS' if n_multi_state == 0 else 'FAIL'}] Customers mapped to >1 State: {n_multi_state}")

# 3c. Row-for-row Customer ID sequence identical between the two files
#     (proves the adaptation only ADDED/MODIFIED columns and never
#     reordered, dropped, or resampled rows)
same_id_sequence = df_original["Customer ID"].astype(str).str.replace(r"\.0$", "", regex=True).reset_index(drop=True).equals(
    df_my["Customer ID"].astype(str).str.replace(r"\.0$", "", regex=True).reset_index(drop=True)
)
print(f"[{'PASS' if same_id_sequence else 'FAIL'}] Row-order Customer ID sequence identical to Phase 1")

# --------------------------------------------
# Step 4: Distribution shape / skewness check
# --------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: SKEWNESS / DISTRIBUTION SHAPE CHECK")
print("=" * 60)

for col in ["Quantity", "Price", "TotalAmount"]:
    s_orig = skew(df_original[col])
    s_my = skew(df_my[col])
    pct_diff = abs(s_my - s_orig) / abs(s_orig) * 100 if s_orig != 0 else float("nan")
    print(f"{col:12s}  Phase1 skew: {s_orig:8.4f}   Malaysian skew: {s_my:8.4f}   "
          f"diff: {pct_diff:5.2f}%")

print("\nExpected pattern: strong positive skew in Quantity & Price, "
      "moderate positive skew in TotalAmount.")
print("(Quantity/Price skew should be IDENTICAL or near-identical, since Price is a flat")
print(" FX*psychological-rounding transform and Quantity is untouched; TotalAmount skew")
print(" may shift slightly due to psychological price rounding in Step 7 of the adaptation.)")

# --------------------------------------------
# Step 5: Quantity byte-for-byte unchanged (sanity check on the untouched column)
# --------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: QUANTITY UNCHANGED CHECK")
print("=" * 60)
qty_unchanged = df_original["Quantity"].reset_index(drop=True).equals(df_my["Quantity"].reset_index(drop=True))
print(f"[{'PASS' if qty_unchanged else 'FAIL'}] Quantity column byte-for-byte identical to Phase 1")

# --------------------------------------------
# Step 6: State distribution (sanity check on the geography weighting)
# --------------------------------------------
print("\n" + "=" * 60)
print("STEP 6: STATE DISTRIBUTION (sanity check)")
print("=" * 60)
print((df_my["State"].value_counts(normalize=True) * 100).round(2))

# --------------------------------------------
# Summary
# --------------------------------------------
print("\n" + "=" * 60)
print("OVERALL VALIDATION SUMMARY")
print("=" * 60)
checks = {
    "Row count == 715,863": rows_match,
    "No nulls introduced": n_nulls_total == 0,
    "Identical Customer ID set": ids_match,
    "Every customer -> exactly 1 State": n_multi_state == 0,
    "Row-order Customer ID sequence preserved": same_id_sequence,
    "Quantity byte-for-byte unchanged": qty_unchanged,
}
for check, passed in checks.items():
    print(f"  [{'PASS' if passed else 'FAIL'}] {check}")

all_passed = all(checks.values())
print(f"\n{'ALL CHECKS PASSED — dataset is safe to proceed to Phase 2 Modelling.' if all_passed else 'SOME CHECKS FAILED — investigate before proceeding.'}")
