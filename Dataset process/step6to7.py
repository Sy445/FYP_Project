"""
Step 6 — Product Description Localization
Step 7 — Psychological Price Rounding
================================================
Builds on malaysian_context_online_retail.csv (output of steps 1-5)

WHY SCOPED, NOT A FULL CATALOG REWRITE
----------------------------------------
Of 5,226 unique product descriptions, only 112 (~1.5% of catalog, but check
row-level share below) carry explicit UK/Western place-branding that makes
no sense sold in Malaysia: Union Jack imagery, "ENGLISH ROSE" homeware
lines, "I LOVE LONDON" tourist merchandise, "NEW ENGLAND" styling,
"PARIS FASHION" labelling. These are replaced with Malaysia-relevant
equivalents that preserve the SAME product type (a doormat stays a
doormat, a mug stays a mug) - only the branding/motif changes.

Generic items (T-LIGHT HOLDER, GLASS JAR, COAT HOOK, PICTURE FRAME) are
left untouched: they are plausible in any gift/homeware retail market and
rewriting all 5,226 SKUs would (a) not be defensible without inventing a
synthetic product catalog from scratch, and (b) risk altering the
category-level structure your clustering/segmentation depends on. This
scoping decision - what was changed and why - should go in your
limitations section.

Replacement rules are applied to the DESCRIPTION TEXT itself (regex-based,
ordered longest-match-first), so the same original description always maps
to the same new description - this keeps StockCode <-> Description
consistency intact automatically, with no risk of the same product code
ending up with two different names.
"""

import pandas as pd
import numpy as np
import re

# ---------------------------------------------------------------------------
# STEP 6: Description localization
# ---------------------------------------------------------------------------
# ordered list of (pattern, replacement) - more specific patterns first
DESC_REPLACEMENTS = [
    (r"\bI LOVE LONDON\b",              "I LOVE MALAYSIA"),
    (r"\bLONDON BUS\b",                 "KL MONORAIL"),
    (r"\bLONDON BRIDGE\b",              "PETRONAS TOWERS"),
    (r"\bLONDON\b",                     "KL"),
    (r"\bPARIS FASHION\b",              "KL FASHION"),
    (r"\bFRENCH STYLE\b",               "COLONIAL STYLE"),
    (r"\bNEW ENGLAND\b",                "PERANAKAN"),
    (r"\bENGLISH ROSE\b",               "BATIK FLORAL"),
    (r"\bUNION (JACK|FLAG)\b",          "HIBISCUS PRINT"),
    (r"\bJUBILEE\b",                    "MERDEKA"),
]

COMPILED_REPLACEMENTS = [(re.compile(pat), repl) for pat, repl in DESC_REPLACEMENTS]


def localize_description(desc: str) -> str:
    if pd.isna(desc):
        return desc
    out = desc
    for pat, repl in COMPILED_REPLACEMENTS:
        out = pat.sub(repl, out)
    return out


def localize_all_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Description_original"] = df["Description"]
    df["Description"] = df["Description"].apply(localize_description)
    return df


# ---------------------------------------------------------------------------
# STEP 7: Psychological price rounding (RM x.00 / x.50 / x.90 convention)
# ---------------------------------------------------------------------------
# Malaysian retail pricing overwhelmingly clusters on a small set of endings
# rather than arbitrary cents - most commonly .00, .50, and .90 (charm
# pricing). We snap each FX-converted price to whichever of these endings
# (within the same or next ringgit) is closest to the raw converted value,
# rather than truncating/flooring, so the rounding error stays small and
# symmetric and doesn't systematically drag prices up or down.
PSYCH_ENDINGS = np.array([0.00, 0.50, 0.90])


def round_to_psychological_price(price: float) -> float:
    if pd.isna(price) or price <= 0:
        return price
    base = np.floor(price)
    candidates = np.concatenate([base + PSYCH_ENDINGS, [base + 1.00]])
    diffs = np.abs(candidates - price)
    return float(candidates[np.argmin(diffs)])


def apply_psychological_pricing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Price_raw_fx"] = df["Price"]  # keep pre-rounding value for traceability
    df["Price"] = df["Price"].apply(round_to_psychological_price).round(2)
    df["TotalAmount"] = (df["Quantity"] * df["Price"]).round(2)
    return df


if __name__ == "__main__":
    df = pd.read_csv("/mnt/user-data/outputs/malaysian_context_online_retail.csv")

    # --- Step 6 ---
    n_before = df["Description"].nunique()
    df = localize_all_descriptions(df)
    changed_rows = (df["Description"] != df["Description_original"]).sum()
    changed_unique = df.loc[df["Description"] != df["Description_original"],
                             "Description_original"].nunique()
    print(f"Unique descriptions before: {n_before}")
    print(f"Unique descriptions localized: {changed_unique}")
    print(f"Rows affected: {changed_rows:,} ({changed_rows/len(df)*100:.2f}% of dataset)")
    print("\nSample localizations:")
    sample = (df.loc[df["Description"] != df["Description_original"],
                      ["Description_original", "Description"]]
              .drop_duplicates().head(10))
    print(sample.to_string(index=False))

    # --- Step 7 ---
    from scipy.stats import skew
    skew_before = skew(df["Price"])
    df = apply_psychological_pricing(df)
    skew_after = skew(df["Price"])

    print(f"\nSkewness Price before psychological rounding: {skew_before:.6f}")
    print(f"Skewness Price after psychological rounding:  {skew_after:.6f}")
    print(f"\nSample price endings (should mostly be .00/.50/.90):")
    print(df["Price"].head(15).tolist())

    ending_counts = (df["Price"] % 1).round(2).value_counts().head(6)
    print(f"\nMost common price endings across dataset:")
    print(ending_counts)

    df.to_csv("/mnt/user-data/outputs/malaysian_context_online_retail.csv", index=False)
    print(f"\nUpdated final dataset saved. Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
