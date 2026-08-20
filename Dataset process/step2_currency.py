"""
Step 2 — Currency Adaptation: GBP -> MYR
================================================
Builds on step1_geography_output.csv

LOGIC / DECISION: (a) flat FX rate, NOT category-adjusted repricing
--------------------------------------------------------------------
You asked whether to (a) apply a straight exchange rate, or (b) apply the
rate and then further adjust individual product categories toward "realistic"
Malaysian retail price points. I'm recommending (a), for a reason that
matters directly to your point (4) requirement:

  Skewness, and every other shape statistic (kurtosis, coefficient of
  variation, relative spread) of a distribution, is INVARIANT under a
  positive linear transform (i.e. multiplying every value by the same
  positive constant). So Price_MYR = Price_GBP * rate produces a histogram
  that is a pure horizontal rescale of the original - same shape, same
  skew, same relative gaps between cheap and expensive items - just a
  different currency on the x-axis.

  A category-specific adjustment (e.g. "homeware +15%, stationery -5%,
  novelty gifts +30%" to mimic Malaysian retail markups) would require you
  to invent per-category multipliers with no defensible empirical source -
  and worse, it would change the SHAPE of the Price distribution (some
  categories compressed, others stretched), which directly conflicts with
  your Phase 1 requirement to preserve the validated distribution shapes.
  That's a real risk, so I'm flagging it explicitly rather than doing it.

So: flat FX multiplier is both the more defensible academic choice (one
transparent, cited, reproducible parameter) AND the one that mathematically
guarantees Quantity/Price/TotalAmount distribution shapes are untouched.

EXCHANGE RATE USED
-------------------
GBP/MYR mid-market rate ~5.50 (early August 2026, multiple live FX sources:
Xe, Pound Sterling Live, Investing.com all clustering around RM5.45-5.52
per GBP at time of writing). Using a single rounded, cited, point-in-time
rate is standard practice for this kind of dataset adaptation - real FX
rates fluctuate daily and using a single fixed rate (rather than trying to
simulate historical GBP/MYR movements across 2010-2011, which the original
transactions span) is a deliberate simplification you should name in your
methodology chapter: it treats currency as a single conversion constant to
preserve reproducibility and avoid layering FX-timing noise into a dataset
that is already synthetic-in-geography.

WHAT GETS RECOMPUTED
---------------------
TotalAmount must be regenerated as Quantity * Price_MYR (rounded to 2dp) to
stay internally consistent - it was originally Quantity * Price_GBP, so if
we only touched Price and left TotalAmount as GBP-derived, every row would
have mismatched currency units between Price and TotalAmount. Quantity
itself is NEVER touched.
"""

import pandas as pd
import numpy as np

GBP_TO_MYR = 5.50  # cited: Xe/Pound Sterling Live/Investing.com mid-market average, ~Aug 2026


def convert_currency(df: pd.DataFrame, rate: float = GBP_TO_MYR) -> pd.DataFrame:
    df = df.copy()

    # keep the original GBP price for traceability / appendix comparison
    df["Price_GBP"] = df["Price"]

    df["Price"] = (df["Price_GBP"] * rate).round(2)

    # recompute TotalAmount for currency consistency (Quantity is untouched)
    df["TotalAmount"] = (df["Quantity"] * df["Price"]).round(2)

    return df


if __name__ == "__main__":
    df = pd.read_csv("/home/claude/step1_geography_output.csv")
    df_before = df.copy()

    df = convert_currency(df)

    print(f"Exchange rate used: 1 GBP = {GBP_TO_MYR} MYR\n")
    print("Price (GBP) summary:")
    print(df["Price_GBP"].describe().round(3))
    print("\nPrice (MYR) summary:")
    print(df["Price"].describe().round(3))

    # prove distribution shape is unchanged: skewness is invariant to positive linear scaling
    from scipy.stats import skew
    print(f"\nSkewness Price (GBP):  {skew(df['Price_GBP']):.6f}")
    print(f"Skewness Price (MYR):  {skew(df['Price']):.6f}")
    print(f"Skewness TotalAmount (GBP, recomputed): {skew(df_before['TotalAmount']):.6f}")
    print(f"Skewness TotalAmount (MYR):             {skew(df['TotalAmount']):.6f}")
    print(f"Skewness Quantity (unchanged either way): {skew(df['Quantity']):.6f}")

    df.to_csv("/home/claude/step2_currency_output.csv", index=False)
    print("\nSaved -> /home/claude/step2_currency_output.csv")

    print("\nSample rows:")
    print(df[["Customer ID", "State", "Price_GBP", "Price", "Quantity", "TotalAmount"]].head(5))
