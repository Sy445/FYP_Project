"""
Step 8 — Climate & Retail-Context Localization
================================================
Builds on the current malaysian_context_online_retail.csv (post Steps 1-7)

SCOPE OF THIS PASS
-------------------
Step 6 handled explicit UK/Western PLACE branding (Union Jack, "I LOVE
LONDON", etc). This pass handles a different problem: product TYPES that
don't make sense for a tropical climate / Malaysian mall context, even
though they carry no place-branding at all.

What's changed and why:
  - HOT WATER BOTTLE / "HOTTIE" -> INSULATED TUMBLER
        A hot water bottle is a cold-climate comfort item. Malaysia's
        equivalent, equally common giftware/homeware category is the
        insulated tumbler/flask (huge retail category there).
  - HAND WARMER -> MINI FAN
        Same logic inverted: a hand warmer solves a problem Malaysia
        doesn't have. A portable/mini fan solves the problem it does have,
        and is sold everywhere from pharmacies to department stores.
  - EGG WARMER / EGG COSY -> MUG WARMER / MUG COSY
        Soft-boiled-egg cosies are a specifically British breakfast object;
        mug warmers/cosies are the equivalent object serving a function
        that generalizes (keeping a drink warm in an air-conditioned
        office - genuinely common in Malaysia).
  - EASTER-themed decor (bunnies, eggs, chicks) -> HARI RAYA-themed decor
        Easter is not a significant commercial/retail event in Malaysian
        malls. Hari Raya (Eid) is one of the largest retail seasons in
        Malaysia, so Easter-specific SKUs are remapped to Raya-equivalent
        decor items (ketupat, lanterns, cookies) rather than dropped.
  - SCARF (wearable) -> SARONG; SCARF KNITTING KIT -> BATIK PAINTING KIT
        Scarves-for-warmth don't fit; sarongs/batik are the direct
        Malaysian analogue as a wearable textile / craft activity.
  - Licensed British character names (Charlie+Lola, Larry the Lamb)
        stripped to generic descriptors - these are foreign IP with no
        Malaysian retail licensing basis, separate from the climate issue.

WHAT WAS DELIBERATELY LEFT ALONE
----------------------------------
Christmas decor (snowflakes, snowmen, sleigh bells, stockings) was NOT
changed. Malaysian shopping malls run large, genuine commercial Christmas
displays every December (Pavilion KL, KLCC, etc. are well known for this),
so snow-themed ornaments are legitimately sold there despite the tropical
climate - the same way they're sold in Singapore, Dubai, or Hong Kong.
Mug warmers, throws/blankets, and knitted décor were also left alone where
plausible, since Malaysia's heavily air-conditioned malls/offices/homes
create genuine demand for warmth-retaining homeware.

This is a judgment-call layer, not a mechanical one - some resulting
phrases will read slightly awkwardly (e.g. "BAKING MOULD RAYA COOKIE MILK
CHOC") because the substitution is lexical, not a full manual rewrite of
each of ~170 flagged SKUs. A before/after mapping CSV is exported
separately so you can manually spot-check and hand-edit any that read
badly before using this in your final report.
"""

import pandas as pd
import re

CHARACTER_STRIP = [
    (re.compile(r"CHARLIE\s*(\+|AND|&)?\s*LOLA", re.IGNORECASE), "KIDS DESIGN"),
    (re.compile(r"LARRY THE LAMB", re.IGNORECASE),        "CUTE LAMB"),
]

# order matters - specific multi-word patterns before single-word ones
CLIMATE_REPLACEMENTS = [
    (re.compile(r"\bWOOLLY HOTTIE\b"),          "INSULATED TUMBLER"),
    (re.compile(r"\bHOTTIE\b"),                 "TUMBLER"),
    (re.compile(r"\bHOT WATER BOTTLE\b"),        "INSULATED TUMBLER"),
    (re.compile(r"\bMINI HAND WARMER\b"),        "MINI FAN"),
    (re.compile(r"\bHAND WARMER\b"),             "MINI FAN"),
    (re.compile(r"\bPOCKET WARMER\b"),            "POCKET FAN"),
    (re.compile(r"\bEGG WARMER\b"),              "MUG WARMER"),
    (re.compile(r"\bEGG COSY\b"),                "MUG COSY"),
    (re.compile(r"\bSCARF KNITTING KIT\b"),       "BATIK PAINTING KIT"),
    (re.compile(r"\bSCARF\b"),                    "SARONG"),
]


def raya_transform(desc: str) -> str:
    """Only fires on descriptions that originally referenced EASTER -
    leaves unrelated mentions of rabbit/egg/chick (e.g. breakfast egg cups,
    rabbit-shaped toys) untouched elsewhere in the catalog."""
    if "EASTER" not in desc.upper():
        return desc
    out = desc
    out = re.sub(r"EASTER", "RAYA", out, flags=re.IGNORECASE)
    out = re.sub(r"BUNNIES", "KETUPAT", out, flags=re.IGNORECASE)
    out = re.sub(r"BUNNY", "KETUPAT", out, flags=re.IGNORECASE)
    out = re.sub(r"RABBIT", "KETUPAT", out, flags=re.IGNORECASE)
    out = re.sub(r"CHICKS", "LANTERNS", out, flags=re.IGNORECASE)
    out = re.sub(r"CHICK", "LANTERN", out, flags=re.IGNORECASE)
    out = re.sub(r"EGGS", "COOKIES", out, flags=re.IGNORECASE)
    out = re.sub(r"EGG", "COOKIE", out, flags=re.IGNORECASE)
    return out


def localize_climate_context(desc: str) -> str:
    if pd.isna(desc):
        return desc
    out = desc
    for pat, repl in CHARACTER_STRIP:
        out = pat.sub(repl, out)
    out = raya_transform(out)
    for pat, repl in CLIMATE_REPLACEMENTS:
        out = pat.sub(repl, out)
    return out


if __name__ == "__main__":
    df = pd.read_csv("/mnt/user-data/outputs/malaysian_context_online_retail.csv")

    before = df["Description"].copy()
    df["Description"] = df["Description"].apply(localize_climate_context)

    changed_mask = df["Description"] != before
    changed_rows = changed_mask.sum()
    print(f"Rows affected in this pass: {changed_rows:,} ({changed_rows/len(df)*100:.2f}%)")

    mapping = (pd.DataFrame({"Before": before[changed_mask], "After": df.loc[changed_mask, "Description"]})
               .drop_duplicates()
               .sort_values("Before")
               .reset_index(drop=True))
    print(f"Unique description changes: {len(mapping)}")
    print("\nSample:")
    print(mapping.head(20).to_string(index=False))

    mapping.to_csv("/mnt/user-data/outputs/description_localization_mapping.csv", index=False)

    # confirm StockCode <-> Description is still 1:1 consistent per code
    consistency = df.groupby("StockCode")["Description"].nunique()
    print(f"\nStockCodes with >1 description after this pass: {(consistency > 1).sum()} "
          f"(pre-existing multi-description StockCodes also existed in the original data, "
          f"this is not something this step introduces)")

    df.to_csv("/mnt/user-data/outputs/malaysian_context_online_retail.csv", index=False)
    print(f"\nUpdated final dataset saved. Shape: {df.shape}")
