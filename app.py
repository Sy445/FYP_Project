"""
================================================================================
FYP PROJECT — Objective 4: Interactive Retail Analytics Dashboard
Predictive Consumer Segmentation and Spending Behaviour in Malaysian Retail
================================================================================

PURPOSE (proposal Section 2.5.2 / methodology Phase 6)
------------------------------------------------------
A Streamlit dashboard that makes the Phase 2 modelling results usable by retail
managers who have no data science training. It translates cluster output and
classifier metrics into plain-language segment descriptions and concrete
recommended actions.

ARCHITECTURE (explain this section in your report)
--------------------------------------------------
The dashboard is a PRESENTATION LAYER ONLY. It performs no modelling: it reads
the artefacts already produced by Phase 2 and saved in phase2_outputs/.

    phase2_objective2_rfm_kmeans.py ──> customer_rfm_segments.csv
                                        segment_profile_summary.csv
    phase2_objective3_predictive_modelling.py ──> model_comparison.csv
                                                  feature_importance.csv
                                                  model_recommendation.json
                                        app.py ──> reads all of the above

This separation matters for three reasons worth stating in the report:
  1. REPRODUCIBILITY — the dashboard cannot silently disagree with the results
     in your report, because it displays the same saved numbers rather than
     recomputing them with possibly-different settings.
  2. PERFORMANCE — no model training on page load. Streamlit reruns the whole
     script on every interaction, so retraining here would make the app
     unusable. Loads are additionally cached with @st.cache_data.
  3. DEPLOYABILITY — Streamlit Community Cloud containers are memory-limited.
     Reading a few small CSVs stays well inside the free tier; refitting a
     Random Forest on 715,863 rows would not.

DESIGN NOTES
------------
* Colour: the four segment colours are a validated categorical palette. Each
  segment is bound to a FIXED colour by name (SEGMENT_COLOURS), never by
  position, so filtering the data never repaints the remaining segments.
* Accessibility: two of the four segment colours fall below a 3:1 contrast
  ratio against a white surface, so every chart carries direct value labels
  and a table view — colour is never the only channel carrying meaning.
* A four-colour scatter was deliberately NOT used: the yellow/orange pair
  falls below the normal-vision separation floor when every pair can appear
  adjacent, as happens in a scatter. Segment comparisons use faceted small
  multiples instead. (See validate_palette.py for the computed check.)

RUNNING LOCALLY
---------------
    pip install -r requirements.txt
    streamlit run app.py

================================================================================
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ==============================================================================
# CONFIGURATION
# ==============================================================================

OUT_DIR = Path(__file__).parent / "phase2_outputs"

st.set_page_config(
    page_title="Malaysian Retail — Customer Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------------
# Colour tokens.
#
# Segment colours are slots 1-4 of a validated categorical palette. The
# assignment is BY SEGMENT NAME, not by list position — this is what guarantees
# that when a manager filters segments out, the survivors keep their colours.
# ------------------------------------------------------------------------------
SEGMENT_COLOURS = {
    "Champions":          "#2a78d6",   # slot 1 — blue
    "At-Risk High Value": "#eb6834",   # slot 2 — orange
    "New & Promising":    "#1baf7a",   # slot 3 — aqua
    "Lost / Dormant":     "#eda100",   # slot 4 — yellow
}

# Chart chrome — recessive by design so the data marks carry the attention.
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#ffffff"      # matches the Streamlit light theme set in config.toml

# De-emphasis grey for the faceted "small multiples" charts, where one segment
# is highlighted and every other customer is drawn as background context.
#
# This value was CHOSEN BY MEASUREMENT, not by eye (validate_palette.py):
#   vs #c3c2b7 every segment colour clears the ΔE 15 normal-vision floor
#       (Champions 29.5, At-Risk 22.0, New & Promising 19.6, Lost/Dormant 15.5)
#   vs #898781 New & Promising measures only 14.5 — below the floor, so that
#       darker grey was rejected
# It is also light enough (1.79:1 against white) to read as recessive context
# rather than competing with the highlighted marks.
CONTEXT_GREY = "#c3c2b7"

# Diverging pair for the "what raises / lowers the prediction" chart. Validated
# all-pairs: CVD ΔE 21.6, normal-vision ΔE 32.3, both above 3:1 on white.
DIVERGING_UP = "#2a78d6"     # raises likelihood
DIVERGING_DOWN = "#e34948"   # lowers likelihood

# ------------------------------------------------------------------------------
# Map the analytical segment labels produced by Objective 2 onto short,
# manager-friendly names. The long labels are precise but unreadable on a chart
# axis; the short names are what a retail manager would actually say out loud.
# ------------------------------------------------------------------------------
SEGMENT_SHORT_NAMES = {
    "Champions (recent, frequent, high spend)": "Champions",
    "At-Risk High Value (lapsing loyal spender)": "At-Risk High Value",
    "New / Promising (recent, low frequency & value)": "New & Promising",
    "Lost / Dormant Low-Value": "Lost / Dormant",
}

# Fixed display order: most to least valuable. Used everywhere so the segments
# always appear in the same sequence regardless of the current filter.
SEGMENT_ORDER = ["Champions", "At-Risk High Value", "New & Promising", "Lost / Dormant"]

# ------------------------------------------------------------------------------
# Plain-language segment content.
#
# This is the layer that turns a cluster number into something a retail manager
# can act on. Each entry carries: a one-line identity, a description in ordinary
# language (no statistical vocabulary), a concrete recommended action with a
# named tactic and a measurable outcome, and the priority the manager should
# assign it.
#
# The actions are deliberately specific about what NOT to do as well — telling a
# manager where to stop spending is often more valuable than another campaign.
# ------------------------------------------------------------------------------
SEGMENT_CONTENT = {
    "Champions": {
        "headline": "Your best customers — protect these first",
        "priority": "Priority 1 · Protect",
        "who": (
            "Bought recently, buy often, and spend far more than anyone else. "
            "They are roughly a fifth of your customer list but generate about "
            "two-thirds of all revenue."
        ),
        "action_title": "Reward loyalty — do not discount",
        "action": """
**What to do**

1. **Launch a tiered loyalty programme.** Give this group early access to new
   stock before general release, and free delivery with no minimum spend.
2. **Assign named contacts** to the top 100 by spend — a real person who
   handles their orders and issues.
3. **Ask them for referrals.** This group is your most credible advocate; a
   referral incentive here costs less per acquired customer than paid ads.

**What NOT to do**

Do **not** send this segment discount vouchers. They already buy willingly at
full price, so a discount simply reduces margin on your most profitable
customers. Discounting is a tool for the At-Risk group, not this one.

**How to measure it:** month-on-month retention rate of this segment, and the
share of total revenue it contributes. Both should hold steady or rise.
""",
    },
    "At-Risk High Value": {
        "headline": "Spent well, then went quiet — your biggest win-back opportunity",
        "priority": "Priority 2 · Win back now",
        "who": (
            "These customers used to buy regularly and spend well, but have not "
            "purchased for around eight months. They are still your second most "
            "valuable group by historical revenue — but they are drifting away."
        ),
        "action_title": "Run a time-boxed win-back sequence",
        "action": """
**What to do — a three-week sequence**

1. **Week 1 — personal reminder.** Email referencing the category they last
   bought from ("your last order was kitchenware — here's what's new"). No
   discount yet; test whether a reminder alone is enough.
2. **Week 2 — single-use voucher.** 10–15% off, expiring in 14 days. The
   deadline matters more than the size of the discount; do not exceed 15% or
   you train them to wait for sales.
3. **Week 3 — direct contact for the top 100** by historical spend, by phone or
   WhatsApp. At this value level a human conversation is worth the cost.

**Then stop.** If three touches produce nothing, move them to the low-cost
treatment used for Lost / Dormant. Continuing to chase non-responders is where
win-back budgets get wasted.

**How to measure it:** reactivation rate within 30 days of the sequence
starting, and revenue recovered against campaign cost. Because this segment
holds a large amount of past revenue, even a modest win-back rate repays a
substantial campaign budget — size the budget against that figure.
""",
    },
    "New & Promising": {
        "headline": "Recent buyers who haven't formed a habit yet",
        "priority": "Priority 3 · Grow",
        "who": (
            "Bought recently, but only once or twice, and spend modestly so far. "
            "This group is genuinely undecided — with the right follow-up they "
            "become Champions, and with none they quietly become Lost."
        ),
        "action_title": "Drive the second and third purchase",
        "action": """
**What to do — a 60-day onboarding sequence**

1. **Day 3 — make the first purchase succeed.** Care instructions, setup tips,
   or styling ideas for what they actually bought. No selling.
2. **Day 21 — one curated recommendation.** Two or three items that genuinely
   complement their first order. Keep it short; a full catalogue converts worse
   than a small, relevant selection.
3. **Day 45 — a modest second-purchase incentive** (free delivery works better
   than a percentage off for low-value baskets).

**Why this sequence:** repeat buying habit forms at the second and third
purchase, not the first. Concentrating effort in the first 60 days is far
cheaper than trying to win the same customer back a year later.

**How to measure it:** the percentage of this segment reaching three or more
orders within 90 days.
""",
    },
    "Lost / Dormant": {
        "headline": "Largest group by headcount, smallest by value",
        "priority": "Priority 4 · Spend little here",
        "who": (
            "Your biggest segment by number of people — about a third of the "
            "list — but the smallest by revenue, contributing only a few percent "
            "of sales. Mostly one-time buyers from more than a year ago."
        ),
        "action_title": "Contain cost and reallocate the budget",
        "action": """
**What to do**

1. **One low-cost bulk email**, once. A single broad reactivation offer costs
   almost nothing to send and will recover a small number of customers.
2. **Suppress non-responders from all paid channels.** Paid retargeting against
   this segment costs more per reactivation than the customer is likely to be
   worth.
3. **Use them to clean the list.** Removing long-dormant addresses improves
   email deliverability for every other segment — a real benefit that is easy
   to overlook.
4. **Reallocate the saved budget to At-Risk High Value,** where the same money
   is chasing far more revenue.

**What NOT to do**

Do not run a deep-discount campaign to "reactivate" this group at scale. The
economics do not work: the average customer here is worth a fraction of a
Champion, and heavy discounting attracts one-off bargain hunters who do not
return.

**How to measure it:** cost per reactivated customer, compared against this
segment's average customer value. If it costs more to win them back than they
spend, stop.
""",
    },
}


# ==============================================================================
# DATA LOADING
#
# @st.cache_data memoises on the function's inputs. Streamlit re-executes this
# whole script top-to-bottom on every widget interaction, so without caching the
# CSVs would be re-read on every click.
# ==============================================================================

@st.cache_data
def load_customers() -> pd.DataFrame:
    """Per-customer RFM values, cluster assignment, and (synthetic) state."""
    df = pd.read_csv(OUT_DIR / "customer_rfm_segments.csv")
    df["Segment"] = df["Segment_Label"].map(SEGMENT_SHORT_NAMES)
    # Fail loudly rather than silently dropping customers if Objective 2's
    # labels ever change without this mapping being updated.
    if df["Segment"].isna().any():
        unmapped = df.loc[df["Segment"].isna(), "Segment_Label"].unique()
        raise ValueError(
            f"Unmapped segment label(s) from Objective 2: {list(unmapped)}. "
            "Update SEGMENT_SHORT_NAMES in app.py."
        )
    return df


@st.cache_data
def load_segment_profile() -> pd.DataFrame:
    """Cluster-level RFM profile produced by Objective 2."""
    df = pd.read_csv(OUT_DIR / "segment_profile_summary.csv")
    df["Segment"] = df["Segment_Label"].map(SEGMENT_SHORT_NAMES)
    return df


@st.cache_data
def load_model_comparison() -> pd.DataFrame:
    return pd.read_csv(OUT_DIR / "model_comparison.csv")


@st.cache_data
def load_feature_importance() -> pd.DataFrame:
    return pd.read_csv(OUT_DIR / "feature_importance.csv")


@st.cache_data
def load_predictions() -> pd.DataFrame:
    """Per-customer scores from the recommended model (Objective 3, Step 12b)."""
    return pd.read_csv(OUT_DIR / "customer_predictions.csv")


@st.cache_data
def load_recommendation() -> dict:
    with open(OUT_DIR / "model_recommendation.json", encoding="utf-8") as f:
        return json.load(f)


# ==============================================================================
# FORMATTING HELPERS
# ==============================================================================

def rm(value: float, decimals: int = 0) -> str:
    """Format a number as Malaysian Ringgit with thousands separators."""
    return f"RM {value:,.{decimals}f}"


def rm_compact(value: float) -> str:
    """Compact Ringgit for headline tiles and chart labels.

    One decimal place is kept in the thousands range: rounding RM 26,342 to
    "RM 26K" and RM 1,302 to "RM 1K" flattens a 20x difference into adjacent-
    looking numbers, which defeats the point of a comparison chart.
    """
    if abs(value) >= 1_000_000:
        return f"RM {value / 1_000_000:.1f}M"
    if abs(value) >= 10_000:
        return f"RM {value / 1_000:.1f}K"
    if abs(value) >= 1_000:
        return f"RM {value / 1_000:.2f}K"
    return f"RM {value:,.0f}"


def style_axes(fig: go.Figure, show_xgrid: bool = True,
               margin: dict | None = None) -> go.Figure:
    """Apply the shared recessive chart chrome to a Plotly figure.

    `margin` MUST be sized per chart. Plotly does not grow the plotting area to
    fit axis text, so a margin too small for the longest tick label silently
    clips it — long category names and log-scale tick labels ("100k") are the
    usual casualties. Each caller passes the space its own labels need.
    """
    fig.update_layout(
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif",
                  color=INK_SECONDARY, size=13),
        margin=margin or dict(l=60, r=20, t=20, b=40),
        hoverlabel=dict(bgcolor=SURFACE, font_size=13,
                        bordercolor=GRIDLINE, font_color=INK_PRIMARY),
    )
    fig.update_xaxes(showgrid=show_xgrid, gridcolor=GRIDLINE, gridwidth=1,
                     zeroline=False, linecolor=GRIDLINE, tickfont=dict(color=INK_MUTED))
    fig.update_yaxes(showgrid=False, zeroline=False, linecolor=GRIDLINE,
                     tickfont=dict(color=INK_SECONDARY))
    return fig


# ==============================================================================
# SIDEBAR — navigation and global filters
# ==============================================================================

def render_sidebar(customers: pd.DataFrame):
    """Draw the sidebar and return (selected_page, filtered_customers)."""
    st.sidebar.title("🛍️ Customer Intelligence")
    st.sidebar.caption("Malaysian Retail — Segmentation & Spend Prediction")

    page = st.sidebar.radio(
        "Go to",
        ["Overview", "Customer Segments", "Prediction Insights"],
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    st.sidebar.subheader("Filters")

    # --- Segment filter -------------------------------------------------------
    available_segments = [s for s in SEGMENT_ORDER if s in set(customers["Segment"])]
    chosen_segments = st.sidebar.multiselect(
        "Customer segment",
        options=available_segments,
        default=available_segments,
        help="Narrow every figure on the page to the selected segments.",
    )

    # --- State filter ---------------------------------------------------------
    # NOTE: see the caption below. State is descriptive only.
    all_states = sorted(customers["State"].dropna().unique())
    chosen_states = st.sidebar.multiselect(
        "State / territory",
        options=all_states,
        default=all_states,
        help="Descriptive filter only — see the note below.",
    )
    st.sidebar.caption(
        "ℹ️ **Geography is synthetic.** States were assigned to customers by a "
        "weighted random draw when this dataset was adapted to a Malaysian "
        "context, so they are useful for filtering and for describing where "
        "customers sit in the data — but differences in spending *between* "
        "states are not real and must not be read as regional insight."
    )

    if not chosen_segments or not chosen_states:
        st.sidebar.warning("Select at least one segment and one state.")

    filtered = customers[
        customers["Segment"].isin(chosen_segments)
        & customers["State"].isin(chosen_states)
    ]

    st.sidebar.divider()
    st.sidebar.caption(
        f"Showing **{len(filtered):,}** of **{len(customers):,}** customers"
    )

    return page, filtered


# ==============================================================================
# PAGE 1 — OVERVIEW
# ==============================================================================

def page_overview(customers: pd.DataFrame, all_customers: pd.DataFrame):
    st.title("Business Overview")
    st.markdown(
        "A summary of the customer base, how it divides into behavioural "
        "segments, and where the revenue actually sits."
    )

    if customers.empty:
        st.warning("No customers match the current filters. Widen the selection "
                   "in the sidebar to see results.")
        return

    total_customers = len(customers)
    total_revenue = customers["Monetary"].sum()
    avg_value = customers["Monetary"].mean()
    n_segments = customers["Segment"].nunique()

    # --- KPI ROW ---------------------------------------------------------------
    # These are headline numbers, so they are stat tiles rather than a chart —
    # a bar chart of four unrelated quantities would be harder to read, not
    # easier.
    st.subheader("Headline figures")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total customers", f"{total_customers:,}")
    c2.metric("Total revenue", rm_compact(total_revenue))
    c3.metric("Average customer value", rm(avg_value))
    c4.metric("Segments in view", f"{n_segments}")

    if len(customers) < len(all_customers):
        share = total_revenue / all_customers["Monetary"].sum() * 100
        st.caption(
            f"Filtered view: these customers represent {share:.1f}% of total "
            f"business revenue ({rm_compact(all_customers['Monetary'].sum())})."
        )

    st.divider()

    # --- HEADLINE INSIGHT ------------------------------------------------------
    # Computed from the *unfiltered* data so the strategic message stays stable
    # regardless of what the manager is currently looking at.
    seg_all = _segment_totals(all_customers)
    if "Champions" in seg_all.index:
        champ = seg_all.loc["Champions"]
        st.info(
            f"**{champ['pct_customers']:.0f}% of customers generate "
            f"{champ['pct_revenue']:.0f}% of revenue.** "
            f"The {int(champ['customers']):,} customers in the *Champions* "
            f"segment account for {rm_compact(champ['revenue'])} of "
            f"{rm_compact(seg_all['revenue'].sum())} in total sales. Protecting "
            "this group matters more than any other single action."
        )
    if "At-Risk High Value" in seg_all.index:
        risk = seg_all.loc["At-Risk High Value"]
        st.warning(
            f"**{rm_compact(risk['revenue'])} of revenue is in the "
            f"*At-Risk High Value* segment.** These "
            f"{int(risk['customers']):,} customers historically spent well but "
            "have not purchased recently. They are the clearest win-back "
            "opportunity in the customer base."
        )

    st.divider()

    # --- WHERE THE REVENUE SITS ------------------------------------------------
    st.subheader("Where the revenue sits")
    st.markdown(
        "Each bar totals 100%. Comparing them shows which segments are "
        "**larger than their value** and which are **smaller than their value** "
        "— the gap between the two bars is the whole argument for segmenting."
    )

    seg = _segment_totals(customers)
    fig = _revenue_concentration_chart(seg)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # Table view. This is not optional decoration: two of the four segment
    # colours sit below a 3:1 contrast ratio on white, so an equivalent
    # non-colour representation of the same data must always be available.
    with st.expander("View as table"):
        table = seg.reset_index()[
            ["Segment", "customers", "pct_customers", "revenue", "pct_revenue",
             "avg_value"]
        ].copy()
        table.columns = ["Segment", "Customers", "% of customers",
                         "Revenue (RM)", "% of revenue", "Avg value (RM)"]
        st.dataframe(
            table.style.format({
                "Customers": "{:,.0f}",
                "% of customers": "{:.1f}%",
                "Revenue (RM)": "{:,.0f}",
                "% of revenue": "{:.1f}%",
                "Avg value (RM)": "{:,.0f}",
            }),
            width="stretch",
            hide_index=True,
        )

    st.divider()

    # --- PROVENANCE ------------------------------------------------------------
    with st.expander("About this data"):
        st.markdown(
            f"""
**Source.** {len(all_customers):,} customers derived from 715,863 retail
transactions spanning December 2009 to December 2011.

**Segments.** Produced by K-means clustering (k=4) on Recency, Frequency and
Monetary value. The segmentation is highly reproducible — bootstrap Adjusted
Rand Index of 0.956 across resamples, well above the 0.75 stability convention.

**How well-separated are the segments?** The silhouette score is 0.360. This is
below the 0.5 target originally set for this project, and the reason is
inherent to the data rather than to the method: customer behaviour forms a
continuous gradient rather than four naturally separated groups, and no
clustering algorithm tested exceeded this range. The segments should therefore
be read as a useful and highly stable way of *dividing* the customer base, not
as evidence that four distinct customer "types" exist in nature.

**Geography is synthetic.** This dataset was adapted from a UK retail dataset
into a Malaysian context. States were assigned to customers by weighted random
draw, so state may be used to filter and describe the customer base, but
differences in spending between states carry no real-world meaning.

**Currency.** Converted from GBP at a fixed rate of 1 GBP = 5.50 MYR.
"""
        )


def _segment_totals(customers: pd.DataFrame) -> pd.DataFrame:
    """Aggregate customers to one row per segment, in fixed display order."""
    agg = customers.groupby("Segment").agg(
        customers=("Customer ID", "size"),
        revenue=("Monetary", "sum"),
        avg_value=("Monetary", "mean"),
    )
    agg["pct_customers"] = agg["customers"] / agg["customers"].sum() * 100
    agg["pct_revenue"] = agg["revenue"] / agg["revenue"].sum() * 100
    present = [s for s in SEGMENT_ORDER if s in agg.index]
    return agg.loc[present]


def _revenue_concentration_chart(seg: pd.DataFrame) -> go.Figure:
    """Two stacked 100% bars: share of customers vs share of revenue.

    Form choice: this is part-to-whole for two comparable wholes, which is a
    stacked bar. Horizontal, because the segment names are long. Direct value
    labels are mandatory here rather than cosmetic — see the module docstring.
    """
    fig = go.Figure()

    for segment in seg.index:
        colour = SEGMENT_COLOURS[segment]
        pct_cust = seg.loc[segment, "pct_customers"]
        pct_rev = seg.loc[segment, "pct_revenue"]

        fig.add_trace(go.Bar(
            y=["Share of customers", "Share of revenue"],
            x=[pct_cust, pct_rev],
            name=segment,
            orientation="h",
            marker=dict(
                color=colour,
                # A 2px surface-coloured gap between stacked segments keeps
                # adjacent fills legible without adding an outline colour.
                line=dict(color=SURFACE, width=2),
            ),
            text=[f"{pct_cust:.0f}%", f"{pct_rev:.0f}%"],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="#ffffff", size=13),
            hovertemplate=(
                f"<b>{segment}</b><br>"
                f"{seg.loc[segment, 'customers']:,.0f} customers "
                f"({pct_cust:.1f}%)<br>"
                f"{rm(seg.loc[segment, 'revenue'])} revenue "
                f"({pct_rev:.1f}%)<extra></extra>"
            ),
        ))

    fig.update_layout(
        barmode="stack",
        height=240,
        showlegend=True,
        # traceorder="normal" keeps the legend in the same order the segments
        # are stacked; Plotly reverses it by default for horizontal stacks,
        # which puts the least valuable segment first and reads backwards.
        legend=dict(orientation="h", yanchor="bottom", y=-0.42,
                    xanchor="left", x=0, font=dict(color=INK_SECONDARY),
                    traceorder="normal"),
        xaxis=dict(range=[0, 100], ticksuffix="%"),
        bargap=0.45,
    )
    # Left margin sized for "Share of customers" — the longest category label.
    return style_axes(fig, show_xgrid=False,
                      margin=dict(l=150, r=20, t=10, b=40))


# ==============================================================================
# PAGE 2 & 3 — built next
# ==============================================================================

def page_segments(customers: pd.DataFrame, all_customers: pd.DataFrame):
    st.title("Customer Segments")
    st.markdown(
        "Every customer has been placed into one of four groups based on three "
        "things: **how recently** they last bought, **how often** they buy, and "
        "**how much** they have spent. Each group needs a different response."
    )

    if customers.empty:
        st.warning("No customers match the current filters. Widen the selection "
                   "in the sidebar to see results.")
        return

    # NOTE ON ARCHITECTURE: segment ASSIGNMENTS are read from the saved
    # Objective 2 output — nothing is re-clustered here. The aggregates below
    # are simple display sums over whichever customers the manager has filtered
    # to, which is what a filter is expected to do.
    seg = _segment_totals(customers)

    # --- HOW THE SEGMENTS COMPARE ---------------------------------------------
    st.subheader("How the segments compare")
    st.markdown(
        "Three separate panels, because the three measures are in different "
        "units and cannot share a scale. Read each panel on its own."
    )
    st.plotly_chart(
        _rfm_comparison_chart(customers),
        width="stretch",
        config={"displayModeBar": False},
    )
    st.caption(
        "**Days since last purchase** — lower is better. "
        "**Orders placed** — higher is better. "
        "**Average total spend** — higher is better."
    )

    st.divider()

    # --- WHERE EACH SEGMENT SITS ----------------------------------------------
    st.subheader("Where each segment sits")
    st.markdown(
        "One panel per segment. The coloured dots are that segment's customers; "
        "the grey dots are everyone else, shown for context. This is the same "
        "customer base viewed four times, so you can see how cleanly the groups "
        "separate."
    )
    st.plotly_chart(
        _segment_facet_chart(customers, all_customers),
        width="stretch",
        config={"displayModeBar": False},
    )
    st.caption(
        "Spending runs on a compressed scale so that customers spending RM 50 "
        "and RM 500,000 both fit on one chart. Each step to the right is ten "
        "times the previous one."
    )

    st.divider()

    # --- SEGMENT PROFILES AND ACTIONS -----------------------------------------
    st.subheader("What to do about each segment")
    st.markdown(
        "Segments are listed in priority order. Each card gives the profile, "
        "then a specific recommended action and how to measure whether it "
        "worked."
    )

    for segment in seg.index:
        content = SEGMENT_CONTENT[segment]
        row = seg.loc[segment]

        with st.container(border=True):
            # Colour swatch ties the card to the same colour used in every
            # chart on the page. The segment NAME is always present alongside
            # it, so colour never carries the identity by itself.
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:10px;'>"
                f"<span style='width:14px;height:14px;border-radius:3px;"
                f"background:{SEGMENT_COLOURS[segment]};display:inline-block;'></span>"
                f"<span style='font-size:1.35rem;font-weight:600;"
                f"color:{INK_PRIMARY};'>{segment}</span>"
                f"<span style='font-size:0.8rem;color:{INK_MUTED};"
                f"border:1px solid {GRIDLINE};border-radius:10px;"
                f"padding:2px 9px;'>{content['priority']}</span></div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**{content['headline']}**")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Customers", f"{int(row['customers']):,}")
            m2.metric("Share of customers", f"{row['pct_customers']:.0f}%")
            m3.metric("Revenue", rm_compact(row["revenue"]))
            m4.metric("Average value", rm(row["avg_value"]))

            st.markdown(f"**Who they are.** {content['who']}")

            with st.expander(f"Recommended action — {content['action_title']}"):
                st.markdown(content["action"])

    st.divider()

    with st.expander("View all segments as a table"):
        table = seg.reset_index()[
            ["Segment", "customers", "pct_customers", "revenue", "pct_revenue",
             "avg_value"]
        ].copy()
        table.columns = ["Segment", "Customers", "% of customers",
                         "Revenue (RM)", "% of revenue", "Avg value (RM)"]
        st.dataframe(
            table.style.format({
                "Customers": "{:,.0f}",
                "% of customers": "{:.1f}%",
                "Revenue (RM)": "{:,.0f}",
                "% of revenue": "{:.1f}%",
                "Avg value (RM)": "{:,.0f}",
            }),
            width="stretch",
            hide_index=True,
        )


def _rfm_comparison_chart(customers: pd.DataFrame) -> go.Figure:
    """Three side-by-side panels comparing the segments on R, F and M.

    Form choice: small multiples rather than one grouped chart, because days,
    order counts and Ringgit cannot share an axis. Putting them on one axis
    would either flatten two of the three measures to invisibility or require a
    second y-axis, which is never acceptable.
    """
    panels = [
        ("Days since last purchase", "Recency", "mean", "{:.0f}"),
        ("Orders placed", "Frequency", "mean", "{:.1f}"),
        ("Average total spend", "Monetary", "mean", None),
    ]

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=[p[0] for p in panels],
                        horizontal_spacing=0.12)

    present = [s for s in SEGMENT_ORDER if s in set(customers["Segment"])]
    agg = customers.groupby("Segment").agg(
        Recency=("Recency", "mean"),
        Frequency=("Frequency", "mean"),
        Monetary=("Monetary", "mean"),
    )

    for col_i, (title, field, _, fmt) in enumerate(panels, start=1):
        # Reversed so the highest-priority segment sits at the top of each panel.
        ordered = list(reversed(present))
        values = [agg.loc[s, field] for s in ordered]
        labels = [rm_compact(v) if fmt is None else fmt.format(v) for v in values]

        fig.add_trace(
            go.Bar(
                y=ordered,
                x=values,
                orientation="h",
                marker=dict(color=[SEGMENT_COLOURS[s] for s in ordered],
                            line=dict(color=SURFACE, width=2)),
                text=labels,
                textposition="outside",
                textfont=dict(color=INK_SECONDARY, size=12),
                cliponaxis=False,
                showlegend=False,
                hovertemplate="<b>%{y}</b><br>" + title + ": %{text}<extra></extra>",
            ),
            row=1, col=col_i,
        )
        # Headroom so the outside value labels are never clipped.
        fig.update_xaxes(range=[0, max(values) * 1.35], row=1, col=col_i,
                         showticklabels=False, showgrid=False)
        # Segment names appear once, on the leftmost panel. All three panels
        # share the same category order, so repeating the names in every panel
        # is redundant and steals width from the bars.
        fig.update_yaxes(showticklabels=(col_i == 1), row=1, col=col_i)

    fig.update_annotations(font=dict(size=13, color=INK_PRIMARY))
    fig.update_layout(height=310, bargap=0.35)
    # Left margin fits "At-Risk High Value"; top margin clears the panel titles.
    return style_axes(fig, show_xgrid=False,
                      margin=dict(l=160, r=20, t=46, b=20))


def _segment_facet_chart(customers: pd.DataFrame,
                         all_customers: pd.DataFrame) -> go.Figure:
    """2x2 small multiples: one segment highlighted against grey context.

    Form choice: this replaces a single four-colour scatter, which FAILED the
    colour checks — with every pair able to appear adjacent in a scatter, the
    yellow/orange pair measures ΔE 13.7, below the 15 normal-vision floor. In
    the faceted version each panel carries only two colours (one segment plus
    the context grey), and every segment clears the floor against that grey.
    """
    present = [s for s in SEGMENT_ORDER if s in set(customers["Segment"])]
    rows = (len(present) + 1) // 2 or 1

    fig = make_subplots(
        rows=rows, cols=2,
        subplot_titles=present,
        horizontal_spacing=0.09, vertical_spacing=0.16,
    )

    # Context layer is drawn from the unfiltered base so the backdrop stays
    # stable while the manager changes filters. Sampled for rendering speed —
    # a scatter of every customer in every panel adds nothing readable.
    context = all_customers.sample(
        min(2500, len(all_customers)), random_state=42
    )

    for i, segment in enumerate(present):
        r, c = i // 2 + 1, i % 2 + 1
        sub = customers[customers["Segment"] == segment]

        fig.add_trace(
            go.Scattergl(
                x=context["Recency"], y=context["Monetary"],
                mode="markers", name="All other customers",
                marker=dict(color=CONTEXT_GREY, size=4, opacity=0.45),
                hoverinfo="skip", showlegend=False,
            ),
            row=r, col=c,
        )
        fig.add_trace(
            go.Scattergl(
                x=sub["Recency"], y=sub["Monetary"],
                mode="markers", name=segment,
                marker=dict(color=SEGMENT_COLOURS[segment], size=5, opacity=0.75),
                showlegend=False,
                hovertemplate=("Days since purchase: %{x}<br>"
                               "Total spend: RM %{y:,.0f}<extra></extra>"),
            ),
            row=r, col=c,
        )

    fig.update_yaxes(type="log", title_text="", gridcolor=GRIDLINE, showgrid=True)
    fig.update_xaxes(title_text="", gridcolor=GRIDLINE, showgrid=False)
    for i in range(1, rows + 1):
        fig.update_yaxes(title_text="Total spend (RM)", row=i, col=1,
                         title_font=dict(size=11, color=INK_MUTED))
    fig.update_xaxes(title_text="Days since last purchase", row=rows, col=1,
                     title_font=dict(size=11, color=INK_MUTED))
    fig.update_xaxes(title_text="Days since last purchase", row=rows, col=2,
                     title_font=dict(size=11, color=INK_MUTED))

    fig.update_annotations(font=dict(size=13, color=INK_PRIMARY))
    fig.update_layout(height=310 * rows)
    # Left margin fits the log tick labels ("100k") plus the axis title;
    # top margin clears the panel titles; bottom clears the x-axis title.
    return style_axes(fig, margin=dict(l=90, r=20, t=46, b=56))


def page_prediction():
    st.title("Prediction Insights")

    comparison = load_model_comparison()
    importance = load_feature_importance()
    rec = load_recommendation()

    # --- WHAT THE MODEL ACTUALLY DOES -----------------------------------------
    st.markdown(
        "Beyond describing customers as they are today, we tested whether past "
        "buying behaviour can **predict** who will spend heavily in the months "
        "ahead."
    )
    with st.container(border=True):
        st.markdown(
            f"""
**The question the model answers**

> *Looking only at what a customer did up to a cut-off date, can we tell whether
> they will be an above-average spender in the following six months?*

The model learned from **{rec['n_customers_modelled']:,} customers** using their
behaviour in the period **{rec['observation_window']}**, and was scored on what
those customers actually went on to spend during **{rec['future_window']}**.

Because the two periods do not overlap, the model is genuinely forecasting
rather than describing something it had already seen — the same test a real
deployment would face.
"""
        )

    st.divider()

    # --- RECOMMENDED MODEL -----------------------------------------------------
    st.subheader("Recommended model")

    best_name = rec["recommended_model"]
    best = comparison[comparison["Model"] == best_name].iloc[0]

    with st.container(border=True):
        st.markdown(f"### ✅ {best_name}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{best['Accuracy']*100:.1f}%",
                  help="How often the model's call is correct.")
        c2.metric("Precision", f"{best['Precision']*100:.1f}%",
                  help="When it flags someone as a high spender, how often it is right.")
        c3.metric("Recall", f"{best['Recall']*100:.1f}%",
                  help="Of all the real high spenders, how many it successfully found.")
        c4.metric("F1 score", f"{best['F1']*100:.1f}%",
                  help="A single balanced score combining precision and recall.")

        st.markdown(
            f"""
**Why this model was chosen.** All three models tested perform within a hair of
each other — the differences are not statistically meaningful (see below). The
choice therefore came down to practical qualities rather than raw scores:

- **It generalises best.** The gap between its performance on data it trained on
  and data it had never seen is just **{best['Overfit_Gap']*100:.1f} percentage
  points** — far tighter than the alternatives. A model that scores well in
  testing but poorly in the real world is worse than useless, and this is the
  measure that catches that.
- **It can be explained.** Each factor gets a single, readable weight, so you
  can always answer "why was this customer flagged?" — which matters when a
  manager has to justify a campaign spend.
- **It is the simplest to run**, needing the least computing power to deploy and
  maintain.
"""
        )

    st.divider()

    # --- MODEL COMPARISON ------------------------------------------------------
    st.subheader("How the three models compared")
    st.plotly_chart(
        _model_comparison_chart(comparison),
        width="stretch",
        config={"displayModeBar": False},
    )

    with st.expander("View comparison as a table"):
        table = comparison[["Model", "Accuracy", "Precision", "Recall", "F1",
                            "ROC_AUC", "Overfit_Gap"]].copy()
        table.columns = ["Model", "Accuracy", "Precision", "Recall", "F1",
                         "Ranking quality (AUC)", "Overfitting gap"]
        st.dataframe(
            table.style.format({c: "{:.3f}" for c in table.columns[1:]}),
            width="stretch", hide_index=True,
        )

    # --- ARE THE DIFFERENCES REAL? --------------------------------------------
    if rec.get("models_statistically_indistinguishable"):
        st.info(
            f"""
**Are these differences real? No — and that matters.**

The models look slightly different in the chart above, but formal statistical
tests show those gaps are within the range of random chance
(p = {rec['corrected_paired_t_p']:.2f} and p = {rec['mcnemar_p']:.2f}; anything
above 0.05 means "no reliable difference").

In plain terms: **you could deploy any of the three and get the same real-world
result.** Picking the top row of a results table and calling it the winner would
be reading noise as signal. That is precisely why the recommendation above rests
on generalisation and explainability instead.
"""
        )

    st.divider()

    # --- WHAT DRIVES A PREDICTION ---------------------------------------------
    st.subheader("What drives a high-spend prediction")
    st.markdown(
        "These are the factors the recommended model relies on, and the "
        "direction each one pushes. Bars to the right raise the likelihood of "
        "being a high future spender; bars to the left reduce it."
    )
    st.plotly_chart(
        _feature_importance_chart(importance, best_name),
        width="stretch",
        config={"displayModeBar": False},
    )

    st.markdown(
        """
**Reading this in plain language**

- **How much they have spent before** is by far the strongest signal. Past
  spending is the best available guide to future spending.
- **Time since the last purchase** works against them — the longer a customer
  has been quiet, the less likely they are to spend heavily next period. This is
  why the At-Risk segment deserves urgent attention.
- **Number of orders placed** raises the likelihood: customers who return often
  keep returning.
- **Average order value** carries a small *negative* weight once total spending
  is accounted for. This is worth pausing on — it means **many smaller orders
  predict future value better than a few large ones**. A customer with a buying
  habit is more valuable than one who made a single expensive purchase.
"""
    )

    st.divider()

    # --- WHO TO CONTACT --------------------------------------------------------
    # This is the section that turns the model from an evaluation exercise into
    # something a manager can act on: the model's actual per-customer output,
    # ranked, filterable, and exportable.
    _render_prediction_table()

    st.divider()

    # --- HONEST LIMITATIONS ----------------------------------------------------
    with st.expander("Limitations — please read before acting on predictions"):
        st.markdown(
            f"""
**Accuracy is around {best['Accuracy']*100:.0f}%, and that is close to the
practical ceiling for this kind of forecast.** Extensive testing — tuning the
models, adding eleven further behavioural measures, and trying a more powerful
algorithm — moved the result by less than a percentage point. Individual
spending is genuinely difficult to predict; the limit is in the behaviour, not
in the modelling.

**The model is strongest at the extremes.** It identifies customers who stop
buying entirely, and customers who spend heavily, considerably more reliably
than it handles the middle of the range. Use it with most confidence at the top
and bottom of the customer base.

**New customers are out of scope.** {rec['excluded_cold_start_customers']:,}
customers had no purchase history before the cut-off and could not be scored. A
behavioural model cannot assess someone who has not yet behaved — predictions
apply to existing customers only.

**Geography was excluded.** {rec['state_exclusion_reason']}

**Treat predictions as a ranked shortlist, not a verdict.** The right use is
prioritising who to contact first, not deciding that a specific individual will
or will not spend.
"""
        )


def _render_prediction_table():
    """The model's actual per-customer output — a prioritised contact list.

    Everything above this point on the page answers "does the model work?".
    This section answers "so who do I contact?", which is the question a retail
    manager actually has.
    """
    st.subheader("Which customers to prioritise")

    preds = load_predictions()
    customers = load_customers()

    # Attach the segment name so a manager can see prediction and segment
    # together — "a Champion we expect to keep spending" and "an At-Risk
    # customer we expect to keep spending" warrant very different actions.
    merged = preds.merge(
        customers[["Customer ID", "Segment", "State"]],
        on="Customer ID", how="left",
    )

    n_flagged = int(merged["Predicted_HighSpender"].sum())
    st.markdown(
        f"The model scores every one of the **{len(merged):,} customers** with a "
        f"probability of being an above-average spender in the next period. It "
        f"flags **{n_flagged:,}** of them. Sort or filter this list to build a "
        "contact list, then export it."
    )

    # --- Controls -------------------------------------------------------------
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        min_prob = st.slider(
            "Minimum confidence", 0.0, 1.0, 0.50, 0.05,
            help="Only show customers the model scores at or above this probability.",
        )
    with c2:
        seg_choice = st.multiselect(
            "Segment",
            options=[s for s in SEGMENT_ORDER if s in set(merged["Segment"].dropna())],
            default=[],
            help="Leave empty to include all segments.",
        )
    with c3:
        honest_only = st.checkbox(
            "Held-out customers only",
            value=False,
            help="Show only customers the model never saw during training — "
                 "the honest measure of real-world performance.",
        )

    view = merged[merged["Probability_HighSpender"] >= min_prob]
    if seg_choice:
        view = view[view["Segment"].isin(seg_choice)]
    if honest_only:
        view = view[view["DataSplit"] == "held-out test"]

    if view.empty:
        st.warning("No customers match these filters. Lower the confidence "
                   "threshold or widen the segment selection.")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Customers in list", f"{len(view):,}")
    m2.metric("Average confidence", f"{view['Probability_HighSpender'].mean():.0%}")
    m3.metric("Model accuracy on this list", f"{view['Correct'].mean():.0%}")

    # --- The list -------------------------------------------------------------
    display = view[[
        "Customer ID", "Segment", "Probability_HighSpender",
        "Predicted_HighSpender", "Actual_FutureSpend", "DataSplit",
    ]].copy()
    display["Predicted_HighSpender"] = display["Predicted_HighSpender"].map(
        {1: "High spender", 0: "Low / no spend"})
    display.columns = ["Customer ID", "Segment", "Confidence", "Prediction",
                       "Actual spend (RM)", "Data split"]

    st.dataframe(
        display.style.format({
            "Confidence": "{:.1%}",
            "Actual spend (RM)": "{:,.0f}",
        }),
        width="stretch",
        hide_index=True,
        height=420,
    )

    st.download_button(
        "⬇ Download this list as CSV",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name="priority_customers.csv",
        mime="text/csv",
        help="Hand this to your marketing team.",
    )

    # --- The honesty control --------------------------------------------------
    # Without this, a manager comparing the 'accuracy on this list' figure
    # against the headline 71.8% would be confused by the gap — and would be
    # reading an inflated number without knowing it.
    n_train = int((view["DataSplit"] == "train").sum())
    if n_train and not honest_only:
        st.caption(
            f"ℹ️ **A note on the accuracy figure above.** {n_train:,} of these "
            f"{len(view):,} customers were used to train the model, so its "
            "predictions for them are unrealistically good — the model has "
            "effectively already seen the answer. Tick **Held-out customers "
            "only** to see performance on customers it has never encountered, "
            "which is the figure that reflects real-world use."
        )

    st.caption(
        "The *Actual spend* column is shown for transparency — it is what the "
        "customer really went on to spend, and lets you check the model's calls "
        "for yourself. In live use this column would not exist yet; that is the "
        "value the model is estimating."
    )


def _model_comparison_chart(comparison: pd.DataFrame) -> go.Figure:
    """Grouped bars: four metrics across the three models.

    Colour: categorical slots 1-3, validated all-pairs (worst CVD ΔE 9.2,
    normal-vision ΔE 24.0). Direct value labels are on every bar, which also
    satisfies the relief requirement for the sub-3:1 aqua slot.
    """
    metrics = [("Accuracy", "Accuracy"), ("Precision", "Precision"),
               ("Recall", "Recall"), ("F1", "F1 score")]
    model_colours = ["#2a78d6", "#eb6834", "#1baf7a"]

    fig = go.Figure()
    for i, model in enumerate(comparison["Model"]):
        row = comparison[comparison["Model"] == model].iloc[0]
        values = [row[m[0]] * 100 for m in metrics]
        fig.add_trace(go.Bar(
            name=model,
            x=[m[1] for m in metrics],
            y=values,
            marker=dict(color=model_colours[i % len(model_colours)],
                        line=dict(color=SURFACE, width=2)),
            text=[f"{v:.1f}" for v in values],
            textposition="outside",
            textfont=dict(color=INK_SECONDARY, size=11),
            cliponaxis=False,
            hovertemplate=f"<b>{model}</b><br>%{{x}}: %{{y:.1f}}%<extra></extra>",
        ))

    fig.update_layout(
        barmode="group",
        height=400,
        yaxis=dict(range=[0, 100], ticksuffix="%", title=""),
        legend=dict(orientation="h", yanchor="bottom", y=-0.24,
                    xanchor="left", x=0, font=dict(color=INK_SECONDARY)),
        bargap=0.32, bargroupgap=0.08,
    )
    return style_axes(fig, show_xgrid=False,
                      margin=dict(l=64, r=20, t=24, b=30))


def _feature_importance_chart(importance: pd.DataFrame,
                              model_name: str) -> go.Figure:
    """Diverging bars: which factors raise vs lower the predicted likelihood.

    Form choice: the values are signed (some factors push the prediction up,
    others down), which is the definition of a diverging encoding — one hue per
    direction with a neutral zero line, never a single-hue ramp.
    """
    df = importance[importance["Model"] == model_name].copy()

    friendly = {
        "Monetary": "How much they have spent before",
        "Recency": "Time since their last purchase",
        "Frequency": "Number of orders placed",
        "Tenure": "How long they have been a customer",
        "AvgOrderValue": "Average value per order",
    }
    df["Label"] = df["Feature"].map(friendly).fillna(df["Feature"])
    df = df.sort_values("Importance")

    colours = [DIVERGING_UP if v > 0 else DIVERGING_DOWN for v in df["Importance"]]

    fig = go.Figure(go.Bar(
        x=df["Importance"],
        y=df["Label"],
        orientation="h",
        marker=dict(color=colours, line=dict(color=SURFACE, width=2)),
        text=[("Raises" if v > 0 else "Lowers") + f" ({v:+.2f})"
              for v in df["Importance"]],
        textposition="outside",
        textfont=dict(color=INK_SECONDARY, size=11),
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Weight: %{x:+.3f}<extra></extra>",
    ))

    span = max(abs(df["Importance"].min()), abs(df["Importance"].max())) * 1.75
    fig.update_layout(
        height=340,
        xaxis=dict(range=[-span, span], title="", zeroline=True,
                   zerolinecolor=INK_MUTED, zerolinewidth=1,
                   showticklabels=False),
        showlegend=False,
    )
    # Generous left margin: these category labels are full sentences
    # ("How much they have spent before"), not short field names.
    return style_axes(fig, show_xgrid=True,
                      margin=dict(l=270, r=20, t=16, b=16))


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    try:
        customers = load_customers()
    except FileNotFoundError:
        st.error(
            "Could not find the Phase 2 output files. Run the modelling scripts "
            "first:\n\n"
            "```\npython phase2_objective2_rfm_kmeans.py\n"
            "python phase2_objective3_predictive_modelling.py\n```"
        )
        st.stop()

    page, filtered = render_sidebar(customers)

    if page == "Overview":
        page_overview(filtered, customers)
    elif page == "Customer Segments":
        page_segments(filtered, customers)
    else:
        # The prediction page reports model-level results, which are properties
        # of the trained model rather than of any customer subset — so the
        # sidebar filters deliberately do not apply here. Filtering the metrics
        # would imply the model was re-evaluated on the subset, which it was not.
        page_prediction()


if __name__ == "__main__":
    main()
