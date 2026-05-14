"""
Where Patients and the FDA Disagree — Streamlit demo
=====================================================

Run locally:
    pip install streamlit pandas numpy plotly joblib scikit-learn
    streamlit run app.py

Deploy free to Streamlit Community Cloud (https://streamlit.io/cloud):
    1. Push this repo to GitHub (include data/results/, models/, app.py, requirements.txt)
    2. Connect your GitHub at share.streamlit.io
    3. Pick the repo + branch + app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path

# ============================================================
# PAGE CONFIG (must be the first Streamlit call)
# ============================================================
st.set_page_config(
    page_title="Where Patients and the FDA Disagree",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Colour palette matching the presentation
CORAL = "#E07A5F"   # patient-side
SLATE = "#3D5A80"   # FDA-side
SAGE  = "#81B29A"   # findings
NAVY  = "#1E3A5F"
DARK  = "#2D3142"
MUTED = "#6C757D"


# ============================================================
# DATA LOADING (cached so reruns are instant)
# ============================================================
@st.cache_data
def load_data():
    """Load the per-drug divergence table and per-review predictions."""
    base = Path("data/results")
    cohort = pd.read_csv(base / "divergence_per_drug.csv")
    flagged = pd.read_csv(base / "divergence_flagged_drugs.csv")
    reviews = pd.read_csv(base / "drugscom_with_predictions.csv")
    by_cond = pd.read_csv(base / "divergence_by_condition.csv")
    return cohort, flagged, reviews, by_cond


@st.cache_resource
def load_model():
    """Load Model B (no-rating side-effects) for live prediction."""
    base = Path("models")
    return {
        "se_model_b":     joblib.load(base / "sideeffects_pipeline_no_rating.joblib"),
        "target_enc":     joblib.load(base / "target_encoders.joblib"),
        "top_conditions": joblib.load(base / "top_conditions.joblib"),
        "feature_cols":   joblib.load(base / "feature_columns.joblib"),
    }


try:
    cohort, flagged, reviews, by_cond = load_data()
    DATA_AVAILABLE = True
except Exception as e:
    DATA_AVAILABLE = False
    st.error(f"Data files not found. Expected files under `data/results/`. Error: {e}")
    st.stop()


# ============================================================
# HEADER
# ============================================================
st.markdown(
    f"""
    <div style='padding: 1rem 0 0.5rem 0;'>
        <h1 style='color: {NAVY}; margin-bottom: 0.2rem; font-size: 2.4rem;'>
            Where Patients and the FDA Disagree
        </h1>
        <p style='color: {MUTED}; font-size: 1.1rem; font-style: italic; margin: 0;'>
            A machine learning approach to drug safety signal detection
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TABS
# ============================================================
tab_explore, tab_map, tab_flagged, tab_condition, tab_about = st.tabs([
    "🔍  Explore a drug",
    "🗺️  Divergence map",
    "🚩  Flagged drugs",
    "🏥  By condition",
    "ℹ️  About",
])


# ============================================================
# TAB 1 — EXPLORE A DRUG
# ============================================================
with tab_explore:
    st.markdown("### Look up a drug's patient vs FDA profile")
    st.caption(
        "Start typing or pick from the dropdown. The list is restricted to the 697 "
        "drugs with sufficient patient reviews and FAERS data for divergence analysis."
    )

    # Sort drugs by review count descending so common drugs appear first
    drug_options = (
        cohort.sort_values("n_reviews", ascending=False)["drugName_clean"].tolist()
    )

    selected_drug = st.selectbox(
        "Drug",
        options=drug_options,
        format_func=lambda d: f"{d.title():<30}  ({cohort.loc[cohort['drugName_clean'] == d, 'n_reviews'].values[0]:.0f} reviews)",
        help="Type to filter the list. Drugs are sorted by review count."
    )

    if selected_drug:
        drug_row = cohort.loc[cohort["drugName_clean"] == selected_drug].iloc[0]

        # --- Top-level metric cards ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Patient reviews", f"{int(drug_row['n_reviews']):,}")
        col2.metric(
            "Patient-reported severity",
            f"{drug_row['patient_expected']:.2f}",
            help="Mean expected severity from Model B (no-rating side-effects). Scale 0–3."
        )
        col3.metric(
            "FDA severity (composite)",
            f"{drug_row['faers_severity']:.3f}",
            help="Weighted sum of FAERS adverse-event outcome rates."
        )
        col4.metric(
            "Divergence (z-score)",
            f"{drug_row['divergence_z']:+.2f}",
            delta=str(drug_row["flag"]),
            delta_color="inverse" if drug_row["flag"] != "agreement" else "off",
        )

        st.divider()

        # --- Interpretive narrative ---
        flag = drug_row["flag"]
        if flag == "patient-worse-than-FAERS":
            st.markdown(
                f"<div style='background:#FDF6F2;border-left:4px solid {CORAL};padding:1rem;border-radius:4px;'>"
                f"<strong style='color:{CORAL};'>🚩 Flagged: patients report worse than FDA suggests</strong><br>"
                f"This drug's patient-reported side-effect severity sits well above what FDA adverse-event "
                f"reporting would suggest. This pattern is common for drugs whose burden is in routine "
                f"day-to-day tolerability rather than catastrophic events."
                f"</div>",
                unsafe_allow_html=True,
            )
        elif flag == "FAERS-worse-than-patient":
            st.markdown(
                f"<div style='background:#F2F4F8;border-left:4px solid {SLATE};padding:1rem;border-radius:4px;'>"
                f"<strong style='color:{SLATE};'>🚩 Flagged: FDA reports worse than patients experience</strong><br>"
                f"This drug carries rare-but-serious adverse-event signals in FDA data, but most patients "
                f"in routine use describe tolerating it well. Both signals are valid — they measure "
                f"different aspects of safety."
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='background:#F0F4F1;border-left:4px solid {SAGE};padding:1rem;border-radius:4px;'>"
                f"<strong style='color:{SAGE};'>✓ Agreement</strong><br>"
                f"Patient experience and FDA reporting tell a consistent story for this drug."
                f"</div>",
                unsafe_allow_html=True,
            )

        st.divider()

        # --- Sample reviews ---
        with st.expander("📝  Sample reviews scored as 'High' severity (up to 5)", expanded=False):
            samples = reviews[
                (reviews["drugName_clean"] == selected_drug)
                & (reviews["se_pred_label"] == "High")
            ].head(5)

            if len(samples) == 0:
                st.write("No reviews predicted as High severity for this drug.")
            else:
                for i, r in samples.iterrows():
                    cond = r["condition_clean"] if pd.notna(r["condition_clean"]) else "—"
                    # Reviews can be very long; truncate for display
                    text = str(r.get("review_clean", "")) if "review_clean" in r else "(review text not stored)"
                    st.markdown(
                        f"**Condition:** *{cond}* &nbsp;·&nbsp; "
                        f"**Rating:** {r['rating']}/10 &nbsp;·&nbsp; "
                        f"**Predicted severity:** {r['se_pred_label']}"
                    )
                    if text and text != "(review text not stored)":
                        st.markdown(f"> {text[:500]}{'...' if len(text) > 500 else ''}")
                    st.divider()


# ============================================================
# TAB 2 — DIVERGENCE MAP
# ============================================================
with tab_map:
    st.markdown("### The divergence map")
    st.caption(
        "Each dot is a drug. The horizontal axis is FDA-reported severity; "
        "the vertical axis is patient-reported severity. Drugs on the diagonal agree. "
        "Hover for details; the marker size is proportional to the square root of review count."
    )

    # Filters
    fcol1, fcol2 = st.columns([1, 3])
    with fcol1:
        show_flag = st.radio(
            "Filter",
            ["All drugs", "Flagged only", "Patient-worse only", "FDA-worse only"],
            label_visibility="collapsed",
        )

    plot_df = cohort.copy()
    if show_flag == "Flagged only":
        plot_df = plot_df[plot_df["flag"] != "agreement"]
    elif show_flag == "Patient-worse only":
        plot_df = plot_df[plot_df["flag"] == "patient-worse-than-FAERS"]
    elif show_flag == "FDA-worse only":
        plot_df = plot_df[plot_df["flag"] == "FDA-worse-than-patient"]

    color_map = {
        "agreement":                "#9aa1a8",
        "patient-worse-than-FAERS": CORAL,
        "FDA-worse-than-patient":   SLATE,
    }

    fig = px.scatter(
        plot_df,
        x="faers_z",
        y="patient_z",
        color="flag",
        color_discrete_map=color_map,
        size=np.sqrt(plot_df["n_reviews"].clip(lower=1)),
        hover_data={
            "drugName_clean": True,
            "n_reviews": True,
            "patient_expected": ":.2f",
            "faers_severity": ":.3f",
            "divergence_z": ":.2f",
            "flag": True,
            "faers_z": False,
            "patient_z": False,
        },
        labels={
            "faers_z": "FDA-reported severity (z-score)",
            "patient_z": "Patient-reported severity (z-score)",
            "flag": "Divergence category",
        },
        height=650,
    )
    # Diagonal reference line
    lim = max(
        abs(plot_df["faers_z"].min()), abs(plot_df["faers_z"].max()),
        abs(plot_df["patient_z"].min()), abs(plot_df["patient_z"].max()),
    )
    fig.add_shape(
        type="line", x0=-lim, x1=lim, y0=-lim, y1=lim,
        line=dict(color="gray", dash="dash", width=1),
    )
    fig.update_layout(
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(gridcolor="#eee", zeroline=False)
    fig.update_yaxes(gridcolor="#eee", zeroline=False)

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "💡 **Tip:** click and drag to zoom; double-click to reset; "
        "use the legend to hide categories."
    )


# ============================================================
# TAB 3 — FLAGGED DRUGS
# ============================================================
with tab_flagged:
    st.markdown("### Drugs flagged as divergent")
    st.caption(
        "All 105 drugs where patient-reported and FDA-reported severity disagree "
        "by 2+ standard deviations. Click column headers to sort."
    )

    direction = st.radio(
        "Direction",
        ["Both directions", "Patient-worse-than-FDA only", "FDA-worse-than-patient only"],
        horizontal=True,
    )

    display = flagged.copy()
    if direction == "Patient-worse-than-FDA only":
        display = display[display["flag"] == "patient-worse-than-FAERS"]
    elif direction == "FDA-worse-than-patient only":
        display = display[display["flag"] == "FDA-worse-than-patient"]

    display = display.sort_values("divergence_z", key=abs, ascending=False)

    show_cols = ["drugName_clean", "flag", "n_reviews",
                 "patient_expected", "faers_severity", "divergence_z"]
    if "ci_lo" in display.columns and "ci_hi" in display.columns:
        show_cols += ["ci_lo", "ci_hi"]

    st.dataframe(
        display[show_cols].rename(columns={
            "drugName_clean": "Drug",
            "flag": "Direction",
            "n_reviews": "# reviews",
            "patient_expected": "Patient severity",
            "faers_severity": "FDA severity",
            "divergence_z": "Divergence z",
            "ci_lo": "CI lower",
            "ci_hi": "CI upper",
        }),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Patient severity": st.column_config.NumberColumn(format="%.2f"),
            "FDA severity":     st.column_config.NumberColumn(format="%.3f"),
            "Divergence z":     st.column_config.NumberColumn(format="%+.2f"),
            "CI lower":         st.column_config.NumberColumn(format="%.2f"),
            "CI upper":         st.column_config.NumberColumn(format="%.2f"),
        },
        height=550,
    )

    st.download_button(
        "📥  Download as CSV",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name="flagged_drugs.csv",
        mime="text/csv",
    )


# ============================================================
# TAB 4 — BY CONDITION
# ============================================================
with tab_condition:
    st.markdown("### Divergence patterns by condition")
    st.caption(
        "For each medical condition with ≥5 drugs in the analysis cohort, "
        "the mean divergence z-score across its drugs. Positive = patients report "
        "worse than FDA suggests."
    )

    by_cond_sorted = by_cond.sort_values("mean_divergence", ascending=True)

    fig2 = px.bar(
        by_cond_sorted.tail(20),  # top 20 by mean divergence
        x="mean_divergence",
        y="primary_condition",
        orientation="h",
        color="mean_divergence",
        color_continuous_scale=[[0, SLATE], [0.5, "#cccccc"], [1, CORAL]],
        labels={
            "mean_divergence": "Mean divergence z-score",
            "primary_condition": "Condition",
        },
        height=600,
        hover_data={"n_drugs": True, "n_patient_worse": True, "n_faers_worse": True},
    )
    fig2.update_layout(plot_bgcolor="white", coloraxis_showscale=False)
    fig2.update_yaxes(gridcolor="#eee")
    fig2.update_xaxes(gridcolor="#eee", zeroline=True, zerolinecolor=DARK)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("##### Full table")
    st.dataframe(
        by_cond.sort_values("mean_divergence", ascending=False).rename(columns={
            "primary_condition": "Condition",
            "n_drugs": "# drugs",
            "mean_divergence": "Mean divergence z",
            "n_patient_worse": "# patient-worse",
            "n_faers_worse": "# FDA-worse",
        }),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# TAB 5 — ABOUT
# ============================================================
with tab_about:
    st.markdown("### About this project")

    cA, cB = st.columns([2, 1])
    with cA:
        st.markdown(
            """
            **Where Patients and the FDA Disagree** is a machine learning capstone project
            comparing patient drug reviews (Drugs.com, Druglib.com) with FDA Adverse Event
            Reporting System (FAERS) data. The goal is to identify drugs where the two
            signal sources tell meaningfully different stories — a potential pharmacovigilance
            signal.

            **How it works:**
            1. Two ordinal classifiers are trained on Druglib (the only labelled source) to
               predict effectiveness and side-effect severity from review text + FAERS features.
            2. The trained models are applied to all 212,698 Drugs.com reviews. Predictions
               are aggregated per drug.
            3. For each drug, the patient-text-derived severity is compared to the
               FAERS-reported severity. Drugs where the two diverge by ≥2 standard deviations
               are flagged.
            4. Bootstrap confidence intervals confirm whether each flag is statistically robust.

            **Headline finding:** the most concentrated pattern is among hormonal contraceptives
            — 32 of 46 birth control drugs in the cohort are flagged as patient-worse-than-FDA.
            This is empirical evidence of a known regulatory blind spot: quality-of-life
            side effects of widely-used medications are systematically under-captured by
            adverse-event reporting databases.

            **Limitations:** Both data sources are self-selected (patient reviews oversample
            extreme experiences; FAERS reports skew toward serious events). The model picks
            up severity language but cannot determine whether the patient ultimately tolerated
            the drug. The two sources also measure different concepts of severity — FAERS
            captures rare-but-catastrophic events, while reviews capture everyday tolerability.
            The divergence quantifies this categorical gap, not "who's right".
            """
        )
    with cB:
        st.markdown(
            f"""
            <div style='background:#FAFBFC;border:1px solid #E5E7EB;padding:1rem;border-radius:6px;'>
              <h4 style='color:{NAVY};margin-top:0;'>By the numbers</h4>
              <ul style='line-height:1.6;'>
                <li><strong>212,698</strong> Drugs.com reviews scored</li>
                <li><strong>697</strong> drugs in analysis cohort</li>
                <li><strong>105</strong> flagged as divergent</li>
                <li><strong>100 of 105</strong> robust to bootstrap CI</li>
                <li><strong>+7–9%</strong> better predictions with FAERS data</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        f"<p style='text-align:center; color:{MUTED}; font-size:0.9rem;'>"
        "Built with scikit-learn, Streamlit, and Plotly. "
        "Source: Drugs.com (Kaggle), Druglib.com (UCI), openFDA FAERS API, RxNorm."
        "</p>",
        unsafe_allow_html=True,
    )
