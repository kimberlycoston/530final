# crrt_clot_app.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# =========================
# CONFIG & CONSTANTS
# =========================

st.set_page_config(
    page_title="CRRT Clot Risk Predictor",
    layout="wide"
)

# Brand colors
BLUE   = "#4C64DE"
ORANGE = "#F9AE37"
RED    = "#FF4631"

# === IMPORTANT: DEFINE YOUR FEATURE LISTS HERE ===
# Replace these placeholders with your actual feature names in the exact order
FULL_FEATURES = ['access_pressure', 'blood_flow', 'citrate', 'current_goal', 'dialysate_rate', 'effluent_pressure', 'filter_pressure', 'heparin_dose', 'hourly_patient_fluid_removal', 'prefilter_replacement_rate', 'postfilter_replacement_rate', 'replacement_rate', 'return_pressure', 'ultrafiltrate_output', 'hematocrit', 'hemoglobin', 'platelet', 'rbc', 'wbc', 'fibrinogen', 'inr', 'pt', 'ptt', 'aniongap', 'bicarbonate', 'bun', 'calcium', 'chloride', 'creatinine', 'glucose', 'sodium', 'potassium', 'lactate', 'ph', 'pco2', 'magnesium', 'phosphate', 'ldh', 'high_pressure', 'bun_creatinine_ratio', 'platelet_ptt_interaction', 'hct_bloodflow_interaction', 'wbc_rbc_ratio', 'platelet_wbc_ratio', 'flow_pressure_ratio', 'ptt_squared', 'platelets_squared', 'hour_of_day', 'day_of_week', 'is_weekend', 'is_night_shift', 'platelet_change', 'platelet_change_rate', 'ptt_change', 'ptt_change_rate', 'creatinine_change', 'creatinine_change_rate', 'hematocrit_change', 'hematocrit_change_rate']


TOP10_FEATURES = ['blood_flow', 'citrate', 'heparin_dose', 'phosphate', 'fibrinogen', 'effluent_pressure', 'filter_pressure', 'prefilter_replacement_rate', 'creatinine', 'replacement_rate']

# Example hypothetical patients (for Page 1 & 3)
# Each is a dict mapping feature -> value
# You MUST fill this with realistic values for your 57 features
HYPOTHETICAL_PATIENTS = {
    "Low clot risk patient": {
        'access_pressure': -75, 
        'blood_flow': 200,
        'citrate': 200,
        'current_goal': 25,
        'dialysate_rate': 1500,
        'effluent_pressure': 75,
        'filter_pressure': 125,
        'heparin_dose': 800,
        'hourly_patient_fluid_removal': 50,
        'prefilter_replacement_rate': 500,
        'postfilter_replacement_rate': 200,
        'replacement_rate': 700,
        'return_pressure': 110,
        'ultrafiltrate_output': 1400,

        'hematocrit': 30,
        'hemoglobin': 10.1,
        'platelet': 220,
        'rbc': 3.3,
        'wbc': 9.0,

        'fibrinogen': 350,
        'inr': 1.1,
        'pt': 13.0,
        'ptt': 32,

        'aniongap': 12,
        'bicarbonate': 23,
        'bun': 35,
        'calcium': 8.8,
        'chloride': 102,
        'creatinine': 2.5,
        'glucose': 120,
        'sodium': 138,
        'potassium': 4.1,
        'lactate': 1.2,
        'ph': 7.38,
        'pco2': 40,

        'magnesium': 1.9,
        'phosphate': 3.5,
        'ldh': 190,

        'high_pressure': 0,
        'bun_creatinine_ratio': 14,
        'platelet_ptt_interaction': 220*32,
        'hct_bloodflow_interaction': 30*200,
        'wbc_rbc_ratio': 9.0/3.3,
        'platelet_wbc_ratio': 220/9.0,
        'flow_pressure_ratio': 200/110,
        'ptt_squared': 32**2,
        'platelets_squared': 220**2,

        'hour_of_day': 10,
        'day_of_week': 2,
        'is_weekend': 0,
        'is_night_shift': 0,

        'platelet_change': 0,
        'platelet_change_rate': 0,
        'ptt_change': 0,
        'ptt_change_rate': 0,
        'creatinine_change': 0,
        'creatinine_change_rate': 0,
        'hematocrit_change': 0,
        'hematocrit_change_rate': 0
    },

    "Moderate clot risk patient": {
        'access_pressure': -110,
        'blood_flow': 180,
        'citrate': 180,
        'current_goal': 20,
        'dialysate_rate': 1600,
        'effluent_pressure': 95,
        'filter_pressure': 165,
        'heparin_dose': 500,
        'hourly_patient_fluid_removal': 80,
        'prefilter_replacement_rate': 600,
        'postfilter_replacement_rate': 200,
        'replacement_rate': 800,
        'return_pressure': 130,
        'ultrafiltrate_output': 1550,

        'hematocrit': 28,
        'hemoglobin': 9.3,
        'platelet': 160,
        'rbc': 3.0,
        'wbc': 13.5,

        'fibrinogen': 400,
        'inr': 1.2,
        'pt': 13.5,
        'ptt': 35,

        'aniongap': 15,
        'bicarbonate': 20,
        'bun': 45,
        'calcium': 8.4,
        'chloride': 104,
        'creatinine': 3.0,
        'glucose': 145,
        'sodium': 140,
        'potassium': 4.5,
        'lactate': 1.8,
        'ph': 7.34,
        'pco2': 38,

        'magnesium': 2.0,
        'phosphate': 4.1,
        'ldh': 250,

        'high_pressure': 0,
        'bun_creatinine_ratio': 15,
        'platelet_ptt_interaction': 160*35,
        'hct_bloodflow_interaction': 28*180,
        'wbc_rbc_ratio': 13.5/3.0,
        'platelet_wbc_ratio': 160/13.5,
        'flow_pressure_ratio': 180/130,
        'ptt_squared': 35**2,
        'platelets_squared': 160**2,

        'hour_of_day': 3,
        'day_of_week': 4,
        'is_weekend': 0,
        'is_night_shift': 1,

        'platelet_change': -10,
        'platelet_change_rate': -10/220,
        'ptt_change': 3,
        'ptt_change_rate': 3/32,
        'creatinine_change': 0.3,
        'creatinine_change_rate': 0.3/2.7,
        'hematocrit_change': -2,
        'hematocrit_change_rate': -2/30
    },

    "High clot risk patient": {
        'access_pressure': -150,
        'blood_flow': 170,
        'citrate': 150,
        'current_goal': 20,
        'dialysate_rate': 1800,
        'effluent_pressure': 125,
        'filter_pressure': 220,
        'heparin_dose': 300,
        'hourly_patient_fluid_removal': 120,
        'prefilter_replacement_rate': 700,
        'postfilter_replacement_rate': 300,
        'replacement_rate': 1000,
        'return_pressure': 150,
        'ultrafiltrate_output': 1500,

        'hematocrit': 26,
        'hemoglobin': 8.7,
        'platelet': 95,
        'rbc': 2.8,
        'wbc': 18.0,

        'fibrinogen': 500,
        'inr': 1.4,
        'pt': 15.8,
        'ptt': 48,

        'aniongap': 18,
        'bicarbonate': 18,
        'bun': 60,
        'calcium': 8.2,
        'chloride': 105,
        'creatinine': 3.8,
        'glucose': 165,
        'sodium': 141,
        'potassium': 4.9,
        'lactate': 2.8,
        'ph': 7.29,
        'pco2': 35,

        'magnesium': 2.1,
        'phosphate': 4.7,
        'ldh': 380,

        'high_pressure': 1,
        'bun_creatinine_ratio': 60/3.8,
        'platelet_ptt_interaction': 95*48,
        'hct_bloodflow_interaction': 26*170,
        'wbc_rbc_ratio': 18.0/2.8,
        'platelet_wbc_ratio': 95/18.0,
        'flow_pressure_ratio': 170/150,
        'ptt_squared': 48**2,
        'platelets_squared': 95**2,

        'hour_of_day': 22,
        'day_of_week': 6,
        'is_weekend': 1,
        'is_night_shift': 1,

        'platelet_change': -35,
        'platelet_change_rate': -35/130,
        'ptt_change': 12,
        'ptt_change_rate': 12/36,
        'creatinine_change': 0.6,
        'creatinine_change_rate': 0.6/3.2,
        'hematocrit_change': -3,
        'hematocrit_change_rate': -3/29
    },

    "Very high clot risk patient": {
        'access_pressure': -180,
        'blood_flow': 150,
        'citrate': 100,
        'current_goal': 15,
        'dialysate_rate': 2000,
        'effluent_pressure': 150,
        'filter_pressure': 260,
        'heparin_dose': 200,
        'hourly_patient_fluid_removal': 150,
        'prefilter_replacement_rate': 800,
        'postfilter_replacement_rate': 400,
        'replacement_rate': 1200,
        'return_pressure': 165,
        'ultrafiltrate_output': 1450,

        'hematocrit': 25,
        'hemoglobin': 8.3,
        'platelet': 65,
        'rbc': 2.6,
        'wbc': 22.0,

        'fibrinogen': 650,
        'inr': 1.5,
        'pt': 17.5,
        'ptt': 52,

        'aniongap': 21,
        'bicarbonate': 16,
        'bun': 70,
        'calcium': 8.0,
        'chloride': 106,
        'creatinine': 4.1,
        'glucose': 185,
        'sodium': 140,
        'potassium': 5.2,
        'lactate': 3.9,
        'ph': 7.24,
        'pco2': 33,

        'magnesium': 2.2,
        'phosphate': 5.2,
        'ldh': 500,

        'high_pressure': 1,
        'bun_creatinine_ratio': 70/4.1,
        'platelet_ptt_interaction': 65*52,
        'hct_bloodflow_interaction': 25*150,
        'wbc_rbc_ratio': 22.0/2.6,
        'platelet_wbc_ratio': 65/22.0,
        'flow_pressure_ratio': 150/165,
        'ptt_squared': 52**2,
        'platelets_squared': 65**2,

        'hour_of_day': 17,
        'day_of_week': 6,
        'is_weekend': 1,
        'is_night_shift': 0,

        'platelet_change': -45,
        'platelet_change_rate': -45/110,
        'ptt_change': 18,
        'ptt_change_rate': 18/34,
        'creatinine_change': 1.0,
        'creatinine_change_rate': 1.0/3.1,
        'hematocrit_change': -4,
        'hematocrit_change_rate': -4/29
    },

    "Anticoagulated bleeding-risk patient": {
        'access_pressure': -70,
        'blood_flow': 200,
        'citrate': 220,
        'current_goal': 30,
        'dialysate_rate': 1400,
        'effluent_pressure': 55,
        'filter_pressure': 90,
        'heparin_dose': 1500,
        'hourly_patient_fluid_removal': 40,
        'prefilter_replacement_rate': 450,
        'postfilter_replacement_rate': 150,
        'replacement_rate': 600,
        'return_pressure': 90,
        'ultrafiltrate_output': 1300,

        'hematocrit': 27,
        'hemoglobin': 9.0,
        'platelet': 110,
        'rbc': 2.9,
        'wbc': 7.5,

        'fibrinogen': 250,
        'inr': 2.2,
        'pt': 22,
        'ptt': 78,

        'aniongap': 10,
        'bicarbonate': 23,
        'bun': 38,
        'calcium': 8.6,
        'chloride': 101,
        'creatinine': 2.2,
        'glucose': 105,
        'sodium': 137,
        'potassium': 4.0,
        'lactate': 0.9,
        'ph': 7.40,
        'pco2': 41,

        'magnesium': 1.7,
        'phosphate': 3.9,
        'ldh': 180,

        'high_pressure': 0,
        'bun_creatinine_ratio': 38/2.2,
        'platelet_ptt_interaction': 110*78,
        'hct_bloodflow_interaction': 27*200,
        'wbc_rbc_ratio': 7.5/2.9,
        'platelet_wbc_ratio': 110/7.5,
        'flow_pressure_ratio': 200/90,
        'ptt_squared': 78**2,
        'platelets_squared': 110**2,

        'hour_of_day': 8,
        'day_of_week': 1,
        'is_weekend': 0,
        'is_night_shift': 0,

        'platelet_change': -5,
        'platelet_change_rate': -5/115,
        'ptt_change': 22,
        'ptt_change_rate': 22/56,
        'creatinine_change': -0.1,
        'creatinine_change_rate': -0.1/2.3,
        'hematocrit_change': 0,
        'hematocrit_change_rate': 0
    }
}

STEP_SIZES = {
    'access_pressure': 5,
    'blood_flow': 10,
    'citrate': 10,
    'current_goal': 1,
    'dialysate_rate': 50,
    'effluent_pressure': 5,
    'filter_pressure': 5,
    'heparin_dose': 50,
    'hourly_patient_fluid_removal': 10,
    'prefilter_replacement_rate': 50,
    'postfilter_replacement_rate': 50,
    'replacement_rate': 50,
    'return_pressure': 5,
    'ultrafiltrate_output': 50,

    'hematocrit': 0.5,
    'hemoglobin': 0.1,
    'platelet': 5,
    'rbc': 0.1,
    'wbc': 0.1,
    'fibrinogen': 10,
    'inr': 0.1,
    'pt': 0.1,
    'ptt': 1,

    'aniongap': 1,
    'bicarbonate': 1,
    'bun': 1,
    'calcium': 0.1,
    'chloride': 1,
    'creatinine': 0.1,
    'glucose': 5,
    'sodium': 1,
    'potassium': 0.1,
    'lactate': 0.1,
    'ph': 0.01,
    'pco2': 1,

    'magnesium': 0.1,
    'phosphate': 0.1,
    'ldh': 10,

    'high_pressure': 1,
    'flow_pressure_ratio': 0.01,

    'hour_of_day': 1,
    'day_of_week': 1,
    'is_weekend': 1,
    'is_night_shift': 1,

    'platelet_change': 1,
    'platelet_change_rate': 0.01,
    'ptt_change': 1,
    'ptt_change_rate': 0.01,
    'creatinine_change': 0.1,
    'creatinine_change_rate': 0.01,
    'hematocrit_change': 0.1,
    'hematocrit_change_rate': 0.01
}

DERIVED_FEATURES = [
    'bun_creatinine_ratio',
    'platelet_ptt_interaction',
    'hct_bloodflow_interaction',
    'wbc_rbc_ratio',
    'platelet_wbc_ratio',
    'flow_pressure_ratio',
    'ptt_squared',
    'platelets_squared',
]

CHANGE_DERIVED = [
    'platelet_change_rate',
    'ptt_change_rate',
    'creatinine_change_rate',
    'hematocrit_change_rate'
]

def compute_derived_features(data):
    d = data.copy()

    # Avoid zero-division where needed
    safe = lambda x: x if x != 0 else 1

    # Core derived features
    d['bun_creatinine_ratio'] = d['bun'] / safe(d['creatinine'])
    d['platelet_ptt_interaction'] = d['platelet'] * d['ptt']
    d['hct_bloodflow_interaction'] = d['hematocrit'] * d['blood_flow']
    d['wbc_rbc_ratio'] = d['wbc'] / safe(d['rbc'])
    d['platelet_wbc_ratio'] = d['platelet'] / safe(d['wbc'])
    d['flow_pressure_ratio'] = d['blood_flow'] / safe(d['return_pressure'])
    d['ptt_squared'] = d['ptt'] ** 2
    d['platelets_squared'] = d['platelet'] ** 2

    # Change-rate derived features
    d['platelet_change_rate'] = d['platelet_change'] / safe(d['platelet'])
    d['ptt_change_rate'] = d['ptt_change'] / safe(d['ptt'])
    d['creatinine_change_rate'] = d['creatinine_change'] / safe(d['creatinine'])
    d['hematocrit_change_rate'] = d['hematocrit_change'] / safe(d['hematocrit'])

    return d


# =========================
# LOAD MODELS & OBJECTS
# =========================

@st.cache_resource
def load_models():
    xgb_full = joblib.load("xgb_full_57.pkl")     # Full 57-feature XGB
    xgb_top10 = joblib.load("xgb_top10.pkl")      # Top-10 XGB
    scaler_full = joblib.load("scaler_full.pkl")  # StandardScaler fit on all features
    return xgb_full, xgb_top10, scaler_full

@st.cache_data
def load_dataset_for_ranges():
    """Load dataset to get min/max values for feature inputs"""
    try:
        df = pd.read_csv("datasetv4.csv")
        return df
    except FileNotFoundError:
        # Return None if dataset not found - will use defaults
        return None

xgb_full, xgb_top10, scaler_full = load_models()
df_ranges = load_dataset_for_ranges()

# SHAP explainer for full model (use TreeExplainer for XGBoost)
@st.cache_resource
def get_shap_explainer_full():
    return shap.TreeExplainer(xgb_full)

@st.cache_resource
def get_shap_explainer_top10():
    return shap.TreeExplainer(xgb_top10)

explainer_full = get_shap_explainer_full()
explainer_top10 = get_shap_explainer_top10()

# =========================
# HELPER FUNCTIONS
# =========================

def scale_full_features(df_row: pd.DataFrame):
    """
    df_row: DataFrame with one row and columns FULL_FEATURES
    Returns scaled numpy array with same structure.
    """
    # Ensure correct column order
    df_row = df_row[FULL_FEATURES]
    arr_scaled = scaler_full.transform(df_row)
    return arr_scaled

def scale_top10_features(df_row_top10: pd.DataFrame):
    """
    df_row_top10: DataFrame with one row containing only TOP10_FEATURES.
    We still use the full scaler, but subset afterwards to match
    how XGB_top10 was trained (on X_train_scaled[TOP10_FEATURES]).
    """
    # Create dummy full row with NaNs, fill only top10, then scale, then slice
    full_row = pd.DataFrame(columns=FULL_FEATURES)
    full_row.loc[0] = np.nan
    for feat in TOP10_FEATURES:
        full_row.loc[0, feat] = df_row_top10.loc[0, feat]
    # Use median for missing features
    full_row = full_row.fillna(0)  # or use stored medians if you have them
    scaled_full = scaler_full.transform(full_row)
    scaled_full_df = pd.DataFrame(scaled_full, columns=FULL_FEATURES)
    scaled_top10 = scaled_full_df[TOP10_FEATURES].values
    return scaled_top10

def plot_shap_bar(shap_values, feature_names, title):
    shap_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_values
    }).sort_values("shap_value", key=np.abs, ascending=False).head(10)

    # Color: red for positive (↑ risk), blue for negative (↓ risk)
    shap_df["color"] = shap_df["shap_value"].apply(
        lambda v: RED if v > 0 else BLUE
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        shap_df["feature"],
        shap_df["shap_value"],
        color=shap_df["color"],
        edgecolor="black",
        linewidth=1.2
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("SHAP Value (Impact on Predicted Clot Risk)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Feature", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig

def generate_llm_explanation_stub(patient_dict, risk_score, top_shap):
    """
    Stub / template for LLM explanation.
    For the class project, you can either:
    - leave this stub text, OR
    - hook in OpenAI / Azure with a real API call.

    patient_dict: raw feature values
    risk_score: float, 0-1
    top_shap: DataFrame with top SHAP features
    """
    risk_pct = risk_score * 100
    lines = []
    lines.append(f"Predicted clot risk: **{risk_pct:.1f}%**.")
    if risk_score >= 0.8:
        lines.append("This patient is at **high risk** of CRRT circuit clotting.")
    elif risk_score >= 0.4:
        lines.append("This patient is at **moderate risk** of CRRT circuit clotting.")
    else:
        lines.append("This patient is at **low risk** of CRRT circuit clotting.")

    lines.append("")
    lines.append("Key drivers of this prediction:")

    for _, row in top_shap.iterrows():
        feat = row["feature"]
        val = patient_dict.get(feat, "N/A")
        direction = "increases" if row["shap_value"] > 0 else "decreases"
        lines.append(f"- **{feat} = {val}** → {direction} clot risk")

    lines.append("")
    lines.append(
        "In a real deployment, this panel could be auto-populated from Epic and an LLM "
        "could generate a more detailed narrative tying these factors to CRRT management decisions."
    )

    return "\n".join(lines)


import plotly.graph_objects as go

def plot_risk_gauge(probability, title="Predicted Clot Risk"):
    """
    probability: float between 0–1
    """
    pct = probability * 100

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={'suffix': "%"},
            title={'text': title, 'font': {'size': 20}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "black"},
                'bar': {'color': "black"},  # gauge needle color

                # Background color bands
                'steps': [
                    {'range': [0, 30],  'color': 'green'},   # Blue = low
                    {'range': [30, 70], 'color': 'yellow'},   # Orange = moderate
                    {'range': [70, 100],'color': '#FF4631'}    # Red = high
                ],

                # Threshold line (thin red line at value)
                'threshold': {
                    'line': {'color': '#000000', 'width': 4},
                    'thickness': 0.8,
                    'value': pct
                }
            }
        )
    )

    fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))

    return fig


# =========================
# SIDEBAR NAVIGATION
# =========================

st.sidebar.title("CRRT Clot Prediction")
page = st.sidebar.radio(
    "Choose a view:",
    [
        "1️⃣ AI Clinical Interpretation (Epic-style, 57 features)",
        "2️⃣ Clinician Input (Top 10 Features)",
        "3️⃣ Full Feature Explorer (57 Features)"
    ]
)

# =========================
# PAGE 1: EPIC-STYLE AI INTERPRETATION
# =========================

if page.startswith("1️⃣"):
    st.title("AI Clinical Interpretation (57 Features)")

    st.markdown(
        "This page simulates a future Epic integration where **all relevant labs, "
        "pressures, and CRRT parameters are auto-pulled**, the full 57-feature model "
        "runs in the background, and an AI system explains *why* clot risk is high or low."
    )

    # --- Select hypothetical patient ---
    patient_choice = st.selectbox(
        "Select hypothetical patient",
        list(HYPOTHETICAL_PATIENTS.keys())
    )
    patient_data = HYPOTHETICAL_PATIENTS[patient_choice]

    # Sanity check
    missing_feats = [f for f in FULL_FEATURES if f not in patient_data]
    if missing_feats:
        st.error(f"Missing values for features: {missing_feats}. Fill HYPOTHETICAL_PATIENTS in the code.")
        st.stop()

    # Create DataFrame
    df_patient = pd.DataFrame([patient_data], columns=FULL_FEATURES)

    # Scale and predict
    scaled_patient = scale_full_features(df_patient)
    prob = xgb_full.predict_proba(scaled_patient)[0, 1]

    st.subheader(f"Predicted Clot Risk: :red[{prob*100:.1f}%]")

    # SHAP values
    shap_vals_full = explainer_full.shap_values(scaled_patient)[0]
    shap_df_full = pd.DataFrame({
        "feature": FULL_FEATURES,
        "shap_value": shap_vals_full
    }).sort_values("shap_value", key=np.abs, ascending=False).head(10)

    # Layout: left = SHAP bar, right = LLM text
    col1, col2 = st.columns([1.1, 0.9])

    with col1:
        st.markdown("### Top Feature Contributors")
        fig_shap = plot_shap_bar(
            shap_vals_full,
            FULL_FEATURES,
            "Top 10 SHAP Contributions (Full 57-feature Model)"
        )
        st.pyplot(fig_shap)

    with col2:
        st.markdown("### Narrative Explanation (LLM-style)")
        explanation_md = generate_llm_explanation_stub(patient_data, prob, shap_df_full)
        st.markdown(explanation_md)

# =========================
# PAGE 2: CLINICIAN INPUT (TOP 10 FEATURES)
# =========================

elif page.startswith("2️⃣"):
    st.title("Clinician Input — Deployed Top 10 Feature Model")

    st.markdown(
        "This page represents a **clinically deployable** version of the model, where a nurse "
        "or provider manually enters a small set of inputs (top 10 features), and the model "
        "returns predicted clot risk and key drivers."
    )

    # Load dataset ranges safely
    df = df_ranges

    # INPUT FORM
    with st.form("top10_form"):

        st.subheader("Enter CRRT parameters and key labs")

        input_values = {}
        cols = st.columns(2)

        for i, feat in enumerate(TOP10_FEATURES):
            with cols[i % 2]:

                # determine min/max/default safely
                if df is not None and feat in df.columns:
                    min_val = float(df[feat].min())
                    max_val = float(df[feat].max())
                    default_val = float(df[feat].mean())
                else:
                    min_val = 0.0
                    max_val = 9999.0
                    default_val = 0.0

                step = float(STEP_SIZES.get(feat, 1.0))

                input_values[feat] = st.number_input(
                    label=feat,
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    step=step,
                    format="%.3f",
                    help="Add tooltip later here."
                )

        submitted = st.form_submit_button("Predict Clot Risk")

    # PROCESS PREDICTION
    if submitted:

        # Convert to dataframe
        df_top10 = pd.DataFrame([input_values], columns=TOP10_FEATURES)

        # Scale features
        scaled_top10 = scale_top10_features(df_top10)

        # COMPUTE PREDICTION
        prob_top10 = xgb_top10.predict_proba(scaled_top10)[0, 1]  # <== NOW DEFINED

        st.subheader("Predicted Clot Risk")

        # === GAUGE VISUALIZATION ===
        gauge_fig = plot_risk_gauge(prob_top10, "Clot Risk (%)")
        st.plotly_chart(gauge_fig, use_container_width=True)

        # === SHAP ===
        shap_vals_top10 = explainer_top10.shap_values(scaled_top10)[0]
        fig_shap_top10 = plot_shap_bar(
            shap_vals_top10,
            TOP10_FEATURES,
            "Top 10 SHAP Contributions (Deployed Top-10 Model)"
        )
        st.pyplot(fig_shap_top10)

        st.markdown(
            "> This model uses **only 10 features**, making it more realistic for "
            "manual or semi-automated data collection at the bedside. The tradeoff is a small "
            "loss of performance versus the full 57-feature model."
        )


# =========================
# PAGE 3: FULL FEATURE EXPLORER
# =========================

elif page.startswith("3️⃣"):
    st.title("Full Feature Explorer (57 Features)")

    st.markdown(
        "This page exposes all 57 features used by the full XGBoost model. In a real system, "
        "these values would be populated directly from Epic and the CRRT machine. "
        "Here, you can tweak them manually to see how risk changes."
    )

    col_left, col_right = st.columns([1.0, 1.2])

    with col_left:
        st.markdown("### Select base patient")
        selected_patient = st.selectbox(
            "Start from hypothetical patient:",
            list(HYPOTHETICAL_PATIENTS.keys())
        )
        base_patient = HYPOTHETICAL_PATIENTS[selected_patient]

        if not base_patient:
            st.error("HYPOTHETICAL_PATIENTS must be populated with all 57 features.")
            st.stop()

        st.subheader("Enter or adjust patient values")

        patient_input = {}

        # Loop through all features
        for feature in FULL_FEATURES:

            # Skip derived features — we compute those AFTER inputs
            if feature in DERIVED_FEATURES or feature in CHANGE_DERIVED:
                continue

            # Default value from hypothetical patient
            default_val = float(base_patient.get(feature, 0.0))

            # Step size based on your STEP_SIZES dictionary
            step = float(STEP_SIZES.get(feature, 1.0))

            # If df_ranges exists, use real min/max; otherwise fallback
            if df_ranges is not None and feature in df_ranges.columns:
                min_val = float(df_ranges[feature].min())
                max_val = float(df_ranges[feature].max())
            else:
                # Fallback generic range
                min_val = default_val - abs(default_val) * 2 - 5
                max_val = default_val + abs(default_val) * 2 + 5

            # Build number input
            patient_input[feature] = st.number_input(
                label=feature,
                min_value=min_val,
                max_value=max_val,
                value=default_val,
                step=step,
                format="%.3f",
                key=f"full_{feature}"
            )

        # Compute derived features AFTER base features entered
        patient_input = compute_derived_features(patient_input)

        st.subheader("Automatically Calculated Derived Features")
        for feature in DERIVED_FEATURES + CHANGE_DERIVED:
            st.write(f"**{feature}:** {patient_input[feature]:.4f}")

        # Run model
        if st.button("Run Full Model"):
            df_full = pd.DataFrame([patient_input], columns=FULL_FEATURES)
            scaled_full = scale_full_features(df_full)
            prob_full = xgb_full.predict_proba(scaled_full)[0, 1]

            st.session_state["full_prob"] = prob_full
            st.session_state["full_input"] = patient_input
            st.session_state["full_scaled"] = scaled_full

    with col_right:
        if "full_prob" in st.session_state:
            prob_full = st.session_state["full_prob"]
            edited_values = st.session_state["full_input"]
            scaled_full = st.session_state["full_scaled"]

            st.subheader(f"Predicted Clot Risk: :red[{prob_full*100:.1f}%]")

            shap_vals_full_explorer = explainer_full.shap_values(scaled_full)[0]
            fig_shap_full = plot_shap_bar(
                shap_vals_full_explorer,
                FULL_FEATURES,
                "Top 10 SHAP Contributions (Full 57-feature Model)"
            )
            st.pyplot(fig_shap_full)

            shap_df_full = pd.DataFrame({
                "feature": FULL_FEATURES,
                "shap_value": shap_vals_full_explorer
            }).sort_values("shap_value", key=np.abs, ascending=False).head(10)

            st.markdown("### Narrative Explanation (Stub)")
            explanation_md = generate_llm_explanation_stub(
                edited_values, prob_full, shap_df_full
            )
            st.markdown(explanation_md)

        else:
            st.info("Adjust features on the left and click **Run Full Model** to see results.")
