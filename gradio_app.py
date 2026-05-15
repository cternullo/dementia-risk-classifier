"""
Gradio interface for the Alzheimer's / Dementia Risk Classifier.

Loads the trained Random Forest model (rf_model.pkl) and StandardScaler
(scaler.pkl) saved by the Jupyter notebook, then exposes a simple web UI
where a clinician or researcher can enter patient values and get a risk
probability and category back.

Run with:  python gradio_app.py
"""

import numpy as np
import pandas as pd
import joblib
import gradio as gr

# ---------------------------------------------------------------------------
# Load artefacts saved by the notebook
# ---------------------------------------------------------------------------
scaler   = joblib.load("scaler.pkl")
rf_model = joblib.load("rf_model.pkl")

FEATURE_ORDER = ["M/F", "Age", "EDUC", "SES", "MMSE", "eTIV", "nWBV", "ASF"]


# ---------------------------------------------------------------------------
# Prediction function
# ---------------------------------------------------------------------------
def predict_dementia_risk(gender, age, educ, ses, mmse, etiv, nwbv, asf):
    """
    Preprocess raw inputs, run the Random Forest, and return a formatted
    probability string plus a risk category label.
    """
    gender_encoded = 0 if gender == "Male" else 1

    raw = pd.DataFrame(
        [[gender_encoded, age, educ, ses, mmse, etiv, nwbv, asf]],
        columns=FEATURE_ORDER
    )
    raw_scaled = scaler.transform(raw)

    prob = rf_model.predict_proba(raw_scaled)[0, 1]
    pct  = prob * 100

    risk_label = "🔴 High Risk" if prob >= 0.5 else "🟢 Low Risk"
    return f"{pct:.1f}%", risk_label


# ---------------------------------------------------------------------------
# Gradio UI layout
# ---------------------------------------------------------------------------
TITLE = "🧠 Cognitive Decline Risk Classifier"

DESCRIPTION = """
Uses a **Random Forest** model trained on the OASIS Longitudinal Dementia
Dataset to estimate the probability that a patient has dementia based on
demographic and neuroimaging features.

**Threshold:** probability ≥ 50% → High Risk; < 50% → Low Risk.
"""

INPUT_INFO = {
    "Age":  "Patient age in years.",
    "EDUC": "Years of education completed.",
    "SES":  "Socioeconomic status — 1 (highest) to 5 (lowest).",
    "MMSE": "Mini-Mental State Examination score (0–30). Lower = more cognitive impairment.",
    "eTIV": "Estimated Total Intracranial Volume (mm³). Typical range 1100–1900.",
    "nWBV": "Normalised Whole Brain Volume — brain volume as a fraction of intracranial volume. "
            "Typical range 0.65–0.83.",
    "ASF":  "Atlas Scaling Factor — MRI intensity normalisation coefficient. Typical range 0.88–1.58.",
}

with gr.Blocks(title=TITLE) as demo:

    gr.Markdown(f"# {TITLE}")
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Patient Inputs")

            gender = gr.Radio(
                choices=["Male", "Female"],
                value="Male",
                label="Sex",
                info="Patient biological sex."
            )
            age = gr.Slider(
                minimum=18, maximum=100, value=72, step=1,
                label="Age (years)",
                info=INPUT_INFO["Age"]
            )
            educ = gr.Slider(
                minimum=0, maximum=23, value=14, step=1,
                label="Education (years)",
                info=INPUT_INFO["EDUC"]
            )
            ses = gr.Slider(
                minimum=1, maximum=5, value=2, step=1,
                label="Socioeconomic Status (SES)",
                info=INPUT_INFO["SES"]
            )
            mmse = gr.Slider(
                minimum=0, maximum=30, value=27, step=1,
                label="MMSE Score",
                info=INPUT_INFO["MMSE"]
            )
            etiv = gr.Number(
                value=1500, label="eTIV (mm³)",
                info=INPUT_INFO["eTIV"]
            )
            nwbv = gr.Number(
                value=0.740, label="nWBV",
                info=INPUT_INFO["nWBV"]
            )
            asf = gr.Number(
                value=1.200, label="ASF",
                info=INPUT_INFO["ASF"]
            )

            submit_btn = gr.Button("Predict Risk", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### Results")

            prob_out  = gr.Textbox(
                label="Dementia Probability",
                placeholder="—",
                interactive=False
            )
            label_out = gr.Textbox(
                label="Risk Category",
                placeholder="—",
                interactive=False
            )

            gr.Markdown(
                """
                **How to interpret:**
                - **🟢 Low Risk** — model estimates < 50% probability of dementia.
                - **🔴 High Risk** — model estimates ≥ 50% probability of dementia.

                > ⚠️ This tool is for research and educational purposes only.
                > It is **not** a clinical diagnostic instrument.
                """
            )

    submit_btn.click(
        fn=predict_dementia_risk,
        inputs=[gender, age, educ, ses, mmse, etiv, nwbv, asf],
        outputs=[prob_out, label_out]
    )

    gr.Examples(
        examples=[
            # Likely non-demented: educated, high MMSE, younger
            ["Male",   65, 16, 1, 29, 1700, 0.780, 1.050],
            # Borderline: older, lower MMSE
            ["Female", 78, 12, 3, 24, 1400, 0.710, 1.200],
            # High risk: advanced age, low MMSE, low education
            ["Male",   85,  8, 4, 18, 1300, 0.670, 1.400],
        ],
        inputs=[gender, age, educ, ses, mmse, etiv, nwbv, asf],
        label="Example Patients"
    )


if __name__ == "__main__":
    demo.launch(share=False, theme=gr.themes.Soft())
