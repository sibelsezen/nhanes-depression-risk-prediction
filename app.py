# ============================================================
# NHANES Depression-Risk Screening Application
# Section 1: Imports, page configuration, and model loading
# ============================================================

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# Configure the Streamlit browser tab and page layout.
# This must be the first Streamlit command in the application.
st.set_page_config(
    page_title="NHANES Depression-Risk Screening",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Build reliable file paths relative to app.py.
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "final_depression_risk_model.joblib"


@st.cache_resource
def load_model_bundle():
    """
    Load the fitted preprocessing and classification pipeline.

    Streamlit caches the result so the model is loaded only once,
    rather than every time the user changes an input.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file was not found at: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


# Load and validate the saved model bundle.
try:
    model_bundle = load_model_bundle()

    required_items = {
        "pipeline",
        "selected_predictors",
        "prediction_threshold",
        "target_column",
    }

    missing_items = required_items.difference(model_bundle.keys())

    if missing_items:
        raise KeyError(
            f"Missing model-bundle items: {sorted(missing_items)}"
        )

    model_pipeline = model_bundle["pipeline"]
    selected_predictors = model_bundle["selected_predictors"]
    prediction_threshold = float(
        model_bundle["prediction_threshold"]
    )
    target_column = model_bundle["target_column"]

except Exception as error:
    st.error(
        "The saved prediction model could not be loaded. "
        "Please verify the model file and application path."
    )
    st.exception(error)
    st.stop()


# ---------------------------------------------------------
# PAGE HEADER
# ---------------------------------------------------------

st.title("Depression Risk Screening Tool")

st.write(
    """
    This application uses a trained machine-learning model to estimate
    elevated depression risk from participant health and lifestyle information.
    """
)

st.info(
    "This tool is for educational and screening purposes only. "
    "It does not provide a medical diagnosis."
)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:
    st.header("Model Information")
    st.success("Model loaded successfully")
    st.write(f"**Input predictors:** {len(selected_predictors)}")
    st.write(f"**Decision threshold:** {prediction_threshold:.4f}")
    st.write("**Model:** Class-balanced Logistic Regression")

# ---------------------------------------------------------
# PARTICIPANT INPUT FORM — NUMERIC FEATURES
# ---------------------------------------------------------

st.header("Participant Information")
st.write("Enter the participant's health and lifestyle information below.")

column_1, column_2, column_3 = st.columns(3)

with column_1:
    age = st.number_input(
        "Age (years)",
        min_value=18,
        max_value=80,
        value=40,
        step=1,
    )

    income_ratio = st.number_input(
        "Family income-to-poverty ratio",
        min_value=0.0,
        max_value=5.0,
        value=2.0,
        step=0.1,
    )

    sedentary_minutes = st.number_input(
        "Sedentary time (minutes/day)",
        min_value=0,
        max_value=1320,
        value=300,
        step=10,
    )

    cigarettes_per_day = st.number_input(
        "Cigarettes per day",
        min_value=0,
        max_value=60,
        value=0,
        step=1,
    )

with column_2:
    alcohol_frequency = st.number_input(
        "Alcohol-use frequency",
        min_value=0.0,
        value=0.0,
        step=1.0,
    )

    average_drinks = st.number_input(
        "Average drinks per day",
        min_value=0.0,
        max_value=15.0,
        value=0.0,
        step=0.5,
    )

    binge_frequency = st.number_input(
        "Binge-drinking frequency",
        min_value=0.0,
        value=0.0,
        step=1.0,
    )

    binge_episodes = st.number_input(
        "Binge-drinking episodes in past 30 days",
        min_value=0,
        max_value=30,
        value=0,
        step=1,
    )

with column_3:
    average_sleep = st.number_input(
        "Average sleep duration (hours)",
        min_value=2.0,
        max_value=14.0,
        value=8.0,
        step=0.5,
    )

    weekend_sleep_difference = st.number_input(
        "Weekend sleep difference (hours)",
        min_value=-9.0,
        max_value=10.0,
        value=0.0,
        step=0.5,
    )

    active_domain_count = st.number_input(
        "Number of active physical-activity domains",
        min_value=0,
        max_value=5,
        value=0,
        step=1,
    )
# ---------------------------------------------------------
# CATEGORICAL FEATURES — DEMOGRAPHIC AND SLEEP
# ---------------------------------------------------------

st.subheader("Demographic and Sleep Information")

def coded_selectbox(label, choices, key):
    selected_label = st.selectbox(label, list(choices.keys()), key=key)
    return choices[selected_label]


demographic_column, sleep_column = st.columns(2)

with demographic_column:
    gender = coded_selectbox(
        "Gender",
        {"Male": 1, "Female": 2},
        "gender",
    )

    race_ethnicity = coded_selectbox(
        "Race and ethnicity",
        {
            "Mexican American": 1,
            "Other Hispanic": 2,
            "Non-Hispanic White": 3,
            "Non-Hispanic Black": 4,
            "Non-Hispanic Asian": 6,
            "Other or multiracial": 7,
        },
        "race_ethnicity",
    )

    education_level = coded_selectbox(
        "Education level",
        {
            "Under age 20": 0,
            "Less than 9th grade": 1,
            "9th–11th grade": 2,
            "High school graduate or GED": 3,
            "Some college or associate degree": 4,
            "College graduate or higher": 5,
        },
        "education_level",
    )

    marital_status = coded_selectbox(
        "Marital status",
        {
            "Under age 20": 0,
            "Married": 1,
            "Widowed": 2,
            "Divorced": 3,
            "Separated": 4,
            "Never married": 5,
            "Living with partner": 6,
        },
        "marital_status",
    )

with sleep_column:
    snoring_frequency = coded_selectbox(
        "How often do you snore?",
        {
            "Never": 0,
            "Rarely — 1–2 nights per week": 1,
            "Occasionally — 3–4 nights per week": 2,
            "Frequently — 5 or more nights per week": 3,
        },
        "snoring_frequency",
    )

    breathing_frequency = coded_selectbox(
        "How often do you snort or stop breathing during sleep?",
        {
            "Never": 0,
            "Rarely — 1–2 nights per week": 1,
            "Occasionally — 3–4 nights per week": 2,
            "Frequently — 5 or more nights per week": 3,
        },
        "breathing_frequency",
    )

    trouble_sleeping = coded_selectbox(
        "Have you told a doctor about trouble sleeping?",
        {"Yes": 1, "No": 2},
        "trouble_sleeping",
    )

    daytime_sleepiness = coded_selectbox(
        "How often do you feel overly sleepy during the day?",
        {
            "Never": 0,
            "Rarely — once per month": 1,
            "Sometimes — 2–4 times per month": 2,
            "Often — 5–15 times per month": 3,
            "Almost always — 16–30 times per month": 4,
        },
        "daytime_sleepiness",
    )
# ---------------------------------------------------------
# CATEGORICAL FEATURES — LIFESTYLE AND ACTIVITY
# ---------------------------------------------------------

st.subheader("Lifestyle and Physical Activity")

lifestyle_column, activity_column = st.columns(2)

with lifestyle_column:
    smoking_status = coded_selectbox(
        "Smoking status",
        {
            "Never smoker": 0,
            "Former smoker": 1,
            "Some-day smoker": 2,
            "Daily smoker": 3,
        },
        "smoking_status",
    )

    alcohol_use_status = coded_selectbox(
        "Alcohol-use status",
        {
            "Missing or unknown": 0,
            "Lifetime abstainer": 1,
            "No alcohol use in past year": 2,
            "Current alcohol user": 3,
        },
        "alcohol_use_status",
    )

    heavy_daily_drinking = coded_selectbox(
        "Ever drank heavily almost every day?",
        {"No": 0, "Yes": 1},
        "heavy_daily_drinking",
    )

with activity_column:
    vigorous_work = coded_selectbox(
        "Vigorous-intensity work activity",
        {"Yes": 1, "No": 2},
        "vigorous_work",
    )

    moderate_work = coded_selectbox(
        "Moderate-intensity work activity",
        {"Yes": 1, "No": 2},
        "moderate_work",
    )

    walking_bicycling = coded_selectbox(
        "Walking or bicycling for transportation",
        {"Yes": 1, "No": 2},
        "walking_bicycling",
    )

    vigorous_recreation = coded_selectbox(
        "Vigorous recreational activity",
        {"Yes": 1, "No": 2},
        "vigorous_recreation",
    )

    moderate_recreation = coded_selectbox(
        "Moderate recreational activity",
        {"Yes": 1, "No": 2},
        "moderate_recreation",
    )
# ---------------------------------------------------------
# CREATE MODEL INPUT AND GENERATE PREDICTION
# ---------------------------------------------------------

st.divider()

if st.button("Evaluate Depression Risk", type="primary"):
    input_values = {
        "RIDAGEYR": age,
        "INDFMPIR": income_ratio,
        "sedentary_minutes_per_day": sedentary_minutes,
        "cigarettes_per_day": cigarettes_per_day,
        "alcohol_frequency": alcohol_frequency,
        "average_drinks_per_day": average_drinks,
        "binge_drinking_frequency": binge_frequency,
        "binge_drinking_episodes_30d": binge_episodes,
        "average_sleep_hours": average_sleep,
        "weekend_sleep_difference": weekend_sleep_difference,
        "active_domain_count": active_domain_count,
        "RIAGENDR": gender,
        "RIDRETH3": race_ethnicity,
        "DMDEDUC2": education_level,
        "DMDMARTL": marital_status,
        "SLQ030": snoring_frequency,
        "SLQ040": breathing_frequency,
        "SLQ050": trouble_sleeping,
        "SLQ120": daytime_sleepiness,
        "smoking_status": smoking_status,
        "alcohol_use_status": alcohol_use_status,
        "ever_heavy_daily_drinking": heavy_daily_drinking,
        "vigorous_work_activity": vigorous_work,
        "moderate_work_activity": moderate_work,
        "walking_or_bicycling": walking_bicycling,
        "vigorous_recreation": vigorous_recreation,
        "moderate_recreation": moderate_recreation,
    }

    # Confirm that every predictor required by the model is available.
    missing_predictors = [
        predictor
        for predictor in selected_predictors
        if predictor not in input_values
    ]

    if missing_predictors:
        st.error(
            "Missing required predictors: "
            + ", ".join(missing_predictors)
        )
    else:
        # Preserve the exact predictor order used during model training.
        participant_data = pd.DataFrame(
            [[input_values[column] for column in selected_predictors]],
            columns=selected_predictors,
        )

        try:
            pipeline = model_bundle["pipeline"]
            positive_class = model_bundle.get("positive_class", 1)
            positive_class_index = list(pipeline.classes_).index(
                positive_class
            )

            risk_score = float(
                pipeline.predict_proba(participant_data)[
                    0, positive_class_index
                ]
            )

            elevated_risk = risk_score >= prediction_threshold

            st.subheader("Screening Result")
            st.metric(
                label="Model risk score",
                value=f"{risk_score:.1%}",
            )

            if elevated_risk:
                st.warning(
                    "The result is above the selected screening "
                    f"threshold of {prediction_threshold:.1%}. "
                    "The participant is classified as having elevated risk."
                )
            else:
                st.success(
                    "The result is below the selected screening "
                    f"threshold of {prediction_threshold:.1%}."
                )

            st.caption(
                "This result is produced by a statistical model and "
                "must not be interpreted as a clinical diagnosis."
            )

        except Exception as prediction_error:
            st.error("The model could not generate a prediction.")
            st.exception(prediction_error)