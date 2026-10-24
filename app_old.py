# ============================================================
# NHANES Depression-Risk Screening Application
# Final tuned model with all 29 predictors
# ============================================================

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="NHANES Depression-Risk Screening",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_depression_risk_model.joblib"
)


@st.cache_resource
def load_model_bundle():
    """Load the finalized model bundle."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file was not found at: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


try:
    model_bundle = load_model_bundle()

    required_items = {
        "pipeline",
        "selected_predictors",
        "prediction_threshold",
    }

    missing_items = required_items.difference(
        model_bundle.keys()
    )

    if missing_items:
        raise KeyError(
            f"Missing model-bundle items: {sorted(missing_items)}"
        )

    model_pipeline = model_bundle["pipeline"]
    selected_predictors = model_bundle[
        "selected_predictors"
    ]
    prediction_threshold = float(
        model_bundle["prediction_threshold"]
    )

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

st.title("NHANES Depression Risk Screening")

st.write(
    """
    This application uses demographic, health, sleep, and
    lifestyle information to estimate elevated
    depressive-symptom risk.
    """
)

st.info(
    "This application is intended for educational screening "
    "support only. It does not provide a medical diagnosis."
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:
    st.header("Model Information")
    st.success("Final model loaded successfully")
    st.write("**Model:** Logistic Regression")
    st.write(
        f"**Input predictors:** {len(selected_predictors)}"
    )
    st.write(
        f"**Screening threshold:** "
        f"{prediction_threshold:.4f}"
    )
    st.write("**Primary selection metric:** Recall")
    st.write("**Imbalance handling:** Balanced class weights")


# ---------------------------------------------------------
# HELPER FUNCTION
# ---------------------------------------------------------

def coded_selectbox(label, choices, key):
    """Display labels while returning model-compatible codes."""

    selected_label = st.selectbox(
        label,
        options=list(choices.keys()),
        key=key,
    )

    return choices[selected_label]


# ---------------------------------------------------------
# PARTICIPANT SURVEY
# ---------------------------------------------------------

st.header("Participant Information")

st.write(
    "Complete the health and lifestyle information below."
)

with st.form("depression_risk_form"):

    # -----------------------------------------------------
    # DEMOGRAPHIC INFORMATION
    # -----------------------------------------------------

    st.subheader("Demographic Information")

    demographic_1, demographic_2, demographic_3 = (
        st.columns(3)
    )

    with demographic_1:
        age = st.number_input(
            "Age (years)",
            min_value=18,
            max_value=80,
            value=51,
            step=1,
        )

        gender = coded_selectbox(
            "Gender",
            {
                "Male": 1.0,
                "Female": 2.0,
            },
            "gender",
        )

    with demographic_2:
        race_ethnicity = coded_selectbox(
            "Race and ethnicity",
            {
                "Mexican American": 1.0,
                "Other Hispanic": 2.0,
                "Non-Hispanic White": 3.0,
                "Non-Hispanic Black": 4.0,
                "Non-Hispanic Asian": 6.0,
                "Other or multiracial": 7.0,
            },
            "race_ethnicity",
        )

        income_ratio = st.number_input(
            "Family income-to-poverty ratio",
            min_value=0.0,
            max_value=5.0,
            value=2.13,
            step=0.1,
        )

    with demographic_3:
        education_level = coded_selectbox(
            "Education level",
            {
                "Under age 20": 0.0,
                "Less than 9th grade": 1.0,
                "9th–11th grade": 2.0,
                "High school graduate or GED": 3.0,
                "Some college or associate degree": 4.0,
                "College graduate or higher": 5.0,
                "Missing or unknown": 7.0,
            },
            "education_level",
        )

        marital_status = coded_selectbox(
            "Marital status",
            {
                "Under age 20 or missing": 0.0,
                "Married": 1.0,
                "Widowed": 2.0,
                "Divorced": 3.0,
                "Separated": 4.0,
                "Never married": 5.0,
                "Living with partner": 6.0,
            },
            "marital_status",
        )

    # -----------------------------------------------------
    # BODY MEASUREMENTS
    # -----------------------------------------------------

    st.subheader("Body Measurements")

    body_1, body_2 = st.columns(2)

    with body_1:
        body_mass_index = st.number_input(
            "Body mass index (BMI)",
            min_value=14.0,
            max_value=86.2,
            value=28.5,
            step=0.1,
        )

    with body_2:
        waist_circumference = st.number_input(
            "Waist circumference (cm)",
            min_value=56.4,
            max_value=169.5,
            value=99.1,
            step=0.1,
        )

    # -----------------------------------------------------
    # SLEEP INFORMATION
    # -----------------------------------------------------

    st.subheader("Sleep Information")

    sleep_1, sleep_2, sleep_3 = st.columns(3)

    with sleep_1:
        snoring_frequency = coded_selectbox(
            "How often do you snore?",
            {
                "Never": 0.0,
                "Rarely — 1–2 nights per week": 1.0,
                "Occasionally — 3–4 nights per week": 2.0,
                "Frequently — 5 or more nights per week": 3.0,
            },
            "snoring_frequency",
        )

        breathing_frequency = coded_selectbox(
            "How often do you snort or stop breathing "
            "during sleep?",
            {
                "Never": 0.0,
                "Rarely — 1–2 nights per week": 1.0,
                "Occasionally — 3–4 nights per week": 2.0,
                "Frequently — 5 or more nights per week": 3.0,
            },
            "breathing_frequency",
        )

    with sleep_2:
        trouble_sleeping = coded_selectbox(
            "Have you told a doctor about trouble sleeping?",
            {
                "Yes": 1.0,
                "No": 2.0,
            },
            "trouble_sleeping",
        )

        daytime_sleepiness = coded_selectbox(
            "How often do you feel overly sleepy "
            "during the day?",
            {
                "Never": 0.0,
                "Rarely — once per month": 1.0,
                "Sometimes — 2–4 times per month": 2.0,
                "Often — 5–15 times per month": 3.0,
                "Almost always — 16–30 times per month": 4.0,
            },
            "daytime_sleepiness",
        )

    with sleep_3:
        average_sleep = st.number_input(
            "Average sleep duration (hours)",
            min_value=2.0,
            max_value=14.0,
            value=7.86,
            step=0.25,
        )

        weekend_sleep_difference = st.number_input(
            "Weekend sleep difference (hours)",
            min_value=-9.0,
            max_value=10.0,
            value=0.0,
            step=0.25,
        )

    # -----------------------------------------------------
    # SMOKING AND ALCOHOL INFORMATION
    # -----------------------------------------------------

    st.subheader("Smoking and Alcohol Information")

    lifestyle_1, lifestyle_2, lifestyle_3 = st.columns(3)

    with lifestyle_1:
        smoking_status = coded_selectbox(
            "Smoking status",
            {
                "Never smoker": 0.0,
                "Former smoker": 1.0,
                "Some-day smoker": 2.0,
                "Daily smoker": 3.0,
            },
            "smoking_status",
        )

        cigarettes_per_day = st.number_input(
            "Cigarettes per day",
            min_value=0.0,
            max_value=60.0,
            value=0.0,
            step=1.0,
        )

    with lifestyle_2:
        alcohol_use_status = coded_selectbox(
            "Alcohol-use status",
            {
                "Missing or unknown": 0.0,
                "Lifetime abstainer": 1.0,
                "No alcohol use in past year": 2.0,
                "Current alcohol user": 3.0,
            },
            "alcohol_use_status",
        )

        alcohol_frequency = st.number_input(
            "Alcohol-use frequency",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=1.0,
        )

        average_drinks = st.number_input(
            "Average drinks per day",
            min_value=0.0,
            max_value=15.0,
            value=1.0,
            step=0.5,
        )

    with lifestyle_3:
        heavy_daily_drinking = coded_selectbox(
            "Ever drank heavily almost every day?",
            {
                "No": 0.0,
                "Yes": 1.0,
            },
            "heavy_daily_drinking",
        )

        binge_frequency = st.number_input(
            "Binge-drinking frequency",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=1.0,
        )

        binge_episodes = st.number_input(
            "Binge-drinking episodes in the past 30 days",
            min_value=0.0,
            max_value=30.0,
            value=0.0,
            step=1.0,
        )

    # -----------------------------------------------------
    # PHYSICAL ACTIVITY
    # -----------------------------------------------------

    st.subheader("Physical Activity and Sedentary Time")

    activity_1, activity_2, activity_3 = st.columns(3)

    activity_choices = {
        "No": 0,
        "Yes": 1,
    }

    with activity_1:
        vigorous_work = coded_selectbox(
            "Vigorous-intensity work activity",
            activity_choices,
            "vigorous_work",
        )

        moderate_work = coded_selectbox(
            "Moderate-intensity work activity",
            activity_choices,
            "moderate_work",
        )

    with activity_2:
        walking_bicycling = coded_selectbox(
            "Walking or bicycling for transportation",
            activity_choices,
            "walking_bicycling",
        )

        vigorous_recreation = coded_selectbox(
            "Vigorous recreational activity",
            activity_choices,
            "vigorous_recreation",
        )

    with activity_3:
        moderate_recreation = coded_selectbox(
            "Moderate recreational activity",
            activity_choices,
            "moderate_recreation",
        )

        sedentary_minutes = st.number_input(
            "Sedentary time (minutes per day)",
            min_value=0.0,
            max_value=1320.0,
            value=300.0,
            step=10.0,
        )

        active_domain_count = st.number_input(
            "Number of active physical-activity domains",
            min_value=0.0,
            max_value=5.0,
            value=1.0,
            step=1.0,
        )

    evaluate_button = st.form_submit_button(
        "Evaluate Depression Risk",
        type="primary",
        use_container_width=True,
    )


# ---------------------------------------------------------
# CREATE MODEL INPUT AND GENERATE PREDICTION
# ---------------------------------------------------------

if evaluate_button:

    input_values = {
        "RIAGENDR": float(gender),
        "RIDAGEYR": float(age),
        "RIDRETH3": float(race_ethnicity),
        "DMDEDUC2": float(education_level),
        "DMDMARTL": float(marital_status),
        "INDFMPIR": float(income_ratio),
        "SLQ030": float(snoring_frequency),
        "SLQ040": float(breathing_frequency),
        "SLQ050": float(trouble_sleeping),
        "SLQ120": float(daytime_sleepiness),
        "BMXBMI": float(body_mass_index),
        "BMXWAIST": float(waist_circumference),
        "sedentary_minutes_per_day": float(
            sedentary_minutes
        ),
        "cigarettes_per_day": float(
            cigarettes_per_day
        ),
        "alcohol_frequency": float(alcohol_frequency),
        "average_drinks_per_day": float(average_drinks),
        "binge_drinking_frequency": float(
            binge_frequency
        ),
        "binge_drinking_episodes_30d": float(
            binge_episodes
        ),
        "smoking_status": float(smoking_status),
        "alcohol_use_status": float(alcohol_use_status),
        "ever_heavy_daily_drinking": float(
            heavy_daily_drinking
        ),
        "average_sleep_hours": float(average_sleep),
        "weekend_sleep_difference": float(
            weekend_sleep_difference
        ),
        "vigorous_work_activity": int(vigorous_work),
        "moderate_work_activity": int(moderate_work),
        "walking_or_bicycling": int(walking_bicycling),
        "vigorous_recreation": int(vigorous_recreation),
        "moderate_recreation": int(
            moderate_recreation
        ),
        "active_domain_count": float(
            active_domain_count
        ),
    }

    missing_predictors = [
        predictor
        for predictor in selected_predictors
        if predictor not in input_values
    ]

    unexpected_predictors = [
        predictor
        for predictor in input_values
        if predictor not in selected_predictors
    ]

    if missing_predictors:
        st.error(
            "Missing required predictors: "
            + ", ".join(missing_predictors)
        )

    elif unexpected_predictors:
        st.error(
            "Unexpected predictors were provided: "
            + ", ".join(unexpected_predictors)
        )

    else:
        participant_data = pd.DataFrame(
            [
                [
                    input_values[predictor]
                    for predictor in selected_predictors
                ]
            ],
            columns=selected_predictors,
        )

        try:
            positive_class = model_bundle.get(
                "positive_class",
                1,
            )

            class_labels = list(model_pipeline.classes_)

            positive_class_index = class_labels.index(
                positive_class
            )

            risk_score = float(
                model_pipeline.predict_proba(
                    participant_data
                )[0, positive_class_index]
            )

            elevated_risk = (
                risk_score >= prediction_threshold
            )

            st.divider()
            st.subheader("Screening Result")

            result_1, result_2 = st.columns(2)

            with result_1:
                st.metric(
                    label="Estimated risk score",
                    value=f"{risk_score:.1%}",
                )

            with result_2:
                st.metric(
                    label="Screening threshold",
                    value=f"{prediction_threshold:.1%}",
                )

            if elevated_risk:
                st.warning(
                    "The estimated score is above the selected "
                    "screening threshold. The participant is "
                    "classified as having elevated "
                    "depressive-symptom risk."
                )
            else:
                st.success(
                    "The estimated score is below the selected "
                    "screening threshold. The participant is not "
                    "classified as having elevated "
                    "depressive-symptom risk."
                )

            st.caption(
                "This result is generated by a statistical "
                "screening model and must not be interpreted "
                "as a diagnosis. A qualified healthcare "
                "professional should perform any clinical "
                "assessment."
            )

        except Exception as prediction_error:
            st.error(
                "The model could not generate a prediction."
            )
            st.exception(prediction_error)