# NHANES Depression Risk Prediction

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B.svg)](https://nhanes-depression-risk.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end machine-learning project that estimates elevated depressive-symptom risk using publicly available health, demographic, sleep, physical-activity, smoking, and alcohol-use data from the CDC National Health and Nutrition Examination Survey (NHANES) 2017–2018 cycle.

**Live application:** [NHANES Depression Risk Screening Tool](https://nhanes-depression-risk.streamlit.app)

> **Important:** This project is for educational and screening-demonstration purposes only. It is not a medical diagnosis tool and must not be used to make clinical decisions.

## Project Overview

Depression is an important public-health concern, but elevated risk can be difficult to identify from any single health or lifestyle measure. This project builds a reproducible binary-classification workflow that:

1. Collects and combines relevant NHANES 2017–2018 files.
2. Cleans the data and engineers analysis-ready health and lifestyle features.
3. Explores class balance, distributions, group differences, and correlations.
4. Compares multiple classification models using stratified cross-validation.
5. Tunes the strongest candidate without using the test set.
6. Selects a screening threshold from out-of-fold predictions.
7. Evaluates the final model once on an untouched holdout test set.
8. Deploys the saved pipeline as an interactive Streamlit application.

## Dataset

- **Source:** CDC/NCHS National Health and Nutrition Examination Survey (NHANES)
- **Cycle:** 2017–2018
- **Final analytic sample:** 5,068 participants
- **Candidate predictors:** 29
- **Final model predictors:** 27
- **Elevated-risk cases:** 459 (9.06%)
- **Target:** Binary elevated depressive-symptom risk derived from the NHANES depression screener

The project keeps participant identifiers separate from the modeling matrix and checks for duplicate IDs, missing target values, predictor leakage, and train/test alignment.

## Modeling Approach

The positive class represents only 9.06% of the analytic sample, so accuracy alone would be misleading. Model selection therefore emphasized:

- Balanced accuracy
- Recall/sensitivity
- Precision
- F1 score
- ROC-AUC
- PR-AUC

The evaluated candidates included:

- Dummy classifier
- Class-balanced logistic regression
- Random forest
- Histogram gradient boosting

The final model is a **class-balanced logistic regression** with L2 regularization and `C = 0.5`. BMI and waist circumference were removed from the final feature set after correlated-feature analysis. The probability threshold was selected from out-of-fold training predictions; the holdout test set was not used during tuning or threshold selection.

## Final Test Performance

The final pipeline was evaluated once on the untouched test set of 1,014 participants.

| Metric | Result |
|---|---:|
| Accuracy | 0.7367 |
| Balanced accuracy | 0.7084 |
| Precision | 0.2074 |
| Recall / sensitivity | 0.6739 |
| Specificity | 0.7430 |
| F1 score | 0.3171 |
| ROC-AUC | 0.7923 |
| PR-AUC | 0.2888 |
| Selected probability threshold | 0.4704 |

At the selected screening threshold, the model identified approximately 67% of elevated-risk cases in the holdout sample. Precision is lower because the positive class is uncommon and the threshold intentionally favors sensitivity for screening-oriented use.

## Repository Structure

```text
nhanes-depression-risk-prediction/
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── models/
│   ├── final_depression_risk_model.joblib
│   └── final_model_metadata.json
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_exploratory_data_analysis.ipynb
│   └── 04_model_development.ipynb
└── reports/
    └── figures/
```

Large or generated datasets are not stored in the repository. Run the notebooks in numerical order to reproduce them locally.

## Workflow

### 1. Data collection

`01_data_collection.ipynb` downloads or reads the required NHANES component files, selects the variables used in the project, and merges participant-level records.

### 2. Data preprocessing

`02_data_preprocessing.ipynb` cleans and recodes the variables, creates derived lifestyle features and the binary target, validates the final dataset, and saves analysis-ready outputs.

### 3. Exploratory data analysis

`03_exploratory_data_analysis.ipynb` examines target prevalence, continuous-feature distributions, group differences, categorical elevated-risk rates, correlations, and multicollinearity.

### 4. Model development

`04_model_development.ipynb` performs the stratified split, constructs leakage-safe preprocessing pipelines, compares models, tunes hyperparameters, evaluates correlated-feature configurations, selects a probability threshold from out-of-fold predictions, evaluates the untouched test set, and saves the final artifacts.

### 5. Deployment

`app.py` loads the saved pipeline and metadata, collects the 27 required inputs, produces a probability score, and compares it with the selected screening threshold.

## Run the Application Locally

Clone the repository:

```bash
git clone https://github.com/sibelsezen/nhanes-depression-risk-prediction.git
cd nhanes-depression-risk-prediction
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies and start Streamlit:

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

## Key Technical Decisions

- Used a stratified 80/20 train–test split to preserve class proportions.
- Kept all preprocessing inside scikit-learn pipelines to reduce leakage risk.
- Used five-fold stratified cross-validation on the training data.
- Applied class balancing to address the minority positive class.
- Used out-of-fold probabilities for threshold selection.
- Reserved the test set for a single final evaluation.
- Saved the complete fitted pipeline so deployment uses the same transformations as training.

## Limitations

- The data are cross-sectional, so the model identifies associations rather than causal relationships.
- The model was internally evaluated on one NHANES cycle and has not been externally or clinically validated.
- Self-reported survey responses may contain recall or reporting bias.
- The positive class is relatively uncommon, which limits precision.
- Performance may not generalize to other populations, time periods, or data-collection settings.
- The output is a statistical screening score, not a diagnosis or treatment recommendation.

## Ethical Use

Mental-health predictions require careful interpretation. This application should not replace the PHQ-9, a clinician, or emergency and crisis services. A high score should not be treated as proof of depression, and a low score does not rule it out. The model should be viewed only as an educational demonstration of an end-to-end data-science workflow.

## Technology Stack

- Python
- pandas and NumPy
- scikit-learn
- Matplotlib and Seaborn
- Jupyter Notebook
- joblib
- Streamlit
- Git and GitHub

## Author

**Sibel Sezen**  
M.S. Data Science

## License

This project is licensed under the [MIT License](LICENSE).
