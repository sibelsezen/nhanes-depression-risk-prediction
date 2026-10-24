# NHANES Depression Risk Prediction

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-AWS%20EC2-FF4B4B.svg)](http://3.144.190.201:8501)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end machine-learning project that estimates elevated depressive-symptom risk using health, demographic, sleep, physical-activity, smoking, and alcohol-use data from the CDC National Health and Nutrition Examination Survey (NHANES) 2017–2018 cycle.

**Live AWS application:** [NHANES Depression Risk Screening Tool](http://3.144.190.201:8501)

> **Important:** This project is an educational screening demonstration. It is not a diagnostic or clinical decision-making tool.

## Project Goal

The goal is to compare multiple classification algorithms fairly and select a model that can identify participants with elevated depressive-symptom risk. Because only 9.06% of participants belong to the positive class, the project prioritizes **recall** and uses **ROC-AUC** as a secondary discrimination metric. Ordinary accuracy is not used for model selection.

## Dataset

- **Source:** CDC/NCHS NHANES
- **Cycle:** 2017–2018
- **Analytic sample:** 5,068 participants
- **Predictors:** 29
- **Elevated-risk cases:** 459 (9.06%)
- **Target:** Binary elevated depressive-symptom risk derived from the NHANES depression screener
- **Split:** Stratified 80% training / 20% untouched test

Participant identifiers are excluded from the predictor matrix. The workflow also checks duplicate IDs, missing target values, predictor leakage, and train/test alignment.

## Models Compared

Ten real classification models were developed and compared:

1. Logistic Regression
2. Support Vector Machine
3. Decision Tree
4. Random Forest
5. AdaBoost
6. Gradient Boosting
7. XGBoost
8. CatBoost
9. LightGBM
10. Histogram Gradient Boosting

A Dummy Classifier is not included in the final computation table.

## Modeling Strategy

All model development is performed on the training set. The test set remains untouched until the final model is selected.

- Numeric variables are imputed and scaled; scaling is especially important for Logistic Regression and SVM.
- Categorical variables are imputed and one-hot encoded.
- Preprocessing is fitted inside each cross-validation fold to prevent leakage.
- Stratified five-fold cross-validation preserves the minority-class proportion.
- Class weights are used as the main imbalance treatment.
- SMOTE is evaluated as a training-only alternative inside an imbalanced-learn pipeline.
- Hyperparameters are tuned using cross-validation.
- Regularized versions of complex models are evaluated to reduce overfitting.
- Feature importance and feature-selection experiments are examined, while the final survey retains all 29 predictors because reducing the feature set did not materially improve performance.
- Each model receives its own threshold selected from out-of-fold training probabilities by maximizing F2, which weights recall more heavily than precision.
- Models are compared only after initial, tuned, imbalance-handled, and overfitting checks are complete.

## Final Model Selection

The selected model is **tuned Logistic Regression with balanced class weights**.

| Selection result | Value |
|---|---:|
| Out-of-fold recall | 0.8038 |
| Out-of-fold ROC-AUC | 0.8077 |
| Out-of-fold specificity | 0.6922 |
| Out-of-fold balanced accuracy | 0.7480 |
| Selected probability threshold | 0.4125 |
| Overfitting assessment | No meaningful overfitting |

Logistic Regression was selected because it produced the highest recall among models that passed the project guardrails, maintained competitive ROC-AUC, showed stable validation behavior, and remained interpretable for a screening-oriented application.

## Untouched Test Performance

The final pipeline was evaluated once on 1,014 previously unseen participants.

| Metric | Result |
|---|---:|
| Recall / sensitivity | 0.7609 |
| ROC-AUC | 0.7866 |
| Specificity | 0.7017 |
| Balanced accuracy | 0.7313 |
| Precision | 0.2029 |
| F1 score | 0.3204 |
| F2 score | 0.4909 |
| PR-AUC | 0.2862 |
| Probability threshold | 0.4125 |

The test set contained 92 positive cases. The model identified 70 of them and missed 22. Its lower precision is expected in a low-prevalence screening setting where the selected threshold deliberately favors sensitivity.

## Repository Structure

```text
nhanes-depression-risk-prediction/
├── app.py
├── requirements.txt
├── requirements-modeling.txt
├── MODEL_CARD.md
├── README.md
├── LICENSE
├── data/
│   ├── raw/
│   └── processed/
├── deployment/
│   ├── README.md
│   └── nhanes-streamlit.service
├── models/
│   ├── final_depression_risk_model.joblib
│   └── final_model_metadata.json
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_exploratory_data_analysis.ipynb
│   ├── 04_model_development.ipynb
│   └── 04_model_development_revised.ipynb
└── reports/
    └── figures/
```

Large or generated datasets are not stored in the repository. Run the first three notebooks in numerical order, then use `04_model_development_revised.ipynb` as the authoritative final modeling notebook. The original `04_model_development.ipynb` is retained only for development history.

## Run Locally

```bash
git clone https://github.com/sibelsezen/nhanes-depression-risk-prediction.git
cd nhanes-depression-risk-prediction
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

The application loads the complete saved pipeline, gathers all 29 inputs, calculates the positive-class probability, and applies the saved threshold of 0.4125.

To reproduce the full modeling notebook, install the additional pinned packages:

```bash
python3 -m pip install -r requirements-modeling.txt
```

See [MODEL_CARD.md](MODEL_CARD.md) for intended use, evaluation details, and limitations.

## AWS Deployment

The application is deployed on an Ubuntu AWS EC2 instance and managed by `systemd`, which starts Streamlit automatically and restarts it after a failure or server reboot.

See [deployment/README.md](deployment/README.md) for the complete reproducible deployment procedure and [deployment/nhanes-streamlit.service](deployment/nhanes-streamlit.service) for the service definition.

The current endpoint uses an EC2 public IP. For a permanent production URL, use an Elastic IP or domain name and place Nginx with HTTPS in front of Streamlit.

## Key Technical Decisions

- Recall is the primary selection metric because false negatives are costly in screening.
- ROC-AUC is the secondary metric because it evaluates ranking across all possible thresholds.
- PR-AUC is retained as a diagnostic metric because the positive class is uncommon.
- Ordinary accuracy is excluded from model selection.
- Thresholds are learned from out-of-fold training probabilities, never from the test set.
- The final test set is evaluated only once.
- The complete fitted preprocessing-and-model pipeline is saved so deployment uses the same transformations as training.

## Limitations

- NHANES data are cross-sectional; the model identifies associations, not causes.
- The model has internal evaluation on one NHANES cycle but no external or clinical validation.
- Survey responses may contain recall or reporting bias.
- Low positive-class prevalence limits precision.
- Results may not generalize to other populations or data-collection settings.
- The AWS endpoint is a demonstration deployment and may change unless an Elastic IP is assigned.
- A risk score is not a diagnosis and must not replace professional assessment.

## Technology Stack

Python, pandas, NumPy, scikit-learn, imbalanced-learn, XGBoost, CatBoost, LightGBM, Jupyter Notebook, joblib, Streamlit, Git/GitHub, and AWS EC2.

## Author

**Sibel Sezen**  
M.S. Data Science

## License

This project is licensed under the [MIT License](LICENSE).
