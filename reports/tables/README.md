# Report Tables

The authoritative final outputs are:

- `final_model_comparison.csv`: the ten selected post-tuning model configurations, ranked primarily by recall after guardrail checks.
- `finalist_model_comparison.csv`: the three highest-ranked eligible configurations.
- `final_logistic_regression_test_results.csv`: the single untouched-test evaluation of the selected Logistic Regression model.

`final_gradient_boosting_test_results.csv` is retained only as a historical experiment from an earlier model-selection stage. It does not describe the deployed model and must not be reported as the final result.
