# Task 6 — MLflow Screenshots Guide and Placeholders

This file is a **submission-ready screenshot guide** for Task 6.

It includes:
- exact commands to run
- where to open MLflow
- what to capture
- suggested screenshot filenames
- ready-made Markdown image placeholders
- short caption guidance for embedding in the final report

---

## Goal

The assignment asks for:
- MLflow experiment hierarchy tracking
- required run groups
- preprocessing ablation study with 6 configurations
- logged parameters, metrics, and artifacts
- parallel coordinates plot screenshot
- registered best models in MLflow Model Registry
- automated promotion logic evidence
- **minimum 15 MLflow screenshots embedded in the report**

This document helps you produce those screenshots in a clean and organized way.

---

## Before taking screenshots

### Step 1 — Generate all MLflow runs

From the `2/` directory, run:

- `venv311/bin/python task6_mlflow.py`

This populates the MLflow tracking store with:
- preprocessing ablation runs
- feature comparison runs
- model comparison runs
- registered best models for each family
- artifacts such as confusion matrices, ROC curves, vocabularies, classification reports, and the preprocessing parallel-coordinates plot

### Step 2 — Launch the MLflow UI

From the `2/` directory, run:

- `venv311/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001`

If that command does not work, run:

- `venv311/bin/python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001`

Then open this in your browser:

- `http://127.0.0.1:5001`

### Step 3 — Save screenshots here

Store all screenshots in:

- `2/screenshots_mlflow/`

This keeps the report organized and makes embedding easier.

---

## Screenshot Checklist

Below are **15 recommended screenshots** that satisfy the assignment cleanly.

---

## Screenshot 1 — MLflow experiments overview

**What to capture:**
- the MLflow landing page showing all experiments
- the following experiment names should be visible if possible:
  - `Assignment2/Preprocessing Ablation`
  - `Assignment2/Feature Comparison`
  - `Assignment2/Model Comparison`
  - `Assignment2/Serving Bootstrap`

**Where from:**
- MLflow UI home page

**Suggested filename:**
- `screenshots_mlflow/mlflow_01_experiments_overview.png`

**Suggested caption:**
- MLflow experiments overview showing the experiment hierarchy used for Task 6.

![MLflow Screenshot 01 Placeholder](screenshots_mlflow/mlflow_01_experiments_overview.png)

---

## Screenshot 2 — Preprocessing Ablation experiment run list

**What to capture:**
- the run table for `Assignment2/Preprocessing Ablation`
- show all 6 preprocessing runs if possible

**Where from:**
- MLflow UI → `Assignment2/Preprocessing Ablation`

**Suggested filename:**
- `screenshots_mlflow/mlflow_02_preprocessing_runs.png`

**Suggested caption:**
- Preprocessing ablation experiment containing the six required configurations.

![MLflow Screenshot 02 Placeholder](screenshots_mlflow/mlflow_02_preprocessing_runs.png)

---

## Screenshot 3 — Feature Comparison experiment run list

**What to capture:**
- the run table for `Assignment2/Feature Comparison`

**Where from:**
- MLflow UI → `Assignment2/Feature Comparison`

**Suggested filename:**
- `screenshots_mlflow/mlflow_03_feature_comparison_runs.png`

**Suggested caption:**
- Feature comparison experiment showing alternative vectorization strategies.

![MLflow Screenshot 03 Placeholder](screenshots_mlflow/mlflow_03_feature_comparison_runs.png)

---

## Screenshot 4 — Model Comparison experiment run list

**What to capture:**
- the run table for `Assignment2/Model Comparison`

**Where from:**
- MLflow UI → `Assignment2/Model Comparison`

**Suggested filename:**
- `screenshots_mlflow/mlflow_04_model_comparison_runs.png`

**Suggested caption:**
- Model comparison experiment showing Naive Bayes, Logistic Regression, and Polynomial LR families.

![MLflow Screenshot 04 Placeholder](screenshots_mlflow/mlflow_04_model_comparison_runs.png)

---

## Screenshot 5 — Parallel coordinates plot

**What to capture:**
- the parallel coordinates plot for preprocessing ablation
- weighted F1 should be visible as the main comparison signal if possible

**Where from:**
- MLflow UI artifact view inside the preprocessing ablation parent run
- or the logged plot artifact from the summary artifacts

**Suggested filename:**
- `screenshots_mlflow/mlflow_05_parallel_coordinates.png`

**Suggested caption:**
- Parallel coordinates plot used to compare preprocessing configurations using weighted F1 as the primary metric.

![MLflow Screenshot 05 Placeholder](screenshots_mlflow/mlflow_05_parallel_coordinates.png)

---

## Screenshot 6 — Preprocessing run parameters

**What to capture:**
- a single preprocessing run details page
- the Parameters section should show items like:
  - dataset sources
  - train/test size
  - tokenizer
  - stopword list
  - normalization method
  - vectorizer settings
  - model type

**Where from:**
- open one child run from `Assignment2/Preprocessing Ablation`

**Suggested filename:**
- `screenshots_mlflow/mlflow_06_preprocessing_run_params.png`

**Suggested caption:**
- Example preprocessing ablation run showing logged parameters required by the assignment.

![MLflow Screenshot 06 Placeholder](screenshots_mlflow/mlflow_06_preprocessing_run_params.png)

---

## Screenshot 7 — Preprocessing run metrics

**What to capture:**
- the Metrics section from the same run
- include:
  - accuracy
  - per-class precision/recall/F1
  - weighted F1
  - ROC-AUC
  - training time

**Where from:**
- same preprocessing run details page

**Suggested filename:**
- `screenshots_mlflow/mlflow_07_preprocessing_run_metrics.png`

**Suggested caption:**
- Example preprocessing ablation run showing the required evaluation metrics.

![MLflow Screenshot 07 Placeholder](screenshots_mlflow/mlflow_07_preprocessing_run_metrics.png)

---

## Screenshot 8 — Preprocessing run artifacts

**What to capture:**
- the Artifacts section from the same run
- show these artifacts if possible:
  - confusion matrix
  - ROC curve
  - TF-IDF vocabulary
  - classification report

**Where from:**
- same preprocessing run details page → Artifacts panel

**Suggested filename:**
- `screenshots_mlflow/mlflow_08_preprocessing_run_artifacts.png`

**Suggested caption:**
- Example preprocessing ablation run showing the required logged artifacts.

![MLflow Screenshot 08 Placeholder](screenshots_mlflow/mlflow_08_preprocessing_run_artifacts.png)

---

## Screenshot 9 — Feature comparison table view

**What to capture:**
- the feature comparison experiment table with metrics visible across runs

**Where from:**
- MLflow UI → `Assignment2/Feature Comparison`

**Suggested filename:**
- `screenshots_mlflow/mlflow_09_feature_comparison_table.png`

**Suggested caption:**
- Feature comparison table used to compare performance across different vectorizer configurations.

![MLflow Screenshot 09 Placeholder](screenshots_mlflow/mlflow_09_feature_comparison_table.png)

---

## Screenshot 10 — Best feature comparison run details

**What to capture:**
- the best run from the feature comparison experiment
- include enough of the page to show both params and metrics if possible

**Where from:**
- open the best feature-comparison run in MLflow

**Suggested filename:**
- `screenshots_mlflow/mlflow_10_best_feature_run.png`

**Suggested caption:**
- Best feature comparison run used to justify the selected representation.

![MLflow Screenshot 10 Placeholder](screenshots_mlflow/mlflow_10_best_feature_run.png)

---

## Screenshot 11 — Model comparison table view

**What to capture:**
- the model comparison experiment table
- show family-level alternatives such as Naive Bayes, Logistic Regression, and Polynomial LR

**Where from:**
- MLflow UI → `Assignment2/Model Comparison`

**Suggested filename:**
- `screenshots_mlflow/mlflow_11_model_comparison_table.png`

**Suggested caption:**
- Model comparison table used to evaluate the major classifier families.

![MLflow Screenshot 11 Placeholder](screenshots_mlflow/mlflow_11_model_comparison_table.png)

---

## Screenshot 12 — Best Logistic Regression run

**What to capture:**
- the best Logistic Regression run details page
- show metrics and artifacts if possible

**Where from:**
- one selected run inside `Assignment2/Model Comparison`

**Suggested filename:**
- `screenshots_mlflow/mlflow_12_best_logistic_regression_run.png`

**Suggested caption:**
- Best Logistic Regression family run used for registration and serving.

![MLflow Screenshot 12 Placeholder](screenshots_mlflow/mlflow_12_best_logistic_regression_run.png)

---

## Screenshot 13 — Best Naive Bayes run

**What to capture:**
- the best Naive Bayes run details page

**Where from:**
- one selected Naive Bayes run inside `Assignment2/Model Comparison`

**Suggested filename:**
- `screenshots_mlflow/mlflow_13_best_naive_bayes_run.png`

**Suggested caption:**
- Best Naive Bayes family run used for model-family registration.

![MLflow Screenshot 13 Placeholder](screenshots_mlflow/mlflow_13_best_naive_bayes_run.png)

---

## Screenshot 14 — Model Registry overview

**What to capture:**
- the MLflow Model Registry page
- show registered models such as:
  - `FakeNewsNaiveBayes`
  - `FakeNewsLogisticRegression`
  - `FakeNewsPolynomialLR`

**Where from:**
- MLflow UI → Models / Model Registry

**Suggested filename:**
- `screenshots_mlflow/mlflow_14_model_registry_overview.png`

**Suggested caption:**
- MLflow Model Registry showing the best model from each algorithm family.

![MLflow Screenshot 14 Placeholder](screenshots_mlflow/mlflow_14_model_registry_overview.png)

---

## Screenshot 15 — Logistic Regression model version and stage

**What to capture:**
- the registered `FakeNewsLogisticRegression` model page
- include:
  - version
  - stage
  - tags or metrics if visible
- this screenshot should ideally show Production or Staging evidence

**Where from:**
- MLflow UI → Model Registry → `FakeNewsLogisticRegression`

**Suggested filename:**
- `screenshots_mlflow/mlflow_15_logistic_regression_model_version.png`

**Suggested caption:**
- Registered Logistic Regression model version showing stage assignment after automated promotion logic.

![MLflow Screenshot 15 Placeholder](screenshots_mlflow/mlflow_15_logistic_regression_model_version.png)

---

## Optional extra screenshots

If you want stronger evidence beyond the minimum 15, add these:

### Optional Screenshot 16 — Naive Bayes model version page
- `screenshots_mlflow/mlflow_16_nb_model_version.png`

### Optional Screenshot 17 — Polynomial LR model version page
- `screenshots_mlflow/mlflow_17_poly_model_version.png`

### Optional Screenshot 18 — Production vs Staging evidence
- `screenshots_mlflow/mlflow_18_stage_promotion_evidence.png`

### Optional Screenshot 19 — Version history table
- `screenshots_mlflow/mlflow_19_version_history.png`

### Optional Screenshot 20 — Recent run history view
- `screenshots_mlflow/mlflow_20_recent_runs.png`

---

## Embedding instructions for the final report

For each screenshot in the final report:
1. place the image below the relevant subsection
2. add a one-line caption under it
3. refer to it in the discussion text

Example style:
- Figure X shows the preprocessing ablation run table in MLflow.
- Figure Y shows the registered Logistic Regression model in the Model Registry.

---

## Quick command summary

From the `2/` directory:

### Generate MLflow runs
- `venv311/bin/python task6_mlflow.py`

### Start MLflow UI
- `venv311/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001`

### Open in browser
- `http://127.0.0.1:5001`

---

## Final checklist

Before submission, confirm that:

- all 15 screenshots are saved in `2/screenshots_mlflow/`
- the images load correctly in Markdown
- the experiment names are visible in at least one screenshot
- the preprocessing ablation includes 6 runs
- the parallel coordinates plot is included
- the model registry is shown
- the Logistic Regression model version and stage are shown

---

## Notes

If MLflow opens but does not show enough runs:
- confirm you ran `venv311/bin/python task6_mlflow.py`
- confirm you launched the UI from the `2/` directory
- confirm the backend store URI is exactly `sqlite:///mlflow.db`

If needed, rerun the experiment generation step and refresh the UI.
