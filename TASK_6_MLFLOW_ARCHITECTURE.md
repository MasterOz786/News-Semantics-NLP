# Task 6 — MLflow Experiment Tracking Architecture

## Purpose

This document defines the MLflow experiment hierarchy **before implementation** for Task 6. It covers the required run groups:

- Preprocessing Ablation
- Feature Comparison
- Model Comparison

It also shows how best models are registered and automatically promoted from **Staging** to **Production**.

---

## Architecture Diagram

```mermaid
flowchart TD
    A[Dataset Sources\nCOVID-19 Fake News\nFakeNewsNet\nISOT\nLIAR] --> B[Dataset Loader\nload_all_datasets / build_text_pipeline_dataset]
    B --> C[Shared Preprocessing Layer\nclean_text\ntokenize\nstopword handling\nnormalization\nmin token filtering]

    C --> D1[MLflow Experiment\nAssignment2/Preprocessing Ablation]
    C --> D2[MLflow Experiment\nAssignment2/Feature Comparison]
    C --> D3[MLflow Experiment\nAssignment2/Model Comparison]

    D1 --> E1[Parent Run\nPreprocessing Ablation Group]
    E1 --> F1[Child Run 1\nBaseline]
    E1 --> F2[Child Run 2\nDefault stopwords]
    E1 --> F3[Child Run 3\nCustom stopwords]
    E1 --> F4[Child Run 4\nPorter stemming]
    E1 --> F5[Child Run 5\nLemmatization]
    E1 --> F6[Child Run 6\nMax features / min token length variation]

    D2 --> E2[Parent Run\nFeature Comparison Group]
    E2 --> G1[Child Run\nBoW + LR]
    E2 --> G2[Child Run\nTF-IDF + LR]
    E2 --> G3[Child Run\nTF-IDF Sublinear + LR]

    D3 --> E3[Parent Run\nModel Comparison Group]
    E3 --> H1[Child Run\nNaive Bayes Family]
    E3 --> H2[Child Run\nLogistic Regression Family]
    E3 --> H3[Child Run\nPolynomial LR Family]

    F1 --> I[Metrics + Artifacts Logger]
    F2 --> I
    F3 --> I
    F4 --> I
    F5 --> I
    F6 --> I
    G1 --> I
    G2 --> I
    G3 --> I
    H1 --> I
    H2 --> I
    H3 --> I

    I --> J[MLflow Tracking Store\nRuns / Params / Metrics / Artifacts]
    J --> K[Best Model Selector\nBest per algorithm family]
    K --> L[MLflow Model Registry\nNaiveBayes / LogisticRegression / PolynomialLR]
    L --> M[Promotion Logic\nPromote to Production only if\nF1-weighted >= current Production + 0.01]
    M --> N[Registry Stages\nNone -> Staging -> Production]
```

---

## Experiment Hierarchy

### 1. Preprocessing Ablation

- **Experiment name:** `Assignment2/Preprocessing Ablation`
- **Parent run:** `preprocessing_ablation_group`
- **Nested child runs:** 6 required configurations

Each child run varies at least one of:
- stopword list
- stemming / lemmatization
- min token length
- TF-IDF `max_features`

### 2. Feature Comparison

- **Experiment name:** `Assignment2/Feature Comparison`
- **Parent run:** `feature_comparison_group`
- **Nested child runs:** one per feature representation

Suggested child runs:
- BoW + Logistic Regression
- TF-IDF + Logistic Regression
- TF-IDF with sublinear TF + Logistic Regression

### 3. Model Comparison

- **Experiment name:** `Assignment2/Model Comparison`
- **Parent run:** `model_comparison_group`
- **Nested child runs:** one per algorithm family

Suggested child runs:
- Naive Bayes family
- Logistic Regression family
- Polynomial Logistic Regression family

---

## Required Logging Contract

Every run logs the following **parameters**:

- dataset sources
- train size
- test size
- tokenizer
- stopword list
- normalization method
- vectorizer settings
- model type

Every run logs the following **metrics**:

- accuracy
- per-class precision
- per-class recall
- per-class F1
- weighted F1
- ROC-AUC
- training time

Every run logs the following **artifacts**:

- confusion matrix image
- ROC curve image
- TF-IDF vocabulary file
- classification report text file

---

## Model Registry Policy

The best-performing model from each algorithm family is registered under a distinct registry name:

- `FakeNewsNaiveBayes`
- `FakeNewsLogisticRegression`
- `FakeNewsPolynomialLR`

---

## Automated Promotion Policy

Promotion from **Staging** to **Production** is allowed only if:

- candidate `weighted_f1 >= current_production_weighted_f1 + 0.01`

Interpretation:
- a candidate must beat the current Production model by **at least 1 percentage point** in weighted F1.
- if no Production model exists yet, the candidate can be promoted directly.

---

## Outputs Produced by This Architecture

- MLflow experiments with nested run groups
- preprocessing ablation comparison table
- parallel coordinates plot for preprocessing ablation
- registered best models for each family
- reproducible version history for API consumption in Task 7
