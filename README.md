# ML Analysis of Differentially Expressed Genes (DEGs)

A machine learning pipeline for identifying robust biomarker genes from Differentially Expressed Gene (DEG) analysis. The workflow integrates multiple feature selection methods, consensus voting, and model validation to identify genes that consistently discriminate between disease and control samples.

---

## Overview

This project performs an end-to-end biomarker discovery workflow using gene expression data. Starting from raw count matrices and a list of significant DEGs, it:

- Cleans and filters candidate genes
- Extracts expression profiles
- Applies multiple machine learning feature selection methods
- Generates a consensus biomarker panel
- Validates the panel using Leave-One-Out Cross Validation (LOOCV)
- Interprets model predictions using SHAP values
- Exports all results for downstream biological analysis

---

## Features

- DEG preprocessing and quality filtering
- Expression matrix generation
- Multiple ML-based feature selection methods:
  - LASSO (Logistic Regression with L1 Regularization)
  - Boruta
  - SVM-RFE
  - XGBoost Feature Importance
  - Mutual Information
- Consensus biomarker selection through voting
- Leave-One-Out Cross Validation (LOOCV)
- ROC Curve generation
- Confusion Matrix
- Classification Report
- Gene expression visualization
- Clustered heatmaps
- SHAP explainability
- Export of selected genes and evaluation metrics

---

## Pipeline

```text
Raw Count Matrix
        │
        ▼
Differentially Expressed Genes
        │
        ▼
Data Cleaning
(Remove LOC genes, pseudogenes, miRNAs, missing IDs)
        │
        ▼
Extract DEG Expression Matrix
        │
        ▼
Feature Selection
 ├── LASSO
 ├── Boruta
 ├── SVM-RFE
 ├── XGBoost
 └── Mutual Information
        │
        ▼
Consensus Gene Panel
        │
        ▼
Random Forest Validation
(LOOCV)
        │
        ├── ROC Curve
        ├── Confusion Matrix
        ├── Classification Report
        ├── Heatmap
        ├── Boxplots
        └── SHAP Interpretation
```

---

## Required Input Files

| File | Description |
|------|-------------|
| `PD-COUNT-FILE.xlsx` | Raw gene count matrix |
| `DEG_List.xlsx` | Differentially expressed genes with Entrez IDs, Symbols and Gene Names |
| `Metadata.xlsx` | Sample labels (Disease / Control) |

---

## Machine Learning Methods

### LASSO

Identifies sparse biomarker signatures using L1-regularized logistic regression.

### Boruta

Uses Random Forest importance scores to identify biologically relevant genes.

### SVM-RFE

Recursively removes the least informative genes using a Support Vector Machine.

### XGBoost

Ranks genes according to gradient boosted decision tree importance.

### Mutual Information

Measures nonlinear relationships between gene expression and disease labels.

---

## Consensus Biomarker Selection

Instead of relying on a single algorithm, this project combines predictions from five independent feature selection methods.

Genes selected by multiple algorithms are considered more reliable biomarkers and are used for model validation.

---

## Validation

The final biomarker panel is evaluated using:

- Leave-One-Out Cross Validation (LOOCV)
- Random Forest Classifier
- ROC Curve & AUC
- Confusion Matrix
- Precision
- Recall
- F1 Score

LOOCV is particularly suitable for small transcriptomic datasets because every sample is used once for testing.

---

## Visualizations

The notebook automatically generates:

- ROC Curve
- Confusion Matrix
- Gene expression boxplots
- Clustered heatmap
- SHAP summary plots

These visualizations aid both predictive evaluation and biological interpretation.

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/ML-Analysis-of-DEGs.git
cd ML-Analysis-of-DEGs
```

Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost Boruta shap openpyxl
```

---

## Dependencies

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- xgboost
- Boruta
- shap
- openpyxl



## Author

**Anoop Nair**

B.Tech Computer Science (Bioinformatics)  
Vellore Institute of Technology
