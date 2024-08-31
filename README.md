# Can integration of Alzheimer's plasma biomarkers with MRI, cardiovascular, genetics, and lifestyle measures improve cognition prediction?

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#installation)
- [Usage](#usage)
  - [1. Configuration](#1-configuration)
  - [2. Running the Pipeline](#2-running-the-pipeline)
- [Outputs](#outputs)
- [References](#references)

## Introduction

This repository contains the mlearning pipeline developed for comparing the perforamnce of plasma biomarkers to predict long-term cognitive outcomes comapred to other biomakrers and risk factors. The pipeline was utilized in [*Accepted*](link_to_paper) to analyze and interpret complex datasets effectively.

The pipeline leverages **AutoGluon** for model training, **SHAP** for interpretability, and additional custom scripts for data preprocessing and visualization.

## Features

- **Data Preprocessing**: Handles missing data imputation and normalization.
- **Model Training**: Utilizes AutoGluon's `TabularPredictor` for robust model training with cross-validation.
- **Feature Importance Analysis**: Computes and visualizes feature importance using both permutation importance and SHAP values.
- **SHAP Analysis**: Generates detailed SHAP summary and dependency plots, including transition point detection.
- **Modular Structure**: Organized into separate scripts for clarity and ease of maintenance.
- **Reproducibility**: Ensures consistent results with fixed random seeds and structured workflows.

## Architecture

The pipeline consists of the following scripts:

1. **`main.py`**: Orchestrates the entire workflow, from data loading to model training and evaluation.
2. **`model_cv.py`**: Handles cross-validation, model training, and integration with AutoGluon.
3. **`support.py`**: Provides utility functions for data imputation, normalization, evaluation, and metrics saving.
4. **`fi_plotting.py`**: Manages feature importance calculations and plotting.
5. **`shap_custom.py`**: Conducts SHAP value computations and generates interpretability plots.

## Prerequisites

- **Python 3.8+**
- **pip** (Python package installer)

## Usage

### 1. Configuration

Before executing the pipeline, configure the `main.py` script with appropriate paths and parameters.

**Edit `main.py` as follows:**

```
import os
import pandas as pd
import model_cv

# Define paths and parameters
main_path = "outputs"  # Directory to save outputs
data_path = "data/data.csv"  # Path to your dataset
model_features = {
    "model_name": ["feature1", "feature2", "feature3"]  # Replace with your actual features
}
label = "target_column"  # Replace with your target column name
_impute = True  # Set whether to impute missing data

# Load data
data = pd.read_csv(data_path)

# Run the pipeline
model_cv.run_the_models(
    main_path=main_path,
    data=data,
    model_features=model_features,
    model_name="model_name",
    column=label,
    _impute=_impute
)
```

### . Running the Pipeline

To execute the pipeline, run the `main.py` script:

```
python main.py
```

This will:

* Train models with cross-validation.
* Calculate and save feature importance.
* Generate SHAP plots for model interpretation.
* Calculate transition points for further interpretaion of the features. Refer to

## Outputs

After running the pipeline, you can expect the following outputs:

* **Cross-validation Results** : Saved under `outputs/cv_*/` directories.
* **Feature Importance** : Plots and CSV files saved under `outputs/FeatureImportances/`.
* **SHAP Analysis** : SHAP summary plots, dependency plots, and CSV files in `outputs/SHAP/`.

## References

* [Paper]()
