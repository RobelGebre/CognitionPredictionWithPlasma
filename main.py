"""
This script sets up and executes a machine learning pipeline for training models, 
evaluating feature importance, and generating SHAP plots for model interpretation.

Workflow:
1. Configuration and Setup:
   - Define paths, variables, and ensure necessary directories exist.
   
2. Model Training and Evaluation (model_cv.py):
   - The script `model_cv.py` handles the core of the model training process.
   - Data is preprocessed, and models are trained using a cross-validation strategy.
   - If retraining is enabled, models are trained from scratch, otherwise, the best model from previous runs is used.
   - This script also handles missing data imputation if needed.
   - The trained model's performance metrics are saved for later evaluation.

3. Feature Importance Analysis (fi_plotting.py):
   - During training, `fi_plotting.py` is invoked to calculate and plot feature importance.
   - This script leverages AutoGluon's feature importance methods and SHAP values to interpret model predictions.
   - Feature importance is saved, and SHAP plots are generated for deeper insights.

4. SHAP Calculations (shap_custom.py):
   - SHAP values are calculated to explain individual model predictions.
   - This script supports dependency plots and transition point analysis to identify significant feature changes.
   - Results from SHAP analysis are saved and can be visualized for further understanding of the model’s behavior.
"""

import os

import model_cv  

main_path = "path/to/main_directory"
data = "data.csv"
model_features = {"model_name": ["feature1", "feature2", "feature3"]} 
label = "name of the traget column"
_impute = True  

os.makedirs(main_path, exist_ok=True)

model_cv.run_the_models(
    main_path=main_path,
    datasets=data,
    model_features=model_features,
    model_name="model_name",
    column=label,
    _impute=_impute
)