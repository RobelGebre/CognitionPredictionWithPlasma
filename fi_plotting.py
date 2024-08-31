import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import shap_custom
import support


def run_feat_importance(predictor, nsh, data, train_data, model_features, column, _title, _save_path):
    _save_path = os.path.join(_save_path, 'FeatureImportances')
    os.makedirs(_save_path, exist_ok=True)

    def save_and_plot(data_subset, suffix):
        save_path = os.path.join(_save_path, 'Permutation')
        os.makedirs(save_path, exist_ok=True)
        file_name = f"{suffix}_{_title}.csv"
        data_subset.to_csv(os.path.join(save_path, file_name))
        
        feat_imp = predictor.feature_importance(data_subset[model_features + [column]], num_shuffle_sets=nsh)
        support.plot_feat_importance(feat_imp, f"{_title}_{suffix}", save_path)

        shap_custom.shap_calculations(
            model_features,
            data_subset[model_features + [column]],
            train_data[model_features + [column]],
            predictor,
            os.path.join(_save_path, 'SHAP'),
            f"{_title}_{suffix}"
        )

        save_and_plot(data, 'filtered')

def plot_feat_importance(featImp, _title, _save_path):
    featImp['importance'] = pd.to_numeric(featImp['importance'], errors='coerce')

    plt.figure(figsize=(15, 20))
    sns.set(font_scale=2)
    sns.barplot(
        x=featImp['importance'],
        y=featImp['feature'],
        data=featImp,
        capsize=20,
        linewidth=2,
        orient='h'
    )
    plt.xlabel('Feature Importance')
    plt.ylabel('Inputs')
    plt.gcf().subplots_adjust(left=0.38)
    plt.savefig(os.path.join(_save_path, f"{_title}_.png"), dpi=300)
    plt.close()