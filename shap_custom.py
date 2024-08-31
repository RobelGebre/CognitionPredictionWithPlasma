import os
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn_extra.cluster import KMedoids
from transition_points import find_transition_point
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

class AutogluonWrapper:
    def __init__(self, predictor, feature_names):
        self.ag_model = predictor
        self.feature_names = feature_names
    
    def predict(self, X):
        if isinstance(X, pd.Series):
            X = X.values.reshape(1, -1)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        return self.ag_model.predict(X)

def execute_shap(test_data, train_data, model_features, column, predictor, _save_path, datapart):
    save_path = os.path.join(_save_path, 'SHAP')
    os.makedirs(save_path, exist_ok=True)

    sample_size = 10
    if len(test_data) >= sample_size:
        test_data = test_data.sample(sample_size, random_state=1).T.drop_duplicates().T

    if not test_data.empty:
        X_test_shap = test_data[[column] + model_features]
        X_train_shap = train_data[[column] + model_features]
        shap_calculations(model_features, X_test_shap, X_train_shap, predictor, save_path, datapart)

def determine_num_clusters(data_size):
    """Calculates the number of clusters based on the data size."""
    min_clusters = 5
    max_clusters = 100
    cluster_percentage = 0.1

    ideal_clusters = int(data_size * cluster_percentage)
    return max(min_clusters, min(ideal_clusters, max_clusters))

def shap_calculations(feature_names, X_test, X_train, predictor, save_path, datapart):
    _save_path = os.path.join(save_path, datapart)
    os.makedirs(_save_path, exist_ok=True)

    X_test.to_csv(os.path.join(_save_path, datapart + '_shap.csv'))
    ag_wrapper = AutogluonWrapper(predictor, feature_names)

    df_SHAP = X_train[feature_names].values
    num_clusters = determine_num_clusters(df_SHAP.shape[0])

    kmedoids = KMedoids(n_clusters=num_clusters, method='pam', init='heuristic', random_state=0).fit(df_SHAP)
    X_train_summary = kmedoids.cluster_centers_

    explainer = shap.KernelExplainer(ag_wrapper.predict, X_train_summary)
    
    print('<>'*10, f'Running SHAP AI explainer on the category: {datapart}', '<>'*10)
    shap_values_test = explainer.shap_values(X_test[feature_names].astype(np.float64))
    df_shap_values = pd.DataFrame(shap_values_test, columns=feature_names)
    df_shap_values.insert(0, 'base value', explainer.expected_value)
    df_shap_values.to_csv(os.path.join(_save_path, datapart + 'SHAPValues.csv'))

    mean_abs_shap_values = np.abs(shap_values_test).mean(axis=0)
    sorted_indices = np.argsort(-mean_abs_shap_values)
    sorted_shap_values = shap_values_test[:, sorted_indices]
    sorted_features = np.array(feature_names)[sorted_indices]

    modified_feature_names = [f"{a} [{str(b)}]" for a, b in zip(sorted_features, np.abs(X_test[sorted_features].values).mean(0).round(2))]

    shap.summary_plot(
        sorted_shap_values,
        X_test[sorted_features].reset_index(drop=True),
        feature_names=modified_feature_names,
        show=False,
        plot_type='dot',
        max_display=20
    )
    plt.gcf().subplots_adjust(left=0.42, bottom=0.1)
    plt.grid(visible=True)
    plt.tight_layout()
    plt.savefig(os.path.join(_save_path, 'SHAPSummary_BeeSwarmPlot.png'), dpi=300)
    plt.close()

    plot_dependency_plots(shap_values_test, X_test, feature_names, _save_path)

def plot_dependency_plots(shap_values_test, test_X, features, save_path):
    shap_values_test *= 100
    _dir = 'Dependency Plots'
    os.makedirs(os.path.join(save_path, _dir), exist_ok=True)

    sns.set_theme(style="whitegrid")
    sns.set_context("notebook", font_scale=2.8)
    
    for i in range(len(features)):
        predictor_variable = features[i]
        feature_dir = predictor_variable
        os.makedirs(os.path.join(save_path, _dir, feature_dir), exist_ok=True)

        print(f'Dependency Plots for: {predictor_variable}')
        inds = shap.approximate_interactions(predictor_variable, shap_values_test, test_X)

        idx = np.where(test_X.columns == predictor_variable)[0][0]
        data = pd.DataFrame({'x': test_X.iloc[:, idx], 'y_sv': shap_values_test[:, idx]})

        x_filtered = data['x']
        y_sv_filtered = data['y_sv'] 
        
        if len(np.unique(x_filtered)) >= 5 and len(x_filtered) > 10:
            transition_points = find_and_plot_transitions(x_filtered, y_sv_filtered, predictor_variable, save_path, feature_dir, inds[i])

def find_and_plot_transitions(x_filtered, y_sv_filtered, predictor_variable, save_path, feature_dir, interaction_index):
    slope_thresholds = [0.05]
    matching_thresholds = [0.05]
    peak_bottom_thresholds = [0.8, 0.9]
    significance_thresholds = [0.001]
    user_defined_s = [0.01, 0.03, 0.04, 0.05, 0.07, 0.09, 0.2, 0.4]
    user_defined_k = [3]

    x_smoothed, y_sv_smoothed, x_spline_smooth, y_spline_smooth, transition_point_found, transition_points = find_transition_point(
        x_filtered, y_sv_filtered, slope_thresholds, matching_thresholds, peak_bottom_thresholds, significance_thresholds, user_defined_s, user_defined_k, handle_duplicates='median')

    if transition_point_found:
        transition_y_values = [y_sv_filtered[(np.abs(x_filtered - x_point)).argmin()] for x_point in transition_points]

        _, ax = plt.subplots(figsize=(12, 10))
        shap.dependence_plot(predictor_variable, shap_values_test, test_X, show=False, interaction_index=interaction_index, dot_size=200, color=plt.get_cmap("jet"), ax=ax)
        
        line_length_x = (np.max(x_filtered) - np.min(x_filtered)) * 0.85
        line_length_y = (np.max(y_sv_smoothed) - np.min(y_sv_smoothed)) * 1.00

        ax.plot(x_spline_smooth, y_spline_smooth, color='black', linewidth=3.2)
        for x_point, y_point in zip(transition_points, transition_y_values):
            plt.hlines(y=y_point, xmin=x_point - line_length_x / 2, xmax=x_point + line_length_x / 2, colors='black', linestyle='-.', linewidth=3.0)
            plt.vlines(x=x_point, ymin=y_point - line_length_y / 2, ymax=y_point + line_length_y / 2, colors='black', linestyle='-.', linewidth=3.0)
            ax.annotate(f"${x_point:.2f}$", xy=(x_point, y_point), xytext=(x_point + line_length_x * 0.05, y_point + line_length_y * 0.1), 
                        horizontalalignment='right', color='darkgreen', path_effects=[patheffects.withStroke(linewidth=3, foreground="white")])

        ax.xaxis.label.set_size(30)
        ax.yaxis.label.set_size(30)
        ax.tick_params(axis='both', which='major', labelsize=35)
        plt.gcf().subplots_adjust(left=0.25)
        plt.savefig(os.path.join(save_path, feature_dir, f"{predictor_variable}_{interaction_index}.png"), dpi=120)
        plt.show()
