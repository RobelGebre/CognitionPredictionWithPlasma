import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import IterativeImputer
from sklearn import metrics, preprocessing
import shutil

random_state = 42

def impute_custom(data, vars_not_to_impute, model_features, impute_missing):
    if impute_missing:
        data_impute = data.drop(columns=vars_not_to_impute)
        missing_columns = data_impute.columns[data_impute.isnull().any()]
        
        imputer = IterativeImputer(max_iter=500, tol=1e-3, initial_strategy='median', random_state=random_state)
        data_imputed = data_impute.copy()
        data_imputed[missing_columns] = imputer.fit_transform(data_impute[missing_columns])
        data_imputed_indexed = pd.concat([data[vars_not_to_impute], data_imputed], axis=1)
        print("Imputation completed.")
    else:
        data_imputed_indexed = data
        print("No imputation applied.")
    return data_imputed_indexed

def normalize(ds, binaries_to_drop, column, skip):
    if not skip:
        ds_scaled = ds.drop(columns=binaries_to_drop + [column])
        scaler = preprocessing.MaxAbsScaler().fit(ds_scaled)
        ds_scaled = pd.DataFrame(scaler.transform(ds_scaled), columns=ds_scaled.columns, index=ds.index)
        ds_scaled = pd.concat([ds_scaled, ds[binaries_to_drop + [column]]], axis=1)
        print("Data normalized.")
    else:
        ds_scaled = ds
    return ds_scaled

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(abs(predictions) - abs(test_labels)).values
    mape = 100 * np.mean(errors / abs(test_labels))
    accuracy = 100 - mape

    print("Model Performance")
    print("Average Error: {:0.4f}".format(np.mean(errors)))
    print("Accuracy = {:0.2f}%.".format(accuracy))
    
    return accuracy

def save_metrics(predictor, test_data, train_data, model_features, save_path, column):
    y_train = train_data[column].astype(np.float64)
    y_pred_train = predictor.predict(train_data[model_features]).astype(np.float64)

    y_test = test_data[column].astype(np.float64)
    y_pred_test = predictor.predict(test_data[model_features]).astype(np.float64)

    perf_train = predictor.evaluate_predictions(y_true=y_train, y_pred=y_pred_train, auxiliary_metrics=True)
    perf_test = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred_test, auxiliary_metrics=True)

    pd.DataFrame([perf_train]).to_csv(os.path.join(save_path, 'ag_training_performance.csv'))
    pd.DataFrame([perf_test]).to_csv(os.path.join(save_path, 'ag_testing_performance.csv'))

    leaderboard = predictor.leaderboard(test_data[model_features + [column]], silent=True)
    leaderboard.to_csv(os.path.join(save_path, 'Leaderboard.csv'))

    metrics_data = {
        'mae_train': metrics.mean_absolute_error(y_train, y_pred_train),
        'mse_train': metrics.mean_squared_error(y_train, y_pred_train),
        'r2_train': metrics.r2_score(y_train, y_pred_train),
        'mae_test': metrics.mean_absolute_error(y_test, y_pred_test),
        'mse_test': metrics.mean_squared_error(y_test, y_pred_test),
        'r2_test': metrics.r2_score(y_test, y_pred_test)
    }

    pd.DataFrame([metrics_data]).to_csv(os.path.join(save_path, 'Sklearn_Performance_metrics.csv'))

    train_data['True_Label'] = y_train
    train_data['Predicted_Label'] = y_pred_train
    train_data['part'] = 'train'

    test_data['True_Label'] = y_test
    test_data['Predicted_Label'] = y_pred_test
    test_data['part'] = 'test'

    pd.concat([train_data, test_data]).to_csv(os.path.join(save_path, 'complete_data_after_training.csv'))

def select_and_save_best_model(df_r2_train, df_r2_test, df_rmse_train, df_rmse_test, main_path, datasets_names, model_name, ft, n_folds):
    merged_df = pd.merge(df_r2_train, df_r2_test, on='cvs', suffixes=('_train', '_test'))
    merged_df = pd.merge(merged_df, df_rmse_train[['cvs', 'RMSE']], on='cvs')
    merged_df = pd.merge(merged_df, df_rmse_test[['cvs', 'RMSE']], on='cvs', suffixes=('_rmse_train', '_rmse_test'))

    merged_df['R2_diff'] = abs(merged_df['R2_train'] - merged_df['R2_test'])
    merged_df = merged_df[(merged_df['R2_train'] > 0) & (merged_df['R2_test'] > 0)]

    best_model = merged_df.sort_values(by=['R2_test', 'R2_diff'], ascending=[False, True]).iloc[0]
    best_model_cvs = best_model['cvs']

    source_dir = os.path.join(main_path, datasets_names[ft], model_name, f'cv_{int(best_model_cvs)}')
    destination_dir = os.path.join(main_path, datasets_names[ft], model_name, 'best_model')

    if os.path.exists(destination_dir):
        shutil.rmtree(destination_dir)
    shutil.copytree(source_dir, destination_dir)

    for n in range(n_folds):
        shutil.rmtree(os.path.join(main_path, datasets_names[ft], model_name, f'cv_{n}'))

    print(f"The best model is based on cross-validation split: {best_model_cvs}")
    pd.DataFrame([best_model]).to_csv(os.path.join(main_path, datasets_names[ft], model_name, 'best_cv.csv'))

def process_metrics(main_path, datasets_names, ft, model_name, n_folds):
    def gather_metrics(n_folds, metric_name):
        metrics_list = []
        for run_cv in range(n_folds):
            metric_path = os.path.join(main_path, datasets_names[ft], model_name, f'cv_{run_cv}', f'ag_{metric_name}_performance.csv')
            metrics_list.append(pd.read_csv(metric_path))
        return pd.concat(metrics_list).reset_index(drop=True)

    df_r2_train = gather_metrics(n_folds, 'training')
    df_r2_test = gather_metrics(n_folds, 'testing')

    select_and_save_best_model(df_r2_train, df_r2_test, df_r2_train, df_r2_test, main_path, datasets_names, model_name, ft, n_folds)
