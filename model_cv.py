import warnings
import os
import random
import pandas as pd
import numpy as np

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from autogluon.tabular import TabularPredictor

import support
import fi_plotting

warnings.filterwarnings('ignore')
random.seed(42)

def run_the_models(main_path, data, model_features, model_name, column, _impute):
    nsh = 10
    n_splits = 5
    n_repeats = 10

    df_final = data.dropna(subset=[column])
    df_final = df_final[df_final[column] != np.nan]

    df_final['target_bins'] = pd.qcut(df_final[column], q=10, labels=False, duplicates='drop')
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    for cv_n, (train_index, test_index) in enumerate(rskf.split(df_final, df_final['target_bins'])):
        _save_path_cv = os.path.join(main_path, f'cv_{cv_n}')
        os.makedirs(_save_path_cv, exist_ok=True)

        train_data, test_data = df_final.iloc[train_index], df_final.iloc[test_index]

        if _impute:
            imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=20, imputation_order='ascending', initial_strategy='median', random_state=0)
            train_data_imputed = imputer.fit_transform(train_data[model_features[model_name]])
            test_data_imputed = imputer.transform(test_data[model_features[model_name]])

            train_data = pd.DataFrame(train_data_imputed, index=train_data.index, columns=model_features[model_name])
            test_data = pd.DataFrame(test_data_imputed, index=test_data.index, columns=model_features[model_name])

        train_data.to_csv(os.path.join(_save_path_cv, 'train_data.csv'))
        test_data.to_csv(os.path.join(_save_path_cv, 'test_data.csv'))

        print(f'Starting training for {column}...')
        predictor = TabularPredictor(label=column, eval_metric='r2', problem_type="regression", path=_save_path_cv).fit(
            train_data.reset_index(drop=True), 
            presets='good_quality', 
            excluded_model_types=['TABPFN','FT_TRANSFORMER','NN_TORCH','FASTAI'], 
            verbosity=2
        )

        support.save_metrics(predictor, test_data, train_data, model_features[model_name], _save_path_cv, column)
        fi_plotting.run_feat_importance(predictor, nsh, test_data, train_data, model_features[model_name], column, "all", _save_path_cv)
