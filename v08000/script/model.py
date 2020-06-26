import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use("seaborn-darkgrid")

import xgboost as xgb
import lightgbm as lgb
import catboost as cat


class LightGBM_Wrapper():

    def __init__(self):
        self.model = None
        self.importance = None

        self.train_bin_path = 'tmp_train_set.bin'
        self.valid_bin_path = 'tmp_valid_set.bin'

    def _remove_bin_file(self, filename):
        if os.path.exists(filename):
            os.remove(filename)

    def dataset_to_binary(self, train_dataset, valid_dataset):
        # Remove Binary Cache.
        self._remove_bin_file(self.train_bin_path)
        self._remove_bin_file(self.valid_bin_path)
        # Save Binary Cache.
        train_dataset.save_binary(self.train_bin_path)
        valid_dataset.save_binary(self.valid_bin_path)
        # Reload Binary Cache.
        train_dataset = lgb.Dataset(self.train_bin_path)
        valid_dataset = lgb.Dataset(self.valid_bin_path)
        return train_dataset, valid_dataset

    def fit(self, params, train_param,
            X_train, y_train, X_valid, y_valid,
            train_weight=None, valid_weight=None):
        train_dataset = lgb.Dataset(X_train, y_train, weight=train_weight)
        valid_dataset = lgb.Dataset(X_valid, y_valid, weight=valid_weight, reference=train_dataset)

        train_dataset, valid_dataset = self.dataset_to_binary(train_dataset, valid_dataset)

        self.model = lgb.train(
            params,
            train_dataset,
            valid_sets=[valid_dataset],
            **train_param
        )
        # Remove Binary Cache.
        self._remove_bin_file(self.train_bin_path)
        self._remove_bin_file(self.valid_bin_path)

    def predict(self, data):
        return self.model.predict(data, num_iteration=self.model.best_iteration)

    def model_importance(self):
        model = self.model
        f_name = model.feature_name()
        f_imp = model.feature_importance(
            importance_type='gain', iteration=model.best_iteration)

        importance = pd.Series(dict(zip(f_name, f_imp)), name='Importance').to_frame()
        importance.sort_values(by='Importance', inplace=True)
        return importance

    def save_importance(self, filepath, max_num_features=50, figsize=(18, 25)):
        imp_df = self.model_importance()
        # Plot Importance DataFrame.
        plt.figure(figsize=figsize)
        imp_df[-max_num_features:].plot(
            kind='barh', title='Feature importance', figsize=figsize,
            y='Importance', align="center"
        )
        plt.savefig(filepath)
        plt.close('all')


class XGBoost_Wrapper():
    def __init__(self):
        self.model = None

    def fit(self, params, train_param,
            X_train, y_train, X_valid, y_valid,
            train_weight=None, valid_weight=None):
        train_dataset = xgb.DMatrix(X_train, label=y_train, weight=train_weight)
        valid_dataset = xgb.DMatrix(X_valid, label=y_valid, weight=valid_weight)

        self.model = xgb.train(
            params,
            train_dataset,
            evals=[(valid_dataset, 'valid')],
            **train_param
        )

    def predict(self, data):
        data = xgb.DMatrix(data)
        return self.model.predict(data, ntree_limit=self.model.best_ntree_limit)

    def save_importance(self, filepath, max_num_features=50, figsize=(18, 25)):
        importance = pd.Series(self.model.get_fscore(), name='Importance').to_frame()
        importance.sort_values(by='Importance', inplace=True)

        # Plot Importance DataFrame.
        plt.figure(figsize=figsize)
        importance[-max_num_features:].plot(
            kind='barh', title='Feature importance', figsize=figsize,
            y='Importance', align="center"
        )
        plt.savefig(filepath)
        plt.close('all')


class Catboost_Wrapper():
    def __init__(self):
        self.model = None

    def fit(self, params, train_params,
            X_train, y_train, X_valid, y_valid,
            train_weight=None, valid_weight=None):

        categorical_cols = X_train.dtypes[X_train.dtypes == 'category'].index.tolist()
        train_pool = cat.Pool(X_train, y_train, cat_features=categorical_cols, weight=train_weight)
        valid_pool = cat.Pool(X_valid, y_valid, cat_features=categorical_cols, weight=valid_weight)

        self.model = cat.CatBoost(params)
        self.model.fit(train_set, eval_set=valid_set, **train_params)

    def predict(self, data):
        return model.predict(data, prediction_type='RawFormulaVal')

    def save_importance(self, filepath, max_num_features=50, figsize=(18, 25)):
        f_importance = self.model.get_feature_importance()
        f_name = self.model.feature_names_

        importance = pd.DataFrame(f_importance, index=f_name, columns=['Importance'])
        importance.sort_values(by='Importance', inplace=True)

        # Plot Importance DataFrame.
        plt.figure(figsize=figsize)
        importance[-max_num_features:].plot(
            kind='barh', title='Feature importance', figsize=figsize,
            y='Importance', align="center"
        )
        plt.savefig(filepath)
        plt.close('all')


class LightGBM_Model():
    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

    def save_importance(self):
        pass


class LightGBM_CV_Model():
    def __init__(self):
        pass
