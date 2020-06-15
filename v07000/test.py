import os
import gc
import re
import math
import json
import pickle
import datetime
from tqdm import tqdm
from typing import Union

from workalendar.usa.texas import Texas
from workalendar.usa.california import California
from workalendar.usa.wisconsin import Wisconsin

import numpy as np
import pandas as pd
pd.options.display.max_columns = None

from typing import Union

import seaborn
import matplotlib.pyplot as plt
plt.style.use("seaborn-darkgrid")

from scipy.stats import mode
from scipy.stats import linregress

from sklearn import preprocessing
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import xgboost as xgb
import lightgbm as lgb

import warnings
warnings.filterwarnings('ignore')

# custom funcs
from script import WRMSSEEvaluator
from script import GaussianTargetEncoder
from script import cache_result
from script import reduce_mem_usage
from script import load_pickle, dump_pickle
from script import get_groups

IS_TEST = True


class CustomWRMSSE(WRMSSEEvaluator):

    def lgb_custom_feval(self, preds, dtrain):
        actual = dtrain.get_label().reshape(28, -1).T
        preds = preds.reshape(28, -1).T

        rmse = np.sqrt(np.mean(np.square(actual - preds), axis=1))
        score = np.sum(self.valid_feval_weight * rmse)
        return 'WRMSSE', score, False

    def xgb_custom_feval(self, preds, dtrain):
        actual = dtrain.get_label().reshape(28, -1).T
        preds = preds.reshape(28, -1).T

        rmse = np.sqrt(np.mean(np.square(actual - preds), axis=1))
        score = np.sum(self.valid_feval_weight * rmse)
        return 'WRMSSE', score

    def get_series_weight(self, data_ids):
        data_ids = data_ids.apply(lambda x: x.rsplit('_', 1)[0])

        weight_df = self.weights * 12
        weight_df.index = weight_df.index.str.replace('--', '_')
        weight_df.columns = ['weight']
        weight_df['scale'] = self.scale

        fobj_weight = weight_df.loc[data_ids, 'weight'].values
        fojb_sclae = weight_df.loc[data_ids, 'scale'].values
        return fobj_weight, fojb_sclae

    def set_feval_weight(self, valid_ids):
        weight, scale = self.get_series_weight(valid_ids)
        self.valid_feval_weight = weight / np.sqrt(scale)

    def set_series_weight_for_fobj(self, train_ids):
        fobj_weight, fojb_scale = self.get_series_weight(train_ids)
        self.custom_fobj_weight = 2 * np.square(fobj_weight) / fojb_scale

    def custom_fobj(self, preds, dtrain):
        actual = dtrain.get_label()
        weight = self.custom_fobj_weight

        grad = weight * np.square(preds - actual)
        hess = weight * 2 * (preds - actual)
        return grad, hess


@cache_result(filename='evaluator', use_cache=False)
def get_evaluator():
    df = pd.read_pickle('../data/reduced/sales_train_evaluation.pkl')
    train_df = df.iloc[:, : -28] if IS_TEST else df
    evaluator = CustomWRMSSE(
        train_df=train_df,  # 最後の29列が重みに使われる
        valid_df=df.iloc[:, -28:],  # evaluator.scoreのときに使用されるラベルデータ
        calendar=pd.read_pickle('../data/reduced/calendar.pkl'),
        prices=pd.read_pickle('../data/reduced/sell_prices.pkl')
    )
    return evaluator


df = pd.read_pickle("features/train_data_FOODS_1.pkl")

catgorical_cols = [
    'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
    'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2'
]

df[catgorical_cols] = df[catgorical_cols].astype('int64')

print(df.shape)
print(df.info(), '\n')


def train_valid_split(df, go_back_days=28):
    valid_duration = 28

    min_thresh_date = df['date'].max() - datetime.timedelta(days=go_back_days)
    max_thresh_date = min_thresh_date + datetime.timedelta(days=valid_duration)

    valid_mask = ((df["date"] > min_thresh_date) & (df["date"] <= max_thresh_date))
    train_mask = (df["date"] <= min_thresh_date)
    return df[train_mask], df[valid_mask]


drop_cols = ['id', 'd', 'sales', 'date', 'wm_yr_wk']
features = [f for f in df.columns if f not in drop_cols]

train_data, valid_data = train_valid_split(df, go_back_days=28)
X_train, y_train = train_data[features], train_data['sales']
X_valid, y_valid = valid_data[features], valid_data['sales']


class LGBM_Model():

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

    def fit(self, params, train_param, X_train, y_train, X_valid, y_valid):
        train_dataset = lgb.Dataset(
            X_train, y_train, feature_name=X_train.columns.tolist(), weight=None)
        valid_dataset = lgb.Dataset(
            X_valid, y_valid, weight=None, reference=train_dataset)

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
        imp_df = pd.DataFrame(
            [self.model.feature_importance()],
            columns=self.model.feature_name(),
            index=['Importance']
        ).T
        imp_df.sort_values(by='Importance', inplace=True)
        return imp_df

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

    def fit(self, params, train_param, X_train, y_train, X_valid, y_valid):
        train_dataset = xgb.DMatrix(X_train, label=y_train)
        valid_dataset = xgb.DMatrix(X_valid, label=y_valid)

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


evaluator = get_evaluator()
evaluator.set_feval_weight(valid_data['id'].drop_duplicates(keep='last'))

# params = {
#     'model_params': {
#         'boosting': 'gbdt',
#         'objective': 'tweedie',  # tweedie, poisson, regression
#         'tweedie_variance_power': 1,  # 1.0=poisson
#         'metric': 'rmse',
#         'num_leaves': 2**7 - 1,
#         'min_data_in_leaf': 50,
#         'seed': 42,
#         'learning_rate': 0.03,  # 0.1
#         'subsample': 0.5,  # ~v05006, 0.8
#         'subsample_freq': 1,
#         'feature_fraction': 0.5,  # ~v05006, 0.8
#         # 'lambda_l1': 0.1, # v06012, 0.554 -> 0.561
#         'lambda_l2': 0.1,  # v06012, 0.554 -> 0.555
#         # 'max_bin': 100,  # Score did not change.
#         'force_row_wise': True,
#         'verbose': -1
#     },
#     'train_params': {
#         'num_boost_round': 1500,  # 2000
#         'early_stopping_rounds': 100,
#         'verbose_eval': 100,
#         'feval': evaluator.lgb_custom_feval
#     }
# }

# lgb_model = LGBM_Model()
# lgb_model.fit(params['model_params'], params['train_params'], X_train, y_train, X_valid, y_valid)
# lgb_model.save_importance('test_importance.png', max_num_features=80, figsize=(25, 30))

# pred = lgb_model.predict(X_valid)

params = {
    'model_params': {
        'booster': 'gbtree',
        'objective': 'reg:tweedie',  # tweedie, poisson, regression
        'tweedie_variance_power': 1,  # 1.0=poisson
        # 'eval_metric': 'rmse',
        'disable_default_eval_metric': 1,
        'max_depth': 7,
        'max_leaves': 2**7 - 1,
        'min_child_weight': 20,
        'seed': 42,
        'eta': 0.03,
        'subsample': 0.5,
        # 'alpha': 0.1,
        'lambda': 0.1,
        # 'max_bin': 100,  # Score did not change.
        'verbose': -1
    },
    'train_params': {
        'num_boost_round': 10,  # 2000
        'early_stopping_rounds': 10,
        'verbose_eval': 1,
        'feval': evaluator.xgb_custom_feval
    }
}

xgb_model = XGBoost_Wrapper()
xgb_model.fit(params['model_params'], params['train_params'], X_train, y_train, X_valid, y_valid)
pred = xgb_model.predict(X_valid)

xgb_model.save_importance('test_imp.png')


def rmse(pred, actual):
    return np.sqrt(np.mean(np.square(pred - actual)))


print(rmse(pred, y_valid))
