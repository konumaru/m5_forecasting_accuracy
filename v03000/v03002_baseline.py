import os
import gc
import re
import json
import pickle
import datetime
from tqdm import tqdm
from typing import Union

import numpy as np
import pandas as pd
pd.options.display.max_columns = None

from typing import Union

import seaborn
import matplotlib.pyplot as plt
plt.style.use("seaborn-darkgrid")

from scipy.stats import linregress

from sklearn import preprocessing
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import lightgbm as lgb

# custom funcs
from script import WRMSSEEvaluator
from script import cache_result
from script import reduce_mem_usage
from script import load_pickle, dump_pickle


# Define global Variables.
SEED = 42
VERSION = str(__file__).split('_')[0]
TARGET = 'sales'
NUM_ITEMS = 30490

FEATURECOLS_PATH = f'result/feature_cols/{VERSION}.pkl'
MODEL_PATH = f'result/model/{VERSION}.pkl'
IMPORTANCE_PATH = f'result/importance/{VERSION}.png'
SCORE_PATH = f'result/score/{VERSION}.pkl'


""" Load Data and Initial Processing
"""
@cache_result(filename='parse_calendar', use_cache=True)
def parse_calendar():
    calendar = pd.read_pickle('../data/reduced/calendar.pkl')
    # Drop Initial Columns.
    drop_features = ['weekday', 'wday', 'month', 'year']
    calendar.drop(drop_features, inplace=True, axis=1)
    # Fill nan feature and label encoding
    nan_features = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    for f in nan_features:
        calendar[f].fillna('none', inplace=True)

        encodaer = preprocessing.LabelEncoder()
        calendar[f] = encodaer.fit_transform(calendar[f])

    calendar['date'] = pd.to_datetime(calendar['date'])
    attrs = [
        "year",
        "quarter",
        "month",
        "week",
        # "weekofyear",
        "day",
        "dayofweek",
        "is_year_end",
        "is_year_start",
        "is_quarter_end",
        "is_quarter_start",
        "is_month_end",
        "is_month_start",
    ]

    for attr in attrs:
        dtype = np.int16 if attr == "year" else np.int8
        calendar[attr] = getattr(calendar['date'].dt, attr).astype(dtype)
    calendar["is_weekend"] = calendar["dayofweek"].isin([5, 6]).astype(np.int8)
    return calendar


@cache_result(filename='parse_sell_prices', use_cache=True)
def parse_sell_prices():
    sell_prices = pd.read_pickle('../data/reduced/sell_prices.pkl')
    return sell_prices


@cache_result(filename='parse_sales_train', use_cache=True)
def parse_sales_train():
    train = pd.read_pickle('../data/reduced/sales_train_validation.pkl')
    # Add Prediction Columns
    start_d = 1914
    end_d = 1969
    for i in range(start_d, end_d + 1):
        train[f'd_{i}'] = 0
    return train


""" Transform
"""
@cache_result(filename='melted_and_merged_train', use_cache=True)
def melted_and_merged_train():
    # Load Data
    calendar = pd.read_pickle('features/parse_calendar.pkl')
    sell_prices = pd.read_pickle('features/parse_sell_prices.pkl')
    df = pd.read_pickle('features/parse_sales_train.pkl')

    idx_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    df = pd.melt(df, id_vars=idx_cols, var_name='d', value_name='sales')
    # Drop very old data.
    nrows = (365 * 2 + 28 * 2) * NUM_ITEMS
    df = df.iloc[-nrows:, :]
    df = pd.merge(df, calendar, how='left', on='d')
    df = pd.merge(df, sell_prices, how='left', on=['store_id', 'item_id', 'wm_yr_wk'])

    # Label Encoding
    label_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    for c in label_cols:
        encodaer = preprocessing.LabelEncoder()
        df[c] = encodaer.fit_transform(df[c])

    cat_cols = [
        'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
        'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2'
    ]
    for c in cat_cols:
        encoder = preprocessing.LabelEncoder()
        df[c] = pd.Series(encoder.fit_transform(df[c])).astype('category')
    print(f'Our final dataset to train has {df.shape[0]} rows and {df.shape[1]} columns\n')
    return df.pipe(reduce_mem_usage)


""" Feature Engineering
"""
@cache_result(filename='simple_fe', use_cache=True)
def simple_fe():
    df = pd.read_pickle('features/melted_and_merged_train.pkl')
    # rolling demand features
    for val in [28, 29, 30]:
        df[f'sales_lag_t{val}'] = df.groupby(['id'])['sales'].transform(lambda x: x.shift(val))

    for val in [7, 30, 60, 90, 180]:
        df[f'sales_rolling_mean_t{val}'] = df.groupby(['id'])['sales'].transform(
            lambda x: x.shift(28).rolling(val).mean())
        df[f'sales_rolling_std_t{val}'] = df.groupby(['id'])['sales'].transform(
            lambda x: x.shift(28).rolling(val).std())

    df['sales_rolling_skew_t30'] = df.groupby(['id'])['sales'].transform(
        lambda x: x.shift(28).rolling(30).skew())
    df['sales_rolling_kurt_t30'] = df.groupby(['id'])['sales'].transform(
        lambda x: x.shift(28).rolling(30).kurt())

    # price features
    df['price_lag_t1'] = df.groupby(['id'])['sell_price'].transform(lambda x: x.shift(1))
    df['price_change_t1'] = (df['price_lag_t1'] - df['sell_price']) / (df['price_lag_t1'])
    df['rolling_price_max_t365'] = df.groupby(['id'])['sell_price'].transform(
        lambda x: x.shift(1).rolling(365).max())
    df['price_change_t365'] = (df['rolling_price_max_t365'] - df['sell_price'])\
        / (df['rolling_price_max_t365'])
    df['price_rolling_std_t7'] = df.groupby(['id'])['sell_price'].transform(
        lambda x: x.rolling(7).std())
    df['price_rolling_std_t30'] = df.groupby(['id'])['sell_price'].transform(
        lambda x: x.rolling(30).std())
    df.drop(['rolling_price_max_t365', 'price_lag_t1'], inplace=True, axis=1)

    return df.pipe(reduce_mem_usage)


""" Define Evaluation Object
- WRMSSEForLightGBM の制約は
    - validation では, すべての id が存在し, 連続する28日のデータであること.
    - validation と prediction の id の順序が同一であること.
"""


class WRMSSEForLightGBM(WRMSSEEvaluator):
    def custom_feval(self, preds, dtrain):
        row, col = self.valid_df[self.valid_target_columns].shape
        preds = preds.reshape(col, row).T

        score = self.score(preds)
        return 'WRMSSE', score, False

    def custom_fobj(self, preds, dtrain):
        weight = 2 * np.power(self.weight, 2) / self.scale
        labels = dtrain.get_label()

        grad = weight * (preds - labels)
        hess = weight
        return grad, hess


def get_evaluator(go_back_days=28):
    pred_days = 28
    train_df = pd.read_pickle('../data/reduced/sales_train_validation.pkl')
    train_fold_df = train_df.iloc[:, :-go_back_days]
    valid_fold_df = train_df.iloc[:, -go_back_days:(-go_back_days + pred_days)].copy()

    evaluator = WRMSSEForLightGBM(
        train_df=train_fold_df,
        valid_df=valid_fold_df,
        calendar=pd.read_pickle('../data/reduced/calendar.pkl'),
        prices=pd.read_pickle('../data/reduced/sell_prices.pkl')
    )
    return evaluator


""" Train Model
"""


def train_submit_split(df):
    oldest_submit_date = '2016-04-25'
    submit_mask = (df['date'] >= oldest_submit_date)
    train_mask = ~submit_mask
    return df[train_mask], df[submit_mask]


def train_valid_split(df, go_back_days=28):
    valid_duration = 28

    min_thresh_date = df['date'].max() - datetime.timedelta(days=go_back_days)
    max_thresh_date = min_thresh_date + datetime.timedelta(days=valid_duration)

    eval_mask = ((df["date"] > min_thresh_date) & (df["date"] <= max_thresh_date))
    train_mask = (df["date"] <= min_thresh_date)
    return df[train_mask], df[eval_mask]


def save_importance(model, filepath, max_num_features=50, figsize=(15, 20)):
    # Define Feature Importance DataFrame.
    imp_df = pd.DataFrame(
        [model.feature_importance()],
        columns=model.feature_name(),
        index=['Importance']
    ).T
    imp_df.sort_values(by='Importance', inplace=True)
    # Plot Importance DataFrame.
    plt.figure(figsize=figsize)
    imp_df[-max_num_features:].plot(
        kind='barh', title='Feature importance', figsize=figsize,
        y='Importance', align="center"
    )
    plt.savefig(filepath)
    plt.close('all')


def run_lgb(data, features):
    x_train = data[data['date'] <= '2016-03-27']
    y_train = x_train['sales']
    x_val = data[(data['date'] > '2016-03-27') & (data['date'] <= '2016-04-24')]
    y_val = x_val['sales']
    test = data[(data['date'] > '2016-04-24')]
    del data
    gc.collect()

    params = {
        'boosting': 'gbdt',
        'metric': 'rmse',
        'objective': 'poisson',
        'n_jobs': -1,
        'seed': 20,
        'learning_rate': 0.1,
        'alpha': 0.1,
        'lambda': 0.1,
        'bagging_fraction': 0.66,
        'bagging_freq': 2,
        'colsample_bytree': 0.77
    }

    train_set = lgb.Dataset(x_train[features], y_train)
    val_set = lgb.Dataset(x_val[features], y_val)

    del x_train, y_train
    model = lgb.train(params, train_set, num_boost_round=2000, early_stopping_rounds=200,
                      valid_sets=[train_set, val_set], verbose_eval=100)

    dump_pickle(model, MODEL_PATH)
    model = load_pickle(MODEL_PATH)

    val_pred = model.predict(x_val[features])
    val_score = np.sqrt(mean_squared_error(val_pred, y_val))
    print(f'Our val rmse score is {val_score}')
    y_pred = model.predict(test[features])
    test['sales'] = y_pred
    save_importance(model, filepath=IMPORTANCE_PATH)
    return test


def run_train():
    go_back_days = 28
    df = load_pickle('features/all_train_data.pkl')

    evaluator = get_evaluator(go_back_days)
    train_data, valid_data = train_valid_split(df, go_back_days)
    del df; gc.collect()

    features = [
        # 'id',
        'item_id',
        'dept_id',
        'cat_id',
        'store_id',
        'state_id',
        # 'd',
        # 'sales',
        # 'date',
        # 'wm_yr_wk',
        'month',
        'year',
        'event_name_1',
        'event_type_1',
        'event_name_2',
        'event_type_2',
        'snap_CA',
        'snap_TX',
        'snap_WI',
        'quarter',
        'day',
        'dayofweek',
        'is_year_end',
        'is_year_start',
        'is_quarter_end',
        'is_quarter_start',
        'is_month_end',
        'is_month_start',
        'is_weekend',
        'sell_price',
        'sales_lag_t28',
        'sales_lag_t29',
        'sales_lag_t30',
        'sales_rolling_mean_t7',
        'sales_rolling_std_t7',
        'sales_rolling_mean_t30',
        'sales_rolling_std_t30',
        'sales_rolling_mean_t60',
        'sales_rolling_std_t60',
        'sales_rolling_mean_t90',
        'sales_rolling_std_t90',
        'sales_rolling_mean_t180',
        'sales_rolling_std_t180',
        'sales_rolling_skew_t30',
        'sales_rolling_kurt_t30',
        'price_change_t1',
        'price_change_t365',
        'price_rolling_std_t7',
        'price_rolling_std_t30'
    ]
    dump_pickle(features, FEATURECOLS_PATH)

    train_set = lgb.Dataset(train_data[features], train_data[TARGET])
    val_set = lgb.Dataset(valid_data[features], valid_data[TARGET])
    del train_data; gc.collect()

    params = {
        'model_params': {
            'boosting': 'gbdt',
            'metric': 'rmse',
            'objective': 'poisson',
            'n_jobs': -1,
            'seed': 20,
            'learning_rate': 0.1,
            'alpha': 0.1,
            'lambda': 0.1,
            'bagging_fraction': 0.66,
            'bagging_freq': 2,
            'colsample_bytree': 0.77
        },
        'train_params': {
            'num_boost_round': 2000,
            'early_stopping_rounds': 200,
            'verbose_eval': 100
        }
    }
    print('Parameters:\n', json.dumps(params, indent=4), '\n')

    # model = lgb.train(
    #     params['model_params'], train_set, valid_sets=[train_set, val_set], **params['train_params']
    # )
    model = load_pickle(MODEL_PATH)
    dump_pickle(model, MODEL_PATH)

    val_pred = model.predict(valid_data[features], num_iteration=model.best_iteration)
    val_score = np.sqrt(mean_squared_error(val_pred, valid_data[TARGET]))
    print(f'Our val rmse score is {val_score}')
    wrmsse = evaluator.score(val_pred)
    print(f'Our val wrmsse score is {wrmsse}')

    save_importance(model, filepath=IMPORTANCE_PATH)
    return model


""" Evaluation
"""


""" Submission
"""


def run_submission():
    model = load_pickle(MODEL_PATH)
    features = load_pickle(FEATURECOLS_PATH)
    submit_data = load_pickle('features/submit_data.pkl')
    submit_data['sales'] = model.predict(submit_data[features], num_iteration=model.best_iteration)

    submission = pd.pivot(submit_data, index='id', columns='d', values='sales').reset_index()
    # split valid and eval
    valid_sub = submission[['id'] + [f'd_{i}' for i in range(1914, 1942)]]
    eval_sub = submission[['id'] + [f'd_{i}' for i in range(1942, 1970)]]
    # rename columns
    valid_sub.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]
    eval_sub.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]
    # rename id only evaluation
    eval_sub = eval_sub.assign(id=lambda x: x['id'].str.replace('validation', 'evaluation'))

    submission = pd.concat([valid_sub, eval_sub], axis=0)
    sample_submission = pd.read_pickle('../data/reduced/sample_submission.pkl')
    submission = sample_submission[['id']].merge(submission, how='left', on='id')
    submission.to_csv(f'submit/{VERSION}.csv.gz', index=False, compression='gzip')

    print(submission.shape)
    print(submission.head())


def main():
    print('\n\n--- Load Data and Initial Processing ---\n\n')
    _ = parse_calendar()
    _ = parse_sell_prices()
    _ = parse_sales_train()

    print('\n\n--- Transform ---\n\n')
    _ = melted_and_merged_train()

    print('\n\n--- Feature Engineering ---\n\n')
    all_data = simple_fe()
    all_train_data, submit_data = train_submit_split(all_data)
    # Dump Split Data.
    print('Cache Train and Submission Data.')
    dump_pickle(all_train_data, 'features/all_train_data.pkl')
    dump_pickle(submit_data, 'features/submit_data.pkl')
    del all_data, all_train_data, submit_data; gc.collect();

    print('\n\n--- Define Evaluation Object ---\n\n')

    print('\n\n--- Train Model ---\n\n')
    _ = run_train()

    print('\n\n--- Evaluation ---\n\n')

    print('\n\n--- Submission ---\n\n')
    run_submission()


if __name__ == "__main__":
    main()
