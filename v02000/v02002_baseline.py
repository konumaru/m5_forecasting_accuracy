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
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error

import lightgbm as lgb

# custom funcs
from script import WRMSSEEvaluator
from script import cache_result
from script import reduce_mem_usage
from script import load_pickle, dump_pickle


SEED = 42
VERSION = str(__file__).split('_')[0]
TARGET = 'sales'

MODEL_PATH = f'result/model/{VERSION}.pkl'
SCORE_PATH = f'result/score/{VERSION}.json'


""" Load Data and Initial Processing
"""
@cache_result(filename='parse_calendar', use_cache=True)
def parse_calendar():
    calendar = pd.read_pickle('../data/reduced/calendar.pkl')
    # fill null feature
    nan_features = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    for f in nan_features:
        calendar[f].fillna('null', inplace=True)
    # label encoding
    cat_features = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    for f in cat_features:
        encodaer = preprocessing.LabelEncoder()
        calendar[f] = encodaer.fit_transform(calendar[f])

    calendar['date'] = pd.to_datetime(calendar['date'])
    attrs = [
        "year",
        # "quarter",
        "month",
        "week",
        # "weekofyear",
        "day",
        "dayofweek",
        # "is_year_end",
        # "is_year_start",
        # "is_quarter_end",
        # "is_quarter_start",
        # "is_month_end",
        # "is_month_start",
    ]

    for attr in attrs:
        calendar[attr] = getattr(calendar['date'].dt, attr)
    # calendar["is_weekend"] = calendar["dayofweek"].isin([5, 6]).astype(np.int8)|

    # drop_features = ['weekday', 'wday', 'month', 'year']
    # features = calendar.columns.tolist()
    # features = [f for f in features if f not in drop_features]
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
    df = pd.merge(df, calendar, how='left', on='d')
    df = pd.merge(df, sell_prices, how='left', on=['store_id', 'item_id', 'wm_yr_wk'])

    # Label Encoding
    cat_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    for c in cat_cols:
        encodaer = preprocessing.LabelEncoder()
        df[c] = encodaer.fit_transform(df[c])

    df.dropna(subset=['sell_price'], axis=0, inplace=True)
    return df.pipe(reduce_mem_usage)


""" Feature Engineering
"""
@cache_result(filename='simple_fe', use_cache=True)
def simple_fe():
    df = pd.read_pickle('features/melted_and_merged_train.pkl')
    # rolling demand features
    df['sales_lag_t28'] = df.groupby(['id'])['sales'].transform(lambda x: x.shift(28))
    df['sales_lag_t29'] = df.groupby(['id'])['sales'].transform(lambda x: x.shift(29))
    df['sales_lag_t30'] = df.groupby(['id'])['sales'].transform(lambda x: x.shift(30))
    df['sales_rolling_mean_t7'] = df.groupby(['id'])['sales'].transform(lambda x: x.shift(28).rolling(7).mean())
    df['sales_rolling_std_t7'] = df.groupby(['id'])['sales'].transform(lambda x: x.shift(28).rolling(7).std())
    df['sales_rolling_mean_t30'] = df.groupby(['id'])['sales'].transform(lambda x: x.shift(28).rolling(30).mean())
    df['sales_rolling_mean_t90'] = df.groupby(['id'])['sales'].transform(lambda x: x.shift(28).rolling(90).mean())
    df['sales_rolling_mean_t180'] = df.groupby(['id'])['sales'].transform(lambda x: x.shift(28).rolling(180).mean())
    df['sales_rolling_std_t30'] = df.groupby(['id'])['sales'].transform(lambda x: x.shift(28).rolling(30).std())
    df['sales_rolling_skew_t30'] = df.groupby(['id'])['sales'].transform(lambda x: x.shift(28).rolling(30).skew())
    df['sales_rolling_kurt_t30'] = df.groupby(['id'])['sales'].transform(lambda x: x.shift(28).rolling(30).kurt())

    # price features
    df['price_lag_t1'] = df.groupby(['id'])['sell_price'].transform(lambda x: x.shift(1))  # after drop.
    df['price_change_t1'] = (df['price_lag_t1'] - df['sell_price']) / (df['price_lag_t1'])
    df['rolling_price_max_t365'] = df.groupby(
        ['id'])['sell_price'].transform(
        lambda x: x.shift(1).rolling(365).max())  # after drop.
    df['price_change_t365'] = (df['rolling_price_max_t365'] - df['sell_price']) / (df['rolling_price_max_t365'])
    df['price_rolling_std_t7'] = df.groupby(['id'])['sell_price'].transform(lambda x: x.rolling(7).std())
    df['price_rolling_std_t30'] = df.groupby(['id'])['sell_price'].transform(lambda x: x.rolling(30).std())
    df.drop(['rolling_price_max_t365', 'price_lag_t1'], inplace=True, axis=1)

    return df.pipe(reduce_mem_usage)


""" Define Evaluation Object
- WRMSSEForLightGBM の制約は
    - validation では, すべての id が存在し, 連続する28日のデータであること.
    - validation と prediction の id の順序が同一であること.
"""


class WRMSSEForLightGBM(WRMSSEEvaluator):
    def feval(self, preds, dtrain):
        row, col = self.valid_df[self.valid_target_columns].shape
        preds = preds.reshape(col, row).T

        score = self.score(preds)
        return 'WRMSSE', score, False

    def get_sample_weight(self, data_idx):
        '''
        sample weight for rmse.
        Weights for doing WRMSSE-like evaluations using RMSE.
        '''
        data_idx = data_idx.apply(lambda x: x.rsplit('_', 1)[0]).values

        weight_df = self.weights * 12
        weight_df.index = weight_df.index.str.replace('--', '_')
        weight_df.columns = ['weight']
        scale = np.where(self.scale != 0, self.scale, 1)
        weight_df['sample_weight'] = weight_df['weight'] / scale

        return weight_df.loc[data_idx, 'sample_weight'].values


@cache_result(filename='evaluator', use_cache=False)
def get_evaluator():
    train_df = pd.read_pickle('../data/reduced/sales_train_validation.pkl')

    train_fold_df = train_df.iloc[:, :-28]
    valid_fold_df = train_df.iloc[:, -28:].copy()

    evaluator = WRMSSEForLightGBM(
        train_df=train_fold_df,
        valid_df=valid_fold_df,
        calendar=pd.read_pickle('../data/reduced/calendar.pkl'),
        prices=pd.read_pickle('../data/reduced/sell_prices.pkl')
    )
    return evaluator


""" Train Model
"""


def train_eval_submit_split(df, eval_days=28):
    oldest_submit_date = '2016-04-25'
    submit_mask = (df["date"] >= oldest_submit_date)

    eval_date = datetime.datetime.strptime(oldest_submit_date, '%Y-%m-%d') \
        - datetime.timedelta(days=eval_days)
    eval_mask = ((df["date"] >= eval_date) & (df["date"] < oldest_submit_date))

    train_mask = ((~eval_mask) & (~submit_mask))
    return df[train_mask], df[eval_mask], df[submit_mask]


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


def run_train(all_train_data, features):
    evaluator = load_pickle('features/evaluator.pkl')

    train_days = 365 * 3
    train_thresh = all_train_data['date'].max() - datetime.timedelta(days=train_days)
    all_train_data = all_train_data[all_train_data['date'] > train_thresh]

    train_data, valid_data = train_test_split(
        all_train_data, test_size=0.1, shuffle=False, random_state=SEED)

    train_set = lgb.Dataset(train_data[features], train_data[TARGET])
    val_set = lgb.Dataset(valid_data[features], valid_data[TARGET], reference=train_set)

    use_weight = False
    if use_weight:
        train_set.set_weight(evaluator.get_sample_weight(train_data['id']))
        val_set.set_weight(evaluator.get_sample_weight(valid_data['id']))

    params = {
        'boosting_type': 'gbdt',
        'metric': 'rmse',
        'objective': 'regression',
        'n_jobs': -1,
        'seed': SEED,
        'learning_rate': 0.1,
        'bagging_fraction': 0.75,
        'bagging_freq': 5,
        'colsample_bytree': 0.75,
        'verbosity': -1
    }

    print(json.dumps(params, indent=4), '\n')

    train_params = {
        'num_boost_round': 2500,
        'early_stopping_rounds': 50,
        'verbose_eval': 100,
    }
    model = lgb.train(params, train_set, valid_sets=[train_set, val_set], **train_params)
    # Save Image of feature importance.
    save_importance(model, filepath=f'result/importance/{VERSION}.png')
    # Export Model.
    dump_pickle(model, filepath=MODEL_PATH)


""" Evaluation
"""


def run_evaluation(eval_data, features):
    model = load_pickle(MODEL_PATH)
    evaluator = load_pickle('features/evaluator.pkl')
    scores = {}

    val_pred = model.predict(eval_data[features])
    scores['RMSE'] = metrics.mean_squared_error(val_pred, eval_data[TARGET], squared=False)

    valid_preds = val_pred.reshape(28, -1).T
    scores['WRMSSE'] = evaluator.score(valid_preds)

    for f_name, score in scores.items():
        print(f'Our val {f_name} score is {score}')

    dump_pickle(scores, SCORE_PATH)


""" Submission
"""


def run_submission(sub_data, features):
    model = load_pickle(MODEL_PATH)
    wrmsse_socre = load_pickle(SCORE_PATH)['WRMSSE']

    sub_pred = model.predict(sub_data[features])

    submission = sub_data[['id', 'd']].copy(deep=True)
    submission['sales'] = sub_pred
    submission = pd.pivot(submission, index='id', columns='d', values='sales').reset_index()
    # split valid and eval
    valid_sub = submission[['id'] + [f'd_{i}' for i in range(1914, 1942)]]
    eval_sub = submission[['id'] + [f'd_{i}' for i in range(1942, 1970)]]
    # rename columns
    valid_sub.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]
    eval_sub.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]
    # rename id evaluation
    eval_sub = eval_sub.assign(id=lambda x: x['id'].str.replace('validation', 'evaluation'))

    submission = pd.concat([valid_sub, eval_sub], axis=0)

    sample_submission = pd.read_pickle('../data/reduced/sample_submission.pkl')
    submission = sample_submission[['id']].merge(submission, how='left', on='id')
    submission.to_csv(
        f'submit/{VERSION}_{wrmsse_socre:.05}.csv.gz', index=False, compression='gzip'
    )

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
    train = simple_fe()

    print('\n\n--- Define Evaluation Object ---\n\n')
    _ = get_evaluator()

    print('\n\n--- Train Model ---\n\n')
    cols_to_drop = ['id', 'd', 'date', 'wm_yr_wk', 'weekday', 'year'] + [TARGET]
    features = train.columns.tolist()
    features = [f for f in features if f not in cols_to_drop]

    all_train_data, eval_data, sub_data = train_eval_submit_split(train)
    del train; gc.collect()
    run_train(all_train_data, features)
    del all_train_data; gc.collect()

    print('\n\n--- Evaluation ---\n\n')
    run_evaluation(eval_data, features)
    del eval_data; gc.collect()

    print('\n\n--- Submission ---\n\n')
    run_submission(sub_data, features)


if __name__ == "__main__":
    main()
