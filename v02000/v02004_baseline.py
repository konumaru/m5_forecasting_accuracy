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


SEED = 42
VERSION = str(__file__).split('_')[0]
TARGET = 'sales'

MODEL_PATH = f'result/model/{VERSION}.pkl'
SCORE_PATH = f'result/score/{VERSION}.pkl'


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
        "quarter",
        "month",
        "week",
        "weekofyear",
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
        calendar[attr] = getattr(calendar['date'].dt, attr)
    calendar["is_weekend"] = calendar["dayofweek"].isin([5, 6]).astype(np.int8)

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
    label_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    for c in label_cols:
        encodaer = preprocessing.LabelEncoder()
        df[c] = encodaer.fit_transform(df[c])

    cat_cols = [
        'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
        'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2',
        'quarter', 'month', 'week', 'day', 'dayofweek', 'weekofyear'
    ]
    for c in cat_cols:
        df[c] = df[c].astype('category')

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


def ordered_d_cols(df_cols, is_reverse=False):
    return sorted(df_cols, key=lambda x: int((re.search(r"\d+", x)).group(0)), reverse=is_reverse)


@cache_result(filename='sample_weight', use_cache=True)
def calc_similar_weight():
    df = pd.read_pickle('features/melted_and_merged_train.pkl')
    # Prepare raw data.
    df = df[['id', 'd', 'sales', 'sell_price']]
    df['sales_value'] = df['sales'] * df['sell_price']
    df.drop(['sell_price'], axis=1, inplace=True)
    # Calculation salse value ratio.
    weight_df = df.pivot(values='sales_value', index='id', columns='d')
    weight_df = weight_df[ordered_d_cols(weight_df.columns)]

    weight_df = weight_df.shift(28, axis=1).rolling(28, axis=1).sum()
    weight_df = weight_df / weight_df.sum(axis=0)

    weight_df = weight_df.reset_index()
    weight_df = pd.melt(weight_df, id_vars='id', var_name='d', value_name='weight').fillna(0)
    # Calculation scale that is Variance of past values.
    scale_df = df.pivot(values='sales', index='id', columns='d')
    scale_df = scale_df[ordered_d_cols(scale_df.columns, is_reverse=False)]

    def est_scale(series):
        series = series[~np.isnan(series)][np.argmax(series != 0):]
        if series.shape[0] > 0:
            scale = np.mean(((series[1:] - series[:-1]) ** 2))
        else:
            scale = 1
        return scale
    scale_df = scale_df.rolling(90, min_periods=28, axis=1).apply(est_scale, raw=True)
    scale_df = scale_df.reset_index()
    scale_df = pd.melt(scale_df, id_vars='id', var_name='d', value_name='scale').fillna(0)
    # Merge weight_df and scale_df.
    weight_df = weight_df.merge(scale_df, how='left', on=['id', 'd'])
    weight_df['sample_weight'] = weight_df['weight'] / (weight_df['scale'].map(np.sqrt) + 1)
    # Min_Max_Scaling sample weight.
    weight_df['sample_weight'] = (weight_df['sample_weight'] - weight_df['sample_weight'].min()) \
        / weight_df['sample_weight'].max() - weight_df['sample_weight'].min()

    df = pd.merge(df, weight_df, how='left', on=['id', 'd'])
    return df[['sample_weight']].pipe(reduce_mem_usage)


@cache_result(filename='sales_features', use_cache=True)
def sales_features():
    pred_days = 28
    dst_df = pd.DataFrame()
    df = pd.read_pickle('features/melted_and_merged_train.pkl')
    grouped_df = df.groupby(["id"])['sales']

    for window in [7, 14, 28]:

        dst_df[f"sales_rolling_ZeroRatio_t{window}"] = grouped_df.transform(
            lambda x: 1 - (x == 0).shift(pred_days).rolling(window).mean())

        dst_df[f"sales_rolling_ZeroCount_t{window}"] = grouped_df.transform(
            lambda x: (x == 0).shift(pred_days).rolling(window).sum())

    # df = df[df[f"sales_rolling_ZeroRatio_t28"] < 0.25]
    return dst_df.pipe(reduce_mem_usage)


@cache_result(filename='all_train_data', use_cache=True)
def get_all_train_data():
    df = simple_fe()
    df = pd.concat([df, sales_features()], axis=1)
    return df


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
        weight_df['sample_weight'] = weight_df['weight']**2 / scale

        weight_max = weight_df['sample_weight'].max()
        weight_min = weight_df['sample_weight'].min()
        weight_df['sample_weight'] = \
            (weight_df['sample_weight'] - weight_min) / (weight_max - weight_min)

        return weight_df.loc[data_idx, 'sample_weight'].values


@cache_result(filename='evaluator', use_cache=False)
def get_evaluator(for_local=False):
    train_df = pd.read_pickle('../data/reduced/sales_train_validation.pkl')
    # If local test, drop sales data latest 28 days.
    if for_local:
        train_df = train_df.iloc[:, :-28]

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
    oldest_submit_date = datetime.datetime.strptime('2016-04-25', '%Y-%m-%d')
    submit_mask = (df["date"] >= oldest_submit_date)

    eval_date = oldest_submit_date - datetime.timedelta(days=eval_days)
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


def lgb_custom_fobj(preds, train_data):
    weight = train_data.get_weight()
    labels = train_data.get_label()

    grad = 2 * weight * (preds - labels)
    hess = 2 * weight
    return grad, hess


def lgb_wrmse(preds, train_data):
    weight = train_data.get_weight()
    labels = train_data.get_label()

    loss = np.sqrt(np.mean(2 * weight * np.power(preds - labels, 2)))
    return 'WRMSE', loss, False


def run_train(all_train_data, features):
    evaluator = load_pickle('features/evaluator.pkl')

    train_days = 365 * 2
    train_thresh = all_train_data['date'].max() - datetime.timedelta(days=train_days)
    all_train_data = all_train_data[all_train_data['date'] > train_thresh]

    train_data, valid_data = train_test_split(
        all_train_data, test_size=0.2, shuffle=False, random_state=SEED)

    train_set = lgb.Dataset(train_data[features], train_data[TARGET])
    valid_set = lgb.Dataset(valid_data[features], valid_data[TARGET], reference=train_set)

    use_weight = True
    if use_weight:
        train_weight = 2 * evaluator.get_sample_weight(train_data['id'])
        train_set.set_weight(train_weight)

        valid_weight = 2 * evaluator.get_sample_weight(valid_data['id'])
        valid_set.set_weight(valid_weight)

    params = {
        'boosting_type': 'gbdt',
        # 'objective': 'regression',  # regression, tweedie, poisson
        # 'tweedie_variance_power': 1.1,
        'metric': 'rmse',
        'subsample': 0.5,
        'subsample_freq': 1,
        'learning_rate': 0.03,  # 0.03
        'num_leaves': 2**11 - 1,
        'min_data_in_leaf': 2**12 - 1,
        'feature_fraction': 0.5,
        # 'max_bin': 100,
        'boost_from_average': False,
        'verbose': -1,
    }

    print(json.dumps(params, indent=4), '\n')

    train_params = {
        'num_boost_round': 2500,
        'early_stopping_rounds': 50,
        'verbose_eval': 100,
        # 'feval': lgb_wrmse,
        'fobj': lgb_custom_fobj
    }
    model = lgb.train(params, train_set, valid_sets=[train_set, valid_set], **train_params)
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
    scores['RMSE'] = mean_squared_error(val_pred, eval_data[TARGET], squared=False)

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
    df = get_all_train_data()
    print('\n', df.dtypes, '\n')

    print('\n\n--- Define Evaluation Object ---\n\n')
    _ = get_evaluator()

    print('\n\n--- Train Model ---\n\n')
    cols_to_drop = [
        'id', 'd', 'date', 'wm_yr_wk', 'weekday', 'year', 'sample_weight',
        'is_year_end', 'is_year_start', 'is_quarter_end', 'is_quarter_start',
        'is_month_end', 'is_month_start'
    ] + [TARGET]
    features = df.columns.tolist()
    features = [f for f in features if f not in cols_to_drop]

    all_train_data, eval_data, sub_data = train_eval_submit_split(df)
    del df; gc.collect()
    run_train(all_train_data, features)
    del all_train_data; gc.collect()

    print('\n\n--- Evaluation ---\n\n')
    run_evaluation(eval_data, features)
    del eval_data; gc.collect()

    print('\n\n--- Submission ---\n\n')
    run_submission(sub_data, features)


if __name__ == "__main__":
    main()
