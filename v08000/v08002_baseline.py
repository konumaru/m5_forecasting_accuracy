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
from script.model import LightGBM_Wrapper


# Define global Variables.
IS_TEST = True
SEED = 42
VERSION = str(__file__).split('_')[0]
TARGET = 'sales'
NUM_ITEMS = 30490

FEATURECOLS_PATH = f'result/feature_cols/{VERSION}.pkl'
EVALUATOR_PATH = 'features/evaluator.pkl'
MODEL_PATH = f'result/model/{VERSION}.pkl'
IMPORTANCE_PATH = f'result/importance/{VERSION}.png'
SCORE_PATH = f'result/score/{VERSION}.pkl'

GROUP_ID = ('store_id',)  # one version, one group
GROUPS = get_groups(GROUP_ID)

""" Transform
Input: raw_data
Output: melted_and_merged_train
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
    # Date Features
    calendar['date'] = pd.to_datetime(calendar['date'])
    attrs = [
        "year",
        "quarter",
        "month",
        "week",
        # "weekofyear",
        "day",
        "dayofweek",
        "dayofyear",
        # "is_year_end",
        # "is_year_start",
        # "is_quarter_end",
        # "is_quarter_start",
        "is_month_end",
        "is_month_start",
    ]

    for attr in attrs:
        calendar[attr] = getattr(calendar['date'].dt, attr)
    calendar["is_weekend"] = calendar["dayofweek"].isin([5, 6]).astype(np.bool)

    work_cals = {'CA': California(), 'TX': Texas(), 'WI': Wisconsin()}
    # 休日フラグを未来に向かって rolling.sum() する。
    holiday_df = pd.DataFrame({'date': pd.date_range(start='2011-01-29', end='2016-07-30')})
    for state, work_cal in work_cals.items():
        holiday_df[f'nwd_{state}'] = [int(work_cal.is_working_day(d)) for d in holiday_df.date]
    reversed_holiday_df = holiday_df.sort_values(by='date', ascending=False)
    for state in ['CA', 'TX', 'WI']:
        holiday_df = holiday_df.assign(**{
            f'nwd_{state}_rolling_t7': reversed_holiday_df[f'nwd_{state}'].rolling(
                7).sum().fillna(-1).astype(int),
            f'nwd_{state}_rolling_t14': reversed_holiday_df[f'nwd_{state}'].rolling(
                14).sum().fillna(-1).astype(int),
            f'nwd_{state}_rolling_t28': reversed_holiday_df[f'nwd_{state}'].rolling(
                28).sum().fillna(-1).astype(int)
        })

    calendar = calendar.merge(holiday_df, how='left', on='date')
    return calendar.pipe(reduce_mem_usage)


@cache_result(filename='parse_sell_prices', use_cache=True)
def parse_sell_prices():
    sell_prices = pd.read_pickle('../data/reduced/sell_prices.pkl')
    # Add Release Feature.
    groupd_df = sell_prices.groupby(['store_id', 'item_id'])
    sell_prices = sell_prices.assign(
        price_max=groupd_df['sell_price'].transform('max'),
        price_min=groupd_df['sell_price'].transform('min'),
        price_std=groupd_df['sell_price'].transform('std'),
        price_mean=groupd_df['sell_price'].transform('mean'),
        price_nunique=groupd_df['sell_price'].transform('nunique'),
        release=sell_prices['wm_yr_wk'] - groupd_df['wm_yr_wk'].transform('min'),
        id_nunique_by_price=sell_prices.groupby(
            ['store_id', 'sell_price'])['item_id'].transform('nunique'),
        price_float=sell_prices['sell_price'].map(lambda x: math.modf(x)[0]),
        price_int=sell_prices['sell_price'].map(lambda x: math.modf(x)[1]),
    )
    sell_prices['price_norm'] = sell_prices['sell_price'] / sell_prices['price_max']

    return sell_prices.pipe(reduce_mem_usage)


@cache_result(filename='parse_sales_train', use_cache=True)
def parse_sales_train():
    train = pd.read_pickle('../data/reduced/sales_train_evaluation.pkl')
    # Add Prediction Columns
    start_d = 1942
    end_d = 1969
    for i in range(start_d, end_d + 1):
        train[f'd_{i}'] = np.nan
    return train


@cache_result(filename='melted_and_merged_train', use_cache=True)
def melted_and_merged_train(n_row):
    # Load Data
    calendar = pd.read_pickle('features/parse_calendar.pkl')
    sell_prices = pd.read_pickle('features/parse_sell_prices.pkl')
    df = pd.read_pickle('features/parse_sales_train.pkl')
    # Melt and Merge
    idx_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    df = pd.melt(df, id_vars=idx_cols, var_name='d', value_name='sales')
    # df = df.iloc[-n_row:, :]    # Sampling Recently data
    df = pd.merge(df, calendar, how='left', on='d')
    df = pd.merge(df, sell_prices, how='left', on=['store_id', 'item_id', 'wm_yr_wk'])
    # d column change type to int
    df['d'] = df['d'].str.replace('d_', '').astype(int)
    # replace sales from 0 to nan.
    isnan_sell_price = df['sell_price'].isnull().values
    df.loc[isnan_sell_price, 'sales'] = np.nan
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

    df = df.reset_index(drop=True)
    return df.pipe(reduce_mem_usage)


def run_trainsform():
    # Transform each raw data.
    _ = parse_calendar()
    _ = parse_sell_prices()
    _ = parse_sales_train()
    # Melt and Merge all data.
    sample_rows = (365 * 3.5 + 28 * 2) * NUM_ITEMS
    _ = melted_and_merged_train(n_row=sample_rows)


""" Create Features
Input: melted_and_merged_train
Output: all_train_data, eval_data, submit_data
"""


@cache_result(filename='sales_lag_and_roll', use_cache=True)
def sales_lag_and_roll():
    # Define variables and dataframes.
    shift_days = 28
    use_cols = ['id', 'sales']
    srd_df = pd.read_pickle('features/melted_and_merged_train.pkl')[use_cols]
    # Creat Features
    agg_funcs = {}
    target = 'sales'
    grouped_df = srd_df.groupby('id')
    # lag features
    lags = list(range(15))
    for i in lags:
        lag_t = shift_days + i
        agg_funcs[f'sales_lag_t{i}'] = grouped_df[target].transform(lambda x: x.shift(lag_t))
    # rolling features
    num_shift = [1, 7, 14]
    num_roll = [7, 14, 30]
    for n_shift in num_shift:
        shift = shift_days + n_shift
        for n_roll in num_roll:
            agg_funcs[f'sales_roll_mean_t{n_shift}_{n_roll}'] = grouped_df[target].transform(
                lambda x: x.shift(shift).rolling(i).mean())
            agg_funcs[f'sales_roll_std_t{n_shift}_{n_roll}'] = grouped_df[target].transform(
                lambda x: x.shift(shift).rolling(i).std())
            # agg_funcs[f'sales_roll_mode_t{n_shift}_{n_roll}'] = grouped_df[target].transform(
            #     lambda x: x.shift(shift).rolling(i).apply(lambda x: x.mode()[0]))
            agg_funcs[f'sales_roll_min_t{n_shift}_{n_roll}'] = grouped_df[target].transform(
                lambda x: x.shift(shift).rolling(i).min())
            agg_funcs[f'sales_roll_max_t{n_shift}_{n_roll}'] = grouped_df[target].transform(
                lambda x: x.shift(shift).rolling(i).max())
    # cunsum sales values
    # agg_funcs['sales_cumsum_values'] = grouped_df[target].transform(lambda x: x.cumsum())
    # agg_funcs['sales_cummax_values'] = grouped_df[target].transform(lambda x: x.cummax())
    # is_zero, is_non_zero features
    for i in [7, 14, 30]:
        agg_funcs[f'sales_rolling_ZeroRatio_t{i}'] = grouped_df[target].transform(
            lambda x: (x == 0).shift(shift_days).rolling(i).mean())
        agg_funcs[f'sales_rolling_ZeroCount_t{i}'] = grouped_df[target].transform(
            lambda x: (x == 0).shift(shift_days).rolling(i).sum())
        agg_funcs[f'sales_rolling_NonZeroRatio_t{i}'] = grouped_df[target].transform(
            lambda x: (x != 0).shift(shift_days).rolling(i).mean())
        agg_funcs[f'sales_rolling_NonZeroCount_t{i}'] = grouped_df[target].transform(
            lambda x: (x != 0).shift(shift_days).rolling(i).sum())

    agg_funcs['sales_rolling_skew_t30'] = grouped_df[target].transform(
        lambda x: x.shift(shift_days).rolling(30).skew())
    agg_funcs['sales_rolling_kurt_t30'] = grouped_df[target].transform(
        lambda x: x.shift(shift_days).rolling(30).kurt())

    dst_df = pd.DataFrame()
    dst_df = dst_df.assign(**agg_funcs)
    return dst_df.pipe(reduce_mem_usage)


@cache_result(filename='total_sales_lag_and_roll', use_cache=True)
def total_sales_lag_and_roll():
    # Define variables and dataframes.
    shift_days = 28
    use_cols = ['id', 'sales', 'sell_price']
    srd_df = pd.read_pickle('features/melted_and_merged_train.pkl')[use_cols]
    srd_df['total_sales'] = srd_df['sales'] * srd_df['sell_price']
    grouped_df = srd_df.groupby('id')
    dst_df = pd.DataFrame()
    # Creat Features
    agg_funcs = {}
    target = 'total_sales'

    lags = list(range(15))
    for i in lags:
        lag_t = shift_days + i
        agg_funcs[f'{target}_lag_t{i}'] = grouped_df[target].transform(lambda x: x.shift(lag_t))

    num_shift = [1, 7, 14]
    num_roll = [7, 14, 30]
    for n_shift in num_shift:
        shift = shift_days + n_shift
        for n_roll in num_roll:
            agg_funcs[f'{target}_roll_mean_t{n_shift}_{n_roll}'] = grouped_df[target].transform(
                lambda x: x.shift(shift).rolling(i).mean())
            agg_funcs[f'{target}_roll_std_t{n_shift}_{n_roll}'] = grouped_df[target].transform(
                lambda x: x.shift(shift).rolling(i).std())

    dst_df = dst_df.assign(**agg_funcs)
    return dst_df.pipe(reduce_mem_usage)


@cache_result(filename='price_simple_feature', use_cache=True)
def price_simple_feature():
    use_cols = ['id', 'sell_price', 'month']
    srd_df = pd.read_pickle('features/melted_and_merged_train.pkl')[use_cols]
    dst_df = pd.DataFrame()
    grouped_df = srd_df.groupby('id')

    dst_df['price_momentum'] = srd_df['sell_price'] /\
        srd_df.groupby('id')['sell_price'].transform(lambda x: x.shift(28))
    dst_df['price_momentum_m'] = srd_df['sell_price'] /\
        srd_df.groupby(['id', 'month'])['sell_price'].transform('mean')

    dst_df = dst_df.reset_index(drop=True)
    return dst_df.pipe(reduce_mem_usage)


@cache_result(filename='days_from_last_sales', use_cache=True)
def days_from_last_sales():
    # Define variables and dataframes.
    shift_days = 28
    use_cols = ['id', 'd', 'sales']
    srd_df = pd.read_pickle('features/melted_and_merged_train.pkl')[use_cols]
    # Convert target to binary
    srd_df['non_zero'] = (srd_df['sales'] > 0)
    srd_df = srd_df.assign(
        non_zero_lag=srd_df.groupby(['id'])['non_zero'].transform(
            lambda x: x.shift(28).rolling(2000, 1).sum()).fillna(-1)
    )

    temp_df = srd_df[['id', 'd', 'non_zero_lag']].drop_duplicates(subset=['id', 'non_zero_lag'])
    temp_df.columns = ['id', 'd_min', 'non_zero_lag']

    srd_df = srd_df.merge(temp_df, on=['id', 'non_zero_lag'], how='left')
    srd_df['days_from_last_sales'] = srd_df['d'] - srd_df['d_min']
    return srd_df[['days_from_last_sales']]


@cache_result(filename='simple_target_encoding', use_cache=True)
def simple_target_encoding():
    # Define variables and dataframes.
    shift_days = 28
    use_cols = [
        'id', 'd', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
        'sales', 'dayofweek', 'week', 'day', 'nwd_CA', 'nwd_TX', 'nwd_WI',
        'is_weekend'
    ]
    df = pd.read_pickle('features/melted_and_merged_train.pkl')[use_cols]

    # df['sales_times_price'] = df['sales'] * df['sell_price']
    # Convert target to binary
    target = 'sales'
    icols = [
        # id groups.
        ['state_id'],
        ['store_id'],
        ['cat_id'],
        ['dept_id'],
        ['state_id', 'cat_id'],
        ['state_id', 'dept_id'],
        ['store_id', 'cat_id'],
        ['store_id', 'dept_id'],
        ['item_id'],
        ['item_id', 'state_id'],
        ['item_id', 'store_id'],
        # datetime groups.
        ['store_id', 'dayofweek'],
        ['dept_id', 'dayofweek'],
        ['item_id', 'dayofweek'],
        ['store_id', 'dept_id', 'dayofweek'],
        ['store_id', 'dept_id', 'week'],
        ['store_id', 'dept_id', 'day'],
        ['store_id', 'item_id', 'dayofweek'],
        ['store_id', 'item_id', 'week'],
        ['store_id', 'item_id', 'day'],
        # holiday
        ['store_id', 'item_id', 'nwd_CA'],
        ['store_id', 'item_id', 'nwd_TX'],
        ['store_id', 'item_id', 'nwd_WI'],
        ['store_id', 'item_id', 'is_weekend'],
    ]

    features = []
    print('Target', target)
    for col in icols:
        print('Encoding', col)
        col_name = '_'.join(col)
        features.extend([f'enc_{target}_mean_by_{col_name}', f'enc_{target}_std_by_{col_name}'])

        df[f'enc_{target}_mean_by_{col_name}'] = df.groupby(col)[target].transform('mean')
        df[f'enc_{target}_std_by_{col_name}'] = df.groupby(col)[target].transform('std')
        df[f'enc_{target}_is_over_mean_by_{col_name}'] = (df.groupby(
            'id')[target].transform('mean') > df.groupby(col)[target].transform('mean'))

    df = df.loc[:, features].reset_index(drop=True)
    return df.pipe(reduce_mem_usage)


@cache_result(filename='simple_total_sales_encoding', use_cache=True)
def simple_total_sales_encoding():
    # Define variables and dataframes.
    shift_days = 28
    use_cols = [
        'id', 'd', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
        'sales', 'sell_price', 'dayofweek'
    ]
    df = pd.read_pickle('features/melted_and_merged_train.pkl')[use_cols]
    df['sales_times_price'] = df['sales'] * df['sell_price']
    # Convert target to binary
    target = 'sales_times_price'
    icols = [
        # id groups.
        ['state_id'],
        ['store_id'],
        ['cat_id'],
        ['dept_id'],
        ['state_id', 'cat_id'],
        ['state_id', 'dept_id'],
        ['store_id', 'cat_id'],
        ['store_id', 'dept_id'],
        ['item_id'],
        ['item_id', 'state_id'],
        ['item_id', 'store_id'],
        # datetime groups.
        ['store_id', 'dayofweek'],
        ['dept_id', 'dayofweek'],
        ['item_id', 'dayofweek'],
        ['store_id', 'dept_id', 'dayofweek'],
        ['store_id', 'item_id', 'dayofweek']
    ]

    features = []
    print('Target', target)
    for col in icols:
        print('Encoding', col)
        col_name = '_'.join(col)
        features.extend([f'enc_{target}_mean_by_{col_name}', f'enc_{target}_std_by_{col_name}'])

        df[f'enc_{target}_mean_by_{col_name}'] = df.groupby(col)[target].transform('mean')
        df[f'enc_{target}_std_by_{col_name}'] = df.groupby(col)[target].transform('std')
        df[f'enc_{target}_is_over_mean_by_{col_name}'] = (df.groupby(
            'id')[target].transform('mean') > df.groupby(col)[target].transform('mean'))

    df = df.loc[:, features].reset_index(drop=True)
    return df.pipe(reduce_mem_usage)


@cache_result(filename='hierarchical_bayesian_target_encoding', use_cache=True)
def hierarchical_bayesian_target_encoding():
    use_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'sales']
    df = pd.read_pickle('features/melted_and_merged_train.pkl')[use_cols]

    groups_and_priors = {
        # singe encodings
        ("state_id",): None,
        ("store_id",): None,
        ("cat_id",): None,
        ("dept_id",): None,
        ("item_id",): None,

        # second-order interactions
        ("state_id", "dept_id"): ["gte_state_id", "gte_dept_id"],
        ("state_id", "item_id"): ["gte_state_id", "gte_item_id"],
        ("store_id", "dept_id"): ["gte_store_id", "gte_dept_id"],
        ("store_id", "item_id"): ["gte_store_id", "gte_item_id"],
    }

    for agg_f in ['mean', 'var']:
        features = []
        for group_cols, prior_cols in groups_and_priors.items():
            features.append(f"gte_{'_'.join(group_cols)}")
            print(f'Add {features[-1]}')

            gte = GaussianTargetEncoder(list(group_cols), "sales", prior_cols)
            df[features[-1]] = gte.fit_transform(df, prior_precision=100, stat_type=agg_f)

        rename_dict = {feat: f'{feat}_{agg_f.upper()}' for feat in features}
        df = df.rename(columns=rename_dict)

    features = df.columns[df.columns.str.startswith('gte')]
    df = df.loc[:, features].reset_index(drop=True)
    return df.pipe(reduce_mem_usage)


@cache_result(filename='all_data', use_cache=True)
def run_create_features():
    # Create Feature
    df = pd.read_pickle('features/melted_and_merged_train.pkl')
    feat_funcs = [
        sales_lag_and_roll,
        total_sales_lag_and_roll,
        price_simple_feature,
        days_from_last_sales,
        simple_target_encoding,
        simple_total_sales_encoding,
        hierarchical_bayesian_target_encoding
    ]
    # Join All Features.
    for f_func in feat_funcs:
        temp_feat_df = f_func()
        df = pd.concat([df, temp_feat_df], axis=1)
        del temp_feat_df; gc.collect()

    sample_rows = (365 * 3 + 28 * 2) * NUM_ITEMS
    df = df.iloc[-sample_rows:, :]    # Sampling Recently data
    df['sales'].fillna(0, inplace=True)

    numeric_cols = df.select_dtypes(include=['float16']).columns
    df = df.assign(**{num_c: df[num_c].fillna(-999) for num_c in numeric_cols})
    # Export Data
    return df


""" Split Data
Input: all data
Output: all_train_data, eval_data, submit_data

# MEMO
- 新しいdataが使えるようになったらeval_dataの期間を設定し直す。
"""


def dump_by_groups(df: pd.DataFrame, data_name: str, gruops: list):
    for i, g_id in enumerate(gruops):
        print(f'{i+1}/{len(gruops)}', end=', ')
        filepath = f"features/{data_name}_{'_'.join(g_id)}.pkl"
        query = ''.join([f'(?=.*{i})' for i in g_id])
        isin_group = df['id'].str.contains(f'^{query}.*$')
        dump_pickle(df[isin_group], filepath)


def run_split_data():
    df = pd.read_pickle('features/all_data.pkl')
    print(df.info(verbose=True), '\n')

    print(f'Dump Train Data.')
    upper_d = 1913 if IS_TEST else 1941  # Submision時は検証データも学習する
    train_mask = (df['d'] <= upper_d)
    dump_by_groups(df[train_mask], 'train_data', GROUPS)

    print(f'Dump Evaluation Data.')
    eval_data_path = 'features/eval_data.pkl'
    lower_d = 1913
    upper_d = 1941
    eval_mask = (df['d'] > lower_d) & (df['d'] <= upper_d)
    dump_pickle(df[eval_mask], eval_data_path)

    print(f'Dump Submisssion Data.')
    submit_data_path = 'features/submit_data.pkl'
    print(f'Split submit_data to {submit_data_path}')
    lower_d = 1913
    submit_mask = (df['d'] > lower_d)
    dump_pickle(df[submit_mask], submit_data_path)

    return None


""" Define Evaluator
"""


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

    def lgb_custom_fobj(self, preds, dtrain):
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


""" Train
Input: all train data
Output: models
"""


def train_valid_split(df, go_back_days=28):
    valid_duration = 28

    min_thresh_date = df['date'].max() - datetime.timedelta(days=go_back_days)
    max_thresh_date = min_thresh_date + datetime.timedelta(days=valid_duration)

    valid_mask = ((df["date"] > min_thresh_date) & (df["date"] <= max_thresh_date))
    train_mask = (df["date"] <= min_thresh_date)
    return df[train_mask], df[valid_mask]


def get_decayed_weights(df, days_keep_weight=0):
    max_d = df['d'].max()

    weight_map = np.flip(np.concatenate([
        [1 for _ in range(days_keep_weight)],
        [1 - 0.0005 * i for i in range(days_keep_weight, max_d)]
    ], axis=0)).round(4)
    weight_map = {i + 1: weight_map[i] for i in range(max_d)}
    weight = df['d'].map(weight_map).values
    return weight


def train_group_models(seed=SEED):
    evaluator = load_pickle(EVALUATOR_PATH)

    params = {
        'model_params': {
            'boosting': 'gbdt',
            'objective': 'tweedie',  # tweedie, poisson, regression
            'tweedie_variance_power': 1,  # 1.0=poisson
            'metric': 'custom',
            'num_leaves': 2**7 - 1,
            'min_data_in_leaf': 50,
            'seed': seed,
            'learning_rate': 0.03,  # 0.1
            'subsample': 0.5,  # ~v05006, 0.8
            'subsample_freq': 1,
            'feature_fraction': 0.5,  # ~v05006, 0.8
            # 'lambda_l1': 0.1, # v06012, 0.554 -> 0.561
            'lambda_l2': 0.1,  # v06012, 0.554 -> 0.555
            'max_bin': 100,  # Score did not change.
            'force_row_wise': True,
            'verbose': -1
        },
        'train_params': {
            'num_boost_round': 1500,  # 2000
            'early_stopping_rounds': 100,
            'verbose_eval': 100,
            'feval': evaluator.lgb_custom_feval
        }
    }
    print('\nParameters:\n', json.dumps(params['model_params'], indent=4), '\n')

    group_models = {}
    for i, g_id in enumerate(GROUPS):
        print(f'\n\nGroup ID: {g_id}, {i+1}/{len(GROUPS)}')
        df = pd.read_pickle(f"features/train_data_{'_'.join(g_id)}.pkl")

        drop_cols = ['id', 'd', 'sales', 'date', 'wm_yr_wk']
        features = [f for f in df.columns if f not in drop_cols]
        dump_pickle(features, FEATURECOLS_PATH)
        # Split Train data.
        train_data, valid_data = train_valid_split(df, go_back_days=28)
        del df; gc.collect()
        query = '''
            release > 30 \
            & days_from_last_sales < 7
        '''
        train_data = train_data.query(query)

        use_weight = True
        if use_weight:
            weight, scale = evaluator.get_series_weight(train_data['id'])
            wrmsse_weight = 10000 * weight / np.log1p(scale)

            weight_decayed = get_decayed_weights(train_data, 90)
            pastNonZeroRatio = train_data['sales_rolling_NonZeroRatio_t30'].values
            sample_weight = weight_decayed + pastNonZeroRatio

            train_weight = wrmsse_weight * sample_weight

        is_set_feval_weight = True
        if is_set_feval_weight:
            evaluator.set_feval_weight(valid_data['id'].drop_duplicates(keep='last'))

        X_train, y_train = train_data[features], train_data['sales']
        X_valid, y_valid = valid_data[features], valid_data['sales']
        del train_data, valid_data; gc.collect()

        if '_'.join(g_id) in ['HOBBIES', 'HOUSEHOLD_1']:
            params['model_params']['learning_rate'] = 0.01

        model = LightGBM_Wrapper()
        model.fit(
            params['model_params'],
            params['train_params'],
            X_train,
            y_train,
            X_valid,
            y_valid,
            train_weight=train_weight,
            valid_weight=None
        )
        # Save Importance
        IMPORTANCE_PATH = f'result/importance/{VERSION}/{seed}/{g_id}.png'
        os.makedirs(f'result/importance/{VERSION}/{seed}', exist_ok=True)
        model.save_importance(filepath=IMPORTANCE_PATH, max_num_features=80, figsize=(25, 30))
        # Add Model
        group_models[''.join(g_id)] = model
    return group_models


def run_train(seed=SEED):
    models = train_group_models(seed)
    print('')
    dump_pickle(models, MODEL_PATH)


""" Evaluation

Input: eval_data, models
Output: validation scores
"""


def run_evaluation(seed=SEED):
    scores = {}
    lgb_models = load_pickle(MODEL_PATH)
    evaluator = load_pickle(EVALUATOR_PATH)
    features = load_pickle(FEATURECOLS_PATH)
    eval_data = load_pickle('features/eval_data.pkl')
    valid_pred = np.zeros(eval_data.shape[0])

    for g_id in GROUPS:
        query = ''.join([f'(?=.*{i})' for i in g_id])
        is_groups = eval_data['id'].str.contains(f'^{query}.*$')
        valid_pred[is_groups] = lgb_models[''.join(g_id)].predict(
            eval_data.loc[is_groups, features])

    scores['RMSE'] = mean_squared_error(valid_pred, eval_data[TARGET], squared=False)
    scores['WRMSSE'] = evaluator.score(valid_pred.reshape(-1, NUM_ITEMS).T)

    print('')
    for f_name, score in scores.items():
        print(f'Our val {f_name} score is {score}')
    # Export Score and evaluation data
    with open(f'result/score/{VERSION}.json', 'w') as f:
        json.dump(scores, f, indent=4)

    os.makedirs(f'result/evaluation/{VERSION}', exist_ok=True)
    dump_filepath = f'result/evaluation/{VERSION}/{seed}_{scores["WRMSSE"]:.05}.csv.gz'
    eval_df = pd.DataFrame({
        'id': eval_data['id'].values,
        'pred': valid_pred
    })
    eval_df.to_csv(dump_filepath, index=False, compression='gzip')


""" Submission
Input: submit_data, models
Output: submision file
"""


def run_submission(seed=SEED):
    # Load Data.
    lgb_models = load_pickle(MODEL_PATH)
    features = load_pickle(FEATURECOLS_PATH)
    submit_data = load_pickle('features/submit_data.pkl')
    submit_pred = np.zeros(submit_data.shape[0])

    # Prediction and Transform Submission Data.
    for g_id in GROUPS:
        query = ''.join([f'(?=.*{i})' for i in g_id])
        is_groups = submit_data['id'].str.contains(f'^{query}.*$')
        submit_pred[is_groups] = lgb_models[''.join(g_id)].predict(
            submit_data.loc[is_groups, features])

    submit_data['sales'] = submit_pred
    submission = pd.pivot(submit_data, index='id', columns='d', values='sales').reset_index()

    valid_sub = submission[['id'] + [i for i in range(1914, 1942)]]
    valid_sub.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]
    valid_sub = valid_sub.assign(id=lambda x: x['id'].str.replace('evaluation', 'validation'))

    eval_sub = submission[['id'] + [i for i in range(1942, 1970)]]
    eval_sub.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]

    submission = pd.concat([valid_sub, eval_sub], axis=0)

    # Dump Submission file.
    with open(f'result/score/{VERSION}.json', 'r') as f:
        scores = json.load(f)['WRMSSE']
    sample_submission = pd.read_pickle('../data/reduced/sample_submission.pkl')
    submission = sample_submission[['id']].merge(submission, how='left', on='id')
    os.makedirs(f'submit/{VERSION}/', exist_ok=True)
    submission.to_csv(
        f'submit/{VERSION}/{seed}_{scores:.05}.csv.gz',
        index=False,
        compression='gzip'
    )

    print(submission.shape)
    print(submission.head())


# ==================================================================


def main():
    print('\n\n--- Transform ---\n\n')
    run_trainsform()

    print('\n\n--- Create Features ---\n\n')
    run_create_features()

    print('\n\n--- Split Data ---\n\n')
    run_split_data()

    print('\n\n--- Define Evaluator ---\n\n')
    _ = get_evaluator()

    seed = SEED
    n_fold = 3
    for i in range(n_fold):
        seed += i
        print('#' * 25)
        print('#' * 5, f'Fold {i+1} / {n_fold}')
        print('#' * 25)

        print('\n\n--- Train ---\n\n')
        run_train(seed)

        print('\n\n--- Evaluation ---\n\n')
        run_evaluation(seed)

        print('\n\n--- Submission ---\n\n')
        run_submission(seed)


if __name__ == '__main__':
    main()
