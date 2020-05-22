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
CATEGORICALS_PATH = f'result/categoricals/{VERSION}.pkl'
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
    # Date Features
    calendar['date'] = pd.to_datetime(calendar['date'])
    attrs = [
        "year",
        "quarter",
        "month",
        "week",
        "weekofyear",
        "day",
        "dayofweek",
        "dayofyear",
        "is_year_end",
        "is_year_start",
        "is_quarter_end",
        "is_quarter_start",
        "is_month_end",
        "is_month_start",
    ]

    for attr in attrs:
        calendar[attr] = getattr(calendar['date'].dt, attr)
    calendar["is_weekend"] = calendar["dayofweek"].isin([5, 6]).astype(np.bool)

    return calendar.pipe(reduce_mem_usage)


@cache_result(filename='parse_sell_prices', use_cache=True)
def parse_sell_prices():
    sell_prices = pd.read_pickle('../data/reduced/sell_prices.pkl')
    # Add Release Feature.
    groupd_df = sell_prices.groupby(['store_id', 'item_id'])
    sell_prices = sell_prices.assign(
        release=sell_prices['wm_yr_wk'] - groupd_df['wm_yr_wk'].transform('min'),
    )
    sell_prices = sell_prices.assign(
        price_max=groupd_df['sell_price'].transform('max'),
        price_min=groupd_df['sell_price'].transform('min'),
        price_std=groupd_df['sell_price'].transform('std'),
        price_mean=groupd_df['sell_price'].transform('mean'),
        price_nunique=groupd_df['sell_price'].transform('nunique'),
        id_nunique_by_price=sell_prices.groupby(
            ['store_id', 'sell_price'])['item_id'].transform('nunique'),
    )
    sell_prices['price_norm'] = sell_prices['sell_price'] / sell_prices['price_max']

    return sell_prices.pipe(reduce_mem_usage)


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

    df = df.reset_index(drop=True)
    return df.pipe(reduce_mem_usage)


""" Feature Engineering
"""
@cache_result(filename='sales_lag_and_roll', use_cache=True)
def sales_lag_and_roll():
    # Define variables and dataframes.
    target = TARGET
    shift_days = 28
    use_cols = ['id', 'sales']
    srd_df = pd.read_pickle('features/melted_and_merged_train.pkl')[use_cols]
    dst_df = pd.DataFrame()
    # Creat Features
    agg_funcs = {}
    grouped_df = srd_df.groupby('id')

    lag_list = [7, 14]  # list(range(15))
    for i in lag_list:
        lag_t = shift_days + i
        agg_funcs[f'{target}_lag_t{shift_days}p{i}'] = grouped_df[target].transform(
            lambda x: x.shift(lag_t - 1))

    roll_values = [14, 30]  # [7, 14, 30, 60]
    for i in roll_values:
        agg_funcs[f'{target}_roll_mean_t{i}'] = grouped_df[target].transform(
            lambda x: x.shift(shift_days).rolling(i).mean())
        agg_funcs[f'{target}_roll_std_t{i}'] = grouped_df[target].transform(
            lambda x: x.shift(shift_days).rolling(i).std())
        agg_funcs[f'{target}_rolling_ZeroRatio_t{i}'] = grouped_df[target].transform(
            lambda x: 1 - (x == 0).shift(shift_days).rolling(i).mean())
        agg_funcs[f'{target}_rolling_ZeroCount_t{i}'] = grouped_df[target].transform(
            lambda x: (x == 0).shift(shift_days).rolling(i).sum())

    agg_funcs['sales_rolling_skew_t30'] = grouped_df[target].transform(
        lambda x: x.shift(shift_days).rolling(30).skew())
    agg_funcs['sales_rolling_kurt_t30'] = grouped_df[target].transform(
        lambda x: x.shift(shift_days).rolling(30).kurt())

    dst_df = dst_df.assign(**agg_funcs)
    dst_df = dst_df.reset_index(drop=True)
    return dst_df.pipe(reduce_mem_usage)


@cache_result(filename='price_simple_feature', use_cache=True)
def price_simple_feature():
    use_cols = ['id', 'sell_price', 'month']
    srd_df = pd.read_pickle('features/melted_and_merged_train.pkl')[use_cols]
    dst_df = pd.DataFrame()
    grouped_df = srd_df.groupby('id')

    dst_df['price_momentum'] = srd_df['sell_price'] / srd_df.groupby(
        'id')['sell_price'].transform(lambda x: x.shift(28))
    dst_df['price_momentum_m'] = srd_df['sell_price'] \
        / srd_df.groupby(['id', 'month'])['sell_price'].transform('mean')

    dst_df = dst_df.reset_index(drop=True)
    return dst_df.pipe(reduce_mem_usage)


@cache_result(filename='days_from_last_sales', use_cache=True)
def days_from_last_sales():
    # Define variables and dataframes.
    target = TARGET
    shift_days = 28
    use_cols = ['id', 'd', 'sales']
    srd_df = pd.read_pickle('features/melted_and_merged_train.pkl')[use_cols]
    dst_df = pd.DataFrame()
    # Convert target to binary
    dst_df['id'] = srd_df['id']
    dst_df['non_zero'] = (srd_df[TARGET] > 0)
    # Make lags to prevent any leakage
    dst_df = dst_df.assign(
        d=srd_df['d'].str.replace('d_', '').astype(int),
        non_zero_lag=dst_df.groupby(['id'])['non_zero'].transform(
            lambda x: x.shift(28).rolling(2000, 1).sum()).fillna(-1)
    )

    temp_df = dst_df[['id', 'd', 'non_zero_lag']].drop_duplicates(subset=['id', 'non_zero_lag'])
    temp_df.columns = ['id', 'd_min', 'non_zero_lag']

    dst_df = dst_df.merge(temp_df, on=['id', 'non_zero_lag'], how='left')
    dst_df['days_from_last_sales'] = dst_df['d'] - dst_df['d_min']
    return dst_df[['days_from_last_sales']].pipe(reduce_mem_usage)


def get_all_features():
    df = pd.read_pickle('features/melted_and_merged_train.pkl')

    feat_funcs = [
        sales_lag_and_roll,
        price_simple_feature,
        days_from_last_sales
    ]

    for f_func in feat_funcs:
        temp_feat_df = f_func()
        df = pd.concat([df, temp_feat_df], axis=1)
        del temp_feat_df; gc.collect()

    # numeric_cols = df.select_dtypes(include=['number']).columns
    # df = df.assign(**{num_c: df[num_c].fillna(-999) for num_c in numeric_cols})
    return df


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

    def get_series_weight(self, data_idx):
        data_idx = data_idx.apply(lambda x: x.rsplit('_', 1)[0]).values

        weight_df = self.weights * 12
        weight_df.index = weight_df.index.str.replace('--', '_')
        weight_df.columns = ['weight']
        weight_df['scale'] = np.where(self.scale != 0, self.scale, 1)

        fobj_weight = weight_df.loc[data_idx, 'weight'].values
        fojb_sclae = weight_df.loc[data_idx, 'scale'].values

        return fobj_weight, fojb_sclae

    def set_series_weight_for_fobj(self, train_idx):
        fobj_weight, fojb_sclae = self.get_series_weight(train_idx)
        self.custom_jobj_weight = 2 * np.power(fobj_weight, 2) / fojb_sclae

    def custom_fobj(self, preds, dtrain):
        actual = dtrain.get_label()
        weight = self.custom_jobj_weight

        grad = weight * (preds - actual)
        hess = weight
        return grad, hess


@cache_result(filename='evaluator', use_cache=True)
def get_evaluator(go_back_days=28):
    pred_days = 28
    end_thresh = (-go_back_days + pred_days) if (-go_back_days + pred_days) != 0 else None
    train_df = pd.read_pickle('../data/reduced/sales_train_validation.pkl')
    train_fold_df = train_df.iloc[:, :-go_back_days]
    valid_fold_df = train_df.iloc[:, -go_back_days:end_thresh].copy()

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


@cache_result(filename='train_decayed_weights', use_cache=True)
def get_decayed_weights(train_data, keep_weight_days=0):
    max_train_d = train_data['d'].str.replace('d_', '').astype('int').max()
    weight_map = np.concatenate([
        [1 for _ in range(keep_weight_days)],
        [1 - 0.0005 * i for i in range(keep_weight_days, max_train_d)]
    ], axis=0)
    weight_map = {f'd_{d+1}': w for d, w in zip(list(range(max_train_d)), weight_map)}
    weight = train_data['d'].map(weight_map).values
    return weight


class LGBM_Model():

    def __init__(
        self, params, train_param, train_data, valid_data,
        target_col, feature_cols, categorical_cols=None, train_weight=None, valid_weight=None
    ):
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.categorical_cols = categorical_cols
        self.train_weight = train_weight
        self.valid_weight = valid_weight

        model = self.fit(params, train_param, train_data, valid_data)

    def _remove_bin_file(self, filepath):
        if os.path.exists(filepath):
            os.remove(filepath)

    def _convert_dataset(self, train_data, valid_data, save_binary=True):
        train_dataset = lgb.Dataset(
            train_data[self.feature_cols], train_data[self.target_col], weight=self.train_weight,
            feature_name=self.feature_cols, categorical_feature=self.categorical_cols
        )
        valid_dataset = lgb.Dataset(
            valid_data[self.feature_cols], valid_data[self.target_col], weight=self.valid_weight,
            reference=train_dataset
        )

        if save_binary:
            # Define cache filepath.
            train_bin_path = 'tmp_train_set.bin'
            valid_bin_path = 'tmp_valid_set.bin'
            # Remove Cached File.
            self._remove_bin_file(train_bin_path)
            self._remove_bin_file(valid_bin_path)
            # Save Binary Cache.
            train_dataset.save_binary(train_bin_path)
            valid_dataset.save_binary(valid_bin_path)
            # Reload Binary Cache.
            train_dataset = lgb.Dataset(train_bin_path)
            valid_dataset = lgb.Dataset(valid_bin_path)

        return train_dataset, valid_dataset

    def fit(self, params, train_param, train_data, valid_data):
        train_dataset, valid_dataset = self._convert_dataset(train_data, valid_data)

        self.model = lgb.train(
            params, train_dataset,
            valid_sets=[valid_dataset],
            **train_param
        )

    def predict(self, data):
        model = self.model
        return model.predict(data, num_iteration=model.best_iteration)

    def get_importance(self):
        # Define Feature Importance DataFrame.
        imp_df = pd.DataFrame(
            [self.model.feature_importance()],
            columns=self.model.feature_name(),
            index=['Importance']
        ).T
        imp_df.sort_values(by='Importance', inplace=True)
        return imp_df

    def save_importance(self, filepath, max_num_features=50, figsize=(18, 25)):
        imp_df = self.get_importance()
        # Plot Importance DataFrame.
        plt.figure(figsize=figsize)
        imp_df[-max_num_features:].plot(
            kind='barh', title='Feature importance', figsize=figsize,
            y='Importance', align="center"
        )
        plt.savefig(filepath)
        plt.close('all')


def run_train():
    go_back_days = 28
    evaluator = get_evaluator(go_back_days)
    df = load_pickle('features/all_train_data.pkl')
    # Split Train data.
    train_data, valid_data = train_valid_split(df, go_back_days)
    dump_pickle(valid_data, './features/valid_data.pkl')
    del df; gc.collect()

    drop_cols = ['id', 'd', 'sales', 'date', 'wm_yr_wk']
    features = train_data.columns.tolist()
    features = [f for f in features if f not in drop_cols]
    categoricals = train_data.dtypes[train_data.dtypes == 'category'].index.tolist()
    dump_pickle(features, FEATURECOLS_PATH)
    dump_pickle(categoricals, CATEGORICALS_PATH)

    params = {
        'model_params': {
            'boosting': 'gbdt',
            'objective': 'tweedie',  # tweedie, poisson
            'tweedie_variance_power': 1.1,  # 1.0=poisson
            'metric': 'custom',
            'num_leaves': 2**7 - 1,
            'min_data_in_leaf': 25,
            'seed': SEED,
            'learning_rate': 0.075,  # 0.1
            'subsample': 0.5,
            'subsample_freq': 1,
            'feature_fraction': 0.8,
            'force_row_wise': True,
            'verbose': -1,
            'num_threads': 2
        },
        'train_params': {
            'num_boost_round': 1500,  # 2000
            'early_stopping_rounds': 100,
            'verbose_eval': 100,
            # 'fobj': evaluator.custom_fobj,
            'feval': evaluator.custom_feval,
        }
    }
    print('\nParameters:\n', json.dumps(params['model_params'], indent=4), '\n')

    lgb_model = LGBM_Model(
        params['model_params'],
        params['train_params'],
        train_data,
        valid_data,
        TARGET,
        features,
        categorical_cols=categoricals,
        train_weight=get_decayed_weights(train_data, keep_weight_days=90),
        valid_weight=None
    )

    dump_pickle(lgb_model, MODEL_PATH)
    lgb_model.save_importance(filepath=IMPORTANCE_PATH, figsize=(18, 25))

    print('\nEvaluation:')
    valid_data = load_pickle('./features/valid_data.pkl')
    valid_pred = lgb_model.predict(valid_data[features])
    scores = {}
    scores['RMSE'] = mean_squared_error(valid_pred, valid_data[TARGET], squared=False)
    scores['WRMSSE'] = evaluator.score(valid_pred.reshape(-1, NUM_ITEMS).T)
    for f_name, score in scores.items():
        print(f'Our val {f_name} score is {score}')

    dump_pickle(scores, SCORE_PATH)
    return lgb_model


""" Submission
"""


def run_submission():
    lgb_model = load_pickle(MODEL_PATH)
    scores = load_pickle(SCORE_PATH)['WRMSSE']
    features = load_pickle(FEATURECOLS_PATH)
    submit_data = load_pickle('features/submit_data.pkl')

    submit_data['sales'] = lgb_model.predict(submit_data[features])
    submission = pd.pivot(submit_data, index='id', columns='d', values='sales').reset_index()
    # Processing for validation.
    valid_sub = submission[['id'] + [f'd_{i}' for i in range(1914, 1942)]]
    valid_sub.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]
    # Processing for evaluation.
    eval_sub = submission[['id'] + [f'd_{i}' for i in range(1942, 1970)]]
    eval_sub.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]
    eval_sub = eval_sub.assign(id=lambda x: x['id'].str.replace('validation', 'evaluation'))
    # Dump Submission file.
    submission = pd.concat([valid_sub, eval_sub], axis=0)
    sample_submission = pd.read_pickle('../data/reduced/sample_submission.pkl')
    submission = sample_submission[['id']].merge(submission, how='left', on='id')
    submission.to_csv(f'submit/{VERSION}__{scores:.05}.csv.gz', index=False, compression='gzip')

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
    all_data = get_all_features()
    print('\n', all_data.info())
    all_train_data, submit_data = train_submit_split(all_data)
    # Dump Split Data.
    print('Cache Train and Submission Data.')
    dump_pickle(all_train_data, 'features/all_train_data.pkl')
    dump_pickle(submit_data, 'features/submit_data.pkl')
    del all_data, all_train_data, submit_data; gc.collect();

    print('\n\n--- Train Model ---\n\n')
    _ = run_train()

    print('\n\n--- Submission ---\n\n')
    run_submission()


if __name__ == "__main__":
    main()
