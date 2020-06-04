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

import warnings
warnings.filterwarnings('ignore')

# custom funcs
from script import WRMSSEEvaluator
from script import cache_result
from script import reduce_mem_usage
from script import load_pickle, dump_pickle
from script import get_groups


# Define global Variables.
IS_TEST = True
SEED = 42
VERSION = str(__file__).split('_')[0]
TARGET = 'sales'
NUM_ITEMS = 30490

FEATURECOLS_PATH = f'result/feature_cols/{VERSION}.pkl'
MODEL_PATH = f'result/model/{VERSION}.pkl'
IMPORTANCE_PATH = f'result/importance/{VERSION}.png'
SCORE_PATH = f'result/score/{VERSION}.pkl'

GROUP_ID = ('dept_id',)  # one version, one group
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
        "weekofyear",
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
    )
    sell_prices['price_norm'] = sell_prices['sell_price'] / sell_prices['price_max']

    return sell_prices.pipe(reduce_mem_usage)


@cache_result(filename='parse_sales_train', use_cache=True)
def parse_sales_train():
    train = pd.read_pickle('../data/reduced/sales_train_validation.pkl')
    # Add Prediction Columns
    start_d = 1914  # 1915
    end_d = 1969
    for i in range(start_d, end_d + 1):
        train[f'd_{i}'] = 0
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
    df = df.iloc[-n_row:, :]    # Sampling Recently data
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

    df = df.reset_index(drop=True)
    return df.pipe(reduce_mem_usage)


def run_trainsform():
    # Transform each raw data.
    _ = parse_calendar()
    _ = parse_sell_prices()
    _ = parse_sales_train()
    # Melt and Merge all data.
    sample_rows = (365 * 3 + 28 * 2) * NUM_ITEMS
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
    grouped_df = srd_df.groupby('id')
    dst_df = pd.DataFrame()
    # Creat Features
    agg_funcs = {}

    lags = list(range(15))
    for i in lags:
        lag_t = shift_days + i
        agg_funcs[f'sales_lag_t{i}'] = grouped_df['sales'].transform(lambda x: x.shift(lag_t))

    num_shift = [1, 7, 14]
    num_roll = [7, 14, 30]
    for n_shift in num_shift:
        shift = shift_days + n_shift
        for n_roll in num_roll:
            agg_funcs[f'sales_roll_mean_t{n_shift}_{n_roll}'] = grouped_df['sales'].transform(
                lambda x: x.shift(shift).rolling(i).mean())
            agg_funcs[f'sales_roll_std_t{n_shift}_{n_roll}'] = grouped_df['sales'].transform(
                lambda x: x.shift(shift).rolling(i).std())

    for i in [7, 14, 30]:
        agg_funcs[f'sales_rolling_ZeroRatio_t{i}'] = grouped_df['sales'].transform(
            lambda x: 1 - (x == 0).shift(shift_days).rolling(i).mean())
        agg_funcs[f'sales_rolling_ZeroCount_t{i}'] = grouped_df['sales'].transform(
            lambda x: (x == 0).shift(shift_days).rolling(i).sum())

    agg_funcs['sales_rolling_skew_t30'] = grouped_df['sales'].transform(
        lambda x: x.shift(shift_days).rolling(30).skew())
    agg_funcs['sales_rolling_kurt_t30'] = grouped_df['sales'].transform(
        lambda x: x.shift(shift_days).rolling(30).kurt())

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
    dst_df = pd.DataFrame()
    # Convert target to binary
    dst_df['id'] = srd_df['id']
    dst_df['non_zero'] = (srd_df['sales'] > 0)
    dst_df['d'] = srd_df['d'].str.replace('d_', '').astype(int)
    dst_df = dst_df.assign(
        non_zero_lag=dst_df.groupby(['id'])['non_zero'].transform(
            lambda x: x.shift(28).rolling(2000, 1).sum()).fillna(-1)
    )

    temp_df = dst_df[['id', 'd', 'non_zero_lag']].drop_duplicates(subset=['id', 'non_zero_lag'])
    temp_df.columns = ['id', 'd_min', 'non_zero_lag']

    dst_df = dst_df.merge(temp_df, on=['id', 'non_zero_lag'], how='left')
    dst_df['days_from_last_sales'] = dst_df['d'] - dst_df['d_min']
    return dst_df[['days_from_last_sales']]


@cache_result(filename='simple_target_encoding', use_cache=True)
def simple_target_encoding():
    # TODO: submission 期間のデータも含んでしまっているので調整する必要がある
    # Define variables and dataframes.
    shift_days = 28
    use_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'sales']
    srd_df = pd.read_pickle('features/melted_and_merged_train.pkl')[use_cols]
    dst_df = pd.DataFrame()
    # Convert target to binary
    icols = [
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
        ['item_id', 'store_id']
    ]

    for col in icols:
        print('Encoding', col)
        col_name = '_'.join(col)
        dst_df[f'enc_{col_name}_mean'] = srd_df.groupby(col)[TARGET].transform('mean')
        dst_df[f'enc_{col_name}_std'] = srd_df.groupby(col)[TARGET].transform('std')
    return dst_df.pipe(reduce_mem_usage)


@cache_result(filename='all_data', use_cache=False)
def run_create_features():
    # Create Feature
    df = pd.read_pickle('features/melted_and_merged_train.pkl')
    feat_funcs = [
        sales_lag_and_roll,
        price_simple_feature,
        days_from_last_sales,
        simple_target_encoding
    ]
    # Join All Features.
    for f_func in feat_funcs:
        temp_feat_df = f_func()
        df = pd.concat([df, temp_feat_df], axis=1)
        del temp_feat_df; gc.collect()

    # numeric_cols = df.select_dtypes(include=['number']).columns
    # df = df.assign(**{num_c: df[num_c].fillna(-999) for num_c in numeric_cols})
    # Export Data
    return df


""" Split Data
Input: all data
Output: all_train_data, eval_data, submit_data

# MEMO
- 新しいdataが使えるようになったらeval_dataの期間を設定し直す。
"""


def run_split_data():
    df = pd.read_pickle('features/all_data.pkl')
    # Define Dump filepath.
    all_train_data_path = 'features/all_train_data.pkl'
    eval_data_path = 'features/eval_data.pkl'
    submit_data_path = 'features/submit_data.pkl'
    # Split Train Data.
    print(f'Split all_train_data to {all_train_data_path}')
    upper_train_date = '2016-04-24'
    train_mask = (df['date'] <= upper_train_date)
    dump_pickle(df[train_mask], all_train_data_path)
    # Split Evaluation Data.
    print(f'Split eval_data to {eval_data_path}')
    lower_eval_date = '2016-03-27'  # '2016-04-24' <- for evaluation.
    upper_eval_date = datetime.datetime.strptime(lower_eval_date, '%Y-%m-%d') +\
        datetime.timedelta(days=28)
    eval_mask = (df['date'] > lower_eval_date) & (df['date'] <= upper_eval_date)
    dump_pickle(df[eval_mask], eval_data_path)
    # Split Submisssion Data.
    print(f'Split submit_data to {submit_data_path}')
    lower_submit_date = '2016-04-24'
    submit_mask = (df['date'] > lower_submit_date)
    dump_pickle(df[submit_mask], submit_data_path)

    print(df.info(verbose=True))
    del df; gc.collect()


""" Train
Input: all train data
Output: models
"""


class WRMSSEForLightGBM(WRMSSEEvaluator):

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

    def custom_feval(self, preds, dtrain):
        actual = dtrain.get_label().reshape(28, -1).T
        preds = preds.reshape(28, -1).T

        rmse = np.sqrt(np.mean(np.square(actual - preds), axis=1))
        score = np.sum(self.valid_feval_weight * rmse)
        return 'WRMSSE', score, False

    def set_series_weight_for_fobj(self, train_ids):
        fobj_weight, fojb_scale = self.get_series_weight(train_ids)
        self.custom_fobj_weight = 2 * np.square(fobj_weight) / fojb_scale

    def custom_fobj(self, preds, dtrain):
        actual = dtrain.get_label()
        weight = self.custom_fobj_weight

        grad = weight * np.square(preds - actual)
        hess = weight * 2 * (preds - actual)
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

        self.train_bin_path = 'tmp_train_set.bin'
        self.valid_bin_path = 'tmp_valid_set.bin'

        train_dataset, valid_dataset = self._convert_dataset(train_data, valid_data)
        model = self.fit(params, train_param, train_dataset, valid_dataset)
        # Remove bin file
        self._remove_bin_file(self.train_bin_path)
        self._remove_bin_file(self.valid_bin_path)

    def _remove_bin_file(self, filepath):
        if os.path.exists(filepath):
            os.remove(filepath)

    def _convert_dataset(self, train_data, valid_data):
        train_dataset = lgb.Dataset(
            train_data[self.feature_cols],
            train_data[self.target_col],
            weight=self.train_weight,
            feature_name=self.feature_cols
        )
        valid_dataset = lgb.Dataset(
            valid_data[self.feature_cols],
            valid_data[self.target_col],
            weight=self.valid_weight,
            reference=train_dataset
        )
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

    def fit(self, params, train_param, train_dataset, valid_dataset):
        self.model = lgb.train(
            params,
            train_dataset,
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


def train_valid_split(df, go_back_days=28):
    valid_duration = 28

    min_thresh_date = df['date'].max() - datetime.timedelta(days=go_back_days)
    max_thresh_date = min_thresh_date + datetime.timedelta(days=valid_duration)

    valid_mask = ((df["date"] > min_thresh_date) & (df["date"] <= max_thresh_date))
    train_mask = (df["date"] <= min_thresh_date)
    return df[train_mask], df[valid_mask]


def get_decayed_weights(df, days_keep_weight=0):
    max_d = df['d'].str.replace('d_', '').astype('int').max()

    weight_map = np.flip(np.concatenate([
        [1 for _ in range(days_keep_weight)],
        [1 - 0.0005 * i for i in range(days_keep_weight, max_d)]
    ], axis=0)).round(4)
    weight_map = {f'd_{i+1}': weight_map[i] for i in range(max_d)}
    weight = df['d'].map(weight_map).values
    return weight


def train_group_models():
    group_models = {}
    evaluator = get_evaluator(go_back_days=28)

    params = {
        'model_params': {
            'boosting': 'gbdt',
            'objective': 'tweedie',  # tweedie, poisson, regression
            'tweedie_variance_power': 1.1,  # 1.0=poisson
            'metric': 'custom',
            'num_leaves': 2**7 - 1,
            'min_data_in_leaf': 50,
            'seed': SEED,
            'learning_rate': 0.03,  # 0.1
            'subsample': 0.5,  # ~v05006, 0.8
            'subsample_freq': 1,
            'feature_fraction': 0.5,  # ~v05006, 0.8
            'force_row_wise': True,
            'verbose': -1
        },
        'train_params': {
            'num_boost_round': 1500,  # 2000
            'early_stopping_rounds': 100,
            'verbose_eval': 100,
            'feval': evaluator.custom_feval
        }
    }
    print('\nParameters:\n', json.dumps(params['model_params'], indent=4), '\n')

    models = []
    for i, g_id in enumerate(GROUPS):
        print(f'\n\nGroup ID: {g_id}, {i+1}/{len(GROUPS)}')
        df = pd.read_pickle('features/all_train_data.pkl')
        query = ''.join([f'(?=.*{i})' for i in g_id])
        df = df[df['id'].str.contains(f'^{query}.*$')]

        drop_cols = ['id', 'd', 'sales', 'date', 'wm_yr_wk']
        features = [f for f in df.columns if f not in drop_cols]
        dump_pickle(features, FEATURECOLS_PATH)
        # Split Train data.
        train_data, valid_data = train_valid_split(df, go_back_days=28)
        del df; gc.collect()
        train_data = train_data[train_data['release'] > 30]  # add at v05006
        train_data = train_data[train_data['days_from_last_sales'] < 7]  # add at v05009

        use_weight = True
        if use_weight:
            train_decayed_weight = get_decayed_weights(train_data, 90)
            weight, scale = evaluator.get_series_weight(train_data['id'])
            train_weight = (10000 * (weight / np.sqrt(scale))) * (1.3 * train_decayed_weight)

            weight, scale = evaluator.get_series_weight(valid_data['id'])
            valid_weight = (10000 * (weight / np.sqrt(scale)))

        is_set_fecal_weight = True
        if is_set_fecal_weight:
            evaluator.set_feval_weight(valid_data['id'].drop_duplicates(keep='last'))

        if query in 'HOBBIES':
            params['model_params']['learning_rate'] = 0.01

        lgb_model = LGBM_Model(
            params['model_params'],
            params['train_params'],
            train_data,
            valid_data,
            TARGET,
            features,
            categorical_cols=None,
            train_weight=train_weight,
            # valid_weight=valid_weight
        )
        # Save Importance
        IMPORTANCE_PATH = f'result/importance/{VERSION}/{g_id}.png'
        os.makedirs(f'result/importance/{VERSION}', exist_ok=True)
        lgb_model.save_importance(filepath=IMPORTANCE_PATH, figsize=(18, 25))
        # Add Model
        group_models[''.join(g_id)] = lgb_model
    return group_models


def run_train():
    models = train_group_models()
    dump_pickle(models, MODEL_PATH)


""" Evaluation

Input: eval_data, models
Output: validation scores
"""


def run_evaluation():
    scores = {}
    lgb_models = load_pickle(MODEL_PATH)
    features = load_pickle(FEATURECOLS_PATH)
    eval_data = load_pickle('./features/eval_data.pkl')
    evaluator = get_evaluator(go_back_days=28)
    valid_pred = np.zeros(eval_data.shape[0])

    for g_id in GROUPS:
        query = ''.join([f'(?=.*{i})' for i in g_id])
        is_groups = eval_data['id'].str.contains(f'^{query}.*$')
        valid_pred[is_groups] = lgb_models[''.join(g_id)].predict(
            eval_data.loc[is_groups, features])

    scores['RMSE'] = mean_squared_error(valid_pred, eval_data[TARGET], squared=False)
    scores['WRMSSE'] = evaluator.score(valid_pred.reshape(-1, NUM_ITEMS).T)
    dump_pickle(scores, SCORE_PATH)

    print('')
    for f_name, score in scores.items():
        print(f'Our val {f_name} score is {score}')


""" Submission
Input: submit_data, models
Output: submision file
"""


def run_submission():
    # Load Data.
    lgb_models = load_pickle(MODEL_PATH)
    scores = load_pickle(SCORE_PATH)['WRMSSE']
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

    valid_sub = submission[['id'] + [f'd_{i}' for i in range(1914, 1942)]]
    eval_sub = submission[['id'] + [f'd_{i}' for i in range(1942, 1970)]]
    valid_sub.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]
    eval_sub.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]
    eval_sub = eval_sub.assign(id=lambda x: x['id'].str.replace('validation', 'evaluation'))
    submission = pd.concat([valid_sub, eval_sub], axis=0)

    # Dump Submission file.
    sample_submission = pd.read_pickle('../data/reduced/sample_submission.pkl')
    submission = sample_submission[['id']].merge(submission, how='left', on='id')
    submission.to_csv(f'submit/{VERSION}__{scores:.05}.csv.gz', index=False, compression='gzip')

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

    print('\n\n--- Train ---\n\n')
    run_train()

    print('\n\n--- Evaluation ---\n\n')
    run_evaluation()

    print('\n\n--- Submission ---\n\n')
    run_submission()


if __name__ == '__main__':
    main()
