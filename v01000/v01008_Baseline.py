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

from typing import Union

import seaborn
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

from scipy.stats import linregress

from sklearn import preprocessing
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error

import lightgbm as lgb

IS_TEST = True
SEED = 42
VERSION = str(__file__).split('_')[0]


''' Load Data
MEMO:
- コンペ終了1ヶ月前には sales_train_evaluation.csv が追加される。
- train = sales_train_validation, test = sales_train_evaluation.
'''


def read_data():
    files = ['calendar', 'sample_submission', 'sales_train_validation', 'sell_prices']

    if os.path.exists('/kaggle/input/m5-forecasting-accuracy'):
        data_dir_path = '/kaggle/input/m5-forecasting-accuracy'
        dst_data = {}
        for file in files:
            print(f'Reading {file} ....')
            dst_data[file] = pd.read_csv(data_dir_path + file + '.csv')
    else:
        data_dir_path = '../data/reduced/'
        dst_data = {}
        for file in files:
            print(f'Reading {file} ....')
            dst_data[file] = pd.read_pickle(data_dir_path + file + '.pkl')
    return dst_data.values()


'''Transform Data
'''


def load_pickle(filepath):
    with open(filepath, 'rb') as file:
        return pickle.load(file)


def dump_pickle(data, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def reduce_mem_usage(df, verbose=True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


def encode_map(filename='encode_map', use_cache=True):
    print('processing encode_map.')
    filepath = f'features/{filename}.pkl'
    if use_cache and os.path.exists(filepath):
        return load_pickle(filepath)

    train = pd.read_pickle('../data/reduced/sales_train_validation.pkl')
    categorical_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    encode_map = {
        col: {label: i for i, label in enumerate(sorted(train[col].unique()))}
        for col in categorical_cols
    }

    dump_pickle(encode_map, filepath)
    return encode_map


def parse_sell_price(filename='encoded_sell_price', use_cache=True):
    print('processing parse_sell_price.')
    filepath = f'features/{filename}.pkl'
    if use_cache and os.path.exists(filepath):
        return pd.read_pickle(filepath)
    # Load Data
    df = pd.read_pickle('../data/reduced/sell_prices.pkl')
    # Initial Processing And Feature Engineearing
    df['Log1p_sell_price'] = np.log1p(df['sell_price'])
    df['area_id'] = df['store_id'].str.extract('(\w+)_\d+')
    df['sell_price_rate_by_wm_yr_wk__item_id'] = df['sell_price'] / \
        df.groupby(['wm_yr_wk', 'item_id'])['sell_price'].transform('mean')
    df['sell_price_rate_by_wm_yr_wk__area__item_id'] = df['sell_price'] / \
        df.groupby(['wm_yr_wk', 'area_id', 'item_id'])['sell_price'].transform('mean')
    df['sell_price_momentum'] = df['sell_price'] / \
        df.groupby(['store_id', 'item_id'])['sell_price'].transform(lambda x: x.shift(1))
    # Export DataFrame
    df.drop(['area_id'], axis=1, inplace=True)
    df = df.pipe(reduce_mem_usage)
    df.to_pickle(filepath)
    return df


def encode_calendar(filename='encoded_calendar', use_cache=True):
    print('processing encode_calendar.')
    filepath = f'features/{filename}.pkl'
    if use_cache and os.path.exists(filepath):
        return pd.read_pickle(filepath)
    # Load Data
    df = pd.read_pickle('../data/reduced/calendar.pkl')
    # Initial Processing And Feature Engineearing
    df['date'] = pd.to_datetime(df['date'])
    attrs = [
        "quarter",
        "month",
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
        df[attr] = getattr(df['date'].dt, attr).astype(np.int8)

    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(np.int8)
    # MEMO: N_Unique of event_name_1 == 31 and event_name_2 == 5.
    event_cols = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    df[event_cols] = df[event_cols].fillna('None')
    for c in event_cols:
        le = preprocessing.LabelEncoder()
        df[c] = le.fit_transform(df[c].values).astype('int8')

    # for c in event_cols:
    #     for diff in [1, 2]:
    #         df[f"{c}_lag_t{diff}"] = df[c].shift(diff)
    # Drop columns.
    cols_to_drop = ['weekday', 'wday', 'year']
    df.drop(cols_to_drop, axis=1, inplace=True)
    # Export DataFrame
    df = df.pipe(reduce_mem_usage)
    df.to_pickle(filepath)
    return df


def hstack_sales_colums(df, latest_d, stack_days=28):
    # MEMO: 予測日数分のデータをtrain/testに水平結合する。
    add_columns = ['d_' + str(i + 1) for i in range(latest_d, latest_d + stack_days)]
    add_df = pd.DataFrame(index=df.index, columns=add_columns).fillna(0)
    return pd.concat([df, add_df], axis=1)


def melt_data(filename='melted_train', use_cache=True):
    print('processing melt_data.')
    filepath = f'features/{filename}.pkl'
    # check is exist cached file.
    if use_cache and os.path.exists(filepath):
        return pd.read_pickle(filepath)
    # Load Data
    latest_d = 1913  # latest_d of evaluation data is 1941
    df = pd.read_pickle('../data/reduced/sales_train_validation.pkl')
    df = hstack_sales_colums(df, latest_d)
    # Melt Main Data and Join Optinal Data.
    id_columns = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    df = pd.melt(df, id_vars=id_columns, var_name='d', value_name='sales')
    # Join calendar.
    calendar = pd.read_pickle('features/encoded_calendar.pkl')
    df = pd.merge(df, calendar, how='left', on='d')
    # Join sell_price.
    sell_price = pd.read_pickle('features/encoded_sell_price.pkl')
    df = pd.merge(df, sell_price, how='left', on=['store_id', 'item_id', 'wm_yr_wk'])
    # Label encoding main dataframe.
    with open('features/encode_map.pkl', 'rb') as file:
        encode_map = pickle.load(file)
    for label, encode_map in encode_map.items():
        df[label] = df[label].map(encode_map)
    # Drop Null Records.
    df.dropna(subset=['sell_price', 'sell_price_momentum'], axis=0, inplace=True)
    # Cache DataFrame.
    df = df.pipe(reduce_mem_usage)
    df.to_pickle(filepath)
    return df


''' Feature Engineering
'''
#     print('')
#     print('\tADD Roll SLOPE Feature', end='')
#     slope_func = lambda x: linregress(np.arange(len(x)), x)[0]
#     df[f"{col}rolling_SLOPE_t30"] = grouped_df.transform(
#           lambda x: x.shift(DAYS_PRED).rolling(30).agg(slope_func))


class BaseFeature():
    def __init__(self, filename, use_cache=True):
        self.filepath = f'features/{filename}.pkl'
        self.use_cache = use_cache
        self.is_exist_cahce = False
        self.df = pd.DataFrame()

    def __enter__(self):
        if self.use_cache:
            self.check_exist_cahce()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.is_exist_cahce:
            with open(self.filepath, 'wb') as file:
                pickle.dump(self.df, file, protocol=pickle.HIGHEST_PROTOCOL)

    def check_exist_cahce(self):
        if os.path.exists(self.filepath):
            self.is_exist_cahce = True

    def get_feature(self, df):
        if self.is_exist_cahce:
            with open(self.filepath, 'rb') as file:
                self.df = pickle.load(file)
            return self.df
        else:
            self.df = self.create_feature(df)
            return self.df

    def create_feature(self, df):
        raise NotImplementedError


class AddBaseSalesFeature(BaseFeature):
    def create_feature(self, df):
        print('Create Sales Feature.')
        DAYS_PRED = 28
        col = 'sales'
        grouped_df = df.groupby(["id"])[col]

        for diff in [0, 1, 2, 4, 5, 6]:
            shift = DAYS_PRED + diff
            df[f"{col}_lag_t{shift}"] = grouped_df.transform(lambda x: x.shift(shift))

        for diff in [1, 2, 4, 5, 6, 7]:
            df[f"{col}_lag_t{shift}"] = grouped_df.transform(
                lambda x: x.shift(DAYS_PRED).diff(diff))

        for diff in [1, 2, 4, 5, 6, 7]:
            df[f"{col}_lag_t{shift}"] = grouped_df.transform(
                lambda x: x.shift(DAYS_PRED).pct_change(diff))

        for window in [7, 30, 60, 90, 180]:
            df[f"{col}_rolling_STD_t{window}"] = grouped_df.transform(
                lambda x: x.shift(DAYS_PRED).rolling(window).std())

        for window in [7, 30, 60, 90, 180]:
            df[f"{col}_rolling_MEAN_t{window}"] = grouped_df.transform(
                lambda x: x.shift(DAYS_PRED).rolling(window).mean())

        for window in [7, 30, 60]:
            df[f"{col}_rolling_MIN_t{window}"] = grouped_df.transform(
                lambda x: x.shift(DAYS_PRED).rolling(window).min())

        for window in [7, 30, 60]:
            df[f"{col}_rolling_MAX_t{window}"] = grouped_df.transform(
                lambda x: x.shift(DAYS_PRED).rolling(window).max())

        for window in [7, 14, 30, 60]:
            df[f"{col}_rolling_ZeroRatio_t{window}"] = grouped_df.transform(
                lambda x: 1 - (x == 0).shift(DAYS_PRED).rolling(window).mean())
            df[f"{col}_rolling_ZeroCount_t{window}"] = grouped_df.transform(
                lambda x: (x == 0).shift(DAYS_PRED).rolling(window).sum())

        df[f"{col}_rolling_SKEW_t30"] = grouped_df.transform(
            lambda x: x.shift(DAYS_PRED).rolling(30).skew())
        df[f"{col}_rolling_KURT_t30"] = grouped_df.transform(
            lambda x: x.shift(DAYS_PRED).rolling(30).kurt())
        return df.pipe(reduce_mem_usage)


def create_features(df, is_use_cache=True):
    '''
    TODO:
        - 当該月の特徴量
            - 月初・月末（１日）の売上
            - 15, 20日などのクレジットカードの締日のごとの統計量
        - 過去１ヶ月間の（特定item_idの売上 / スーパー全体の売上）
            - （特定item_idの売上個数 / スーパー全体の売上個数）
            - （特定item_idの売上個数 / スーパー全体の売上個数）
        - pd.DataFrame().shift(DAYS_PRED).diff(n) これつかっていきたい。
            - nがn日前のデータとの差という意味
            - pct_change()もあるらしい
                - (A - B) / B
                - 過去のデータから何倍変化があったかわかる。
    '''

    with AddBaseSalesFeature(filename='add_sales_train', use_cache=is_use_cache) as feat:
        df = feat.get_feature(df)

    return df.reset_index(drop=True)


''' Train Model
'''


def split_train_eval_submit(df, pred_interval=28):
    latest_date = df['date'].max()
    submit_date = latest_date - datetime.timedelta(days=pred_interval)
    submit_mask = (df["date"] > submit_date)

    eval_date = latest_date - datetime.timedelta(days=pred_interval * 2)
    eval_mask = ((df["date"] > eval_date) & (df["date"] <= submit_date))

    train_mask = ((~eval_mask) & (~submit_mask))
    return df[train_mask], df[eval_mask], df[submit_mask]


def drop_null_rows(df):
    print('Drop Null Rows.')
    check_cols = ['lag', 'rolling']
    cheked_regex = '|'.join(check_cols)
    target_cols = df.columns[df.columns.str.contains(cheked_regex)]
    is_contain_null_rows = df[target_cols].isnull().any(axis=1)
    return df.loc[~(is_contain_null_rows), :]


def estimate_rmsle(actual, preds, weight=None):
    return np.sqrt(mean_squared_log_error(actual, preds, sample_weight=weight))


def lgbm_rmsle(preds, data):
    return 'RMSLE', estimate_rmsle(data.get_label(), preds), False


def estimate_rmse(actual, pred, weight=None):
    return mean_squared_error(actual, pred, squared=False, sample_weight=weight)


def lgbm_rmse(preds, data):
    return 'RMSE', estimate_rmse(data.get_label(), preds), False


class LGBM_Model():
    def __init__(self, X, y, cv_param, model_param, train_param, groups=None, custom_metric=None):
        self.cv = self.set_cv(cv_param)
        self.custom_metric = custom_metric
        self.models = self.fit(X, y, model_param, train_param, groups)

    def set_cv(self, cv_param):
        cv = TimeSeriesSplit(n_splits=cv_param['n_splits'], max_train_size=cv_param['max_train_size'])
        return cv

    def fit(self, X, y, model_param, train_param, groups):
        models = []
        for n_fold, (train_idx, valid_idx) in enumerate(self.cv.split(X)):
            print(f"\n{n_fold + 1} of {self.cv.get_n_splits()} Fold:\n")
            train_X, train_y = X.iloc[train_idx], y.iloc[train_idx]
            valid_X, valid_y = X.iloc[valid_idx], y.iloc[valid_idx]

            print('Train DataFrame Size:', train_X.shape)
            print('Valid DataFrame Size:', valid_X.shape)
            train_dataset = lgb.Dataset(train_X, label=train_y)
            valid_dataset = lgb.Dataset(valid_X, label=valid_y, reference=train_dataset)
            model = lgb.train(
                model_param,
                train_dataset,
                valid_sets=[train_dataset, valid_dataset],
                valid_names=["train", "valid"],
                **train_param
            )
            models.append(model)
            del train_X, valid_X, train_y, valid_y; gc.collect()
        return models

    def get_models(self):
        return self.models

    def predict(self, data):
        models = self.get_models()
        return [m.predict(data.values, num_iteration=m.best_iteration) for m in models]

    def save_importance(self, filepath, max_num_features=50, figsize=(20, 25), plot=False):
        models = self.get_models()
        # Define Feature Importance DataFrame.
        imp_df = pd.DataFrame(
            [m.feature_importance() for m in models],
            columns=models[0].feature_name()
        ).T
        imp_df['AVG_Importance'] = imp_df.iloc[:, :len(models)].mean(axis=1)
        imp_df['STD_Importance'] = imp_df.iloc[:, :len(models)].std(axis=1)
        imp_df.sort_values(by='AVG_Importance', inplace=True)
        # Plot Importance DataFrame.
        plt.figure(figsize=figsize)
        imp_df[-max_num_features:].plot(
            kind='barh', title='Feature importance', figsize=figsize,
            y='AVG_Importance', xerr='STD_Importance', align="center"
        )
        if plot:
            plt.show()
        plt.savefig(filepath)
        plt.close('all')


class RondomSeed_LGBM_Model():
    def __init__(self, data, feature, target,
                 n_fold, test_days, max_train_days, model_param, train_param,
                 weight=None):
        self.n_fold = n_fold
        self.weight = weight
        train_dataset, valid_dataset = self.split_data(
            data, feature, target, test_days, max_train_days)
        self.models = self.fit(train_dataset, valid_dataset, model_param, train_param)

    def split_data(self, data, features, target, test_days, max_train_days):
        latest_date = data['date'].max()
        oldest_valid_date = latest_date - datetime.timedelta(days=test_days)
        valid_mask = (data["date"] > oldest_valid_date)
        oldest_train_date = oldest_valid_date - datetime.timedelta(days=max_train_days)
        train_mask = (data["date"] > oldest_train_date) & (data["date"] <= oldest_valid_date)

        train_X, train_y = data.loc[train_mask, features], data.loc[train_mask, target]
        valid_X, valid_y = data.loc[valid_mask, features], data.loc[valid_mask, target]

        train_dataset = lgb.Dataset(train_X, label=train_y)
        valid_dataset = lgb.Dataset(valid_X, label=valid_y, reference=train_dataset)

        if self.weight is not None:
            train_dataset.set_weight(self.weight.loc[train_mask])
            valid_dataset.set_weight(self.weight.loc[valid_mask])

        print('Train DataFrame Size:', train_mask.sum())
        print('Valid DataFrame Size:', valid_mask.sum())
        return train_dataset, valid_dataset

    def fit(self, train_dataset, valid_dataset, model_param, train_param):
        models = []
        for n in range(self.n_fold):
            print(f"\n{n + 1} of {self.n_fold} Fold:\n")
            model_param['seed'] = model_param['seed'] + 1
            model = lgb.train(
                model_param,
                train_dataset,
                valid_sets=[train_dataset, valid_dataset],
                valid_names=["train", "valid"],
                **train_param
            )
            models.append(model)
        return models

    def get_models(self):
        return self.models

    def predict(self, data):
        models = self.get_models()
        preds = [m.predict(data, num_iteration=m.best_iteration) for m in models]
        avg_pred = np.mean(preds, axis=0)
        return avg_pred

    def save_importance(self, filepath, max_num_features=50, figsize=(20, 25), plot=False):
        models = self.get_models()
        # Define Feature Importance DataFrame.
        imp_df = pd.DataFrame(
            [m.feature_importance() for m in models],
            columns=models[0].feature_name()
        ).T
        imp_df['AVG_Importance'] = imp_df.iloc[:, :len(models)].mean(axis=1)
        imp_df['STD_Importance'] = imp_df.iloc[:, :len(models)].std(axis=1)
        imp_df.sort_values(by='AVG_Importance', inplace=True)
        # Plot Importance DataFrame.
        plt.figure(figsize=figsize)
        imp_df[-max_num_features:].plot(
            kind='barh', title='Feature importance', figsize=figsize,
            y='AVG_Importance', xerr='STD_Importance', align="center"
        )
        if plot:
            plt.show()
        plt.savefig(filepath)
        plt.close('all')


def train_model(df, feature, target):
    n_fold = 3
    max_train_days = 2.5 * 365
    test_days = 90

    model_param = {
        "boosting_type": "gbdt",
        "metric": "rmse",
        "objective": "regression",
        "seed": SEED,
        "learning_rate": 0.3,
        "num_leaves": 2**6,
        'min_data_in_leaf': 50,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "feature_fraction": 1.0,
        "lambda_l2": 0.1,
        "verbosity": -1
    }

    train_param = {
        "num_boost_round": 100000,
        "early_stopping_rounds": 50,
        "verbose_eval": 100
    }

    print(f'n_fold: {n_fold}')
    print(f'max_train_days: {max_train_days}')
    print(f'test_days: {test_days}')
    print(model_param)
    print(train_param)

    df = drop_null_rows(df)
    cat_feat = ['item_id', 'dept_id', 'store_id', 'cat_id',
                'state_id', "event_name_1", "event_name_2", "event_type_1", "event_type_2"]
    lgbm_model = RondomSeed_LGBM_Model(
        df, feature, target, n_fold, test_days, max_train_days,
        model_param, train_param
    )
    lgbm_model.save_importance(filepath=f'result/importance/{VERSION}.png')
    dump_pickle(lgbm_model, f'result/model/{VERSION}.pkl')


''' Evaluation Model
'''


class WRMSSEEvaluator(object):

    group_ids = ('all_id', 'state_id', 'store_id', 'cat_id', 'dept_id', 'item_id',
                 ['state_id', 'cat_id'], ['state_id', 'dept_id'], ['store_id', 'cat_id'],
                 ['store_id', 'dept_id'], ['item_id', 'state_id'], ['item_id', 'store_id'])

    def __init__(self,
                 train_df: pd.DataFrame,
                 valid_df: pd.DataFrame,
                 calendar: pd.DataFrame,
                 prices: pd.DataFrame):
        '''
        intialize and calculate weights
        '''
        self.calendar = calendar
        self.prices = prices
        self.train_df = train_df
        self.valid_df = valid_df
        self.train_target_columns = [i for i in self.train_df.columns if i.startswith('d_')]
        self.weight_columns = self.train_df.iloc[:, -28:].columns.tolist()

        self.train_df['all_id'] = "all"

        self.id_columns = [i for i in self.train_df.columns if not i.startswith('d_')]
        self.valid_target_columns = [i for i in self.valid_df.columns if i.startswith('d_')]

        if not all([c in self.valid_df.columns for c in self.id_columns]):
            self.valid_df = pd.concat([self.train_df[self.id_columns], self.valid_df],
                                      axis=1,
                                      sort=False)
        self.train_series = self.trans_30490_to_42840(self.train_df,
                                                      self.train_target_columns,
                                                      self.group_ids)
        self.valid_series = self.trans_30490_to_42840(self.valid_df,
                                                      self.valid_target_columns,
                                                      self.group_ids)
        self.weights = self.get_weight_df()
        self.scale = self.get_scale()
        self.train_series = None
        self.train_df = None
        self.prices = None
        self.calendar = None

    def get_scale(self):
        '''
        scaling factor for each series ignoring starting zeros
        '''
        scales = []
        for i in range(len(self.train_series)):
            series = self.train_series.iloc[i].values
            series = series[np.argmax(series != 0):]
            scale = ((series[1:] - series[:-1]) ** 2).mean()
            scales.append(scale)
        return np.array(scales)

    def get_name(self, i):
        '''
        convert a str or list of strings to unique string
        used for naming each of 42840 series
        '''
        if type(i) == str or type(i) == int:
            return str(i)
        else:
            return "--".join(i)

    def get_weight_df(self) -> pd.DataFrame:
        """
        returns weights for each of 42840 series in a dataFrame
        """
        day_to_week = self.calendar.set_index("d")["wm_yr_wk"].to_dict()
        weight_df = self.train_df[["item_id", "store_id"] + self.weight_columns].set_index(
            ["item_id", "store_id"]
        )
        weight_df = (
            weight_df.stack().reset_index().rename(columns={"level_2": "d", 0: "value"})
        )
        weight_df["wm_yr_wk"] = weight_df["d"].map(day_to_week)
        weight_df = weight_df.merge(
            self.prices, how="left", on=["item_id", "store_id", "wm_yr_wk"]
        )
        weight_df["value"] = weight_df["value"] * weight_df["sell_price"]
        weight_df = weight_df.set_index(["item_id", "store_id", "d"]).unstack(level=2)[
            "value"
        ]
        weight_df = weight_df.loc[
            zip(self.train_df.item_id, self.train_df.store_id), :
        ].reset_index(drop=True)
        weight_df = pd.concat(
            [self.train_df[self.id_columns], weight_df], axis=1, sort=False
        )
        weights_map = {}
        for i, group_id in enumerate(self.group_ids):
            lv_weight = weight_df.groupby(group_id)[self.weight_columns].sum().sum(axis=1)
            lv_weight = lv_weight / lv_weight.sum()
            for i in range(len(lv_weight)):
                weights_map[self.get_name(lv_weight.index[i])] = np.array(
                    [lv_weight.iloc[i]]
                )
        weights = pd.DataFrame(weights_map).T / len(self.group_ids)

        return weights

    def trans_30490_to_42840(self, df, cols, group_ids, dis=False):
        series_map = {}
        for i, group_id in enumerate(self.group_ids):
            tr = df.groupby(group_id)[cols].sum()
            for i in range(len(tr)):
                series_map[self.get_name(tr.index[i])] = tr.iloc[i].values
        return pd.DataFrame(series_map).T

    def get_rmsse(self, valid_preds) -> pd.Series:
        score = ((self.valid_series - valid_preds) ** 2).mean(axis=1)
        self.scale = np.where(self.scale != 0, self.scale, 1)
        rmsse = (score / self.scale).map(np.sqrt)
        return rmsse

    def score(self, valid_preds: Union[pd.DataFrame, np.ndarray]) -> float:
        assert self.valid_df[self.valid_target_columns].shape == valid_preds.shape

        if isinstance(valid_preds, np.ndarray):
            valid_preds = pd.DataFrame(valid_preds, columns=self.valid_target_columns)

        valid_preds = pd.concat([self.valid_df[self.id_columns], valid_preds],
                                axis=1,
                                sort=False)
        valid_preds = self.trans_30490_to_42840(valid_preds,
                                                self.valid_target_columns,
                                                self.group_ids,
                                                True)
        self.rmsse = self.get_rmsse(valid_preds)
        self.contributors = pd.concat([self.weights, self.rmsse],
                                      axis=1,
                                      sort=False).prod(axis=1)
        return np.sum(self.contributors)


def estimate_wrmsse(eval_data, preds, eval_days=28):
    def ordered_d_cols(df_cols):
        return sorted(df_cols, key=lambda x: int((re.search(r"\d+", x)).group(0)))
    # Processing train data.
    df = pd.read_pickle('../data/reduced/sales_train_validation.pkl')
    train_idx_labels = df.iloc[:, :-eval_days].copy(deep=True)
    # Processing eval label data.
    eval_labels = pd.pivot_table(
        eval_data, index='id', columns='d', values='sales', fill_value=0).reset_index()
    eval_labels = eval_labels[ordered_d_cols(eval_labels.drop('id', axis=1).columns.tolist())]
    # Processing eval predict data.
    eval_data['pred'] = preds
    pred_labels = pd.pivot_table(
        eval_data, index='id', columns='d', values='pred', fill_value=0).reset_index()
    pred_labels = pred_labels[ordered_d_cols(pred_labels.drop('id', axis=1).columns.tolist())]
    # Estimate WRMSSE score.
    calendar = pd.read_pickle('../data/reduced/calendar.pkl')
    sell_prices = pd.read_pickle('../data/reduced/sell_prices.pkl')
    e = WRMSSEEvaluator(train_idx_labels, eval_labels, calendar, sell_prices)
    score = e.score(pred_labels)
    return score


def evaluation_model(eval_data, feature):
    lgbm_model = load_pickle(f'result/model/{VERSION}.pkl')
    preds = lgbm_model.predict(eval_data[feature].values)
    metric_scores = {}
    metric_scores['RMSE'] = mean_squared_error(eval_data['sales'].values, preds, squared=False)
    metric_scores['WRMSSE'] = estimate_wrmsse(eval_data, preds)
    for metric, score in metric_scores.items():
        print(f'{metric}: {score}')
    # Dump metric_scores as Json file.
    with open(f'result/score/{VERSION}.json', 'w') as file:
        json.dump(metric_scores, file, indent=4)
    return metric_scores


''' Submission
'''


def create_submission_file(submit_data, feature, score):
    submission = pd.read_pickle('../data/reduced/sample_submission.pkl')['id'].to_frame()
    lgbm_model = load_pickle(f'result/model/{VERSION}.pkl')

    submit_data['pred'] = lgbm_model.predict(submit_data[feature].values)
    sub_validation = pd.pivot(submit_data, index='id', columns='date', values='pred').reset_index()
    sub_validation.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]

    submission = pd.merge(submission, sub_validation, how='left', on='id')
    submission.fillna(0, inplace=True)
    submission.to_csv(f'submit/{VERSION}_{score:.04f}.csv.gz', index=False, compression='gzip')

    print('Submit DataFrame:', submission.shape)
    print(submission.head())


def main():
    now = str(datetime.datetime.now())
    print('\n\n' + '=' * 80, '\n\n')
    print(f'RUNNING at {now}\n')

    print('\n--- Load Data ---\n')
    print('Loading and initial processing have already been completed.')

    print('\n--- Transfrom Data ---\n')
    _ = encode_map(filename='encode_map', use_cache=True)
    _ = parse_sell_price(filename='encoded_sell_price', use_cache=True)
    _ = encode_calendar(filename='encoded_calendar', use_cache=True)

    train = melt_data(filename='melted_train', use_cache=True)
    print('\nTrain DataFrame:', train.shape)
    print('Memory Usage:', train.memory_usage().sum() / 1024 ** 2, 'Mb')
    print(train.head())

    print('\n--- Feature Engineering ---\n')
    train = create_features(train, is_use_cache=True)
    print('Train DataFrame:', train.shape)
    print('Memory Usage:', train.memory_usage().sum() / 1024 ** 2, 'Mb')
    print(train.head())

    print('\n--- Train Model ---\n')
    target = 'sales'
    cols_to_drop = ['id', 'wm_yr_wk', 'd', 'date'] + [target]
    features = train.columns.tolist()
    features = [f for f in features if f not in cols_to_drop]
    train_data, eval_data, submit_data = split_train_eval_submit(train)
    del train; gc.collect()
    # train_data['sales'] = np.log1p(train_data['sales'])
    train_model(train_data, features, target)

    print('\n--- Evaluation ---\n')
    metric_scores = evaluation_model(eval_data, features)

    print('\n--- Submission ---\n')
    create_submission_file(submit_data, features, metric_scores['WRMSSE'])


if __name__ == '__main__':
    main()
