import os
import gc
import re
import datetime
from tqdm import tqdm

import numpy as np
import pandas as pd

from typing import Union

import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

from scipy.stats import linregress

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

import lightgbm as lgb


VERSION = str(__file__).split('_')[0]
IS_TEST = True

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


def hstack_sales_colums(df, last_d, stack_days=28):
    # MEMO: 予測日数分のデータをtrain/testに水平結合する。
    add_columns = ['d_' + str(i + 1) for i in range(last_d, last_d + stack_days)]
    add_df = pd.DataFrame(index=df.index, columns=add_columns).fillna(0)
    return pd.concat([df, add_df], axis=1)


def encode_calendar(df, filename='encoded_calendar', use_cache=True):
    filepath = f'features/{filename}.pkl'
    if use_cache and os.path.exists(filepath):
        return pd.read_pickle(filepath)

    # Drop Columns
    cols_to_drop = ['weekday', 'wday', 'year']
    df.drop(cols_to_drop, axis=1, inplace=True)
    # Parse Date Feature
    dt_col = 'date'
    df[dt_col] = pd.to_datetime(df[dt_col])
    attrs = [
        "quarter",
        "month",
        "week",
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
        df[attr] = getattr(df[dt_col].dt, attr).astype(dtype)

    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(np.int8)
    # MEMO: N_Unique of event_name_1 == 31 and event_name_2 == 5.
    event_cols = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    df[event_cols] = df[event_cols].fillna('None')
    for c in event_cols:
        le = preprocessing.LabelEncoder()
        df[c] = le.fit_transform(df[c].values).astype('int8')
    df.to_pickle(filepath)
    return df


def melt_data(df, calendar, sell_prices, encode_maps, filename, use_cache=True):
    filepath = f'features/{filename}.pkl'
    if use_cache and os.path.exists(filepath):
        return pd.read_pickle(filepath)
    # MEMO: ラベルは全データ共通なので、train/test/sell_prices の LabelEncode に使える。
    for label, encode_map in encode_maps.items():
        df[label] = df[label].map(encode_map)
        if label in ['item_id', 'store_id']:
            sell_prices[label] = sell_prices[label].map(encode_map)
    # Melt Main Data and Join Optinal Data.
    id_columns = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    df = pd.melt(df, id_vars=id_columns, var_name='d', value_name='sales')
    df = pd.merge(df, calendar, how='left', on='d')
    df = pd.merge(df, sell_prices, how='left', on=['store_id', 'item_id', 'wm_yr_wk'])
    # MEMO: sell_price を直近価格で過去の値を埋める。
    # - 値が入っていない商品はないが、値が入っていない週は存在する。
    # - 欠損値の原因は、欠品なのか、計測漏れなのか理由が定かでないので、安易に保管するのは危険。
    # - 様々な保管方法を試して、後で、Null Importanceで確かめるなどしたほうがよい。
    # - 上記の理由から、今は欠損値のまま扱う。
    # df['sell_price'] = df.groupby('item_id')['sell_price'].bfill()
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
            self.df.to_pickle(self.filepath)

    def check_exist_cahce(self):
        if os.path.exists(self.filepath):
            self.is_exist_cahce = True

    def get_feature(self, df):
        if self.is_exist_cahce:
            self.df = pd.read_pickle(self.filepath)
            return self.df
        else:
            self.df = self.create_feature(df)
            return self.df

    def create_feature(self, df):
        raise NotImplementedError


class AddSalesFeature(BaseFeature):
    def create_feature(self, df):
        DAYS_PRED = 28
        col = 'sales'
        grouped_df = df.groupby(["id"])[col]

        for diff in [0, 1, 2, 4, 5, 6]:
            shift = DAYS_PRED + diff
            df[f"{col}_lag_t{shift}"] = grouped_df.transform(lambda x: x.shift(shift))

        for window in [7, 30, 60, 90, 180]:
            df[f"{col}_rolling_STD_t{window}"] = grouped_df.transform(
                lambda x: x.shift(DAYS_PRED).rolling(window).std())

        for window in [7, 30, 60, 90, 180]:
            df[f"{col}rolling_MEAN_t{window}"] = grouped_df.transform(
                lambda x: x.shift(DAYS_PRED).rolling(window).mean())

        for window in [7, 30, 60]:
            df[f"{col}rolling_MIN_t{window}"] = grouped_df.transform(lambda x: x.shift(DAYS_PRED).rolling(window).min())

        for window in [7, 30, 60]:
            df[f"{col}rolling_MAX_t{window}"] = grouped_df.transform(lambda x: x.shift(DAYS_PRED).rolling(window).max())

        df[f"{col}_rolling_SKEW_t30"] = grouped_df.transform(lambda x: x.shift(DAYS_PRED).rolling(30).skew())
        df[f"{col}_rolling_KURT_t30"] = grouped_df.transform(lambda x: x.shift(DAYS_PRED).rolling(30).kurt())
        return df


class AddPriceFeature(BaseFeature):
    def create_feature(self, df):
        DAYS_PRED = 28
        col = 'sell_price'
        grouped_df = df.groupby(["id"])[col]

        for diff in [0]:
            shift = DAYS_PRED + diff
            df[f"{col}_lag_t{shift}"] = grouped_df.transform(lambda x: x.shift(shift))

        df[f"{col}_rolling_price_MAX_t365"] = grouped_df.transform(lambda x: x.shift(DAYS_PRED).rolling(365).max())
        df[f"{col}_price_change_t365"] = \
            (df[f"{col}_rolling_price_MAX_t365"] - df["sell_price"]) / (df[f"{col}_rolling_price_MAX_t365"])

        df[f"{col}_rolling_price_std_t7"] = grouped_df.transform(lambda x: x.shift(DAYS_PRED).rolling(7).std())
        df[f"{col}_rolling_price_std_t30"] = grouped_df.transform(lambda x: x.shift(DAYS_PRED).rolling(30).std())
        return df.drop([f"{col}_rolling_price_MAX_t365"], axis=1)


class AddWeight(BaseFeature):
    def create_feature(self, df):
        grouped_df = df.groupby(["id"])['sales']
        df['weight'] = grouped_df.transform(lambda x: 1 - (x == 0).rolling(28).mean())
        return df


def create_features(df):
    '''
    # TODO
        - (PRED_INTERVAL + N)日rollingした統計量
        - 翌日休日フラグ・連続休日フラグ
        - 当該月の特徴量
            - 月初・月末（１日）の売上
            - 15, 20日などのクレジットカードの締日のごとの統計量
        - 過去１ヶ月間の（特定item_idの売上 / スーパー全体の売上）
            - （特定item_idの売上個数 / スーパー全体の売上個数）
            - （特定item_idの売上個数 / スーパー全体の売上個数）
    '''
    with AddSalesFeature(filename='add_sales_train', use_cache=True) as feat:
        df = feat.get_feature(df).pipe(reduce_mem_usage)

    with AddPriceFeature(filename='add_price_train', use_cache=True) as feat:
        df = feat.get_feature(df).pipe(reduce_mem_usage)

    with AddWeight(filename='add_weight', use_cache=True) as feat:
        df = feat.get_feature(df).pipe(reduce_mem_usage)
    return df


''' Train Model
'''


def plot_cv_indices(cv, X, y, dt_col, lw=10):
    n_splits = cv.get_n_splits()
    fig, ax = plt.subplots(figsize=(20, n_splits))

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(
            X[dt_col],
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=plt.cm.coolwarm,
            vmin=-0.2,
            vmax=1.2,
        )

    # Formatting
    MIDDLE = 15
    LARGE = 20
    ax.set_xlabel("Datetime", fontsize=LARGE)
    ax.set_xlim([X[dt_col].min(), X[dt_col].max()])
    ax.set_ylabel("CV iteration", fontsize=LARGE)
    ax.set_yticks(np.arange(n_splits) + 0.5)
    ax.set_yticklabels(list(range(n_splits)))
    ax.invert_yaxis()
    ax.tick_params(axis="both", which="major", labelsize=MIDDLE)
    ax.set_title("{}".format(type(cv).__name__), fontsize=LARGE)
    fig.savefig(f"result/cv_split/{VERSION}.png")


class CustomTimeSeriesSplitter:
    def __init__(self, n_splits=5, train_days=80, test_days=20, dt_col="date"):
        self.n_splits = n_splits
        self.train_days = train_days
        self.test_days = test_days
        self.dt_col = dt_col

    def split(self, X, y=None, groups=None):
        sec = (X[self.dt_col] - X[self.dt_col][0]).dt.total_seconds()
        duration = sec.max() - sec.min()

        train_sec = 3600 * 24 * self.train_days
        test_sec = 3600 * 24 * self.test_days
        total_sec = test_sec + train_sec
        step = (duration - total_sec) / (self.n_splits - 1)

        for idx in range(self.n_splits):
            train_start = idx * step
            train_end = train_start + train_sec
            test_end = train_end + test_sec

            if idx == self.n_splits - 1:
                test_mask = sec >= train_end
            else:
                test_mask = (sec >= train_end) & (sec < test_end)

            train_mask = (sec >= train_start) & (sec < train_end)
            test_mask = (sec >= train_end) & (sec < test_end)

            yield sec[train_mask].index.values, sec[test_mask].index.values

    def get_n_splits(self):
        return self.n_splits


def get_feature_importance(models):
    feature_importance = pd.DataFrame(
        [model.feature_importance() for model in models],
        columns=models[0].feature_name()
    ).T

    feature_importance['Agerage_Importance'] = feature_importance.iloc[:, :len(models)].mean(axis=1)
    feature_importance['importance_std'] = feature_importance.iloc[:, :len(models)].std(axis=1)
    feature_importance.sort_values(by='Agerage_Importance', inplace=True)
    return feature_importance


def plot_importance(models, max_num_features=50, figsize=(15, 20)):
    feature_importance = get_feature_importance(models)
    plt.figure(figsize=figsize)

    feature_importance[-max_num_features:].plot(
        kind='barh', title='Feature importance', figsize=figsize,
        y='Agerage_Importance', xerr='importance_std',
        grid=True, align="center"
    )
    plt.savefig(f'result/importance/{VERSION}.png')


def rmsele(preds, actual, weight=None):
    return mean_squared_error(
        np.log1p(actual), np.log1p(preds), sample_weight=weight, squared=False)


def rmsle(preds, data):
    weight = data.get_weight()
    metric_name = 'RMSLE' if weight is None else 'WRMSLE'
    return metric_name, rmsele(preds, data.get_label(), weight), False


def train_lgb(bst_params, fit_params, X, y, cv, drop_when_train=None):
    models = []

    if drop_when_train is None:
        drop_when_train = []

    for idx_fold, (idx_trn, idx_val) in enumerate(cv.split(X, y)):
        print(f"\n---------- Fold: ({idx_fold + 1} / {cv.get_n_splits()}) ----------\n")

        X_trn, X_val = X.iloc[idx_trn], X.iloc[idx_val]
        y_trn, y_val = y.iloc[idx_trn], y.iloc[idx_val]
        train_set = lgb.Dataset(X_trn.drop(drop_when_train, axis=1), label=y_trn, weight=X_trn['weight'])
        val_set = lgb.Dataset(X_val.drop(drop_when_train, axis=1), label=y_val, weight=X_val['weight'])

        model = lgb.train(
            bst_params,
            train_set,
            valid_sets=[train_set, val_set],
            valid_names=["train", "valid"],
            **fit_params,
            feval=rmsle,
        )
        models.append(model)

        del idx_trn, idx_val, X_trn, X_val, y_trn, y_val
        gc.collect()

    plot_importance(models)
    return models


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
        for i in tqdm(range(len(self.train_series))):
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
        for i, group_id in enumerate(tqdm(self.group_ids, leave=False)):
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
        for i, group_id in enumerate(tqdm(self.group_ids, leave=False, disable=dis)):
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


''' Submission
'''


def main():
    print('\n--- Load Data ---\n')
    # TODO: sales_train_evaluation.csv が公開されたらtestに代入
    calendar, submission, train, sell_prices = read_data()
    test = train.sample(10000).copy(deep=True)

    print('\n--- Transform Data ---\n')
    train = hstack_sales_colums(train, last_d=1913)
    test = hstack_sales_colums(train, last_d=1913)
    calendar = encode_calendar(calendar, filename='encoded_calendar', use_cache=True)

    encode_maps = {}
    categorical_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    encode_maps = {col: {label: i for i, label in enumerate(sorted(train[col].unique()))}
                   for col in categorical_cols}

    train = melt_data(train, calendar, sell_prices, encode_maps,
                      filename='melted_train', use_cache=True)

    print('Train DataFrame:', train.shape)
    print(train.head())

    print('\n--- Feature Engineering ---\n')
    train = create_features(train)

    DAYS_PRED = 28
    num_unique_id = train['id'].nunique()
    max_roll_days = 180

    skip_raws = num_unique_id * (max_roll_days + DAYS_PRED)
    train = train.iloc[skip_raws:].reset_index(drop=True)

    print('Train DataFrame:', train.shape)
    print(train.head())

    print('\n--- Train Model ---')
    cv = CustomTimeSeriesSplitter(**{
        "n_splits": 5,
        "train_days": 365 * 2,
        "test_days": 28,
        "dt_col": 'date',
    })
    # Plotting all the points takes long time.
    plot_cv_indices(cv, train.iloc[::1000][['date']].reset_index(drop=True), None, 'date')

    PRED_INTERVAL = 28
    target_col = 'sales'
    features = train.columns.tolist()

    cols_to_drop = ['id', 'wm_yr_wk', 'd', 'date'] + [target_col]
    features = [f for f in features if f not in cols_to_drop]

    latest_date = train['date'].max()
    submit_date = latest_date - datetime.timedelta(days=PRED_INTERVAL)
    submit_mask = (train["date"] > submit_date)

    eval_date = latest_date - datetime.timedelta(days=PRED_INTERVAL * 2)
    eval_mask = ((train["date"] > eval_date) & (train["date"] <= submit_date))

    train_mask = ((~eval_mask) & (~submit_mask))

    X_train, y_train = train[train_mask][['date'] + features], train[train_mask][target_col]
    X_eval, y_eval = train[eval_mask][['date'] + features], train[eval_mask][target_col]
    X_submit, y_submit = train[submit_mask][['date'] + features], train[submit_mask][target_col]

    bst_params = {
        "boosting_type": "gbdt",
        "metric": "None",  # "rmse",
        "objective": "poisson",
        "seed": 11,
        "learning_rate": 0.3,
        'max_depth': 5,
        'num_leaves': 32,
        'min_data_in_leaf': 50,
        "bagging_fraction": 0.8,
        "bagging_freq": 10,
        "feature_fraction": 0.8,
        "verbosity": -1,
    }

    fit_params = {
        "num_boost_round": 100_000,
        "early_stopping_rounds": 50,
        "verbose_eval": 100,
    }

    models = train_lgb(
        bst_params, fit_params, X_train, y_train, cv, drop_when_train=['date', 'weight']
    )

    del X_train, y_train; gc.collect()

    print('\n--- Evaluation ---\n')
    preds = np.mean([m.predict(
        X_eval.drop(['date', 'weight'], axis=1), num_iteration=m.best_iteration)
        for m in models], axis=0)

    rmse = mean_squared_error(y_eval.values, preds, squared=False)
    print(f'RMSE: {rmse}')
    rmsle = rmsele(preds, y_eval.values)
    print(f'RMSLE: {rmsle}')

    train_label_df = train[train_mask][['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'd', 'sales']]
    train_label_df['sales'] = train[train_mask][target_col].astype(int)

    train_label_df = pd.pivot_table(train_label_df,
                                    index=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
                                    columns='d',
                                    values='sales'
                                    ).reset_index()

    def reverse_map(d):
        return {v: k for k, v in d.items()}

    for label, encode_map in encode_maps.items():
        train_label_df[label] = train_label_df[label].map(reverse_map(encode_map))

    fill_cols = train_label_df.columns[train_label_df.columns.str.startswith('d_')]
    train_label_df[fill_cols] = train_label_df[fill_cols].fillna(0).astype(int)

    fill_cols = sorted(fill_cols, key=lambda x: int((re.search(r"\d+", x)).group(0)))
    cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'] + fill_cols
    train_label_df[cols] = train_label_df[cols]

    valid_label_df = train[eval_mask][['id', 'd']]
    valid_label_df['sales'] = train[eval_mask][target_col]

    valid_label_df = pd.pivot(valid_label_df, index='id', columns='d', values='sales').reset_index()
    fill_cols = valid_label_df.columns[valid_label_df.columns.str.startswith('d_')]
    valid_label_df[fill_cols] = valid_label_df[fill_cols].fillna(0).astype(int)

    valid_label_df.drop('id', axis=1, inplace=True)

    valid_pred_df = train[eval_mask][['id', 'date']]
    valid_pred_df['sales'] = preds

    valid_pred_df = pd.pivot(valid_pred_df, index='id', columns='date', values='sales').reset_index()
    # valid_pred_df.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]
    valid_pred_df.columns = ['id'] + valid_label_df.columns.tolist()
    valid_pred_df.fillna(0, inplace=True)

    valid_pred_df.drop('id', axis=1, inplace=True)

    calendar = pd.read_pickle('../data/reduced/calendar.pkl')
    sell_prices = pd.read_pickle('../data/reduced/sell_prices.pkl')
    e = WRMSSEEvaluator(train_label_df, valid_label_df, calendar, sell_prices)

    wrmsse = e.score(valid_pred_df)
    print(f'WRMSSE: {wrmsse}')

    print('\n--- Submission ---\n')
    sub_val_df = train[submit_mask][['id', 'date']]
    sub_val_df['sales'] = np.mean(
        [m.predict(X_submit.drop(['date', 'weight'], axis=1)) for m in models], axis=0)

    sub_val_df = pd.pivot(sub_val_df, index='id', columns='date', values='sales').reset_index()
    sub_val_df.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]

    sub_df = pd.DataFrame()
    sub_df['id'] = submission['id']

    sub_df = pd.merge(sub_df, sub_val_df, how='left', on='id')
    sub_df.fillna(0, inplace=True)
    sub_df.to_csv(f'submit/{VERSION}_{wrmsse:.04f}.csv.gz', index=False, compression='gzip')

    print('Submit DataFrame:', sub_df.shape)
    print(sub_df.head())


if __name__ == '__main__':
    main()
