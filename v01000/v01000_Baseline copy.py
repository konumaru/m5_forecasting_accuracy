import os
import gc
import re
import datetime
from tqdm import tqdm

import numpy as np
import pandas as pd

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


def check_dir_exist(dirs):
    for path in dirs:
        if os.path.exists(path):
            continue
        else:
            print(f'{path} is not exist, so create it.')
            os.mkdir(path)


'''Transform Initial Data
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


def encode_calendar(df, filename='encoded_calendar', use_cache=True):
    filepath = f'features/{filename}.pkl'

    if use_cache and os.path.exists(filepath):
        return pd.read_pickle(filepath)

    # Drop Columns
    cols_to_drop = ['weekday', 'year']
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


def melt_data(df, calendar, sell_prices, encode_maps, filepath, use_cache=True):
    filepath = f'features/{filepath}.pkl'

    if use_cache and os.path.exists(filepath):
        return pd.read_pickle(filepath)

    # MEMO: ラベルは全データ共通なので、train/test/sell_prices のmappingに使える。
    for label, encode_map in encode_maps.items():
        df[label] = df[label].map(encode_map)
        if label in ['item_id', 'store_id']:
            sell_prices[label] = sell_prices[label].map(encode_map)

    # melt and join data
    id_columns = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    df = pd.melt(df, id_vars=id_columns, var_name='d', value_name='sales')
    df = pd.merge(df, calendar, how='left', on='d')
    df = pd.merge(df, sell_prices, how='left', on=['store_id', 'item_id', 'wm_yr_wk'])
    # MEMO: sell_price を直近価格で過去の値を埋める。
    # df['sell_price'] = df.groupby('item_id')['sell_price'].bfill()
    df = df.pipe(reduce_mem_usage)
    df.to_pickle(filepath)
    return df


''' Create Features
'''
#     print('')
#     print('\tADD Roll SLOPE Feature', end='')
#     slope_func = lambda x: linregress(np.arange(len(x)), x)[0]
#     df[f"{col}rolling_SLOPE_t30"] = grouped_df.transform(
#           lambda x: x.shift(DAYS_PRED).rolling(30).agg(slope_func))


def add_sales_features(df):
    print('Create sales Features: ')
    DAYS_PRED = 28
    col = 'sales'
    grouped_df = df.groupby(["id"])[col]
    print('\tAdd Shift Feature', end='')
    for diff in [0, 1, 2]:
        print(f' {diff}', end='')
        shift = DAYS_PRED + diff
        df[f"{col}_shift_t{shift}"] = grouped_df.transform(lambda x: x.shift(shift))

    print('')
    print('\tADD Roll STD Feature', end='')
    for window in [7, 30, 60, 90, 180]:
        print(f' {window}', end='')
        df[f"{col}_rolling_STD_t{window}"] = grouped_df.transform(lambda x: x.shift(DAYS_PRED).rolling(window).std())

    print('')
    print('\tADD Roll MEAN Feature', end='')
    for window in [7, 30, 60, 90, 180]:
        print(f' {window}', end='')
        df[f"{col}rolling_MEAN_t{window}"] = grouped_df.transform(lambda x: x.shift(DAYS_PRED).rolling(window).mean())

    print('')
    print('\tADD Roll MIN Feature', end='')
    for window in [7, 30, 60]:
        print(f' {window}', end='')
        df[f"{col}rolling_MIN_t{window}"] = grouped_df.transform(lambda x: x.shift(DAYS_PRED).rolling(window).min())

    print('')
    print('\tADD Roll MAX Feature', end='')
    for window in [7, 30, 60]:
        print(f' {window}', end='')
        df[f"{col}rolling_MAX_t{window}"] = grouped_df.transform(lambda x: x.shift(DAYS_PRED).rolling(window).max())

    print('')
    print('\tADD Roll SKEW and KURT Feature', end='')
    df[f"{col}rolling_SKEW_t30"] = grouped_df.transform(lambda x: x.shift(DAYS_PRED).rolling(30).skew())
    df[f"{col}rolling_KURT_t30"] = grouped_df.transform(lambda x: x.shift(DAYS_PRED).rolling(30).kurt())

    print('')
    print('\tAll Feature Created')
    return df


def add_price_features(df):
    col = 'sell_price'
    grouped_df = df.groupby(["id"])[col]
    df[f"{col}_shift_price_t1"] = grouped_df.transform(lambda x: x.shift(1))
    df[f"{col}_rolling_price_MAX_t365"] = grouped_df.transform(lambda x: x.shift(1).rolling(365).max())

    df[f"{col}__change_t1"] = (df[f"{col}_shift_price_t1"] - df[col]) / (df[f"{col}_shift_price_t1"])
    df[f"{col}__change_t365"] = (df[f"{col}_rolling_price_MAX_t365"] - df[col]
                                 ) / (df[f"{col}_rolling_price_MAX_t365"])
    df[f"{col}_rolling_price_std_t7"] = grouped_df.transform(lambda x: x.rolling(7).std())
    df[f"{col}_rolling_price_std_t30"] = grouped_df.transform(lambda x: x.rolling(30).std())
    return df.drop([f"{col}_rolling_price_MAX_t365", f"{col}_shift_price_t1"], axis=1)


def create_features(df, filename, use_cache=True):
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
    filepath = f'features/{filename}.pkl'

    if use_cache and os.path.exists(filepath):
        return pd.read_pickle(filepath)\

    df = add_sales_features(df).pipe(reduce_mem_usage)
    # df = add_price_features(df).pipe(reduce_mem_usage)

    df.dropna(axis=0, inplace=True)
    df.to_pickle(filepath)
    return df


''' Train Model
'''


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
    fig.savefig("figure/cv_indices.png")


''' Evaluation Model
'''

''' Submission
'''


def main():
    check_dir_exist(dirs=[
        'features', 'result/importance', 'result/scores', 'submit'
    ])

    # print('\n--- Load Data ---\n')
    # # TODO: sales_train_evaluation.csv が公開されたらtestに代入
    # calendar, submission, train, sell_prices = read_data()
    # test = train.sample(10000).copy(deep=True)

    # print('\n--- Transform Initial Data ---\n')
    # calendar = encode_calendar(calendar, filename='encoded_calendar', use_cache=True)

    # encode_maps = {}
    # categorical_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    # encode_maps = {col: {label: i for i, label in enumerate(sorted(train[col].unique()))}
    #                for col in categorical_cols}
    # train = melt_data(train, calendar, sell_prices, encode_maps,
    #                   filepath='melted_train', use_cache=False)
    # test = melt_data(test, calendar, sell_prices, encode_maps,
    #                  filepath='melted_test', use_cache=False)

    # del calendar, sell_prices, test; gc.collect()

    # print('\n--- Create Features ---\n')
    # train = create_features(train, filename='train_created_feature', use_cache=False)

    # print(train.shape)
    # print(train.head())

    # print('\n--- Train Model ---')
    # cv_params = {
    #     "n_splits": 5,
    #     "train_days": 365 * 2,
    #     "test_days": 28,
    #     "dt_col": 'date',
    # }
    # cv = CustomTimeSeriesSplitter(**cv_params)
    # plot_cv_indices(cv, train.iloc[::1000][['date']].reset_index(drop=True), None, 'date')
    # TODO:
    # - dataをtrain, validation, evaluation, （今回のみの）submissionに分ける。
    # - LightGBMで学習する。
    #   - model, feature importance, train_loss（一旦RMSEでよい。） がわかるようにする。

    # print('\n--- Prediction ---\n')
    # TODO: Train Modelで手に入ったモデルを使って、testデータで予測するが、今回はない。
    # test = pd.read_pickle('features/melted_test.pkl')

    # print('\n--- Submission ---\n')


if __name__ == '__main__':
    main()
