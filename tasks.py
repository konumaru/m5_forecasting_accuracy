import os
import re

import numpy as np
import pandas as pd


def hello():
    print("Hello Poetry !!")


def CreateNewExperiment():
    def get_version_num(dir_name: str):
        match = re.match(r'v(\d{2})\d{3}', dir_name)
        if match:
            return int(match.groups()[0])
        else:
            return 0

    top_dirs = os.listdir()
    versions = [get_version_num(d) for d in top_dirs]
    next_version = str(max(versions) + 1)
    new_exp_path = 'v' + next_version.zfill(2) + '000'

    os.makedirs(new_exp_path, exist_ok=True)

    with open(os.path.join(new_exp_path, 'version_reference.md'), 'w') as f:
        f.write('# Version Reference')
    print(f'Create {new_exp_path} Experiment Directory')

    def check_dir_exist(dirs):
        for path in dirs:
            if os.path.exists(path):
                continue
            else:
                dir_path = os.path.join(new_exp_path, path)
                print(f'{dir_path} is not exist, so create it.')
                os.makedirs(dir_path, exist_ok=True)
                with open(dir_path + '/.gitkeep', 'w') as f:
                    f.write('')

    check_dir_exist(dirs=[
        'features', 'submit', 'result/importance', 'result/score', 'result/log', 'result/model',
        'result/feature_cols', 'result/evaluation'
    ])


def ReduceData():
    SRC_FILES = [
        'calendar.csv',
        'sales_train_validation.csv',
        'sales_train_evaluation.csv',
        'sample_submission.csv',
        'sell_prices.csv'
    ]

    def reduce_mem_usage(df, verbose=True):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = df.memory_usage().sum() / 1024**2
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
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
        end_mem = df.memory_usage().sum() / 1024**2
        if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)\n'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))
        return df

    for src_file in SRC_FILES:
        print(f'Processing {src_file}')
        # Load and Transform
        raw_df = pd.read_csv('./data/raw/' + src_file)
        reduced_df = reduce_mem_usage(raw_df)
        # Export
        os.makedirs('./data/reduced/', exist_ok=True)
        filename = src_file.split('.')[0]
        reduced_df.to_pickle('./data/reduced/' + filename + '.pkl')
