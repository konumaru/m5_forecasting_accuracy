import os
import sys
sys.path.append('..')

import warnings
warnings.filterwarnings('ignore')

import yaml
import datetime

import numpy as np
import pandas as pd

from model import Model, GroupKfoldModel, rmsle
from utils import get_latest_version_num, load_config, dump_config,\
    load_dataset, export_submission_file
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.model_selection import train_test_split

LOCAL_TEST = False
# Experiment Major Version
VERSION = 1000

seed = 42
now = datetime.datetime.now()


def main():
    # Set Experiment Version
    sub_version = get_latest_version_num(VERSION)
    # Laod Config
    config = load_config()
    # Laod Data
    X_train_all, y_train_all, X_test = load_dataset(config['dataset'])

    if LOCAL_TEST:
        config['params']['bagging_fraction'] = 0.01

    # Define And Train Model
    model = Model(model_type=config['model_ref'])
    y_preds, scores, models = model.train_and_predict(
        config['params'], config['train_params'],
        X_train_all, y_train_all, X_test,
        n_fold=3, is_shuffle=True, seed=seed
    )

    model.save_feature_importance(filepath=f'v{VERSION}/{sub_version}')
    cv_score = np.mean(scores)

    # Prediction And Eport Submission file
    sub_df = pd.DataFrame()
    sub_df[config['ID_name']] = pd.read_pickle('./data/input/test.pkl')[config['ID_name']]
    sub_df[config['target_name']] = np.clip(np.mean(y_preds, axis=0), 0.0, None)

    print(f'\nExport Submission file ...')
    sub_dir = f'./data/submit/v{VERSION}'
    sub_filename = f'sub_{now:%Y%m%d%H%M}_{sub_version}_{model.model_type}_{cv_score:02f}.csv'
    export_submission_file(sub_df, os.path.join(sub_dir, sub_filename))

    # Dump config
    filepath = f'./config/version/v{VERSION}/{sub_version}_{model.model_type}_{cv_score:02f}.yml'
    dump_config(config, filepath)

    # Print Submit Message

    print(
        f'''
        kaggle competitions submit -c ashrae-energy-prediction \
            -f {os.path.join(sub_dir, sub_filename)} \
            -m "{config['note']}"
        '''
    )


if __name__ == '__main__':
    main()
