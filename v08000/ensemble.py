import glob

import numpy as np
import pandas as pd

import lightgbm as lgb

from sklearn.model_selection import train_test_split

from script import load_pickle, dump_pickle

from v08000_baseline import CustomWRMSSE

import warnings
warnings.filterwarnings('ignore')


def main():
    # Submission
    preds_list = []
    preds_list += glob.glob("submit/v08005/*")
    preds_list += glob.glob("submit/v08007/*")

    submit = pd.read_pickle('../data/reduced/sample_submission.pkl')
    for i, f in enumerate(preds_list):
        tmp_df = pd.read_csv(f, compression='gzip')
        submit.iloc[:, 1:] += tmp_df.iloc[:, 1:]

    submit.iloc[:, 1:] = submit.iloc[:, 1:] / len(preds_list)

    # Evaluation
    eval_data = submit.iloc[:30490, 1:].values
    evaluator = load_pickle('features/evaluator.pkl')
    score = evaluator.score(eval_data)
    print(f'Averaged WRMSSE Score: {score}')

    # Dump Submission file.
    submit.to_csv(
        f'submit/ensamble_{score:.05}.csv.gz',
        index=False,
        compression='gzip'
    )

    print(submit.shape)
    print(submit.head())


if __name__ == '__main__':
    main()
