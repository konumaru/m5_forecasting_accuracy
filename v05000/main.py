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


def main():
    print('this is main fucntion.')


if __name__ == '__main__':
    main()
