# Import standard modules
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
from pathlib import Path


BASE_DIR = Path(__file__).absolute().parent.parent # Uncomment for python files
#BASE_DIR = os.path.dirname(os.getcwd()) # Comment for python files
sys.path.insert(0, str(Path(BASE_DIR) / 'utils'))
print(BASE_DIR)
import config
from treasuries import *

# Global variables
DATA_DIR = Path(config.DATA_DIR)
DATA_DIR.mkdir(parents=True, exist_ok=True)
QUOTE_DATE = '2024-10-31'

FILE_PATH = Path(config.DATA_DIR) / f"treasury_quotes_{QUOTE_DATE}.xlsx"


# Extract data:
RESTRICT_YLD = True
RESTRICT_TIPS = True

treasuries_all = pd.read_excel(FILE_PATH, sheet_name='quotes')
t_curr_date = treasuries_all['quote date'].values[0]
treasuries_restr = filter_treasuries(treasuries_all, date_curr=t_curr_date, filter_tips=RESTRICT_TIPS, filter_yld=RESTRICT_YLD)

cf_restr = calc_cashflows(treasuries_restr)
cf_ols = filter_treasury_cashflows(cf_restr, filter_benchmark_dates=True, filter_maturity_dates=True, filter_cf_strict=True)

# Perform OLS
prices_filtered = treasuries_restr['price'][cf_ols.index]
#param_OLS = estimate_rate_curve(model_name='OLS', cf=cf_ols, prices=prices_filtered, date_current=t_curr_date)
#int_rates = discount_to_int_rate(discount=param_OLS[1], time_to_maturity=param_OLS[0], n_compound=2)

param_OLS = estimate_rate_curve(model_name='nelson_siegel', cf=cf_ols, prices=prices_filtered, date_current=t_curr_date)
int_rates = discount_to_int_rate(discount=param_OLS[1], time_to_maturity=param_OLS[0], n_compound=2)
plt.plot(param_OLS[0], int_rates)