I have a repository containing modularized Python files, each focusing on specific tasks. Functions are compact, well-documented with type hints, and include helper functions for reusable logic.

I will show examples from portfolio_management.py and its supporting data_utils.py. Review these files to identify key patterns in coding style, modularization, and documentation.

In the next prompt, I'll provide several functions meant for treasuries.py. These need to be refactored to match the best practices observed in my modules. Focus on:

1. Code readability and conciseness.
2. Consistent type hinting and documentation.
3. Proper error handling and efficient use of imports.
4. Ensure the new treasuries.py integrates seamlessly with the existing codebase.


`data_utils.py`:
```python
import datetime
import logging
import re
import sys
import zipfile
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

BASE_DIR = Path(__file__).absolute().parent.parent.parent
sys.path.insert(0, str(Path(BASE_DIR) / 'src'))

# Local imports
import config


# =============================================================================
# Global Configuration
# =============================================================================
RAW_DATA_DIR = Path(config.RAW_DATA_DIR)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = Path(config.PROCESSED_DATA_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Helper Functions (Caching, Reading/Writing Files)
# =============================================================================

def save_figure(
        fig: plt.Figure,
        plot_name_prefix: str,
) -> None:
    """
    Saves a matplotlib figure to a PNG file if save_plot is True.
    The filename pattern is "<prefix>_YYYYMMDD_HHMMSS.png".

    Parameters:
    fig (plt.Figure): The matplotlib figure to save.
    plot_name_prefix (str): The prefix for the plot filename.
    """
    filename  = f"{plot_name_prefix}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png"
    plot_path = OUTPUT_DIR / filename
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    logging.info(f"Plot saved to {plot_path}")


def read_excel_default(excel_name: str,
                       sheet_name: str = None, 
                       index_col : int = 0,
                       parse_dates: bool =True,
                       print_sheets: bool = False,
                       **kwargs):
    """
    Reads an Excel file and returns a DataFrame with specified options.

    Parameters:
    excel_name (str): The path to the Excel file.
    index_col (int, default=0): Column to use as the row index labels of the DataFrame.
    parse_dates (bool, default=True): Boolean to parse dates.
    print_sheets (bool, default=False): If True, prints the names and first few rows of all sheets.
    sheet_name (str or int, default=None): Name or index of the sheet to read. If None, reads the first sheet.
    **kwargs: Additional arguments passed to `pd.read_excel`.

    Returns:
    pd.DataFrame: DataFrame containing the data from the specified Excel sheet.

    Notes:
    - If `print_sheets` is True, the function will print the names and first few rows of all sheets and return None.
    - The function ensures that the index name is set to 'date' if the index column name is 'date', 'dates' or 'datatime', or if the index contains date-like values.
    """

    if print_sheets:
        excel_file = pd.ExcelFile(excel_name)  # Load the Excel file to get sheet names
        sheet_names = excel_file.sheet_names
        n = 0
        while True:
            try:
                sheet = pd.read_excel(excel_name, sheet_name=n)
                print(f'Sheet name: {sheet_names[n]}')
                print("Columns: " + ", ".join(list(sheet.columns)))
                print(sheet.head(3))
                n += 1
                print('-' * 70)
                print('\n')
            except:
                return
    sheet_name = 0 if sheet_name is None else sheet_name
    df = pd.read_excel(excel_name, index_col=index_col, parse_dates=parse_dates,  sheet_name=sheet_name, **kwargs)
    df.columns = [col.lower() for col in df.columns]
    if df.index.name is not None:
        if df.index.name in ['date', 'dates', 'datetime']:
            df.index.name = 'date'
    elif isinstance(df.index[0], (datetime.date, datetime.datetime)):
        df.index.name = 'date'
    return df


def read_csv_default(csv_name: str,
                     index_col: int = 0,
                     parse_dates: bool = True,
                     print_data: bool = False,
                     keep_cols: Union[List, str] = None,
                     drop_cols: Union[List, str] = None,
                     **kwargs):
    """
    Reads a CSV file and returns a DataFrame with specified options.

    Parameters:
    csv_name (str): The path to the CSV file.
    index_col (int, default=0): Column to use as the row index labels of the DataFrame.
    parse_dates (bool, default=True): Boolean to parse dates.
    print_data (bool, default=False): If True, prints the first few rows of the DataFrame.
    keep_cols (list or str, default=None): Columns to read from the CSV file.
    drop_cols (list or str, default=None): Columns to drop from the DataFrame.
    **kwargs: Additional arguments passed to `pd.read_csv`.

    Returns:
    pd.DataFrame: DataFrame containing the data from the CSV file.

    Notes:
    - The function ensures that the index name is set to 'date' if the index column name is 'date', 'dates' or 'datatime', or if the index contains date-like values.
    """

    df = pd.read_csv(csv_name, index_col=index_col, parse_dates=parse_dates, **kwargs)
    df.columns = [col.lower() for col in df.columns]

    # Filter columns if keep_cols is specified
    if keep_cols is not None:
        if isinstance(keep_cols, str):
            keep_cols = [keep_cols]
        df = df[keep_cols]

    # Drop columns if drop_cols is specified
    if drop_cols is not None:
        if isinstance(drop_cols, str):
            drop_cols = [drop_cols]
        df = df.drop(columns=drop_cols, errors='ignore')

    # Print data if print_data is True
    if print_data:
        print("Columns:", ", ".join(df.columns))
        print(df.head(3))
        print('-' * 70)
    
    # Set index name to 'date' if appropriate
    if df.index.name is not None:
        if df.index.name in ['date', 'dates', 'datetime']:
            df.index.name = 'date'
    elif isinstance(df.index[0], (datetime.date, datetime.datetime)):
        df.index.name = 'date'
    
    return df


# =============================================================================
# Helper Functions (Manipulating DataFrames)
# =============================================================================

def time_series_to_df(returns: Union[pd.DataFrame, pd.Series, List[pd.Series]], name: str = "Returns"):
    """
    Converts returns to a DataFrame if it is a Series or a list of Series.

    Parameters:
    returns (pd.DataFrame, pd.Series or List or pd.Series): Time series of returns.

    Returns:
    pd.DataFrame: DataFrame of returns.
    """
    if isinstance(returns, pd.DataFrame):
        returns = returns.copy()
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()
    elif isinstance(returns, list):
        returns_list = returns.copy()
        returns = pd.DataFrame({})

        for series in returns_list:
            if isinstance(series, pd.Series):
                returns = returns.merge(series, right_index=True, left_index=True, how='outer')
            else:
                raise TypeError(f'{name} must be either a pd.DataFrame or a list of pd.Series')
            
    # Convert returns to float
    try:
        returns = returns.apply(lambda x: x.astype(float))
    except ValueError:
        print(f'Could not convert {name} to float. Check if there are any non-numeric values')
        pass

    return returns


def fix_dates_index(returns: pd.DataFrame):
    """
    Fixes the date index of a DataFrame if it is not in datetime format and convert returns to float.

    Parameters:
    returns (pd.DataFrame): DataFrame of returns.

    Returns:
    pd.DataFrame: DataFrame with datetime index.
    """
    # Check if 'date' is in the columns and set it as the index

    # Set index name to 'date' if appropriate
    
    if returns.index.name is not None:
        if returns.index.name.lower() in ['date', 'dates', 'datetime']:
            returns.index.name = 'date'
    elif isinstance(returns.index[0], (datetime.date, datetime.datetime)):
        returns.index.name = 'date'
    elif 'date' in returns.columns.str.lower():
        returns = returns.rename({'Date': 'date'}, axis=1)
        returns = returns.set_index('date')
    elif 'datetime' in returns.columns.str.lower():
        returns = returns.rename({'Datetime': 'date'}, axis=1)
        returns = returns.rename({'datetime': 'date'}, axis=1)
        returns = returns.set_index('date')

    # Convert dates to datetime if not already in datetime format or if minutes are 0
    try:
        returns.index = pd.to_datetime(returns.index, utc=True)
    except ValueError:
        print('Could not convert the index to datetime. Check the index format for invalid dates.')
    if not isinstance(returns.index, pd.DatetimeIndex) or (returns.index.minute == 0).all():
        returns.index = pd.to_datetime(returns.index.map(lambda x: x.date()))
        
    # Convert returns to float
    try:
        returns = returns.apply(lambda x: x.astype(float))
    except ValueError:
        print('Could not convert returns to float. Check if there are any non-numeric values')
        pass

    return returns


def filter_columns_and_indexes(
    df: pd.DataFrame,
    keep_columns: Union[list, str],
    drop_columns: Union[list, str],
    keep_indexes: Union[list, str],
    drop_indexes: Union[list, str]
):
    """
    Filters a DataFrame based on specified columns and indexes.

    Parameters:
    df (pd.DataFrame): DataFrame to be filtered.
    keep_columns (list or str): Columns to keep in the DataFrame.
    drop_columns (list or str): Columns to drop from the DataFrame.
    keep_indexes (list or str): Indexes to keep in the DataFrame.
    drop_indexes (list or str): Indexes to drop from the DataFrame.

    Returns:
    pd.DataFrame: The filtered DataFrame.
    """

    if not isinstance(df, (pd.DataFrame, pd.Series)):
        return df
    
    df = df.copy()

    # Columns
    if keep_columns is not None:
        keep_columns = [re.escape(col) for col in keep_columns]
        keep_columns = "(?i).*(" + "|".join(keep_columns) + ").*" if isinstance(keep_columns, list) else "(?i).*" + keep_columns + ".*"
        df = df.filter(regex=keep_columns)
        if drop_columns is not None:
            print('Both "keep_columns" and "drop_columns" were specified. "drop_columns" will be ignored.')

    elif drop_columns is not None:
        drop_columns = [re.escape(col) for col in drop_columns]
        drop_columns = "(?i).*(" + "|".join(drop_columns) + ").*" if isinstance(drop_columns, list) else "(?i).*" + drop_columns + ".*"
        df = df.drop(columns=df.filter(regex=drop_columns).columns)

    # Indexes
    if keep_indexes is not None:
        keep_indexes = [re.escape(col) for col in keep_indexes]
        keep_indexes = "(?i).*(" + "|".join(keep_indexes) + ").*" if isinstance(keep_indexes, list) else "(?i).*" + keep_indexes + ".*"
        df = df.filter(regex=keep_indexes, axis=0)
        if drop_indexes is not None:
            print('Both "keep_indexes" and "drop_indexes" were specified. "drop_indexes" will be ignored.')

    elif drop_indexes is not None:
        drop_indexes = [re.escape(col) for col in drop_indexes]
        drop_indexes = "(?i).*(" + "|".join(drop_indexes) + ").*" if isinstance(drop_indexes, list) else "(?i).*" + drop_indexes + ".*"
        df = df.filter(regex=keep_indexes, axis=0)
    
    return df
```

`portfolio_management.py`:
```python
import datetime
import math
import re
import sys
from pathlib import Path
from typing import Callable, List, Union, model

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from arch import arch_model
from scipy.optimize import minimize
from scipy.stats import kurtosis, norm, skew
import statsmodels.api as sm

BASE_DIR = Path(__file__).absolute().parent.parent.parent
sys.path.insert(0, str(Path(BASE_DIR) / 'src'))

# Local imports
import config
from utils.data_utils import *

# =============================================================================
# Global Configuration
# =============================================================================

PLOT_WIDTH, PLOT_HEIGHT = 12, 8

# =============================================================================
# Portfolio Management Functions
# =============================================================================

def plot_cumulative_returns(
    cumulative_returns: Union[pd.DataFrame, pd.Series],
    name: str = None,
    save_plot: bool = False,
    plot_name: str = None
) -> None:
    """
    Plots cumulative returns from a time series of cumulative returns and optionally saves the plot.

    Parameters:
    -----------
    cumulative_returns (pd.DataFrame or pd.Series): Time series of cumulative returns.
    name (str, optional): Name for the title of the plot.
    save_plot (bool, default=False): If True, saves the plot as a PNG.
    plot_name (str, optional): Custom filename prefix. If None, defaults to 'cumulative_returns'.

    Returns: None
    """

    # Handle index formatting for hours, minutes, and seconds
    indexes_cum_ret = cumulative_returns.index
    if indexes_cum_ret[0].hour == 0 and indexes_cum_ret[0].minute == 0 and indexes_cum_ret[0].second == 0:
        formatted_index = indexes_cum_ret.strftime('%Y-%m-%d')
    else:
        formatted_index = indexes_cum_ret.strftime('%Y-%m-%d\n%H:%M:%S')

    continuous_index = range(len(indexes_cum_ret))

    # Plot cumulative returns
    fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    if isinstance(cumulative_returns, pd.Series):
        ax.plot(continuous_index, cumulative_returns, label='Cumulative Returns', linewidth=1.5, color='blue')
    elif isinstance(cumulative_returns, pd.DataFrame):
        for column in cumulative_returns.columns:
            ax.plot(continuous_index, cumulative_returns[column], label=column, linewidth=1.5)
    else:
        raise ValueError("`cumulative_returns` must be a pandas DataFrame or Series.")

    # Format x-axis with formatted dates
    num_ticks = 20
    tick_indices = np.linspace(0, len(continuous_index) - 1, num=num_ticks, dtype=int)
    tick_labels = [formatted_index[i] for i in tick_indices]
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=8)
    ax.set_xlim([0, len(formatted_index) - 1])

    # Add percentage formatting for y-axis
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=2))

    # Add zero line
    ax.axhline(0, color='darkgrey', linewidth=1, linestyle='-')
    ax.set_title(f'Cumulative Returns {name}' if name else 'Cumulative Returns', fontsize=14)

    # Style grid and background
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.legend(fontsize=10)

    # Show the plot
    plt.show()

    # If save_plot is True, save the figure
    if save_plot:
        plot_name_prefix = plot_name if plot_name else (f'cumulative_returns_{name}' if name else 'cumulative_returns')
        save_figure(fig, plot_name_prefix)



def calc_cumulative_returns(
    returns: Union[pd.DataFrame, pd.Series, List[pd.Series]],
    plot_returns: bool = True,
    name: str = None,
    return_series: bool = True,
    timeframes: Union[None, dict] = None
) -> Union[pd.DataFrame, pd.Series]:
    """
    Calculates cumulative returns from a time series of returns.

    Parameters:
    returns (pd.DataFrame, pd.Series or List or pd.Series): Time series of returns.
    plot_returns (bool, default=True): If True, plots the cumulative returns.
    name (str, default=None): Name for the cumulative return series.
    return_series (bool, default=True): If True, returns the cumulative returns as a Series.
    timeframes (dict or None, default=None): Dictionary of timeframes to calculate cumulative returns for each period.
    
    Returns:
    pd.DataFrame: Returns cumulative returns DataFrame
    """

    returns = time_series_to_df(returns)  # Convert returns to DataFrame if it is a Series or a list of Series
    fix_dates_index(returns)  # Fix the date index of the DataFrame if it is not in datetime format and convert returns to float

    if timeframes is not None:
        results = {}
        for name, timeframe in timeframes.items():
            if timeframe[0] and timeframe[1]:
                timeframe_returns = returns.loc[timeframe[0]:timeframe[1]]
            elif timeframe[0]:
                timeframe_returns = returns.loc[timeframe[0]:]
            elif timeframe[1]:
                timeframe_returns = returns.loc[:timeframe[1]]
            else:
                timeframe_returns = returns.copy()

            if len(timeframe_returns.index) == 0:
                raise Exception(f'No returns data for {name} timeframe')

            cumulative_returns = calc_cumulative_returns(
                timeframe_returns,
                return_series=True,
                plot_returns=plot_returns,
                name=name,
                timeframes=None
            )
            results[name] = cumulative_returns
        return results

    cumulative_returns = (1 + returns).cumprod() - 1

    if plot_returns:
        plot_cumulative_returns(cumulative_returns, name=name)

    if return_series:
        return cumulative_returns
    

def calc_returns_statistics(
    returns: Union[pd.DataFrame, pd.Series, List[pd.Series]],
    annual_factor: int = None,
    provided_excess_returns: bool = None,
    rf_returns: Union[pd.Series, pd.DataFrame] = None,
    var_quantile: Union[float , List] = .05,
    timeframes: Union[None, dict] = None,
    return_tangency_weights: bool = False,
    correlations: Union[bool, List] = False,
    tail_risks: bool = True,
    keep_columns: Union[list, str] = None,
    drop_columns: Union[list, str] = None,
    keep_indexes: Union[list, str] = None,
    drop_indexes: Union[list, str] = None,
    _timeframe_name: str = None,
) -> pd.DataFrame:
    """
    Calculates summary statistics for a time series of returns.   

    Parameters:
    returns (pd.DataFrame, pd.Series or List or pd.Series): Time series of returns.
    annual_factor (int, default=None): Factor for annualizing returns.
    provided_excess_returns (bool, default=None): Whether excess returns are already provided.
    rf (pd.Series or pd.DataFrame, default=None): Risk-free rate data.
    var_quantile (float or list, default=0.05): Quantile for Value at Risk (VaR) calculation.
    timeframes (dict or None, default=None): Dictionary of timeframes [start, finish] to calculate statistics for each period.
    return_tangency_weights (bool, default=True): If True, returns tangency portfolio weights.
    correlations (bool or list, default=True): If True, returns correlations, or specify columns for correlations.
    tail_risks (bool, default=True): If True, include tail risk statistics.
    keep_columns (list or str, default=None): Columns to keep in the resulting DataFrame.
    drop_columns (list or str, default=None): Columns to drop from the resulting DataFrame.
    keep_indexes (list or str, default=None): Indexes to keep in the resulting DataFrame.
    drop_indexes (list or str, default=None): Indexes to drop from the resulting DataFrame.

    Returns:
    pd.DataFrame: Summary statistics of the returns.
    """

    returns = time_series_to_df(returns) # Convert returns to DataFrame if it is a Series or a list of Series
    fix_dates_index(returns) # Fix the date index of the DataFrame if it is not in datetime format and convert returns to float

    if rf_returns is not None:
        rf_returns = time_series_to_df(rf_returns) # Convert returns to DataFrame if it is a Series
        fix_dates_index(rf_returns) # Fix the date index of the DataFrame if it is not in datetime format and convert returns to float
        rf_returns = rf_returns.reindex(returns.index).dropna()
        
        if len(rf_returns.index) != len(returns.index):
            raise Exception('"rf_returns" has missing data to match "returns" index')
        if type(rf_returns) == pd.DataFrame:
            rf = rf_returns.iloc[:, 0].to_list()
        elif type(rf_returns) == pd.Series:
            rf = rf_returns.to_list()

    # Assume annualization factor of 12 for monthly returns if None and notify user
    if annual_factor is None:
        print('Assuming monthly returns with annualization term of 12')
        annual_factor = 12

    
    if keep_columns is None:
        keep_columns = ['Accumulated Return', 'Annualized Mean', 'Annualized Vol', 'Annualized Sharpe', 'Min', 'Mean', 'Max', 'Correlation']
        if tail_risks == True:
            keep_columns += ['Skewness', 'Excess Kurtosis', f'Historical VaR', f'Annualized Historical VaR', 
                                f'Historical CVaR', f'Annualized Historical CVaR', 'Max Drawdown', 
                                'Peak Date', 'Bottom Date', 'Recovery', 'Duration (days)']
    if return_tangency_weights == True:
        keep_columns += ['Tangency Portfolio']
    if correlations != False:
        keep_columns += ['Correlation']

    # Iterate to calculate statistics for multiple timeframes
    if isinstance(timeframes, dict):
        all_timeframes_summary_statistics = pd.DataFrame({})
        for name, timeframe in timeframes.items():
            if timeframe[0] and timeframe[1]:
                timeframe_returns = returns.loc[timeframe[0]:timeframe[1]]
            elif timeframe[0]:
                timeframe_returns = returns.loc[timeframe[0]:]
            elif timeframe[1]:
                timeframe_returns = returns.loc[:timeframe[1]]
            else:
                timeframe_returns = returns.copy()
            if len(timeframe_returns.index) == 0:
                raise Exception(f'No returns data for {name} timeframe')
            
            timeframe_returns = timeframe_returns.rename(columns=lambda col: col + f' ({name})')
            timeframe_summary_statistics = calc_returns_statistics(
                returns=timeframe_returns,
                annual_factor=annual_factor,
                provided_excess_returns=provided_excess_returns,
                rf_returns=rf_returns,
                var_quantile=var_quantile,
                timeframes=None,
                return_tangency_weights=return_tangency_weights,
                correlations=correlations,
                tail_risks=tail_risks,
                _timeframe_name=name,
                keep_columns=keep_columns,
                drop_columns=drop_columns,
                keep_indexes=keep_indexes,
                drop_indexes=drop_indexes
            )
            all_timeframes_summary_statistics = pd.concat(
                [all_timeframes_summary_statistics, timeframe_summary_statistics],
                axis=0
            )
        return all_timeframes_summary_statistics

    # Calculate summary statistics for a single timeframe
    summary_statistics = pd.DataFrame(index=returns.columns)
    summary_statistics['Mean'] = returns.mean()
    summary_statistics['Annualized Mean'] = returns.mean() * annual_factor
    summary_statistics['Vol'] = returns.std()
    summary_statistics['Annualized Vol'] = returns.std() * np.sqrt(annual_factor)
    if provided_excess_returns is True:
        if rf_returns is not None:
            print('Excess returns and risk-free were both provided.'
                ' Excess returns will be consider as is, and risk-free rate given will be ignored.\n')
        summary_statistics['Sharpe'] = returns.mean() / returns.std()
    else:
        try:
            if rf_returns is None:
                print('No risk-free rate provided. Interpret "Sharpe" as "Mean/Volatility".\n')
                summary_statistics['Sharpe'] = returns.mean() / returns.std()
            else:
                excess_returns = returns.subtract(rf_returns.iloc[:, 0], axis=0)
                summary_statistics['Sharpe'] = excess_returns.mean() / returns.std()
        except Exception as e:
            print(f'Could not calculate Sharpe: {e}')

    summary_statistics['Annualized Sharpe'] = summary_statistics['Sharpe'] * np.sqrt(annual_factor)
    summary_statistics['Min'] = returns.min()
    summary_statistics['Max'] = returns.max()

    summary_statistics['Win Rate'] = (returns > 0).mean()
    
    if tail_risks == True:
        tail_risk_stats = stats_tail_risk(returns,
                                        annual_factor=annual_factor,
                                        var_quantile=var_quantile,
                                        keep_indexes=keep_indexes,
                                        drop_indexes=drop_indexes)
        
        summary_statistics = summary_statistics.join(tail_risk_stats)
        
    if return_tangency_weights is True:
        tangency_weights = calc_tangency_port(returns, name = 'Tangency')
        summary_statistics = summary_statistics.join(tangency_weights)

    if correlations is True or isinstance(correlations, list):
               
        returns_corr = returns.corr()
        if _timeframe_name:
            returns_corr = returns_corr.rename(columns=lambda col: col.replace(f' ({_timeframe_name})', ''))

        if isinstance(correlations, list):
            # Check if all selected columns exist in returns_corr
            corr_not_in_returns_corr = [col for col in correlations if col not in returns_corr.columns]
            if len(corr_not_in_returns_corr) > 0:
                not_in_returns_corr = ", ".join([c for c in corr_not_in_returns_corr])
                raise Exception(f'{not_in_returns_corr} not in returns columns')
            
            returns_corr = returns_corr[[col for col in correlations]]
            
        returns_corr = returns_corr.rename(columns=lambda col: col + ' Correlation')
        
        # Specify a suffix to be added to overlapping columns
        summary_statistics = summary_statistics.join(returns_corr)
    
    return filter_columns_and_indexes(
        summary_statistics,
        keep_columns=keep_columns,
        drop_columns=drop_columns,
        keep_indexes=keep_indexes,
        drop_indexes=drop_indexes
    )


def stats_tail_risk(
    returns: Union[pd.DataFrame, pd.Series, List[pd.Series]],
    annual_factor: int = None,
    var_quantile: Union[float , List] = .05,
    keep_columns: Union[list, str] = None,
    drop_columns: Union[list, str] = None,
    keep_indexes: Union[list, str] = None,
    drop_indexes: Union[list, str] = None,
) -> pd.DataFrame:
    """
    Calculates tail risk summary statistics for a time series of returns.   

    Parameters:
    returns (pd.DataFrame, pd.Series or List or pd.Series): Time series of returns.
    annual_factor (int, default=None): Factor for annualizing returns.
    var_quantile (float or list, default=0.05): Quantile for Value at Risk (VaR) calculation.
    keep_columns (list or str, default=None): Columns to keep in the resulting DataFrame.
    drop_columns (list or str, default=None): Columns to drop from the resulting DataFrame.
    keep_indexes (list or str, default=None): Indexes to keep in the resulting DataFrame.
    drop_indexes (list or str, default=None): Indexes to drop from the resulting DataFrame.

    Returns:
    pd.DataFrame: tail risk summary statistics of the returns.
    """

    returns = time_series_to_df(returns) # Convert returns to DataFrame if it is a Series or a list of Series
    fix_dates_index(returns) # Fix the date index of the DataFrame if it is not in datetime format and convert returns to float

    tail_risk_stats = pd.DataFrame(index=returns.columns)

    tail_risk_stats['Skewness'] = returns.skew()
    tail_risk_stats['Excess Kurtosis'] = returns.kurtosis()
    var_quantile = [var_quantile] if isinstance(var_quantile, (float, int)) else var_quantile
    for var_q in var_quantile:
        tail_risk_stats[f'Historical VaR ({var_q:.1%})'] = returns.quantile(var_q, axis = 0)
        tail_risk_stats[f'Historical CVaR ({var_q:.1%})'] = returns[returns <= returns.quantile(var_q, axis = 0)].mean()
        if annual_factor:
            tail_risk_stats[f'Annualized Historical VaR ({var_q:.1%})'] = returns.quantile(var_q, axis = 0) * np.sqrt(annual_factor)
            tail_risk_stats[f'Annualized Historical CVaR ({var_q:.1%})'] = returns[returns <= returns.quantile(var_q, axis = 0)].mean() * np.sqrt(annual_factor)
    
    cum_returns = (1 + returns).cumprod()
    maximum = cum_returns.cummax()
    drawdown = cum_returns / maximum - 1

    tail_risk_stats['Accumulated Return'] = cum_returns.iloc[-1] - 1
    tail_risk_stats['Max Drawdown'] = drawdown.min()
    tail_risk_stats['Peak Date'] = [maximum[col][:drawdown[col].idxmin()].idxmax() for col in maximum.columns]
    tail_risk_stats['Bottom Date'] = drawdown.idxmin()
    
    recovery_date = []
    for col in cum_returns.columns:
        prev_max = maximum[col][:drawdown[col].idxmin()].max()
        recovery_wealth = pd.DataFrame([cum_returns[col][drawdown[col].idxmin():]]).T
        recovery_date.append(recovery_wealth[recovery_wealth[col] >= prev_max].index.min())
    tail_risk_stats['Recovery'] = recovery_date

    tail_risk_stats["Duration (days)"] = [
        (i - j).days if i != pd.NaT else "-" for i, j in
        zip(tail_risk_stats["Recovery"], tail_risk_stats["Bottom Date"])
    ]

    return filter_columns_and_indexes(
        tail_risk_stats,
        keep_columns=keep_columns,
        drop_columns=drop_columns,
        keep_indexes=keep_indexes,
        drop_indexes=drop_indexes
    )

    
def calc_correlations(
    returns: Union[pd.DataFrame, pd.Series, List[pd.Series]],
    print_highest_lowest: bool = True,
    show_heatmap: bool = True,
    return_matrix: bool = False,
    keep_columns: Union[list, str] = None,
    drop_columns: Union[list, str] = None,
    keep_indexes: Union[list, str] = None,
    drop_indexes: Union[list, str] = None,
    save_plot: bool = False,
    plot_name: str = None
) -> Union[sns.heatmap, pd.DataFrame]:
    """
    Calculates the correlation matrix of the provided returns and optionally prints or visualizes it.

    Parameters:
    returns (pd.DataFrame, pd.Series or List or pd.Series): Time series of returns.
    print_highest_lowest (bool, default=True): If True, prints the highest and lowest correlations.
    show_heatmap (bool, default=False): If True, returns a heatmap of the correlation matrix.
    keep_columns (list or str, default=None): Columns to keep in the resulting DataFrame.
    drop_columns (list or str, default=None): Columns to drop from the resulting DataFrame.
    keep_indexes (list or str, default=None): Indexes to keep in the resulting DataFrame.
    drop_indexes (list or str, default=None): Indexes to drop from the resulting DataFrame.
    save_plot (bool, default=False): If True and show_heatmap is also True, saves the heatmap as a PNG.
    plot_name (str, optional): Custom filename prefix. If None, defaults to 'heatmap_correlations'.

    Returns:
    sns.heatmap or pd.DataFrame: Heatmap of the correlation matrix or the correlation matrix itself.
    """

    returns = time_series_to_df(returns)  # convert to DataFrame if needed
    fix_dates_index(returns)             # ensure datetime index and float dtype

    returns = filter_columns_and_indexes(
        returns,
        keep_columns=keep_columns,
        drop_columns=drop_columns,
        keep_indexes=keep_indexes,
        drop_indexes=drop_indexes
    )

    correlation_matrix = returns.corr()

    if print_highest_lowest:
        highest_lowest_corr = (
            correlation_matrix
            .unstack()
            .sort_values()
            .reset_index()
            .set_axis(['asset_1', 'asset_2', 'corr'], axis=1)
            .loc[lambda df: df.asset_1 != df.asset_2]
        )
        highest_corr = highest_lowest_corr.iloc[-1, :]
        lowest_corr = highest_lowest_corr.iloc[0, :]
        print(f'The highest correlation ({highest_corr["corr"]:.4f}) is between {highest_corr.asset_1} and {highest_corr.asset_2}')
        print(f'The lowest correlation ({lowest_corr["corr"]:.4f}) is between {lowest_corr.asset_1} and {lowest_corr.asset_2}')

    if show_heatmap:
        fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
        heatmap = sns.heatmap(
            correlation_matrix, 
            xticklabels=correlation_matrix.columns,
            yticklabels=correlation_matrix.columns,
            annot=True,
            ax=ax
        )
        plt.show()

        # Save the figure if requested
        if save_plot:
            plot_name_prefix = plot_name if plot_name else "heatmap_correlations"
            save_figure(fig, plot_name_prefix)

        # Return the heatmap object if needed (though typically you don't "return" a heatmap)
        if return_matrix:
            return correlation_matrix
        else:
            return heatmap

    if return_matrix:
        return correlation_matrix
    else:
        return None
    


def calc_ewma_volatility(
        returns: pd.Series,
        theta : float = 0.94,
        ewma_initial_annual_vol : float = None,
        annual_factor: int = None
    ) -> pd.Series:
    """
    Calculates the EWMA (Exponentially Weighted Moving Average) volatility of a time series of returns.

    Parameters:
    returns (pd.Series): Time series of returns.
    theta (float, default=0.94): Theta parameter for the EWMA volatility calculation.
    ewma_initial_annual_vol (float, default=None): Initial annual volatility for the EWMA calculation.
    annual_factor (int, default=None): Factor for annualizing returns.

    Returns:
    pd.Series: Time series of EWMA volatility.
    """

    if ewma_initial_annual_vol is not None:
        if annual_factor is None:
            print('Assuming monthly returns with annualization term of 12')
            annual_factor = 12
        ewma_initial_annual_vol = ewma_initial_annual_vol / np.sqrt(annual_factor)
    else:
        ewma_initial_annual_vol = returns.std()
    
    var_t0 = ewma_initial_annual_vol ** 2
    ewma_var = [var_t0]
    for i in range(len(returns.index)):
        new_ewma_var = ewma_var[-1] * theta + (returns.iloc[i] ** 2) * (1 - theta)
        ewma_var.append(new_ewma_var)
    ewma_var.pop(0) # Remove var_t0
    ewma_vol = [np.sqrt(v) for v in ewma_var]
    return pd.Series(ewma_vol, index=returns.index)


def calc_garch_volatility(
        returns: pd.Series,
        p: int = 1,
        q: int = 1
    ) -> pd.Series:

    model = arch_model(returns, vol='Garch', p=p, q=q)
    fitted_model = model.fit(disp='off')
    fitted_values = fitted_model.conditional_volatility
    return pd.Series(fitted_values, index=returns.index)


def calc_var_cvar_summary(
    returns: Union[pd.Series, pd.DataFrame],
    percentile: Union[None, float] = .05,
    window: Union[None, str] = None,
    return_hit_ratio: bool = False,
    filter_first_hit_ratio_date: Union[None, str, datetime.date] = None,
    z_score: float = None,
    shift: int = 1,
    std_formula: bool = False,
    ewma_theta : float = .94,
    ewma_initial_annual_vol: float = None,
    include_garch: bool = False,
    garch_p: int = 1,
    garch_q: int = 1,
    annual_factor: int = None,
    return_stats: Union[str, list] = ['Returns', 'VaR', 'CVaR', 'Vol'],
    keep_columns: Union[list, str] = None,
    drop_columns: Union[list, str] = None,
    keep_indexes: Union[list, str] = None,
    drop_indexes: Union[list, str] = None
) -> pd.DataFrame:
    """
    Calculates a summary of VaR (Value at Risk), CVaR (Conditional VaR), kurtosis, and skewness for the provided returns.

    Parameters:
    returns (pd.Series or pd.DataFrame): Time series of returns.
    percentile (float or None, default=0.05): Percentile to calculate the VaR and CVaR.
    window (str or None, default=None): Window size for rolling calculations.
    return_hit_ratio (bool, default=False): If True, returns the hit ratio for the VaR.
    filter_first_hit_ratio_date (str, datetime.date or None, default=None): Date to filter the hit ratio calculation from then on.
    z_score (float, default=None): Z-score for parametric VaR calculation, in case no percentile is provided.
    shift (int, default=1): Period shift for VaR/CVaR calculations to avoid look-ahead bias.
    std_formula (bool, default=False): If True, uses the normal volatility formula with .std(). Else, use squared returns.
    ewma_theta (float, default=0.94): Theta parameter for the EWMA volatility calculation.
    ewma_initial_annual_vol (float, default=None): Initial annual volatility for the EWMA calculation.
    include_garch (bool, default=False): If True, includes GARCH volatility in the summary.
    garch_p (int, default=1): Order of the GARCH model.
    garch_q (int, default=1): Order of the GARCH model.
    annual_factor (int, default=None): Factor for annualizing returns.
    return_stats (str or list, default=['Returns', 'VaR', 'CVaR', 'Vol']): Statistics to return in the summary.
    keep_columns (list or str, default=None): Columns to keep in the resulting DataFrame.
    drop_columns (list or str, default=None): Columns to drop from the resulting DataFrame.
    keep_indexes (list or str, default=None): Indexes to keep in the resulting DataFrame.
    drop_indexes (list or str, default=None): Indexes to drop from the resulting DataFrame.

    Returns:
    pd.DataFrame: Summary of VaR and CVaR statistics.
    """
    if annual_factor is None:
        print('Assuming monthly returns with annualization term of 12')
        annual_factor = 12
    if window is None:
        print('Using "window" of 60 periods, since none was specified')
        window = 60
    if isinstance(returns, pd.DataFrame):
        returns_series = returns.iloc[:, 0]
        returns_series.index = returns.index
        returns = returns_series.copy()
    elif isinstance(returns, pd.Series):
        returns = returns.copy()
    else:
        raise TypeError('returns must be either a pd.DataFrame or a pd.Series')

    summary = pd.DataFrame({})

    # Returns
    summary[f'Returns'] = returns

    # Kurtosis
    summary[f'Expanding Kurtosis'] = returns.expanding(window).apply(lambda x: kurtosis(x, fisher=True, bias=False))
    summary[f'Rolling Kurtosis ({window})'] = returns.rolling(window).apply(lambda x: kurtosis(x, fisher=True, bias=False))

    # Skewness
    summary[f'Expanding Skewness'] = returns.expanding(window).apply(lambda x: skew(x, bias=False))
    summary[f'Rolling Skewness ({window})'] = returns.rolling(window).apply(lambda x: skew(x, bias=False))

    # VaR
    summary[f'Expanding {window} Historical VaR ({percentile:.0%})'] = returns.expanding(min_periods=window).quantile(percentile)
    summary[f'Rolling {window} Historical VaR ({percentile:.0%})'] = returns.rolling(window=window).quantile(percentile)
    if std_formula:
        summary[f'Expanding {window} Volatility'] = returns.expanding(window).std()
        summary[f'Rolling {window} Volatility'] = returns.rolling(window).std()
    else: # Volaility assuming zero mean returns
        summary[f'Expanding {window} Volatility'] = np.sqrt((returns ** 2).expanding(window).mean())
        summary[f'Rolling {window} Volatility'] = np.sqrt((returns ** 2).rolling(window).mean())
    summary[f'EWMA {ewma_theta:.2f} Volatility'] = calc_ewma_volatility(returns, theta=ewma_theta, ewma_initial_annual_vol=ewma_initial_annual_vol, annual_factor=annual_factor)
    if include_garch:
        summary[f'GARCH({garch_p:.0f}, {garch_q:.0f}) Volatility'] = calc_garch_volatility(returns, p=garch_p, q=garch_q)
    
    # Parametric VaR assuming zero mean returns
    z_score = norm.ppf(percentile) if z_score is None else z_score
    summary[f'Expanding {window} Parametric VaR ({percentile:.0%})'] = summary[f'Expanding {window} Volatility'] * z_score
    summary[f'Rolling {window} Parametric VaR ({percentile:.0%})'] = summary[f'Rolling {window} Volatility'] * z_score
    summary[f'EWMA {ewma_theta:.2f} Parametric VaR ({percentile:.0%})'] = summary[f'EWMA {ewma_theta:.2f} Volatility'] * z_score
    if include_garch:
        summary[f'GARCH({garch_p:.0f}, {garch_q:.0f}) Parametric VaR ({percentile:.0%})'] = summary[f'GARCH({garch_p:.0f}, {garch_q:.0f}) Volatility'] * z_score

    if return_hit_ratio:
        var_stats = [
            f'Expanding {window} Historical VaR ({percentile:.0%})',
            f'Rolling {window} Historical VaR ({percentile:.0%})',
            f'Expanding {window} Parametric VaR ({percentile:.0%})',
            f'Rolling {window} Parametric VaR ({percentile:.0%})',
            f'EWMA {ewma_theta:.2f} Parametric VaR ({percentile:.0%})'
        ]
        if include_garch:
            var_stats.append(f'GARCH({garch_p:.0f}, {garch_q:.0f}) Parametric VaR ({percentile:.0%})')
        
        summary_hit_ratio = summary.copy()
        summary_hit_ratio[var_stats] = summary_hit_ratio[var_stats].shift()
        if filter_first_hit_ratio_date:
            if isinstance(filter_first_hit_ratio_date, (datetime.date, datetime.datetime)):
                filter_first_hit_ratio_date = filter_first_hit_ratio_date.strftime("%Y-%m-%d")
            summary_hit_ratio = summary.loc[filter_first_hit_ratio_date:]
        summary_hit_ratio = summary_hit_ratio.dropna(axis=0)
        summary_hit_ratio[var_stats] = summary_hit_ratio[var_stats].apply(lambda x: (x - summary['Returns']) > 0)
        
        hit_ratio = pd.DataFrame(summary_hit_ratio[var_stats].mean(), columns=['Hit Ratio'])
        hit_ratio['Hit Ratio Error'] = (hit_ratio['Hit Ratio'] - percentile) / percentile
        hit_ratio['Hit Ratio Absolute Error'] = abs(hit_ratio['Hit Ratio Error'])
        hit_ratio = hit_ratio.sort_values('Hit Ratio Absolute Error')

        if z_score is not None:
            hit_ratio = hit_ratio.rename(lambda col: re.sub(r'VaR \(\d+%\)', f'VaR ({z_score:.2f})', col), axis=1) # Rename columns
        return filter_columns_and_indexes(
            hit_ratio,
            keep_columns=keep_columns,
            drop_columns=drop_columns,
            keep_indexes=keep_indexes,
            drop_indexes=drop_indexes
        )

    # CVaR
    summary[f'Expanding {window} Historical CVaR ({percentile:.0%})'] = returns.expanding(window).apply(lambda x: x[x < x.quantile(percentile)].mean())
    summary[f'Rolling {window} Historical CVaR ({percentile:.0%})'] = returns.rolling(window).apply(lambda x: x[x < x.quantile(percentile)].mean())
    summary[f'Expanding {window} Parametrical CVaR ({percentile:.0%})'] = - norm.pdf(z_score) / percentile * summary[f'Expanding {window} Volatility']
    summary[f'Rolling {window} Parametrical CVaR ({percentile:.0%})'] = - norm.pdf(z_score) / percentile * summary[f'Rolling {window} Volatility']
    summary[f'EWMA {ewma_theta:.2f} Parametrical CVaR ({percentile:.0%})'] = - norm.pdf(z_score) / percentile * summary[f'EWMA {ewma_theta:.2f} Volatility']
    if include_garch:
        summary[f'GARCH({garch_p:.0f}, {garch_q:.0f}) Parametrical CVaR ({percentile:.0%})'] = - norm.pdf(z_score) / percentile * summary[f'GARCH({garch_p:.0f}, {garch_q:.0f}) Volatility']

    if shift > 0:
        shift_columns = [c for c in summary.columns if not bool(re.search("returns", c))]
        summary[shift_columns] = summary[shift_columns].shift(shift).dropna()
        print(f'VaR and CVaR are given shifted by {shift:0f} period(s).')
    else:
        print('VaR and CVaR are given in-sample.')

    return_stats = [return_stats.lower()] if isinstance(return_stats, str) else [s.lower() for s in return_stats]
    return_stats = list(map(lambda x: 'volatility' if x == 'vol' else x, return_stats))
    
    if z_score is not None:
        summary = summary.rename(lambda col: re.sub(r'VaR \(\d+%\)', f'VaR ({z_score:.2f})', col), axis=1)

    if return_stats == ['all'] or set(return_stats) == set(['returns', 'var', 'cvar', 'volatility']):
        summary = summary.loc[:, lambda df: df.columns.map(lambda c: bool(re.search(r"\b" + r"\b|\b".join(return_stats) + r"\b", c.lower())))]
        
    return filter_columns_and_indexes(
        summary,
        keep_columns=keep_columns,
        drop_columns=drop_columns,
        keep_indexes=keep_indexes,
        drop_indexes=drop_indexes
    )


def plot_var(
        returns: Union[pd.DataFrame, pd.Series],
        var: Union[pd.DataFrame, pd.Series, List[pd.Series]],
        percentile: Union[None, float] = .05,
        figsize: tuple = (PLOT_WIDTH, PLOT_HEIGHT),
        limit = True,
        colors: Union[list, str] = ["blue", "red", "orange", "green", "purple", "black", "grey", "pink", "brown", "cyan", "magenta", "yellow"],
        var_column: str = None,
        is_excess_returns: bool = False,
        save_plot: bool = False,
        plot_name: str = None
) -> None:
    """
    Plots a variance graph with returns and highlights returns < VaR 

    Parameters:
    returns (pd.DataFrame, pd.Series or None): Time series of returns.
    var (pd.DataFrame, pd.Series or List or pd.Series): Time series of VaR.
    percentile (float or None, default=.05): Percentile to calculate the hit ratio.
    limit (bool, default=True): If True, limits the y-axis to the minimum return.
    figsize (tuple, default=(PLOT_WIDTH, PLOT_HEIGHT)): Size of the plot.
    colors (list or str, default=["blue", "red", "orange", "green", "purple", "black", "grey", "pink", "brown", "cyan", "magenta", "yellow"]): Colors for the plot.
    var_column (str, default='VaR'): Name for the VaR column to be uses
    is_excess_returns (bool, default=False): If True, adjust y-axis label accordingly.
    save_plot (bool, default=False): If True, saves the plot as a PNG.
    plot_name (str, optional): Custom filename prefix. If None, defaults to 'plot_var'.

    """
    var = time_series_to_df(var, "VaR")  # convert to DataFrame if needed
    fix_dates_index(var)

    returns = time_series_to_df(returns, "Returns")
    fix_dates_index(returns)
    returns = pd.merge(returns, var, left_index=True, right_index=True).dropna()

    asset_name = returns.columns[0]
    if asset_name == 0:
        asset_name = "Asset"

    # If var_name isn't provided, try to derive it
    if var_column is None:
        if isinstance(var, pd.DataFrame):
            var_column = var.columns[0]
            if var_column == 0:
                var_column = "VaR"

    returns.columns = [asset_name, var_column]

    plt.figure(figsize=figsize)
    plt.axhline(y=0, linestyle='--', color='black', alpha=0.5)

    # Plot returns
    plt.plot(
        returns.index,
        returns[asset_name],
        color=colors[2],
        label=f"{asset_name} Returns",
        alpha=0.2
    )

    # If var has only one column
    if var.shape[1] == 1:
        plt.plot(
            returns.index,
            returns[var_column],
            color=colors[0],
            label=var_column
        )
        # Points where returns < VaR
        excess_returns_surpass_var = (
            returns
            .dropna()
            .loc[lambda df: df[asset_name] < df[var_column]]
        )
        plt.plot(
            excess_returns_surpass_var.index,
            excess_returns_surpass_var[asset_name],
            linestyle="",
            marker="o",
            color=colors[1],
            label=f"Return < {var_column}",
            markersize=1.5
        )

        if limit:
            plt.ylim(min(returns[asset_name]), 0.01)

        hit_ratio = len(excess_returns_surpass_var.index) / len(returns.index)
        hit_ratio_error = abs((hit_ratio / percentile) - 1)
        plt.title(f"{var_column} of {asset_name} Returns")
        plt.xlabel(f"Hit Ratio: {hit_ratio:.2%}; Hit Ratio Error: {hit_ratio_error:.2%}")
        plt.ylabel("Excess Returns" if is_excess_returns else "Returns")
        plt.legend()
        plt.show()

    else:
        fig, ax = plt.subplots(figsize=figsize)

        # If var has multiple columns
        for idx, var_series in enumerate(var.columns):
            ax.plot(
                returns.index,
                returns[var_series],
                color=colors[idx],
                label=var_series
            )

        ax.set_title(f"VaR of {asset_name} Returns")
        ax.set_ylabel("Excess Returns" if is_excess_returns else "Returns")
        ax.legend()

        plt.show()

    # Save the plot if requested
    if save_plot:
        plot_name_prefix = plot_name if plot_name else (f'plot_var_{asset_name}' if asset_name != 'Asset' else 'plot_var')
        save_figure(fig, plot_name_prefix)



def calc_tangency_port(
    returns: Union[pd.DataFrame, List[pd.Series]],
    expected_returns: Union[pd.Series, dict, None] = None,
    cov_matrix_factor: str = 1,
    target_return: Union[None, float] = None,
    annual_factor: int = 12,
    show_graphic: bool = False,
    return_port_returns: bool = False,
    name: str = 'Tangency'
) -> Union[pd.DataFrame, pd.Series]:
    """
    Calculates tangency portfolio weights based on the covariance matrix of returns.
        When `target_return` is provided, the weights are rescaled to achieve the target return:
            - If returns are the "excess returns", then the rescaled tangency portfolio is also in the ~MV frontier.

    Parameters:
    returns (pd.DataFrame or List of pd.Series): Time series of returns.
    expected_returns (pd.Series, dict or None, default=None): Expected returns for each asset. If None, uses the mean returns as a proxy for expected returns.
    cov_matrix_factor (str, default=1): Weight for the covariance matrix. If 1, uses the sample covariance matrix, otherwise uses a shrinkage estimator.
    target_return (float or None, default=None): Target return for rescaling weights (annualized).
    annual_factor (int, default=12): Factor for annualizing returns.
    show_graphic (bool, default=False): If True, plots the tangency weights.
    return_port_returns (bool, default=False): If True, returns the portfolio returns. Otherwise, returns portfolio weights.
    name (str, default='Tangency'): Name for labeling the weights and portfolio.

    Returns:
    pd.DataFrame or pd.Series: Tangency portfolio weights or portfolio returns if `return_port_ret` is True.
    """

    returns = time_series_to_df(returns) # Convert returns to DataFrame if it is a Series or a list of Series
    fix_dates_index(returns) # Fix the date index of the DataFrame if it is not in datetime format and convert returns to float

    # Calculate the covariance matrix
    if cov_matrix_factor == 1:
        cov_matrix = returns.cov()
    else:
        cov_matrix = returns.cov()
        cov_matrix_diag = np.diag(np.diag(cov_matrix))
        cov_matrix = cov_matrix_factor * cov_matrix + (1-cov_matrix_factor) * cov_matrix_diag
    
    cov_matrix_inv = np.linalg.pinv(cov_matrix)
    ones = np.ones(len(returns.columns))
    if expected_returns is not None:
        if isinstance(expected_returns, dict):
            expected_returns = pd.Series(expected_returns)
        elif isinstance(expected_returns, pd.DataFrame):
            expected_returns = expected_returns.iloc[:, 0]
        else:
            raise TypeError('expected_returns must be a pd.Series or a dictionary')
        
        mu = expected_returns.reindex(returns.columns)
        if mu.isnull().any():
            not_in_returns_mu = mu[mu.isnull()].index
            raise Exception(f'{not_in_returns_mu} not in returns columns')
    else:
        mu = returns.mean() # Use mean monthly excess returns as a proxy for expected excess returns: (mu)

    # Calculate the tangency portfolio weights
    scaling = 1 / (ones.T @ cov_matrix_inv @ mu)
    tangency_wts = scaling * (cov_matrix_inv @ mu)
    tangency_wts = pd.DataFrame(index=returns.columns, data=tangency_wts, columns=[f'{name} Portfolio'])
    
    # Calculate the portfolio returns
    port_returns = returns @ tangency_wts

    # Rescale weights to target return
    if isinstance(target_return, (float, int)):
        if annual_factor is None:
            print(f'Assuming monthly returns with annualization term of 12 since none was provided')
            annual_factor = 12
        scaler = target_return / (port_returns[f'{name} Portfolio'].mean() * annual_factor)
        tangency_wts[[f'{name} Portfolio']] *= scaler
        port_returns *= scaler
        
        tangency_wts = tangency_wts.rename({f'{name} Portfolio': f'{name} Portfolio (rescaled {target_return:.1%} p.a.)'},axis=1)
        port_returns = port_returns.rename({f'{name} Portfolio': f'{name} Portfolio (rescaled {target_return:.1%} p.a.)'},axis=1)

    
    # Plot the tangency weights
    if show_graphic == True:
        ax = tangency_wts.plot(kind='bar', title=f'{name} Portfolio Weights')
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2%}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    if cov_matrix_factor != 1:
        if target_return is None:
            tangency_wts = tangency_wts.rename({f'{name} Portfolio': f'{name} Portfolio (regularized {cov_matrix_factor:.1f})'},axis=1)
            port_returns = port_returns.rename({f'{name} Portfolio':f'{name} Portfolio (regularized {cov_matrix_factor:.1f})'},axis=1)
        else:
            tangency_wts = tangency_wts.rename({f'{name} Portfolio (rescaled {target_return:.1%} p.a.)':
                                                f'{name} Portfolio (regularized {cov_matrix_factor:.1f}, rescaled {target_return:.1%} p.a.)'},axis=1)
            port_returns = port_returns.rename({f'{name} Portfolio (rescaled {target_return:.1%} p.a.)':
                                                f'{name} Portfolio (regularized {cov_matrix_factor:.1f}, rescaled {target_return:.1%} p.a.)'},axis=1)
            
        
    if return_port_returns:
        return port_returns
    return tangency_wts

```
"

