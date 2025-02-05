I have many scripts with many files that I use as tool set. Some of them are in the section below. Take a look at the formatting, documentation, which functions I have. Then I will give you a new set of functions that you will need to refactor and give me in a new code so that I can include them in my code base. You don't need to re-write any of my current functions.

## Current functions from my script

```python

import datetime
from typing import Union, List, Optional, Tuple

import numpy as np
import pandas as pd

from scipy import interpolate
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

from .fixed_income import int_rate_to_discount, discount_to_int_rate, get_maturity_delta

# ===============================================================================================
# Curve Models
# ===============================================================================================


def bootstrap(
    params: Tuple[np.ndarray, np.ndarray],
    maturity: Union[float, np.ndarray]
) -> np.ndarray:
    """
    Applies piecewise interpolation of discount factors to compute rates.

    Parameters:
    params (tuple of np.ndarray): (estimated_maturities, discount_factors).
    maturity (float or np.ndarray): Maturity(ies) for which to compute rates.

    Returns:
    np.ndarray: Interpolated interest rates.
    """
    estimated_maturities, betas = params
    # Convert discount factors to interest rates (continuous by default)
    est_rates = [discount_to_int_rate(b, m, None) for b, m in zip(betas, estimated_maturities)]

    f = interpolate.interp1d(
        estimated_maturities,
        est_rates,
        bounds_error=False,
        fill_value='extrapolate'
    )
    return f(maturity)


def estimate_curve_ols(
    CF: pd.DataFrame,
    prices: Union[pd.DataFrame, pd.Series, np.ndarray],
    interpolate: bool = False
) -> np.ndarray:
    """
    Fits a linear regression (with no intercept) using 'CF' values as predictors and 'prices' as targets, 
    returning an array of discount factors. If 'interpolate' is True, discount factors are further 
    interpolated over a maturity grid.

    Parameters:
    CF (pd.DataFrame): Cashflow matrix where rows are bonds and columns are dates, containing coupon payments.
    prices (pd.DataFrame, pd.Series, or np.ndarray): Observed bond prices. If a DataFrame/Series, values are matched to CF's rows.
    interpolate (bool, default=False): Whether to interpolate discount factors over the maturity grid.

    Returns:
    np.ndarray: Discount factors (one per column of CF if no interpolation, otherwise an interpolated array).
    """
    # If 'prices' is a DataFrame/Series, align it with CF's index
    if isinstance(prices, (pd.DataFrame, pd.Series)):
        prices = prices[CF.index].values

    # Fit linear regression with no intercept (discount factors = regression coefficients)
    model = LinearRegression(fit_intercept=False).fit(CF.values, prices)

    # By default, the discount factors are the regression coefficients
    discounts = model.coef_

    # If interpolation is requested
    if interpolate:
        # Create a grid of "maturity" for each column in CF relative to the first column
        matgrid = get_maturity_delta(CF.columns, CF.columns.min())

        # Filter out any coefficients that seem invalid (e.g., negative or too large)
        valid_mask = np.logical_and(discounts > 0, discounts < 1.25)

        xold = matgrid[valid_mask]
        yold = discounts[valid_mask]
        xnew = matgrid

        f = interpolate.interp1d(xold, yold, bounds_error=False, fill_value='extrapolate')
        discounts = f(xnew)

    return discounts


def nelson_siegel(params: List[float], maturity: Union[float]) -> np.ndarray:
    """
    Nelson-Siegel model: r(t) = beta0 + (beta1 + beta2)*[1-exp(-t/tau)]/(t/tau) - beta2*exp(-t/tau).

    Parameters:
    params (list of float): [beta0, beta1, beta2, tau].
    maturity (float): Maturity.

    Returns:
    np.ndarray: Modeled rates.
    """
    rate = params[0] + (params[1] + params[2]) * (1 - np.exp(-maturity/params[3]))/(maturity/params[3]) - params[2] * np.exp(-maturity/params[3])
    
    return rate


def nelson_siegel_extended(params: List[float], maturity: float) -> np.ndarray:
    """
    Extended Nelson-Siegel: r(t) = standard NS + beta4 * ([1-exp(-t/tau2)]/(t/tau2) - exp(-t/tau2)).

    Parameters:
    params (list of float): [beta0, beta1, beta2, tau, beta4, tau2].
    maturity (float): Maturity.

    Returns:
    np.ndarray: Modeled rates.
    """
    rate = params[0] + (params[1] + params[2]) * (1 - np.exp(-maturity/params[3]))/(maturity/params[3]) - params[2] * np.exp(-maturity/params[3]) + params[4] *((1-np.exp(-maturity/params[5]))/(maturity/params[5]) - np.exp(-maturity/params[5]))
    
    return rate


def price_with_rate_model(
    params: Union[List[float], np.ndarray, Tuple[np.ndarray, np.ndarray]],
    cf: pd.DataFrame,
    date_current: Union[str, datetime.date, datetime.datetime, np.datetime64],
    func_model,
    convert_to_discount: bool = True,
    price_coupons: bool = False
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Computes bond prices from model parameters and a cashflow matrix.

    Parameters:
    params (list or np.ndarray or tuple): Model parameters (e.g. [beta0,beta1,beta2,tau] or (maturities, discounts)).
    cf (pd.DataFrame): Cashflow matrix, rows=bonds, columns=dates.
    date_current (str or date or datetime or np.datetime64): Current date to compute maturities.
    func_model (callable): Function of form func_model(params, maturity)->rate or discount.
    convert_to_discount (bool, default=True): If True, model returns rates which get converted to discount factors.
    price_coupons (bool, default=False): If True, returns discounted CF matrix; else returns sum across columns.

    Returns:
    np.ndarray or pd.DataFrame: Prices (or matrix of discounted CF).
    """
    maturity = get_maturity_delta(cf.columns, date_current)
    maturity = maturity.values  # 1D array of column maturities

    # Convert interest rates -> discount factors if needed
    if convert_to_discount:
        discount_factors = []
        for matur in maturity:
            rate = func_model(params, matur)
            discount_factors.append(int_rate_to_discount(rate, matur))
        discount_factors = np.array(discount_factors)
    else:
        # The model itself returns discount factors
        discount_factors = np.array([func_model(params, matur) for matur in maturity])

    if price_coupons:
        # Return the matrix of discounted cashflows
        price = cf.mul(discount_factors, axis='columns') 
    else:
        # Price is sum across columns
        price = cf.values @ discount_factors
    
    return price


def pricing_errors(
    params: Union[List[float], np.ndarray, Tuple[np.ndarray, np.ndarray]],
    cf: pd.DataFrame,
    date_current: Union[str, datetime.date, datetime.datetime, np.datetime64],
    func_model,
    observed_prices: Union[pd.DataFrame, pd.Series, np.ndarray]
) -> float:
    """
    Objective function for curve-fitting: sum of squared errors 
    between observed prices and modeled prices.

    Parameters:
    params (list, np.ndarray or tuple): Model parameters.
    cf (pd.DataFrame): Cashflow matrix.
    date_current (str or date or datetime or np.datetime64): Current date for maturity calc.
    func_model (callable): Rate model function.
    observed_prices (pd.DataFrame, pd.Series or np.ndarray): Observed bond prices.

    Returns:
    float: Sum of squared errors.
    """
    modeled_prices = price_with_rate_model(params, cf, date_current, func_model)

    if isinstance(observed_prices, (pd.DataFrame, pd.Series)):
        observed_prices = observed_prices.values

    return float(np.sum((observed_prices - modeled_prices) ** 2))


def estimate_rate_curve(
    model_name: str,
    cf: pd.DataFrame,
    date_current: Union[str, datetime.date, datetime.datetime, np.datetime64],
    prices: Union[pd.DataFrame, pd.Series, np.ndarray],
    x0: Optional[Union[List[float], np.ndarray]] = None
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Estimates parameters by fitting bond prices to a curve model.

    Parameters:
    model_name (str): Model to estimate interest rate curve (e.g. bootstrap, nelson_siegel, etc.).
    cf (pd.DataFrame): Cashflow matrix [n_bonds x n_dates].
    date_curr (str or date or datetime or np.datetime64): Current date to calculate maturity.
    prices (pd.DataFrame, pd.Series or np.ndarray): Observed bond prices.
    x0 (list or np.ndarray, optional): Initial guess for model parameters.

    Returns:
    np.ndarray or tuple: Fitted parameters (for bootstrap returns (maturities, discounts)).
    """
    p = prices.copy()
    if isinstance(p, (pd.DataFrame, pd.Series)):
        p = p[cf.index].values

    # If model == 'OLS', use linear regression
    if model_name == 'OLS':
        # OLS on CF matrix to get discount(s)
        discounts = estimate_curve_ols(cf, prices, interpolate=False)
        cf_intervals = get_maturity_delta(cf.columns.to_series(),date_current=date_current).values    
        params_est = [cf_intervals, discounts]

    elif model_name == 'bootstrap':
        discounts = estimate_curve_ols(cf, prices, interpolate=False)
        cf_intervals = get_maturity_delta(cf.columns.to_series(),date_current=date_current).values    
        params_est = [cf_intervals, discounts]

    # Otherwise, numeric optimization
    else:
        if model_name == 'nelson_siegel_extended':
            model_obj = nelson_siegel_extended
            if x0 is None:
                x0 = np.ones(6)   
        else:
            if model_name != 'nelson_siegel': print(f"Model '{model_name}' not recognized. Using default 'nelson_siegel'.")	
            model_obj = nelson_siegel
            if x0 is None: x0 = np.ones(4) * 0.1  # some default

        # optimizer iteratively updates params (starting from x0) to minimize the sum of squared errors calculated in pricing_errors.
        result = minimize(pricing_errors, x0, args=(cf, date_current, model_obj, prices))
        params_est = result.x

    return params_est

"""
treasuries.py

This module provides functions for working with Treasury bond data, including business-day calculations, 
coupon date extraction, cashflow filtering, yield curve modeling, and more.

"""
import datetime
from typing import Union, List, Optional, Tuple

import numpy as np
import pandas as pd

from scipy import interpolate
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

from .fixed_income import int_rate_to_discount, discount_to_int_rate, get_maturity_delta

# ===============================================================================================
# Curve Models
# ===============================================================================================


def bootstrap(
    params: Tuple[np.ndarray, np.ndarray],
    maturity: Union[float, np.ndarray]
) -> np.ndarray:
    """
    Applies piecewise interpolation of discount factors to compute rates.

    Parameters:
    params (tuple of np.ndarray): (estimated_maturities, discount_factors).
    maturity (float or np.ndarray): Maturity(ies) for which to compute rates.

    Returns:
    np.ndarray: Interpolated interest rates.
    """
    estimated_maturities, betas = params
    # Convert discount factors to interest rates (continuous by default)
    est_rates = [discount_to_int_rate(b, m, None) for b, m in zip(betas, estimated_maturities)]

    f = interpolate.interp1d(
        estimated_maturities,
        est_rates,
        bounds_error=False,
        fill_value='extrapolate'
    )
    return f(maturity)


def estimate_curve_ols(
    CF: pd.DataFrame,
    prices: Union[pd.DataFrame, pd.Series, np.ndarray],
    interpolate: bool = False
) -> np.ndarray:
    """
    Fits a linear regression (with no intercept) using 'CF' values as predictors and 'prices' as targets, 
    returning an array of discount factors. If 'interpolate' is True, discount factors are further 
    interpolated over a maturity grid.

    Parameters:
    CF (pd.DataFrame): Cashflow matrix where rows are bonds and columns are dates, containing coupon payments.
    prices (pd.DataFrame, pd.Series, or np.ndarray): Observed bond prices. If a DataFrame/Series, values are matched to CF's rows.
    interpolate (bool, default=False): Whether to interpolate discount factors over the maturity grid.

    Returns:
    np.ndarray: Discount factors (one per column of CF if no interpolation, otherwise an interpolated array).
    """
    # If 'prices' is a DataFrame/Series, align it with CF's index
    if isinstance(prices, (pd.DataFrame, pd.Series)):
        prices = prices[CF.index].values

    # Fit linear regression with no intercept (discount factors = regression coefficients)
    model = LinearRegression(fit_intercept=False).fit(CF.values, prices)

    # By default, the discount factors are the regression coefficients
    discounts = model.coef_

    # If interpolation is requested
    if interpolate:
        # Create a grid of "maturity" for each column in CF relative to the first column
        matgrid = get_maturity_delta(CF.columns, CF.columns.min())

        # Filter out any coefficients that seem invalid (e.g., negative or too large)
        valid_mask = np.logical_and(discounts > 0, discounts < 1.25)

        xold = matgrid[valid_mask]
        yold = discounts[valid_mask]
        xnew = matgrid

        f = interpolate.interp1d(xold, yold, bounds_error=False, fill_value='extrapolate')
        discounts = f(xnew)

    return discounts


def nelson_siegel(params: List[float], maturity: Union[float]) -> np.ndarray:
    """
    Nelson-Siegel model: r(t) = beta0 + (beta1 + beta2)*[1-exp(-t/tau)]/(t/tau) - beta2*exp(-t/tau).

    Parameters:
    params (list of float): [beta0, beta1, beta2, tau].
    maturity (float): Maturity.

    Returns:
    np.ndarray: Modeled rates.
    """
    rate = params[0] + (params[1] + params[2]) * (1 - np.exp(-maturity/params[3]))/(maturity/params[3]) - params[2] * np.exp(-maturity/params[3])
    
    return rate


def nelson_siegel_extended(params: List[float], maturity: float) -> np.ndarray:
    """
    Extended Nelson-Siegel: r(t) = standard NS + beta4 * ([1-exp(-t/tau2)]/(t/tau2) - exp(-t/tau2)).

    Parameters:
    params (list of float): [beta0, beta1, beta2, tau, beta4, tau2].
    maturity (float): Maturity.

    Returns:
    np.ndarray: Modeled rates.
    """
    rate = params[0] + (params[1] + params[2]) * (1 - np.exp(-maturity/params[3]))/(maturity/params[3]) - params[2] * np.exp(-maturity/params[3]) + params[4] *((1-np.exp(-maturity/params[5]))/(maturity/params[5]) - np.exp(-maturity/params[5]))
    
    return rate


def price_with_rate_model(
    params: Union[List[float], np.ndarray, Tuple[np.ndarray, np.ndarray]],
    cf: pd.DataFrame,
    date_current: Union[str, datetime.date, datetime.datetime, np.datetime64],
    func_model,
    convert_to_discount: bool = True,
    price_coupons: bool = False
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Computes bond prices from model parameters and a cashflow matrix.

    Parameters:
    params (list or np.ndarray or tuple): Model parameters (e.g. [beta0,beta1,beta2,tau] or (maturities, discounts)).
    cf (pd.DataFrame): Cashflow matrix, rows=bonds, columns=dates.
    date_current (str or date or datetime or np.datetime64): Current date to compute maturities.
    func_model (callable): Function of form func_model(params, maturity)->rate or discount.
    convert_to_discount (bool, default=True): If True, model returns rates which get converted to discount factors.
    price_coupons (bool, default=False): If True, returns discounted CF matrix; else returns sum across columns.

    Returns:
    np.ndarray or pd.DataFrame: Prices (or matrix of discounted CF).
    """
    maturity = get_maturity_delta(cf.columns, date_current)
    maturity = maturity.values  # 1D array of column maturities

    # Convert interest rates -> discount factors if needed
    if convert_to_discount:
        discount_factors = []
        for matur in maturity:
            rate = func_model(params, matur)
            discount_factors.append(int_rate_to_discount(rate, matur))
        discount_factors = np.array(discount_factors)
    else:
        # The model itself returns discount factors
        discount_factors = np.array([func_model(params, matur) for matur in maturity])

    if price_coupons:
        # Return the matrix of discounted cashflows
        price = cf.mul(discount_factors, axis='columns') 
    else:
        # Price is sum across columns
        price = cf.values @ discount_factors
    
    return price


def pricing_errors(
    params: Union[List[float], np.ndarray, Tuple[np.ndarray, np.ndarray]],
    cf: pd.DataFrame,
    date_current: Union[str, datetime.date, datetime.datetime, np.datetime64],
    func_model,
    observed_prices: Union[pd.DataFrame, pd.Series, np.ndarray]
) -> float:
    """
    Objective function for curve-fitting: sum of squared errors 
    between observed prices and modeled prices.

    Parameters:
    params (list, np.ndarray or tuple): Model parameters.
    cf (pd.DataFrame): Cashflow matrix.
    date_current (str or date or datetime or np.datetime64): Current date for maturity calc.
    func_model (callable): Rate model function.
    observed_prices (pd.DataFrame, pd.Series or np.ndarray): Observed bond prices.

    Returns:
    float: Sum of squared errors.
    """
    modeled_prices = price_with_rate_model(params, cf, date_current, func_model)

    if isinstance(observed_prices, (pd.DataFrame, pd.Series)):
        observed_prices = observed_prices.values

    return float(np.sum((observed_prices - modeled_prices) ** 2))


def estimate_rate_curve(
    model_name: str,
    cf: pd.DataFrame,
    date_current: Union[str, datetime.date, datetime.datetime, np.datetime64],
    prices: Union[pd.DataFrame, pd.Series, np.ndarray],
    x0: Optional[Union[List[float], np.ndarray]] = None
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Estimates parameters by fitting bond prices to a curve model.

    Parameters:
    model_name (str): Model to estimate interest rate curve (e.g. bootstrap, nelson_siegel, etc.).
    cf (pd.DataFrame): Cashflow matrix [n_bonds x n_dates].
    date_curr (str or date or datetime or np.datetime64): Current date to calculate maturity.
    prices (pd.DataFrame, pd.Series or np.ndarray): Observed bond prices.
    x0 (list or np.ndarray, optional): Initial guess for model parameters.

    Returns:
    np.ndarray or tuple: Fitted parameters (for bootstrap returns (maturities, discounts)).
    """
    p = prices.copy()
    if isinstance(p, (pd.DataFrame, pd.Series)):
        p = p[cf.index].values

    # If model == 'OLS', use linear regression
    if model_name == 'OLS':
        # OLS on CF matrix to get discount(s)
        discounts = estimate_curve_ols(cf, prices, interpolate=False)
        cf_intervals = get_maturity_delta(cf.columns.to_series(),date_current=date_current).values    
        params_est = [cf_intervals, discounts]

    elif model_name == 'bootstrap':
        discounts = estimate_curve_ols(cf, prices, interpolate=False)
        cf_intervals = get_maturity_delta(cf.columns.to_series(),date_current=date_current).values    
        params_est = [cf_intervals, discounts]

    # Otherwise, numeric optimization
    else:
        if model_name == 'nelson_siegel_extended':
            model_obj = nelson_siegel_extended
            if x0 is None:
                x0 = np.ones(6)   
        else:
            if model_name != 'nelson_siegel': print(f"Model '{model_name}' not recognized. Using default 'nelson_siegel'.")	
            model_obj = nelson_siegel
            if x0 is None: x0 = np.ones(4) * 0.1  # some default

        # optimizer iteratively updates params (starting from x0) to minimize the sum of squared errors calculated in pricing_errors.
        result = minimize(pricing_errors, x0, args=(cf, date_current, model_obj, prices))
        params_est = result.x

    return params_est
    
```

## New functions that you will refactor:


def extract_spot_curves(quote_date, filepath=None, model=nelson_siegel, delta_maturity = .25, T=30,calc_forward=False, delta_forward_multiple = 1, filter_maturity_dates=False, filter_tips=True):

    if filepath is None:
        filepath = f'../data/treasury_quotes_{quote_date}.xlsx'
        
    rawdata = pd.read_excel(filepath,sheet_name='quotes')
    
    rawdata.columns = rawdata.columns.str.upper()
    rawdata.sort_values('TMATDT',inplace=True)
    rawdata.set_index('KYTREASNO',inplace=True)

    t_check = rawdata['CALDT'].values[0]
    if rawdata['CALDT'].eq(t_check).all():
        t_current = t_check
    else:
        warnings.warn('Quotes are from multiple dates.')
        t_current = None

    rawprices = (rawdata['TDBID'] + rawdata['TDASK'])/2 + rawdata['TDACCINT']
    rawprices.name = 'price'

    ###
    data = filter_treasuries(rawdata, t_date=t_current, filter_tips=filter_tips)

    CF = filter_treasury_cashflows(calc_cashflows(data),filter_maturity_dates=filter_maturity_dates)
    prices = rawprices[CF.index]

    ###
    params = estimate_rate_curve(model,CF,t_current,prices)
    
    if model == nelson_siegel_extended:
        params0 = estimate_rate_curve(nelson_siegel,CF,t_current,prices)
        x0 = np.concatenate((params0,(1,1)))
        params = estimate_rate_curve(model,CF,t_current,prices,x0=x0)
        
    else:
        params = estimate_rate_curve(model,CF,t_current,prices)

    ###
    maturity_grid = np.arange(0,T+delta_maturity,delta_maturity)
    maturity_grid[0] = .01
    
    curves = pd.DataFrame(index = pd.Index(maturity_grid,name='maturity'))
    # adjust earliest maturity from 0 to epsion
    curves.columns.name = quote_date
    
    curves['spot rate']= model(params,maturity_grid)

    curves['spot discount'] = intrate_to_discount(curves['spot rate'].values,curves.index.values)
    
    
    
    if calc_forward:
        delta_forward = delta_forward_multiple * delta_maturity
        
        curves['forward discount'] = curves['spot discount'] / curves['spot discount'].shift(delta_forward_multiple)

        # first value of forward is spot rate
        maturity_init = curves.index[0:delta_forward_multiple]
        curves.loc[maturity_init,'forward discount'] = curves.loc[maturity_init,'spot discount']
        
        curves.insert(2,'forward rate', -np.log(curves['forward discount'])/delta_forward)
        
    return curves



def process_treasury_quotes(quote_date):
    
    filepath_rawdata = f'../data/treasury_quotes_{quote_date}.xlsx'
    rawdata = pd.read_excel(filepath_rawdata,sheet_name='quotes')
    rawdata.columns = rawdata.columns.str.upper()
    rawdata.sort_values('TMATDT',inplace=True)
    rawdata.set_index('KYTREASNO',inplace=True)

    t_check = rawdata['CALDT'].values[0]
    if rawdata['CALDT'].eq(t_check).all():
        t_current = t_check
    else:
        warnings.warn('Quotes are from multiple dates.')
        t_current = None

    rawprices = (rawdata['TDBID'] + rawdata['TDASK'])/2 + rawdata['TDACCINT']
    rawprices.name = 'price'

    maturity_delta = get_maturity_delta(rawdata['TMATDT'],t_current)
    maturity_delta.name = 'maturity delta'

    metrics = rawdata.copy()[['TDATDT','TMATDT','TDPUBOUT','TCOUPRT','TDYLD','TDDURATN']]
    metrics.columns = ['issue date','maturity date','outstanding','coupon rate','yld','duration']
    metrics['yld'] *= 365
    metrics['duration'] /= 365
    metrics['outstanding'] *= 1e6
    metrics['maturity interval'] = get_maturity_delta(metrics['maturity date'], t_current)
    metrics['price'] = rawprices
    
    return metrics


def get_bond(quote_date,maturity=None,coupon=None,selection='nearest'):
    
    metrics = process_treasury_quotes(quote_date)

    if coupon is not None:
        metrics = metrics[metrics['coupon rate']==coupon]
    
    if maturity is not None:
        mats = metrics['maturity interval']

        if type(maturity) is float:
            maturity = [maturity]

        idx = list()

        for m in maturity:

            if selection == 'nearest':
                idx.append(mats.sub(m).abs().idxmin())
            elif selection == 'ceil':
                idx.append(mats.sub(m).where(mats > 0, np.inf).argmin())
            elif selection == 'floor':
                idx.append(mats.sub(m).where(mats < 0, -np.inf).argmax())

        metrics = metrics.loc[idx,:]

    return metrics


def get_bond_raw(quote_date):
    
    filepath_rawdata = f'../data/treasury_quotes_{quote_date}.xlsx'
    rawdata = pd.read_excel(filepath_rawdata,sheet_name='quotes')
    rawdata.columns = rawdata.columns.str.upper()
    rawdata.sort_values('TMATDT',inplace=True)
    rawdata.set_index('KYTREASNO',inplace=True)

    t_check = rawdata['CALDT'].values[0]
    if rawdata['CALDT'].eq(t_check).all():
        t_current = t_check
    else:
        warnings.warn('Quotes are from multiple dates.')
        t_current = None
        
    return rawdata, t_current


def bootstrap_spot_rates(df):
    """
    Bootstraps spot rates from a dataframe of bond information.
    
    :param df: Pandas DataFrame with columns 'price', 'cpn rate', and 'ttm'
    :return: Pandas Series of spot rates indexed by TTM
    """
    # Ensure the DataFrame is sorted by TTM
    df = df.sort_values(by='ttm')
    
    # Initialize a dictionary to store spot rates
    spot_rates = {}

    # Iterate over each bond
    for index, row in df.iterrows():
        ttm, coupon_rate, price = row['ttm'], row['cpn rate'], row['price']
        cash_flows = [coupon_rate / 2] * round(ttm * 2)  # Semi-annual coupons
        cash_flows[-1] += 100  # Add the face value to the last cash flow

        # Function to calculate the present value of cash flows
        def pv_of_cash_flows(spot_rate):
            pv = 0
            for t in range(1, len(cash_flows) + 1):
                if t/2 in spot_rates:
                    rate = spot_rates[t/2]
                else:
                    rate = spot_rate
                pv += cash_flows[t - 1] / ((1 + rate / 2) ** t)
            return pv

        # Solve for the spot rate that sets the present value of cash flows equal to the bond price
        spot_rate_guess = (cash_flows[-1] / price) ** (1/(ttm*2)) - 1
        spot_rate = fsolve(lambda r: pv_of_cash_flows(r) - price, x0=spot_rate_guess)[0]

        # Store the calculated spot rate
        spot_rates[ttm] = spot_rate

    return pd.Series(spot_rates)


def process_wrds_treasury_data(rawdata,keys_extra=[]):

    DAYS_YEAR = 365.25
    FREQ = 2
    # could pull this directly from TNIPPY

    data = rawdata.copy()
    
    data.columns = data.columns.str.upper()

    data.sort_values('TMATDT',inplace=True)
    data.set_index('KYTREASNO',inplace=True)

    data = data[['CALDT','TDBID','TDASK','TDNOMPRC','TDACCINT','TDYLD','TDATDT','TMATDT','TCOUPRT','ITYPE','TDDURATN','TDPUBOUT','TDTOTOUT']]


    ### List issue type
    dict_type = {
        1: 'bond',
        2: 'note',
        4: 'bill',
        11: 'TIPS note',
        12: 'TIPS bond'
    }
    
    data['ITYPE'] = data['ITYPE'].replace(dict_type)
    

    ### Rename columns
    data.rename(columns={'CALDT':'quote date',
                         'TDATDT':'issue date',
                         'TMATDT':'maturity date',
                         'TCOUPRT':'cpn rate',
                         'TDTOTOUT':'total size',
                         'TDPUBOUT':'public size',
                         'TDDURATN':'duration',
                         'ITYPE':'type',
                         'TDBID':'bid',
                         'TDASK':'ask',
                         'TDNOMPRC':'price',
                         'TDACCINT':'accrued int',
                         'TDYLD':'ytm'
                        },inplace=True)


    ### Calculate time-to-maturity (TTM)
    data['maturity date'] = pd.to_datetime(data['maturity date'])
    data['issue date'] = pd.to_datetime(data['issue date'])
    data['quote date'] = pd.to_datetime(data['quote date'])
    data['ttm'] = (data['maturity date'] - data['quote date']).dt.days.astype(float)/DAYS_YEAR

    
    ### Dirty price
    data['dirty price'] = data['price'] + data['accrued int']    


    ### duration
    data['duration'] /= 365
    data['total size'] *= 1e6
    data['public size'] *= 1e6
    
    
    ### Annualize YTM for semi-compounding
    def tempfunc(x):
        return (np.exp(x*DAYS_YEAR/FREQ)-1)*FREQ

    data['ytm'] = data['ytm'].apply(tempfunc)

    
    ### accrual fraction
    data['accrual fraction'] = data['accrued int'] / (data['cpn rate'] / FREQ)

    idx = data['accrual fraction'].isna()
    data.loc[idx,'accrual fraction'] = 1 - (data.loc[idx,'ttm']-round(data.loc[idx,'ttm']))*FREQ

    
    ### Reorganize columns
    keys = ['type',
            'quote date',
            'issue date',
            'maturity date',
            'ttm',
            'accrual fraction',
            'cpn rate',
            'bid',
            'ask',
            'price',
            'accrued int',
            'dirty price',
            'ytm']
    
    data = data[keys+keys_extra]

    
    return data            


from pandas.tseries.offsets import DateOffset


def select_maturities(rawdata,periods=20,freq='6ME'):

    data = rawdata.copy()[['quote date','issue date','maturity date']]
    
    # Convert DATE and columns in 'data' to datetime
    DATE = data['quote date'].iloc[0]
    DAYS_YEAR = 365

    FREQ = freq

    # Generate 6-month intervals from DATE
    six_month_intervals = pd.date_range(start=DATE, periods=periods+1, freq=FREQ)[1:]
    
    
    def find_closest_date(interval, data):
    
        # Calculate the absolute difference between each MATURITY date and the interval
        data['difference'] = abs(data['maturity date'] - interval)
        
        # Ensure we only consider future dates relative to DATE
        DATE = data['quote date'].iloc[0]
        future_dates = data[data['maturity date'] > DATE]
        if not future_dates.empty:
            # Find the row with the minimum difference
            min_diff = future_dates['difference'].min()
            closest_dates = future_dates[future_dates['difference'] == min_diff]
            # Resolve ties by 'tdatdt' date
            return closest_dates.sort_values('issue date', ascending=False).iloc[0]
            
        return None
    
    # Apply the function to each interval
    selected_rows = [find_closest_date(interval, data) for interval in six_month_intervals]
    
    # Remove None values and ensure uniqueness
    selected_rows = [row for row in selected_rows if row is not None]
    select_ids = [row.name for row in selected_rows]

    # algorithm includes
    #select_ids = select_ids[1:]
    return select_ids