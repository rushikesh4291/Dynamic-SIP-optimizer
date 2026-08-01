import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, norm
from scipy.optimize import newton

# --- Performance Metrics ---
def calculate_performance_metrics(daily_returns, annual_rf_rate=0.065):
    """
    Calculates true Sharpe and Sortino ratios using an annualized hurdle rate.
    """
    daily_rf = (1 + annual_rf_rate) ** (1 / 252) - 1
    excess_returns = daily_returns - daily_rf
    
    # Sharpe with ddof=1 for sample standard deviation
    sharpe = (np.mean(excess_returns) / np.std(daily_returns, ddof=1)) * np.sqrt(252)
    
    downside_diff = np.minimum(0, excess_returns)
    downside_deviation = np.sqrt(np.mean(np.square(downside_diff)))
    
    sortino = np.inf if downside_deviation == 0 else (np.mean(excess_returns) / downside_deviation) * np.sqrt(252)
    
    return sharpe, sortino


# --- Risk Metrics ---
class RiskEngine:
    """High-speed vectorized risk calculations."""
    
    @staticmethod
    def calculate_max_drawdown(daily_returns):
        wealth_index = np.cumprod(1 + daily_returns)
        running_max = np.maximum.accumulate(wealth_index)
        drawdowns = (wealth_index - running_max) / running_max
        return np.abs(np.min(drawdowns))

    @staticmethod
    def calculate_modified_cvar(daily_returns, confidence_level=0.05):
        """
        Calculates the Cornish-Fisher Modified CVaR (Expected Shortfall) to account for fat tails.
        """
        mu = np.mean(daily_returns)
        sigma = np.std(daily_returns, ddof=1)
        S = skew(daily_returns)
        K = kurtosis(daily_returns)
        
        z = norm.ppf(confidence_level)
        z_cf = (z + (1/6)*(z**2 - 1)*S + (1/24)*(z**3 - 3*z)*K - (1/36)*(2*z**3 - 5*z)*S**2)
        
        mod_var = mu + z_cf * sigma
        tail_events = daily_returns[daily_returns <= mod_var]
        
        return np.mean(tail_events) if len(tail_events) > 0 else mod_var


# --- Return Calculations ---
def xirr_npv(rate, cashflows, days):
    """Calculates Net Present Value for a given rate."""
    return np.sum(cashflows / ((1 + rate) ** (days / 365.0)))

def calculate_xirr(cashflows, dates):
    """
    Roots out the exact annualized return based on irregular cash flows using Newton-Raphson.
    """
    days = np.array([(d - dates[0]).days for d in dates])
    try:
        # Newton-Raphson method to find the rate where NPV == 0
        return newton(lambda r: xirr_npv(r, cashflows, days), 0.10)
    except RuntimeError:
        return np.nan
