import numpy as np
import pandas as pd
from scipy.optimize import minimize

def calculate_log_returns(prices):
    """Calculates log returns from a DataFrame of prices."""
    return np.log(prices / prices.shift(1)).dropna()

def calculate_portfolio_stats(weights, mean_returns, cov_matrix):
    """
    Calculates the expected annual return, volatility (std dev),
    and Sharpe ratio for a given set of portfolio weights.
    
    Assumes 252 trading days in a year.
    """
    portfolio_return = np.dot(weights, mean_returns) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    
    # Assume a risk-free rate (e.g., 2% annual)
    risk_free_rate = 0.02
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    
    return portfolio_return, portfolio_volatility, sharpe_ratio

def get_optimizer_objective(target_function):
    """
    Returns a function that minimizes the negative of the target function.
    (Used to maximize Sharpe ratio, as optimizer only minimizes).
    """
    def objective(weights, mean_returns, cov_matrix):
        # We pass stats_func (e.g., calculate_sharpe)
        # We want to maximize sharpe, so we minimize -sharpe
        p_return, p_vol, p_sharpe = calculate_portfolio_stats(weights, mean_returns, cov_matrix)
        
        if target_function == 'sharpe':
            return -p_sharpe
        elif target_function == 'volatility':
            return p_vol
        else:
            raise ValueError("Invalid target_function")

    return objective


def find_optimal_weights(mean_returns, cov_matrix, target_function='sharpe'):
    """
    Finds the optimal portfolio weights to either maximize Sharpe ratio
    or minimize volatility.
    """
    num_assets = len(mean_returns)
    
    # Define the objective function
    objective_func = get_optimizer_objective(target_function)
    
    # Define constraints (weights must sum to 1)
    constraints = ({
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    })
    
    # Define bounds (each weight between 0 and 1, no short-selling)
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    # Initial guess (equal weights)
    initial_weights = np.array([1 / num_assets] * num_assets)
    
    # Run the optimization
    solution = minimize(
        objective_func,
        initial_weights,
        args=(mean_returns, cov_matrix),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    if not solution.success:
        raise Exception(f"Optimization failed: {solution.message}")
        
    return solution.x

def run_monte_carlo_simulation(num_portfolios, mean_returns, cov_matrix):
    """
    Runs a Monte Carlo simulation to generate random portfolio
    weights and their corresponding returns, volatilities, and Sharpe ratios.
    """
    num_assets = len(mean_returns)
    results = np.zeros((3, num_portfolios)) # [Return, Volatility, Sharpe]
    weights_record = []

    for i in range(num_portfolios):
        # Generate random weights
        weights = np.random.random(num_assets)
        # Normalize weights to sum to 1
        weights /= np.sum(weights)
        weights_record.append(weights)
        
        # Calculate stats for these weights
        p_return, p_vol, p_sharpe = calculate_portfolio_stats(weights, mean_returns, cov_matrix)
        
        results[0, i] = p_return
        results[1, i] = p_vol
        results[2, i] = p_sharpe
        
    return results, weights_record