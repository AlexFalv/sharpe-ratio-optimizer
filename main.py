import numpy as np
import pandas as pd
from data_fetcher import fetch_stock_data
from optimizer import (
    calculate_log_returns,
    calculate_portfolio_stats,
    find_optimal_weights,
    run_monte_carlo_simulation
)
from visualizer import plot_efficient_frontier

# --- CONFIGURATION ---
# Define the 20 stocks for your portfolio
# A diverse mix of tech, finance, healthcare, and consumer goods
TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN',  # Tech
    'JPM', 'V', 'BAC',               # Finance
    'JNJ', 'PFE', 'MRK',             # Healthcare
    'PG', 'KO', 'WMT', 'MCD',       # Consumer Goods
    'XOM', 'CVX',                    # Energy
    'TSLA', 'NVDA',                  # High-growth Tech
    'UNH', 'HD'                      # Other
]
START_DATE = '2020-01-01'
END_DATE = '2024-12-31'
NUM_PORTFOLIOS_MC = 20000 # Number of simulations for Monte Carlo

# --- 1. DATA FETCHING ---
price_data = fetch_stock_data(TICKERS, START_DATE, END_DATE)

# --- 2. CALCULATIONS ---
log_returns = calculate_log_returns(price_data)
mean_returns = log_returns.mean()
cov_matrix = log_returns.cov()

# --- 3. OPTIMIZATION ---
print("Finding optimal portfolios...")

# Find portfolio that maximizes Sharpe Ratio
max_sharpe_weights = find_optimal_weights(mean_returns, cov_matrix, target_function='sharpe')
max_sharpe_stats = calculate_portfolio_stats(max_sharpe_weights, mean_returns, cov_matrix)

# Find portfolio that minimizes Volatility
min_vol_weights = find_optimal_weights(mean_returns, cov_matrix, target_function='volatility')
min_vol_stats = calculate_portfolio_stats(min_vol_weights, mean_returns, cov_matrix)

print("Optimization complete.")

# --- 4. MONTE CARLO SIMULATION ---
print(f"Running {NUM_PORTFOLIOS_MC} Monte Carlo simulations...")
mc_results, _ = run_monte_carlo_simulation(NUM_PORTFOLIOS_MC, mean_returns, cov_matrix)
print("Simulation complete.")

# --- 5. RESULTS & VISUALIZATION ---

# Print optimal portfolio details
print("\n--- Optimal Portfolio (Max Sharpe Ratio) ---")
print(f"  Return: {max_sharpe_stats[0]:.4f}")
print(f"  Volatility: {max_sharpe_stats[1]:.4f}")
print(f"  Sharpe Ratio: {max_sharpe_stats[2]:.4f}")
print("  Weights:")
max_sharpe_weights_series = pd.Series(max_sharpe_weights, index=TICKERS)
print(max_sharpe_weights_series[max_sharpe_weights_series > 0.01].sort_values(ascending=False)) # Show non-trivial weights

print("\n--- Optimal Portfolio (Min Volatility) ---")
print(f"  Return: {min_vol_stats[0]:.4f}")
print(f"  Volatility: {min_vol_stats[1]:.4f}")
print(f"  Sharpe Ratio: {min_vol_stats[2]:.4f}")
print("  Weights:")
min_vol_weights_series = pd.Series(min_vol_weights, index=TICKERS)
print(min_vol_weights_series[min_vol_weights_series > 0.01].sort_values(ascending=False))

# Plot the efficient frontier
optimal_portfolios_for_plot = {
    'Max Sharpe Portfolio': (max_sharpe_stats[0], max_sharpe_stats[1], max_sharpe_weights),
    'Min Volatility Portfolio': (min_vol_stats[0], min_vol_stats[1], min_vol_weights)
}
plot_efficient_frontier(mc_results, optimal_portfolios_for_plot)