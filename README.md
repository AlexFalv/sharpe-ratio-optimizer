# sharpe-ratio-optimizer
A Python project to find the optimal stock portfolio
Markowitz Portfolio Optimizer

A Python-based tool to construct and analyze an optimal investment portfolio using Modern Portfolio Theory (Markowitz). This script fetches historical stock data, calculates the efficient frontier, and identifies the portfolios that either maximize the Sharpe Ratio or minimize volatility.

Core Features

Data Fetching: Pulls historical stock data from Yahoo Finance using yfinance.

Mean-Variance Optimization: Implements Markowitz optimization to find the optimal portfolio weights.

Optimization Targets:

Maximizes the Sharpe Ratio (tangency portfolio).

Minimizes Volatility (minimum variance portfolio).

Monte Carlo Simulation: Generates thousands of random portfolios to visualize the full efficient frontier.

Visualization: Plots the efficient frontier using matplotlib, highlighting the optimal portfolios.

Technology Stack

Python 3

NumPy: For numerical operations and linear algebra.

Pandas: For data manipulation and time-series analysis.

SciPy: For the core minimize (SLSQP) optimization algorithm.

yfinance: For sourcing historical market data.

Matplotlib: For plotting the efficient frontier.

How to Run

Clone the repository:

git clone [https://github.com/](https://github.com/)[YOUR_USERNAME]/sharpe-ratio-optimizer.git
cd sharpe-ratio-optimizer


Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate


Install the required libraries:

pip install -r requirements.txt


Run the main script:

python main.py


Example Output

The script will print the optimal portfolio weights to the console and generate a plot named efficient_frontier.png.

Console Output:

--- Optimal Portfolio (Max Sharpe Ratio) ---
  Return: 0.2350
  Volatility: 0.1820
  Sharpe Ratio: 1.1813
  Weights:
NVDA     0.350013
MSFT     0.220147
...


Plot:
<img width="1008" height="732" alt="Capture d’écran 2025-11-08 à 22 16 51" src="https://github.com/user-attachments/assets/1a3aa90c-9f6a-4ca1-a8e5-ad1f47d6dcfd" />
