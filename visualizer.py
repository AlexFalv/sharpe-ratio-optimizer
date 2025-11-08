import matplotlib.pyplot as plt

def plot_efficient_frontier(mc_results, optimal_portfolios):
    """
    Plots the Efficient Frontier using Monte Carlo results
    and highlights the optimal portfolios.
    
    mc_results: (3, N) numpy array [returns, volatilities, sharpes]
    optimal_portfolios: dict of {'name': (return, volatility, weights)}
    """
    
    # Create the scatter plot of all simulated portfolios
    plt.figure(figsize=(12, 8))
    
    # Scatter plot: color by Sharpe ratio
    scatter = plt.scatter(
        mc_results[1, :], # Volatility (x-axis)
        mc_results[0, :], # Return (y-axis)
        c=mc_results[2, :], # Color (Sharpe ratio)
        cmap='viridis',
        marker='o',
        s=10,
        alpha=0.3
    )
    
    # Add color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Sharpe Ratio')
    
    # Plot the optimal portfolios as large, distinct markers
    for name, stats in optimal_portfolios.items():
        p_return, p_vol, p_weights = stats
        plt.scatter(
            p_vol,
            p_return,
            marker='*', # Star marker
            color='red' if 'Sharpe' in name else 'blue',
            s=300, # Large size
            label=f'{name} (Sharpe: {p_return/p_vol:.2f})'
        )

    # Set labels and title
    plt.title('Efficient Frontier (Monte Carlo Simulation)')
    plt.xlabel('Annualized Volatility')
    plt.ylabel('Annualized Return')
    plt.legend(loc='upper left')
    plt.grid(True)
    
    # Save the plot to a file
    plot_filename = 'efficient_frontier.png'
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    
    # Show the plot
    plt.show()