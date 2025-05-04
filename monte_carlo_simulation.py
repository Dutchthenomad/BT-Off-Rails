import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime


def run_monte_carlo_simulation(trades_df, initial_capital=50.0, num_simulations=1000, confidence_level=0.95):
    """
    Run a Monte Carlo simulation on trade results to estimate strategy performance
    
    Parameters:
    - trades_df: DataFrame with trade results (must contain 'return_pct' column)
    - initial_capital: Starting capital amount (default: $50)
    - num_simulations: Number of Monte Carlo simulations to run (default: 1000)
    - confidence_level: Confidence level for calculating drawdown and profit metrics (default: 0.95)
    
    Returns:
    - Dictionary with simulation results
    """
    # Extract returns from trades DataFrame
    returns = trades_df['return_pct'].values
    
    # Create array to store simulation results
    simulation_results = np.zeros((num_simulations, len(returns) + 1))
    simulation_results[:, 0] = initial_capital
    
    # Run simulations
    for i in range(num_simulations):
        # Shuffle the returns to create a random sequence
        np.random.shuffle(returns)
        
        # Calculate equity curve for this simulation
        for j in range(len(returns)):
            simulation_results[i, j+1] = simulation_results[i, j] * (1 + returns[j])
    
    # Calculate metrics from simulations
    final_capitals = simulation_results[:, -1]
    
    # Calculate drawdowns for each simulation
    max_drawdowns = np.zeros(num_simulations)
    for i in range(num_simulations):
        # Calculate running maximum
        running_max = np.maximum.accumulate(simulation_results[i, :])
        # Calculate drawdown
        drawdown = (simulation_results[i, :] - running_max) / running_max
        # Store maximum drawdown
        max_drawdowns[i] = np.min(drawdown)
    
    # Calculate confidence intervals
    lower_idx = int((1 - confidence_level) / 2 * num_simulations)
    upper_idx = int((1 - (1 - confidence_level) / 2) * num_simulations)
    
    # Sort results for percentile calculations
    sorted_final_capitals = np.sort(final_capitals)
    sorted_max_drawdowns = np.sort(max_drawdowns)
    
    # Calculate metrics
    metrics = {
        'mean_final_capital': np.mean(final_capitals),
        'median_final_capital': np.median(final_capitals),
        'min_final_capital': np.min(final_capitals),
        'max_final_capital': np.max(final_capitals),
        'lower_ci_capital': sorted_final_capitals[lower_idx],
        'upper_ci_capital': sorted_final_capitals[upper_idx],
        'mean_return': (np.mean(final_capitals) / initial_capital) - 1,
        'median_return': (np.median(final_capitals) / initial_capital) - 1,
        'mean_max_drawdown': np.mean(max_drawdowns),
        'median_max_drawdown': np.median(max_drawdowns),
        'worst_drawdown': np.min(max_drawdowns),
        'lower_ci_drawdown': sorted_max_drawdowns[lower_idx],
        'upper_ci_drawdown': sorted_max_drawdowns[upper_idx],
        'probability_profit': np.mean(final_capitals > initial_capital),
        'probability_loss': np.mean(final_capitals < initial_capital),
        'probability_double': np.mean(final_capitals > 2 * initial_capital),
        'probability_half': np.mean(final_capitals < 0.5 * initial_capital),
        'simulation_data': simulation_results
    }
    
    return metrics


def plot_monte_carlo_results(metrics, initial_capital=50.0, confidence_level=0.95, num_paths_to_plot=100):
    """
    Plot Monte Carlo simulation results
    
    Parameters:
    - metrics: Dictionary with simulation results from run_monte_carlo_simulation
    - initial_capital: Starting capital amount (default: $50)
    - confidence_level: Confidence level used in simulation (default: 0.95)
    - num_paths_to_plot: Number of random paths to plot (default: 100)
    """
    # Create output directory if it doesn't exist
    os.makedirs('monte_carlo_results', exist_ok=True)
    
    # Get simulation data
    simulation_data = metrics['simulation_data']
    num_simulations = simulation_data.shape[0]
    num_periods = simulation_data.shape[1]
    
    # Select random paths to plot
    if num_simulations > num_paths_to_plot:
        indices = np.random.choice(num_simulations, num_paths_to_plot, replace=False)
    else:
        indices = range(num_simulations)
    
    # Create figure for equity curves
    plt.figure(figsize=(12, 8))
    
    # Plot random paths
    for i in indices:
        plt.plot(simulation_data[i, :], 'b-', alpha=0.1)
    
    # Calculate and plot percentiles
    percentiles = [5, 25, 50, 75, 95]
    percentile_values = np.percentile(simulation_data, percentiles, axis=0)
    
    # Plot percentile lines
    colors = ['r', 'm', 'g', 'm', 'r']
    labels = ['5th', '25th', '50th (Median)', '75th', '95th']
    
    for i, percentile in enumerate(percentiles):
        plt.plot(percentile_values[i, :], colors[i], linewidth=2, label=f'{labels[i]} Percentile')
    
    # Plot initial capital line
    plt.axhline(y=initial_capital, color='k', linestyle='--', label='Initial Capital')
    
    # Add labels and title
    plt.xlabel('Trade Number')
    plt.ylabel('Account Balance ($)')
    plt.title(f'Monte Carlo Simulation: {num_simulations} Paths ({confidence_level*100:.0f}% Confidence Interval)')
    plt.legend()
    plt.grid(True)
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'monte_carlo_results/equity_curves_{timestamp}.png', dpi=300)
    
    # Create figure for final capital distribution
    plt.figure(figsize=(12, 8))
    
    # Get final capitals
    final_capitals = simulation_data[:, -1]
    
    # Plot histogram
    plt.hist(final_capitals, bins=50, alpha=0.75)
    
    # Plot vertical lines for key metrics
    plt.axvline(x=initial_capital, color='k', linestyle='--', label='Initial Capital')
    plt.axvline(x=metrics['mean_final_capital'], color='r', linestyle='-', label='Mean Final Capital')
    plt.axvline(x=metrics['median_final_capital'], color='g', linestyle='-', label='Median Final Capital')
    plt.axvline(x=metrics['lower_ci_capital'], color='m', linestyle=':', label=f'{confidence_level*100:.0f}% CI Lower')
    plt.axvline(x=metrics['upper_ci_capital'], color='m', linestyle=':', label=f'{confidence_level*100:.0f}% CI Upper')
    
    # Add labels and title
    plt.xlabel('Final Account Balance ($)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Final Account Balances')
    plt.legend()
    plt.grid(True)
    
    # Save figure
    plt.savefig(f'monte_carlo_results/final_capital_distribution_{timestamp}.png', dpi=300)
    
    # Print summary statistics
    print("\nMonte Carlo Simulation Results:")
    print(f"Number of simulations: {num_simulations}")
    print(f"Confidence level: {confidence_level*100:.0f}%")
    print("\nFinal Capital Statistics:")
    print(f"Mean final capital: ${metrics['mean_final_capital']:.2f} (Return: {metrics['mean_return']*100:.2f}%)")
    print(f"Median final capital: ${metrics['median_final_capital']:.2f} (Return: {metrics['median_return']*100:.2f}%)")
    print(f"Min final capital: ${metrics['min_final_capital']:.2f}")
    print(f"Max final capital: ${metrics['max_final_capital']:.2f}")
    print(f"{confidence_level*100:.0f}% Confidence Interval: ${metrics['lower_ci_capital']:.2f} to ${metrics['upper_ci_capital']:.2f}")
    print("\nDrawdown Statistics:")
    print(f"Mean maximum drawdown: {metrics['mean_max_drawdown']*100:.2f}%")
    print(f"Median maximum drawdown: {metrics['median_max_drawdown']*100:.2f}%")
    print(f"Worst drawdown: {metrics['worst_drawdown']*100:.2f}%")
    print(f"{confidence_level*100:.0f}% Confidence Interval: {metrics['lower_ci_drawdown']*100:.2f}% to {metrics['upper_ci_drawdown']*100:.2f}%")
    print("\nProbability Analysis:")
    print(f"Probability of profit: {metrics['probability_profit']*100:.2f}%")
    print(f"Probability of loss: {metrics['probability_loss']*100:.2f}%")
    print(f"Probability of doubling account: {metrics['probability_double']*100:.2f}%")
    print(f"Probability of losing half of account: {metrics['probability_half']*100:.2f}%")
    
    # Show plots
    plt.show()


def run_simulation_from_csv(csv_file, initial_capital=50.0, num_simulations=1000, confidence_level=0.95):
    """
    Run Monte Carlo simulation from a CSV file containing trade results
    
    Parameters:
    - csv_file: Path to CSV file with trade results (must contain 'return_pct' column)
    - initial_capital: Starting capital amount (default: $50)
    - num_simulations: Number of Monte Carlo simulations to run (default: 1000)
    - confidence_level: Confidence level for calculating metrics (default: 0.95)
    """
    # Load trade results
    trades_df = pd.read_csv(csv_file)
    
    # Check if return_pct column exists
    if 'return_pct' not in trades_df.columns:
        # Try to calculate it from other columns if available
        if 'pnl' in trades_df.columns and 'position_size' in trades_df.columns:
            trades_df['return_pct'] = trades_df['pnl'] / trades_df['position_size']
        else:
            raise ValueError("CSV file must contain 'return_pct' column or both 'pnl' and 'position_size' columns")
    
    # Run simulation
    metrics = run_monte_carlo_simulation(
        trades_df, 
        initial_capital=initial_capital,
        num_simulations=num_simulations,
        confidence_level=confidence_level
    )
    
    # Plot results
    plot_monte_carlo_results(
        metrics, 
        initial_capital=initial_capital,
        confidence_level=confidence_level
    )
    
    return metrics


def run_monte_carlo_by_symbol(results_dir='results_aggressive', initial_capital=50.0, num_simulations=1000):
    """
    Run Monte Carlo simulations for each symbol in the results directory
    
    Parameters:
    - results_dir: Directory containing results CSV files (default: 'results_aggressive')
    - initial_capital: Starting capital amount (default: $50)
    - num_simulations: Number of Monte Carlo simulations to run (default: 1000)
    """
    # Get list of all_trades.csv files in the results directory
    trades_file = os.path.join(results_dir, 'all_trades.csv')
    
    if os.path.exists(trades_file):
        # Load all trades
        all_trades_df = pd.read_csv(trades_file)
        
        # Get unique symbols
        symbols = all_trades_df['symbol'].unique()
        
        # Run simulation for each symbol
        for symbol in symbols:
            print(f"\n=== Running Monte Carlo Simulation for {symbol} ===")
            
            # Filter trades for this symbol
            symbol_trades = all_trades_df[all_trades_df['symbol'] == symbol]
            
            # Run simulation
            metrics = run_monte_carlo_simulation(
                symbol_trades, 
                initial_capital=initial_capital,
                num_simulations=num_simulations
            )
            
            # Plot results
            plot_monte_carlo_results(metrics, initial_capital=initial_capital)
    else:
        print(f"Error: Could not find trades file at {trades_file}")


# Example usage
if __name__ == "__main__":
    # Run simulation from all trades
    run_simulation_from_csv('results_aggressive/all_trades.csv', initial_capital=50.0, num_simulations=1000)
    
    # Alternatively, run simulations for each symbol separately
    # run_monte_carlo_by_symbol(results_dir='results_aggressive', initial_capital=50.0, num_simulations=1000)
