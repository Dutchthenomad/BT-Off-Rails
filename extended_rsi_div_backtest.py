import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from datetime import datetime
import glob
import warnings
warnings.filterwarnings('ignore')

# Constants and parameters
RSI_LENGTH = 6
ATR_LENGTH = 6
ATR_MULTIPLIER = 1.1
PROFIT_TARGET_PCT = 0.01
STOP_LOSS_PCT = 0.008
SWING_WINDOW = 3
INITIAL_CAPITAL = 1_000_000
COMMISSION_PCT = 0.001  # 0.1%

# Data paths
DATA_PATH = "F:\\Master_Data\\market_data"
RESULTS_DIR = "results"

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# Function to calculate RSI
def calculate_rsi(prices, length=14):
    # Convert to Series if numpy array is passed
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
        
    # Calculate price changes
    delta = prices.diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss over the specified period
    avg_gain = gain.rolling(window=length).mean()
    avg_loss = loss.rolling(window=length).mean()
    
    # Calculate relative strength (RS)
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# Function to calculate ATR
def calculate_atr(high, low, close, length=14):
    # Convert to Series if numpy arrays are passed
    if isinstance(high, np.ndarray):
        high = pd.Series(high)
    if isinstance(low, np.ndarray):
        low = pd.Series(low)
    if isinstance(close, np.ndarray):
        close = pd.Series(close)
    
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate Average True Range
    atr = tr.rolling(window=length).mean()
    
    return atr

# Function to find local extrema
def find_local_extrema(series, order=3, mode='both'):
    if mode not in ['min', 'max', 'both']:
        raise ValueError("Mode must be 'min', 'max', or 'both'")
    
    # Initialize result series
    if mode == 'min' or mode == 'both':
        mins = pd.Series(0, index=series.index)
    if mode == 'max' or mode == 'both':
        maxs = pd.Series(0, index=series.index)
    
    # Find local minima
    if mode == 'min' or mode == 'both':
        for i in range(order, len(series) - order):
            if all(series.iloc[i] < series.iloc[i-j] for j in range(1, order+1)) and \
               all(series.iloc[i] < series.iloc[i+j] for j in range(1, order+1)):
                mins.iloc[i] = 1
    
    # Find local maxima
    if mode == 'max' or mode == 'both':
        for i in range(order, len(series) - order):
            if all(series.iloc[i] > series.iloc[i-j] for j in range(1, order+1)) and \
               all(series.iloc[i] > series.iloc[i+j] for j in range(1, order+1)):
                maxs.iloc[i] = 1
    
    if mode == 'min':
        return mins
    elif mode == 'max':
        return maxs
    else:  # mode == 'both'
        return mins, maxs

# Function to detect RSI divergences
def detect_rsi_divergence(df, swing_window=3):
    # Find local extrema for price
    df['price_local_min'], df['price_local_max'] = find_local_extrema(df['Close'], order=swing_window, mode='both')
    
    # Find local extrema for RSI
    df['rsi_local_min'], df['rsi_local_max'] = find_local_extrema(df['RSI'], order=swing_window, mode='both')
    
    # Initialize divergence columns
    df['bullish_div'] = 0
    df['bearish_div'] = 0
    
    # Detect bullish divergence (price makes lower low but RSI makes higher low)
    for i in range(swing_window * 2, len(df)):
        if df['price_local_min'].iloc[i] == 1:
            # Look back for another local minimum in price
            for j in range(i - swing_window, i - swing_window * 5, -1):
                if j < 0:
                    break
                if df['price_local_min'].iloc[j] == 1:
                    # Check if price made a lower low
                    if df['Close'].iloc[i] < df['Close'].iloc[j]:
                        # Check if RSI made a higher low
                        if df['RSI'].iloc[i] > df['RSI'].iloc[j]:
                            df['bullish_div'].iloc[i] = 1
                    break
    
    # Detect bearish divergence (price makes higher high but RSI makes lower high)
    for i in range(swing_window * 2, len(df)):
        if df['price_local_max'].iloc[i] == 1:
            # Look back for another local maximum in price
            for j in range(i - swing_window, i - swing_window * 5, -1):
                if j < 0:
                    break
                if df['price_local_max'].iloc[j] == 1:
                    # Check if price made a higher high
                    if df['Close'].iloc[i] > df['Close'].iloc[j]:
                        # Check if RSI made a lower high
                        if df['RSI'].iloc[i] < df['RSI'].iloc[j]:
                            df['bearish_div'].iloc[i] = 1
                    break
    
    return df

# RSI Divergence Strategy with Position Sizing
class RsiDivergenceStrategy(Strategy):
    # Strategy parameters
    rsi_length = RSI_LENGTH
    atr_length = ATR_LENGTH
    atr_multiplier = ATR_MULTIPLIER
    profit_target_pct = PROFIT_TARGET_PCT
    stop_loss_pct = STOP_LOSS_PCT
    swing_window = SWING_WINDOW
    
    # Position sizing parameters
    base_position_size = 0.01  # Base position size as percentage of equity
    volatility_factor = 1.0    # Scaling factor for volatility-based sizing
    max_position_size = 0.05   # Maximum position size as percentage of equity
    
    def init(self):
        # Calculate RSI
        self.rsi = self.I(calculate_rsi, self.data.Close, self.rsi_length)
        
        # Calculate ATR
        self.atr = self.I(calculate_atr, self.data.High, self.data.Low, self.data.Close, self.atr_length)
        
        # Calculate volatility as percentage of price
        self.volatility = self.I(lambda x, y: x / y, self.atr, self.data.Close)
        
        # Store divergence signals
        self.bullish_div = self.I(lambda x: x, np.zeros(len(self.data)))
        self.bearish_div = self.I(lambda x: x, np.zeros(len(self.data)))
        
        # Precompute divergences
        df = pd.DataFrame({
            'Close': self.data.Close,
            'RSI': self.rsi
        })
        
        # Detect divergences
        df = detect_rsi_divergence(df, self.swing_window)
        
        # Store divergence signals
        self.bullish_div = self.I(lambda x: x, df['bullish_div'].values)
        self.bearish_div = self.I(lambda x: x, df['bearish_div'].values)
        
        # For tracking trade reasons
        self.trade_reasons = {}
    
    def next(self):
        # Skip if not enough data
        if len(self.data) < self.rsi_length + self.swing_window * 5:
            return
        
        # Calculate position size based on volatility
        current_volatility = self.volatility[-1]
        position_size_pct = self.base_position_size / (current_volatility * 100 * self.volatility_factor)
        position_size_pct = min(position_size_pct, self.max_position_size)  # Cap at maximum position size
        
        # Check for bullish divergence (long entry)
        if self.bullish_div[-1] == 1 and not self.position:
            # Calculate position size
            size = position_size_pct * self.equity / self.data.Close[-1]
            
            # Enter long position
            order = self.buy(size=size)
            if order:
                self.trade_reasons[order] = 'bullish_div'
            
            # Set stop loss and take profit levels
            self.stop_loss_price = self.data.Close[-1] * (1 - self.stop_loss_pct)
            self.take_profit_price = self.data.Close[-1] * (1 + self.profit_target_pct)
        
        # Check for bearish divergence (short entry)
        elif self.bearish_div[-1] == 1 and not self.position:
            # Calculate position size
            size = position_size_pct * self.equity / self.data.Close[-1]
            
            # Enter short position
            order = self.sell(size=size)
            if order:
                self.trade_reasons[order] = 'bearish_div'
            
            # Set stop loss and take profit levels
            self.stop_loss_price = self.data.Close[-1] * (1 + self.stop_loss_pct)
            self.take_profit_price = self.data.Close[-1] * (1 - self.profit_target_pct)
        
        # Check for exit conditions if in a position
        if self.position:
            # For long positions
            if self.position.is_long:
                # Exit if stop loss or take profit is hit
                if self.data.Low[-1] <= self.stop_loss_price:
                    order = self.position.close()
                    if order:
                        self.trade_reasons[order] = 'stop_loss'
                elif self.data.High[-1] >= self.take_profit_price:
                    order = self.position.close()
                    if order:
                        self.trade_reasons[order] = 'take_profit'
            
            # For short positions
            elif self.position.is_short:
                # Exit if stop loss or take profit is hit
                if self.data.High[-1] >= self.stop_loss_price:
                    order = self.position.close()
                    if order:
                        self.trade_reasons[order] = 'stop_loss'
                elif self.data.Low[-1] <= self.take_profit_price:
                    order = self.position.close()
                    if order:
                        self.trade_reasons[order] = 'take_profit'

# Function to load data for a specific cryptocurrency
def load_crypto_data(crypto, timeframe="5m"):
    # Define possible path patterns to try
    base_dir = os.path.join(DATA_PATH, crypto.lower(), timeframe)
    
    # Check if directory exists
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return None
    
    # Try different file naming patterns
    patterns = [
        f"{crypto.lower()}_{timeframe}_*.csv",  # Standard pattern
        f"{crypto.lower()}_*.csv",             # Without timeframe
        "*.csv"                                 # Any CSV file
    ]
    
    files = []
    for pattern in patterns:
        path_pattern = os.path.join(base_dir, pattern)
        found_files = glob.glob(path_pattern)
        if found_files:
            files.extend(found_files)
            break
    
    if not files:
        print(f"No data files found for {crypto} with timeframe {timeframe}")
        return None
    
    # Sort files by name (assuming naming convention includes dates)
    files.sort()
    
    # Load the most recent file
    latest_file = files[-1]
    print(f"Loading data from {latest_file}")
    
    try:
        df = pd.read_csv(latest_file)
        
        # Check column names (case-insensitive)
        columns_lower = [col.lower() for col in df.columns]
        
        # Ensure required columns exist
        required_columns = ['timestamp', 'open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in columns_lower and col not in df.columns]
        
        if missing_columns:
            # Try alternative column names
            alt_mappings = {
                'timestamp': ['time', 'date', 'datetime'],
                'open': ['o'],
                'high': ['h'],
                'low': ['l'],
                'close': ['c'],
                'volume': ['vol', 'v']
            }
            
            for missing_col in missing_columns:
                for alt in alt_mappings.get(missing_col, []):
                    if alt in columns_lower or alt in df.columns:
                        missing_columns.remove(missing_col)
                        break
        
        if missing_columns:
            print(f"Missing required columns in {latest_file}: {missing_columns}")
            print(f"Available columns: {df.columns.tolist()}")
            return None
        
        # Standardize column names
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower == 'timestamp' or col_lower == 'time' or col_lower == 'date' or col_lower == 'datetime':
                column_mapping[col] = 'Date'
            elif col_lower == 'open' or col_lower == 'o':
                column_mapping[col] = 'Open'
            elif col_lower == 'high' or col_lower == 'h':
                column_mapping[col] = 'High'
            elif col_lower == 'low' or col_lower == 'l':
                column_mapping[col] = 'Low'
            elif col_lower == 'close' or col_lower == 'c':
                column_mapping[col] = 'Close'
            elif col_lower == 'volume' or col_lower == 'vol' or col_lower == 'v':
                column_mapping[col] = 'Volume'
        
        df = df.rename(columns=column_mapping)
        
        # If Volume is missing, add a dummy column
        if 'Volume' not in df.columns:
            df['Volume'] = 0
        
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            try:
                # Try milliseconds first
                df['Date'] = pd.to_datetime(df['Date'], unit='ms')
            except:
                try:
                    # Try seconds
                    df['Date'] = pd.to_datetime(df['Date'], unit='s')
                except:
                    try:
                        # Try standard datetime format
                        df['Date'] = pd.to_datetime(df['Date'])
                    except Exception as e:
                        print(f"Failed to convert timestamp: {e}")
                        return None
        
        # Set Date as index
        df = df.set_index('Date')
        
        # Sort by date
        df = df.sort_index()
        
        # Ensure we have enough data
        if len(df) < 100:
            print(f"Insufficient data points for {crypto}: {len(df)} rows")
            return None
        
        return df
    
    except Exception as e:
        print(f"Error loading data for {crypto}: {e}")
        return None

# Function to run backtest for a specific cryptocurrency
def run_backtest(crypto, timeframe="5m"):
    # Load data
    df = load_crypto_data(crypto, timeframe)
    
    if df is None or len(df) < 100:
        print(f"Insufficient data for {crypto}")
        return None
    
    # Run backtest
    bt = Backtest(df, RsiDivergenceStrategy, cash=INITIAL_CAPITAL, commission=COMMISSION_PCT)
    stats = bt.run()
    
    # Print results
    print(f"\nResults for {crypto}:")
    print(f"Return: {stats['Return']:.2%}")
    print(f"Win Rate: {stats['Win Rate']:.2%}")
    print(f"Profit Factor: {stats['Profit Factor']:.2f}")
    print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
    print(f"Max. Drawdown: {stats['Max. Drawdown']:.2%}")
    print(f"# Trades: {stats['# Trades']}")
    
    # Save detailed results to CSV
    result_file = os.path.join(RESULTS_DIR, f"{crypto}_results.csv")
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(result_file)
    
    # Plot equity curve and save to file
    fig, ax = plt.subplots(figsize=(10, 6))
    stats['_equity_curve']['Equity'].plot(ax=ax)
    ax.set_title(f"{crypto} Equity Curve")
    ax.set_ylabel("Equity")
    ax.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, f"{crypto}_equity_curve.png"))
    plt.close()
    
    # Return stats for summary
    return {
        'Symbol': crypto,
        'Return': stats['Return'],
        'Win Rate': stats['Win Rate'],
        'Profit Factor': stats['Profit Factor'],
        'Sharpe Ratio': stats['Sharpe Ratio'],
        'Max. Drawdown': stats['Max. Drawdown'],
        '# Trades': stats['# Trades']
    }

# Function to run backtests for multiple cryptocurrencies
def run_multi_crypto_backtest(cryptos, timeframe="5m"):
    results = []
    
    for crypto in cryptos:
        print(f"\nBacktesting {crypto}...")
        result = run_backtest(crypto, timeframe)
        if result:
            results.append(result)
    
    # Create summary DataFrame
    if results:
        summary_df = pd.DataFrame(results)
        summary_df = summary_df.sort_values('Return', ascending=False)
        
        # Save summary to CSV
        summary_file = os.path.join(RESULTS_DIR, "summary_results.csv")
        summary_df.to_csv(summary_file, index=False)
        
        # Print summary
        print("\nSummary Results (sorted by Return):")
        print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.2%}" if isinstance(x, float) and x < 10 else f"{x:.2f}"))
        
        return summary_df
    
    return None

# Function to run parameter optimization for a specific cryptocurrency
def optimize_parameters(crypto, timeframe="5m"):
    # Load data
    df = load_crypto_data(crypto, timeframe)
    
    if df is None or len(df) < 100:
        print(f"Insufficient data for {crypto}")
        return None
    
    # Define parameter ranges to test
    param_ranges = {
        'rsi_length': range(4, 15, 2),            # 4, 6, 8, 10, 12, 14
        'atr_length': range(4, 15, 2),            # 4, 6, 8, 10, 12, 14
        'atr_multiplier': np.arange(0.8, 1.6, 0.2),  # 0.8, 1.0, 1.2, 1.4
        'profit_target_pct': np.arange(0.005, 0.021, 0.005),  # 0.5%, 1.0%, 1.5%, 2.0%
        'stop_loss_pct': np.arange(0.005, 0.016, 0.005),      # 0.5%, 1.0%, 1.5%
        'swing_window': range(2, 6)               # 2, 3, 4, 5
    }
    
    # Run optimization
    bt = Backtest(df, RsiDivergenceStrategy, cash=INITIAL_CAPITAL, commission=COMMISSION_PCT)
    stats, heatmap = bt.optimize(
        rsi_length=param_ranges['rsi_length'],
        atr_length=param_ranges['atr_length'],
        atr_multiplier=param_ranges['atr_multiplier'],
        profit_target_pct=param_ranges['profit_target_pct'],
        stop_loss_pct=param_ranges['stop_loss_pct'],
        swing_window=param_ranges['swing_window'],
        maximize='Return',
        return_heatmap=True
    )
    
    # Print optimization results
    print(f"\nOptimized Parameters for {crypto}:")
    print(f"RSI Length: {stats['rsi_length']}")
    print(f"ATR Length: {stats['atr_length']}")
    print(f"ATR Multiplier: {stats['atr_multiplier']:.1f}")
    print(f"Profit Target: {stats['profit_target_pct']:.2%}")
    print(f"Stop Loss: {stats['stop_loss_pct']:.2%}")
    print(f"Swing Window: {stats['swing_window']}")
    print(f"\nOptimized Performance:")
    print(f"Return: {stats['Return']:.2%}")
    print(f"Win Rate: {stats['Win Rate']:.2%}")
    print(f"Profit Factor: {stats['Profit Factor']:.2f}")
    print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
    print(f"Max. Drawdown: {stats['Max. Drawdown']:.2%}")
    print(f"# Trades: {stats['# Trades']}")
    
    # Save optimization results
    result_file = os.path.join(RESULTS_DIR, f"{crypto}_optimized_params.csv")
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(result_file)
    
    # Return optimized parameters
    return {
        'Symbol': crypto,
        'rsi_length': stats['rsi_length'],
        'atr_length': stats['atr_length'],
        'atr_multiplier': stats['atr_multiplier'],
        'profit_target_pct': stats['profit_target_pct'],
        'stop_loss_pct': stats['stop_loss_pct'],
        'swing_window': stats['swing_window'],
        'Return': stats['Return'],
        'Win Rate': stats['Win Rate'],
        'Profit Factor': stats['Profit Factor'],
        'Sharpe Ratio': stats['Sharpe Ratio'],
        'Max. Drawdown': stats['Max. Drawdown'],
        '# Trades': stats['# Trades']
    }

# Function to run parameter optimization for multiple cryptocurrencies
def optimize_multi_crypto(cryptos, timeframe="5m"):
    results = []
    
    for crypto in cryptos:
        print(f"\nOptimizing parameters for {crypto}...")
        result = optimize_parameters(crypto, timeframe)
        if result:
            results.append(result)
    
    # Create summary DataFrame
    if results:
        summary_df = pd.DataFrame(results)
        summary_df = summary_df.sort_values('Return', ascending=False)
        
        # Save summary to CSV
        summary_file = os.path.join(RESULTS_DIR, "optimized_params_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        
        # Print summary
        print("\nOptimization Summary (sorted by Return):")
        print(summary_df[["Symbol", "Return", "Win Rate", "Profit Factor", "# Trades", 
                          "rsi_length", "atr_length", "atr_multiplier", 
                          "profit_target_pct", "stop_loss_pct", "swing_window"]].to_string(
                              index=False, 
                              float_format=lambda x: f"{x:.2%}" if isinstance(x, float) and x < 10 else f"{x:.2f}"
                          ))
        
        return summary_df
    
    return None

# Main execution
if __name__ == "__main__":
    # List of cryptocurrencies to test
    cryptos = [
        "BTC", "ETH", "SOL", "XRP", "HYPE", "SUI", "TRUMP", "FARTCOIN", 
        "kPEPE", "kBONK", "WIF", "DOGE", "PAXG", "AI16Z", "PNUT"
    ]
    
    # Run extended backtests with position sizing
    print("Running extended backtests with position sizing...")
    summary = run_multi_crypto_backtest(cryptos)
    
    # Ask if user wants to run optimization
    run_opt = input("\nDo you want to run parameter optimization for each cryptocurrency? (y/n): ")
    if run_opt.lower() == 'y':
        # Select top performers for optimization
        if summary is not None:
            top_performers = summary.head(5)['Symbol'].tolist()
            print(f"\nRunning optimization for top 5 performers: {', '.join(top_performers)}")
            optimize_multi_crypto(top_performers)
        else:
            print("No valid backtest results to select top performers.")
    
    print("\nBacktesting complete. Results saved to the 'results' directory.")
