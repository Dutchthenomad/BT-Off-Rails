import os
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import matplotlib.pyplot as plt
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
    price_local_min, price_local_max = find_local_extrema(df['Close'], order=swing_window, mode='both')
    df['price_local_min'] = price_local_min
    df['price_local_max'] = price_local_max
    
    # Find local extrema for RSI
    rsi_local_min, rsi_local_max = find_local_extrema(df['RSI'], order=swing_window, mode='both')
    df['rsi_local_min'] = rsi_local_min
    df['rsi_local_max'] = rsi_local_max
    
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
        
        # Pre-calculate divergences
        df = pd.DataFrame({
            'Close': self.data.Close,
            'RSI': self.rsi
        })
        
        df = detect_rsi_divergence(df, self.swing_window)
        
        # Store signals
        self.bullish_div = self.I(lambda x: x, df['bullish_div'].values)
        self.bearish_div = self.I(lambda x: x, df['bearish_div'].values)
    
    def next(self):
        # Skip if not enough data
        if len(self.data) < self.rsi_length + self.swing_window * 5:
            return
        
        # Calculate position size based on volatility
        current_volatility = self.volatility[-1]
        
        # Avoid division by zero or very small values
        if current_volatility < 0.001:
            current_volatility = 0.001
            
        # Calculate position size inversely proportional to volatility
        position_size_pct = self.base_position_size / (current_volatility * 100 * self.volatility_factor)
        
        # Cap at maximum position size
        position_size_pct = min(position_size_pct, self.max_position_size)
        
        # Calculate actual size in shares/contracts
        size = position_size_pct * self.equity / self.data.Close[-1]
        
        # Entry signals
        if not self.position:  # If not in a position
            # Bullish divergence - go long
            if self.bullish_div[-1] == 1:
                self.buy(size=size)
                # Set stop loss and take profit prices
                self.stop_loss_price = self.data.Close[-1] * (1 - self.stop_loss_pct)
                self.take_profit_price = self.data.Close[-1] * (1 + self.profit_target_pct)
            
            # Bearish divergence - go short
            elif self.bearish_div[-1] == 1:
                self.sell(size=size)
                # Set stop loss and take profit prices
                self.stop_loss_price = self.data.Close[-1] * (1 + self.stop_loss_pct)
                self.take_profit_price = self.data.Close[-1] * (1 - self.profit_target_pct)
        
        # Exit signals
        elif self.position.is_long:  # If in a long position
            # Exit if stop loss or take profit is hit
            if self.data.Low[-1] <= self.stop_loss_price or self.data.High[-1] >= self.take_profit_price:
                self.position.close()
        
        elif self.position.is_short:  # If in a short position
            # Exit if stop loss or take profit is hit
            if self.data.High[-1] >= self.stop_loss_price or self.data.Low[-1] <= self.take_profit_price:
                self.position.close()

# Function to load data for a specific cryptocurrency
def load_crypto_data(crypto, timeframe="5m"):
    # Try different directory paths
    possible_paths = [
        os.path.join(DATA_PATH, crypto.lower(), timeframe),
        os.path.join(DATA_PATH, crypto.lower()),
        os.path.join(DATA_PATH, crypto.upper(), timeframe),
        os.path.join(DATA_PATH, crypto.upper())
    ]
    
    for base_dir in possible_paths:
        if not os.path.exists(base_dir):
            continue
        
        # Try different file patterns
        patterns = [
            f"{crypto.lower()}_{timeframe}_*.csv",
            f"{crypto.lower()}_*.csv",
            f"{crypto.upper()}_{timeframe}_*.csv",
            f"{crypto.upper()}_*.csv",
            "*.csv"
        ]
        
        for pattern in patterns:
            files = glob.glob(os.path.join(base_dir, pattern))
            if files:
                # Sort files by name
                files.sort()
                latest_file = files[-1]
                print(f"Loading data from {latest_file}")
                
                try:
                    # Load the CSV file
                    df = pd.read_csv(latest_file)
                    
                    # Check column names (case-insensitive)
                    columns_lower = [col.lower() for col in df.columns]
                    
                    # Map column names
                    column_mapping = {}
                    for col in df.columns:
                        col_lower = col.lower()
                        if col_lower in ['timestamp', 'time', 'date', 'datetime']:
                            column_mapping[col] = 'Date'
                        elif col_lower in ['open', 'o']:
                            column_mapping[col] = 'Open'
                        elif col_lower in ['high', 'h']:
                            column_mapping[col] = 'High'
                        elif col_lower in ['low', 'l']:
                            column_mapping[col] = 'Low'
                        elif col_lower in ['close', 'c']:
                            column_mapping[col] = 'Close'
                        elif col_lower in ['volume', 'vol', 'v']:
                            column_mapping[col] = 'Volume'
                    
                    # Rename columns
                    df = df.rename(columns=column_mapping)
                    
                    # Check if required columns exist
                    required_columns = ['Date', 'Open', 'High', 'Low', 'Close']
                    if not all(col in df.columns for col in required_columns):
                        print(f"Missing required columns in {latest_file}. Available columns: {df.columns.tolist()}")
                        continue
                    
                    # Add Volume column if missing
                    if 'Volume' not in df.columns:
                        df['Volume'] = 0
                    
                    # Convert timestamp to datetime
                    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                        try:
                            # Try milliseconds first
                            df['Date'] = pd.to_datetime(df['Date'], unit='ms')
                        except Exception:
                            try:
                                # Try seconds
                                df['Date'] = pd.to_datetime(df['Date'], unit='s')
                            except Exception:
                                try:
                                    # Try standard format
                                    df['Date'] = pd.to_datetime(df['Date'])
                                except Exception as e:
                                    print(f"Failed to convert timestamp: {e}")
                                    continue
                    
                    # Set Date as index
                    df = df.set_index('Date')
                    
                    # Sort by date
                    df = df.sort_index()
                    
                    # Ensure we have enough data
                    if len(df) < 100:
                        print(f"Insufficient data points for {crypto}: {len(df)} rows")
                        continue
                    
                    return df
                
                except Exception as e:
                    print(f"Error loading file {latest_file}: {e}")
    
    print(f"No valid data found for {crypto} with timeframe {timeframe}")
    return None

# Function to run backtest for a specific cryptocurrency
def run_backtest(crypto, timeframe="5m"):
    # Load data
    df = load_crypto_data(crypto, timeframe)
    
    if df is None:
        print(f"Skipping backtest for {crypto} due to data loading issues")
        return None
    
    # Print data info
    print(f"Data loaded for {crypto}: {len(df)} rows from {df.index[0]} to {df.index[-1]}")
    
    # Run backtest
    bt = Backtest(df, RsiDivergenceStrategy, cash=INITIAL_CAPITAL, commission=COMMISSION_PCT)
    stats = bt.run()
    
    # Print results
    print(f"\nResults for {crypto}:")
    print(f"Return: {stats['Return']:.2%}")
    print(f"Win Rate: {stats['Win Rate']:.2%}")
    print(f"Profit Factor: {stats.get('Profit Factor', 'N/A')}")
    print(f"Sharpe Ratio: {stats.get('Sharpe Ratio', 'N/A')}")
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
        'Profit Factor': stats.get('Profit Factor', float('nan')),
        'Sharpe Ratio': stats.get('Sharpe Ratio', float('nan')),
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
        print(summary_df.to_string(index=False))
        
        return summary_df
    
    return None

# Main execution
if __name__ == "__main__":
    # List of cryptocurrencies to test
    cryptos = [
        "BTC", "ETH", "SOL", "XRP"
    ]
    
    # Run backtests with position sizing
    print("Running backtests with position sizing...")
    summary = run_multi_crypto_backtest(cryptos)
    
    print("\nBacktesting complete. Results saved to the 'results' directory.")
