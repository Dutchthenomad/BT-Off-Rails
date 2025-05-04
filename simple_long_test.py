import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import matplotlib.pyplot as plt
import os
import glob

# --- Configuration ---
INITIAL_CASH = 1000000.0
COMMISSION_RATE = 0.001  # 0.1%
RESULTS_DIR = "results"

# --- Strategy Parameters ---
RSI_LENGTH = 6
ATR_LENGTH = 6
ATR_MULTIPLIER = 1.1
PROFIT_TARGET_PCT = 0.01
STOP_LOSS_PCT = 0.008
SWING_WINDOW = 3

# --- Position Sizing Parameters ---
BASE_POSITION_SIZE = 0.01  # Base position size as percentage of equity
VOLATILITY_FACTOR = 1.0    # Scaling factor for volatility-based sizing
MAX_POSITION_SIZE = 0.05   # Maximum position size as percentage of equity

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- RSI Calculation ---
def calculate_rsi(prices, length=14):
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

# --- ATR Calculation ---
def calculate_atr(df, length=14):
    # Calculate True Range
    tr1 = df['High'] - df['Low']
    tr2 = abs(df['High'] - df['Close'].shift(1))
    tr3 = abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate Average True Range
    atr = tr.rolling(window=length).mean()
    
    return atr

# --- Find Local Extrema ---
def find_local_extrema(series, order=3, mode='min'):
    result = pd.Series(0, index=series.index)
    
    # Find local minima
    if mode == 'min':
        for i in range(order, len(series) - order):
            if all(series.iloc[i] < series.iloc[i-j] for j in range(1, order+1)) and \
               all(series.iloc[i] < series.iloc[i+j] for j in range(1, order+1)):
                result.iloc[i] = 1
    
    # Find local maxima
    elif mode == 'max':
        for i in range(order, len(series) - order):
            if all(series.iloc[i] > series.iloc[i-j] for j in range(1, order+1)) and \
               all(series.iloc[i] > series.iloc[i+j] for j in range(1, order+1)):
                result.iloc[i] = 1
    
    return result

# --- Prepare Data ---
def prepare_data(df):
    # Calculate RSI
    df['RSI'] = calculate_rsi(df['Close'], length=RSI_LENGTH)
    
    # Calculate ATR
    df['ATR'] = calculate_atr(df, length=ATR_LENGTH)
    
    # Calculate volatility as percentage of price
    df['Volatility'] = df['ATR'] / df['Close']
    
    # Find local extrema for price
    df['price_local_min'] = find_local_extrema(df['Close'], order=SWING_WINDOW, mode='min')
    df['price_local_max'] = find_local_extrema(df['Close'], order=SWING_WINDOW, mode='max')
    
    # Find local extrema for RSI
    df['rsi_local_min'] = find_local_extrema(df['RSI'], order=SWING_WINDOW, mode='min')
    df['rsi_local_max'] = find_local_extrema(df['RSI'], order=SWING_WINDOW, mode='max')
    
    # Initialize divergence columns
    df['bullish_div'] = 0
    
    # Detect bullish divergence (price makes lower low but RSI makes higher low)
    for i in range(SWING_WINDOW * 2, len(df)):
        if df['price_local_min'].iloc[i] == 1:
            # Look back for another local minimum in price
            for j in range(i - SWING_WINDOW, i - SWING_WINDOW * 5, -1):
                if j < 0:
                    break
                if df['price_local_min'].iloc[j] == 1:
                    # Check if price made a lower low
                    if df['Close'].iloc[i] < df['Close'].iloc[j]:
                        # Check if RSI made a higher low
                        if df['RSI'].iloc[i] > df['RSI'].iloc[j]:
                            df.loc[df.index[i], 'bullish_div'] = 1
                    break
    
    return df

# --- RSI Divergence Strategy with Position Sizing ---
class LongOnlyRSIDivergenceStrategy(Strategy):
    def init(self):
        # Store indicators
        self.rsi = self.I(lambda: self.data.RSI)
        self.atr = self.I(lambda: self.data.ATR)
        self.volatility = self.I(lambda: self.data.Volatility)
        self.bullish_div = self.I(lambda: self.data.bullish_div)
    
    def next(self):
        # Skip if not enough data
        if len(self.data) < SWING_WINDOW * 5:
            return
        
        # Entry signals
        if not self.position and self.bullish_div[-1] == 1:
            # Calculate position size based on volatility
            current_volatility = self.volatility[-1]
            
            # Avoid division by zero or very small values
            if current_volatility < 0.001:
                current_volatility = 0.001
                
            # Calculate position size inversely proportional to volatility
            position_size_pct = BASE_POSITION_SIZE / (current_volatility * 100 * VOLATILITY_FACTOR)
            
            # Cap at maximum position size
            position_size_pct = min(position_size_pct, MAX_POSITION_SIZE)
            
            # Calculate actual size in shares/contracts
            size = position_size_pct * self.equity / self.data.Close[-1]
            
            # Enter long position
            self.buy(size=size)
            
            # Set stop loss and take profit prices
            self.stop_loss_price = self.data.Close[-1] * (1 - STOP_LOSS_PCT)
            self.take_profit_price = self.data.Close[-1] * (1 + PROFIT_TARGET_PCT)
        
        # Exit signals
        elif self.position:
            # Exit if stop loss or take profit is hit
            if self.data.Low[-1] <= self.stop_loss_price:
                self.position.close()
            elif self.data.High[-1] >= self.take_profit_price:
                self.position.close()

# --- Load Data ---
def load_data(file_path):
    try:
        # Load CSV file
        df = pd.read_csv(file_path)
        
        # Standardize column names
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
        
        df = df.rename(columns=column_mapping)
        
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
                    # Try standard format
                    df['Date'] = pd.to_datetime(df['Date'])
        
        # Set Date as index
        df = df.set_index('Date')
        
        # Sort by date
        df = df.sort_index()
        
        return df
    
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

# --- Find Data File ---
def find_data_file(symbol, base_path="F:\\Master_Data\\market_data"):
    # Try different directory paths
    possible_paths = [
        os.path.join(base_path, symbol.lower(), "5m"),
        os.path.join(base_path, symbol.lower()),
        os.path.join(base_path, symbol.upper(), "5m"),
        os.path.join(base_path, symbol.upper())
    ]
    
    for path in possible_paths:
        if not os.path.exists(path):
            continue
        
        # Try different file patterns
        patterns = [
            f"{symbol.lower()}_5m_*.csv",
            f"{symbol.lower()}_*.csv",
            f"{symbol.upper()}_5m_*.csv",
            f"{symbol.upper()}_*.csv",
            "*.csv"
        ]
        
        for pattern in patterns:
            files = glob.glob(os.path.join(path, pattern))
            if files:
                # Sort files by name
                files.sort()
                return files[-1]  # Use the most recent file
    
    return None

# --- Run Backtest ---
def run_backtest(symbol):
    # Find data file
    data_file = find_data_file(symbol)
    if not data_file:
        print(f"No data file found for {symbol}")
        return None
    
    print(f"Found data file for {symbol}: {data_file}")
    
    # Load data
    df = load_data(data_file)
    if df is None:
        print(f"Failed to load data for {symbol}")
        return None
    
    # Check if we have enough data
    if len(df) < 100:
        print(f"Insufficient data for {symbol}: {len(df)} rows")
        return None
    
    print(f"Loaded {len(df)} rows of data for {symbol} from {df.index[0]} to {df.index[-1]}")
    
    # Prepare data
    df = prepare_data(df)
    
    # Run backtest
    bt = Backtest(df, LongOnlyRSIDivergenceStrategy, cash=INITIAL_CASH, commission=COMMISSION_RATE)
    stats = bt.run()
    
    # Print results
    print(f"\nResults for {symbol}:")
    print(f"Return: {stats['Return']:.2%}")
    print(f"Win Rate: {stats['Win Rate']:.2%}")
    print(f"Profit Factor: {stats.get('Profit Factor', 'N/A')}")
    print(f"Sharpe Ratio: {stats.get('Sharpe Ratio', 'N/A')}")
    print(f"Max. Drawdown: {stats['Max. Drawdown']:.2%}")
    print(f"# Trades: {stats['# Trades']}")
    
    # Save equity curve
    plt.figure(figsize=(10, 6))
    stats['_equity_curve']['Equity'].plot()
    plt.title(f"{symbol} - Equity Curve")
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, f"{symbol}_equity_curve.png"))
    plt.close()
    
    # Save trades
    trades_df = stats['_trades']
    trades_df.to_csv(os.path.join(RESULTS_DIR, f"{symbol}_trades.csv"))
    
    # Analyze position sizing
    if len(trades_df) > 0:
        # Calculate position size as percentage of equity
        trades_df['Position Size'] = trades_df['Size'] * trades_df['EntryPrice'] / INITIAL_CASH
        
        # Group trades by position size buckets
        trades_df['Position Size Bucket'] = pd.cut(trades_df['Position Size'], 
                                                 bins=[0, 0.01, 0.02, 0.03, 0.04, 0.05, 1.0],
                                                 labels=['0-1%', '1-2%', '2-3%', '3-4%', '4-5%', '>5%'])
        
        # Analyze performance by position size
        size_analysis = trades_df.groupby('Position Size Bucket').agg({
            'ReturnPct': ['mean', 'count'],
            'ReturnPct': lambda x: (x > 0).mean()  # Win rate
        })
        
        print("\nPosition Size Analysis:")
        print(size_analysis)
        
        # Plot position size vs. return
        plt.figure(figsize=(10, 6))
        plt.scatter(trades_df['Position Size'] * 100, trades_df['ReturnPct'] * 100)
        plt.title(f"{symbol} - Position Size vs. Return")
        plt.xlabel("Position Size (% of Equity)")
        plt.ylabel("Return (%)")
        plt.grid(True)
        plt.savefig(os.path.join(RESULTS_DIR, f"{symbol}_position_size_vs_return.png"))
        plt.close()
    
    return stats

# --- Main Function ---
if __name__ == "__main__":
    # Test with ETH (Ethereum)
    print("Running backtest for ETH...")
    run_backtest("ETH")
    
    # Test with SOL (Solana)
    print("\nRunning backtest for SOL...")
    run_backtest("SOL")
    
    print("\nBacktesting complete. Results saved to the 'results' directory.")
