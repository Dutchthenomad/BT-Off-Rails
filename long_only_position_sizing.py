import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import logging
import os
import glob
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# --- Configuration ---
INITIAL_CASH = 1000000.0  # Initial cash for each cryptocurrency
COMMISSION_RATE = 0.001  # 0.1% commission rate
RESULTS_DIR = "results"  # Directory to store results

# --- Strategy Parameters ---
RSI_LENGTH = 6  # Optimized value
ATR_LENGTH = 6  # Optimized value
ATR_MULTIPLIER = 1.1  # Optimized value
PROFIT_TARGET_PCT = 0.01  # Optimized value
STOP_LOSS_PCT = 0.008  # Optimized value
SWING_WINDOW = 3  # Optimized value

# --- Position Sizing Parameters ---
BASE_POSITION_SIZE = 0.01  # Base position size as percentage of equity
VOLATILITY_FACTOR = 1.0    # Scaling factor for volatility-based sizing
MAX_POSITION_SIZE = 0.05   # Maximum position size as percentage of equity

# --- Cryptocurrencies to Test ---
CRYPTOS = [
    {"name": "Bitcoin", "symbol": "BTC", "emoji": "🪙"},
    {"name": "Ethereum", "symbol": "ETH", "emoji": "💠"},
    {"name": "Solana", "symbol": "SOL", "emoji": "😎"},
    {"name": "Ripple", "symbol": "XRP", "emoji": "🌊"},
    {"name": "Hyperliquid", "symbol": "HYPE", "emoji": "🔥"},
    {"name": "Sui", "symbol": "SUI", "emoji": "💧"},
    {"name": "Trump", "symbol": "TRUMP", "emoji": "🇺🇸"},
    {"name": "Fartcoin", "symbol": "FARTCOIN", "emoji": "💨"},
    {"name": "Kabosu PEPE", "symbol": "kPEPE", "emoji": "🐸"},
    {"name": "Kabosu BONK", "symbol": "kBONK", "emoji": "🔨"},
    {"name": "Dogwifhat", "symbol": "WIF", "emoji": "🐶"},
    {"name": "Dogecoin", "symbol": "DOGE", "emoji": "🐕"},
    {"name": "Pax Gold", "symbol": "PAXG", "emoji": "🥇"},
    {"name": "AI16Z", "symbol": "AI16Z", "emoji": "🤖"},
    {"name": "Peanut", "symbol": "PNUT", "emoji": "🥜"}
]

# --- RSI Calculation ---
def calculate_rsi(prices, length=14):
    """Calculate RSI indicator"""
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

# --- ATR Calculation ---
def calculate_atr(high, low, close, length=14):
    """Calculate Average True Range"""
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

# --- Find Local Extrema ---
def find_local_extrema(series, order=5, mode='max'):
    """Find local maxima or minima in a series"""
    # Import here to avoid import error if not used
    from scipy.signal import argrelextrema
    
    if mode == 'max':
        indices = argrelextrema(series.values, np.greater, order=order)[0]
        result = pd.Series(0, index=series.index)
        if len(indices) > 0:
            result.iloc[indices] = 1
        return result
    elif mode == 'min':
        indices = argrelextrema(series.values, np.less, order=order)[0]
        result = pd.Series(0, index=series.index)
        if len(indices) > 0:
            result.iloc[indices] = 1
        return result
    else:
        raise ValueError("Mode must be 'max' or 'min'")

# --- Load and Prepare Data ---
def load_data(file_path):
    """Load data from CSV file"""
    try:
        # Load CSV file
        df = pd.read_csv(file_path)
        
        # Check if required columns exist
        required_columns = ['timestamp', 'open', 'high', 'low', 'close']
        missing_columns = []
        for col in required_columns:
            if col not in df.columns and col.capitalize() not in df.columns:
                # Check for alternative column names
                alt_columns = {
                    'timestamp': ['time', 'date', 'datetime'],
                    'open': ['o'],
                    'high': ['h'],
                    'low': ['l'],
                    'close': ['c']
                }
                
                found = False
                for alt in alt_columns.get(col, []):
                    if alt in df.columns or alt.capitalize() in df.columns:
                        found = True
                        break
                
                if not found:
                    missing_columns.append(col)
        
        if missing_columns:
            logging.error(f"Missing required columns: {missing_columns}")
            return None
        
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
                    try:
                        # Try standard format
                        df['Date'] = pd.to_datetime(df['Date'])
                    except Exception as e:
                        logging.error(f"Failed to convert timestamp: {e}")
                        return None
        
        # Set Date as index
        df = df.set_index('Date')
        
        # Sort by date
        df = df.sort_index()
        
        return df
    
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        return None

# --- Prepare Data for Backtesting ---
def prepare_data(df):
    """Calculate indicators and detect divergences"""
    # Calculate RSI
    df['RSI'] = calculate_rsi(df['Close'], length=RSI_LENGTH)
    
    # Calculate ATR
    df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'], length=ATR_LENGTH)
    
    # Calculate volatility as percentage of price
    df['Volatility'] = df['ATR'] / df['Close']
    
    # Detect divergences
    df = detect_divergences(df)
    
    return df

# --- Detect Divergences ---
def detect_divergences(df):
    """Detect bullish and bearish divergences"""
    # Find local extrema for price
    df['price_local_min'] = find_local_extrema(df['Close'], order=SWING_WINDOW, mode='min')
    df['price_local_max'] = find_local_extrema(df['Close'], order=SWING_WINDOW, mode='max')
    
    # Find local extrema for RSI
    df['rsi_local_min'] = find_local_extrema(df['RSI'], order=SWING_WINDOW, mode='min')
    df['rsi_local_max'] = find_local_extrema(df['RSI'], order=SWING_WINDOW, mode='max')
    
    # Initialize divergence columns
    df['bullish_div'] = 0
    df['bearish_div'] = 0
    
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
    
    # Detect bearish divergence (price makes higher high but RSI makes lower high)
    for i in range(SWING_WINDOW * 2, len(df)):
        if df['price_local_max'].iloc[i] == 1:
            # Look back for another local maximum in price
            for j in range(i - SWING_WINDOW, i - SWING_WINDOW * 5, -1):
                if j < 0:
                    break
                if df['price_local_max'].iloc[j] == 1:
                    # Check if price made a higher high
                    if df['Close'].iloc[i] > df['Close'].iloc[j]:
                        # Check if RSI made a lower high
                        if df['RSI'].iloc[i] < df['RSI'].iloc[j]:
                            df.loc[df.index[i], 'bearish_div'] = 1
                    break
    
    return df

# --- RSI Divergence Strategy with Position Sizing (Long Only) ---
class LongOnlyRSIDivergenceStrategy(Strategy):
    """RSI Divergence Trading Strategy with Volatility-Based Position Sizing (Long Only)"""
    # Strategy parameters
    rsi_length = RSI_LENGTH
    atr_length = ATR_LENGTH
    atr_multiplier = ATR_MULTIPLIER
    profit_target_pct = PROFIT_TARGET_PCT
    stop_loss_pct = STOP_LOSS_PCT
    swing_window = SWING_WINDOW
    
    # Position sizing parameters
    base_position_size = BASE_POSITION_SIZE
    volatility_factor = VOLATILITY_FACTOR
    max_position_size = MAX_POSITION_SIZE
    
    def init(self):
        """Initialize strategy"""
        # Store indicators
        self.rsi = self.I(lambda: self.data.RSI)
        self.atr = self.I(lambda: self.data.ATR)
        self.volatility = self.I(lambda: self.data.Volatility)
        
        # Store divergence signals
        self.bullish_div = self.I(lambda: self.data.bullish_div)
    
    def next(self):
        """Execute strategy for each candle"""
        # Skip if not enough data
        if len(self.data) < self.swing_window * 5:
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
                # Enter long position with calculated size
                self.buy(size=size)
                
                # Set stop loss and take profit prices
                self.stop_loss_price = self.data.Close[-1] * (1 - self.stop_loss_pct)
                self.take_profit_price = self.data.Close[-1] * (1 + self.profit_target_pct)
        
        # Exit signals
        elif self.position:  # If in a position
            # Exit if stop loss or take profit is hit
            if self.data.Low[-1] <= self.stop_loss_price:
                self.position.close()
            elif self.data.High[-1] >= self.take_profit_price:
                self.position.close()

# --- Find Data Files ---
def find_data_files():
    """Find data files for each cryptocurrency"""
    data_files = {}
    base_path = "F:\\Master_Data\\market_data"
    timeframe = "5m"
    
    for crypto in CRYPTOS:
        symbol = crypto["symbol"]
        logging.info(f"Looking for data files for {symbol}...")
        
        # Try different directory paths
        possible_paths = [
            os.path.join(base_path, symbol.lower(), timeframe),
            os.path.join(base_path, symbol.lower()),
            os.path.join(base_path, symbol.upper(), timeframe),
            os.path.join(base_path, symbol.upper())
        ]
        
        found_file = None
        
        for path in possible_paths:
            if not os.path.exists(path):
                continue
            
            # Try different file patterns
            patterns = [
                f"{symbol.lower()}_{timeframe}_*.csv",
                f"{symbol.lower()}_*.csv",
                f"{symbol.upper()}_{timeframe}_*.csv",
                f"{symbol.upper()}_*.csv",
                "*.csv"
            ]
            
            for pattern in patterns:
                files = glob.glob(os.path.join(path, pattern))
                if files:
                    # Sort files by name
                    files.sort()
                    found_file = files[-1]  # Use the most recent file
                    break
            
            if found_file:
                break
        
        if found_file:
            logging.info(f"Found data file for {symbol}: {found_file}")
            data_files[symbol] = found_file
        else:
            logging.warning(f"No data file found for {symbol}")
    
    return data_files

# --- Run Backtest for Single Cryptocurrency ---
def run_backtest(crypto, data_file):
    """Run backtest for a single cryptocurrency"""
    symbol = crypto["symbol"]
    name = crypto["name"]
    emoji = crypto["emoji"]
    
    logging.info(f"Running backtest for {name} ({symbol}) {emoji}...")
    
    # Load and prepare data
    df = load_data(data_file)
    if df is None:
        logging.warning(f"Failed to load data for {symbol}")
        return None
    
    # Check if we have enough data
    if len(df) < 100:
        logging.warning(f"Insufficient data for {symbol}: {len(df)} rows")
        return None
    
    logging.info(f"Loaded {len(df)} rows of data for {symbol} from {df.index[0]} to {df.index[-1]}")
    
    # Prepare data for backtesting
    df = prepare_data(df)
    
    # Run backtest
    bt = Backtest(df, LongOnlyRSIDivergenceStrategy, cash=INITIAL_CASH, commission=COMMISSION_RATE)
    stats = bt.run()
    
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Save equity curve
    plt.figure(figsize=(10, 6))
    stats['_equity_curve']['Equity'].plot()
    plt.title(f"{name} ({symbol}) {emoji} - Equity Curve")
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, f"{symbol}_equity_curve.png"))
    plt.close()
    
    # Save trades
    trades_df = stats['_trades']
    trades_df.to_csv(os.path.join(RESULTS_DIR, f"{symbol}_trades.csv"))
    
    # Print results
    logging.info(f"Results for {name} ({symbol}) {emoji}:")
    logging.info(f"Return: {stats['Return']:.2%}")
    logging.info(f"Win Rate: {stats['Win Rate']:.2%}")
    logging.info(f"Profit Factor: {stats.get('Profit Factor', 'N/A')}")
    logging.info(f"Sharpe Ratio: {stats.get('Sharpe Ratio', 'N/A')}")
    logging.info(f"Max. Drawdown: {stats['Max. Drawdown']:.2%}")
    logging.info(f"# Trades: {stats['# Trades']}")
    
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
        
        # Save position size analysis
        size_analysis.to_csv(os.path.join(RESULTS_DIR, f"{symbol}_position_size_analysis.csv"))
        
        # Calculate average position size
        avg_position_size = trades_df['Position Size'].mean()
    else:
        avg_position_size = 0
    
    # Return stats for summary
    return {
        "Symbol": symbol,
        "Name": name,
        "Emoji": emoji,
        "Return": stats["Return"],
        "Win Rate": stats["Win Rate"],
        "Profit Factor": stats.get("Profit Factor", float('nan')),
        "Sharpe Ratio": stats.get("Sharpe Ratio", float('nan')),
        "Max. Drawdown": stats["Max. Drawdown"],
        "# Trades": stats["# Trades"],
        "Avg. Volatility": df["Volatility"].mean(),
        "Avg. Position Size": avg_position_size
    }

# --- Main Function ---
def main():
    """Main function"""
    logging.info("Starting long-only RSI divergence backtesting with position sizing...")
    
    # Find data files
    data_files = find_data_files()
    
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Run backtests
    results = []
    for crypto in CRYPTOS:
        symbol = crypto["symbol"]
        if symbol in data_files:
            result = run_backtest(crypto, data_files[symbol])
            if result:
                results.append(result)
    
    # Create summary DataFrame
    if results:
        summary_df = pd.DataFrame(results)
        summary_df = summary_df.sort_values("Return", ascending=False)
        
        # Save summary to CSV
        summary_df.to_csv(os.path.join(RESULTS_DIR, "summary_results.csv"), index=False)
        
        # Print summary
        logging.info("\nSummary Results (sorted by Return):")
        for _, row in summary_df.iterrows():
            logging.info(f"{row['Emoji']} {row['Symbol']}: {row['Return']:.2%} return, {row['Win Rate']:.2%} win rate, {row['# Trades']} trades, {row['Avg. Position Size']:.2%} avg position size")
        
        # Create summary plot
        plt.figure(figsize=(12, 8))
        plt.bar(summary_df["Symbol"], summary_df["Return"] * 100)
        plt.title("RSI Divergence Strategy Returns by Cryptocurrency")
        plt.xlabel("Cryptocurrency")
        plt.ylabel("Return (%)")
        plt.xticks(rotation=45)
        plt.grid(True, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "summary_returns.png"))
        plt.close()
        
        # Create position size vs. volatility plot
        plt.figure(figsize=(10, 6))
        plt.scatter(summary_df["Avg. Volatility"], summary_df["Avg. Position Size"] * 100)
        for i, row in summary_df.iterrows():
            plt.annotate(row["Symbol"], (row["Avg. Volatility"], row["Avg. Position Size"] * 100))
        plt.title("Average Volatility vs. Position Size")
        plt.xlabel("Average Volatility")
        plt.ylabel("Average Position Size (%)")
        plt.grid(True)
        plt.savefig(os.path.join(RESULTS_DIR, "volatility_vs_position_size.png"))
        plt.close()
    
    logging.info("Backtesting complete. Results saved to the 'results' directory.")

if __name__ == "__main__":
    main()
