import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import logging
from scipy.signal import argrelextrema
import os
import glob

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
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr = tr.rolling(window=length).mean()
    return atr

# --- Find Local Extrema ---
def find_local_extrema(series, order=5, mode='max'):
    """Find local maxima or minima in a series"""
    if mode == 'max':
        idx = argrelextrema(series.values, np.greater_equal, order=order)[0]
    else:
        idx = argrelextrema(series.values, np.less_equal, order=order)[0]
    extrema = np.zeros(series.shape, dtype=bool)
    extrema[idx] = True
    return pd.Series(extrema, index=series.index)

# --- Load and Prepare Data ---
def load_data(file_path):
    """Load data from CSV file"""
    logging.info(f"Loading data from {file_path}")
    
    try:
        # Load data
        df = pd.read_csv(file_path)
        
        # Convert timestamp to datetime and set as index
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        elif 'date' in df.columns or 'Date' in df.columns:
            date_col = 'date' if 'date' in df.columns else 'Date'
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        
        # Rename columns to match Backtest.py requirements
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'open' in col_lower:
                column_mapping[col] = 'Open'
            elif 'high' in col_lower:
                column_mapping[col] = 'High'
            elif 'low' in col_lower:
                column_mapping[col] = 'Low'
            elif 'close' in col_lower:
                column_mapping[col] = 'Close'
            elif 'volume' in col_lower:
                column_mapping[col] = 'Volume'
        
        df.rename(columns=column_mapping, inplace=True)
        
        # Convert to numeric
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'Volume' in df.columns:
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        else:
            df['Volume'] = 0
        
        # Drop rows with NaN values
        df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
        
        logging.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

# --- Prepare Data for Backtesting ---
def prepare_data(df):
    """Calculate indicators and detect divergences"""
    logging.info("Preparing data for backtesting...")
    
    # Calculate RSI
    df['RSI'] = calculate_rsi(df['Close'], length=RSI_LENGTH)
    
    # Calculate ATR
    df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'], length=ATR_LENGTH)
    
    # Find local extrema
    df['local_max'] = find_local_extrema(df['Close'], order=SWING_WINDOW, mode='max')
    df['local_min'] = find_local_extrema(df['Close'], order=SWING_WINDOW, mode='min')
    df['rsi_local_max'] = find_local_extrema(df['RSI'], order=SWING_WINDOW, mode='max')
    df['rsi_local_min'] = find_local_extrema(df['RSI'], order=SWING_WINDOW, mode='min')
    
    # Initialize divergence columns
    df['bullish_div'] = False
    df['bearish_div'] = False
    
    # Log the number of extrema points found
    logging.info(f"Found {df['local_max'].sum()} price maxima and {df['local_min'].sum()} price minima")
    logging.info(f"Found {df['rsi_local_max'].sum()} RSI maxima and {df['rsi_local_min'].sum()} RSI minima")
    
    # Detect divergences
    detect_divergences(df)
    
    # Drop rows with NaN values from indicator calculations
    df.dropna(inplace=True)
    
    logging.info(f"Data preparation complete. Found {df['bullish_div'].sum()} bullish and {df['bearish_div'].sum()} bearish divergences")
    return df

# --- Detect Divergences ---
def detect_divergences(df):
    """Detect bullish and bearish divergences"""
    logging.info("Detecting divergences...")
    
    # Loop through each bar
    for i in range(SWING_WINDOW, len(df) - SWING_WINDOW):
        # Check for bullish divergence (price makes lower low, RSI makes higher low)
        if df.iloc[i]['local_min']:
            # Find previous price minimum
            prev_mins = np.where(df.iloc[max(0, i-20):i]['local_min'].values)[0]
            if len(prev_mins) > 0:
                prev_price_idx = max(0, i-20) + prev_mins[-1]
                
                # Check if current price is lower than previous
                if df.iloc[i]['Close'] < df.iloc[prev_price_idx]['Close']:
                    # Find corresponding RSI values
                    curr_rsi = df.iloc[i]['RSI']
                    prev_rsi = df.iloc[prev_price_idx]['RSI']
                    
                    # Check if RSI is making higher low (bullish divergence)
                    if curr_rsi > prev_rsi:
                        df.loc[df.index[i], 'bullish_div'] = True
        
        # Check for bearish divergence (price makes higher high, RSI makes lower high)
        if df.iloc[i]['local_max']:
            # Find previous price maximum
            prev_maxs = np.where(df.iloc[max(0, i-20):i]['local_max'].values)[0]
            if len(prev_maxs) > 0:
                prev_price_idx = max(0, i-20) + prev_maxs[-1]
                
                # Check if current price is higher than previous
                if df.iloc[i]['Close'] > df.iloc[prev_price_idx]['Close']:
                    # Find corresponding RSI values
                    curr_rsi = df.iloc[i]['RSI']
                    prev_rsi = df.iloc[prev_price_idx]['RSI']
                    
                    # Check if RSI is making lower high (bearish divergence)
                    if curr_rsi < prev_rsi:
                        df.loc[df.index[i], 'bearish_div'] = True

# --- RSI Divergence Strategy ---
class RSIDivergenceStrategy(Strategy):
    """RSI Divergence Trading Strategy"""
    
    def init(self):
        # Store indicators for the strategy
        self.rsi = self.I(lambda x: x, self.data.RSI)
        self.atr = self.I(lambda x: x, self.data.ATR)
        self.bullish_div = self.I(lambda x: x, self.data.bullish_div)
        self.bearish_div = self.I(lambda x: x, self.data.bearish_div)
        
        # Position management variables
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
    
    def next(self):
        price = self.data.Close[-1]
        atr_val = self.atr[-1]
        
        # Log current price and indicators for debugging only when divergences are detected
        if self.bullish_div[-1] or self.bearish_div[-1]:
            logging.info(f"Price={price:.2f}, RSI={self.rsi[-1]:.2f}, ATR={atr_val:.2f}")
            logging.info(f"Bullish div: {self.bullish_div[-1]}, Bearish div: {self.bearish_div[-1]}")
        
        # Manage existing positions
        if self.position:
            # Check stop loss and take profit for long positions
            if self.position.is_long:
                if price <= self.stop_loss or price >= self.take_profit:
                    self.position.close()
                    logging.info(f"Closed LONG position at {self.data.index[-1]}, price={price:.2f}")
            
            # Check stop loss and take profit for short positions
            elif self.position.is_short:
                if price >= self.stop_loss or price <= self.take_profit:
                    self.position.close()
                    logging.info(f"Closed SHORT position at {self.data.index[-1]}, price={price:.2f}")
            
            return
        
        # Entry logic
        if self.bullish_div[-1]:
            # Long entry on bullish divergence
            self.entry_price = price
            self.stop_loss = price * (1 - STOP_LOSS_PCT)
            self.take_profit = price * (1 + PROFIT_TARGET_PCT)
            
            # Enter long position
            self.buy()
            logging.info(f"LONG entry at {self.data.index[-1]}, price={price:.2f}, stop={self.stop_loss:.2f}, target={self.take_profit:.2f}")
        
        elif self.bearish_div[-1]:
            # Short entry on bearish divergence
            self.entry_price = price
            self.stop_loss = price * (1 + STOP_LOSS_PCT)
            self.take_profit = price * (1 - PROFIT_TARGET_PCT)
            
            # Enter short position
            self.sell()
            logging.info(f"SHORT entry at {self.data.index[-1]}, price={price:.2f}, stop={self.stop_loss:.2f}, target={self.take_profit:.2f}")

# --- Find Data Files ---
def find_data_files():
    """Find data files for each cryptocurrency"""
    base_dir = r'F:\Master_Data\market_data'
    crypto_files = {}
    
    for crypto in CRYPTOS:
        symbol = crypto['symbol'].lower()
        
        # Check if directory exists
        crypto_dir = os.path.join(base_dir, symbol)
        if not os.path.exists(crypto_dir):
            logging.warning(f"No directory found for {crypto['name']} at {crypto_dir}")
            continue
        
        # Check for 5m data
        data_dir_5m = os.path.join(crypto_dir, '5m')
        if os.path.exists(data_dir_5m):
            # Try different file naming patterns
            patterns = [
                f'{symbol}_5m_*.csv',  # Standard pattern
                f'*{symbol}*5m*.csv',   # Alternative pattern
                '*.csv'                # Any CSV file
            ]
            
            found = False
            for pattern in patterns:
                files = glob.glob(os.path.join(data_dir_5m, pattern))
                if files:
                    # Sort files by modification time (newest first)
                    files.sort(key=os.path.getmtime, reverse=True)
                    crypto_files[symbol] = files[0]
                    logging.info(f"Found data file for {crypto['name']}: {os.path.basename(files[0])}")
                    found = True
                    break
            
            if not found:
                logging.warning(f"No data files found for {crypto['name']} in {data_dir_5m}")
        else:
            # Check for data in other timeframes
            timeframes = ['1m', '15m', '1h', '4h', '1d']
            for tf in timeframes:
                data_dir_alt = os.path.join(crypto_dir, tf)
                if os.path.exists(data_dir_alt):
                    files = glob.glob(os.path.join(data_dir_alt, f'*{symbol}*{tf}*.csv'))
                    if files:
                        files.sort(key=os.path.getmtime, reverse=True)
                        crypto_files[symbol] = files[0]
                        logging.info(f"Found alternative timeframe data for {crypto['name']}: {os.path.basename(files[0])}")
                        break
            else:
                logging.warning(f"No data files found for {crypto['name']} in any timeframe")
    
    return crypto_files

# --- Run Backtest for Single Cryptocurrency ---
def run_backtest(crypto, data_file):
    """Run backtest for a single cryptocurrency"""
    try:
        # Create results directory if it doesn't exist
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        
        # Results filename
        results_file = os.path.join(RESULTS_DIR, f"{crypto['symbol'].lower()}_rsi_div_results.txt")
        
        # Load and prepare data
        data = load_data(data_file)
        data = prepare_data(data)
        
        # Run backtest
        logging.info(f"Running backtest for {crypto['name']} ({crypto['symbol']})...")
        bt = Backtest(data, RSIDivergenceStrategy, cash=INITIAL_CASH, commission=COMMISSION_RATE)
        stats = bt.run()
        
        # Print results
        print(f"\n--- {crypto['emoji']} {crypto['name']} ({crypto['symbol']}) Backtest Results ---")
        print(f"Return: {stats['Return [%]']:.2f}%")
        print(f"Win Rate: {stats['Win Rate [%]']:.2f}%")
        print(f"Profit Factor: {stats['Profit Factor']:.2f}")
        print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
        print(f"Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")
        print(f"# Trades: {stats['# Trades']}")
        
        # Save detailed results to file
        with open(results_file, "w") as f:
            f.write(f"Backtest Results for RSI Divergence Strategy on {crypto['name']} ({crypto['symbol']})\n")
            f.write(f"Data: {data_file}\n")
            f.write(f"Initial Cash: {INITIAL_CASH}\n")
            f.write(f"Commission: {COMMISSION_RATE}\n\n")
            f.write(stats.to_string())
        
        logging.info(f"Results saved to {results_file}")
        
        # Skip plotting to avoid errors
        logging.info("Skipping plot generation to avoid errors")
        
        return {
            "symbol": crypto['symbol'],
            "name": crypto['name'],
            "emoji": crypto['emoji'],
            "return": stats['Return [%]'],
            "win_rate": stats['Win Rate [%]'],
            "profit_factor": stats['Profit Factor'],
            "sharpe": stats['Sharpe Ratio'],
            "max_drawdown": stats['Max. Drawdown [%]'],
            "num_trades": stats['# Trades']
        }
    
    except Exception as e:
        logging.error(f"Error running backtest for {crypto['symbol']}: {e}")
        return {
            "symbol": crypto['symbol'],
            "name": crypto['name'],
            "emoji": crypto['emoji'],
            "error": str(e)
        }

# --- Main Function ---
def main():
    try:
        # Find data files for each cryptocurrency
        crypto_files = find_data_files()
        
        # Run backtest for each cryptocurrency
        results = []
        for crypto in CRYPTOS:
            symbol = crypto['symbol'].lower()
            if symbol in crypto_files:
                result = run_backtest(crypto, crypto_files[symbol])
                results.append(result)
        
        # Generate summary report
        print("\n--- 📊 RSI Divergence Strategy Summary Report ---")
        print(f"{'Symbol':<6} {'Name':<12} {'Return':<10} {'Win Rate':<10} {'Profit Factor':<15} {'Sharpe':<10} {'Max DD':<10} {'# Trades':<10}")
        print("-" * 80)
        
        for result in results:
            if 'error' in result:
                print(f"{result['symbol']:<6} {result['name']:<12} ERROR: {result['error']}")
            else:
                print(f"{result['symbol']:<6} {result['name']:<12} {result['return']:.2f}% {result['win_rate']:.2f}% {result['profit_factor']:.2f} {result['sharpe']:.2f} {result['max_drawdown']:.2f}% {result['num_trades']}")
        
        # Save summary to file
        summary_file = os.path.join(RESULTS_DIR, "rsi_div_summary.txt")
        with open(summary_file, "w") as f:
            f.write("RSI Divergence Strategy Summary Report\n")
            f.write(f"{'Symbol':<6} {'Name':<12} {'Return':<10} {'Win Rate':<10} {'Profit Factor':<15} {'Sharpe':<10} {'Max DD':<10} {'# Trades':<10}\n")
            f.write("-" * 80 + "\n")
            
            for result in results:
                if 'error' in result:
                    f.write(f"{result['symbol']:<6} {result['name']:<12} ERROR: {result['error']}\n")
                else:
                    f.write(f"{result['symbol']:<6} {result['name']:<12} {result['return']:.2f}% {result['win_rate']:.2f}% {result['profit_factor']:.2f} {result['sharpe']:.2f} {result['max_drawdown']:.2f}% {result['num_trades']}\n")
        
        logging.info(f"Summary report saved to {summary_file}")
        
    except Exception as e:
        logging.error(f"Error in main function: {e}")

if __name__ == "__main__":
    main()
