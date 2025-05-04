import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import logging
from scipy.signal import argrelextrema
import os
import glob
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# --- Default Configuration --- (Will be overridden by args)
_DEFAULT_INITIAL_CASH = 1000000.0
_DEFAULT_COMMISSION_RATE = 0.001
_DEFAULT_RESULTS_DIR = "results"
_DEFAULT_DATA_DIR = "data" 
_DEFAULT_FILENAME_PATTERN = "{symbol}USDT_1d.csv" 

# --- Default Strategy Parameters --- (Will be overridden by args)
_DEFAULT_RSI_LENGTH = 6
_DEFAULT_ATR_LENGTH = 6
_DEFAULT_ATR_MULTIPLIER = 1.1
_DEFAULT_PROFIT_TARGET_PCT = 0.01
_DEFAULT_STOP_LOSS_PCT = 0.008
_DEFAULT_SWING_WINDOW = 3

# --- RSI Calculation --- (Accepts length parameter)
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

# --- ATR Calculation --- (Accepts length parameter)
def calculate_atr(high, low, close, length=14):
    """Calculate Average True Range"""
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr = tr.rolling(window=length).mean()
    return atr

# --- Find Local Extrema --- (Accepts order parameter)
def find_local_extrema(series, order=5, mode='max'):
    """Find local maxima or minima in a series"""
    try:
        if mode == 'max':
            idx = argrelextrema(series.values, np.greater_equal, order=order)[0]
        else:
            idx = argrelextrema(series.values, np.less_equal, order=order)[0]
        extrema = np.zeros(series.shape, dtype=bool)
        extrema[idx] = True
        return pd.Series(extrema, index=series.index)
    except Exception as e:
        logging.error(f"Error finding local extrema: {e}")
        # Return a series of False if error occurs
        return pd.Series(False, index=series.index)

# --- Load and Prepare Data --- (Remains largely the same)
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

# --- Prepare Data for Backtesting --- (Accepts parameters)
def prepare_data(df, rsi_length, atr_length, swing_window):
    """Calculate indicators and detect divergences using provided parameters"""
    logging.info("Preparing data for backtesting...")
    
    # Calculate RSI
    df['RSI'] = calculate_rsi(df['Close'], length=rsi_length)
    
    # Calculate ATR
    df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'], length=atr_length)
    
    # Find local extrema
    df['local_max'] = find_local_extrema(df['Close'], order=swing_window, mode='max')
    df['local_min'] = find_local_extrema(df['Close'], order=swing_window, mode='min')
    df['rsi_local_max'] = find_local_extrema(df['RSI'], order=swing_window, mode='max')
    df['rsi_local_min'] = find_local_extrema(df['RSI'], order=swing_window, mode='min')
    
    # Initialize divergence columns
    df['bullish_div'] = False
    df['bearish_div'] = False
    
    # Log the number of extrema points found
    logging.info(f"Found {df['local_max'].sum()} price maxima and {df['local_min'].sum()} price minima using swing window {swing_window}")
    logging.info(f"Found {df['rsi_local_max'].sum()} RSI maxima and {df['rsi_local_min'].sum()} RSI minima using swing window {swing_window}")
    
    # Detect divergences
    detect_divergences(df, swing_window) # Pass swing_window
    
    # Drop rows with NaN values from indicator calculations
    df.dropna(inplace=True)
    
    logging.info(f"Data preparation complete. Found {df['bullish_div'].sum()} bullish and {df['bearish_div'].sum()} bearish divergences")
    return df

# --- Detect Divergences --- (Accepts swing_window)
def detect_divergences(df, swing_window):
    """Detect bullish and bearish divergences"""
    logging.info("Detecting divergences...")
    search_window = swing_window * 5 # Look back a reasonable window
    
    # Loop through each bar
    for i in range(swing_window, len(df) - swing_window):
        # Check for bullish divergence (price makes lower low, RSI makes higher low)
        if df.iloc[i]['local_min']:
            # Find previous price minimum within search window
            prev_mins_indices = np.where(df.iloc[max(0, i-search_window):i]['local_min'].values)[0]
            if len(prev_mins_indices) > 0:
                prev_price_idx = max(0, i-search_window) + prev_mins_indices[-1]
                
                # Check if current price is lower than previous
                if df.iloc[i]['Low'] < df.iloc[prev_price_idx]['Low']:
                    # Find corresponding RSI values
                    curr_rsi = df.iloc[i]['RSI']
                    prev_rsi = df.iloc[prev_price_idx]['RSI']
                    
                    # Check if RSI is making higher low (bullish divergence)
                    if curr_rsi > prev_rsi:
                        df.iloc[i, df.columns.get_loc('bullish_div')] = True
                        # logging.debug(f"Bullish divergence detected at index {i} (Price {df.iloc[i]['Close']:.2f} < {df.iloc[prev_price_idx]['Close']:.2f}, RSI {curr_rsi:.2f} > {prev_rsi:.2f})")

        # Check for bearish divergence (price makes higher high, RSI makes lower high)
        if df.iloc[i]['local_max']:
            # Find previous price maximum within search window
            prev_max_indices = np.where(df.iloc[max(0, i-search_window):i]['local_max'].values)[0]
            if len(prev_max_indices) > 0:
                prev_price_idx = max(0, i-search_window) + prev_max_indices[-1]
                
                # Check if current price is higher than previous
                if df.iloc[i]['High'] > df.iloc[prev_price_idx]['High']:
                    # Find corresponding RSI values
                    curr_rsi = df.iloc[i]['RSI']
                    prev_rsi = df.iloc[prev_price_idx]['RSI']
                    
                    # Check if RSI is making lower high (bearish divergence)
                    if curr_rsi < prev_rsi:
                        df.iloc[i, df.columns.get_loc('bearish_div')] = True
                        # logging.debug(f"Bearish divergence detected at index {i} (Price {df.iloc[i]['Close']:.2f} > {df.iloc[prev_price_idx]['Close']:.2f}, RSI {curr_rsi:.2f} < {prev_rsi:.2f})")

# --- RSI Divergence Strategy --- (Uses parameters passed by Backtest)
class RSIDivergenceStrategy(Strategy):
    """ RSI Divergence Trading Strategy using parameters """
    # Define parameters that Backtest will pass to the strategy
    rsi_length = _DEFAULT_RSI_LENGTH
    atr_length = _DEFAULT_ATR_LENGTH
    atr_multiplier = _DEFAULT_ATR_MULTIPLIER
    profit_target_pct = _DEFAULT_PROFIT_TARGET_PCT
    stop_loss_pct = _DEFAULT_STOP_LOSS_PCT
    # swing_window is used in prepare_data, not directly here

    def init(self):
        # Access parameters passed by Backtest
        self.rsi = self.I(calculate_rsi, self.data.Close, length=self.rsi_length)
        self.atr = self.I(calculate_atr, self.data.High, self.data.Low, self.data.Close, length=self.atr_length)
        # Access pre-calculated divergences from the dataframe passed to Backtest
        self.bullish_div = self.data.bullish_div
        self.bearish_div = self.data.bearish_div

    def next(self):
        # Check for open positions
        if self.position:
            # --- Profit Target & Stop Loss for Long --- 
            if self.position.is_long:
                # Simple % based TP/SL
                if self.data.Close[-1] >= self.position.entry_price * (1 + self.profit_target_pct):
                    self.position.close() # Take profit
                elif self.data.Close[-1] <= self.position.entry_price * (1 - self.stop_loss_pct):
                    self.position.close() # Stop loss
            # --- Profit Target & Stop Loss for Short --- 
            # Note: Shorting logic can be added here if desired
            # elif self.position.is_short:
                # if self.data.Close[-1] <= self.position.entry_price * (1 - self.profit_target_pct):
                #     self.position.close() # Take profit
                # elif self.data.Close[-1] >= self.position.entry_price * (1 + self.stop_loss_pct):
                #     self.position.close() # Stop loss
            return # Don't enter new position if one is open

        # --- Entry Conditions ---
        # Bullish divergence detected
        if self.bullish_div[-1]:
            # Calculate dynamic stop loss based on ATR 
            # sl = self.data.Close[-1] - self.atr[-1] * self.atr_multiplier
            # Calculate dynamic take profit 
            # tp = self.data.Close[-1] + (self.data.Close[-1] - sl) * 2 # Example: 2:1 Risk/Reward
            # Buy with calculated SL/TP
            # self.buy(sl=sl, tp=tp)
            # Using simpler % based SL for consistency with current params
            self.buy()

        # Bearish divergence detected (Example for adding shorting later)
        # elif self.bearish_div[-1]:
            # sl = self.data.Close[-1] + self.atr[-1] * self.atr_multiplier
            # tp = self.data.Close[-1] - (sl - self.data.Close[-1]) * 2
            # self.sell(sl=sl, tp=tp)
            # pass # Currently long-only focus

# --- Find Data Files --- (Uses args)
def find_data_files(data_dir, symbols, filename_pattern):
    """Find data files for the specified cryptocurrencies in the data directory."""
    data_files = {}
    logging.info(f"Looking for data files in '{data_dir}' for symbols: {', '.join(symbols)}")
    logging.info(f"Using filename pattern: '{filename_pattern}'")
    
    found_symbols = set()
    missing_symbols = set(symbols)
    
    # Attempt to find files using the pattern
    for symbol in symbols:
        try:
            # Format the pattern with the current symbol
            pattern = os.path.join(data_dir, filename_pattern.format(symbol=symbol))
            files = glob.glob(pattern)
            
            if files:
                # Use the first file found if multiple match (e.g., different date ranges)
                # Consider adding logic to select the most appropriate file if needed
                data_file_path = files[0] 
                data_files[symbol] = data_file_path
                found_symbols.add(symbol)
                missing_symbols.remove(symbol)
                logging.info(f"  Found: {data_file_path} for {symbol}")
            else:
                logging.warning(f"  Could not find data file for {symbol} using pattern: {pattern}")
        except Exception as e:
             logging.error(f"Error processing symbol {symbol} with pattern '{filename_pattern}': {e}")

    if not data_files:
        logging.error(f"No data files found in '{data_dir}' for the specified symbols and pattern.")
    elif missing_symbols:
        logging.warning(f"Could not find data for symbols: {', '.join(missing_symbols)}")
        
    return data_files

# --- Run Backtest for Single Cryptocurrency --- (Uses args)
def run_backtest(symbol, data_file, results_dir, initial_cash, commission_rate, strategy_params):
    """Run backtest for a single cryptocurrency."""
    logging.info(f"--- Starting backtest for {symbol} --- ")
    
    try:
        # Load data
        data = load_data(data_file)
        if data is None or data.empty:
            logging.error(f"No data loaded for {symbol}, skipping backtest.")
            return None
        
        # Prepare data with specific parameters
        data = prepare_data(data,
                            rsi_length=strategy_params['rsi_length'],
                            atr_length=strategy_params['atr_length'],
                            swing_window=strategy_params['swing_window'])
        
        if data.empty:
             logging.error(f"Data empty after preparation for {symbol}, skipping backtest.")
             return None

        # Create Backtest instance
        # Pass strategy parameters directly to Backtest
        bt = Backtest(data,
                      RSIDivergenceStrategy,
                      cash=initial_cash,
                      commission=commission_rate,
                      trade_on_close=True,
                      exclusive_orders=True)
        
        # Run the backtest, passing strategy parameters
        stats = bt.run(**strategy_params) # Pass params like rsi_length, etc.
        
        logging.info(f"Backtest for {symbol} completed.")
        print(f"\nResults for {symbol}:\n{stats}")
        
        # Ensure results directory exists
        os.makedirs(results_dir, exist_ok=True)
        
        # Save results and plot
        results_filename_base = os.path.join(results_dir, f"{symbol}_rsi_div")
        stats_file = f"{results_filename_base}_results.txt"
        plot_file = f"{results_filename_base}_plot.html"
        trades_file = f"{results_filename_base}_trades.csv"
        
        with open(stats_file, 'w') as f:
            f.write(str(stats))
        logging.info(f"Stats saved to {stats_file}")
        
        try:
            bt.plot(filename=plot_file, open_browser=False)
            logging.info(f"Plot saved to {plot_file}")
        except Exception as plot_error:
            logging.error(f"Could not generate plot for {symbol}: {plot_error}")
        
        # Save trades
        trades_df = stats['_trades']
        if not trades_df.empty:
            trades_df.to_csv(trades_file)
            logging.info(f"Trades saved to {trades_file}")
        else:
            logging.info(f"No trades executed for {symbol}.")
            
        return stats
        
    except Exception as e:
        logging.error(f"Error running backtest for {symbol}: {e}")
        return None

# --- Main Function --- (Parses arguments)
def main():
    parser = argparse.ArgumentParser(description="Run RSI Divergence backtests on specified cryptocurrency data.")
    
    # --- File/Directory Arguments ---
    parser.add_argument('--data-dir', type=str, default=_DEFAULT_DATA_DIR,
                        help=f"Directory containing the cryptocurrency data CSV files. Default: '{_DEFAULT_DATA_DIR}'")
    parser.add_argument('--results-dir', type=str, default=_DEFAULT_RESULTS_DIR,
                        help=f"Directory to save backtest results and plots. Default: '{_DEFAULT_RESULTS_DIR}'")
    parser.add_argument('--filename-pattern', type=str, default=_DEFAULT_FILENAME_PATTERN,
                        help=f"Pattern for data filenames, use {{symbol}} as placeholder. Default: '{_DEFAULT_FILENAME_PATTERN}'")

    # --- Symbol Argument ---
    parser.add_argument('--symbols', nargs='+', required=True,
                        help="List of cryptocurrency symbols to backtest (e.g., BTC ETH SOL).")

    # --- Backtest Configuration Arguments ---
    parser.add_argument('--initial-cash', type=float, default=_DEFAULT_INITIAL_CASH,
                        help=f"Initial cash for backtesting. Default: {_DEFAULT_INITIAL_CASH}")
    parser.add_argument('--commission', type=float, default=_DEFAULT_COMMISSION_RATE,
                        help=f"Commission rate per trade. Default: {_DEFAULT_COMMISSION_RATE}")

    # --- Strategy Parameter Arguments ---
    parser.add_argument('--rsi-length', type=int, default=_DEFAULT_RSI_LENGTH,
                        help=f"RSI period length. Default: {_DEFAULT_RSI_LENGTH}")
    parser.add_argument('--atr-length', type=int, default=_DEFAULT_ATR_LENGTH,
                        help=f"ATR period length. Default: {_DEFAULT_ATR_LENGTH}")
    parser.add_argument('--atr-multiplier', type=float, default=_DEFAULT_ATR_MULTIPLIER,
                        help=f"ATR multiplier for stop loss calculation (if used dynamically). Default: {_DEFAULT_ATR_MULTIPLIER}")
    parser.add_argument('--profit-target', type=float, default=_DEFAULT_PROFIT_TARGET_PCT,
                        help=f"Profit target percentage (e.g., 0.01 for 1%). Default: {_DEFAULT_PROFIT_TARGET_PCT}")
    parser.add_argument('--stop-loss', type=float, default=_DEFAULT_STOP_LOSS_PCT,
                        help=f"Stop loss percentage (e.g., 0.008 for 0.8%). Default: {_DEFAULT_STOP_LOSS_PCT}")
    parser.add_argument('--swing-window', type=int, default=_DEFAULT_SWING_WINDOW,
                        help=f"Window size for detecting local price/RSI extrema. Default: {_DEFAULT_SWING_WINDOW}")

    args = parser.parse_args()

    # Prepare strategy parameters dictionary
    strategy_params = {
        'rsi_length': args.rsi_length,
        'atr_length': args.atr_length,
        'atr_multiplier': args.atr_multiplier,
        'profit_target_pct': args.profit_target,
        'stop_loss_pct': args.stop_loss,
        'swing_window': args.swing_window
    }

    # Find data files
    data_files = find_data_files(args.data_dir, args.symbols, args.filename_pattern)
    
    if not data_files:
        print("Exiting due to missing data files.")
        return

    all_stats = {}
    # Run backtests for each found cryptocurrency
    for symbol, data_file in data_files.items():
        stats = run_backtest(symbol, 
                             data_file, 
                             args.results_dir, 
                             args.initial_cash, 
                             args.commission, 
                             strategy_params)
        if stats is not None:
            all_stats[symbol] = stats

    # Optional: Summarize results across all tested symbols
    if all_stats:
        print("\n--- Overall Summary ---")
        summary_df = pd.DataFrame({
            'Return [%]': {s: st['Return [%]'] for s, st in all_stats.items()},
            'Win Rate [%]': {s: st['Win Rate [%]'] for s, st in all_stats.items()},
            'Profit Factor': {s: st['Profit Factor'] for s, st in all_stats.items()},
            '# Trades': {s: st['# Trades'] for s, st in all_stats.items()}
        }).sort_values('Return [%]', ascending=False)
        print(summary_df)
        
        # Save summary
        summary_file = os.path.join(args.results_dir, "_combined_summary.txt")
        summary_df.to_string(summary_file)
        logging.info(f"Combined summary saved to {summary_file}")

    print("\nAll backtests finished.")

if __name__ == "__main__":
    main()
