import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import logging
from scipy.signal import argrelextrema

# --- Configuration ---
INITIAL_CASH = 1000000.0  # Increased to handle BTC price levels
COMMISSION_RATE = 0.001  # 0.1%
RESULTS_FILENAME = "simple_rsi_div_results.txt"

# --- Optimized Parameters ---
# Ultra-sensitive settings to detect more trading opportunities
RSI_LENGTH = 4  # Very short RSI period for maximum responsiveness
ATR_LENGTH = 4  # Very short ATR period
ATR_MULTIPLIER = 0.8  # Very tight stops
PROFIT_TARGET_PCT = 0.005  # Smaller profit target for more frequent wins
STOP_LOSS_PCT = 0.004  # Very tight stop loss
SWING_WINDOW = 1  # Minimum window to detect all possible swings
ALIGN_WINDOW = 1  # Minimum alignment window

# --- Manual Indicator Calculation ---
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

def calculate_atr(high, low, close, length=14):
    """Calculate Average True Range"""
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr = tr.rolling(window=length).mean()
    return atr

def find_local_extrema(series, order=5, mode='max'):
    """Find local maxima or minima in a series"""
    if mode == 'max':
        idx = argrelextrema(series.values, np.greater_equal, order=order)[0]
    else:
        idx = argrelextrema(series.values, np.less_equal, order=order)[0]
    extrema = np.zeros(series.shape, dtype=bool)
    extrema[idx] = True
    return pd.Series(extrema, index=series.index)

def detect_rsi_divergence(df, swing_window=SWING_WINDOW, align_window=ALIGN_WINDOW):
    """Detect RSI divergences with enhanced sensitivity"""
    # Find local extrema in price and RSI with smaller window for more signals
    df['local_max'] = find_local_extrema(df['Close'], order=swing_window, mode='max')
    df['local_min'] = find_local_extrema(df['Close'], order=swing_window, mode='min')
    df['rsi_local_max'] = find_local_extrema(df['RSI'], order=swing_window, mode='max')
    df['rsi_local_min'] = find_local_extrema(df['RSI'], order=swing_window, mode='min')
    
    # Count the extrema points for debugging
    num_price_max = df['local_max'].sum()
    num_price_min = df['local_min'].sum()
    num_rsi_max = df['rsi_local_max'].sum()
    num_rsi_min = df['rsi_local_min'].sum()
    
    logging.info(f"Found {num_price_max} price maxima, {num_price_min} price minima")
    logging.info(f"Found {num_rsi_max} RSI maxima, {num_rsi_min} RSI minima")
    df['bullish_div'] = False
    df['bearish_div'] = False
    df['div_strength'] = np.nan
    
    # For each divergence point, find the next two local extrema in price and RSI
    for i in range(len(df) - align_window):
        if df.iloc[i]['local_min'] and df.iloc[i:i+align_window]['rsi_local_min'].any():
            price_idx = i
            rsi_indices = np.where(df.iloc[i:i+align_window]['rsi_local_min'].values)[0]
            if len(rsi_indices) > 0:
                rsi_idx = i + rsi_indices[0]
                
                # Check if price is making lower low but RSI is making higher low (bullish div)
                if price_idx > swing_window and rsi_idx > swing_window:
                    # Find previous price minimum
                    prev_price_indices = np.where(df.iloc[max(0, price_idx-swing_window*2):price_idx]['local_min'].values)[0]
                    if len(prev_price_indices) > 0:
                        price_prev_idx = max(0, price_idx-swing_window*2) + prev_price_indices[-1]
                        
                        # Find previous RSI minimum
                        prev_rsi_indices = np.where(df.iloc[max(0, rsi_idx-swing_window*2):rsi_idx]['rsi_local_min'].values)[0]
                        if len(prev_rsi_indices) > 0:
                            rsi_prev_idx = max(0, rsi_idx-swing_window*2) + prev_rsi_indices[-1]
                            
                            # Relaxed condition for bullish divergence - any RSI improvement counts
                            if df.iloc[price_idx]['Close'] <= df.iloc[price_prev_idx]['Close'] and \
                               df.iloc[rsi_idx]['RSI'] >= df.iloc[rsi_prev_idx]['RSI']:
                                # Calculate divergence strength
                                strength = (df.iloc[rsi_idx]['RSI'] - df.iloc[rsi_prev_idx]['RSI']) / \
                                           (df.iloc[price_prev_idx]['Close'] - df.iloc[price_idx]['Close']) * 100
                                df.loc[df.index[price_idx], 'div_strength'] = strength
                                df.loc[df.index[price_idx], 'bullish_div'] = True
        
        if df.iloc[i]['local_max'] and df.iloc[i:i+align_window]['rsi_local_max'].any():
            price_idx = i
            rsi_indices = np.where(df.iloc[i:i+align_window]['rsi_local_max'].values)[0]
            if len(rsi_indices) > 0:
                rsi_idx = i + rsi_indices[0]
                
                # Check if price is making higher high but RSI is making lower high (bearish div)
                if price_idx > swing_window and rsi_idx > swing_window:
                    # Find previous price maximum
                    prev_price_indices = np.where(df.iloc[max(0, price_idx-swing_window*2):price_idx]['local_max'].values)[0]
                    if len(prev_price_indices) > 0:
                        price_prev_idx = max(0, price_idx-swing_window*2) + prev_price_indices[-1]
                        
                        # Find previous RSI maximum
                        prev_rsi_indices = np.where(df.iloc[max(0, rsi_idx-swing_window*2):rsi_idx]['rsi_local_max'].values)[0]
                        if len(prev_rsi_indices) > 0:
                            rsi_prev_idx = max(0, rsi_idx-swing_window*2) + prev_rsi_indices[-1]
                            
                            # Relaxed condition for bearish divergence - any RSI deterioration counts
                            if df.iloc[price_idx]['Close'] >= df.iloc[price_prev_idx]['Close'] and \
                               df.iloc[rsi_idx]['RSI'] <= df.iloc[rsi_prev_idx]['RSI']:
                                # Calculate divergence strength
                                strength = (df.iloc[rsi_prev_idx]['RSI'] - df.iloc[rsi_idx]['RSI']) / \
                                           (df.iloc[price_idx]['Close'] - df.iloc[price_prev_idx]['Close']) * 100
                                df.loc[df.index[price_idx], 'div_strength'] = strength
                                df.loc[df.index[price_idx], 'bearish_div'] = True
    return df

def prepare_data(filename):
    """Load and prepare data for backtesting from a file"""
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    logging.info(f"Loading data from {filename}...")
    
    try:
        # Load CSV data
        df = pd.read_csv(filename)
        return prepare_data_from_df(df)
    
    except Exception as e:
        logging.error(f"Error preparing data from file: {e}")
        raise

def prepare_data_from_df(df):
    """Prepare a DataFrame for backtesting"""
    try:
        # Check if we have a timestamp/date column
        date_col = None
        for col in df.columns:
            if 'time' in col.lower() or 'date' in col.lower():
                date_col = col
                break
        
        # If DataFrame is not already indexed by datetime
        if date_col and not isinstance(df.index, pd.DatetimeIndex):
            # Convert to datetime and set as index
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        
        # Standardize column names if needed
        if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            col_map = {}
            for col in df.columns:
                col_lower = col.lower()
                if 'open' in col_lower:
                    col_map[col] = 'Open'
                elif 'high' in col_lower:
                    col_map[col] = 'High'
                elif 'low' in col_lower:
                    col_map[col] = 'Low'
                elif 'close' in col_lower:
                    col_map[col] = 'Close'
                elif 'volume' in col_lower:
                    col_map[col] = 'Volume'
            
            # Rename columns
            df.rename(columns=col_map, inplace=True)
        
        # Make sure we have all required columns
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert to numeric
        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'Volume' in df.columns:
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        else:
            df['Volume'] = 0
        
        # Drop rows with NaN values
        df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
        
        # Calculate indicators
        df['RSI'] = calculate_rsi(df['Close'], length=RSI_LENGTH)
        df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'], length=ATR_LENGTH)
        
        # Detect divergences
        df = detect_rsi_divergence(df, swing_window=SWING_WINDOW, align_window=ALIGN_WINDOW)
        
        # Add signal column
        df['signal'] = 0
        df.loc[df['bullish_div'], 'signal'] = 1
        df.loc[df['bearish_div'], 'signal'] = -1
        
        # Drop NaN rows from indicator calculations
        df.dropna(inplace=True)
        
        logging.info(f"Data prepared successfully. Shape: {df.shape}")
        logging.info(f"Bullish divergences: {df['bullish_div'].sum()}")
        logging.info(f"Bearish divergences: {df['bearish_div'].sum()}")
        
        return df
    
    except Exception as e:
        logging.error(f"Error preparing data from DataFrame: {e}")
        raise

# --- Strategy Definition ---
class RSIDivergenceStrategy(Strategy):
    """RSI Divergence strategy with optimized parameters"""
    
    def init(self):
        self.rsi = self.I(lambda x: x, self.data.RSI)
        self.atr = self.I(lambda x: x, self.data.ATR)
        self.bullish_div = self.I(lambda x: x, self.data.bullish_div)
        self.bearish_div = self.I(lambda x: x, self.data.bearish_div)
        
        # Position management variables
        self.entry_price = None
        self.trailing_stop = None
        self.take_profit = None
        self.highest = None
        self.lowest = None
    
    def next(self):
        price = self.data.Close[-1]
        atr_val = self.atr[-1]
        
        # Manage open positions
        if self.position:
            if self.position.is_long:
                # Update highest price since entry
                if self.highest is None or price > self.highest:
                    self.highest = price
                
                # Update trailing stop
                trail_dist = max(ATR_MULTIPLIER * atr_val, price * STOP_LOSS_PCT)
                new_stop = self.highest - trail_dist
                if self.trailing_stop is None or new_stop > self.trailing_stop:
                    self.trailing_stop = new_stop
                
                # Check exit conditions
                if price <= self.trailing_stop:
                    self.position.close()
                    self.trailing_stop = None
                    self.take_profit = None
                    self.highest = None
                    return
                
                if self.take_profit is not None and price >= self.take_profit:
                    self.position.close()
                    self.trailing_stop = None
                    self.take_profit = None
                    self.highest = None
                    return
            
            elif self.position.is_short:
                # Update lowest price since entry
                if self.lowest is None or price < self.lowest:
                    self.lowest = price
                
                # Update trailing stop
                trail_dist = max(ATR_MULTIPLIER * atr_val, price * STOP_LOSS_PCT)
                new_stop = self.lowest + trail_dist
                if self.trailing_stop is None or new_stop < self.trailing_stop:
                    self.trailing_stop = new_stop
                
                # Check exit conditions
                if price >= self.trailing_stop:
                    self.position.close()
                    self.trailing_stop = None
                    self.take_profit = None
                    self.lowest = None
                    return
                
                if self.take_profit is not None and price <= self.take_profit:
                    self.position.close()
                    self.trailing_stop = None
                    self.take_profit = None
                    self.lowest = None
                    return
            return
        
        # Entry logic
        if self.bullish_div[-1]:
            # Long entry
            self.buy(size=0.95)
            self.entry_price = price
            self.highest = price
            
            # Set up stop loss and take profit
            trail_dist = max(ATR_MULTIPLIER * atr_val, price * STOP_LOSS_PCT)
            self.trailing_stop = price - trail_dist
            self.take_profit = price * (1 + PROFIT_TARGET_PCT)
            
            logging.info(f"LONG entry at {self.data.index[-1]}, price={price:.2f}, stop={self.trailing_stop:.2f}, target={self.take_profit:.2f}")
        
        elif self.bearish_div[-1]:
            # Short entry
            self.sell(size=0.95)
            self.entry_price = price
            self.lowest = price
            
            # Set up stop loss and take profit
            trail_dist = max(ATR_MULTIPLIER * atr_val, price * STOP_LOSS_PCT)
            self.trailing_stop = price + trail_dist
            self.take_profit = price * (1 - PROFIT_TARGET_PCT)
            
            logging.info(f"SHORT entry at {self.data.index[-1]}, price={price:.2f}, stop={self.trailing_stop:.2f}, target={self.take_profit:.2f}")

def combine_data_files(directory_path, pattern='btc_5m_*.csv'):
    """Combine multiple CSV files into a single DataFrame"""
    import glob
    import os
    
    # Get list of all matching files
    file_pattern = os.path.join(directory_path, pattern)
    files = glob.glob(file_pattern)
    
    if not files:
        raise ValueError(f"No files found matching pattern {pattern} in {directory_path}")
    
    logging.info(f"Found {len(files)} data files to combine")
    
    # Initialize empty list to store dataframes
    dfs = []
    
    # Process each file
    for file in files:
        logging.info(f"Processing {os.path.basename(file)}...")
        try:
            # Load the file
            df = pd.read_csv(file)
            
            # Check if we have a timestamp/date column
            date_col = None
            for col in df.columns:
                if 'time' in col.lower() or 'date' in col.lower():
                    date_col = col
                    break
            
            if date_col:
                # Convert to datetime
                df[date_col] = pd.to_datetime(df[date_col])
                df.set_index(date_col, inplace=True)
            
            # Standardize column names
            col_map = {}
            for col in df.columns:
                col_lower = col.lower()
                if 'open' in col_lower:
                    col_map[col] = 'Open'
                elif 'high' in col_lower:
                    col_map[col] = 'High'
                elif 'low' in col_lower:
                    col_map[col] = 'Low'
                elif 'close' in col_lower:
                    col_map[col] = 'Close'
                elif 'volume' in col_lower:
                    col_map[col] = 'Volume'
            
            # Rename columns
            df.rename(columns=col_map, inplace=True)
            
            # Append to list
            dfs.append(df)
            
        except Exception as e:
            logging.warning(f"Error processing {file}: {e}")
    
    if not dfs:
        raise ValueError("No valid data files could be processed")
    
    # Combine all dataframes
    combined_df = pd.concat(dfs)
    
    # Sort by index (timestamp)
    combined_df = combined_df.sort_index()
    
    # Remove duplicates
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
    
    logging.info(f"Combined data shape: {combined_df.shape}")
    
    return combined_df

# --- Main function ---
def main():
    # Directory containing all data files
    data_dir = r'F:\Master_Data\market_data\btc\5m'
    
    try:
        # Combine all data files
        combined_data = combine_data_files(data_dir)
        
        # Prepare the combined data
        data = prepare_data_from_df(combined_data)
        
        # Run backtest with more trades
        logging.info("Setting up backtest...")
        bt = Backtest(data, RSIDivergenceStrategy, cash=INITIAL_CASH, commission=COMMISSION_RATE, trade_on_close=True, hedging=True)
        
        logging.info("Running backtest...")
        stats = bt.run()
        
        # Print results
        print("\n--- Backtest Results ---")
        print(stats)
        
        # Save results to file
        with open(RESULTS_FILENAME, "w") as f:
            f.write("Backtest Results for RSI Divergence Strategy\n")
            f.write(f"Data: Combined BTC 5m data from {data_dir}\n")
            f.write(f"Initial Cash: {INITIAL_CASH}\n")
            f.write(f"Commission: {COMMISSION_RATE}\n\n")
            f.write(stats.to_string())
        
        logging.info(f"Results saved to {RESULTS_FILENAME}")
        
        # Plot results
        logging.info("Plotting results...")
        bt.plot()
        
    except Exception as e:
        logging.error(f"Error in main function: {e}")

if __name__ == "__main__":
    main()
