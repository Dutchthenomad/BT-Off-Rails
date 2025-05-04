import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import logging
from scipy.signal import argrelextrema
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# --- Configuration ---
INITIAL_CASH = 1000000.0  # Increased to handle BTC price levels
COMMISSION_RATE = 0.001  # 0.1%
RESULTS_FILENAME = "basic_rsi_div_results.txt"

# --- Strategy Parameters ---
RSI_LENGTH = 6  # Optimized value
ATR_LENGTH = 6  # Optimized value
ATR_MULTIPLIER = 1.1  # Optimized value
PROFIT_TARGET_PCT = 0.01  # Optimized value
STOP_LOSS_PCT = 0.008  # Optimized value
SWING_WINDOW = 3  # Optimized value

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
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Rename columns to match Backtest.py requirements
        df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }, inplace=True)
        
        # Convert to numeric
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
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

# --- Main Function ---
def main():
    try:
        # File path
        file_path = r'F:\Master_Data\market_data\btc\5m\btc_5m_20240101_to_20250415.csv'
        
        # Load data
        data = load_data(file_path)
        
        # Prepare data
        data = prepare_data(data)
        
        # Run backtest
        logging.info("Running backtest...")
        bt = Backtest(data, RSIDivergenceStrategy, cash=INITIAL_CASH, commission=COMMISSION_RATE)
        stats = bt.run()
        
        # Print results
        print("\n--- Backtest Results ---")
        print(stats)
        
        # Save results to file
        with open(RESULTS_FILENAME, "w") as f:
            f.write("Backtest Results for Basic RSI Divergence Strategy\n")
            f.write(f"Data: {file_path}\n")
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
