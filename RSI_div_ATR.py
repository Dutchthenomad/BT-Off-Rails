import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import logging
from scipy.signal import argrelextrema
import itertools

# --- Configuration ---
DATA_PATH = r'F:\Master_Data\market_data\btc\5m\btc_5m_20240101_to_20250415.csv'
INITIAL_CASH = 50000.0
COMMISSION_RATE = 0.001  # 0.1%
RESULTS_FILENAME = "rsi_divergence_backtest_results.txt"

# --- Parameters (can be optimized) ---
# More sensitive parameters to generate more trading signals
RSI_LENGTH = 6  # Optimized from 14
ATR_LENGTH = 6  # Optimized from 14
ATR_MULTIPLIER = 1.1  # Optimized from 1.5
PROFIT_TARGET_PCT = 0.01  # Keep original
STOP_LOSS_PCT = 0.008  # Optimized from 0.01
SWING_WINDOW = 3  # Optimized from 5
ALIGN_WINDOW = 3  # Keep original

# --- Batch Optimization Parameters ---
BATCH_OPTIMIZE = False  # Set to False for final test
OPTIMIZE_RESULTS_CSV = "rsi_divergence_optimization_results.csv"

PARAM_GRID = {
    'RSI_LENGTH': [6, 8, 10, 12],
    'ATR_LENGTH': [6, 8, 10, 12],
    'ATR_MULTIPLIER': [1.1, 1.2, 1.3, 1.4],
    'PROFIT_TARGET_PCT': [0.01, 0.012, 0.014],
    'STOP_LOSS_PCT': [0.008, 0.009, 0.011],
    'SWING_WINDOW': [2, 3, 4],
}

# --- Manual Indicator Calculation ---
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

def calculate_atr(high, low, close, length=14):
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr = tr.rolling(window=length).mean()
    return atr

def compute_indicators(df):
    logging.info("Computing indicators...")
    try:
        df['RSI'] = calculate_rsi(df['Close'], length=RSI_LENGTH)
    except Exception as e:
        logging.error(f"Error calculating RSI: {e}")
    try:
        df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'], length=ATR_LENGTH)
    except Exception as e:
        logging.error(f"Error calculating ATR: {e}")
    detect_rsi_divergence(df, SWING_WINDOW, ALIGN_WINDOW)
    df['signal'] = 0
    df.loc[df['bullish_div'], 'signal'] = 1
    df['rsi_value_at_signal'] = np.nan
    df.loc[df['signal'] == 1, 'rsi_value_at_signal'] = df.loc[df['signal'] == 1, 'RSI']
    return df

# --- Robust Local Extrema Detection ---
def find_local_extrema(series, order=5, mode='max'):
    """
    Use scipy.signal.argrelextrema to find local maxima/minima.
    Returns a boolean array where True indicates a local max/min.
    """
    if mode == 'max':
        idx = argrelextrema(series.values, np.greater_equal, order=order)[0]
    else:
        idx = argrelextrema(series.values, np.less_equal, order=order)[0]
    extrema = np.zeros(series.shape, dtype=bool)
    extrema[idx] = True
    return pd.Series(extrema, index=series.index)

def detect_rsi_divergence(df, swing_window=SWING_WINDOW, align_window=ALIGN_WINDOW):
    """
    Optimized robust divergence detection for large datasets.
    Lower swing_window and align_window values will generate more signals.
    """
    # Find local extrema in price and RSI
    df['local_max'] = find_local_extrema(df['Close'], order=swing_window, mode='max')
    df['local_min'] = find_local_extrema(df['Close'], order=swing_window, mode='min')
    df['rsi_local_max'] = find_local_extrema(df['RSI'], order=swing_window, mode='max')
    df['rsi_local_min'] = find_local_extrema(df['RSI'], order=swing_window, mode='min')
    df['bullish_div'] = False
    df['bearish_div'] = False
    # For each divergence point, find the next two local extrema in price and RSI
    for i in range(len(df) - align_window):
        if df.iloc[i]['local_min'] and df.iloc[i:i+align_window]['rsi_local_min'].any():
            price_idx = i
            rsi_idx = i + df.iloc[i:i+align_window]['rsi_local_min'].idxmax() - i
            
            # Check if price is making lower low but RSI is making higher low (bullish div)
            if price_idx > swing_window and rsi_idx > swing_window:
                price_prev_idx = df.iloc[max(0, price_idx-swing_window*2):price_idx]['local_min'].idxmax()
                rsi_prev_idx = df.iloc[max(0, rsi_idx-swing_window*2):rsi_idx]['rsi_local_min'].idxmax()
                
                if (df.iloc[price_idx]['Close'] < df.iloc[price_prev_idx]['Close'] and 
                    df.iloc[rsi_idx]['RSI'] > df.iloc[rsi_prev_idx]['RSI']):
                    # Calculate divergence strength and store it
                    df.loc[price_idx, 'div_strength'] = (df.iloc[rsi_idx]['RSI'] - df.iloc[rsi_prev_idx]['RSI']) / \
                               (df.iloc[price_prev_idx]['Close'] - df.iloc[price_idx]['Close']) * 100
                    df.loc[price_idx, 'bullish_div'] = True
        
        if df.iloc[i]['local_max'] and df.iloc[i:i+align_window]['rsi_local_max'].any():
            price_idx = i
            rsi_idx = i + df.iloc[i:i+align_window]['rsi_local_max'].idxmax() - i
            
            # Check if price is making higher high but RSI is making lower high (bearish div)
            if price_idx > swing_window and rsi_idx > swing_window:
                price_prev_idx = df.iloc[max(0, price_idx-swing_window*2):price_idx]['local_max'].idxmax()
                rsi_prev_idx = df.iloc[max(0, rsi_idx-swing_window*2):rsi_idx]['rsi_local_max'].idxmax()
                
                if (df.iloc[price_idx]['Close'] > df.iloc[price_prev_idx]['Close'] and 
                    df.iloc[rsi_idx]['RSI'] < df.iloc[rsi_prev_idx]['RSI']):
                    # Calculate divergence strength and store it
                    df.loc[price_idx, 'div_strength'] = (df.iloc[rsi_prev_idx]['RSI'] - df.iloc[rsi_idx]['RSI']) / \
                               (df.iloc[price_idx]['Close'] - df.iloc[price_prev_idx]['Close']) * 100
                    df.loc[price_idx, 'bearish_div'] = True
    return df

# --- Load and Prepare Data ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logging.info(f"Loading data from {DATA_PATH}...")
try:
    # Load data from CSV format with specific timestamp parsing
    data = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
    
    # Rename columns to match Backtest.py requirements (OHLC format)
    data.rename(columns={
        'timestamp': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    
    # Set index to datetime
    data.set_index('Date', inplace=True)
    
    # Convert to numeric and clean
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    data.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
    logging.info(f"Data loaded successfully. Shape: {data.shape}")
except FileNotFoundError:
    logging.critical(f"Error: Data file not found at {DATA_PATH}")
    exit()
except Exception as e:
    logging.critical(f"Error loading or processing data: {e}", exc_info=True)
    exit()

if BATCH_OPTIMIZE:
    results = []
    param_names = list(PARAM_GRID.keys())
    for values in itertools.product(*[PARAM_GRID[k] for k in param_names]):
        # Set parameters
        params = dict(zip(param_names, values))
        # Copy and prepare data
        try:
            # Load data with the same approach as the main code
            df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
            
            # Rename columns to match Backtest.py requirements (OHLC format)
            df.rename(columns={
                'timestamp': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }, inplace=True)
            
            # Set index to datetime
            df.set_index('Date', inplace=True)
        except Exception as e:
            logging.error(f"Error loading data for optimization: {e}")
            continue
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
        # Compute indicators and divergences using manual calculations
        try:
            df['RSI'] = calculate_rsi(df['Close'], length=params['RSI_LENGTH'])
            df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'], length=params['ATR_LENGTH'])
            df = detect_rsi_divergence(df, swing_window=params['SWING_WINDOW'], align_window=3)
        except Exception as e:
            logging.error(f"Error in optimization loop: {e}")
            continue
        df.dropna(inplace=True)
        # Define strategy class dynamically
        class OptStrategy(Strategy):
            rsi_length = params['RSI_LENGTH']
            atr_length = params['ATR_LENGTH']
            atr_multiplier = params['ATR_MULTIPLIER']
            profit_target_pct = params['PROFIT_TARGET_PCT']
            stop_loss_pct = params['STOP_LOSS_PCT']
            swing_window = params['SWING_WINDOW']
            def init(self):
                self.rsi = self.I(lambda x: x, self.data.RSI, name="RSI")
                self.atr = self.I(lambda x: x, self.data.ATR, name="ATR")
                self.bullish_div = self.I(lambda x: x, self.data.bullish_div, name="bullish_div")
                self.bearish_div = self.I(lambda x: x, self.data.bearish_div, name="bearish_div")
                self.active_stop = None
                self.active_target = None
            def next(self):
                price = self.data.Close[-1]
                atr_val = self.atr[-1]
                if self.position:
                    if self.position.is_long:
                        if price <= self.active_stop or price >= self.active_target:
                            self.position.close()
                            self.active_stop = None
                            self.active_target = None
                            return
                    if self.position.is_short:
                        if price >= self.active_stop or price <= self.active_target:
                            self.position.close()
                            self.active_stop = None
                            self.active_target = None
                            return
                    return
                if self.bullish_div[-1]:
                    self.buy(size=0.95)
                    self.active_stop = price - max(self.atr_multiplier * atr_val, price * self.stop_loss_pct)
                    self.active_target = price + price * self.profit_target_pct
                elif self.bearish_div[-1]:
                    self.sell(size=0.95)
                    self.active_stop = price + max(self.atr_multiplier * atr_val, price * self.stop_loss_pct)
                    self.active_target = price - price * self.profit_target_pct
        # Run backtest
        bt = Backtest(df, OptStrategy, cash=INITIAL_CASH, commission=COMMISSION_RATE)
        stats = bt.run()
        results.append({**params,
            'CAGR': stats['CAGR [%]'],
            'Sharpe': stats['Sharpe Ratio'],
            'MaxDD': stats['Max. Drawdown [%]'],
            'WinRate': stats['Win Rate [%]'],
            'ProfitFactor': stats['Profit Factor'],
            'Trades': stats['# Trades'],
            'EquityFinal': stats['Equity Final [$]'],
        })
    # Save results
    pd.DataFrame(results).to_csv(OPTIMIZE_RESULTS_CSV, index=False)
    print(f"Batch optimization complete. Results saved to {OPTIMIZE_RESULTS_CSV}")
    exit()

# --- Compute Indicators and Divergence ---
data = compute_indicators(data)

# Log number of detected extrema and divergences
def count_true(series):
    return int(series.sum()) if hasattr(series, 'sum') else 0
logging.info(f"local_max: {count_true(data['local_max'])}")
logging.info(f"local_min: {count_true(data['local_min'])}")
logging.info(f"rsi_local_max: {count_true(data['rsi_local_max'])}")
logging.info(f"rsi_local_min: {count_true(data['rsi_local_min'])}")
bullish_count = data['bullish_div'].sum()
bearish_count = data['bearish_div'].sum()
logging.info(f"Detected bullish divergences: {bullish_count}")
logging.info(f"Detected bearish divergences: {bearish_count}")

# Drop initial rows with NaNs from indicator calculations
data.dropna(inplace=True)

# --- Strategy Definition ---
class RSIDivergenceStrategy(Strategy):
    rsi_length = RSI_LENGTH
    atr_length = ATR_LENGTH
    atr_multiplier = ATR_MULTIPLIER
    profit_target_pct = PROFIT_TARGET_PCT
    stop_loss_pct = STOP_LOSS_PCT
    swing_window = SWING_WINDOW

    def init(self):
        self.rsi = self.I(lambda x: x, self.data.RSI, name="RSI")
        self.atr = self.I(lambda x: x, self.data.ATR, name="ATR")
        self.bullish_div = self.I(lambda x: x, self.data.bullish_div, name="bullish_div")
        self.bearish_div = self.I(lambda x: x, self.data.bearish_div, name="bearish_div")
        self.entry_price = None
        self.trailing_stop = None
        self.active_target = None
        self.highest = None
        self.lowest = None

    def next(self):
        price = self.data.Close[-1]
        atr_val = self.atr[-1]
        # Manage open position
        if self.position:
            if self.position.is_long:
                # Update highest price since entry
                if self.highest is None or price > self.highest:
                    self.highest = price
                # Trailing stop logic
                trail_dist = max(self.atr_multiplier * atr_val, price * self.stop_loss_pct)
                new_stop = self.highest - trail_dist
                if self.trailing_stop is None or new_stop > self.trailing_stop:
                    self.trailing_stop = new_stop
                # Exit conditions
                if price <= self.trailing_stop:
                    self.position.close()
                    self.trailing_stop = None
                    self.active_target = None
                    self.highest = None
                    return
                if price >= self.active_target:
                    self.position.close()
                    self.trailing_stop = None
                    self.active_target = None
                    self.highest = None
                    return
            if self.position.is_short:
                # Update lowest price since entry
                if self.lowest is None or price < self.lowest:
                    self.lowest = price
                # Trailing stop logic
                trail_dist = max(self.atr_multiplier * atr_val, price * self.stop_loss_pct)
                new_stop = self.lowest + trail_dist
                if self.trailing_stop is None or new_stop < self.trailing_stop:
                    self.trailing_stop = new_stop
                # Exit conditions
                if price >= self.trailing_stop:
                    self.position.close()
                    self.trailing_stop = None
                    self.active_target = None
                    self.lowest = None
                    return
                if price <= self.active_target:
                    self.position.close()
                    self.trailing_stop = None
                    self.active_target = None
                    self.lowest = None
                    return
            return
        # Entry logic
        if self.bullish_div[-1]:
            logging.info(f"Long entry signal at {self.data.index[-1]}, price={price}")
            self.buy(size=0.95)
            self.entry_price = price
            self.highest = price
            trail_dist = max(self.atr_multiplier * atr_val, price * self.stop_loss_pct)
            self.trailing_stop = price - trail_dist
            self.active_target = price + price * self.profit_target_pct
        elif self.bearish_div[-1]:
            logging.info(f"Short entry signal at {self.data.index[-1]}, price={price}")
            self.sell(size=0.95)
            self.entry_price = price
            self.lowest = price
            trail_dist = max(self.atr_multiplier * atr_val, price * self.stop_loss_pct)
            self.trailing_stop = price + trail_dist
            self.active_target = price - price * self.profit_target_pct

# --- Run Backtest ---
logging.info("Setting up Backtest...")
bt = Backtest(data, RSIDivergenceStrategy, cash=INITIAL_CASH, commission=COMMISSION_RATE)
logging.info("Running Backtest...")
stats = bt.run()
logging.info("Backtest finished.")

# --- Print and Save Results ---
print("\n--- Backtest Results ---")
print(stats)
try:
    with open(RESULTS_FILENAME, "w") as f:
        f.write(f"Backtest Results for RSIDivergenceStrategy\n")
        f.write(f"Data: {DATA_PATH}\n")
        f.write(f"Initial Cash: {INITIAL_CASH}\n")
        f.write(f"Commission: {COMMISSION_RATE}\n\n")
        stats_str = stats.to_string()
        f.write(stats_str)
    logging.info(f"Results saved to {RESULTS_FILENAME}")
except Exception as e:
    logging.error(f"Error saving results to file: {e}")

# --- Plot Results ---
logging.info("Plotting results...")
try:
    bt.plot()
    logging.info("Plotting complete. Check the generated HTML file.")
except Exception as e:
    logging.warning(f"Could not generate plot: {e}")
