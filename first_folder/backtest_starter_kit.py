"""
Backtest using BTC/USDT 5m data from the Master Data Wrapper
Simple RSI strategy using backtesting.py with manual indicator calculation
"""
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy

# --- Configuration ---
DATA_PATH = r'F:\Master_Data\market_data\btc\5m\btc_5m_20240101_to_20250415.csv'
INITIAL_CASH = 50000.0
COMMISSION_RATE = 0.001  # 0.1%

# --- Parameters ---
RSI_LENGTH = 6
ATR_LENGTH = 6
ATR_MULTIPLIER = 1.1
PROFIT_TARGET_PCT = 0.01  # 1.0%
STOP_LOSS_PCT = 0.008  # 0.8%

# --- Load and Prepare Data ---
print(f"Loading data from {DATA_PATH}...")
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
print(f"Loaded {len(data)} rows of data")

# --- Manual indicator calculations ---
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

def calculate_atr(high, low, close, length=14):
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    
    # Calculate Average True Range
    atr = tr.rolling(window=length).mean()
    return atr

# Calculate indicators
print("Calculating indicators...")
data['RSI'] = calculate_rsi(data['Close'], length=RSI_LENGTH)
data['ATR'] = calculate_atr(data['High'], data['Low'], data['Close'], length=ATR_LENGTH)

# Remove NaN values
data.dropna(inplace=True)
print(f"Data ready with {len(data)} rows after removing NaN values")

# --- Simple RSI Strategy ---
class SimpleRSIStrategy(Strategy):
    def init(self):
        # Make indicators available to the strategy
        self.rsi = self.I(lambda x: x, self.data.RSI)
        self.atr = self.I(lambda x: x, self.data.ATR)
        
    def next(self):
        price = self.data.Close[-1]
        
        # Exit logic (if we have a position)
        if self.position:
            if self.position.is_long and price >= self.position.entry_price * (1 + PROFIT_TARGET_PCT):
                self.position.close()
            elif self.position.is_long and price <= self.position.entry_price * (1 - STOP_LOSS_PCT):
                self.position.close()
            elif self.position.is_short and price <= self.position.entry_price * (1 - PROFIT_TARGET_PCT):
                self.position.close()
            elif self.position.is_short and price >= self.position.entry_price * (1 + STOP_LOSS_PCT):
                self.position.close()
            return
        
        # Entry logic
        if self.rsi[-1] < 30:  # Oversold - potential buy signal
            self.buy(size=0.95)
        elif self.rsi[-1] > 70:  # Overbought - potential sell signal
            self.sell(size=0.95)

# --- Run Backtest ---
print("Running backtest...")

try:
    # Create and run backtest
    bt = Backtest(data, SimpleRSIStrategy, cash=INITIAL_CASH, commission=COMMISSION_RATE)
    stats = bt.run()
    print("\nBacktest Results:\n")
    print(stats)
    
    # Try to plot results
    try:
        print("\nGenerating plot...")
        bt.plot(plot_volume=True, plot_return=True)
    except Exception as plot_error:
        print(f"Error generating plot: {plot_error}")
        print("Trying alternative plot settings...")
        try:
            bt.plot(plot_volume=False, plot_return=False)
        except Exception as alt_plot_error:
            print(f"Alternative plot also failed: {alt_plot_error}")
            print("Showing stats only.")

except Exception as backtest_error:
    print(f"Error running backtest: {backtest_error}")