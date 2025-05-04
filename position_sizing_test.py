import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

# Constants and parameters
RSI_LENGTH = 6
ATR_LENGTH = 6
ATR_MULTIPLIER = 1.1
PROFIT_TARGET_PCT = 0.01
STOP_LOSS_PCT = 0.008
SWING_WINDOW = 3
INITIAL_CAPITAL = 1_000_000
COMMISSION_PCT = 0.001  # 0.1%

# Position sizing parameters
BASE_POSITION_SIZE = 0.01  # Base position size as percentage of equity
VOLATILITY_FACTOR = 1.0    # Scaling factor for volatility-based sizing
MAX_POSITION_SIZE = 0.05   # Maximum position size as percentage of equity

# Data paths
DATA_PATH = "F:\\Master_Data\\market_data"
RESULTS_DIR = "results"

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# Function to calculate RSI
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

# Function to calculate ATR
def calculate_atr(df, length=14):
    # Calculate True Range
    tr1 = df['High'] - df['Low']
    tr2 = abs(df['High'] - df['Close'].shift(1))
    tr3 = abs(df['Low'] - df['Close'].shift(1))
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

# Function to run manual backtest with position sizing
def run_manual_backtest(df, rsi_length=RSI_LENGTH, atr_length=ATR_LENGTH, 
                       swing_window=SWING_WINDOW, profit_target_pct=PROFIT_TARGET_PCT,
                       stop_loss_pct=STOP_LOSS_PCT, initial_capital=INITIAL_CAPITAL,
                       commission_pct=COMMISSION_PCT):
    
    # Calculate indicators
    df['RSI'] = calculate_rsi(df['Close'], length=rsi_length)
    df['ATR'] = calculate_atr(df, length=atr_length)
    df['Volatility'] = df['ATR'] / df['Close']
    
    # Detect divergences
    df = detect_rsi_divergence(df, swing_window=swing_window)
    
    # Initialize backtest variables
    equity = initial_capital
    position = 0  # 0 = no position, 1 = long position
    entry_price = 0
    stop_loss_price = 0
    take_profit_price = 0
    trades = []
    equity_curve = [initial_capital]
    
    # Position sizing parameters
    base_position_size = BASE_POSITION_SIZE
    volatility_factor = VOLATILITY_FACTOR
    max_position_size = MAX_POSITION_SIZE
    
    # Loop through data
    for i in range(swing_window * 5, len(df)):
        current_date = df.index[i]
        current_price = df['Close'].iloc[i]
        current_volatility = df['Volatility'].iloc[i]
        
        # Update equity curve
        if position == 1:  # Long position
            # Calculate current equity including unrealized P&L
            current_equity = equity + (position_size * (current_price - entry_price))
            equity_curve.append(current_equity)
        else:
            equity_curve.append(equity)
        
        # Check for exit if in a position
        if position == 1:  # Long position
            # Check stop loss
            if df['Low'].iloc[i] <= stop_loss_price:
                # Close position at stop loss price
                exit_price = stop_loss_price
                pnl = position_size * (exit_price - entry_price)
                commission = position_size * exit_price * commission_pct
                equity += pnl - commission
                
                # Record trade
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position_size': position_size,
                    'pnl': pnl,
                    'commission': commission,
                    'exit_reason': 'stop_loss'
                })
                
                # Reset position
                position = 0
            
            # Check take profit
            elif df['High'].iloc[i] >= take_profit_price:
                # Close position at take profit price
                exit_price = take_profit_price
                pnl = position_size * (exit_price - entry_price)
                commission = position_size * exit_price * commission_pct
                equity += pnl - commission
                
                # Record trade
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position_size': position_size,
                    'pnl': pnl,
                    'commission': commission,
                    'exit_reason': 'take_profit'
                })
                
                # Reset position
                position = 0
        
        # Check for entry if not in a position
        if position == 0:
            # Check for bullish divergence
            if df['bullish_div'].iloc[i] == 1:
                # Calculate position size based on volatility
                if current_volatility < 0.001:
                    current_volatility = 0.001  # Avoid division by zero
                
                # Calculate position size inversely proportional to volatility
                position_size_pct = base_position_size / (current_volatility * 100 * volatility_factor)
                position_size_pct = min(position_size_pct, max_position_size)  # Cap at maximum
                
                # Calculate actual position size
                position_size = (equity * position_size_pct) / current_price
                
                # Enter long position
                entry_price = current_price
                entry_date = current_date
                position = 1
                
                # Calculate stop loss and take profit prices
                stop_loss_price = entry_price * (1 - stop_loss_pct)
                take_profit_price = entry_price * (1 + profit_target_pct)
                
                # Apply commission
                commission = position_size * entry_price * commission_pct
                equity -= commission
    
    # Close any open position at the end
    if position == 1:
        exit_price = df['Close'].iloc[-1]
        pnl = position_size * (exit_price - entry_price)
        commission = position_size * exit_price * commission_pct
        equity += pnl - commission
        
        # Record trade
        trades.append({
            'entry_date': entry_date,
            'exit_date': df.index[-1],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_size': position_size,
            'pnl': pnl,
            'commission': commission,
            'exit_reason': 'end_of_data'
        })
    
    # Calculate performance metrics
    trades_df = pd.DataFrame(trades)
    
    if len(trades_df) > 0:
        # Calculate win rate
        trades_df['is_win'] = trades_df['pnl'] > 0
        win_rate = trades_df['is_win'].mean()
        
        # Calculate profit factor
        total_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        total_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Calculate return
        total_return = (equity - initial_capital) / initial_capital
        
        # Calculate max drawdown
        equity_series = pd.Series(equity_curve)
        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
    else:
        win_rate = 0
        profit_factor = 0
        total_return = 0
        max_drawdown = 0
    
    # Print results
    print(f"\nBacktest Results:")
    print(f"Total Return: {total_return:.2%}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Number of Trades: {len(trades)}")
    
    # Plot equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve)
    plt.title('Equity Curve')
    plt.xlabel('Trade Number')
    plt.ylabel('Equity')
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, 'equity_curve.png'))
    plt.close()
    
    # Save trades to CSV
    if len(trades_df) > 0:
        trades_df.to_csv(os.path.join(RESULTS_DIR, 'trades.csv'))
    
    # Return results
    return {
        'total_return': total_return,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'num_trades': len(trades),
        'equity_curve': equity_curve,
        'trades': trades
    }

# Function to analyze position sizing effectiveness
def analyze_position_sizing(trades):
    if not trades or len(trades) == 0:
        print("No trades to analyze")
        return
    
    trades_df = pd.DataFrame(trades)
    
    # Calculate position size as percentage of equity at entry
    trades_df['position_size_pct'] = trades_df['position_size'] * trades_df['entry_price'] / INITIAL_CAPITAL
    
    # Group trades by position size buckets
    trades_df['position_size_bucket'] = pd.cut(trades_df['position_size_pct'], 
                                             bins=[0, 0.01, 0.02, 0.03, 0.04, 0.05, 1.0],
                                             labels=['0-1%', '1-2%', '2-3%', '3-4%', '4-5%', '>5%'])
    
    # Analyze performance by position size
    size_analysis = trades_df.groupby('position_size_bucket').agg({
        'pnl': ['sum', 'mean', 'count'],
        'is_win': 'mean'
    })
    
    print("\nPosition Sizing Analysis:")
    print(size_analysis)
    
    # Plot position size vs. return
    plt.figure(figsize=(10, 6))
    plt.scatter(trades_df['position_size_pct'], trades_df['pnl'] / trades_df['position_size'] / trades_df['entry_price'])
    plt.title('Position Size vs. Return')
    plt.xlabel('Position Size (% of Equity)')
    plt.ylabel('Return (%)')
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, 'position_size_vs_return.png'))
    plt.close()
    
    return size_analysis

# Main execution
if __name__ == "__main__":
    # Load data for BTC
    print("Loading BTC data...")
    df = load_crypto_data("BTC", timeframe="5m")
    
    if df is not None:
        print(f"Data loaded: {len(df)} rows from {df.index[0]} to {df.index[-1]}")
        
        # Run backtest with position sizing
        print("\nRunning backtest with position sizing...")
        results = run_manual_backtest(df)
        
        # Analyze position sizing effectiveness
        if results['trades']:
            print("\nAnalyzing position sizing effectiveness...")
            analyze_position_sizing(results['trades'])
        
        print("\nBacktesting complete. Results saved to the 'results' directory.")
    else:
        print("Failed to load data. Exiting.")
