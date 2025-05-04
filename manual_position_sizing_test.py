import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# --- Configuration ---
INITIAL_CAPITAL = 1000000.0
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

# --- Manual Backtest ---
def manual_backtest(df):
    # Initialize variables
    equity = INITIAL_CAPITAL
    position = False
    position_size = 0
    entry_price = 0
    entry_date = None
    stop_loss_price = 0
    take_profit_price = 0
    trades = []
    equity_curve = [INITIAL_CAPITAL]
    
    # Loop through data
    for i in range(SWING_WINDOW * 5, len(df)):
        current_date = df.index[i]
        current_price = df['Close'].iloc[i]
        
        # Update equity curve
        if position:
            # Calculate current equity including unrealized P&L
            current_equity = equity + (position_size * (current_price - entry_price)) - (position_size * current_price * COMMISSION_RATE)
            equity_curve.append(current_equity)
        else:
            equity_curve.append(equity)
        
        # Check for exit if in a position
        if position:
            # Check stop loss
            if df['Low'].iloc[i] <= stop_loss_price:
                # Close position at stop loss price
                exit_price = stop_loss_price
                pnl = position_size * (exit_price - entry_price)
                commission = position_size * exit_price * COMMISSION_RATE
                equity += pnl - commission
                
                # Record trade
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position_size': position_size,
                    'position_size_pct': position_size * entry_price / INITIAL_CAPITAL,
                    'pnl': pnl,
                    'return_pct': pnl / (position_size * entry_price),
                    'commission': commission,
                    'exit_reason': 'stop_loss',
                    'volatility': df['Volatility'].iloc[i-1]
                })
                
                # Reset position
                position = False
            
            # Check take profit
            elif df['High'].iloc[i] >= take_profit_price:
                # Close position at take profit price
                exit_price = take_profit_price
                pnl = position_size * (exit_price - entry_price)
                commission = position_size * exit_price * COMMISSION_RATE
                equity += pnl - commission
                
                # Record trade
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position_size': position_size,
                    'position_size_pct': position_size * entry_price / INITIAL_CAPITAL,
                    'pnl': pnl,
                    'return_pct': pnl / (position_size * entry_price),
                    'commission': commission,
                    'exit_reason': 'take_profit',
                    'volatility': df['Volatility'].iloc[i-1]
                })
                
                # Reset position
                position = False
        
        # Check for entry if not in a position
        if not position and df['bullish_div'].iloc[i] == 1:
            # Calculate position size based on volatility
            current_volatility = df['Volatility'].iloc[i]
            
            # Avoid division by zero or very small values
            if current_volatility < 0.001:
                current_volatility = 0.001
                
            # Calculate position size inversely proportional to volatility
            position_size_pct = BASE_POSITION_SIZE / (current_volatility * 100 * VOLATILITY_FACTOR)
            
            # Cap at maximum position size
            position_size_pct = min(position_size_pct, MAX_POSITION_SIZE)
            
            # Calculate actual position size
            position_size = (equity * position_size_pct) / current_price
            
            # Enter long position
            entry_price = current_price
            entry_date = current_date
            position = True
            
            # Calculate stop loss and take profit prices
            stop_loss_price = entry_price * (1 - STOP_LOSS_PCT)
            take_profit_price = entry_price * (1 + PROFIT_TARGET_PCT)
            
            # Apply commission
            commission = position_size * entry_price * COMMISSION_RATE
            equity -= commission
    
    # Close any open position at the end
    if position:
        exit_price = df['Close'].iloc[-1]
        pnl = position_size * (exit_price - entry_price)
        commission = position_size * exit_price * COMMISSION_RATE
        equity += pnl - commission
        
        # Record trade
        trades.append({
            'entry_date': entry_date,
            'exit_date': df.index[-1],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_size': position_size,
            'position_size_pct': position_size * entry_price / INITIAL_CAPITAL,
            'pnl': pnl,
            'return_pct': pnl / (position_size * entry_price),
            'commission': commission,
            'exit_reason': 'end_of_data',
            'volatility': df['Volatility'].iloc[-1]
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
        total_return = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL
        
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
    plt.xlabel('Bar Number')
    plt.ylabel('Equity')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'equity_curve.png'))
    plt.close()
    
    # Save trades to CSV
    if len(trades_df) > 0:
        trades_df.to_csv(os.path.join(RESULTS_DIR, 'trades.csv'))
        
        # Analyze position sizing
        print("\nPosition Sizing Analysis:")
        
        # Group trades by position size buckets
        trades_df['position_size_bucket'] = pd.cut(trades_df['position_size_pct'], 
                                                 bins=[0, 0.01, 0.02, 0.03, 0.04, 0.05, 1.0],
                                                 labels=['0-1%', '1-2%', '2-3%', '3-4%', '4-5%', '>5%'])
        
        # Analyze performance by position size
        size_analysis = trades_df.groupby('position_size_bucket').agg({
            'pnl': ['sum', 'mean'],
            'return_pct': 'mean',
            'is_win': 'mean',
            'pnl': 'count'
        })
        
        print(size_analysis)
        
        # Plot position size vs. return
        plt.figure(figsize=(10, 6))
        plt.scatter(trades_df['position_size_pct'] * 100, trades_df['return_pct'] * 100)
        plt.title('Position Size vs. Return')
        plt.xlabel('Position Size (% of Equity)')
        plt.ylabel('Return (%)')
        plt.grid(True)
        plt.savefig(os.path.join(RESULTS_DIR, 'position_size_vs_return.png'))
        plt.close()
        
        # Plot volatility vs. position size
        plt.figure(figsize=(10, 6))
        plt.scatter(trades_df['volatility'] * 100, trades_df['position_size_pct'] * 100)
        plt.title('Volatility vs. Position Size')
        plt.xlabel('Volatility (%)')
        plt.ylabel('Position Size (% of Equity)')
        plt.grid(True)
        plt.savefig(os.path.join(RESULTS_DIR, 'volatility_vs_position_size.png'))
        plt.close()
    
    return {
        'total_return': total_return,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'num_trades': len(trades),
        'equity_curve': equity_curve,
        'trades': trades
    }

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
    
    # Run manual backtest
    print(f"Running backtest for {symbol}...")
    results = manual_backtest(df)
    
    return results

# --- Main Function ---
if __name__ == "__main__":
    # Test with ETH (Ethereum)
    print("Testing with ETH...")
    eth_results = run_backtest("ETH")
    
    # Test with SOL (Solana)
    print("\nTesting with SOL...")
    sol_results = run_backtest("SOL")
    
    print("\nBacktesting complete. Results saved to the 'results' directory.")
