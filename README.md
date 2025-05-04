# BT-Off-Rails: Cryptocurrency Backtesting Framework

_This repository contains a collection of backtesting scripts and strategies, primarily developed for experimenting with the `backtesting.py` and `Freqtrade` platforms. While initially for personal exploration, these files are shared in the hope they might offer some value or starting points for those new to algorithmic crypto trading._

## _Overview_

BT-Off-Rails is a Python-based backtesting framework designed for developing, testing, and analyzing cryptocurrency trading strategies, with a primary focus on RSI (Relative Strength Index) divergence patterns and various position sizing algorithms.

It provides tools and examples for running simulations using both the `backtesting.py` library and the `Freqtrade` platform.

## Key Features

- **RSI Divergence Strategies:** Includes multiple implementations for detecting bullish and bearish RSI divergences.
- **Advanced Position Sizing:** Explore different position sizing models (simple, aggressive, volatility-based, manual testing scripts).
- **Multi-Platform Support:** Provides strategy examples compatible with:
  - `backtesting.py`: For straightforward backtesting and optimization.
  - `Freqtrade`: For more complex backtesting, hyperparameter optimization, and potential live deployment.
- **Multi-Coin Backtesting:** Scripts to run simulations across a portfolio of cryptocurrencies.
- **Data Handling:** Utilities for fetching and preparing cryptocurrency market data (requires user configuration/data sources).
- **Results Storage:** Saves backtest results, statistics, and potentially plots to dedicated directories.
- **Monte Carlo Simulation:** Includes tools for running Monte Carlo simulations to assess strategy robustness.

## Project Structure

```
backtesting_v1/
├── Master-Data-Wrapper/  # Potential data handling/fetching utilities (Needs verification)
├── first_folder/         # Example starter kit (Needs verification)
├── monte_carlo_results/  # Directory for storing Monte Carlo simulation outputs
├── results/              # Directory for storing general backtest results (backtesting.py)
├── results_aggressive/   # Directory for storing aggressive strategy results
├── *.py                  # Various Python scripts for strategies, backtests, utilities
├── README.md             # This file
├── setup.py              # Project setup and core dependencies
├── .gitignore            # Specifies intentionally untracked files
└── ...                   # Other data files, images, etc.
```

## Dependencies

Core dependencies are listed in `setup.py`. Additional libraries are required depending on which scripts you run:

- `pandas`
- `numpy`
- `ccxt`: For interacting with cryptocurrency exchanges (fetching data, etc.).
- `backtesting.py`: Required for scripts like `multi_coin_rsi_div.py`.
- `TA-Lib`: Required for technical indicators. **Note:** Requires specific installation steps (see below).
- `scipy`: Used for scientific and technical computing (e.g., finding extrema).
- `matplotlib`: Used for plotting results.
- `freqtrade`: Required _only_ if using `rsi_divergence_strategy.py` or similar Freqtrade-compatible strategies. This is a full trading bot framework.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Dutchthenomad/BT-Off-Rails.git
   cd BT-Off-Rails
   ```

2. **Set up a Python Environment:** (Recommended)

   ```bash
   python -m venv venv
   # Activate the environment (Windows example)
   .\venv\Scripts\activate
   ```

3. **Install TA-Lib:**

   - TA-Lib requires pre-installation of its underlying C library. Follow the instructions specific to your operating system:
     - **Windows:** Download `TA-Lib` wheel (`.whl`) file matching your Python version and system architecture (e.g., from [UCI Python Libraries](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)) and install using `pip install TA_Lib‑0.4.xx‑cp3x‑cp3x‑win_amd64.whl`.
     - **macOS:** `brew install ta-lib`
     - **Linux (Debian/Ubuntu):** `sudo apt-get install build-essential libta-lib-dev`
   - After installing the C library, install the Python wrapper:

     ```bash
     pip install TA-Lib
     ```

4. **Install Core Python Dependencies:**

   ```bash
   # Install the package itself and basic dependencies from setup.py
   pip install -e .

   # Install other common dependencies for backtesting.py usage
   pip install backtesting scipy matplotlib
   ```

5. **Install Freqtrade (Optional):**

   - If you plan to use the Freqtrade-compatible strategies (`rsi_divergence_strategy.py`), follow the official [Freqtrade installation guide](https://www.freqtrade.io/en/stable/installation/). This typically involves more setup steps.

## Usage

### Using `multi_coin_rsi_div.py` (Command-Line Backtester)

This script allows you to run backtests for the RSI Divergence strategy using the `backtesting.py` library directly from the command line. It's designed to be flexible, allowing you to specify data locations, symbols, output directories, and strategy parameters.

1.  **Prepare Data:** Ensure you have historical market data (e.g., 1-day candles) in CSV format. Each file should correspond to a single symbol (e.g., `BTCUSDT_1d.csv`). Place these files in a dedicated data directory.

2.  **Run the Backtest via Command Line:**

    ```bash
    python multi_coin_rsi_div.py --symbols <SYMBOL1> [SYMBOL2 ...] --data-dir <path/to/your/data> [options]
    ```

    **Required Arguments:**

    - `--symbols SYMBOL1 [SYMBOL2 ...]`: Specify one or more symbols to test (e.g., `BTC ETH DOGE`).
    - `--data-dir <path/to/your/data>`: Path to the directory containing your CSV data files.

    **Optional Arguments:**

    - `--results-dir <path>`: Directory to save results. (Default: `results`)
    - `--filename-pattern "{symbol}USDT_1d.csv"`: Pattern for data filenames (`{symbol}` placeholder). (Default: `"{symbol}USDT_1d.csv"`)
    - `--initial-cash <amount>`: Starting cash. (Default: 1,000,000)
    - `--commission <rate>`: Commission rate (e.g., 0.001). (Default: 0.001)
    - `--rsi-length <int>`: RSI period. (Default: 6)
    - `--atr-length <int>`: ATR period. (Default: 6)
    - `--atr-multiplier <float>`: ATR multiplier for dynamic SL (currently illustrative). (Default: 1.1)
    - `--profit-target <float>`: Profit target % (e.g., 0.01). (Default: 0.01)
    - `--stop-loss <float>`: Stop loss % (e.g., 0.008). (Default: 0.008)
    - `--swing-window <int>`: Window for finding extrema. (Default: 3)

    **Example (Reproducing notable results):**

    Assuming your data for DOGE, HYPE, etc., is in `./data` and follows the `DOGEUSDT_1d.csv` pattern:

    ```bash
    python multi_coin_rsi_div.py --symbols DOGE HYPE AI16Z XRP --data-dir ./data
    ```

    (This uses the default optimized parameters)

    **Example (Testing different parameters for BTC):**

    ```bash
    python multi_coin_rsi_div.py --symbols BTC --data-dir ./ohlcv --rsi-length 14 --stop-loss 0.01 --profit-target 0.02
    ```

3.  **Check Results:** Backtest statistics (`_results.txt`, `_combined_summary.txt`), plots (`_plot.html`), and trade lists (`_trades.csv`) will be saved in the specified results directory.

### Using `Freqtrade` Strategy (e.g., `rsi_divergence_strategy.py`)

1. **Set up Freqtrade:** Ensure Freqtrade is installed and configured according to its documentation.
2. **Place Strategy File:** Copy `rsi_divergence_strategy.py` into your Freqtrade `user_data/strategies/` directory.
3. **Configure Freqtrade:** Edit your Freqtrade `config.json` to select the `RsiDivergenceStrategy`, set the exchange, pairs, timeframe (`5m` recommended), and other parameters.
4. **Download Data:** Use Freqtrade commands to download the historical data needed for your configured pairs and timeframe.

   ```bash
   # Example Freqtrade command
   freqtrade download-data --exchange your_exchange -t 5m --pairs HYPE/USD SOL/USD XRP/USD ETH/USD BTC/USD
   ```

5. **Run Backtest via Freqtrade:**

   ```bash
   # Example Freqtrade command
   freqtrade backtesting --config config.json --strategy RsiDivergenceStrategy
   ```

6. **Analyze Results:** Freqtrade provides detailed backtesting reports.

## Included Strategies & Concepts

- **RSI Divergence:** The core concept involves identifying discrepancies between price movement and RSI oscillator movement.
  - _Bullish Divergence:_ Price makes lower lows, RSI makes higher lows (potential buy signal).
  - _Bearish Divergence:_ Price makes higher highs, RSI makes lower highs (potential sell/short signal).
- **Position Sizing Scripts:**
  - `simple_position_sizing.py`
  - `aggressive_position_sizing.py`
  - `volatility_based_rsi_div.py`
  - `long_only_position_sizing.py`
  - (Explore these scripts to understand their specific logic)
- **Freqtrade Strategy (`rsi_divergence_strategy.py`):**
  - Optimized for 5m timeframe.
  - Targets specific pairs (HYPE, SOL, XRP, ETH, BTC).
  - Uses aggressive settings (leverage, risk per trade).
  - Includes token-specific leverage rules.
  - Features a "YOLO" trade mode for high-conviction signals.

## _Notable Backtest Results_

The backtests using the RSI Divergence logic within the `multi_coin_rsi_div.py` script (leveraging `backtesting.py`) yielded several promising results across different assets. Below are some highlights based on the summary found in `results/rsi_div_summary.txt`:

- **DOGE:** Achieved a remarkable **136.48% Return** with an **87.50% Win Rate** and a **6.86 Profit Factor**.
- **HYPE:** Showed a strong **134.12% Return**, **80.34% Win Rate**, and **4.32 Profit Factor**.
- **AI16Z:** While having a lower total return (**34.80%**), it demonstrated an exceptional **95.65% Win Rate** and **14.27 Profit Factor**.
- **XRP:** Delivered a solid **50.72% Return** with a **70.65% Win Rate** and **2.46 Profit Factor**.

**Where to find:**

- The primary strategy logic for these results is within `multi_coin_rsi_div.py`.
- The Freqtrade-specific implementation is in `rsi_divergence_strategy.py`.
- Detailed results for each coin, including the summary file and trade logs (`trades.csv`), are located in the `results/` directory.

You can examine these files to understand the strategy parameters and potentially adapt the logic for your own trading bot or platform.

## License

This project is licensed under the MIT License - see the LICENSE file for details (if available, otherwise assume MIT based on common practice).
