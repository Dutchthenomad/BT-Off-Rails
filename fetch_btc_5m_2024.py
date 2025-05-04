from master_data_collection.fetchers.enhanced_hyperliquid_fetcher import EnhancedHyperliquidFetcher
from datetime import datetime

if __name__ == "__main__":
    fetcher = EnhancedHyperliquidFetcher()
    # Calculate days_back for 2024 (from Jan 1, 2024 to today)
    start_date = datetime(2024, 1, 1)
    today = datetime.now()
    days_back = (today - start_date).days

    # Fetch BTC 5m candles for all of 2024 up to today
    btc_data = fetcher.fetch_historical_data("BTC", "5m", days_back=days_back, save_csv=True)
    print(f"Fetched {len(btc_data)} 5-minute candles for BTC (2024)")
