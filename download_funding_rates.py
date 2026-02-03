"""
Binance Funding Rate Downloader
===============================
Downloads historical funding rate data from Binance Futures API.
This is separate from the bulk data download because funding rates
are only available via the Binance API, not from data.binance.vision.

Funding rates are settled every 8 hours on Binance (00:00, 08:00, 16:00 UTC).
"""

import datetime
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('funding_rate_download.log')
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Trading pairs (in Binance format)
TRADING_PAIRS = [
    "AAVEUSDC",
    "ARBUSDC",
    "AVAXUSDC",
    "BTCUSDC",
    "DOGEUSDC",
    "DOTUSDC",
    "ENSUSDC",
    "ETHUSDC",
    "ETHFIUSDC",
    "FILUSDC",
    "HYPEUSDC",
    "IPUSDC",
    "JTOUSDC",
    "LINKUSDC",
    "NEARUSDC",
    "OPUSDC",
    "PEPEUSDC",
    "SEIUSDC",
    "SOLUSDC",
    "SUIUSDC",
    "TAOUSDC",
    "TRUMPUSDC",
    "WLDUSDC",
]

# Number of years of funding rate data to fetch
YEARS_TO_FETCH = 4

# Output directory
OUTPUT_DIR = "./binance_data/funding_rates"

# Binance Futures API endpoint
BINANCE_FUTURES_API = "https://fapi.binance.com"

# Rate limiting
REQUEST_DELAY = 0.5  # seconds between requests to avoid rate limits
MAX_LIMIT = 1000  # max records per request


# ============================================================================
# API FUNCTIONS
# ============================================================================

def get_funding_rate_history(
    symbol: str,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    limit: int = 1000
) -> list[dict]:
    """
    Fetch funding rate history from Binance Futures API.
    
    API Endpoint: GET /fapi/v1/fundingRate
    
    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDC")
        start_time: Start time in milliseconds
        end_time: End time in milliseconds
        limit: Number of records to fetch (max 1000)
    
    Returns:
        List of funding rate records
    """
    url = f"{BINANCE_FUTURES_API}/fapi/v1/fundingRate"
    
    params = {
        "symbol": symbol,
        "limit": limit
    }
    
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed for {symbol}: {e}")
        return []


def fetch_all_funding_rates(
    symbol: str,
    start_date: datetime.date,
    end_date: datetime.date
) -> pd.DataFrame:
    """
    Fetch all funding rate data for a symbol within a date range.
    Uses pagination to get all records.
    
    Args:
        symbol: Trading pair symbol
        start_date: Start date
        end_date: End date
    
    Returns:
        DataFrame with all funding rate records
    """
    all_records = []
    
    # Convert dates to milliseconds
    start_time = int(datetime.datetime.combine(start_date, datetime.time.min).timestamp() * 1000)
    end_time = int(datetime.datetime.combine(end_date, datetime.time.max).timestamp() * 1000)
    
    current_start = start_time
    
    while current_start < end_time:
        logger.debug(f"Fetching {symbol} from {current_start}")
        
        records = get_funding_rate_history(
            symbol=symbol,
            start_time=current_start,
            end_time=end_time,
            limit=MAX_LIMIT
        )
        
        if not records:
            break
        
        all_records.extend(records)
        
        # Get the timestamp of the last record + 1ms for next batch
        last_timestamp = records[-1].get("fundingTime", 0)
        current_start = last_timestamp + 1
        
        # Check if we got fewer than limit (means we're done)
        if len(records) < MAX_LIMIT:
            break
        
        # Rate limiting
        time.sleep(REQUEST_DELAY)
    
    if not all_records:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_records)
    
    # Convert types
    df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms', utc=True)
    df['fundingRate'] = pd.to_numeric(df['fundingRate'])
    df['markPrice'] = pd.to_numeric(df['markPrice'])
    
    # Rename columns for clarity
    df = df.rename(columns={
        'fundingTime': 'datetime',
        'fundingRate': 'funding_rate',
        'markPrice': 'mark_price'
    })
    
    # Sort by datetime
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['datetime'], keep='first')
    
    # Add date column for compatibility with consumers expecting 'date'
    df['date'] = df['datetime'].dt.date
    
    # Reorder columns to have date first
    cols = ['date', 'symbol', 'datetime', 'funding_rate', 'mark_price']
    df = df[[c for c in cols if c in df.columns]]
    
    return df


def save_funding_rates(df: pd.DataFrame, symbol: str, output_dir: Path) -> bool:
    """
    Save funding rate data to feather file.
    
    Args:
        df: DataFrame with funding rate data
        symbol: Trading pair symbol
        output_dir: Output directory
    
    Returns:
        True if successful
    """
    if df.empty:
        logger.warning(f"No data to save for {symbol}")
        return False
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert symbol to output format (e.g., BTCUSDC -> BTC_USDC_USDC)
        if symbol.endswith("USDC"):
            base = symbol[:-4]
            output_symbol = f"{base}_USDC_USDC"
        elif symbol.endswith("USDT"):
            base = symbol[:-4]
            output_symbol = f"{base}_USDT_USDT"
        else:
            output_symbol = symbol
        
        output_path = output_dir / f"{output_symbol}-funding.feather"
        df.to_feather(output_path)
        
        logger.info(f"Saved {len(df)} funding rate records to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving funding rates for {symbol}: {e}")
        return False


def main():
    """
    Main function to download funding rate data for all symbols.
    """
    logger.info("=" * 60)
    logger.info("Binance Funding Rate Downloader")
    logger.info("=" * 60)
    
    # Calculate date range
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=YEARS_TO_FETCH * 365)
    
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Symbols to process: {len(TRADING_PAIRS)}")
    
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for i, symbol in enumerate(TRADING_PAIRS, 1):
        logger.info("-" * 40)
        logger.info(f"[{i}/{len(TRADING_PAIRS)}] Processing {symbol}")
        logger.info("-" * 40)
        
        try:
            # Fetch funding rates
            df = fetch_all_funding_rates(symbol, start_date, end_date)
            
            if df.empty:
                logger.warning(f"No funding rate data found for {symbol}")
                results.append({'symbol': symbol, 'success': False, 'records': 0})
                continue
            
            # Save to feather
            success = save_funding_rates(df, symbol, output_dir)
            
            results.append({
                'symbol': symbol,
                'success': success,
                'records': len(df),
                'start': df['datetime'].min(),
                'end': df['datetime'].max()
            })
            
            # Rate limiting between symbols
            time.sleep(REQUEST_DELAY)
            
        except Exception as e:
            logger.error(f"Failed to process {symbol}: {e}")
            results.append({'symbol': symbol, 'success': False, 'records': 0})
            continue
    
    # Summary
    logger.info("=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    logger.info(f"Successful: {len(successful)}/{len(TRADING_PAIRS)}")
    logger.info(f"Failed: {len(failed)}/{len(TRADING_PAIRS)}")
    
    total_records = sum(r['records'] for r in results)
    logger.info(f"Total funding rate records: {total_records:,}")
    
    if failed:
        logger.warning("Failed symbols:")
        for r in failed:
            logger.warning(f"  - {r['symbol']}")
    
    logger.info(f"Output directory: {output_dir.absolute()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
