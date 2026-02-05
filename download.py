"""
Binance Historical Data Dumper
==============================
Downloads the last 4 years of historical data for specified crypto pairs.
Uses the binance-historical-data package to fetch data from Binance.

https://pypi.org/project/binance-historical-data/

Data Types Downloaded:
- Klines: Standard OHLCV candlestick data
- Mark Price Klines: Mark price data for liquidation monitoring
- Metrics: Open interest and other trading metrics (5-min intervals)
  - sum_open_interest: Total open interest (number of contracts)
  - sum_open_interest_value: Total open interest value (in USDT)
  - Long/short ratios and taker volume ratios
- Funding Rates: Downloaded from HyperLiquid via ccxt (1-hour resolution)
  - Better than Binance's 8-hour funding rate data
  - Stored incrementally in JSON format

Configuration:
- Asset Class: USDT Margined Futures (um) for cross margin trading
- Timeframes: Configurable (default: 1h, 4h, 1d)
"""

import datetime
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import ccxt
import pandas as pd
from binance_historical_data import BinanceDataDumper

# Configure logging with unbuffered stdout
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
file_handler = logging.FileHandler('binance_data_dump.log')
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Get the root logger and add handlers
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(stream_handler)
logger.addHandler(file_handler)

print("Script starting...", flush=True)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Trading pairs in ccxt format - will be converted to Binance format
TRADING_PAIRS_CCXT = [
    "AAVE/USDC:USDC",
    "ARB/USDC:USDC",
    "ASTER/USDC:USDC",
    "AVAX/USDC:USDC",
    "BTC/USDC:USDC",
    "DOGE/USDC:USDC",
    "DOT/USDC:USDC",
    "ENS/USDC:USDC",
    "ETH/USDC:USDC",
    "ETHFI/USDC:USDC",
    "FIL/USDC:USDC",
    "HYPE/USDC:USDC",
    "IP/USDC:USDC",
    "JTO/USDC:USDC",
    "LINK/USDC:USDC",
    "NEAR/USDC:USDC",
    "OP/USDC:USDC",
    "SEI/USDC:USDC",
    "SOL/USDC:USDC",
    "SUI/USDC:USDC",
    "TAO/USDC:USDC",
    "TRUMP/USDC:USDC",
    "WLD/USDC:USDC",
]

# Timeframes to download (can modify as needed)
TIMEFRAMES = ["1h", "4h", "1d"]

# Data types to download from data.binance.vision
# For futures (um/cm): aggTrades, klines, trades, indexPriceKlines, markPriceKlines, premiumIndexKlines, metrics
# For spot: aggTrades, klines, trades
# Note: Funding rates are NOT available via bulk download - they require the Binance API (/fapi/v1/fundingRate)
DATA_TYPES = [
    "klines",           # Standard OHLCV futures data (traded prices)
    "markPriceKlines",  # Mark price data (for liquidation monitoring)
]

# Data types that don't require timeframe (they have fixed 5-min intervals)
# Metrics contains: create_time, symbol, sum_open_interest, sum_open_interest_value, count_toptrader_long_short_ratio, 
#                   sum_toptrader_long_short_ratio, count_long_short_ratio, sum_taker_long_short_vol_ratio
NO_TIMEFRAME_DATA_TYPES = [
    "metrics",          # Open interest and other trading metrics (5-min intervals, futures only)
]

# Number of years to fetch
YEARS_TO_FETCH = 4

# Output directory for the data
OUTPUT_DIR = "./binance_data"

# Asset class options:
# - "spot": Spot trading data
# - "um": USDT Margined Futures (Cross Margin)
# - "cm": Coin Margined Futures
ASSET_CLASS = "um"

# ============================================================================
# FUNDING RATE CONFIGURATION (HyperLiquid via ccxt)
# ============================================================================
# HyperLiquid provides 1-hour funding rates (better than Binance's 8-hour)
FUNDING_RATE_ENABLED = False
FUNDING_RATE_EXCHANGE = "hyperliquid"
FUNDING_RATE_OUTPUT_DIR = "./binance_data/funding_rates"
FUNDING_RATE_BATCH_SIZE = 500  # HyperLiquid's max per request
FUNDING_RATE_RETRY_DELAYS = [2, 4, 8, 16, 32, 64, 128]  # Exponential backoff for rate limits


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def convert_ccxt_to_binance_ticker(ccxt_symbol: str) -> str:
    """
    Convert a ccxt format symbol to Binance API format.
    
    Examples:
        "AAVE/USDC:USDC" -> "AAVEUSDC"
        "BTC/USDC:USDC" -> "BTCUSDC"
        "ETH/USDT" -> "ETHUSDT"
    
    Args:
        ccxt_symbol: Symbol in ccxt format (e.g., "BTC/USDC:USDC")
    
    Returns:
        Binance format ticker (e.g., "BTCUSDC")
    """
    # Remove the settlement currency part (e.g., ":USDC")
    base_symbol = ccxt_symbol.split(":")[0]
    # Remove the slash to get Binance format
    binance_ticker = base_symbol.replace("/", "")
    return binance_ticker


def get_date_range(years: int = 4) -> tuple[datetime.date, datetime.date]:
    """
    Calculate the date range for data fetching.
    
    Args:
        years: Number of years of historical data to fetch
    
    Returns:
        Tuple of (start_date, end_date)
    """
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=years * 365)
    return start_date, end_date


def create_output_directory(output_dir: str) -> Path:
    """
    Create the output directory if it doesn't exist.
    
    Args:
        output_dir: Path to output directory
    
    Returns:
        Path object for the output directory
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def dump_data_for_timeframe(
    output_dir: str,
    tickers: list[str],
    timeframe: str,
    data_type: str,
    date_start: datetime.date,
    date_end: datetime.date,
    asset_class: str = "um"
) -> None:
    """
    Dump data for a specific timeframe and data type.
    
    Args:
        output_dir: Directory to store the data
        tickers: List of Binance format tickers
        timeframe: Candlestick timeframe (e.g., "1h", "4h", "1d")
        data_type: Type of data (klines, markPriceKlines, etc.)
        date_start: Start date for data
        date_end: End date for data  
        asset_class: Binance asset class ("spot", "um", "cm")
    """
    logger.info(f"Starting data dump for {data_type} @ {timeframe}")
    logger.info(f"Date range: {date_start} to {date_end}")
    logger.info(f"Tickers: {tickers}")
    logger.info(f"Asset class: {asset_class}")
    
    try:
        # Initialize the data dumper
        data_dumper = BinanceDataDumper(
            path_dir_where_to_dump=output_dir,
            asset_class=asset_class,
            data_type=data_type,
            data_frequency=timeframe,
        )
        
        # Dump the data
        data_dumper.dump_data(
            tickers=tickers,
            date_start=date_start,
            date_end=date_end,
            is_to_update_existing=True,  # Update if data already exists
        )
        
        # Clean up outdated daily results (optional - saves disk space)
        data_dumper.delete_outdated_daily_results()
        
        logger.info(f"Successfully completed data dump for {data_type} @ {timeframe}")
        
    except Exception as e:
        logger.error(f"Error dumping {data_type} for timeframe {timeframe}: {e}")
        raise


def dump_data_no_timeframe(
    output_dir: str,
    tickers: list[str],
    data_type: str,
    date_start: datetime.date,
    date_end: datetime.date,
    asset_class: str = "um"
) -> None:
    """
    Dump data for data types that don't require timeframe (e.g., metrics).
    
    The metrics data type includes:
    - create_time: Timestamp of the data point
    - symbol: Trading pair symbol
    - sum_open_interest: Total open interest (number of contracts)
    - sum_open_interest_value: Total open interest value (in USDT)
    - count_toptrader_long_short_ratio: Top traders long/short ratio by accounts
    - sum_toptrader_long_short_ratio: Top traders long/short ratio by positions
    - count_long_short_ratio: Long/short ratio by accounts
    - sum_taker_long_short_vol_ratio: Taker buy/sell volume ratio
    
    Args:
        output_dir: Directory to store the data
        tickers: List of Binance format tickers
        data_type: Type of data (metrics, etc.)
        date_start: Start date for data
        date_end: End date for data  
        asset_class: Binance asset class ("um" for metrics, only supported for USD-M futures)
    """
    logger.info(f"Starting data dump for {data_type} (no timeframe)")
    logger.info(f"Date range: {date_start} to {date_end}")
    logger.info(f"Tickers: {tickers}")
    logger.info(f"Asset class: {asset_class}")
    
    if data_type == "metrics" and asset_class != "um":
        logger.warning(f"metrics data type is only supported for 'um' (USD-M futures), but got '{asset_class}'. Skipping.")
        return
    
    try:
        # Initialize the data dumper without data_frequency for non-timeframe data
        data_dumper = BinanceDataDumper(
            path_dir_where_to_dump=output_dir,
            asset_class=asset_class,
            data_type=data_type,
        )
        
        # Dump the data
        data_dumper.dump_data(
            tickers=tickers,
            date_start=date_start,
            date_end=date_end,
            is_to_update_existing=True,  # Update if data already exists
        )
        
        # Clean up outdated daily results (optional - saves disk space)
        data_dumper.delete_outdated_daily_results()
        
        logger.info(f"Successfully completed data dump for {data_type}")
        
    except Exception as e:
        logger.error(f"Error dumping {data_type}: {e}")
        raise


# ============================================================================
# FUNDING RATE DOWNLOAD FUNCTIONS (HyperLiquid via ccxt)
# ============================================================================

def convert_ccxt_pair_to_hyperliquid(ccxt_symbol: str) -> str:
    """
    Convert a ccxt format symbol to HyperLiquid format.
    
    Examples:
        "AAVE/USDC:USDC" -> "AAVE/USDC:USDC" (HyperLiquid uses similar format)
        "BTC/USDC:USDC" -> "BTC/USDC:USDC"
    
    Args:
        ccxt_symbol: Symbol in ccxt format (e.g., "BTC/USDC:USDC")
    
    Returns:
        HyperLiquid format symbol
    """
    # HyperLiquid uses the same ccxt format
    return ccxt_symbol


def get_funding_rate_json_path(pair: str, output_dir: Path) -> Path:
    """
    Get the JSON file path for funding rate data.
    
    Args:
        pair: Trading pair in ccxt format (e.g., "BTC/USDC:USDC")
        output_dir: Output directory
    
    Returns:
        Path to the JSON file
    """
    # Format filename: ETH_USDC_USDC-1h-funding_rate.json
    safe_pair = pair.replace('/', '_').replace(':', '_')
    return output_dir / f"{safe_pair}-1h-funding_rate.json"


def load_existing_funding_data(json_path: Path) -> tuple[Optional[list], Optional[int]]:
    """
    Load existing funding rate data from JSON file.
    
    Args:
        json_path: Path to JSON file
    
    Returns:
        Tuple of (existing_data, last_timestamp_ms)
    """
    if not json_path.exists():
        return None, None
    
    try:
        with open(json_path, 'r') as f:
            existing_data = json.load(f)
        if existing_data and len(existing_data) > 0:
            # Get the last timestamp from existing data (first element of last row)
            last_timestamp_ms = existing_data[-1][0]
            logger.info(f"Found existing funding rate data, last timestamp: {pd.Timestamp(last_timestamp_ms, unit='ms', tz='UTC')}")
            return existing_data, last_timestamp_ms
    except Exception as e:
        logger.warning(f"Failed to load existing JSON file: {e}")
    
    return None, None


def download_funding_rates_for_pair(
    pair: str,
    start_date: datetime.date,
    end_date: datetime.date,
    output_dir: Path,
    exchange_id: str = "hyperliquid",
    batch_size: int = 500,
    retry_delays: list[int] = None
) -> bool:
    """
    Download funding rate data for a single pair using ccxt.
    Supports incremental downloads - only fetches new data since last stored candle.
    
    Args:
        pair: Trading pair in ccxt format (e.g., "BTC/USDC:USDC")
        start_date: Start date for data
        end_date: End date for data
        output_dir: Output directory for JSON files
        exchange_id: Exchange ID for ccxt (default: hyperliquid)
        batch_size: Number of records per request
        retry_delays: List of retry delays for rate limiting
    
    Returns:
        True if successful
    """
    if retry_delays is None:
        retry_delays = FUNDING_RATE_RETRY_DELAYS
    
    try:
        # Get JSON file path
        json_path = get_funding_rate_json_path(pair, output_dir)
        
        # Load existing data
        existing_data, last_timestamp_ms = load_existing_funding_data(json_path)
        
        # Calculate time range in milliseconds
        start_ms = int(datetime.datetime.combine(start_date, datetime.time.min).timestamp() * 1000)
        end_ms = int(datetime.datetime.combine(end_date, datetime.time.max).timestamp() * 1000)
        now_ms = int(pd.Timestamp.now(tz='UTC').timestamp() * 1000)
        until_ms = max(end_ms, now_ms)
        
        # Determine fetch start time
        if last_timestamp_ms is not None:
            # We have existing data - only fetch from last timestamp + 1 hour (3600000 ms)
            since_ms = last_timestamp_ms + 3600000
            
            # Check if we actually need to fetch new data
            if since_ms >= until_ms:
                logger.debug(f"Funding rate data for {pair} is up to date, no download needed")
                return True
            
            logger.info(f"Incremental download for {pair}: fetching from {pd.Timestamp(since_ms, unit='ms', tz='UTC')}")
        else:
            # No existing data - fetch full range
            since_ms = start_ms
            logger.info(f"Full download for {pair}: fetching from {pd.Timestamp(since_ms, unit='ms', tz='UTC')}")
        
        # Initialize exchange
        exchange_class = getattr(ccxt, exchange_id, None)
        if exchange_class is None:
            logger.warning(f"Exchange {exchange_id} not found in ccxt, skipping funding rate download")
            return False
        
        exchange = exchange_class({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        
        # Fetch funding rates with pagination
        all_funding = []
        current_since = since_ms
        
        while current_since < until_ms:
            funding = None
            last_error = None
            
            # Retry logic with exponential backoff for rate limiting
            for attempt in range(len(retry_delays) + 1):
                try:
                    funding = exchange.fetch_funding_rate_history(pair, since=current_since, limit=batch_size)
                    break
                except Exception as e:
                    error_str = str(e).lower()
                    if '429' in error_str or 'too many requests' in error_str or 'rate limit' in error_str:
                        last_error = e
                        if attempt < len(retry_delays):
                            wait_time = retry_delays[attempt]
                            logger.warning(f"Rate limited (429) for {pair}, waiting {wait_time}s before retry {attempt + 1}/{len(retry_delays)}...")
                            time.sleep(wait_time)
                        else:
                            logger.warning(f"Failed to download funding rate for {pair} after {len(retry_delays)} retries: {e}")
                            raise
                    else:
                        raise
            
            if not funding:
                break
            
            # Rate limiting delay
            time.sleep(2)
            
            all_funding.extend(funding)
            
            # Get the last timestamp
            last_ts = funding[-1]['timestamp']
            
            # If we got fewer records than batch_size, we've reached the end
            if len(funding) < batch_size:
                break
            
            # Move to next batch
            current_since = last_ts + 1
            
            logger.debug(f"Fetched {len(funding)} funding records, total: {len(all_funding)}")
        
        # Process new funding data
        if all_funding:
            # Remove duplicates based on timestamp
            seen_timestamps = set()
            unique_funding = []
            for entry in all_funding:
                ts = entry['timestamp']
                if ts not in seen_timestamps:
                    seen_timestamps.add(ts)
                    unique_funding.append(entry)
            
            logger.info(f"Downloaded {len(unique_funding)} new funding rate entries for {pair}")
            
            # Convert new data to JSON format: [timestamp, rate, rate, rate, rate, 0.0]
            new_json_data = []
            for entry in unique_funding:
                ts = entry['timestamp']
                rate = entry['fundingRate']
                new_json_data.append([ts, rate, rate, rate, rate, 0.0])
            
            # Merge with existing data if available
            if existing_data is not None and len(existing_data) > 0:
                # Create a dict for deduplication (timestamp -> row)
                data_dict = {row[0]: row for row in existing_data}
                # Add new data (overwrites duplicates)
                for row in new_json_data:
                    data_dict[row[0]] = row
                # Sort by timestamp and convert back to list
                combined_data = sorted(data_dict.values(), key=lambda x: x[0])
            else:
                combined_data = sorted(new_json_data, key=lambda x: x[0])
            
            # Save to JSON format
            with open(json_path, 'w') as f:
                json.dump(combined_data, f)
            logger.info(f"Saved {len(combined_data)} funding rate entries for {pair} to {json_path}")
            
        elif existing_data is not None and len(existing_data) > 0:
            logger.debug(f"No new funding data for {pair}, existing {len(existing_data)} entries unchanged")
        else:
            logger.warning(f"No funding data found for {pair}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to download funding rate for {pair}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


def dump_funding_rates(
    pairs: list[str],
    date_start: datetime.date,
    date_end: datetime.date,
    output_dir: str,
    exchange_id: str = "hyperliquid"
) -> dict:
    """
    Download funding rate data for all pairs.
    
    Args:
        pairs: List of trading pairs in ccxt format
        date_start: Start date
        date_end: End date
        output_dir: Output directory
        exchange_id: Exchange ID for ccxt
    
    Returns:
        Dict with results summary
    """
    logger.info("#" * 60)
    logger.info("Processing: Funding Rates (HyperLiquid via ccxt)")
    logger.info("#" * 60)
    logger.info(f"Date range: {date_start} to {date_end}")
    logger.info(f"Pairs: {len(pairs)}")
    logger.info(f"Exchange: {exchange_id}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {'successful': [], 'failed': []}
    
    for i, pair in enumerate(pairs, 1):
        logger.info("-" * 40)
        logger.info(f"[{i}/{len(pairs)}] Processing {pair}")
        logger.info("-" * 40)
        
        success = download_funding_rates_for_pair(
            pair=pair,
            start_date=date_start,
            end_date=date_end,
            output_dir=output_path,
            exchange_id=exchange_id,
            batch_size=FUNDING_RATE_BATCH_SIZE,
            retry_delays=FUNDING_RATE_RETRY_DELAYS
        )
        
        if success:
            results['successful'].append(pair)
        else:
            results['failed'].append(pair)
        
        # Rate limiting between pairs
        time.sleep(0.5)
    
    logger.info("=" * 40)
    logger.info(f"Funding rates: {len(results['successful'])}/{len(pairs)} successful")
    if results['failed']:
        logger.warning(f"Failed pairs: {results['failed']}")
    
    return results


def main():
    """
    Main function to orchestrate the data dumping process.
    """
    print("Entering main()...", flush=True)
    print(f"Logger name: {logger.name}, handlers: {logger.handlers}, level: {logger.level}", flush=True)
    logger.info("=" * 60)
    print("After first logger.info call", flush=True)
    logger.info("Binance Historical Data Dumper")
    logger.info("=" * 60)
    
    # Convert ccxt symbols to Binance format
    binance_tickers = [convert_ccxt_to_binance_ticker(pair) for pair in TRADING_PAIRS_CCXT]
    logger.info(f"Converted {len(binance_tickers)} trading pairs to Binance format")
    logger.info(f"Tickers: {binance_tickers}")
    
    # Create output directory
    output_path = create_output_directory(OUTPUT_DIR)
    logger.info(f"Output directory: {output_path.absolute()}")
    
    # Calculate date range
    date_start, date_end = get_date_range(YEARS_TO_FETCH)
    logger.info(f"Fetching data from {date_start} to {date_end} ({YEARS_TO_FETCH} years)")

    # Download funding rates from HyperLiquid (1-hour resolution)
    try:
        logger.info("#" * 60)
        logger.info(f"Processing funding rates")
        logger.info("#" * 60)

        dump_funding_rates(
            pairs=TRADING_PAIRS_CCXT,
            date_start=date_start,
            date_end=date_end,
            output_dir=FUNDING_RATE_OUTPUT_DIR,
            exchange_id=FUNDING_RATE_EXCHANGE
        )
    except Exception as e:
        logger.error(f"Failed to download funding rates: {e}")
    
    # Dump data for data types that don't require timeframe (e.g., metrics with open interest)
    for data_type in NO_TIMEFRAME_DATA_TYPES:
        logger.info("#" * 60)
        logger.info(f"Processing data type: {data_type} (no timeframe)")
        logger.info("#" * 60)
        
        try:
            dump_data_no_timeframe(
                output_dir=str(output_path),
                tickers=binance_tickers,
                data_type=data_type,
                date_start=date_start,
                date_end=date_end,
                asset_class=ASSET_CLASS,
            )
        except Exception as e:
            logger.error(f"Failed to complete dump for {data_type}: {e}")
            continue


        # Dump data for each data type and timeframe
        for data_type in DATA_TYPES:
            logger.info("#" * 60)
            logger.info(f"Processing data type: {data_type}")
            logger.info("#" * 60)
            
            for timeframe in TIMEFRAMES:
                logger.info("-" * 40)
                logger.info(f"Processing {data_type} @ {timeframe}")
                logger.info("-" * 40)
                
                try:
                    dump_data_for_timeframe(
                        output_dir=str(output_path),
                        tickers=binance_tickers,
                        timeframe=timeframe,
                        data_type=data_type,
                        date_start=date_start,
                        date_end=date_end,
                        asset_class=ASSET_CLASS,
                    )
                except Exception as e:
                    logger.error(f"Failed to complete dump for {data_type} @ {timeframe}: {e}")
                    continue
        
    
    logger.info("=" * 60)
    logger.info("Data dump completed!")
    logger.info(f"Data saved to: {output_path.absolute()}")
    if FUNDING_RATE_ENABLED:
        logger.info(f"Funding rates saved to: {Path(FUNDING_RATE_OUTPUT_DIR).absolute()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
