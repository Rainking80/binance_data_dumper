"""
Binance Data Condenser - Combined Data
======================================
Combines all data sources into a single consolidated feather file per symbol:
- klines: Standard OHLCV candlestick data (traded prices)
- markPriceKlines: Mark price data (for liquidation monitoring)
- metrics: Open interest and trading metrics (5-min intervals, resampled to 1h)
- funding_rates: Funding rate data (1-hour resolution from HyperLiquid)

All data is aligned by timestamp. Rows with missing values at the beginning
(before all data sources have data) are removed.

Output Directory Structure:
    condensed_data/
    ├── AAVE_USDC_USDC-1h-combined.feather
    ├── BTC_USDC_USDC-1h-combined.feather
    └── ...
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('condense.log')
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Base directory where the raw data is stored
RAW_DATA_DIR = "./binance_data/futures/um"

# Funding rates directory
FUNDING_RATES_DIR = "./binance_data/funding_rates"

# Output directory for condensed feather files
OUTPUT_DIR = "./condensed_data"

# Target timeframe for output (currently 1h - metrics will be resampled)
TARGET_TIMEFRAME = "1h"

# Column names for the CSV files (Binance klines format)
KLINE_COLUMNS = [
    'open_time',      # Kline open time (Unix timestamp in ms)
    'open',           # Open price
    'high',           # High price
    'low',            # Low price
    'close',          # Close price
    'volume',         # Volume
    'close_time',     # Kline close time (Unix timestamp in ms)
    'quote_volume',   # Quote asset volume
    'count',          # Number of trades
    'taker_buy_volume',       # Taker buy base asset volume
    'taker_buy_quote_volume', # Taker buy quote asset volume
    'ignore'          # Ignore
]

# Data types for optimization
KLINE_DTYPES = {
    'open_time': 'int64',
    'open': 'float64',
    'high': 'float64',
    'low': 'float64',
    'close': 'float64',
    'volume': 'float64',
    'close_time': 'int64',
    'quote_volume': 'float64',
    'count': 'int64',
    'taker_buy_volume': 'float64',
    'taker_buy_quote_volume': 'float64',
    'ignore': 'int64'
}

# Metrics columns
METRICS_COLUMNS = [
    'create_time',
    'symbol',
    'sum_open_interest',
    'sum_open_interest_value',
    'count_toptrader_long_short_ratio',
    'sum_toptrader_long_short_ratio',
    'count_long_short_ratio',
    'sum_taker_long_short_vol_ratio'
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def convert_binance_to_ccxt_symbol(binance_ticker: str) -> str:
    """
    Convert a Binance format ticker to ccxt-style format for output naming.
    
    Examples:
        "BTCUSDC" -> "BTC_USDC_USDC"
        "ETHUSDC" -> "ETH_USDC_USDC"
    
    Args:
        binance_ticker: Ticker in Binance format (e.g., "BTCUSDC")
    
    Returns:
        ccxt-style format (e.g., "BTC_USDC_USDC")
    """
    # Extract base and quote from the ticker
    # Common quote currencies
    quotes = ['USDC', 'USDT', 'BUSD', 'BTC', 'ETH']
    
    for quote in quotes:
        if binance_ticker.endswith(quote):
            base = binance_ticker[:-len(quote)]
            return f"{base}_{quote}_{quote}"
    
    # Fallback: return as-is with underscores
    return binance_ticker


def collect_csv_files(base_dir: Path, data_type: str, timeframe: Optional[str] = None) -> dict:
    """
    Collect all CSV files for each symbol/timeframe combination.
    Searches both daily and monthly directories.
    
    Args:
        base_dir: Base directory (e.g., ./binance_data/futures/um)
        data_type: Type of data (klines, markPriceKlines, metrics)
        timeframe: Timeframe for klines data (None for metrics)
    
    Returns:
        Dictionary mapping symbol -> list of CSV file paths
    """
    csv_files = {}
    
    # Search in both daily and monthly directories
    for period in ['daily', 'monthly']:
        period_dir = base_dir / period / data_type
        
        if not period_dir.exists():
            logger.debug(f"Directory not found: {period_dir}")
            continue
        
        # Iterate over symbol directories
        for symbol_dir in period_dir.iterdir():
            if not symbol_dir.is_dir():
                continue
                
            symbol = symbol_dir.name
            
            if symbol not in csv_files:
                csv_files[symbol] = []
            
            if timeframe is not None:
                # For klines/markPriceKlines - need to go into timeframe subdirectory
                tf_dir = symbol_dir / timeframe
                if tf_dir.exists():
                    for csv_file in tf_dir.glob("*.csv"):
                        csv_files[symbol].append(csv_file)
            else:
                # For metrics - files are directly in symbol dir
                for csv_file in symbol_dir.glob("*.csv"):
                    csv_files[symbol].append(csv_file)
    
    return csv_files


def read_and_concat_klines(csv_paths: list[Path]) -> Optional[pd.DataFrame]:
    """
    Read multiple kline CSV files and concatenate them into a single DataFrame.
    Handles deduplication and sorting by timestamp.
    
    Args:
        csv_paths: List of paths to CSV files
    
    Returns:
        Concatenated DataFrame or None if no data
    """
    if not csv_paths:
        return None
    
    dfs = []
    
    for csv_path in sorted(csv_paths):
        try:
            df = pd.read_csv(
                csv_path,
                dtype=KLINE_DTYPES,
                on_bad_lines='skip'
            )
            
            if 'open_time' not in df.columns:
                logger.warning(f"Skipping {csv_path}: missing 'open_time' column")
                continue
            
            dfs.append(df)
            
        except Exception as e:
            logger.error(f"Error reading {csv_path}: {e}")
            continue
    
    if not dfs:
        return None
    
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=['open_time'], keep='first')
    combined = combined.sort_values('open_time').reset_index(drop=True)
    
    return combined


def read_and_concat_metrics(csv_paths: list[Path]) -> Optional[pd.DataFrame]:
    """
    Read multiple metrics CSV files and concatenate them.
    
    Args:
        csv_paths: List of paths to CSV files
    
    Returns:
        Concatenated DataFrame or None if no data
    """
    if not csv_paths:
        return None
    
    dfs = []
    
    for csv_path in sorted(csv_paths):
        try:
            df = pd.read_csv(csv_path, on_bad_lines='skip')
            
            if 'create_time' not in df.columns:
                logger.warning(f"Skipping {csv_path}: missing 'create_time' column")
                continue
            
            dfs.append(df)
            
        except Exception as e:
            logger.error(f"Error reading {csv_path}: {e}")
            continue
    
    if not dfs:
        return None
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Parse create_time and convert to timestamp
    combined['create_time'] = pd.to_datetime(combined['create_time'], utc=True)
    combined = combined.drop_duplicates(subset=['create_time'], keep='first')
    combined = combined.sort_values('create_time').reset_index(drop=True)
    
    return combined


def resample_metrics_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 5-minute metrics data to hourly intervals.
    Uses mean for ratios and last value for open interest.
    
    Args:
        df: Metrics DataFrame with 5-min intervals
    
    Returns:
        Resampled DataFrame with hourly intervals
    """
    if df is None or len(df) == 0:
        return None
    
    # Set create_time as index for resampling
    df = df.set_index('create_time')
    
    # Define aggregation method for each column
    agg_dict = {
        'sum_open_interest': 'last',
        'sum_open_interest_value': 'last',
        'count_toptrader_long_short_ratio': 'mean',
        'sum_toptrader_long_short_ratio': 'mean',
        'count_long_short_ratio': 'mean',
        'sum_taker_long_short_vol_ratio': 'mean'
    }
    
    # Only aggregate columns that exist
    agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
    
    if not agg_dict:
        return None
    
    # Resample to hourly, using floor (start of hour)
    resampled = df.resample('1h', label='left', closed='left').agg(agg_dict)
    resampled = resampled.reset_index()
    
    return resampled


def load_funding_rates(symbol_ccxt: str, funding_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load funding rate data from JSON file.
    
    Args:
        symbol_ccxt: Symbol in ccxt format (e.g., "BTC_USDC_USDC")
        funding_dir: Directory containing funding rate JSON files
    
    Returns:
        DataFrame with funding rate data or None if not found
    """
    json_path = funding_dir / f"{symbol_ccxt}-1h-funding_rate.json"
    
    if not json_path.exists():
        logger.debug(f"Funding rate file not found: {json_path}")
        return None
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if not data:
            return None
        
        # Data format: [timestamp_ms, rate, rate, rate, rate, 0.0]
        # We only need timestamp and the funding rate
        df = pd.DataFrame(data, columns=['timestamp', 'funding_rate', 'fr2', 'fr3', 'fr4', 'ignore'])
        df = df[['timestamp', 'funding_rate']].copy()
        
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        
        # Floor to hour to ensure alignment
        df['open_time'] = df['datetime'].dt.floor('h')
        df['open_time_ms'] = (df['open_time'].astype('int64') // 10**6).astype('int64')
        
        # Drop duplicates (keep first if same hour)
        df = df.drop_duplicates(subset=['open_time_ms'], keep='first')
        
        return df[['open_time_ms', 'funding_rate']].rename(columns={'open_time_ms': 'open_time'})
        
    except Exception as e:
        logger.error(f"Error loading funding rates from {json_path}: {e}")
        return None


def process_klines_data(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """
    Process raw klines data and add proper datetime columns.
    
    Args:
        df: Raw klines DataFrame
        prefix: Prefix for column names (e.g., "mark_" for mark price)
    
    Returns:
        Processed DataFrame
    """
    processed = df.copy()
    
    # Convert timestamps to datetime
    processed['datetime'] = pd.to_datetime(processed['open_time'], unit='ms', utc=True)
    
    # Create result with prefixed column names
    p = prefix
    result = pd.DataFrame({
        'open_time': processed['open_time'],
        'datetime': processed['datetime'],
        f'{p}open': processed['open'],
        f'{p}high': processed['high'],
        f'{p}low': processed['low'],
        f'{p}close': processed['close'],
        f'{p}volume': processed['volume'],
        'close_time': processed['close_time'],
        f'{p}quote_volume': processed['quote_volume'],
        f'{p}trade_count': processed['count'],
        f'{p}taker_buy_volume': processed['taker_buy_volume'],
        f'{p}taker_buy_quote_volume': processed['taker_buy_quote_volume'],
    })
    
    return result


def combine_all_data(
    klines_df: Optional[pd.DataFrame],
    mark_df: Optional[pd.DataFrame],
    metrics_df: Optional[pd.DataFrame],
    funding_df: Optional[pd.DataFrame],
    symbol: str
) -> tuple[Optional[pd.DataFrame], dict]:
    """
    Combine all data sources into a single DataFrame, aligned by timestamp.
    
    Args:
        klines_df: Klines DataFrame (main OHLCV)
        mark_df: Mark price klines DataFrame
        metrics_df: Metrics DataFrame (resampled to hourly)
        funding_df: Funding rate DataFrame
        symbol: Symbol name for logging
    
    Returns:
        Tuple of (combined DataFrame, stats dict)
    """
    stats = {
        'symbol': symbol,
        'start_datetime': None,
        'end_datetime': None,
        'total_rows': 0,
        'rows_removed': 0,
        'original_rows': 0,
    }
    
    if klines_df is None or len(klines_df) == 0:
        logger.warning(f"No klines data for {symbol}")
        return None, stats
    
    # Process klines as base
    base_df = process_klines_data(klines_df, prefix="")
    stats['original_rows'] = len(base_df)
    
    # Merge mark price data if available
    if mark_df is not None and len(mark_df) > 0:
        mark_processed = process_klines_data(mark_df, prefix="mark_")
        # Keep only mark-specific columns (not open_time, datetime, close_time which are duplicates)
        mark_cols = ['open_time', 'mark_open', 'mark_high', 'mark_low', 'mark_close']
        mark_processed = mark_processed[mark_cols]
        
        base_df = base_df.merge(mark_processed, on='open_time', how='left')
        logger.debug(f"Merged mark price data: {len(mark_processed)} rows")
    
    # Merge metrics data if available
    if metrics_df is not None and len(metrics_df) > 0:
        # Convert create_time to open_time (ms timestamp)
        metrics_df = metrics_df.copy()
        metrics_df['open_time'] = (metrics_df['create_time'].astype('int64') // 10**6).astype('int64')
        
        # Select metrics columns to merge
        metrics_cols = ['open_time', 'sum_open_interest', 'sum_open_interest_value',
                       'count_toptrader_long_short_ratio', 'sum_toptrader_long_short_ratio',
                       'count_long_short_ratio', 'sum_taker_long_short_vol_ratio']
        metrics_cols = [c for c in metrics_cols if c in metrics_df.columns]
        metrics_subset = metrics_df[metrics_cols].drop_duplicates(subset=['open_time'], keep='first')
        
        base_df = base_df.merge(metrics_subset, on='open_time', how='left')
        logger.debug(f"Merged metrics data: {len(metrics_subset)} rows")
    
    # Merge funding rate data if available
    if funding_df is not None and len(funding_df) > 0:
        base_df = base_df.merge(funding_df, on='open_time', how='left')
        logger.debug(f"Merged funding rate data: {len(funding_df)} rows")
    
    # Add date column
    base_df['date'] = base_df['datetime'].dt.date
    
    # Reorder columns - put common columns first
    common_cols = ['date', 'datetime', 'open_time', 'open', 'high', 'low', 'close', 'volume',
                   'close_time', 'quote_volume', 'trade_count', 'taker_buy_volume', 'taker_buy_quote_volume']
    mark_cols = ['mark_open', 'mark_high', 'mark_low', 'mark_close']
    metrics_cols = ['sum_open_interest', 'sum_open_interest_value',
                    'count_toptrader_long_short_ratio', 'sum_toptrader_long_short_ratio',
                    'count_long_short_ratio', 'sum_taker_long_short_vol_ratio']
    funding_cols = ['funding_rate']
    
    # Build final column order
    final_cols = []
    for col in common_cols + mark_cols + metrics_cols + funding_cols:
        if col in base_df.columns:
            final_cols.append(col)
    
    # Add any remaining columns
    for col in base_df.columns:
        if col not in final_cols:
            final_cols.append(col)
    
    base_df = base_df[final_cols]
    
    # Find first row where all added columns have values (not NaN at the start)
    # Define columns that were added via merge
    added_cols = ['mark_open', 'sum_open_interest', 'funding_rate']
    added_cols = [c for c in added_cols if c in base_df.columns]
    
    if added_cols:
        # Find first row where all added columns are not NaN
        first_valid_idx = 0
        for col in added_cols:
            col_first_valid = base_df[col].first_valid_index()
            if col_first_valid is not None:
                first_valid_idx = max(first_valid_idx, col_first_valid)
        
        # Remove rows from beginning where columns are not set
        if first_valid_idx > 0:
            stats['rows_removed'] = first_valid_idx
            base_df = base_df.iloc[first_valid_idx:].reset_index(drop=True)
    
    # Update stats
    stats['total_rows'] = len(base_df)
    if len(base_df) > 0:
        stats['start_datetime'] = base_df['datetime'].iloc[0]
        stats['end_datetime'] = base_df['datetime'].iloc[-1]
    
    return base_df, stats


# ============================================================================
# FEATURE ENGINEERING MODULE
# ============================================================================
# Calculates 25 advanced features for LSTM meta-labeling:
#
# VPVR (Volume Profile) Features:
# 1. dist_to_poc          - Distance to Point of Control (fair value)
# 2. dist_to_hvn          - Distance to High Volume Node (S/R detection)
# 3. vol_above_poc_pct    - Volume above POC ratio (bearish pressure)
# 4. va_position          - Position within Value Area (-1 to +1)
#
# FVG (Fair Value Gap) Features:
# 5. fvg_bull_dist        - Distance to nearest bullish FVG (support)
# 6. fvg_bear_dist        - Distance to nearest bearish FVG (resistance)
# 7. fvg_imbalance        - Net FVG imbalance (-1 to +1)
#
# Order Flow & Volume:
# 8. aggression_score     - Order flow imbalance (stationary CVD proxy)
# 9. volume_zscore        - Volume anomaly detection (whale footprints)
# 10. oi_zscore           - Open interest anomaly detection
#
# Trend & Momentum:
# 11. sma_slope_1h        - Short-term trend direction
# 12. sma_slope_4h        - Medium-term trend direction
# 13. rsi_divergence      - RSI divergence (better than raw RSI)
# 14. own_log_ret_6       - Fast own momentum (6-period)
# 15. momentum_diff       - Trend acceleration (fast - slow momentum)
#
# Price Structure:
# 16. close_zscore        - Normalized price position
# 17. atr_normalized      - Normalized volatility
# 18. volatility_percentile - Volatility regime (0-1 percentile)
#
# Cross-Asset & Sentiment:
# 19. btc_log_ret         - BTC sentiment (market leader)
# 20. corr_btc            - Rolling correlation with BTC (regime)
# 21. funding_zscore      - Crowd sentiment (contrarian signal)
#
# Time Encoding:
# 22. hour_sin            - Cyclic hour encoding (sine)
# 23. hour_cos            - Cyclic hour encoding (cosine)
# 24. day_sin             - Cyclic day-of-week encoding (sine) - weekend effect!
# 25. day_cos             - Cyclic day-of-week encoding (cosine)
# ============================================================================


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate RSI (Relative Strength Index).
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate ATR (Average True Range).
    """
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    
    return atr


# ============================================================================
# NUMBA JIT-COMPILED VPVR (Volume Profile Visible Range)
# ============================================================================
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logger.warning("Numba not installed - VPVR will use slower Python fallback")
    # Create dummy decorator
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


@njit(parallel=True)
def _calculate_vvrp_fast(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    lookback: int,
    num_bins: int
) -> tuple:
    """
    Numba JIT-compiled VPVR calculation with enhanced outputs.
    
    Returns 6 features per candle:
    - poc: Point of Control (highest volume price level)
    - hvn: High Volume Node (2nd highest volume level)
    - dist_to_poc: Normalized distance from close to POC
    - dist_to_hvn: Normalized distance from close to HVN
    - vol_above_poc_pct: % of volume above POC (bearish if high)
    - va_position: Position within Value Area (-1 to +1, 0 = at POC)
    """
    n = len(highs)
    poc = np.zeros(n)
    hvn = np.zeros(n)
    dist_to_poc = np.zeros(n)
    dist_to_hvn = np.zeros(n)
    vol_above_poc_pct = np.zeros(n)
    va_position = np.zeros(n)

    for i in prange(n):
        if i < lookback:
            # Not enough data - use current close as all values
            poc[i] = closes[i]
            hvn[i] = closes[i]
            dist_to_poc[i] = 0.0
            dist_to_hvn[i] = 0.0
            vol_above_poc_pct[i] = 0.5
            va_position[i] = 0.0
            continue
            
        window_highs = highs[i-lookback:i]
        window_lows = lows[i-lookback:i]
        window_volumes = volumes[i-lookback:i]
        
        min_p = np.min(window_lows)
        max_p = np.max(window_highs)
        
        if max_p == min_p:
            poc[i] = closes[i]
            hvn[i] = closes[i]
            dist_to_poc[i] = 0.0
            dist_to_hvn[i] = 0.0
            vol_above_poc_pct[i] = 0.5
            va_position[i] = 0.0
            continue
            
        bin_size = (max_p - min_p) / num_bins
        bins = np.zeros(num_bins)
        
        # Distribute volume across bins
        for j in range(lookback):
            c_high = window_highs[j]
            c_low = window_lows[j]
            c_vol = window_volumes[j]
            
            start_bin = int((c_low - min_p) / bin_size)
            end_bin = int((c_high - min_p) / bin_size)
            
            # Clamp bins
            start_bin = max(0, min(start_bin, num_bins - 1))
            end_bin = max(0, min(end_bin, num_bins - 1))
            
            num_bins_covered = end_bin - start_bin + 1
            vol_per_bin = c_vol / num_bins_covered
            
            for b in range(start_bin, end_bin + 1):
                bins[b] += vol_per_bin
        
        # POC: Bin with max volume
        max_idx = np.argmax(bins)
        poc_price = min_p + (max_idx + 0.5) * bin_size
        poc[i] = poc_price
        
        # HVN: 2nd highest volume node
        bins_copy = bins.copy()
        bins_copy[max_idx] = 0  # Zero out POC bin
        second_max_idx = np.argmax(bins_copy)
        hvn_price = min_p + (second_max_idx + 0.5) * bin_size
        hvn[i] = hvn_price
        
        # Distances (normalized by price)
        close_price = closes[i]
        if close_price > 0:
            dist_to_poc[i] = (close_price - poc_price) / close_price
            dist_to_hvn[i] = (close_price - hvn_price) / close_price
        
        # Volume above POC ratio (bearish pressure if > 0.5)
        total_vol = np.sum(bins)
        vol_above = np.sum(bins[max_idx+1:]) if max_idx < num_bins - 1 else 0.0
        if total_vol > 0:
            vol_above_poc_pct[i] = vol_above / total_vol
        else:
            vol_above_poc_pct[i] = 0.5
        
        # Value Area Position: Where is price relative to 70% volume area?
        # Calculate Value Area (bins containing 70% of volume around POC)
        va_volume_target = total_vol * 0.70
        va_volume = bins[max_idx]
        va_low_idx = max_idx
        va_high_idx = max_idx
        
        while va_volume < va_volume_target and (va_low_idx > 0 or va_high_idx < num_bins - 1):
            # Add bin with more volume (expand left or right)
            left_vol = bins[va_low_idx - 1] if va_low_idx > 0 else 0
            right_vol = bins[va_high_idx + 1] if va_high_idx < num_bins - 1 else 0
            
            if left_vol >= right_vol and va_low_idx > 0:
                va_low_idx -= 1
                va_volume += bins[va_low_idx]
            elif va_high_idx < num_bins - 1:
                va_high_idx += 1
                va_volume += bins[va_high_idx]
            elif va_low_idx > 0:
                va_low_idx -= 1
                va_volume += bins[va_low_idx]
            else:
                break
        
        # Value Area boundaries
        val = min_p + va_low_idx * bin_size  # Value Area Low
        vah = min_p + (va_high_idx + 1) * bin_size  # Value Area High
        
        # Position within Value Area: -1 (below VAL), 0 (at POC), +1 (above VAH)
        if close_price < val:
            va_position[i] = -1.0  # Below value area (potential long)
        elif close_price > vah:
            va_position[i] = 1.0   # Above value area (potential short)
        else:
            # Within value area: normalize to [-0.5, 0.5]
            va_range = vah - val
            if va_range > 0:
                va_position[i] = (close_price - poc_price) / va_range
            else:
                va_position[i] = 0.0

    return poc, hvn, dist_to_poc, dist_to_hvn, vol_above_poc_pct, va_position


# ============================================================================
# FVG (Fair Value Gap) Detection
# ============================================================================
@njit
def _detect_fvg_fast(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    min_gap_pct: float = 0.001  # Minimum 0.1% gap to count
) -> tuple:
    """
    Detect Fair Value Gaps (FVGs) and calculate distance to nearest unfilled gaps.
    
    Bullish FVG: Gap between candle[i-2].high and candle[i].low (price jumped up)
    Bearish FVG: Gap between candle[i-2].low and candle[i].high (price jumped down)
    
    Returns:
    - fvg_bull_dist: Normalized distance to nearest bullish FVG below (support)
    - fvg_bear_dist: Normalized distance to nearest bearish FVG above (resistance)
    - fvg_imbalance: Net imbalance score (-1 to +1, negative = more bearish FVGs above)
    """
    n = len(highs)
    fvg_bull_dist = np.zeros(n)
    fvg_bear_dist = np.zeros(n)
    fvg_imbalance = np.zeros(n)
    
    # Track active (unfilled) FVGs
    # Store as: [bottom_price, top_price, creation_index]
    max_fvgs = 100  # Maximum FVGs to track
    bull_fvgs = np.zeros((max_fvgs, 3))  # Bullish FVGs (below price = support)
    bear_fvgs = np.zeros((max_fvgs, 3))  # Bearish FVGs (above price = resistance)
    num_bull_fvgs = 0
    num_bear_fvgs = 0
    
    for i in range(2, n):
        current_close = closes[i]
        current_high = highs[i]
        current_low = lows[i]
        
        # =====================================================================
        # 1. Detect new FVGs
        # =====================================================================
        
        # Bullish FVG: candle[i-2].high < candle[i].low (gap up)
        prev2_high = highs[i-2]
        curr_low = lows[i]
        if curr_low > prev2_high:
            gap_size = curr_low - prev2_high
            gap_pct = gap_size / current_close
            if gap_pct >= min_gap_pct and num_bull_fvgs < max_fvgs:
                # Add new bullish FVG
                bull_fvgs[num_bull_fvgs, 0] = prev2_high  # Bottom of gap
                bull_fvgs[num_bull_fvgs, 1] = curr_low     # Top of gap
                bull_fvgs[num_bull_fvgs, 2] = i            # Creation index
                num_bull_fvgs += 1
        
        # Bearish FVG: candle[i-2].low > candle[i].high (gap down)
        prev2_low = lows[i-2]
        curr_high = highs[i]
        if prev2_low > curr_high:
            gap_size = prev2_low - curr_high
            gap_pct = gap_size / current_close
            if gap_pct >= min_gap_pct and num_bear_fvgs < max_fvgs:
                # Add new bearish FVG
                bear_fvgs[num_bear_fvgs, 0] = curr_high     # Bottom of gap
                bear_fvgs[num_bear_fvgs, 1] = prev2_low     # Top of gap
                bear_fvgs[num_bear_fvgs, 2] = i             # Creation index
                num_bear_fvgs += 1
        
        # =====================================================================
        # 2. Check for filled FVGs and remove them
        # =====================================================================
        
        # Bullish FVGs are filled if price drops into the gap
        j = 0
        while j < num_bull_fvgs:
            gap_top = bull_fvgs[j, 1]
            gap_bottom = bull_fvgs[j, 0]
            # Filled if price enters the gap (low touches or goes below gap top)
            if current_low <= gap_top:
                # Remove this FVG by shifting array
                for k in range(j, num_bull_fvgs - 1):
                    bull_fvgs[k, 0] = bull_fvgs[k+1, 0]
                    bull_fvgs[k, 1] = bull_fvgs[k+1, 1]
                    bull_fvgs[k, 2] = bull_fvgs[k+1, 2]
                num_bull_fvgs -= 1
            else:
                j += 1
        
        # Bearish FVGs are filled if price rises into the gap
        j = 0
        while j < num_bear_fvgs:
            gap_bottom = bear_fvgs[j, 0]
            gap_top = bear_fvgs[j, 1]
            # Filled if price enters the gap (high touches or goes above gap bottom)
            if current_high >= gap_bottom:
                # Remove this FVG
                for k in range(j, num_bear_fvgs - 1):
                    bear_fvgs[k, 0] = bear_fvgs[k+1, 0]
                    bear_fvgs[k, 1] = bear_fvgs[k+1, 1]
                    bear_fvgs[k, 2] = bear_fvgs[k+1, 2]
                num_bear_fvgs -= 1
            else:
                j += 1
        
        # =====================================================================
        # 3. Calculate distances to nearest FVGs
        # =====================================================================
        
        # Distance to nearest bullish FVG (below current price = support)
        nearest_bull_dist = 1.0  # Default: no FVG nearby (max distance)
        for j in range(num_bull_fvgs):
            gap_top = bull_fvgs[j, 1]
            if gap_top < current_close:  # FVG is below current price
                dist = (current_close - gap_top) / current_close
                if dist < nearest_bull_dist:
                    nearest_bull_dist = dist
        fvg_bull_dist[i] = nearest_bull_dist
        
        # Distance to nearest bearish FVG (above current price = resistance)
        nearest_bear_dist = 1.0  # Default: no FVG nearby
        for j in range(num_bear_fvgs):
            gap_bottom = bear_fvgs[j, 0]
            if gap_bottom > current_close:  # FVG is above current price
                dist = (gap_bottom - current_close) / current_close
                if dist < nearest_bear_dist:
                    nearest_bear_dist = dist
        fvg_bear_dist[i] = nearest_bear_dist
        
        # =====================================================================
        # 4. Calculate imbalance score
        # =====================================================================
        # Positive = more bullish FVGs below (support) = bullish
        # Negative = more bearish FVGs above (resistance) = bearish
        total_fvgs = num_bull_fvgs + num_bear_fvgs
        if total_fvgs > 0:
            fvg_imbalance[i] = (num_bull_fvgs - num_bear_fvgs) / total_fvgs
        else:
            fvg_imbalance[i] = 0.0
    
    return fvg_bull_dist, fvg_bear_dist, fvg_imbalance


def engineer_features(df: pd.DataFrame, btc_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Calculate all LSTM features from raw OHLCV + metrics data.
    
    Args:
        df: DataFrame with columns: open, high, low, close, volume, 
            taker_buy_volume, sum_open_interest, funding_rate, datetime
        btc_df: Optional BTC DataFrame for cross-asset features
        
    Returns:
        DataFrame with all engineered features added
    """
    result = df.copy()
    
    # Ensure datetime index for time-based operations
    if 'datetime' in result.columns:
        result = result.set_index('datetime', drop=False)
    
    # =========================================================================
    # 1. AGGRESSION SCORE (Stationary CVD Proxy)
    # =========================================================================
    # Buy aggression = taker_buy_volume / total_volume
    # Score = (buy_aggression - 0.5) * 2  -> normalized to [-1, 1]
    if 'taker_buy_volume' in result.columns and 'volume' in result.columns:
        buy_ratio = result['taker_buy_volume'] / result['volume'].replace(0, np.nan)
        # Stationary: use the deviation from mean, not cumulative
        result['aggression_score'] = (buy_ratio - 0.5) * 2
        # Fill any NaN with 0 (neutral)
        result['aggression_score'] = result['aggression_score'].fillna(0)
    else:
        result['aggression_score'] = 0
        logger.warning("  Missing taker_buy_volume or volume - aggression_score set to 0")
    
    # =========================================================================
    # 2. SMA SLOPES (Trend Direction)
    # =========================================================================
    # 1h SMA slope (using 12 periods for 1h data = half day)
    sma_1h = result['close'].rolling(window=12, min_periods=1).mean()
    result['sma_slope_1h'] = sma_1h.diff(1) / sma_1h.shift(1)  # Percentage change
    result['sma_slope_1h'] = result['sma_slope_1h'].fillna(0)
    
    # 4h SMA slope (using 48 periods for 1h data = 2 days)
    sma_4h = result['close'].rolling(window=48, min_periods=1).mean()
    result['sma_slope_4h'] = sma_4h.diff(4) / sma_4h.shift(4)  # 4-period change
    result['sma_slope_4h'] = result['sma_slope_4h'].fillna(0)
    
    # =========================================================================
    # 3. VPVR FEATURES (Volume Profile Visible Range)
    # =========================================================================
    # Calculate VPVR using the fast Numba implementation (6 outputs)
    poc, hvn, dist_to_poc_raw, dist_to_hvn_raw, vol_above_poc_pct, va_position = _calculate_vvrp_fast(
        highs=result['high'].values,
        lows=result['low'].values,
        closes=result['close'].values,
        volumes=result['volume'].values,
        lookback=200,
        num_bins=50
    )
    
    # 3a. Distance to POC (Point of Control)
    result['dist_to_poc'] = dist_to_poc_raw
    result['dist_to_poc'] = result['dist_to_poc'].fillna(0).clip(-0.5, 0.5)
    
    # 3b. Distance to HVN (High Volume Node - 2nd highest volume level)
    # Useful for support/resistance detection
    result['dist_to_hvn'] = dist_to_hvn_raw
    result['dist_to_hvn'] = result['dist_to_hvn'].fillna(0).clip(-0.5, 0.5)
    
    # 3c. Volume Above POC Ratio (bearish pressure indicator)
    # > 0.5 means more volume above = overhead resistance
    # < 0.5 means more volume below = underlying support
    result['vol_above_poc_pct'] = vol_above_poc_pct
    result['vol_above_poc_pct'] = result['vol_above_poc_pct'].fillna(0.5)
    
    # 3d. Value Area Position (-1 below VAL, 0 at POC, +1 above VAH)
    # Contrarian signal: extreme values often revert to mean
    result['va_position'] = va_position
    result['va_position'] = result['va_position'].fillna(0)
    
    # =========================================================================
    # 4. FVG FEATURES (Fair Value Gaps)
    # =========================================================================
    # Calculate FVG using the fast Numba implementation (3 outputs)
    fvg_bull_dist, fvg_bear_dist, fvg_imbalance = _detect_fvg_fast(
        highs=result['high'].values,
        lows=result['low'].values,
        closes=result['close'].values,
        min_gap_pct=0.001  # Minimum 0.1% gap
    )
    
    # 4a. Distance to nearest bullish FVG (support below)
    # Lower = stronger support nearby = bullish
    result['fvg_bull_dist'] = fvg_bull_dist
    result['fvg_bull_dist'] = result['fvg_bull_dist'].clip(0, 0.1)  # Cap at 10%
    
    # 4b. Distance to nearest bearish FVG (resistance above)
    # Lower = stronger resistance nearby = bearish
    result['fvg_bear_dist'] = fvg_bear_dist
    result['fvg_bear_dist'] = result['fvg_bear_dist'].clip(0, 0.1)  # Cap at 10%
    
    # 4c. FVG Imbalance score
    # Positive = more bullish FVGs (support) = bullish
    # Negative = more bearish FVGs (resistance) = bearish
    result['fvg_imbalance'] = fvg_imbalance
    result['fvg_imbalance'] = result['fvg_imbalance'].fillna(0)
    
    # =========================================================================
    # 5. CLOSE Z-SCORE (Normalized Price Position)
    # =========================================================================
    # Rolling z-score over 100 periods
    close_mean = result['close'].rolling(window=100, min_periods=20).mean()
    close_std = result['close'].rolling(window=100, min_periods=20).std()
    result['close_zscore'] = (result['close'] - close_mean) / close_std.replace(0, np.nan)
    result['close_zscore'] = result['close_zscore'].fillna(0).clip(-5, 5)  # Clip extremes
    
    # =========================================================================
    # 5. BTC LOG RETURNS (Market Sentiment)
    # =========================================================================
    if btc_df is not None and len(btc_df) > 0:
        # Align BTC data with current symbol using datetime column
        # First, ensure btc_df has datetime as a column (not just index)
        btc_temp = btc_df.copy()
        if 'datetime' not in btc_temp.columns and btc_temp.index.name == 'datetime':
            btc_temp = btc_temp.reset_index()
        
        # Merge on datetime to align BTC close prices
        if 'datetime' in result.columns and 'datetime' in btc_temp.columns:
            btc_close_map = btc_temp.set_index('datetime')['close'].to_dict()
            result['btc_close'] = result['datetime'].map(btc_close_map)
            result['btc_close'] = result['btc_close'].ffill()  # Forward fill gaps
            result['btc_log_ret'] = np.log(result['btc_close'] / result['btc_close'].shift(1))
            result['btc_log_ret'] = result['btc_log_ret'].fillna(0)
            result = result.drop(columns=['btc_close'])  # Clean up temp column
        else:
            # Fallback: use own log returns
            result['btc_log_ret'] = np.log(result['close'] / result['close'].shift(1))
            result['btc_log_ret'] = result['btc_log_ret'].fillna(0)
    else:
        # Use own log returns as fallback (for BTC itself)
        result['btc_log_ret'] = np.log(result['close'] / result['close'].shift(1))
        result['btc_log_ret'] = result['btc_log_ret'].fillna(0)
    
    # =========================================================================
    # 6. ATR NORMALIZED (Volatility)
    # =========================================================================
    atr = calculate_atr(result['high'], result['low'], result['close'], period=14)
    result['atr_normalized'] = atr / result['close'].replace(0, np.nan)
    result['atr_normalized'] = result['atr_normalized'].fillna(0)
    
    # =========================================================================
    # 7. RSI DIVERGENCE (Better than raw RSI)
    # =========================================================================
    # RSI divergence = difference between price direction and RSI direction
    # Positive divergence: price making lows but RSI making highs (bullish)
    # Negative divergence: price making highs but RSI making lows (bearish)
    rsi = calculate_rsi(result['close'], period=14)
    
    # Calculate slopes over 5 periods
    price_slope = result['close'].diff(5) / result['close'].shift(5).abs()
    rsi_slope = rsi.diff(5)
    
    # Divergence: when price and RSI move in opposite directions
    # Normalize RSI slope to similar scale as price slope
    rsi_slope_normalized = rsi_slope / 100  # RSI is 0-100
    
    result['rsi_divergence'] = rsi_slope_normalized - price_slope
    result['rsi_divergence'] = result['rsi_divergence'].fillna(0).clip(-1, 1)
    
    # =========================================================================
    # 8. FUNDING Z-SCORE (Crowd Sentiment - Contrarian)
    # =========================================================================
    if 'funding_rate' in result.columns:
        # Rolling z-score of funding rate (extreme values = contrarian signal)
        fr_mean = result['funding_rate'].rolling(window=168, min_periods=24).mean()  # 7 days
        fr_std = result['funding_rate'].rolling(window=168, min_periods=24).std()
        result['funding_zscore'] = (result['funding_rate'] - fr_mean) / fr_std.replace(0, np.nan)
        result['funding_zscore'] = result['funding_zscore'].fillna(0).clip(-5, 5)
    else:
        result['funding_zscore'] = 0
        logger.warning("  Missing funding_rate - funding_zscore set to 0")
    
    # =========================================================================
    # 9. CORRELATION WITH BTC (Regime Detection)
    # =========================================================================
    if btc_df is not None and len(btc_df) > 0:
        # Rolling correlation with BTC returns using datetime alignment
        btc_temp = btc_df.copy()
        if 'datetime' not in btc_temp.columns and btc_temp.index.name == 'datetime':
            btc_temp = btc_temp.reset_index()
        
        if 'datetime' in result.columns and 'datetime' in btc_temp.columns:
            # Map BTC close to result using datetime
            btc_close_map = btc_temp.set_index('datetime')['close'].to_dict()
            result['btc_close_temp'] = result['datetime'].map(btc_close_map)
            result['btc_close_temp'] = result['btc_close_temp'].ffill()
            
            btc_ret = np.log(result['btc_close_temp'] / result['btc_close_temp'].shift(1))
            own_ret = np.log(result['close'] / result['close'].shift(1))
            
            result['corr_btc'] = own_ret.rolling(window=168, min_periods=24).corr(btc_ret)
            result['corr_btc'] = result['corr_btc'].fillna(0)
            result = result.drop(columns=['btc_close_temp'])
        else:
            # Fallback
            result['corr_btc'] = 1.0
    else:
        # For BTC itself, set to 1 (perfect self-correlation)
        result['corr_btc'] = 1.0
    
    # =========================================================================
    # 10. VOLUME Z-SCORE (Whale Footprints)
    # =========================================================================
    vol_mean = result['volume'].rolling(window=168, min_periods=24).mean()
    vol_std = result['volume'].rolling(window=168, min_periods=24).std()
    result['volume_zscore'] = (result['volume'] - vol_mean) / vol_std.replace(0, np.nan)
    result['volume_zscore'] = result['volume_zscore'].fillna(0).clip(-5, 5)
    
    # =========================================================================
    # 11. OPEN INTEREST Z-SCORE
    # =========================================================================
    if 'sum_open_interest' in result.columns:
        oi_mean = result['sum_open_interest'].rolling(window=168, min_periods=24).mean()
        oi_std = result['sum_open_interest'].rolling(window=168, min_periods=24).std()
        result['oi_zscore'] = (result['sum_open_interest'] - oi_mean) / oi_std.replace(0, np.nan)
        result['oi_zscore'] = result['oi_zscore'].fillna(0).clip(-5, 5)
    else:
        result['oi_zscore'] = 0
        logger.warning("  Missing sum_open_interest - oi_zscore set to 0")
    
    # =========================================================================
    # 12. TIME ENCODING (Cyclic Hour of Day + Day of Week)
    # =========================================================================
    if 'datetime' in result.columns:
        dt = pd.to_datetime(result['datetime'])
        hour = dt.dt.hour
        day_of_week = dt.dt.dayofweek  # Monday=0, Sunday=6
    else:
        hour = result.index.hour
        day_of_week = result.index.dayofweek
    
    # Hour encoding (captures intraday patterns)
    result['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    result['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    
    # Day of week encoding (captures weekend effect - critical in crypto!)
    result['day_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    result['day_cos'] = np.cos(2 * np.pi * day_of_week / 7)
    
    # =========================================================================
    # 13. VOLATILITY PERCENTILE (Regime Detection)
    # =========================================================================
    # Where is current volatility relative to recent history? (0-1 scale)
    # High percentile = extreme volatility = different market behavior
    atr = calculate_atr(result['high'], result['low'], result['close'], period=14)
    result['volatility_percentile'] = atr.rolling(window=168, min_periods=24).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    result['volatility_percentile'] = result['volatility_percentile'].fillna(0.5)
    
    # =========================================================================
    # 14. OWN MOMENTUM (Asset's own returns at different scales)
    # =========================================================================
    # Fast momentum: 6-period log return (~6 hours)
    result['own_log_ret_6'] = np.log(result['close'] / result['close'].shift(6))
    result['own_log_ret_6'] = result['own_log_ret_6'].fillna(0).clip(-0.5, 0.5)
    
    # Slow momentum: 24-period log return (~1 day)
    own_log_ret_24 = np.log(result['close'] / result['close'].shift(24))
    own_log_ret_24 = own_log_ret_24.fillna(0)
    
    # Momentum difference: Trend acceleration
    # Positive = accelerating up (fast > slow)
    # Negative = decelerating or reversing
    result['momentum_diff'] = result['own_log_ret_6'] - own_log_ret_24
    result['momentum_diff'] = result['momentum_diff'].fillna(0).clip(-0.3, 0.3)
    
    # Reset index if we set it
    if result.index.name == 'datetime':
        result = result.reset_index(drop=True)
    
    return result


def save_to_feather(df: pd.DataFrame, output_path: Path) -> bool:
    """
    Save DataFrame to feather format.
    
    Args:
        df: DataFrame to save
        output_path: Path to save the feather file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_feather(output_path)
        logger.info(f"Saved: {output_path} ({len(df)} rows)")
        return True
        
    except Exception as e:
        logger.error(f"Error saving {output_path}: {e}")
        return False


def condense_symbol(
    symbol: str,
    klines_files: list[Path],
    mark_files: list[Path],
    metrics_files: list[Path],
    funding_dir: Path,
    output_dir: Path,
    timeframe: str = "1h",
    btc_df: Optional[pd.DataFrame] = None
) -> dict:
    """
    Condense all data for a single symbol into one feather file.
    
    Args:
        symbol: Binance symbol (e.g., "BTCUSDC")
        klines_files: List of klines CSV files
        mark_files: List of mark price klines CSV files
        metrics_files: List of metrics CSV files
        funding_dir: Directory containing funding rate JSON files
        output_dir: Output directory for feather files
        timeframe: Target timeframe
        btc_df: Optional BTC DataFrame for cross-asset feature calculation
    
    Returns:
        Dictionary with results (including 'combined_df' for BTC reuse)
    """
    symbol_ccxt = convert_binance_to_ccxt_symbol(symbol)
    
    logger.info(f"Processing {symbol} ({symbol_ccxt}) @ {timeframe}")
    
    # Load klines data
    logger.debug(f"Loading klines: {len(klines_files)} files")
    klines_df = read_and_concat_klines(klines_files)
    if klines_df is not None:
        logger.info(f"  Klines: {len(klines_df)} rows")
    
    # Load mark price data
    logger.debug(f"Loading mark prices: {len(mark_files)} files")
    mark_df = read_and_concat_klines(mark_files)
    if mark_df is not None:
        logger.info(f"  Mark prices: {len(mark_df)} rows")
    
    # Load metrics data
    logger.debug(f"Loading metrics: {len(metrics_files)} files")
    metrics_df = read_and_concat_metrics(metrics_files)
    if metrics_df is not None:
        original_metrics_rows = len(metrics_df)
        # Resample to hourly
        metrics_df = resample_metrics_to_hourly(metrics_df)
        if metrics_df is not None:
            logger.info(f"  Metrics: {original_metrics_rows} -> {len(metrics_df)} rows (resampled to 1h)")
    
    # Load funding rate data
    funding_df = load_funding_rates(symbol_ccxt, funding_dir)
    if funding_df is not None:
        logger.info(f"  Funding rates: {len(funding_df)} rows")
    
    # Combine all data
    combined_df, stats = combine_all_data(klines_df, mark_df, metrics_df, funding_df, symbol)
    
    if combined_df is None or len(combined_df) == 0:
        logger.warning(f"No combined data for {symbol}")
        return {'symbol': symbol, 'success': False, 'stats': stats, 'combined_df': None}
    
    # Log data range info
    logger.info(f"  Combined: {stats['total_rows']} rows")
    logger.info(f"  Start: {stats['start_datetime']}")
    logger.info(f"  End: {stats['end_datetime']}")
    logger.info(f"  Rows removed (missing values at start): {stats['rows_removed']}")
    
    # ==========================================================================
    # FEATURE ENGINEERING
    # ==========================================================================
    logger.info(f"  Engineering features...")
    try:
        combined_df = engineer_features(combined_df, btc_df=btc_df)
        
        # Count engineered features
        feature_cols = [
            'aggression_score', 'sma_slope_1h', 'sma_slope_4h', 'dist_to_poc',
            'close_zscore', 'btc_log_ret', 'atr_normalized', 'rsi_divergence',
            'funding_zscore', 'corr_btc', 'volume_zscore', 'oi_zscore',
            'hour_sin', 'hour_cos'
        ]
        available_features = [f for f in feature_cols if f in combined_df.columns]
        logger.info(f"  Engineered {len(available_features)} features: {', '.join(available_features)}")
        
        # Remove rows with NaN in feature columns (from rolling windows)
        initial_rows = len(combined_df)
        combined_df = combined_df.dropna(subset=available_features)
        dropped = initial_rows - len(combined_df)
        if dropped > 0:
            logger.info(f"  Dropped {dropped} rows with NaN in features (rolling warmup)")
            stats['rows_removed'] += dropped
            stats['total_rows'] = len(combined_df)
        
    except Exception as e:
        logger.error(f"  Feature engineering failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Save to feather
    output_path = output_dir / f"{symbol_ccxt}-{timeframe}-combined.feather"
    success = save_to_feather(combined_df, output_path)
    
    return {
        'symbol': symbol,
        'symbol_ccxt': symbol_ccxt,
        'success': success,
        'output_file': str(output_path) if success else None,
        'stats': stats,
        'combined_df': combined_df  # Return for BTC reuse
    }


def main():
    """
    Main function to condense all downloaded data into combined feather files.
    """
    logger.info("=" * 60)
    logger.info("Binance Data Condenser - Combined Data")
    logger.info("=" * 60)
    
    base_dir = Path(RAW_DATA_DIR)
    funding_dir = Path(FUNDING_RATES_DIR)
    output_dir = Path(OUTPUT_DIR)
    timeframe = TARGET_TIMEFRAME
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir.absolute()}")
    
    # Collect all CSV files for each data type
    logger.info("Collecting CSV files...")
    klines_by_symbol = collect_csv_files(base_dir, "klines", timeframe)
    mark_by_symbol = collect_csv_files(base_dir, "markPriceKlines", timeframe)
    metrics_by_symbol = collect_csv_files(base_dir, "metrics", None)  # No timeframe for metrics
    
    logger.info(f"Found klines for {len(klines_by_symbol)} symbols")
    logger.info(f"Found mark prices for {len(mark_by_symbol)} symbols")
    logger.info(f"Found metrics for {len(metrics_by_symbol)} symbols")
    
    # Get all unique symbols
    all_symbols = set(klines_by_symbol.keys())
    logger.info(f"Total symbols to process: {len(all_symbols)}")
    
    # ==========================================================================
    # PROCESS BTC FIRST (for cross-asset features)
    # ==========================================================================
    btc_df = None
    btc_symbols = [s for s in all_symbols if s.startswith('BTC')]
    
    if btc_symbols:
        # Process first BTC symbol (usually BTCUSDC or BTCUSDT)
        btc_symbol = sorted(btc_symbols)[0]
        logger.info("=" * 60)
        logger.info(f"PROCESSING BTC FIRST ({btc_symbol}) FOR CROSS-ASSET FEATURES")
        logger.info("=" * 60)
        
        try:
            btc_result = condense_symbol(
                symbol=btc_symbol,
                klines_files=klines_by_symbol.get(btc_symbol, []),
                mark_files=mark_by_symbol.get(btc_symbol, []),
                metrics_files=metrics_by_symbol.get(btc_symbol, []),
                funding_dir=funding_dir,
                output_dir=output_dir,
                timeframe=timeframe,
                btc_df=None  # BTC uses itself
            )
            
            if btc_result['success'] and btc_result.get('combined_df') is not None:
                btc_df = btc_result['combined_df']
                logger.info(f"BTC data ready: {len(btc_df)} rows for cross-asset features")
            
            # Remove BTC from remaining symbols to process
            all_symbols = all_symbols - {btc_symbol}
            
        except Exception as e:
            logger.error(f"Failed to process BTC ({btc_symbol}): {e}")
            import traceback
            logger.error(traceback.format_exc())
    else:
        logger.warning("No BTC symbol found - cross-asset features will use fallback")
    
    # ==========================================================================
    # PROCESS REMAINING SYMBOLS
    # ==========================================================================
    logger.info("=" * 60)
    logger.info("PROCESSING REMAINING SYMBOLS")
    logger.info("=" * 60)
    
    results = []
    
    # Add BTC result if we processed it
    if 'btc_result' in dir() and btc_result['success']:
        results.append(btc_result)
    
    for symbol in sorted(all_symbols):
        logger.info("-" * 40)
        
        klines_files = klines_by_symbol.get(symbol, [])
        mark_files = mark_by_symbol.get(symbol, [])
        metrics_files = metrics_by_symbol.get(symbol, [])
        
        try:
            result = condense_symbol(
                symbol=symbol,
                klines_files=klines_files,
                mark_files=mark_files,
                metrics_files=metrics_files,
                funding_dir=funding_dir,
                output_dir=output_dir,
                timeframe=timeframe,
                btc_df=btc_df  # Pass BTC data for cross-asset features
            )
            results.append(result)
            
        except Exception as e:
            logger.error(f"Failed to process {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    # Summary
    logger.info("=" * 60)
    logger.info("CONDENSATION SUMMARY")
    logger.info("=" * 60)
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    logger.info(f"Total symbols processed: {len(results)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    
    total_rows = sum(r['stats']['total_rows'] for r in results)
    total_removed = sum(r['stats']['rows_removed'] for r in results)
    
    logger.info(f"Total rows in output: {total_rows:,}")
    logger.info(f"Total rows removed (missing values): {total_removed:,}")
    
    # Detailed per-symbol summary
    logger.info("")
    logger.info("Per-symbol summary:")
    logger.info("-" * 80)
    logger.info(f"{'Symbol':<20} {'Start':<22} {'End':<22} {'Rows':>10} {'Removed':>10}")
    logger.info("-" * 80)
    
    for r in sorted(results, key=lambda x: x['symbol']):
        s = r['stats']
        start_str = str(s['start_datetime'])[:19] if s['start_datetime'] else 'N/A'
        end_str = str(s['end_datetime'])[:19] if s['end_datetime'] else 'N/A'
        logger.info(f"{r['symbol']:<20} {start_str:<22} {end_str:<22} {s['total_rows']:>10,} {s['rows_removed']:>10,}")
    
    logger.info("-" * 80)
    
    # List created files
    logger.info("\nCreated files:")
    for feather_file in sorted(output_dir.glob("*-combined.feather")):
        size_mb = feather_file.stat().st_size / (1024 * 1024)
        logger.info(f"  - {feather_file.name} ({size_mb:.2f} MB)")
    
    logger.info("=" * 60)
    logger.info("Condensation completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
