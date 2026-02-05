"""Comprehensive debug to find the exact merge failure point."""
import json
import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 70)
print("COMPREHENSIVE MERGE DEBUG")
print("=" * 70)

# ============================================================================
# STEP 1: Load klines exactly like condense.py
# ============================================================================
print("\n[1] LOADING KLINES")
print("-" * 50)

KLINE_DTYPES = {
    'open_time': 'int64', 'open': 'float64', 'high': 'float64', 'low': 'float64',
    'close': 'float64', 'volume': 'float64', 'close_time': 'int64',
    'quote_volume': 'float64', 'count': 'int64', 'taker_buy_volume': 'float64',
    'taker_buy_quote_volume': 'float64', 'ignore': 'int64'
}

klines_dir = Path('./binance_data/futures/um/daily/klines/BTCUSDC/1h')
csv_files = sorted(list(klines_dir.glob("*.csv")))
print(f"Found {len(csv_files)} klines files")

dfs = []
for f in csv_files:
    df = pd.read_csv(f, dtype=KLINE_DTYPES, on_bad_lines='skip')
    dfs.append(df)

klines_df = pd.concat(dfs, ignore_index=True)
klines_df = klines_df.drop_duplicates(subset=['open_time'], keep='first')
klines_df = klines_df.sort_values('open_time').reset_index(drop=True)

print(f"Klines: {len(klines_df)} rows")
print(f"Klines open_time range: {klines_df['open_time'].min()} to {klines_df['open_time'].max()}")
print(f"Klines open_time dtype: {klines_df['open_time'].dtype}")

# Process like process_klines_data does
klines_df['datetime'] = pd.to_datetime(klines_df['open_time'], unit='ms', utc=True)
print(f"Klines datetime range: {klines_df['datetime'].min()} to {klines_df['datetime'].max()}")

# ============================================================================
# STEP 2: Load and resample metrics exactly like condense.py
# ============================================================================
print("\n[2] LOADING & RESAMPLING METRICS")
print("-" * 50)

metrics_dir = Path('./binance_data/futures/um/daily/metrics/BTCUSDC')
csv_files = sorted(list(metrics_dir.glob("*.csv")))
print(f"Found {len(csv_files)} metrics files")

dfs = []
for f in csv_files:
    df = pd.read_csv(f, on_bad_lines='skip')
    dfs.append(df)

metrics_raw = pd.concat(dfs, ignore_index=True)
metrics_raw['create_time'] = pd.to_datetime(metrics_raw['create_time'], utc=True)
metrics_raw = metrics_raw.drop_duplicates(subset=['create_time'], keep='first')
metrics_raw = metrics_raw.sort_values('create_time').reset_index(drop=True)

print(f"Raw metrics: {len(metrics_raw)} rows")
print(f"Raw metrics time range: {metrics_raw['create_time'].min()} to {metrics_raw['create_time'].max()}")

# Resample exactly like resample_metrics_to_hourly
df_resample = metrics_raw.copy()
df_resample = df_resample.set_index('create_time')

agg_dict = {
    'sum_open_interest': 'last',
    'sum_open_interest_value': 'last',
    'count_toptrader_long_short_ratio': 'mean',
    'sum_toptrader_long_short_ratio': 'mean',
    'count_long_short_ratio': 'mean',
    'sum_taker_long_short_vol_ratio': 'mean'
}
agg_dict = {k: v for k, v in agg_dict.items() if k in df_resample.columns}

metrics_resampled = df_resample.resample('1h', label='left', closed='left').agg(agg_dict)
metrics_resampled = metrics_resampled.reset_index()

print(f"Resampled metrics: {len(metrics_resampled)} rows")
print(f"Resampled create_time dtype: {metrics_resampled['create_time'].dtype}")
print(f"Resampled time range: {metrics_resampled['create_time'].min()} to {metrics_resampled['create_time'].max()}")

# ============================================================================
# STEP 3: Convert metrics create_time to open_time like combine_all_data
# ============================================================================
print("\n[3] CONVERTING METRICS TO OPEN_TIME")
print("-" * 50)

metrics_df = metrics_resampled.copy()
# This is line 441 from condense.py:
metrics_df['open_time'] = (metrics_df['create_time'].astype('int64') // 10**6).astype('int64')

print(f"Metrics open_time dtype: {metrics_df['open_time'].dtype}")
print(f"Metrics open_time range: {metrics_df['open_time'].min()} to {metrics_df['open_time'].max()}")

# ============================================================================
# STEP 4: Check timestamp overlap
# ============================================================================
print("\n[4] CHECKING TIMESTAMP OVERLAP")
print("-" * 50)

klines_set = set(klines_df['open_time'].tolist())
metrics_set = set(metrics_df['open_time'].tolist())
overlap = klines_set.intersection(metrics_set)

print(f"Klines unique timestamps: {len(klines_set)}")
print(f"Metrics unique timestamps: {len(metrics_set)}")
print(f"OVERLAP: {len(overlap)}")

if len(overlap) == 0:
    print("\n*** CRITICAL: NO OVERLAP! ***")
    print(f"Sample klines open_time: {sorted(list(klines_set))[:5]}")
    print(f"Sample metrics open_time: {sorted(list(metrics_set))[:5]}")
    
    # Check if there's a timezone issue
    k_sample = sorted(list(klines_set))[:3]
    m_sample = sorted(list(metrics_set))[:3]
    print(f"\nAs datetimes:")
    for ts in k_sample:
        print(f"  Klines: {ts} -> {pd.to_datetime(ts, unit='ms', utc=True)}")
    for ts in m_sample:
        print(f"  Metrics: {ts} -> {pd.to_datetime(ts, unit='ms', utc=True)}")
else:
    print(f"Sample overlapping timestamps: {list(overlap)[:5]}")

# ============================================================================
# STEP 5: Try the actual merge
# ============================================================================
print("\n[5] ATTEMPTING MERGE")
print("-" * 50)

# Create base_df like process_klines_data would
base_df = klines_df[['open_time', 'datetime', 'open', 'high', 'low', 'close', 'volume']].copy()

# Select metrics columns to merge (like combine_all_data line 444-448)
metrics_cols = ['open_time', 'sum_open_interest', 'sum_open_interest_value',
               'count_toptrader_long_short_ratio', 'sum_toptrader_long_short_ratio',
               'count_long_short_ratio', 'sum_taker_long_short_vol_ratio']
metrics_cols = [c for c in metrics_cols if c in metrics_df.columns]
metrics_subset = metrics_df[metrics_cols].drop_duplicates(subset=['open_time'], keep='first')

print(f"Merging on 'open_time'...")
print(f"base_df open_time dtype: {base_df['open_time'].dtype}")
print(f"metrics_subset open_time dtype: {metrics_subset['open_time'].dtype}")

merged = base_df.merge(metrics_subset, on='open_time', how='left')
post_merge = merged['sum_open_interest'].notna().sum() if 'sum_open_interest' in merged.columns else 0

print(f"RESULT: {post_merge} non-null OI values out of {len(merged)} rows")

if post_merge == 0:
    print("\n*** MERGE FAILED! ***")
    # Debug further
    print(f"\nbase_df open_time sample values: {base_df['open_time'].iloc[:5].values}")
    print(f"metrics_subset open_time sample values: {metrics_subset['open_time'].iloc[:5].values}")
    print(f"\nbase_df open_time type: {type(base_df['open_time'].iloc[0])}")
    print(f"metrics_subset open_time type: {type(metrics_subset['open_time'].iloc[0])}")
