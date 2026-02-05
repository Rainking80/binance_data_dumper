"""Debug why merges fail despite overlapping timestamps."""
import json
import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 60)
print("DEBUGGING MERGE FAILURE")
print("=" * 60)

# ============================================================================
# 1. Simulate the exact funding rate merge logic
# ============================================================================
print("\n1. FUNDING RATE MERGE SIMULATION")
print("-" * 40)

# Load funding rate the same way condense.py does
json_path = Path('./binance_data/funding_rates/BTC_USDC_USDC-1h-funding_rate.json')
with open(json_path, 'r') as f:
    data = json.load(f)

df_fr = pd.DataFrame(data, columns=['timestamp', 'funding_rate', 'fr2', 'fr3', 'fr4', 'ignore'])
df_fr = df_fr[['timestamp', 'funding_rate']].copy()
df_fr['datetime'] = pd.to_datetime(df_fr['timestamp'], unit='ms', utc=True)
df_fr['open_time'] = df_fr['datetime'].dt.floor('h')
df_fr['open_time_ms'] = (df_fr['open_time'].astype('int64') // 10**6).astype('int64')

# Rename to match condense.py
funding_df = df_fr[['open_time_ms', 'funding_rate']].rename(columns={'open_time_ms': 'open_time'})

print(f"Funding DF shape: {funding_df.shape}")
print(f"Funding open_time dtype: {funding_df['open_time'].dtype}")
print(f"Funding open_time sample: {funding_df['open_time'].iloc[:3].tolist()}")

# ============================================================================
# 2. Load a raw kline CSV to simulate base_df
# ============================================================================
print("\n2. RAW KLINE SIMULATION") 
print("-" * 40)

# Find kline files
klines_dir = Path('./binance_data/futures/um/daily/klines/BTCUSDC/1h')
csv_files = sorted(list(klines_dir.glob("*.csv")))[:5]  # Just a few files

# Read with the same dtypes as condense.py
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

dfs = []
for f in csv_files:
    df = pd.read_csv(f, dtype=KLINE_DTYPES)
    dfs.append(df)
    
klines_df = pd.concat(dfs, ignore_index=True)
klines_df = klines_df.drop_duplicates(subset=['open_time'], keep='first')
klines_df = klines_df.sort_values('open_time')

print(f"Klines shape: {klines_df.shape}")
print(f"Klines open_time dtype: {klines_df['open_time'].dtype}")
print(f"Klines open_time sample: {klines_df['open_time'].iloc[:3].tolist()}")

# ============================================================================
# 3. Try the merge
# ============================================================================
print("\n3. MERGE TEST")
print("-" * 40)

# Confirm data type compatibility
print(f"Klines open_time dtype: {klines_df['open_time'].dtype}")
print(f"Funding open_time dtype: {funding_df['open_time'].dtype}")

# Check if values match
klines_times = set(klines_df['open_time'].tolist())
funding_times = set(funding_df['open_time'].tolist())
overlap = klines_times.intersection(funding_times)
print(f"Klines unique times: {len(klines_times)}")
print(f"Funding unique times: {len(funding_times)}")
print(f"Overlapping: {len(overlap)}")

if len(overlap) > 0:
    print(f"Sample overlap: {list(overlap)[:5]}")

# Do the actual merge
merged = klines_df.merge(funding_df, on='open_time', how='left')
non_null = merged['funding_rate'].notna().sum()
print(f"\nMerge result: {non_null} non-null funding_rate values out of {len(merged)} rows")

# ============================================================================
# 4. Check what the CONDENSED file has for open_time
# ============================================================================
print("\n4. CONDENSED FILE CHECK")
print("-" * 40)

cdf = pd.read_feather('condensed_data/BTC_USDC_USDC-1h-combined.feather')
print(f"Condensed open_time dtype: {cdf['open_time'].dtype}")
print(f"Condensed open_time sample: {cdf['open_time'].iloc[:3].tolist()}")

# Compare types
print(f"\nKlines type: {type(klines_df['open_time'].iloc[0])}")
print(f"Funding type: {type(funding_df['open_time'].iloc[0])}")
print(f"Condensed type: {type(cdf['open_time'].iloc[0])}")

# ============================================================================
# 5. Check metrics merge
# ============================================================================
print("\n5. METRICS MERGE SIMULATION")
print("-" * 40)

metrics_dir = Path('./binance_data/futures/um/daily/metrics/BTCUSDC')
csv_files = sorted(list(metrics_dir.glob("*.csv")))[:5]

dfs = []
for f in csv_files:
    df = pd.read_csv(f)
    dfs.append(df)

metrics_df = pd.concat(dfs, ignore_index=True)
metrics_df['create_time'] = pd.to_datetime(metrics_df['create_time'], utc=True)
metrics_df = metrics_df.drop_duplicates(subset=['create_time'], keep='first')
metrics_df = metrics_df.sort_values('create_time')

print(f"Metrics shape: {metrics_df.shape}")
print(f"Metrics create_time dtype: {metrics_df['create_time'].dtype}")

# Convert create_time to open_time (ms timestamp) - same as condense.py line 441
metrics_df['open_time'] = (metrics_df['create_time'].astype('int64') // 10**6).astype('int64')
print(f"Metrics open_time dtype: {metrics_df['open_time'].dtype}")
print(f"Metrics open_time sample: {metrics_df['open_time'].iloc[:3].tolist()}")

# Check overlap with klines
metrics_times = set(metrics_df['open_time'].tolist())
overlap = klines_times.intersection(metrics_times)
print(f"Metrics unique times: {len(metrics_times)}")
print(f"Overlapping with klines: {len(overlap)}")

if len(overlap) == 0:
    # Show sample values to understand the mismatch
    print("\n*** NO OVERLAP! ***")
    print(f"Sample klines times: {sorted(list(klines_times))[:5]}")
    print(f"Sample metrics times: {sorted(list(metrics_times))[:5]}")
    
    # Convert to readable dates
    klines_sample_dt = [pd.to_datetime(t, unit='ms') for t in sorted(list(klines_times))[:3]]
    metrics_sample_dt = [pd.to_datetime(t, unit='ms') for t in sorted(list(metrics_times))[:3]]
    print(f"Klines dates: {klines_sample_dt}")
    print(f"Metrics dates: {metrics_sample_dt}")
