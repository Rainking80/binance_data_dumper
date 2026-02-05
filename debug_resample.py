"""Debug the resampled metrics to understand why merge fails."""
import pandas as pd
from pathlib import Path

print("=" * 60)
print("DEBUGGING RESAMPLED METRICS")
print("=" * 60)

# Load and resample metrics the same way condense.py does
metrics_dir = Path('./binance_data/futures/um/daily/metrics/BTCUSDC')
csv_files = sorted(list(metrics_dir.glob("*.csv")))[:10]

dfs = []
for f in csv_files:
    df = pd.read_csv(f)
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
combined['create_time'] = pd.to_datetime(combined['create_time'], utc=True)
combined = combined.drop_duplicates(subset=['create_time'], keep='first')
combined = combined.sort_values('create_time').reset_index(drop=True)

print(f"1. After read_and_concat_metrics:")
print(f"   Shape: {combined.shape}")
print(f"   Columns: {list(combined.columns)}")
print(f"   create_time dtype: {combined['create_time'].dtype}")
print(f"   create_time sample: {combined['create_time'].iloc[:3].tolist()}")

# Now resample like resample_metrics_to_hourly() does
df = combined.copy()
df = df.set_index('create_time')

agg_dict = {
    'sum_open_interest': 'last',
    'sum_open_interest_value': 'last',
    'count_toptrader_long_short_ratio': 'mean',
    'sum_toptrader_long_short_ratio': 'mean',
    'count_long_short_ratio': 'mean',
    'sum_taker_long_short_vol_ratio': 'mean'
}
agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}

resampled = df.resample('1h', label='left', closed='left').agg(agg_dict)
resampled = resampled.reset_index()

print(f"\n2. After resample_metrics_to_hourly:")
print(f"   Shape: {resampled.shape}")
print(f"   Columns: {list(resampled.columns)}")
print(f"   create_time dtype: {resampled['create_time'].dtype}")
print(f"   create_time sample: {resampled['create_time'].iloc[:3].tolist()}")

# Now simulate what combine_all_data does (line 441)
print(f"\n3. Simulating combine_all_data conversion:")
print(f"   Before conversion - create_time dtype: {resampled['create_time'].dtype}")

# This is line 441 from condense.py:
metrics_df = resampled.copy()
metrics_df['open_time'] = (metrics_df['create_time'].astype('int64') // 10**6).astype('int64')

print(f"   After conversion - open_time sample: {metrics_df['open_time'].iloc[:3].tolist()}")

# Compare to klines timestamps
klines_dir = Path('./binance_data/futures/um/daily/klines/BTCUSDC/1h')
klines_files = sorted(list(klines_dir.glob("*.csv")))

# Find a file that overlaps
KLINE_DTYPES = {'open_time': 'int64', 'open': 'float64', 'high': 'float64', 'low': 'float64', 
                'close': 'float64', 'volume': 'float64', 'close_time': 'int64', 
                'quote_volume': 'float64', 'count': 'int64', 'taker_buy_volume': 'float64',
                'taker_buy_quote_volume': 'float64', 'ignore': 'int64'}

# Get a sample from early 2024
for f in klines_files:
    if '2024-01' in f.name:
        df = pd.read_csv(f, dtype=KLINE_DTYPES)
        print(f"\n4. Klines from {f.name}:")
        print(f"   open_time sample: {df['open_time'].iloc[:3].tolist()}")
        
        # Convert to readable
        sample_ts = df['open_time'].iloc[0]
        print(f"   First timestamp as datetime: {pd.to_datetime(sample_ts, unit='ms')}")
        
        # Compare to metrics
        metrics_ts = metrics_df['open_time'].iloc[0]
        print(f"   First metrics timestamp as datetime: {pd.to_datetime(metrics_ts, unit='ms')}")
        
        # Check overlap
        klines_times = set(df['open_time'].tolist())
        metrics_times = set(metrics_df['open_time'].tolist())
        overlap = klines_times.intersection(metrics_times)
        print(f"   Overlapping: {len(overlap)}")
        
        if len(overlap) > 0:
            print(f"   Sample overlap: {list(overlap)[:3]}")
        break
