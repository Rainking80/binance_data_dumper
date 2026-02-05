"""Debug script to check funding rate and metrics loading"""
import pandas as pd
from pathlib import Path
import json

# Check funding rate file
funding_file = Path('binance_data/funding_rates/BTC_USDC_USDC-1h-funding_rate.json')
if funding_file.exists():
    with open(funding_file) as f:
        data = json.load(f)
    print(f'Funding rate file found: {len(data)} records')
    if data:
        print(f'Sample: {data[0]}')
        ts = data[0][0]
        print(f'Timestamp: {ts}')
        print(f'As datetime: {pd.to_datetime(ts, unit="ms")}')
else:
    print('Funding file NOT FOUND!')

# Check what the load_funding_rates function returns
from condense import load_funding_rates
funding_df = load_funding_rates('BTC_USDC_USDC', Path('binance_data/funding_rates'))
if funding_df is not None:
    print(f'\nLoaded funding_df: {len(funding_df)} rows')
    print(f'Columns: {funding_df.columns.tolist()}')
    print(funding_df.head(3))
else:
    print('\nload_funding_rates returned None!')

# Check klines open_time to compare
klines_sample = pd.read_feather('condensed_data/BTC_USDC_USDC-1h-combined.feather')
print(f'\nKlines open_time range:')
print(f'  Min: {klines_sample["open_time"].min()} = {pd.to_datetime(klines_sample["open_time"].min(), unit="ms")}')
print(f'  Max: {klines_sample["open_time"].max()} = {pd.to_datetime(klines_sample["open_time"].max(), unit="ms")}')

if funding_df is not None:
    print(f'\nFunding open_time range:')
    print(f'  Min: {funding_df["open_time"].min()} = {pd.to_datetime(funding_df["open_time"].min(), unit="ms")}')
    print(f'  Max: {funding_df["open_time"].max()} = {pd.to_datetime(funding_df["open_time"].max(), unit="ms")}')
    
    # Check overlap
    klines_set = set(klines_sample['open_time'])
    funding_set = set(funding_df['open_time'])
    overlap = klines_set.intersection(funding_set)
    print(f'\nOverlap: {len(overlap)} matching timestamps out of {len(funding_set)} funding records')
