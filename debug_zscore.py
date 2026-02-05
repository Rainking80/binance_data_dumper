"""Debug script to investigate why funding_zscore and oi_zscore are always 0."""
import json
import pandas as pd
from pathlib import Path

print("=" * 60)
print("DEBUGGING Z-SCORE ISSUES")
print("=" * 60)

# ============================================================================
# 1. Check funding rate loading and timestamp alignment
# ============================================================================
print("\n1. FUNDING RATE DATA")
print("-" * 40)

json_path = Path('./binance_data/funding_rates/BTC_USDC_USDC-1h-funding_rate.json')
with open(json_path, 'r') as f:
    data = json.load(f)

df_fr = pd.DataFrame(data, columns=['timestamp', 'funding_rate', 'fr2', 'fr3', 'fr4', 'ignore'])
df_fr = df_fr[['timestamp', 'funding_rate']].copy()

# Convert timestamp to datetime
df_fr['datetime'] = pd.to_datetime(df_fr['timestamp'], unit='ms', utc=True)

# Floor to hour to ensure alignment
df_fr['open_time'] = df_fr['datetime'].dt.floor('h')
df_fr['open_time_ms'] = (df_fr['open_time'].astype('int64') // 10**6).astype('int64')

print(f"Loaded funding rate entries: {len(df_fr)}")
print(f"Open time ms samples: {df_fr['open_time_ms'].iloc[:5].tolist()}")
print(f"Time range: {df_fr['datetime'].iloc[0]} to {df_fr['datetime'].iloc[-1]}")

# ============================================================================
# 2. Check condensed data
# ============================================================================
print("\n2. CONDENSED DATA")
print("-" * 40)

cdf = pd.read_feather('condensed_data/BTC_USDC_USDC-1h-combined.feather')
print(f"Condensed shape: {cdf.shape}")
print(f"Open time samples: {cdf['open_time'].iloc[:5].tolist()}")
print(f"Time range: {cdf['datetime'].iloc[0]} to {cdf['datetime'].iloc[-1]}")

# ============================================================================
# 3. Check timestamp overlap
# ============================================================================
print("\n3. TIMESTAMP OVERLAP")
print("-" * 40)

funding_times = set(df_fr['open_time_ms'].tolist())
condensed_times = set(cdf['open_time'].tolist())
overlap = funding_times.intersection(condensed_times)
print(f"Funding rate unique timestamps: {len(funding_times)}")
print(f"Condensed unique timestamps: {len(condensed_times)}")
print(f"OVERLAPPING timestamps: {len(overlap)}")

if len(overlap) == 0:
    print("\n*** NO OVERLAP! Checking why...")
    # Sample from each
    fr_sample = sorted(list(funding_times))[:5]
    cd_sample = sorted(list(condensed_times))[:5]
    print(f"Funding samples: {fr_sample}")
    print(f"Condensed samples: {cd_sample}")
    
    # Check data types
    print(f"\nFunding open_time dtype: {df_fr['open_time_ms'].dtype}")
    print(f"Condensed open_time dtype: {cdf['open_time'].dtype}")

# ============================================================================
# 4. Check metrics (open interest) data
# ============================================================================
print("\n4. METRICS (OPEN INTEREST) DATA")
print("-" * 40)

metrics_dir = Path('./binance_data/futures/um/daily/metrics')
if metrics_dir.exists():
    symbols = list(metrics_dir.iterdir())
    print(f"Metrics directory contains: {len(symbols)} symbol directories")
    
    # Find BTCUSDC
    btc_dir = metrics_dir / "BTCUSDC"
    if btc_dir.exists():
        csv_files = sorted(list(btc_dir.glob("*.csv")))
        print(f"BTCUSDC has {len(csv_files)} CSV files")
        
        if csv_files:
            df_m = pd.read_csv(csv_files[0])
            print(f"First CSV columns: {list(df_m.columns)}")
            print(f"First CSV rows: {len(df_m)}")
            
            # Check create_time format
            print(f"create_time sample: {df_m['create_time'].iloc[:3].tolist()}")
            
            # Parse and convert to check alignment
            df_m['create_time'] = pd.to_datetime(df_m['create_time'], utc=True)
            # Convert to ms
            first_ts_ms = int(df_m['create_time'].iloc[0].timestamp() * 1000)
            print(f"First timestamp (ms): {first_ts_ms}")
            
            # Check if it would match with klines
            floored = df_m['create_time'].dt.floor('h')
            floored_ms = (floored.astype('int64') // 10**6).astype('int64')
            print(f"Floored first timestamp (ms): {floored_ms.iloc[0]}")
    else:
        print("BTCUSDC directory not found in metrics!")
        print(f"Available: {[s.name for s in symbols[:5]]}")
else:
    print("Metrics directory does not exist!")

# ============================================================================
# 5. Check how condense.py collects and maps symbols
# ============================================================================
print("\n5. SYMBOL MAPPING CHECK")
print("-" * 40)

# The issue might be in how symbols are matched
print("Binance symbol: BTCUSDC")
# From condense.py convert_binance_to_ccxt_symbol logic:
binance_ticker = "BTCUSDC"
quotes = ['USDC', 'USDT', 'BUSD', 'BTC', 'ETH']
for quote in quotes:
    if binance_ticker.endswith(quote):
        base = binance_ticker[:-len(quote)]
        ccxt = f"{base}_{quote}_{quote}"
        print(f"Converted: {binance_ticker} -> {ccxt}")
        break

# Check what condense.py actually uses for metrics lookup
print("\nLooking for the symbol mapping in collect_csv_files...")
