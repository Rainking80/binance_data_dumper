"""
Label Distribution Analysis
"""
import pandas as pd
import numpy as np
from pathlib import Path


def get_daily_vol(close, span0=100):
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0]:])
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1
    df0 = df0.ewm(span=span0).std()
    return df0


BIG_CAPS = ['BTC', 'ETH', 'BNB', 'SOL']


def get_directional_labels(symbol, close, daily_vol):
    base = symbol.split('_')[0] if '_' in symbol else symbol
    asset_type = 'big' if base in BIG_CAPS else 'alt'
    
    t_events = close.index[::10]
    t_events = t_events[t_events.isin(daily_vol.index)]
    
    if len(t_events) == 0:
        return pd.DataFrame(columns=['label'])
    
    # Updated: ALTs now use 24h expiry (was 12h)
    expiry_hours = 48 if asset_type == 'big' else 24
    
    t1 = close.index.searchsorted(t_events + pd.Timedelta(hours=expiry_hours))
    t1 = t1[t1 < close.shape[0]]
    t1 = pd.Series(close.index[t1], index=t_events[:t1.shape[0]])
    
    valid_indices = t1.index.intersection(daily_vol.index)
    if len(valid_indices) == 0:
        return pd.DataFrame(columns=['label'])
    
    t1 = t1.loc[valid_indices]
    target = daily_vol.loc[valid_indices]
    
    # Updated: ALTs now use 1.5% min_ret (was 2%)
    min_ret = 0.0 if asset_type == 'big' else 0.015
    if min_ret > 0:
        target = target.apply(lambda x: max(x, min_ret))
    
    labels = pd.DataFrame(index=valid_indices)
    labels['label'] = 0
    
    for loc, t1_time in t1.items():
        if pd.isna(t1_time):
            continue
        path = close[loc:t1_time]
        if len(path) < 2:
            continue
        entry_price = close[loc]
        trgt = target.loc[loc]
        returns = (path / entry_price) - 1
        
        pt_level = trgt
        sl_level = -trgt
        
        long_tp_times = returns[returns > pt_level].index
        long_sl_times = returns[returns < sl_level].index
        long_tp_hit = long_tp_times.min() if len(long_tp_times) > 0 else pd.NaT
        long_sl_hit = long_sl_times.min() if len(long_sl_times) > 0 else pd.NaT
        
        short_tp_times = returns[returns < sl_level].index
        short_sl_times = returns[returns > pt_level].index
        short_tp_hit = short_tp_times.min() if len(short_tp_times) > 0 else pd.NaT
        short_sl_hit = short_sl_times.min() if len(short_sl_times) > 0 else pd.NaT
        
        long_win = pd.notna(long_tp_hit) and (pd.isna(long_sl_hit) or long_tp_hit < long_sl_hit)
        short_win = pd.notna(short_tp_hit) and (pd.isna(short_sl_hit) or short_tp_hit < short_sl_hit)
        
        if long_win and not short_win:
            labels.loc[loc, 'label'] = 1
        elif short_win and not long_win:
            labels.loc[loc, 'label'] = -1
    
    return labels


def main():
    total = {'long': 0, 'short': 0, 'nothing': 0}
    
    print("=" * 70)
    print("LABEL DISTRIBUTION PER SYMBOL")
    print("=" * 70)
    
    for fpath in sorted(Path('condensed_data').glob('*.feather')):
        df = pd.read_feather(fpath)
        symbol = fpath.stem.replace('-1h-combined', '')
        base = symbol.split('_')[0]
        asset_type = 'BIG' if base in BIG_CAPS else 'ALT'
        
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime')
        
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        if 'close' in df.columns:
            daily_vol = get_daily_vol(df['close'])
            labels = get_directional_labels(symbol, df['close'], daily_vol)
            
            if 'label' in labels.columns and len(labels) > 0:
                counts = labels['label'].value_counts()
                n_long = counts.get(1, 0)
                n_short = counts.get(-1, 0)
                n_nothing = counts.get(0, 0)
                t = n_long + n_short + n_nothing
                
                total['long'] += n_long
                total['short'] += n_short
                total['nothing'] += n_nothing
                
                pct_nothing = n_nothing / t * 100 if t > 0 else 0
                flag = ' << SEVERE IMBALANCE' if pct_nothing > 70 else (' << IMBALANCED' if pct_nothing > 60 else '')
                
                print(f"{symbol:25s} [{asset_type}]: Long={n_long:4d}({n_long/t*100:4.0f}%), "
                      f"Short={n_short:4d}({n_short/t*100:4.0f}%), "
                      f"Nothing={n_nothing:4d}({pct_nothing:4.0f}%){flag}")
    
    # Summary
    print("\n" + "=" * 70)
    print("TOTAL LABEL DISTRIBUTION")
    print("=" * 70)
    
    grand = sum(total.values())
    long_pct = total['long'] / grand * 100
    short_pct = total['short'] / grand * 100
    nothing_pct = total['nothing'] / grand * 100
    
    print(f"  Long:    {total['long']:6d} ({long_pct:5.1f}%)")
    print(f"  Short:   {total['short']:6d} ({short_pct:5.1f}%)")
    print(f"  Nothing: {total['nothing']:6d} ({nothing_pct:5.1f}%)")
    
    # Analysis
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    
    if nothing_pct > 60:
        print(f"\n❌ SEVERE CLASS IMBALANCE DETECTED!")
        print(f"   'Nothing' class represents {nothing_pct:.1f}% of all labels.")
        print(f"   This means the model can achieve ~{nothing_pct:.0f}% accuracy by always predicting 'Nothing'!")
        print(f"\n   ROOT CAUSE: The Triple Barrier Method is too strict.")
        print(f"   Most price movements don't hit TP or SL within the time window,")
        print(f"   so they get labeled as 'Nothing' (time barrier hit first).")
        print(f"\n   SOLUTIONS:")
        print(f"   1. Increase expiry_hours (give more time to hit barriers)")
        print(f"   2. Lower the barrier thresholds (use smaller TP/SL)")
        print(f"   3. Use asymmetric barriers (e.g., 1.5x TP, 1x SL)")
        print(f"   4. Undersample the 'Nothing' class during training")
    
    if abs(long_pct - short_pct) > 10:
        dominant = "Long" if long_pct > short_pct else "Short"
        print(f"\n⚠️  LONG/SHORT IMBALANCE: {dominant} signals dominate ({max(long_pct, short_pct):.1f}% vs {min(long_pct, short_pct):.1f}%)")
        print(f"   This may indicate the training data covers a trending period.")


if __name__ == "__main__":
    main()
