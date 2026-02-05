"""
Data Analysis Script
====================
Investigates:
1. Why FVG/funding features are showing 0 importance
2. Label distribution for class imbalance
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Features to investigate
ZERO_IMPORTANCE_FEATURES = [
    'fvg_bull_dist', 'fvg_bear_dist', 'fvg_imbalance',
    'funding_zscore', 'oi_zscore'
]

NEGATIVE_IMPORTANCE_FEATURES = [
    'atr_normalized', 'volume_zscore', 'dist_to_hvn'
]

def analyze_feature(df, feature_name):
    """Analyze a single feature's distribution."""
    if feature_name not in df.columns:
        return {"status": "MISSING", "reason": "Column not in data"}
    
    series = df[feature_name]
    stats = {
        "count": len(series),
        "null_count": series.isna().sum(),
        "null_pct": f"{series.isna().mean()*100:.2f}%",
        "min": series.min(),
        "max": series.max(),
        "mean": series.mean(),
        "std": series.std(),
        "unique_values": series.nunique(),
        "value_counts_top5": series.value_counts().head(5).to_dict(),
    }
    
    # Check for problematic patterns
    issues = []
    if series.std() < 0.0001:
        issues.append("VERY LOW VARIANCE - feature is essentially constant")
    if series.nunique() <= 3:
        issues.append(f"VERY FEW UNIQUE VALUES ({series.nunique()})")
    if (series == 0).mean() > 0.9:
        issues.append(f"90%+ values are 0 ({(series == 0).mean()*100:.1f}%)")
    if (series == series.iloc[0]).mean() > 0.9:
        issues.append(f"90%+ values are identical to first value")
    
    # Check for default values (1.0 for FVG distances means "no FVG nearby")
    if 'fvg' in feature_name.lower() and 'dist' in feature_name.lower():
        pct_at_cap = (series >= 0.1).mean()
        if pct_at_cap > 0.5:
            issues.append(f"CAPPED AT 0.1: {pct_at_cap*100:.1f}% of values are at cap (no FVGs detected)")
    
    stats["issues"] = issues if issues else ["OK"]
    return stats


def analyze_labels(df, close_series=None):
    """Analyze label distribution from get_directional_labels logic."""
    # Recreate labeling to understand distribution
    pass


def main():
    data_dir = Path("condensed_data")
    
    if not data_dir.exists():
        print(f"ERROR: {data_dir} not found!")
        return
    
    feather_files = list(data_dir.glob("*.feather"))
    print(f"Found {len(feather_files)} feather files\n")
    
    if not feather_files:
        print("No feather files found!")
        return
    
    # =========================================================================
    # PART 1: Analyze Features Across All Symbols
    # =========================================================================
    print("=" * 80)
    print("PART 1: FEATURE ANALYSIS (Zero/Negative Importance Features)")
    print("=" * 80)
    
    all_features_to_check = ZERO_IMPORTANCE_FEATURES + NEGATIVE_IMPORTANCE_FEATURES
    
    # Collect stats across all symbols
    feature_summary = {f: {"working": 0, "broken": 0, "details": []} for f in all_features_to_check}
    
    for fpath in feather_files:
        symbol = fpath.stem.replace("-1h-combined", "")
        df = pd.read_feather(fpath)
        
        print(f"\n--- {symbol} ({len(df)} rows) ---")
        
        for feature in all_features_to_check:
            stats = analyze_feature(df, feature)
            
            if stats.get("status") == "MISSING":
                print(f"  {feature}: ⚠️ MISSING")
                feature_summary[feature]["broken"] += 1
                feature_summary[feature]["details"].append(f"{symbol}: MISSING")
            elif stats["issues"] != ["OK"]:
                print(f"  {feature}: ❌ ISSUES: {stats['issues']}")
                print(f"    Range: [{stats['min']:.6f}, {stats['max']:.6f}], Std: {stats['std']:.6f}")
                print(f"    Unique values: {stats['unique_values']}, Top: {list(stats['value_counts_top5'].items())[:3]}")
                feature_summary[feature]["broken"] += 1
                feature_summary[feature]["details"].append(f"{symbol}: {stats['issues']}")
            else:
                print(f"  {feature}: ✅ OK (range: [{stats['min']:.4f}, {stats['max']:.4f}], std: {stats['std']:.4f})")
                feature_summary[feature]["working"] += 1
    
    # Summary
    print("\n" + "=" * 80)
    print("FEATURE HEALTH SUMMARY")
    print("=" * 80)
    for feature, summary in feature_summary.items():
        status = "✅" if summary["broken"] == 0 else "❌" if summary["working"] == 0 else "⚠️"
        print(f"{status} {feature}: {summary['working']} working, {summary['broken']} broken")
        if summary["details"]:
            for detail in summary["details"][:3]:  # Show first 3 issues
                print(f"    - {detail}")
    
    # =========================================================================
    # PART 2: Deep Dive on FVG Features
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 2: DEEP DIVE - FVG FEATURES")
    print("=" * 80)
    
    # Check one symbol in detail
    sample_file = feather_files[0]
    df = pd.read_feather(sample_file)
    symbol = sample_file.stem.replace("-1h-combined", "")
    
    print(f"\nDetailed FVG analysis for {symbol}:")
    
    for fvg_feat in ['fvg_bull_dist', 'fvg_bear_dist', 'fvg_imbalance']:
        if fvg_feat in df.columns:
            series = df[fvg_feat]
            print(f"\n  {fvg_feat}:")
            print(f"    Value distribution:")
            print(f"      - Values = 0.0:   {(series == 0).sum()} ({(series == 0).mean()*100:.1f}%)")
            print(f"      - Values = 0.1:   {(series == 0.1).sum()} ({(series == 0.1).mean()*100:.1f}%) [cap for 'no FVG']")
            print(f"      - Values = 1.0:   {(series == 1.0).sum()} ({(series == 1.0).mean()*100:.1f}%)")
            print(f"      - Other values:   {((series != 0) & (series != 0.1) & (series != 1.0)).sum()}")
            print(f"    Histogram of non-zero values:")
            non_zero = series[series != 0]
            if len(non_zero) > 0:
                print(f"      Min: {non_zero.min():.6f}, Max: {non_zero.max():.6f}, Mean: {non_zero.mean():.6f}")
            else:
                print(f"      ALL VALUES ARE ZERO!")
    
    # =========================================================================
    # PART 3: Funding Rate Analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 3: DEEP DIVE - FUNDING RATE")
    print("=" * 80)
    
    for fpath in feather_files[:3]:  # Check first 3 symbols
        df = pd.read_feather(fpath)
        symbol = fpath.stem.replace("-1h-combined", "")
        
        print(f"\n  {symbol}:")
        
        # Check raw funding_rate first
        if 'funding_rate' in df.columns:
            fr = df['funding_rate']
            print(f"    funding_rate: {fr.notna().sum()} non-null values out of {len(fr)}")
            print(f"      - NaN count: {fr.isna().sum()} ({fr.isna().mean()*100:.1f}%)")
            if fr.notna().sum() > 0:
                print(f"      - Range: [{fr.min():.8f}, {fr.max():.8f}]")
                print(f"      - Unique: {fr.nunique()}")
        else:
            print(f"    funding_rate: COLUMN MISSING!")
        
        # Check funding_zscore
        if 'funding_zscore' in df.columns:
            fz = df['funding_zscore']
            print(f"    funding_zscore: std={fz.std():.6f}, unique={fz.nunique()}")
            if fz.std() < 0.01:
                print(f"      ⚠️ VERY LOW VARIANCE - z-score calculation likely failed")
        else:
            print(f"    funding_zscore: COLUMN MISSING!")
    
    # =========================================================================
    # PART 4: Label Distribution Analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 4: LABEL DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    # Import the labeling logic from train_lstm
    import sys
    sys.path.insert(0, '.')
    
    try:
        from train_lstm import get_directional_labels, get_daily_vol, BIG_CAPS
        
        total_labels = {"long": 0, "short": 0, "nothing": 0}
        
        for fpath in feather_files:
            df = pd.read_feather(fpath)
            symbol = fpath.stem.replace("-1h-combined", "")
            base = symbol.split("_")[0]
            
            # Set datetime index
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.set_index('datetime')
            
            # Calculate labels
            if 'close' in df.columns:
                daily_vol = get_daily_vol(df['close'])
                labels = get_directional_labels(base, df['close'], daily_vol)
                
                if 'label' in labels.columns and len(labels) > 0:
                    counts = labels['label'].value_counts()
                    n_long = counts.get(1, 0)
                    n_short = counts.get(-1, 0)
                    n_nothing = counts.get(0, 0)
                    total = n_long + n_short + n_nothing
                    
                    total_labels["long"] += n_long
                    total_labels["short"] += n_short
                    total_labels["nothing"] += n_nothing
                    
                    print(f"\n  {symbol} (type: {'BIG' if base in BIG_CAPS else 'ALT'}):")
                    print(f"    Long(1):    {n_long:5d} ({n_long/total*100:5.1f}%)")
                    print(f"    Short(-1):  {n_short:5d} ({n_short/total*100:5.1f}%)")
                    print(f"    Nothing(0): {n_nothing:5d} ({n_nothing/total*100:5.1f}%)")
                    
                    if n_nothing / total > 0.6:
                        print(f"    ⚠️ IMBALANCED: 'Nothing' class dominates!")
        
        # Overall summary
        print("\n" + "-" * 40)
        print("OVERALL LABEL DISTRIBUTION:")
        print("-" * 40)
        grand_total = sum(total_labels.values())
        if grand_total > 0:
            print(f"  Long(1):    {total_labels['long']:6d} ({total_labels['long']/grand_total*100:5.1f}%)")
            print(f"  Short(-1):  {total_labels['short']:6d} ({total_labels['short']/grand_total*100:5.1f}%)")
            print(f"  Nothing(0): {total_labels['nothing']:6d} ({total_labels['nothing']/grand_total*100:5.1f}%)")
            
            # Recommendations
            print("\n" + "-" * 40)
            print("RECOMMENDATIONS:")
            print("-" * 40)
            nothing_pct = total_labels['nothing'] / grand_total
            if nothing_pct > 0.5:
                print(f"  1. ❌ SEVERE CLASS IMBALANCE: 'Nothing' is {nothing_pct*100:.1f}% of labels")
                print(f"     Model will learn to always predict 'Nothing' for easy accuracy!")
                print(f"     Options:")
                print(f"       a) Use class_weight in training (already done)")
                print(f"       b) Undersample 'Nothing' class")
                print(f"       c) Lower the barrier thresholds to get more Long/Short labels")
                print(f"       d) Increase min_ret threshold for less choppy labeling")
            
            long_short_ratio = total_labels['long'] / max(total_labels['short'], 1)
            if long_short_ratio > 2 or long_short_ratio < 0.5:
                print(f"  2. ⚠️ LONG/SHORT IMBALANCE: Ratio is {long_short_ratio:.2f}")
                print(f"     Consider checking if data is from a trending up/down period")
    
    except ImportError as e:
        print(f"Could not import train_lstm: {e}")
        print("Skipping label analysis - run this script from the project root")


if __name__ == "__main__":
    main()
