import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import joblib

# ==========================================
# 0. CONFIGURATION & CLASSIFIER
# ==========================================
# Define what you consider "Big Caps" (Pure Volatility Logic)
BIG_CAPS = ['BTC', 'ETH', 'BNB', 'SOL']

def classify_asset(symbol):
    """
    Returns 'big' for major pairs, 'alt' for everything else.
    Assumes format like 'BTC/USDT' or just 'BTC'.
    """
    base_currency = symbol.split('/')[0] if '/' in symbol else symbol
    if base_currency in BIG_CAPS:
        return 'big'
    return 'alt'

# ==========================================
# 1. FRACTIONAL DIFFERENTIATION (The Feature)
# ==========================================
def get_weights_ffd(d, thres):
    w, k = [1.], 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres: break
        w.append(w_)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)

def frac_diff_ffd(series, d, thres=1e-5):
    w = get_weights_ffd(d, thres)
    width = len(w) - 1
    df = {}
    for name in series.columns:
        series_f = series[[name]].ffill().dropna()
        df_ = pd.Series(dtype=float)
        for iloc1 in range(width, series_f.shape[0]):
            loc0, loc1 = series_f.index[iloc1 - width], series_f.index[iloc1]
            if not np.isfinite(series.loc[loc1, name]): continue
            val = np.dot(w.T, series_f.loc[loc0:loc1])[0, 0]
            df_[loc1] = val
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df

# ==========================================
# 2. TRIPLE BARRIER LABELING (The Target)
# ==========================================
def get_daily_vol(close, span0=100):
    """
    Computes dynamic volatility (standard deviation).
    """
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0]:])
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1
    df0 = df0.ewm(span=span0).std()
    return df0

def apply_triple_barrier(close, events, pt_sl, molecule, min_ret=0.0):
    """
    HYBRID LOGIC IMPLEMENTED HERE:
    - If min_ret > 0 (Alts), we force a minimum target floor.
    - If min_ret == 0 (Big Caps), we trust pure volatility.
    """
    events_ = events.loc[molecule]
    out = pd.DataFrame(index=events_.index)
    
    # --- DYNAMIC TARGET CALCULATION ---
    target = events_['trgt']
    
    if min_ret > 0:
        # For ALTS: Ensure target is at least min_ret (e.g., 2%)
        # This prevents training on noise during low-volatility consolidation
        target = target.apply(lambda x: max(x, min_ret))
    
    # Define barriers based on the (possibly modified) target
    if pt_sl[0] > 0: pt = pt_sl[0] * target
    else: pt = pd.Series(index=events.index)  # NaNs
    
    if pt_sl[1] > 0: sl = -pt_sl[1] * target
    else: sl = pd.Series(index=events.index)  # NaNs

    # Search for barrier touches
    for loc, t1 in events_['t1'].fillna(close.index[-1]).items():
        df0 = close[loc:t1]  # Path prices
        df0 = (df0 / close[loc] - 1) * events_.at[loc, 'side'] # Returns
        
        # Earliest stop loss hit
        out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min()
        # Earliest profit take hit
        out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min()
        
    return out

# ==========================================
# 3. META-LABELING SETUP (Dispatcher)
# ==========================================
def get_directional_labels(symbol, close, daily_vol):
    """
    Generate 3-class directional labels using Triple Barrier Method.
    
    Labels:
    - 1 = Long opportunity (price went up and hit TP before SL)
    - -1 = Short opportunity (price went down and hit TP before SL)  
    - 0 = No clear trade (hit time barrier or stopped out both directions)
    
    This is more balanced than binary meta-labeling because:
    - We evaluate BOTH long and short opportunities at each point
    - "Nothing" labels capture choppy/sideways markets
    """
    asset_type = classify_asset(symbol)
    print(f"   -> Processing {symbol} as [{asset_type.upper()}] Strategy (3-class)")

    # A. Sample every N candles (event trigger)
    t_events = close.index[::10]
    t_events = t_events[t_events.isin(daily_vol.index)]
    
    if len(t_events) == 0:
        print(f"   Warning: No valid events for {symbol}")
        return pd.DataFrame(columns=['label'])
    
    # B. Define Vertical Barrier (Time Limit)
    # Big caps (BTC/ETH) need longer expiry due to lower volatility
    expiry_hours = 48 if asset_type == 'big' else 12
    
    t1 = close.index.searchsorted(t_events + pd.Timedelta(hours=expiry_hours))
    t1 = t1[t1 < close.shape[0]]
    t1 = pd.Series(close.index[t1], index=t_events[:t1.shape[0]])
    
    # Only use indices that exist in daily_vol
    valid_indices = t1.index.intersection(daily_vol.index)
    if len(valid_indices) == 0:
        print(f"   Warning: No valid events after filtering for {symbol}")
        return pd.DataFrame(columns=['label'])
    
    t1 = t1.loc[valid_indices]
    target = daily_vol.loc[valid_indices]
    
    # Apply minimum return threshold for alts
    min_ret = 0.0 if asset_type == 'big' else 0.02
    if min_ret > 0:
        target = target.apply(lambda x: max(x, min_ret))
    
    # C. Evaluate BOTH Long and Short at each event
    labels = pd.DataFrame(index=valid_indices)
    labels['label'] = 0  # Default: no trade
    
    for loc, t1_time in t1.items():
        if t1_time is pd.NaT:
            continue
            
        path = close[loc:t1_time]
        if len(path) < 2:
            continue
            
        entry_price = close[loc]
        trgt = target.loc[loc]
        
        # Calculate returns from entry
        returns = (path / entry_price) - 1
        
        # Define barrier levels
        pt_level = trgt   # Take profit level
        sl_level = -trgt  # Stop loss level
        
        # Check for Long: price > +target (TP) before < -target (SL)
        long_tp_times = returns[returns > pt_level].index
        long_sl_times = returns[returns < sl_level].index
        
        long_tp_hit = long_tp_times.min() if len(long_tp_times) > 0 else pd.NaT
        long_sl_hit = long_sl_times.min() if len(long_sl_times) > 0 else pd.NaT
        
        # Check for Short: price < -target (TP for short) before > +target (SL for short)
        short_tp_times = returns[returns < sl_level].index  # Short profits when price drops
        short_sl_times = returns[returns > pt_level].index  # Short stops when price rises
        
        short_tp_hit = short_tp_times.min() if len(short_tp_times) > 0 else pd.NaT
        short_sl_hit = short_sl_times.min() if len(short_sl_times) > 0 else pd.NaT
        
        # Determine label
        long_win = pd.notna(long_tp_hit) and (pd.isna(long_sl_hit) or long_tp_hit < long_sl_hit)
        short_win = pd.notna(short_tp_hit) and (pd.isna(short_sl_hit) or short_tp_hit < short_sl_hit)
        
        if long_win and not short_win:
            labels.loc[loc, 'label'] = 1   # Long opportunity
        elif short_win and not long_win:
            labels.loc[loc, 'label'] = -1  # Short opportunity
        # else: 0 (both hit, neither hit, or conflicting signals)
    
    return labels


# Keep old function for backwards compatibility but mark as deprecated
def get_meta_labels(symbol, close, daily_vol):
    """DEPRECATED: Use get_directional_labels instead for 3-class labeling."""
    return get_directional_labels(symbol, close, daily_vol)

# ==========================================
# 4. LSTM MODEL (Enhanced for 25 features, 3-class output)
# ==========================================
def create_lstm(input_shape, learning_rate=0.001, num_classes=3):
    """
    Create an LSTM model optimized for 25-feature input.
    
    Architecture:
    - 2 LSTM layers with increasing-then-decreasing units
    - Batch normalization for training stability
    - Dropout + L2 regularization to prevent overfitting
    - 3-class output: Long (1), Short (-1/mapped to 2), Nothing (0)
    """
    from tensorflow.keras.regularizers import l2
    
    model = tf.keras.models.Sequential([
        # First LSTM layer - larger to capture feature interactions
        tf.keras.layers.LSTM(
            128, 
            return_sequences=True, 
            input_shape=input_shape,
            kernel_regularizer=l2(1e-4),
            recurrent_regularizer=l2(1e-4)
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        # Second LSTM layer - distill patterns
        tf.keras.layers.LSTM(
            64,
            return_sequences=False,
            kernel_regularizer=l2(1e-4),
            recurrent_regularizer=l2(1e-4)
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        # Dense layers for classification
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l2(1e-4)),
        tf.keras.layers.Dropout(0.2),
        # 3-class output: 0=Nothing, 1=Long, 2=Short
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # Optimizer with configurable learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Use sparse categorical crossentropy (labels are integers, not one-hot)
    model.compile(
        optimizer=optimizer, 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    return model


def get_callbacks(patience=10, min_delta=0.001):
    """
    Create training callbacks for early stopping and learning rate reduction.
    """
    callbacks = [
        # Early stopping - stop if validation loss doesn't improve
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate when plateauing
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-6,
            verbose=1
        )
    ]
    return callbacks

# ==========================================
# MAIN EXECUTION (Loading from Feather Files)
# ==========================================
import os
from pathlib import Path

# ==========================================
# FEATURE COLUMNS FOR LSTM MODEL
# ==========================================
# These 25 features are engineered by condense.py:
# - VPVR: dist_to_poc, dist_to_hvn, vol_above_poc_pct, va_position
# - FVG: fvg_bull_dist, fvg_bear_dist, fvg_imbalance
# - Order Flow: aggression_score, volume_zscore, oi_zscore
# - Trend: sma_slope_1h, sma_slope_4h, rsi_divergence, own_log_ret_6, momentum_diff
# - Price Structure: close_zscore, atr_normalized, volatility_percentile
# - Cross-Asset: btc_log_ret, corr_btc
# - Sentiment: funding_zscore
# - Time: hour_sin, hour_cos, day_sin, day_cos
# ==========================================
FEATURE_COLUMNS = [
    # --- VPVR (Volume Profile) ---
    'dist_to_poc',          # Distance to Point of Control (fair value)
    'dist_to_hvn',          # Distance to High Volume Node (S/R detection)
    'vol_above_poc_pct',    # Volume above POC ratio (bearish pressure)
    'va_position',          # Position within Value Area (-1 to +1)
    
    # --- FVG (Fair Value Gaps) ---
    'fvg_bull_dist',        # Distance to nearest bullish FVG (support)
    'fvg_bear_dist',        # Distance to nearest bearish FVG (resistance)
    'fvg_imbalance',        # Net FVG imbalance (-1 to +1)
    
    # --- ORDER FLOW ---
    'aggression_score',     # Stationary CVD proxy (buy/sell imbalance)
    
    # --- TREND & MOMENTUM ---
    'sma_slope_1h',         # Short-term trend direction
    'sma_slope_4h',         # Medium-term trend direction
    'own_log_ret_6',        # Fast own momentum (6-period)
    'momentum_diff',        # Trend acceleration (fast - slow momentum)
    
    # --- PRICE STRUCTURE ---
    'close_zscore',         # Normalized price position
    
    # --- CROSS-ASSET SENTIMENT ---
    'btc_log_ret',          # BTC momentum (market leader)
    'corr_btc',             # Rolling correlation with BTC (regime detection)
    
    # --- VOLATILITY ---
    'atr_normalized',       # Normalized ATR
    'volatility_percentile', # Volatility regime (0-1, high = extreme vol)
    
    # --- MOMENTUM ---
    'rsi_divergence',       # RSI divergence (better than raw RSI)
    
    # --- DERIVATIVES SENTIMENT ---
    'funding_zscore',       # Crowd sentiment (contrarian signal)
    
    # --- VOLUME ---
    'volume_zscore',        # Volume anomaly detection (whale footprints)
    'oi_zscore',            # Open interest anomaly detection
    
    # --- TIME ENCODING ---
    'hour_sin',             # Cyclic hour encoding (sine)
    'hour_cos',             # Cyclic hour encoding (cosine)
    'day_sin',              # Cyclic day-of-week encoding (sine) - weekend effect!
    'day_cos',              # Cyclic day-of-week encoding (cosine)
]

# Critical columns that must exist (subset of FEATURE_COLUMNS)
REQUIRED_FEATURES = ['dist_to_poc', 'aggression_score', 'close_zscore', 'atr_normalized']

def load_feather_data(data_dir='condensed_data'):
    """
    Load all feather files from the condensed_data directory.
    Returns a dictionary mapping symbol -> DataFrame with proper datetime index.
    
    The feather files are expected to contain engineered features from condense.py.
    """
    data_feeds = {}
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory '{data_dir}' not found!")
    
    feather_files = list(data_path.glob('*.feather'))
    
    if not feather_files:
        raise FileNotFoundError(f"No feather files found in '{data_dir}'!")
    
    print(f"Found {len(feather_files)} feather files in {data_dir}")
    print(f"Expected features: {len(FEATURE_COLUMNS)} ({', '.join(FEATURE_COLUMNS[:5])}...)")
    
    for fpath in feather_files:
        # Extract symbol from filename (e.g., "BTC_USDC_USDC-1h-combined.feather" -> "BTC_USDC_USDC")
        filename = fpath.stem  # e.g., "BTC_USDC_USDC-1h-combined"
        # Get everything before "-1h-combined" or similar pattern
        symbol_full = filename.split('-')[0]  # e.g., "BTC_USDC_USDC"
        base_currency = symbol_full.split('_')[0]  # e.g., "BTC" (for classification)
        
        try:
            df = pd.read_feather(fpath)
            
            # Convert datetime column to index
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.set_index('datetime')
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
            else:
                print(f"  Warning: No datetime column found in {fpath.name}, skipping...")
                continue
            
            # Convert timezone-aware index to timezone-naive (required for get_daily_vol)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            # Sort by index to ensure chronological order
            df = df.sort_index()
            
            # Check for required features (critical columns that must exist)
            missing_required = [f for f in REQUIRED_FEATURES if f not in df.columns]
            if missing_required:
                print(f"  Warning: {fpath.name} missing required features: {missing_required}")
                print(f"           Run condense.py to regenerate data with features")
                continue
            
            # Check which features are available
            available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
            missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
            
            if missing_features:
                print(f"  Note: {fpath.name} missing optional features: {missing_features}")
            
            # Verify 'close' column exists (needed for labeling)
            if 'close' not in df.columns:
                print(f"  Warning: No 'close' column in {fpath.name}, skipping...")
                continue
            
            # Fill any remaining NaN in feature columns (should be minimal after condense.py)
            for col in available_features:
                if df[col].isna().any():
                    nan_count = df[col].isna().sum()
                    df[col] = df[col].ffill().bfill()
                    print(f"    Filled {nan_count} NaN values in {col}")
            
            # Drop rows with any remaining NaN in feature columns
            initial_len = len(df)
            df = df.dropna(subset=available_features)
            if len(df) < initial_len:
                print(f"    Dropped {initial_len - len(df)} rows with NaN values")
            
            print(f"  Loaded {symbol_full}: {len(df)} rows, {len(available_features)}/{len(FEATURE_COLUMNS)} features")
            
            # Store with full symbol name as key
            data_feeds[symbol_full] = {
                'df': df,
                'base_currency': base_currency,  # For BIG_CAPS classification
                'available_features': available_features  # Track which features are present
            }
            
        except Exception as e:
            print(f"  Error loading {fpath.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return data_feeds


if __name__ == "__main__":
    import json
    
    # 1. Create models directory
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # 1. Load Data from Feather Files
    print("--- LOADING DATA FROM FEATHER FILES ---")
    data_feeds = load_feather_data('condensed_data')
    
    if not data_feeds:
        raise ValueError("No valid data feeds loaded! Check your condensed_data directory.")

    # Global lists to accumulate training data from all coins
    X_all = []
    y_all = []
    window_size = 60
    
    # Store per-symbol data for individual training
    symbol_data = {}
    
    # Determine the common features across all symbols
    all_available_features = None
    for symbol_full, feed_data in data_feeds.items():
        features = set(feed_data['available_features'])
        if all_available_features is None:
            all_available_features = features
        else:
            all_available_features = all_available_features.intersection(features)
    
    # Convert to ordered list (maintain original order)
    common_features = [f for f in FEATURE_COLUMNS if f in all_available_features]
    num_features = len(common_features)
    
    print(f"\n--- FEATURE SUMMARY ---")
    print(f"Total features defined: {len(FEATURE_COLUMNS)}")
    print(f"Common features across all symbols: {num_features}")
    print(f"Features: {', '.join(common_features)}")

    print("\n--- STARTING HYBRID PIPELINE ---")
    print("Note: Using pre-engineered features from condense.py (already stationary)")

    for symbol_full, feed_data in data_feeds.items():
        df = feed_data['df']
        base_currency = feed_data['base_currency']
        
        print(f"\nProcessing {symbol_full}...")
        
        # 2. Volatility (based on close price)
        daily_vol = get_daily_vol(df['close'])

        # 3. Labeling (3-class: Long/Short/Nothing)
        # Use base_currency (e.g., "BTC") for classification
        labels = get_directional_labels(base_currency, df['close'], daily_vol)
        
        if 'label' not in labels.columns or len(labels) == 0:
            print(f"   Warning: No valid labels for {symbol_full}, skipping...")
            continue
        
        # Convert labels from (-1, 0, 1) to (0, 1, 2) for sparse categorical crossentropy
        # Mapping: -1 (Short) -> 2, 0 (Nothing) -> 0, 1 (Long) -> 1
        labels['label_mapped'] = labels['label'].map({-1: 2, 0: 0, 1: 1})
        
        # 4. Features - Use pre-engineered features from condense.py
        # These features are already stationary (z-scores, normalized values, etc.)
        # No need for Fractional Differentiation on already stationary features
        feature_df = df[common_features].copy()
        
        # Ensure no NaN values
        feature_df = feature_df.dropna()
        
        if len(feature_df) == 0:
            print(f"   Warning: No valid feature data for {symbol_full}, skipping...")
            continue
        
        # 5. Align Features & Labels
        common_indices = labels.index.intersection(feature_df.index)
        
        if len(common_indices) == 0:
            print(f"   Warning: No common indices for {symbol_full}, skipping...")
            continue
        
        # Collect samples for this symbol
        X_symbol = []
        y_symbol = []
        
        for idx in common_indices:
            i = feature_df.index.get_loc(idx)
            if i < window_size: 
                continue
            
            window = feature_df.iloc[i-window_size:i].values
            label = labels.loc[idx, 'label_mapped']
            X_all.append(window)
            y_all.append(label)
            X_symbol.append(window)
            y_symbol.append(label)
        
        # Store per-symbol data if we have enough samples
        if len(X_symbol) >= 100:  # Minimum samples for individual training
            symbol_data[symbol_full] = {
                'X': np.array(X_symbol),
                'y': np.array(y_symbol)
            }
            print(f"   Added {len(X_symbol)} samples from {symbol_full} (stored for individual training)")
        else:
            print(f"   Added {len(X_symbol)} samples from {symbol_full} (too few for individual training)")
            
    # Convert to Numpy
    X = np.array(X_all)
    y = np.array(y_all)
    
    # Reshape: (samples, timesteps, features)
    actual_num_features = X.shape[2] if len(X.shape) == 3 else 1
    X = X.reshape((X.shape[0], window_size, actual_num_features))

    print(f"\n--- DATA SUMMARY ---")
    print(f"Combined Training Data: X shape: {X.shape}, y shape: {y.shape}")
    print(f"Features per timestep: {actual_num_features}")
    print(f"Label distribution: Nothing(0)={np.sum(y==0)}, Long(1)={np.sum(y==1)}, Short(2)={np.sum(y==2)}")
    print(f"Symbols with enough data for individual training: {len(symbol_data)}")

    # ==========================================
    # 6. Train Global Model with K-Fold Cross-Validation
    # ==========================================
    print("\n--- TRAINING GLOBAL LSTM MODEL (5-Fold Time Series CV) ---")
    print("    (3-class: Long/Short/Nothing)")
    print("    (Batch normalization, L2 regularization, early stopping)")
    print("    (Class weighting to handle imbalance)")
    
    # Training hyperparameters (optimized for 25 features)
    N_FOLDS = 5
    EPOCHS = 50           # More epochs with early stopping
    BATCH_SIZE = 64       # Larger batch for stability with more features
    PATIENCE = 10         # Early stopping patience
    LEARNING_RATE = 0.001 # Initial learning rate
    
    tscv = TimeSeriesSplit(n_splits=N_FOLDS)
    
    fold_results = []
    best_fold_accuracy = 0
    best_model = None
    best_scaler = None
    
    # Store test data from the last fold for feature importance
    final_X_test = None
    final_y_test = None
    
    # Get callbacks for training
    callbacks = get_callbacks(patience=PATIENCE)
    
    # Calculate class weights to handle imbalance
    # Nothing(0) is majority, Long(1) and Short(2) are minority - weight them higher
    from sklearn.utils.class_weight import compute_class_weight
    unique_classes = np.unique(y)
    class_weights_array = compute_class_weight('balanced', classes=unique_classes, y=y)
    class_weights = {int(c): float(w) for c, w in zip(unique_classes, class_weights_array)}
    print(f"    Class weights: {class_weights}")
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        print(f"\n  Fold {fold}/{N_FOLDS}:")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Normalize data - fit scaler on training data only (avoid data leakage)
        scaler = StandardScaler()
        # Reshape to 2D for scaler, then back to 3D
        n_samples_train, n_timesteps, n_features = X_train.shape
        n_samples_test = X_test.shape[0]
        
        X_train_2d = X_train.reshape(-1, n_features)
        X_test_2d = X_test.reshape(-1, n_features)
        
        X_train_scaled = scaler.fit_transform(X_train_2d).reshape(n_samples_train, n_timesteps, n_features)
        X_test_scaled = scaler.transform(X_test_2d).reshape(n_samples_test, n_timesteps, n_features)
        
        print(f"    Train: {len(X_train)} samples | Test: {len(X_test)} samples")
        print(f"    Features: {n_features} | Window: {n_timesteps} | Epochs: up to {EPOCHS}")
        
        # Create fresh model for each fold with enhanced architecture
        fold_model = create_lstm((window_size, actual_num_features), learning_rate=LEARNING_RATE)
        
        # Train with callbacks (early stopping + LR reduction) and class weighting
        history = fold_model.fit(
            X_train_scaled, y_train, 
            epochs=EPOCHS, 
            batch_size=BATCH_SIZE,
            validation_data=(X_test_scaled, y_test), 
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=0
        )
        
        # Get actual epochs trained (may be less due to early stopping)
        actual_epochs = len(history.history['loss'])
        
        # Evaluate on normalized data
        loss, accuracy = fold_model.evaluate(X_test_scaled, y_test, verbose=0)
        fold_results.append({
            'fold': fold,
            'accuracy': float(accuracy),
            'loss': float(loss),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'epochs_trained': actual_epochs
        })
        
        print(f"    Epochs: {actual_epochs}/{EPOCHS} | Accuracy: {accuracy:.4f} | Loss: {loss:.4f}")
        
        # Keep the best model and its scaler
        if accuracy > best_fold_accuracy:
            best_fold_accuracy = accuracy
            best_model = fold_model
            best_scaler = scaler
            final_X_test = X_test_scaled
            final_y_test = y_test
    
    # Calculate average metrics
    avg_accuracy = np.mean([r['accuracy'] for r in fold_results])
    avg_loss = np.mean([r['loss'] for r in fold_results])
    std_accuracy = np.std([r['accuracy'] for r in fold_results])
    
    print(f"\n  --- K-Fold Summary ---")
    print(f"  Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  Best Fold: {fold_results[[r['accuracy'] for r in fold_results].index(best_fold_accuracy)]['fold']} ({best_fold_accuracy:.4f})")
    
    # Save the best model as global model
    global_model = best_model
    global_model_path = models_dir / "global_model.keras"
    global_model.save(str(global_model_path))
    print(f"\n✓ Global Model (best fold) Saved as {global_model_path}")
    
    # Save the scaler for inference
    scaler_path = models_dir / "global_scaler.joblib"
    joblib.dump(best_scaler, str(scaler_path))
    print(f"✓ Global Scaler saved as {scaler_path}")
    
    # Save K-Fold results with feature metadata
    kfold_results = {
        'n_folds': N_FOLDS,
        'average_accuracy': float(avg_accuracy),
        'std_accuracy': float(std_accuracy),
        'average_loss': float(avg_loss),
        'best_fold_accuracy': float(best_fold_accuracy),
        'normalization': 'StandardScaler',
        'window_size': window_size,
        'feature_columns': common_features,  # Essential for inference
        'num_features': num_features,
        'folds': fold_results
    }
    kfold_path = models_dir / "global_kfold_results.json"
    with open(kfold_path, 'w') as f:
        json.dump(kfold_results, f, indent=2)
    print(f"✓ K-Fold results saved to {kfold_path}")
    
    # Use final test set for feature importance
    X_test = final_X_test
    y_test = final_y_test

    # ==========================================
    # 7. Train Per-Pair Models with K-Fold CV
    # ==========================================
    print("\n--- TRAINING PER-PAIR LSTM MODELS (3-Fold Time Series CV) ---")
    print("    (Enhanced model with early stopping)")
    pair_results = {}
    
    PAIR_FOLDS = 3        # Fewer folds for individual pairs (less data)
    PAIR_EPOCHS = 30      # Fewer epochs for per-pair (less data)
    pair_callbacks = get_callbacks(patience=5)  # Shorter patience for smaller datasets
    
    for symbol_full, data in symbol_data.items():
        # symbol_full is already in correct format (e.g., "BTC_USDC_USDC")
        print(f"\nTraining model for {symbol_full}...")
        
        X_sym = data['X'].reshape((data['X'].shape[0], window_size, actual_num_features))
        y_sym = data['y']
        
        if len(X_sym) < 100:
            print(f"   Skipping {symbol_full}: insufficient samples ({len(X_sym)})")
            continue
        
        # K-Fold for this pair
        pair_tscv = TimeSeriesSplit(n_splits=PAIR_FOLDS)
        pair_fold_results = []
        best_pair_acc = 0
        best_pair_model = None
        best_pair_scaler = None
        
        for fold, (train_idx, test_idx) in enumerate(pair_tscv.split(X_sym), 1):
            X_train_sym, X_test_sym = X_sym[train_idx], X_sym[test_idx]
            y_train_sym, y_test_sym = y_sym[train_idx], y_sym[test_idx]
            
            if len(X_train_sym) < 50 or len(X_test_sym) < 10:
                continue
            
            # Normalize data for this fold
            pair_scaler = StandardScaler()
            n_samples_train, n_timesteps, n_features = X_train_sym.shape
            n_samples_test = X_test_sym.shape[0]
            
            X_train_2d = X_train_sym.reshape(-1, n_features)
            X_test_2d = X_test_sym.reshape(-1, n_features)
            
            X_train_scaled = pair_scaler.fit_transform(X_train_2d).reshape(n_samples_train, n_timesteps, n_features)
            X_test_scaled = pair_scaler.transform(X_test_2d).reshape(n_samples_test, n_timesteps, n_features)
            
            # Create and train model for this fold (with enhanced architecture)
            pair_model = create_lstm((window_size, actual_num_features), learning_rate=LEARNING_RATE)
            pair_model.fit(
                X_train_scaled, y_train_sym, 
                epochs=PAIR_EPOCHS, 
                batch_size=BATCH_SIZE,
                validation_data=(X_test_scaled, y_test_sym), 
                callbacks=pair_callbacks,
                verbose=0
            )
            
            # Evaluate
            loss, accuracy = pair_model.evaluate(X_test_scaled, y_test_sym, verbose=0)
            pair_fold_results.append({'fold': fold, 'accuracy': accuracy, 'loss': loss})
            
            if accuracy > best_pair_acc:
                best_pair_acc = accuracy
                best_pair_model = pair_model
                best_pair_scaler = pair_scaler
        
        if not pair_fold_results:
            print(f"   Skipping {symbol_full}: no valid folds")
            continue
        
        # Calculate average metrics for this pair
        pair_avg_acc = np.mean([r['accuracy'] for r in pair_fold_results])
        pair_avg_loss = np.mean([r['loss'] for r in pair_fold_results])
        pair_std_acc = np.std([r['accuracy'] for r in pair_fold_results])
        
        pair_results[symbol_full] = {
            'avg_accuracy': float(pair_avg_acc),
            'std_accuracy': float(pair_std_acc),
            'avg_loss': float(pair_avg_loss),
            'best_accuracy': float(best_pair_acc),
            'n_folds': len(pair_fold_results),
            'samples': len(X_sym)
        }
        
        # Save best model and scaler for this pair
        pair_model_path = models_dir / f"{symbol_full}_model.keras"
        best_pair_model.save(str(pair_model_path))
        
        pair_scaler_path = models_dir / f"{symbol_full}_scaler.joblib"
        joblib.dump(best_pair_scaler, str(pair_scaler_path))
        
        print(f"   ✓ {symbol_full}: avg_acc={pair_avg_acc:.4f}±{pair_std_acc:.4f}, best={best_pair_acc:.4f}")
    
    # Save pair results summary
    pair_results_path = models_dir / "pair_training_results.json"
    with open(pair_results_path, 'w') as f:
        json.dump(pair_results, f, indent=2)
    print(f"\n✓ Per-pair training results saved to {pair_results_path}")

    # ==========================================
    # 8. Feature Importance Analysis
    # ==========================================
    print("\n--- FEATURE IMPORTANCE ANALYSIS ---")
    
    # Use permutation importance on the global model
    # Baseline accuracy
    baseline_loss, baseline_acc = global_model.evaluate(X_test, y_test, verbose=0)
    print(f"Baseline accuracy: {baseline_acc:.4f}")
    
    feature_importance = {}
    
    for feat_idx, feat_name in enumerate(common_features):
        # Create a copy and shuffle this feature across all timesteps
        X_permuted = X_test.copy()
        
        # Permute the feature across all samples (shuffle within the feature column)
        for t in range(window_size):
            np.random.shuffle(X_permuted[:, t, feat_idx])
        
        # Evaluate with permuted feature
        perm_loss, perm_acc = global_model.evaluate(X_permuted, y_test, verbose=0)
        
        # Importance = drop in accuracy when feature is permuted
        importance = baseline_acc - perm_acc
        feature_importance[feat_name] = {
            'importance': float(importance),
            'baseline_accuracy': float(baseline_acc),
            'permuted_accuracy': float(perm_acc),
            'accuracy_drop': float(importance),
            'relative_importance_pct': float((importance / baseline_acc) * 100) if baseline_acc > 0 else 0
        }
        
        print(f"   {feat_name}: importance={importance:.4f} (acc drop: {baseline_acc:.4f} -> {perm_acc:.4f})")
    
    # Sort by importance
    sorted_importance = dict(sorted(feature_importance.items(), 
                                     key=lambda x: x[1]['importance'], 
                                     reverse=True))
    
    # Add metadata
    importance_report = {
        'model': 'global_model',
        'baseline_accuracy': float(baseline_acc),
        'baseline_loss': float(baseline_loss),
        'test_samples': len(X_test),
        'features': sorted_importance,
        'feature_ranking': list(sorted_importance.keys())
    }
    
    # Save feature importance
    importance_path = models_dir / "feature_importance.json"
    with open(importance_path, 'w') as f:
        json.dump(importance_report, f, indent=2)
    
    print(f"\n--- FEATURE RANKING (by importance) ---")
    for i, (feat, data) in enumerate(sorted_importance.items(), 1):
        print(f"   {i}. {feat}: {data['importance']:.4f} ({data['relative_importance_pct']:.2f}%)")
    
    print(f"\n✓ Feature importance saved to {importance_path}")
    print("\n=== TRAINING COMPLETE ===")