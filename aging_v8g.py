"""
V8g: Aging curves conditioned on Lead Time and Earliness
=========================================================
Instead of: OO_ratio(site, TF, lag) = flat average
Now:         OO_ratio(site, TF, lag) = α + β₁·LT + β₂·OE

Same for FC coverage and FC bias.
Falls back to flat curve when insufficient data or poor fit.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from numpy.linalg import lstsq

RECENCY_POWER = 2
MIN_OBS_FLAT = 3
MIN_OBS_COND = 8      # Need more obs for regression
MIN_R2_IMPROVE = 0.05  # Conditioned must improve R² by at least this
MIN_LT_FOR_NORM = 0.5  # Minimum LT (months) for normalization

df = pd.read_csv('Dummy Training Data Customer 1 Try 2 11-Feb-2026.csv',
                 parse_dates=['Reference_Month', 'Target_Period_Start', 'Target_Period_End'])
if 'Unnamed: 0' in df.columns: df = df.drop(columns=['Unnamed: 0'])
if 'GSA' not in df.columns: df['GSA'] = 'Customer 1'
df['gsa_site'] = df['GSA'].astype(str) + '|' + df['Site'].astype(str)
df['Avg_Weighted_Order_Earliness'] = df['Avg_Weighted_Order_Earliness'].fillna(0)
df['Avg_Weighted_Lead_Time'] = df['Avg_Weighted_Lead_Time'].fillna(0)
df = df[~df['Site'].isin(['Site A'])].copy()

df['oo_ratio'] = np.where(df['Actual_Sales'] > 0, df['Open_Orders'] / df['Actual_Sales'], np.nan)
df['oo_ratio_norm'] = np.where(
    (df['Actual_Sales'] > 0) & (df['Avg_Weighted_Lead_Time'] > MIN_LT_FOR_NORM),
    df['Open_Orders'] / (df['Actual_Sales'] * df['Avg_Weighted_Lead_Time']),
    np.nan)
df['fc_coverage'] = np.where(df['Actual_Sales'] > 0, df['Covered_Orders'] / df['Actual_Sales'], np.nan)
df['fc_bias'] = np.where(df['Covered_Orders'] > 0, df['Forecast_Value'] / df['Covered_Orders'], np.nan)

clean_sites = [gs for gs in sorted(df['gsa_site'].unique()) if 'Site E' not in gs]
top_sites = ['Customer 1|Site W', 'Customer 1|Site J', 'Customer 1|Site S', 'Customer 1|Site T']

cutoff = df['Reference_Month'].quantile(0.8)
train = df[df['Reference_Month'] <= cutoff].copy()
test = df[df['Reference_Month'] > cutoff].copy()
test_months = sorted(test['Reference_Month'].unique())
gsa_sites = sorted(df['gsa_site'].unique())

# ============================================================
# BUILD CONDITIONED CURVES
# ============================================================
def build_conditioned_curves(train_df, ratio_col, start=None):
    """Build both flat and LT+OE conditioned curves.
    Returns: flat_curves, cond_curves
    flat: {(gs, tf, lag): value}
    cond: {(gs, tf, lag): {'alpha': a, 'beta_lt': b1, 'beta_oe': b2, 'r2_gain': r2diff}}
    """
    td = train_df if start is None else train_df[train_df['Reference_Month'] >= start]
    flat = {}
    cond = {}
    
    for (gs, tf, lag), grp in td.groupby(['gsa_site', 'Timeframe', 'Prediction_Lag']):
        valid = grp[grp[ratio_col].notna() & (grp[ratio_col] > 0) & (grp[ratio_col] < 10)].copy()
        
        if len(valid) < MIN_OBS_FLAT:
            continue
        
        # Flat curve (recency-weighted)
        days = (valid['Reference_Month'] - valid['Reference_Month'].min()).dt.days / 365
        wts = (1 + days.values) ** RECENCY_POWER
        flat_val = np.average(valid[ratio_col], weights=wts)
        flat[(gs, tf, lag)] = flat_val
        
        # Conditioned curve: weighted regression
        if len(valid) >= MIN_OBS_COND:
            y = valid[ratio_col].values
            lt = valid['Avg_Weighted_Lead_Time'].values
            oe = valid['Avg_Weighted_Order_Earliness'].values
            
            # Check variance in predictors
            if np.std(lt) < 0.1 and np.std(oe) < 0.1:
                continue
            
            # Weighted least squares: ratio = alpha + beta_lt * LT + beta_oe * OE
            X = np.column_stack([np.ones(len(valid)), lt, oe])
            W = np.diag(np.sqrt(wts))
            Xw = W @ X
            yw = W @ y
            
            try:
                coef, _, _, _ = lstsq(Xw, yw, rcond=None)
                pred_cond = X @ coef
                pred_flat = np.full(len(y), flat_val)
                
                ss_tot = np.sum(wts * (y - np.average(y, weights=wts))**2)
                ss_res_flat = np.sum(wts * (y - pred_flat)**2)
                ss_res_cond = np.sum(wts * (y - pred_cond)**2)
                
                r2_flat = 1 - ss_res_flat / ss_tot if ss_tot > 0 else 0
                r2_cond = 1 - ss_res_cond / ss_tot if ss_tot > 0 else 0
                
                if r2_cond - r2_flat >= MIN_R2_IMPROVE:
                    cond[(gs, tf, lag)] = {
                        'alpha': coef[0],
                        'beta_lt': coef[1],
                        'beta_oe': coef[2],
                        'r2_gain': r2_cond - r2_flat,
                        'r2_cond': r2_cond,
                        'flat_val': flat_val,
                        'lt_mean': np.mean(lt),
                        'oe_mean': np.mean(oe),
                        'y_min': np.min(y),
                        'y_max': np.max(y),
                    }
            except:
                pass
    
    # Fallback curves (pooled across TFs)
    for (gs, lag), grp in td.groupby(['gsa_site', 'Prediction_Lag']):
        valid = grp[grp[ratio_col].notna() & (grp[ratio_col] > 0) & (grp[ratio_col] < 10)]
        if len(valid) >= MIN_OBS_FLAT:
            days = (valid['Reference_Month'] - valid['Reference_Month'].min()).dt.days / 365
            wts = (1 + days.values) ** RECENCY_POWER
            flat[('FB', gs, lag)] = np.average(valid[ratio_col], weights=wts)
            
            if len(valid) >= MIN_OBS_COND:
                y = valid[ratio_col].values
                lt = valid['Avg_Weighted_Lead_Time'].values
                oe = valid['Avg_Weighted_Order_Earliness'].values
                if np.std(lt) > 0.1 or np.std(oe) > 0.1:
                    X = np.column_stack([np.ones(len(valid)), lt, oe])
                    W = np.diag(np.sqrt(wts))
                    try:
                        coef, _, _, _ = lstsq(W @ X, W @ y, rcond=None)
                        flat_val = flat[('FB', gs, lag)]
                        pred_cond = X @ coef
                        ss_tot = np.sum(wts * (y - np.average(y, weights=wts))**2)
                        ss_res_flat = np.sum(wts * (y - flat_val)**2)
                        ss_res_cond = np.sum(wts * (y - pred_cond)**2)
                        r2_flat = 1 - ss_res_flat / ss_tot if ss_tot > 0 else 0
                        r2_cond = 1 - ss_res_cond / ss_tot if ss_tot > 0 else 0
                        if r2_cond - r2_flat >= MIN_R2_IMPROVE:
                            cond[('FB', gs, lag)] = {
                                'alpha': coef[0], 'beta_lt': coef[1], 'beta_oe': coef[2],
                                'r2_gain': r2_cond - r2_flat, 'r2_cond': r2_cond,
                                'flat_val': flat_val, 'lt_mean': np.mean(lt), 'oe_mean': np.mean(oe),
                                'y_min': np.min(y), 'y_max': np.max(y),
                            }
                    except:
                        pass
    
    return flat, cond

def get_curve_value(flat, cond, gs, tf, lag, lt, oe):
    """Get curve value: use conditioned if available, else flat, else fallback."""
    for key in [(gs, tf, lag), ('FB', gs, lag)]:
        if key in cond:
            m = cond[key]
            val = m['alpha'] + m['beta_lt'] * lt + m['beta_oe'] * oe
            # Clip to reasonable range based on training data
            val = np.clip(val, max(m['y_min'] * 0.5, 0.001), m['y_max'] * 1.5)
            return val, 'cond'
        if key in flat:
            return flat[key], 'flat'
    return None, None

print("=" * 100)
print("V8g: BUILDING CONDITIONED AGING CURVES")
print("=" * 100)

# Build curves
oo_flat, oo_cond = build_conditioned_curves(train, 'oo_ratio', start='2023-01-01')
cov_flat, cov_cond = build_conditioned_curves(train, 'fc_coverage')
bias_flat, bias_cond = build_conditioned_curves(train, 'fc_bias')

# Build LT-normalized OO curves (V8i)
oo_norm_flat, _ = build_conditioned_curves(train, 'oo_ratio_norm', start='2023-01-01')

print(f"\n  OO curves:  {len(oo_flat)} flat, {len(oo_cond)} conditioned")
print(f"  OO LT-norm: {len(oo_norm_flat)} flat curves for oo_ratio/LT")
print(f"  FC cov:     {len(cov_flat)} flat, {len(cov_cond)} conditioned")
print(f"  FC bias:    {len(bias_flat)} flat, {len(bias_cond)} conditioned")

# Build reference LT+OE per cell from training (V8j)
ref_lt_oe = {}
for (gs, tf, lag), grp in train.groupby(['gsa_site', 'Timeframe', 'Prediction_Lag']):
    valid = grp[(grp['Avg_Weighted_Lead_Time'] > 0) | (grp['Avg_Weighted_Order_Earliness'] > 0)]
    if len(valid) >= MIN_OBS_FLAT:
        days = (valid['Reference_Month'] - valid['Reference_Month'].min()).dt.days / 365
        wts = (1 + days.values) ** RECENCY_POWER
        ref_lt_oe[(gs, tf, lag)] = np.average(
            valid['Avg_Weighted_Lead_Time'] + valid['Avg_Weighted_Order_Earliness'], weights=wts)
# Pooled fallback (across timeframes)
for (gs, lag), grp in train.groupby(['gsa_site', 'Prediction_Lag']):
    valid = grp[(grp['Avg_Weighted_Lead_Time'] > 0) | (grp['Avg_Weighted_Order_Earliness'] > 0)]
    if len(valid) >= MIN_OBS_FLAT:
        days = (valid['Reference_Month'] - valid['Reference_Month'].min()).dt.days / 365
        wts = (1 + days.values) ** RECENCY_POWER
        ref_lt_oe[('FB', gs, lag)] = np.average(
            valid['Avg_Weighted_Lead_Time'] + valid['Avg_Weighted_Order_Earliness'], weights=wts)
print(f"  LT+OE ref: {len([k for k in ref_lt_oe if k[0] != 'FB'])} cell + {len([k for k in ref_lt_oe if k[0] == 'FB'])} fallback")

# Show conditioned curve details for top sites
print(f"\n  Conditioned OO curves (top sites):")
print(f"  {'Key':>30s} | {'α':>6s} | {'β_LT':>7s} | {'β_OE':>7s} | {'R² gain':>8s} | {'R² cond':>8s}")
print("  " + "-" * 80)
for key in sorted(oo_cond.keys()):
    gs = key[0] if isinstance(key[0], str) and '|' in key[0] else key[1] if len(key) > 1 and isinstance(key[1], str) and '|' in key[1] else None
    if gs and gs in top_sites:
        m = oo_cond[key]
        kstr = f"{key}"[:30]
        print(f"  {kstr:>30s} | {m['alpha']:>+5.3f} | {m['beta_lt']:>+6.4f} | {m['beta_oe']:>+6.4f} | {m['r2_gain']:>+6.3f} | {m['r2_cond']:>6.3f}")

# ============================================================
# APPLY CURVES AND COMPUTE IMPLIED ACTUALS
# ============================================================
def apply_curves(dset, oo_f, oo_c, cov_f, cov_c, bias_f, bias_c):
    """Apply conditioned curves to compute implied actuals."""
    oo_impl = np.full(len(dset), np.nan)
    fc_impl = np.full(len(dset), np.nan)
    oo_types = {'cond': 0, 'flat': 0}
    fc_types = {'cond': 0, 'flat': 0}
    
    for i, (idx, row) in enumerate(dset.iterrows()):
        gs, tf, lag = row['gsa_site'], row['Timeframe'], row['Prediction_Lag']
        lt, oe = row['Avg_Weighted_Lead_Time'], row['Avg_Weighted_Order_Earliness']
        
        # OO implied
        f, ftype = get_curve_value(oo_f, oo_c, gs, tf, lag, lt, oe)
        if f and f > 0.001:
            oo_impl[i] = row['Open_Orders'] / f
            if ftype: oo_types[ftype] += 1
        
        # FC implied
        cf, cft = get_curve_value(cov_f, cov_c, gs, tf, lag, lt, oe)
        bf, bft = get_curve_value(bias_f, bias_c, gs, tf, lag, lt, oe)
        if cf and bf and cf > 0.001 and bf > 0.001 and row['Forecast_Value'] > 0:
            fc_impl[i] = row['Forecast_Value'] / bf / cf
            fc_types['cond' if cft == 'cond' or bft == 'cond' else 'flat'] += 1
    
    dset['oo_implied'] = np.maximum(oo_impl, 0)
    dset['fc_implied'] = np.maximum(fc_impl, 0)
    return oo_types, fc_types

# Also build flat-only (V8d baseline) for comparison
oo_flat_only, _ = build_conditioned_curves(train, 'oo_ratio', start='2023-01-01')
# Force no conditioned curves for baseline
def apply_flat_only(dset, oo_f, cov_f, bias_f):
    oo_impl = np.full(len(dset), np.nan)
    fc_impl = np.full(len(dset), np.nan)
    for i, (idx, row) in enumerate(dset.iterrows()):
        gs, tf, lag = row['gsa_site'], row['Timeframe'], row['Prediction_Lag']
        for key in [(gs, tf, lag), ('FB', gs, lag)]:
            if key in oo_f:
                oo_impl[i] = row['Open_Orders'] / oo_f[key] if oo_f[key] > 0.001 else np.nan
                break
        for key in [(gs, tf, lag), ('FB', gs, lag)]:
            cf = cov_f.get(key)
            bf = bias_f.get(key)
            if cf and bf and cf > 0.001 and bf > 0.001 and row['Forecast_Value'] > 0:
                fc_impl[i] = row['Forecast_Value'] / bf / cf
                break
    dset['oo_implied_flat'] = np.maximum(oo_impl, 0)
    dset['fc_implied_flat'] = np.maximum(fc_impl, 0)

def apply_lt_normalized(dset, oo_norm_f, oo_unnorm_f, cov_f, bias_f):
    """Apply LT-normalized OO curves + flat FC curves.
    OO implied = Open_Orders / (norm_curve * LT_current).
    Falls back to unnormalized flat if LT is missing/tiny.
    """
    oo_impl = np.full(len(dset), np.nan)
    fc_impl = np.full(len(dset), np.nan)
    oo_types = {'lt_norm': 0, 'flat_fallback': 0}

    for i, (idx, row) in enumerate(dset.iterrows()):
        gs, tf, lag = row['gsa_site'], row['Timeframe'], row['Prediction_Lag']
        lt = row['Avg_Weighted_Lead_Time']

        # OO: try LT-normalized curve
        used_norm = False
        if lt > MIN_LT_FOR_NORM:
            for key in [(gs, tf, lag), ('FB', gs, lag)]:
                if key in oo_norm_f and oo_norm_f[key] > 0.001:
                    oo_impl[i] = row['Open_Orders'] / (oo_norm_f[key] * lt)
                    oo_types['lt_norm'] += 1
                    used_norm = True
                    break

        # Fallback: unnormalized flat curve (same as V8d)
        if not used_norm:
            for key in [(gs, tf, lag), ('FB', gs, lag)]:
                if key in oo_unnorm_f and oo_unnorm_f[key] > 0.001:
                    oo_impl[i] = row['Open_Orders'] / oo_unnorm_f[key]
                    oo_types['flat_fallback'] += 1
                    break

        # FC: flat (same as V8d)
        for key in [(gs, tf, lag), ('FB', gs, lag)]:
            cf = cov_f.get(key)
            bf = bias_f.get(key)
            if cf and bf and cf > 0.001 and bf > 0.001 and row['Forecast_Value'] > 0:
                fc_impl[i] = row['Forecast_Value'] / bf / cf
                break

    dset['oo_implied_norm'] = np.maximum(oo_impl, 0)
    dset['fc_implied_norm'] = np.maximum(fc_impl, 0)
    return oo_types

def apply_effective_lag(dset, oo_f, cov_f, bias_f, ref_ltoe):
    """V8j: Additive effective lag shift.
    effective_lag = lag + (ref_LT_OE - current_LT_OE)
    Interpolates flat OO curve between integer lags. FC stays flat (no shift).
    """
    oo_impl = np.full(len(dset), np.nan)
    fc_impl = np.full(len(dset), np.nan)
    stats = {'shifted': 0, 'flat_fallback': 0, 'shifts': []}

    for i, (idx, row) in enumerate(dset.iterrows()):
        gs, tf, lag = row['gsa_site'], row['Timeframe'], row['Prediction_Lag']
        lt_oe_cur = row['Avg_Weighted_Lead_Time'] + row['Avg_Weighted_Order_Earliness']

        # Get reference LT+OE for this cell
        ref = None
        for key in [(gs, tf, lag), ('FB', gs, lag)]:
            if key in ref_ltoe:
                ref = ref_ltoe[key]
                break

        if ref is not None:
            delta = ref - lt_oe_cur
            eff_lag = np.clip(lag + delta, 1.0, 12.0)

            # Interpolate between integer lags in flat curve
            lag_lo = int(np.floor(eff_lag))
            lag_hi = int(np.ceil(eff_lag))
            frac = eff_lag - lag_lo

            val_lo, val_hi = None, None
            for k in [(gs, tf, lag_lo), ('FB', gs, lag_lo)]:
                if k in oo_f:
                    val_lo = oo_f[k]; break
            for k in [(gs, tf, lag_hi), ('FB', gs, lag_hi)]:
                if k in oo_f:
                    val_hi = oo_f[k]; break

            if lag_lo == lag_hi and val_lo is not None:
                oo_curve = val_lo
            elif val_lo is not None and val_hi is not None:
                oo_curve = val_lo * (1 - frac) + val_hi * frac
            elif val_lo is not None:
                oo_curve = val_lo
            elif val_hi is not None:
                oo_curve = val_hi
            else:
                oo_curve = None

            if oo_curve and oo_curve > 0.001:
                oo_impl[i] = row['Open_Orders'] / oo_curve
                stats['shifted'] += 1
                stats['shifts'].append(delta)
            else:
                for k in [(gs, tf, lag), ('FB', gs, lag)]:
                    if k in oo_f and oo_f[k] > 0.001:
                        oo_impl[i] = row['Open_Orders'] / oo_f[k]
                        stats['flat_fallback'] += 1; break
        else:
            for k in [(gs, tf, lag), ('FB', gs, lag)]:
                if k in oo_f and oo_f[k] > 0.001:
                    oo_impl[i] = row['Open_Orders'] / oo_f[k]
                    stats['flat_fallback'] += 1; break

        # FC: flat (no shift, same as V8d)
        for k in [(gs, tf, lag), ('FB', gs, lag)]:
            cf, bf = cov_f.get(k), bias_f.get(k)
            if cf and bf and cf > 0.001 and bf > 0.001 and row['Forecast_Value'] > 0:
                fc_impl[i] = row['Forecast_Value'] / bf / cf; break

    dset['oo_implied_elag'] = np.maximum(oo_impl, 0)
    dset['fc_implied_elag'] = np.maximum(fc_impl, 0)
    avg_s = np.mean(stats['shifts']) if stats['shifts'] else 0
    print(f"    Shifted: {stats['shifted']}, Flat fallback: {stats['flat_fallback']}, "
          f"Avg shift: {avg_s:+.2f} months")
    return stats

MIN_PROGRESS = 0.05   # Floor to avoid division by near-zero progress
MAX_CORRECTION = 3.0   # Cap multiplicative correction
MIN_CORRECTION = 0.2

def apply_progress_correction(dset, oo_f, cov_f, bias_f, ref_ltoe):
    """V8k: Progress-based multiplicative correction.
    correction = progress(lt_oe_current, lag) / progress(lt_oe_ref, lag)
    where progress = max(eps, 1 - lag / lt_oe).
    Corrected_ratio = flat_ratio * correction.
    """
    oo_impl = np.full(len(dset), np.nan)
    fc_impl = np.full(len(dset), np.nan)
    stats = {'corrected': 0, 'flat_fallback': 0, 'corrections': []}

    for i, (idx, row) in enumerate(dset.iterrows()):
        gs, tf, lag = row['gsa_site'], row['Timeframe'], row['Prediction_Lag']
        lt_oe_cur = row['Avg_Weighted_Lead_Time'] + row['Avg_Weighted_Order_Earliness']

        # Get reference LT+OE for this cell
        ref = None
        for key in [(gs, tf, lag), ('FB', gs, lag)]:
            if key in ref_ltoe:
                ref = ref_ltoe[key]
                break

        # Get flat OO ratio
        flat_val = None
        for k in [(gs, tf, lag), ('FB', gs, lag)]:
            if k in oo_f and oo_f[k] > 0.001:
                flat_val = oo_f[k]; break

        if flat_val is not None and ref is not None and ref > 0.5 and lt_oe_cur > 0.5:
            prog_ref = max(MIN_PROGRESS, 1.0 - lag / ref)
            prog_cur = max(MIN_PROGRESS, 1.0 - lag / lt_oe_cur)
            correction = np.clip(prog_cur / prog_ref, MIN_CORRECTION, MAX_CORRECTION)
            corrected_ratio = flat_val * correction
            oo_impl[i] = row['Open_Orders'] / corrected_ratio
            stats['corrected'] += 1
            stats['corrections'].append(correction)
        elif flat_val is not None:
            oo_impl[i] = row['Open_Orders'] / flat_val
            stats['flat_fallback'] += 1

        # FC: flat (no correction, same as V8d)
        for k in [(gs, tf, lag), ('FB', gs, lag)]:
            cf, bf = cov_f.get(k), bias_f.get(k)
            if cf and bf and cf > 0.001 and bf > 0.001 and row['Forecast_Value'] > 0:
                fc_impl[i] = row['Forecast_Value'] / bf / cf; break

    dset['oo_implied_prog'] = np.maximum(oo_impl, 0)
    dset['fc_implied_prog'] = np.maximum(fc_impl, 0)
    avg_c = np.mean(stats['corrections']) if stats['corrections'] else 1.0
    med_c = np.median(stats['corrections']) if stats['corrections'] else 1.0
    print(f"    Corrected: {stats['corrected']}, Flat fallback: {stats['flat_fallback']}, "
          f"Avg correction: {avg_c:.3f}, Median: {med_c:.3f}")
    return stats

# Apply conditioned curves
print("\n--- Applying curves ---")
oo_t, fc_t = apply_curves(train, oo_flat, oo_cond, cov_flat, cov_cond, bias_flat, bias_cond)
oo_te, fc_te = apply_curves(test, oo_flat, oo_cond, cov_flat, cov_cond, bias_flat, bias_cond)
print(f"  Test OO: {oo_te}")
print(f"  Test FC: {fc_te}")

# Apply flat curves for baseline
apply_flat_only(train, oo_flat, cov_flat, bias_flat)
apply_flat_only(test, oo_flat, cov_flat, bias_flat)

# Apply LT-normalized OO curves (V8i)
print("  Applying LT-normalized OO curves...")
norm_types_train = apply_lt_normalized(train, oo_norm_flat, oo_flat, cov_flat, bias_flat)
norm_types_test = apply_lt_normalized(test, oo_norm_flat, oo_flat, cov_flat, bias_flat)
print(f"  Test OO norm types: {norm_types_test}")

# Apply effective lag shift (V8j)
print("  Applying effective lag shift (V8j)...")
print("    Train:")
elag_train = apply_effective_lag(train, oo_flat, cov_flat, bias_flat, ref_lt_oe)
print("    Test:")
elag_test = apply_effective_lag(test, oo_flat, cov_flat, bias_flat, ref_lt_oe)

# Apply progress-based correction (V8k)
print("  Applying progress correction (V8k)...")
print("    Train:")
prog_train = apply_progress_correction(train, oo_flat, cov_flat, bias_flat, ref_lt_oe)
print("    Test:")
prog_test = apply_progress_correction(test, oo_flat, cov_flat, bias_flat, ref_lt_oe)

# ============================================================
# SIGNAL QUALITY: Conditioned vs Flat
# ============================================================
print(f"\n\n{'='*100}")
print("SIGNAL QUALITY: Conditioned vs Flat curves")
print("=" * 100)

print(f"\n  {'Signal':>25s} | {'WMAPE':>7s} | {'Bias':>7s} | {'R²':>6s}")
print("  " + "-" * 55)
for label, col in [('OO flat', 'oo_implied_flat'), ('OO conditioned', 'oo_implied'),
                    ('FC flat', 'fc_implied_flat'), ('FC conditioned', 'fc_implied')]:
    v = test[test[col].notna() & (test[col]>0) & (test['Actual_Sales']>0) & test['gsa_site'].isin(clean_sites)]
    y, p = v['Actual_Sales'].values, v[col].values
    w = np.sum(np.abs(y-p))/np.sum(y)*100
    b = np.mean((p-y)/y)*100
    r2 = 1 - np.sum((y-p)**2)/np.sum((y-np.mean(y))**2)
    print(f"  {label:>25s} | {w:>5.1f}% | {b:>+5.1f}% | {r2:>.3f}")

# Per site
print(f"\n  OO implied per-site:")
print(f"  {'Site':>15s} | {'Flat':>12s} | {'Conditioned':>12s} | {'Delta':>8s}")
print("  " + "-" * 55)
for gs in top_sites:
    site = gs.split('|')[1]
    for col, lbl in [('oo_implied_flat', 'flat'), ('oo_implied', 'cond')]:
        v = test[(test['gsa_site']==gs) & test[col].notna() & (test[col]>0) & (test['Actual_Sales']>0)]
        if len(v) > 0:
            w = np.sum(np.abs(v['Actual_Sales']-v[col]))/np.sum(v['Actual_Sales'])*100
            b = np.mean((v[col]-v['Actual_Sales'])/v['Actual_Sales'])*100
            if lbl == 'flat': fw, fb = w, b
            else: cw, cb = w, b
    print(f"  {site:>15s} | {fw:>5.1f}%{fb:>+5.0f}% | {cw:>5.1f}%{cb:>+5.0f}% | {cw-fw:>+5.1f}pp")

print(f"\n  FC implied per-site:")
print(f"  {'Site':>15s} | {'Flat':>12s} | {'Conditioned':>12s} | {'Delta':>8s}")
print("  " + "-" * 55)
for gs in top_sites:
    site = gs.split('|')[1]
    for col, lbl in [('fc_implied_flat', 'flat'), ('fc_implied', 'cond')]:
        v = test[(test['gsa_site']==gs) & test[col].notna() & (test[col]>0) & (test['Actual_Sales']>0)]
        if len(v) > 0:
            w = np.sum(np.abs(v['Actual_Sales']-v[col]))/np.sum(v['Actual_Sales'])*100
            b = np.mean((v[col]-v['Actual_Sales'])/v['Actual_Sales'])*100
            if lbl == 'flat': fw, fb = w, b
            else: cw, cb = w, b
    print(f"  {site:>15s} | {fw:>5.1f}%{fb:>+5.0f}% | {cw:>5.1f}%{cb:>+5.0f}% | {cw-fw:>+5.1f}pp")

# ============================================================
# SIGNAL QUALITY: LT-normalized OO (V8i)
# ============================================================
print(f"\n\n{'='*100}")
print("SIGNAL QUALITY: OO LT-normalized vs Flat vs Conditioned")
print("=" * 100)

print(f"\n  {'Signal':>25s} | {'WMAPE':>7s} | {'Bias':>7s} | {'R²':>6s}")
print("  " + "-" * 55)
for label, col in [('OO flat', 'oo_implied_flat'), ('OO conditioned', 'oo_implied'),
                    ('OO LT-normalized', 'oo_implied_norm')]:
    v = test[test[col].notna() & (test[col]>0) & (test['Actual_Sales']>0) & test['gsa_site'].isin(clean_sites)]
    y, p = v['Actual_Sales'].values, v[col].values
    w = np.sum(np.abs(y-p))/np.sum(y)*100
    b = np.mean((p-y)/y)*100
    r2 = 1 - np.sum((y-p)**2)/np.sum((y-np.mean(y))**2)
    print(f"  {label:>25s} | {w:>5.1f}% | {b:>+5.1f}% | {r2:>.3f}")

print(f"\n  OO implied per-site (flat vs LT-normalized):")
print(f"  {'Site':>15s} | {'Flat':>12s} | {'LT-norm':>12s} | {'Delta':>8s}")
print("  " + "-" * 55)
for gs in top_sites:
    site = gs.split('|')[1]
    fw, fb, nw, nb = 0, 0, 0, 0
    for col, lbl in [('oo_implied_flat', 'flat'), ('oo_implied_norm', 'norm')]:
        v = test[(test['gsa_site']==gs) & test[col].notna() & (test[col]>0) & (test['Actual_Sales']>0)]
        if len(v) > 0:
            w = np.sum(np.abs(v['Actual_Sales']-v[col]))/np.sum(v['Actual_Sales'])*100
            b = np.mean((v[col]-v['Actual_Sales'])/v['Actual_Sales'])*100
            if lbl == 'flat': fw, fb = w, b
            else: nw, nb = w, b
    print(f"  {site:>15s} | {fw:>5.1f}%{fb:>+5.0f}% | {nw:>5.1f}%{nb:>+5.0f}% | {nw-fw:>+5.1f}pp")

# ============================================================
# SIGNAL QUALITY: Effective Lag OO (V8j)
# ============================================================
print(f"\n\n{'='*100}")
print("SIGNAL QUALITY: OO Effective Lag vs Flat")
print("=" * 100)

print(f"\n  {'Signal':>25s} | {'WMAPE':>7s} | {'Bias':>7s} | {'R²':>6s}")
print("  " + "-" * 55)
for label, col in [('OO flat', 'oo_implied_flat'), ('OO eff-lag (V8j)', 'oo_implied_elag')]:
    v = test[test[col].notna() & (test[col]>0) & (test['Actual_Sales']>0) & test['gsa_site'].isin(clean_sites)]
    y, p = v['Actual_Sales'].values, v[col].values
    w = np.sum(np.abs(y-p))/np.sum(y)*100
    b = np.mean((p-y)/y)*100
    r2 = 1 - np.sum((y-p)**2)/np.sum((y-np.mean(y))**2)
    print(f"  {label:>25s} | {w:>5.1f}% | {b:>+5.1f}% | {r2:>.3f}")

print(f"\n  OO implied per-site (flat vs effective-lag):")
print(f"  {'Site':>15s} | {'Flat':>12s} | {'Eff-lag':>12s} | {'Delta':>8s}")
print("  " + "-" * 55)
for gs in top_sites:
    site = gs.split('|')[1]
    fw, fb, ew, eb = 0, 0, 0, 0
    for col, lbl in [('oo_implied_flat', 'flat'), ('oo_implied_elag', 'elag')]:
        v = test[(test['gsa_site']==gs) & test[col].notna() & (test[col]>0) & (test['Actual_Sales']>0)]
        if len(v) > 0:
            w = np.sum(np.abs(v['Actual_Sales']-v[col]))/np.sum(v['Actual_Sales'])*100
            b = np.mean((v[col]-v['Actual_Sales'])/v['Actual_Sales'])*100
            if lbl == 'flat': fw, fb = w, b
            else: ew, eb = w, b
    print(f"  {site:>15s} | {fw:>5.1f}%{fb:>+5.0f}% | {ew:>5.1f}%{eb:>+5.0f}% | {ew-fw:>+5.1f}pp")

# ============================================================
# SIGNAL QUALITY: Progress-corrected OO (V8k)
# ============================================================
print(f"\n\n{'='*100}")
print("SIGNAL QUALITY: OO Progress-corrected vs Flat")
print("=" * 100)

print(f"\n  {'Signal':>25s} | {'WMAPE':>7s} | {'Bias':>7s} | {'R²':>6s}")
print("  " + "-" * 55)
for label, col in [('OO flat', 'oo_implied_flat'), ('OO prog-corr (V8k)', 'oo_implied_prog')]:
    v = test[test[col].notna() & (test[col]>0) & (test['Actual_Sales']>0) & test['gsa_site'].isin(clean_sites)]
    y, p = v['Actual_Sales'].values, v[col].values
    w = np.sum(np.abs(y-p))/np.sum(y)*100
    b = np.mean((p-y)/y)*100
    r2 = 1 - np.sum((y-p)**2)/np.sum((y-np.mean(y))**2)
    print(f"  {label:>25s} | {w:>5.1f}% | {b:>+5.1f}% | {r2:>.3f}")

print(f"\n  OO implied per-site (flat vs progress-corrected):")
print(f"  {'Site':>15s} | {'Flat':>12s} | {'Prog-corr':>12s} | {'Delta':>8s}")
print("  " + "-" * 55)
for gs in top_sites:
    site = gs.split('|')[1]
    fw, fb, pw, pb = 0, 0, 0, 0
    for col, lbl in [('oo_implied_flat', 'flat'), ('oo_implied_prog', 'prog')]:
        v = test[(test['gsa_site']==gs) & test[col].notna() & (test[col]>0) & (test['Actual_Sales']>0)]
        if len(v) > 0:
            w = np.sum(np.abs(v['Actual_Sales']-v[col]))/np.sum(v['Actual_Sales'])*100
            b = np.mean((v[col]-v['Actual_Sales'])/v['Actual_Sales'])*100
            if lbl == 'flat': fw, fb = w, b
            else: pw, pb = w, b
    print(f"  {site:>15s} | {fw:>5.1f}%{fb:>+5.0f}% | {pw:>5.1f}%{pb:>+5.0f}% | {pw-fw:>+5.1f}pp")

# ============================================================
# WEIGHTED AVERAGE COMBINATION
# ============================================================
def weighted_avg(test_df, train_df, test_months, oo_col='oo_implied', fc_col='fc_implied'):
    results = test_df.copy()
    results['pred'] = np.nan
    hist = {}
    for gs in gsa_sites:
        for tf in train_df['Timeframe'].unique():
            sub = train_df[(train_df['gsa_site']==gs)&(train_df['Timeframe']==tf)]
            if len(sub)>0:
                d = (sub['Reference_Month']-sub['Reference_Month'].min()).dt.days/365
                w = (1+d.values)**RECENCY_POWER
                hist[(gs,tf)] = np.average(sub['Actual_Sales'], weights=w)
    
    for mi, month in enumerate(test_months):
        if mi==0:
            prior = train_df[train_df['Reference_Month'] >= train_df['Reference_Month'].quantile(0.8)]
        else:
            prior = results[results['Reference_Month'].isin(test_months[max(0,mi-3):mi])]
        cmask = results['Reference_Month']==month
        for gs in results.loc[cmask, 'gsa_site'].unique():
            for tf in results['Timeframe'].unique():
                ps = prior[(prior['gsa_site']==gs)&(prior['Timeframe']==tf)&(prior['Actual_Sales']>0)]
                oo_v = ps[ps[oo_col].notna()&(ps[oo_col]>0)]
                oo_mape = np.mean(np.abs(oo_v[oo_col]-oo_v['Actual_Sales'])/oo_v['Actual_Sales']) if len(oo_v)>=1 else 1.0
                fc_v = ps[ps[fc_col].notna()&(ps[fc_col]>0)]
                fc_mape = np.mean(np.abs(fc_v[fc_col]-fc_v['Actual_Sales'])/fc_v['Actual_Sales']) if len(fc_v)>=1 else 1.0
                
                rmask = cmask&(results['gsa_site']==gs)&(results['Timeframe']==tf)
                for idx in results[rmask].index:
                    row = results.loc[idx]
                    lag = row['Prediction_Lag']
                    has_oo = not np.isnan(row[oo_col]) and row[oo_col]>0
                    has_fc = not np.isnan(row[fc_col]) and row[fc_col]>0
                    oo_f = max(0.3, 1.0-0.06*(lag-1)) if has_oo else 0
                    fc_f = max(0.3, 1.0-0.06*(lag-1)) if has_fc else 0
                    w_oo = (1/max(oo_mape,0.01))*oo_f if has_oo else 0
                    w_fc = (1/max(fc_mape,0.01))*fc_f if has_fc else 0
                    hv = hist.get((gs,tf), 0)
                    w_h = (1/0.5)*0.1*lag/12 if hv>0 else 0
                    tot = w_oo+w_fc+w_h
                    if tot>0:
                        results.at[idx, 'pred'] = max(0,
                            (w_oo/tot)*(row[oo_col] if has_oo else 0)+
                            (w_fc/tot)*(row[fc_col] if has_fc else 0)+
                            (w_h/tot)*hv)
                    elif hv>0:
                        results.at[idx, 'pred'] = hv
    return results['pred']

print(f"\n\n{'='*100}")
print("FULL MODEL: V8d (flat) vs V8g (conditioned) vs V8i (LT-norm) vs V8j (eff-lag)")
print("=" * 100)

# V8d baseline
print("\n  Running V8d (flat curves)...")
test['v8d'] = weighted_avg(test, train, test_months, 'oo_implied_flat', 'fc_implied_flat')

# V8g conditioned
print("  Running V8g (conditioned curves)...")
test['v8g'] = weighted_avg(test, train, test_months, 'oo_implied', 'fc_implied')

# V8i LT-normalized
print("  Running V8i (LT-normalized OO + flat FC)...")
test['v8i'] = weighted_avg(test, train, test_months, 'oo_implied_norm', 'fc_implied_norm')

# V8j effective lag
print("  Running V8j (effective lag OO + flat FC)...")
test['v8j'] = weighted_avg(test, train, test_months, 'oo_implied_elag', 'fc_implied_elag')

# V8k progress-corrected
print("  Running V8k (progress-corrected OO + flat FC)...")
test['v8k'] = weighted_avg(test, train, test_months, 'oo_implied_prog', 'fc_implied_prog')

m2 = test['Reference_Month'] > test_months[0]

print(f"\n  {'Model':>15s} | {'WMAPE':>7s} | {'Bias':>7s} | {'R²':>6s} | {'Acct':>6s}")
print("  " + "-" * 55)
for label, col in [('V8d (flat)', 'v8d'), ('V8g (cond)', 'v8g'), ('V8i (LT-norm)', 'v8i'), ('V8j (eff-lag)', 'v8j'), ('V8k (prog)', 'v8k')]:
    ts = test[m2 & test[col].notna() & test['gsa_site'].isin(clean_sites)]
    y, p = ts['Actual_Sales'].values, ts[col].values
    w = np.sum(np.abs(y-p))/np.sum(y)*100
    b = np.mean((p-y)/y)*100
    r2 = 1 - np.sum((y-p)**2)/np.sum((y-np.mean(y))**2)
    tf12 = ts[(ts['Timeframe']==12)&(ts['Prediction_Lag']==1)]
    by_m = tf12.groupby('Reference_Month').agg({'Actual_Sales':'sum'})
    by_m['pr'] = tf12.groupby('Reference_Month').apply(lambda g: test.loc[g.index, col].sum()).values
    aw = np.sum(np.abs(by_m['Actual_Sales']-by_m['pr']))/np.sum(by_m['Actual_Sales'])*100
    print(f"  {label:>15s} | {w:>5.1f}% | {b:>+5.1f}% | {r2:>.3f} | {aw:>4.1f}%")

# Per-site
print(f"\n  Per-site (months 2+):")
print(f"  {'Site':>20s} | {'V8d':>12s} | {'V8j':>12s} | {'V8k':>12s} | {'d-k':>8s}")
print("  " + "-" * 75)
for gs in sorted(clean_sites):
    site = gs.split('|')[1]
    sub = test[m2 & (test['gsa_site']==gs) & (test['Actual_Sales']>0)]
    bv = sub[sub['v8d'].notna()]
    jv = sub[sub['v8j'].notna()]
    kv = sub[sub['v8k'].notna()]
    if len(bv)>0 and len(jv)>0 and len(kv)>0:
        bw = np.sum(np.abs(bv['Actual_Sales']-bv['v8d']))/np.sum(bv['Actual_Sales'])*100
        jw = np.sum(np.abs(jv['Actual_Sales']-jv['v8j']))/np.sum(jv['Actual_Sales'])*100
        kw = np.sum(np.abs(kv['Actual_Sales']-kv['v8k']))/np.sum(kv['Actual_Sales'])*100
        bb = np.mean((bv['v8d']-bv['Actual_Sales'])/bv['Actual_Sales'])*100
        jb = np.mean((jv['v8j']-jv['Actual_Sales'])/jv['Actual_Sales'])*100
        kb = np.mean((kv['v8k']-kv['Actual_Sales'])/kv['Actual_Sales'])*100
        print(f"  {site:>20s} | {bw:>5.1f}%{bb:>+5.0f}% | {jw:>5.1f}%{jb:>+5.0f}% | {kw:>5.1f}%{kb:>+5.0f}% | {kw-bw:>+5.1f}pp")

# Per-lag
print(f"\n  WMAPE by lag:")
print(f"  {'Lag':>4s} | {'V8d':>8s} | {'V8j':>8s} | {'V8k':>8s} | {'d-k':>8s}")
print("  " + "-" * 45)
for lag in sorted(test['Prediction_Lag'].unique()):
    sub = test[m2 & (test['Prediction_Lag']==lag) & (test['Actual_Sales']>0) & test['gsa_site'].isin(clean_sites)]
    bv = sub[sub['v8d'].notna()]
    jv = sub[sub['v8j'].notna()]
    kv = sub[sub['v8k'].notna()]
    if len(bv)>0:
        bw = np.sum(np.abs(bv['Actual_Sales']-bv['v8d']))/np.sum(bv['Actual_Sales'])*100
        jw = np.sum(np.abs(jv['Actual_Sales']-jv['v8j']))/np.sum(jv['Actual_Sales'])*100
        kw = np.sum(np.abs(kv['Actual_Sales']-kv['v8k']))/np.sum(kv['Actual_Sales'])*100
        print(f"  {lag:>4d} | {bw:>6.1f}% | {jw:>6.1f}% | {kw:>6.1f}% | {kw-bw:>+6.1f}pp")

# Account monthly
print(f"\n  Account TF=12/L=1:")
print(f"  {'Month':>12s} | {'Actual':>8s} | {'V8d':>15s} | {'V8k':>15s}")
print("  " + "-" * 65)
for ref in test_months[:7]:
    sub = test[(test['Reference_Month']==ref) & (test['Timeframe']==12) &
                (test['Prediction_Lag']==1) & test['gsa_site'].isin(clean_sites)]
    act = sub['Actual_Sales'].sum()
    bp = sub['v8d'].sum()
    kp = sub['v8k'].sum()
    if act > 0:
        print(f"  {ref.date()} | ${act/1e6:>5.0f}M | ${bp/1e6:>5.0f}M ({(bp-act)/act*100:>+5.1f}%) | ${kp/1e6:>5.0f}M ({(kp-act)/act*100:>+5.1f}%)")

print(f"\n\nV7 reference: WMAPE=8.8%  bias=+0.5%  Acct=2.2%")
print("\nDone!")
