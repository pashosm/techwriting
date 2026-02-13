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

def apply_progress_correction(dset, oo_f, cov_f, bias_f, ref_ltoe, power=1.0,
                              oo_out_col='oo_implied_prog', fc_out_col='fc_implied_prog'):
    """Progress-based multiplicative correction with power-CDF.
    correction = (prog_cur / prog_ref)^power
    power=1.0: uniform CDF (V8k). power<1: front-loaded orders (gentler).
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
            ratio = np.clip(prog_cur / prog_ref, MIN_CORRECTION, MAX_CORRECTION)
            correction = ratio ** power
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

    dset[oo_out_col] = np.maximum(oo_impl, 0)
    dset[fc_out_col] = np.maximum(fc_impl, 0)
    avg_c = np.mean(stats['corrections']) if stats['corrections'] else 1.0
    med_c = np.median(stats['corrections']) if stats['corrections'] else 1.0
    print(f"    p={power:.1f}: Corrected: {stats['corrected']}, Flat fallback: {stats['flat_fallback']}, "
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

# Apply progress-based correction with power sweep
P_VALUES = [0.2, 0.3, 0.5, 0.7, 1.0]
print("  Applying progress correction (power sweep)...")
for p in P_VALUES:
    oo_col = f'oo_implied_p{int(p*10)}'
    fc_col = f'fc_implied_p{int(p*10)}'
    apply_progress_correction(train, oo_flat, cov_flat, bias_flat, ref_lt_oe,
                              power=p, oo_out_col=oo_col, fc_out_col=fc_col)
    apply_progress_correction(test, oo_flat, cov_flat, bias_flat, ref_lt_oe,
                              power=p, oo_out_col=oo_col, fc_out_col=fc_col)

# Build FC debiasing curves: learn correction factor per (site, tf, lag) from training
# fc_debias[(gs,tf,lag)] = recency-weighted mean of (Actual_Sales / fc_implied_flat)
# At prediction: fc_implied_debiased = fc_implied_flat * fc_debias_factor
print("  Building FC debiasing curves...")
fc_debias = {}
for (gs, tf, lag), grp in train.groupby(['gsa_site', 'Timeframe', 'Prediction_Lag']):
    v = grp[grp['fc_implied_flat'].notna() & (grp['fc_implied_flat'] > 0) & (grp['Actual_Sales'] > 0)].copy()
    if len(v) >= MIN_OBS_FLAT:
        ratio = v['Actual_Sales'].values / v['fc_implied_flat'].values
        mask = (ratio > 0.1) & (ratio < 10)
        if mask.sum() >= MIN_OBS_FLAT:
            v_m = v.iloc[mask]
            days = (v_m['Reference_Month'] - v_m['Reference_Month'].min()).dt.days / 365
            wts = (1 + days.values) ** RECENCY_POWER
            fc_debias[(gs, tf, lag)] = np.average(ratio[mask], weights=wts)

# Fallback: pooled across TFs
for (gs, lag), grp in train.groupby(['gsa_site', 'Prediction_Lag']):
    v = grp[grp['fc_implied_flat'].notna() & (grp['fc_implied_flat'] > 0) & (grp['Actual_Sales'] > 0)].copy()
    if len(v) >= MIN_OBS_FLAT:
        ratio = v['Actual_Sales'].values / v['fc_implied_flat'].values
        mask = (ratio > 0.1) & (ratio < 10)
        if mask.sum() >= MIN_OBS_FLAT:
            v_m = v.iloc[mask]
            days = (v_m['Reference_Month'] - v_m['Reference_Month'].min()).dt.days / 365
            wts = (1 + days.values) ** RECENCY_POWER
            fc_debias[('FB', gs, lag)] = np.average(ratio[mask], weights=wts)

print(f"  FC debias curves: {len(fc_debias)} ({sum(1 for k in fc_debias if k[0]!='FB')} cell + {sum(1 for k in fc_debias if k[0]=='FB')} fallback)")

# Apply debiased FC to train and test
for ds in [train, test]:
    fc_db = np.full(len(ds), np.nan)
    for i, (idx, row) in enumerate(ds.iterrows()):
        gs, tf, lag = row['gsa_site'], row['Timeframe'], row['Prediction_Lag']
        fc_val = row['fc_implied_flat']
        if np.isnan(fc_val) or fc_val <= 0:
            continue
        for key in [(gs, tf, lag), ('FB', gs, lag)]:
            if key in fc_debias:
                fc_db[i] = fc_val * fc_debias[key]
                break
    ds['fc_implied_debiased'] = np.maximum(fc_db, 0)

# Also create a NaN OO column for FC-only models
for ds in [train, test]:
    ds['oo_none'] = np.nan

# ============================================================
# SIGNAL QUALITY: Conditioned vs Flat
# ============================================================
print(f"\n\n{'='*100}")
print("SIGNAL QUALITY: Conditioned vs Flat curves")
print("=" * 100)

print(f"\n  {'Signal':>25s} | {'WMAPE':>7s} | {'Bias':>7s} | {'R²':>6s}")
print("  " + "-" * 55)
for label, col in [('OO flat', 'oo_implied_flat'), ('OO conditioned', 'oo_implied'),
                    ('FC flat', 'fc_implied_flat'), ('FC conditioned', 'fc_implied'),
                    ('FC debiased', 'fc_implied_debiased')]:
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
# SIGNAL QUALITY: Power-CDF sweep
# ============================================================
print(f"\n\n{'='*100}")
print("SIGNAL QUALITY: OO Power-CDF correction sweep (p=0 is flat, p=1 is uniform)")
print("=" * 100)

print(f"\n  {'Signal':>25s} | {'WMAPE':>7s} | {'Bias':>7s} | {'R²':>6s}")
print("  " + "-" * 55)
label, col = 'OO flat (p=0)', 'oo_implied_flat'
v = test[test[col].notna() & (test[col]>0) & (test['Actual_Sales']>0) & test['gsa_site'].isin(clean_sites)]
y_f, p_f = v['Actual_Sales'].values, v[col].values
ww = np.sum(np.abs(y_f-p_f))/np.sum(y_f)*100
bb = np.mean((p_f-y_f)/y_f)*100
r2v = 1 - np.sum((y_f-p_f)**2)/np.sum((y_f-np.mean(y_f))**2)
print(f"  {label:>25s} | {ww:>5.1f}% | {bb:>+5.1f}% | {r2v:>.3f}")
for pv in P_VALUES:
    oo_col = f'oo_implied_p{int(pv*10)}'
    label = f'OO p={pv:.1f}'
    v = test[test[oo_col].notna() & (test[oo_col]>0) & (test['Actual_Sales']>0) & test['gsa_site'].isin(clean_sites)]
    y_v, p_v = v['Actual_Sales'].values, v[oo_col].values
    ww = np.sum(np.abs(y_v-p_v))/np.sum(y_v)*100
    bb = np.mean((p_v-y_v)/y_v)*100
    r2v = 1 - np.sum((y_v-p_v)**2)/np.sum((y_v-np.mean(y_v))**2)
    print(f"  {label:>25s} | {ww:>5.1f}% | {bb:>+5.1f}% | {r2v:>.3f}")

print(f"\n  OO implied per-site by power (WMAPE / bias):")
print(f"  {'Site':>10s} | {'flat':>10s}", end="")
for pv in P_VALUES:
    print(f" | {'p='+str(pv):>10s}", end="")
print()
print("  " + "-" * (14 + 13 * (1 + len(P_VALUES))))
for gs in top_sites:
    site = gs.split('|')[1]
    v = test[(test['gsa_site']==gs) & test['oo_implied_flat'].notna() & (test['oo_implied_flat']>0) & (test['Actual_Sales']>0)]
    if len(v) == 0: continue
    fw = np.sum(np.abs(v['Actual_Sales']-v['oo_implied_flat']))/np.sum(v['Actual_Sales'])*100
    fb = np.mean((v['oo_implied_flat']-v['Actual_Sales'])/v['Actual_Sales'])*100
    print(f"  {site:>10s} | {fw:>4.0f}%{fb:>+4.0f}%", end="")
    for pv in P_VALUES:
        oo_col = f'oo_implied_p{int(pv*10)}'
        vp = test[(test['gsa_site']==gs) & test[oo_col].notna() & (test[oo_col]>0) & (test['Actual_Sales']>0)]
        if len(vp) > 0:
            pw = np.sum(np.abs(vp['Actual_Sales']-vp[oo_col]))/np.sum(vp['Actual_Sales'])*100
            pb = np.mean((vp[oo_col]-vp['Actual_Sales'])/vp['Actual_Sales'])*100
            print(f" | {pw:>4.0f}%{pb:>+4.0f}%", end="")
        else:
            print(f" | {'n/a':>10s}", end="")
    print()

# ============================================================
# DIAGNOSTIC: Attach per-row curve values to test DataFrame
# ============================================================
print(f"\n\n{'='*100}")
print("DIAGNOSTICS: Understanding V8d error structure")
print("=" * 100)

# Attach flat curve values to each test row for drift analysis
for ds in [test, train]:
    oo_cv = np.full(len(ds), np.nan)
    cov_cv = np.full(len(ds), np.nan)
    bias_cv = np.full(len(ds), np.nan)
    for i, (idx, row) in enumerate(ds.iterrows()):
        gs, tf, lag = row['gsa_site'], row['Timeframe'], row['Prediction_Lag']
        for key in [(gs, tf, lag), ('FB', gs, lag)]:
            if key in oo_flat:
                oo_cv[i] = oo_flat[key]; break
        for key in [(gs, tf, lag), ('FB', gs, lag)]:
            if key in cov_flat:
                cov_cv[i] = cov_flat[key]; break
        for key in [(gs, tf, lag), ('FB', gs, lag)]:
            if key in bias_flat:
                bias_cv[i] = bias_flat[key]; break
    ds['oo_curve_val'] = oo_cv
    ds['cov_curve_val'] = cov_cv
    ds['bias_curve_val'] = bias_cv

# Filter: test set, clean sites, positive actuals
t = test[test['gsa_site'].isin(clean_sites) & (test['Actual_Sales'] > 0)].copy()

# ============================================================
# DIAGNOSTIC 1: Source of WMAPE
# ============================================================
print(f"\n--- DIAGNOSTIC 1: Source of WMAPE ---")

# 1a: Per-signal WMAPE by lag
print(f"\n  1a. Per-signal WMAPE by lag:")
print(f"  {'Lag':>4s} | {'OO WMAPE':>9s} {'N':>5s} | {'FC WMAPE':>9s} {'N':>5s} | {'Gap':>7s}")
print("  " + "-" * 50)
for lag in sorted(t['Prediction_Lag'].unique()):
    sub = t[t['Prediction_Lag'] == lag]
    oo_v = sub[sub['oo_implied_flat'].notna() & (sub['oo_implied_flat'] > 0)]
    fc_v = sub[sub['fc_implied_flat'].notna() & (sub['fc_implied_flat'] > 0)]
    oo_w = np.sum(np.abs(oo_v['Actual_Sales'] - oo_v['oo_implied_flat'])) / np.sum(oo_v['Actual_Sales']) * 100 if len(oo_v) > 0 else np.nan
    fc_w = np.sum(np.abs(fc_v['Actual_Sales'] - fc_v['fc_implied_flat'])) / np.sum(fc_v['Actual_Sales']) * 100 if len(fc_v) > 0 else np.nan
    gap = oo_w - fc_w if not (np.isnan(oo_w) or np.isnan(fc_w)) else np.nan
    gap_s = f"{gap:>+5.1f}pp" if not np.isnan(gap) else "   n/a"
    print(f"  {lag:>4d} | {oo_w:>7.1f}% {len(oo_v):>5d} | {fc_w:>7.1f}% {len(fc_v):>5d} | {gap_s}")

# 1b: Per-signal WMAPE by timeframe
print(f"\n  1b. Per-signal WMAPE by timeframe:")
print(f"  {'TF':>4s} | {'OO WMAPE':>9s} {'N':>5s} | {'FC WMAPE':>9s} {'N':>5s} | {'Gap':>7s}")
print("  " + "-" * 50)
for tf in sorted(t['Timeframe'].unique()):
    sub = t[t['Timeframe'] == tf]
    oo_v = sub[sub['oo_implied_flat'].notna() & (sub['oo_implied_flat'] > 0)]
    fc_v = sub[sub['fc_implied_flat'].notna() & (sub['fc_implied_flat'] > 0)]
    oo_w = np.sum(np.abs(oo_v['Actual_Sales'] - oo_v['oo_implied_flat'])) / np.sum(oo_v['Actual_Sales']) * 100 if len(oo_v) > 0 else np.nan
    fc_w = np.sum(np.abs(fc_v['Actual_Sales'] - fc_v['fc_implied_flat'])) / np.sum(fc_v['Actual_Sales']) * 100 if len(fc_v) > 0 else np.nan
    gap = oo_w - fc_w if not (np.isnan(oo_w) or np.isnan(fc_w)) else np.nan
    gap_s = f"{gap:>+5.1f}pp" if not np.isnan(gap) else "   n/a"
    print(f"  {tf:>4d} | {oo_w:>7.1f}% {len(oo_v):>5d} | {fc_w:>7.1f}% {len(fc_v):>5d} | {gap_s}")

# 1c: Per-site WMAPE with volume weighting
total_sales = t['Actual_Sales'].sum()
print(f"\n  1c. Per-site WMAPE with volume contribution:")
print(f"  {'Site':>15s} | {'Sales $M':>8s} | {'OO WMAPE':>9s} | {'FC WMAPE':>9s} | {'OO contrib':>10s} | {'FC contrib':>10s}")
print("  " + "-" * 72)
for gs in sorted(clean_sites):
    site = gs.split('|')[1]
    sub = t[t['gsa_site'] == gs]
    sales = sub['Actual_Sales'].sum()
    oo_v = sub[sub['oo_implied_flat'].notna() & (sub['oo_implied_flat'] > 0)]
    fc_v = sub[sub['fc_implied_flat'].notna() & (sub['fc_implied_flat'] > 0)]
    oo_ae = np.sum(np.abs(oo_v['Actual_Sales'] - oo_v['oo_implied_flat'])) if len(oo_v) > 0 else 0
    fc_ae = np.sum(np.abs(fc_v['Actual_Sales'] - fc_v['fc_implied_flat'])) if len(fc_v) > 0 else 0
    oo_w = oo_ae / np.sum(oo_v['Actual_Sales']) * 100 if len(oo_v) > 0 else np.nan
    fc_w = fc_ae / np.sum(fc_v['Actual_Sales']) * 100 if len(fc_v) > 0 else np.nan
    oo_c = oo_ae / total_sales * 100
    fc_c = fc_ae / total_sales * 100
    oo_ws = f"{oo_w:>7.1f}%" if not np.isnan(oo_w) else "     n/a"
    fc_ws = f"{fc_w:>7.1f}%" if not np.isnan(fc_w) else "     n/a"
    print(f"  {site:>15s} | {sales/1e6:>6.1f}M | {oo_ws:>9s} | {fc_ws:>9s} | {oo_c:>8.1f}pp | {fc_c:>8.1f}pp")

# 1d: Blend improvement check
both = t[t['oo_implied_flat'].notna() & (t['oo_implied_flat'] > 0) &
         t['fc_implied_flat'].notna() & (t['fc_implied_flat'] > 0)].copy()
both['simple_avg'] = (both['oo_implied_flat'] + both['fc_implied_flat']) / 2
print(f"\n  1d. Blend improvement (rows with both signals, N={len(both)}):")
print(f"  {'Signal':>15s} | {'WMAPE':>7s} | {'Bias':>7s}")
print("  " + "-" * 35)
for label, col in [('OO alone', 'oo_implied_flat'), ('FC alone', 'fc_implied_flat'), ('Simple avg', 'simple_avg')]:
    w = np.sum(np.abs(both['Actual_Sales'] - both[col])) / np.sum(both['Actual_Sales']) * 100
    b = np.sum(both[col] - both['Actual_Sales']) / np.sum(both['Actual_Sales']) * 100
    print(f"  {label:>15s} | {w:>5.1f}% | {b:>+5.1f}%")

# ============================================================
# DIAGNOSTIC 2: Source of Bias
# ============================================================
print(f"\n--- DIAGNOSTIC 2: Source of Bias ---")

# 2a: Per-signal bias by lag (volume-weighted)
print(f"\n  2a. Per-signal volume-weighted bias by lag:")
print(f"  {'Lag':>4s} | {'OO bias':>8s} | {'FC bias':>8s} | {'OO-FC':>7s}")
print("  " + "-" * 35)
for lag in sorted(t['Prediction_Lag'].unique()):
    sub = t[t['Prediction_Lag'] == lag]
    oo_v = sub[sub['oo_implied_flat'].notna() & (sub['oo_implied_flat'] > 0)]
    fc_v = sub[sub['fc_implied_flat'].notna() & (sub['fc_implied_flat'] > 0)]
    oo_b = np.sum(oo_v['oo_implied_flat'] - oo_v['Actual_Sales']) / np.sum(oo_v['Actual_Sales']) * 100 if len(oo_v) > 0 else np.nan
    fc_b = np.sum(fc_v['fc_implied_flat'] - fc_v['Actual_Sales']) / np.sum(fc_v['Actual_Sales']) * 100 if len(fc_v) > 0 else np.nan
    oo_s = f"{oo_b:>+6.1f}%" if not np.isnan(oo_b) else "    n/a"
    fc_s = f"{fc_b:>+6.1f}%" if not np.isnan(fc_b) else "    n/a"
    gap = oo_b - fc_b if not (np.isnan(oo_b) or np.isnan(fc_b)) else np.nan
    gap_s = f"{gap:>+5.1f}pp" if not np.isnan(gap) else "   n/a"
    print(f"  {lag:>4d} | {oo_s:>8s} | {fc_s:>8s} | {gap_s}")

# 2b: Per-signal bias by site
print(f"\n  2b. Per-signal volume-weighted bias by site:")
print(f"  {'Site':>15s} | {'Sales $M':>8s} | {'OO bias':>8s} | {'FC bias':>8s}")
print("  " + "-" * 48)
for gs in sorted(clean_sites):
    site = gs.split('|')[1]
    sub = t[t['gsa_site'] == gs]
    sales = sub['Actual_Sales'].sum()
    oo_v = sub[sub['oo_implied_flat'].notna() & (sub['oo_implied_flat'] > 0)]
    fc_v = sub[sub['fc_implied_flat'].notna() & (sub['fc_implied_flat'] > 0)]
    oo_b = np.sum(oo_v['oo_implied_flat'] - oo_v['Actual_Sales']) / np.sum(oo_v['Actual_Sales']) * 100 if len(oo_v) > 0 else np.nan
    fc_b = np.sum(fc_v['fc_implied_flat'] - fc_v['Actual_Sales']) / np.sum(fc_v['Actual_Sales']) * 100 if len(fc_v) > 0 else np.nan
    oo_s = f"{oo_b:>+6.1f}%" if not np.isnan(oo_b) else "    n/a"
    fc_s = f"{fc_b:>+6.1f}%" if not np.isnan(fc_b) else "    n/a"
    print(f"  {site:>15s} | {sales/1e6:>6.1f}M | {oo_s:>8s} | {fc_s:>8s}")

# 2c: Per-signal bias by timeframe
print(f"\n  2c. Per-signal volume-weighted bias by timeframe:")
print(f"  {'TF':>4s} | {'OO bias':>8s} | {'FC bias':>8s}")
print("  " + "-" * 25)
for tf in sorted(t['Timeframe'].unique()):
    sub = t[t['Timeframe'] == tf]
    oo_v = sub[sub['oo_implied_flat'].notna() & (sub['oo_implied_flat'] > 0)]
    fc_v = sub[sub['fc_implied_flat'].notna() & (sub['fc_implied_flat'] > 0)]
    oo_b = np.sum(oo_v['oo_implied_flat'] - oo_v['Actual_Sales']) / np.sum(oo_v['Actual_Sales']) * 100 if len(oo_v) > 0 else np.nan
    fc_b = np.sum(fc_v['fc_implied_flat'] - fc_v['Actual_Sales']) / np.sum(fc_v['Actual_Sales']) * 100 if len(fc_v) > 0 else np.nan
    oo_s = f"{oo_b:>+6.1f}%" if not np.isnan(oo_b) else "    n/a"
    fc_s = f"{fc_b:>+6.1f}%" if not np.isnan(fc_b) else "    n/a"
    print(f"  {tf:>4d} | {oo_s:>8s} | {fc_s:>8s}")

# ============================================================
# DIAGNOSTIC 3: Curve Drift
# ============================================================
print(f"\n--- DIAGNOSTIC 3: Curve Drift (train curve vs test-period actual) ---")

# Compute per-row drift: (curve_value - actual_ratio) / actual_ratio
td = t.copy()
td['oo_drift'] = np.where(td['oo_ratio'].notna() & (td['oo_ratio'] > 0.001) & td['oo_curve_val'].notna(),
                           (td['oo_curve_val'] - td['oo_ratio']) / td['oo_ratio'] * 100, np.nan)
td['cov_drift'] = np.where(td['fc_coverage'].notna() & (td['fc_coverage'] > 0.001) & td['cov_curve_val'].notna(),
                            (td['cov_curve_val'] - td['fc_coverage']) / td['fc_coverage'] * 100, np.nan)
td['bias_drift'] = np.where(td['fc_bias'].notna() & (td['fc_bias'] > 0.001) & td['bias_curve_val'].notna(),
                              (td['bias_curve_val'] - td['fc_bias']) / td['fc_bias'] * 100, np.nan)

# 3a: Cell-level drift summary
print(f"\n  3a. Overall drift summary (positive = curve overestimates ratio):")
print(f"  {'Metric':>20s} | {'OO ratio':>10s} | {'FC coverage':>12s} | {'FC bias':>10s}")
print("  " + "-" * 60)
for label, col in [('Median drift %', 'med'), ('Vol-wtd mean %', 'vw'), ('% cells curve>act', 'pct')]:
    vals = []
    for dcol in ['oo_drift', 'cov_drift', 'bias_drift']:
        valid = td[td[dcol].notna()]
        if label == 'Median drift %':
            vals.append(f"{valid[dcol].median():>+8.1f}%")
        elif label == 'Vol-wtd mean %':
            wm = np.average(valid[dcol], weights=valid['Actual_Sales'])
            vals.append(f"{wm:>+8.1f}%")
        else:
            pct = (valid[dcol] > 0).mean() * 100
            vals.append(f"{pct:>7.0f}% ")
    print(f"  {label:>20s} | {vals[0]:>10s} | {vals[1]:>12s} | {vals[2]:>10s}")
print(f"  {'N rows':>20s} | {td['oo_drift'].notna().sum():>10d} | {td['cov_drift'].notna().sum():>12d} | {td['bias_drift'].notna().sum():>10d}")

# 3b: Drift by lag
print(f"\n  3b. Volume-weighted drift by lag:")
print(f"  {'Lag':>4s} | {'OO drift':>9s} | {'Cov drift':>10s} | {'Bias drift':>11s}")
print("  " + "-" * 42)
for lag in sorted(td['Prediction_Lag'].unique()):
    sub = td[td['Prediction_Lag'] == lag]
    vals = []
    for dcol in ['oo_drift', 'cov_drift', 'bias_drift']:
        v = sub[sub[dcol].notna()]
        if len(v) > 0:
            wm = np.average(v[dcol], weights=v['Actual_Sales'])
            vals.append(f"{wm:>+7.1f}%")
        else:
            vals.append("     n/a")
    print(f"  {lag:>4d} | {vals[0]:>9s} | {vals[1]:>10s} | {vals[2]:>11s}")

# 3c: Drift by site
print(f"\n  3c. Volume-weighted drift by site:")
print(f"  {'Site':>15s} | {'Sales $M':>8s} | {'OO drift':>9s} | {'Cov drift':>10s} | {'Bias drift':>11s}")
print("  " + "-" * 62)
for gs in sorted(clean_sites):
    site = gs.split('|')[1]
    sub = td[td['gsa_site'] == gs]
    sales = sub['Actual_Sales'].sum()
    vals = []
    for dcol in ['oo_drift', 'cov_drift', 'bias_drift']:
        v = sub[sub[dcol].notna()]
        if len(v) > 0:
            wm = np.average(v[dcol], weights=v['Actual_Sales'])
            vals.append(f"{wm:>+7.1f}%")
        else:
            vals.append("     n/a")
    print(f"  {site:>15s} | {sales/1e6:>6.1f}M | {vals[0]:>9s} | {vals[1]:>10s} | {vals[2]:>11s}")

# 3d: Temporal trend — volume-weighted average ratios by month
print(f"\n  3d. Temporal trend (volume-weighted avg ratios by month):")
full = df[df['gsa_site'].isin(clean_sites) & (df['Actual_Sales'] > 0)].copy()
all_months = sorted(full['Reference_Month'].unique())

def print_temporal_trend(data, months, cutoff_date, label="Overall"):
    print(f"\n  [{label}]")
    print(f"  {'Month':>12s} | {'Period':>6s} | {'OO ratio':>9s} | {'FC cov':>7s} | {'FC bias':>8s} | {'N':>4s}")
    print("  " + "-" * 58)
    for m in months:
        sub = data[data['Reference_Month'] == m]
        if len(sub) == 0:
            continue
        period = "TRAIN" if m <= cutoff_date else "TEST"
        vals = []
        for col in ['oo_ratio', 'fc_coverage', 'fc_bias']:
            v = sub[sub[col].notna() & (sub[col] > 0) & (sub[col] < 10)]
            if len(v) > 0:
                wm = np.average(v[col], weights=v['Actual_Sales'])
                vals.append(f"{wm:>7.3f}")
            else:
                vals.append("    n/a")
        sep = "  " if period == "TRAIN" else ">>"
        print(f"  {m.date()} | {period:>6s} | {vals[0]:>9s} | {vals[1]:>7s} | {vals[2]:>8s} | {len(sub):>4d}")

print_temporal_trend(full, all_months, cutoff, "Overall")
for gs in top_sites:
    site = gs.split('|')[1]
    site_data = full[full['gsa_site'] == gs]
    site_months = sorted(site_data['Reference_Month'].unique())
    print_temporal_trend(site_data, site_months, cutoff, site)

# 3e: Drift direction vs bias direction (2x2 contingency)
print(f"\n  3e. Drift direction vs prediction bias direction (per cell):")
print(f"      (OO curve drift > 0 means curve overestimates OO ratio -> divides by too much -> underpredicts)")
cells = td.groupby(['gsa_site', 'Timeframe', 'Prediction_Lag']).agg(
    oo_drift_mean=('oo_drift', 'mean'),
    oo_pred_bias=('oo_implied_flat', lambda x: np.nan if x.isna().all() else
                  (x.dropna().values - td.loc[x.dropna().index, 'Actual_Sales'].values).sum() /
                  td.loc[x.dropna().index, 'Actual_Sales'].values.sum() * 100)
).dropna()
if len(cells) > 0:
    drift_pos = cells['oo_drift_mean'] > 0
    bias_neg = cells['oo_pred_bias'] < 0
    print(f"\n  {'':>25s} | {'Pred bias < 0':>14s} | {'Pred bias > 0':>14s}")
    print("  " + "-" * 58)
    print(f"  {'OO curve > test (drift+)':>25s} | {(drift_pos & bias_neg).sum():>10d}    | {(drift_pos & ~bias_neg).sum():>10d}")
    print(f"  {'OO curve < test (drift-)':>25s} | {(~drift_pos & bias_neg).sum():>10d}    | {(~drift_pos & ~bias_neg).sum():>10d}")
    n_diag = (drift_pos & bias_neg).sum() + (~drift_pos & ~bias_neg).sum()
    n_off = (drift_pos & ~bias_neg).sum() + (~drift_pos & bias_neg).sum()
    print(f"\n  Diagonal (drift explains bias): {n_diag}/{len(cells)} = {n_diag/len(cells)*100:.0f}%")
    print(f"  Off-diagonal (drift contradicts): {n_off}/{len(cells)} = {n_off/len(cells)*100:.0f}%")

# ============================================================
# WEIGHTED AVERAGE COMBINATION
# ============================================================
def weighted_avg(test_df, train_df, test_months, oo_col='oo_implied', fc_col='fc_implied',
                 visibility_adj=False, ref_ltoe=None):
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

                    # Visibility adjustment: down-weight OO when LT+OE has dropped
                    if visibility_adj and ref_ltoe is not None and has_oo:
                        lt_oe_cur = row['Avg_Weighted_Lead_Time'] + row['Avg_Weighted_Order_Earliness']
                        ref = None
                        for key in [(gs, tf, lag), ('FB', gs, lag)]:
                            if key in ref_ltoe:
                                ref = ref_ltoe[key]; break
                        if ref is not None and ref > 0.5 and lt_oe_cur > 0.5:
                            prog_ref = max(0.05, 1.0 - lag / ref)
                            prog_cur = max(0.05, 1.0 - lag / lt_oe_cur)
                            visibility = np.clip(prog_cur / prog_ref, 0.1, 1.0)
                        else:
                            visibility = 1.0
                        oo_f = oo_f * visibility

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

# FC-only (no OO signal)
print("  Running FC-only...")
test['fc_only'] = weighted_avg(test, train, test_months, 'oo_none', 'fc_implied_flat')

# FC-only debiased
print("  Running FC-only debiased...")
test['fc_debiased'] = weighted_avg(test, train, test_months, 'oo_none', 'fc_implied_debiased')

# V8g conditioned
print("  Running V8g (conditioned curves)...")
test['v8g'] = weighted_avg(test, train, test_months, 'oo_implied', 'fc_implied')

# V8i LT-normalized
print("  Running V8i (LT-normalized OO + flat FC)...")
test['v8i'] = weighted_avg(test, train, test_months, 'oo_implied_norm', 'fc_implied_norm')

# V8j effective lag
print("  Running V8j (effective lag OO + flat FC)...")
test['v8j'] = weighted_avg(test, train, test_months, 'oo_implied_elag', 'fc_implied_elag')

# Power-CDF sweep through full model
print("  Running power-CDF sweep through full model...")
for pv in P_VALUES:
    oo_col = f'oo_implied_p{int(pv*10)}'
    fc_col = f'fc_implied_p{int(pv*10)}'
    col_name = f'v8_p{int(pv*10)}'
    test[col_name] = weighted_avg(test, train, test_months, oo_col, fc_col)

# Also with visibility weighting at each p
print("  Running power-CDF + visibility sweep...")
for pv in P_VALUES:
    oo_col = f'oo_implied_p{int(pv*10)}'
    fc_col = f'fc_implied_p{int(pv*10)}'
    col_name = f'v8_p{int(pv*10)}_vis'
    test[col_name] = weighted_avg(test, train, test_months, oo_col, fc_col,
                                  visibility_adj=True, ref_ltoe=ref_lt_oe)

m2 = test['Reference_Month'] > test_months[0]

# Summary table: p sweep
print(f"\n  {'Model':>20s} | {'WMAPE':>7s} | {'Bias':>7s} | {'R²':>6s} | {'Acct':>6s}")
print("  " + "-" * 60)
# V8d baseline
ts = test[m2 & test['v8d'].notna() & test['gsa_site'].isin(clean_sites)]
y, p = ts['Actual_Sales'].values, ts['v8d'].values
w = np.sum(np.abs(y-p))/np.sum(y)*100
b = np.mean((p-y)/y)*100
r2 = 1 - np.sum((y-p)**2)/np.sum((y-np.mean(y))**2)
tf12 = ts[(ts['Timeframe']==12)&(ts['Prediction_Lag']==1)]
by_m = tf12.groupby('Reference_Month').agg({'Actual_Sales':'sum'})
by_m['pr'] = tf12.groupby('Reference_Month').apply(lambda g: test.loc[g.index, 'v8d'].sum()).values
aw = np.sum(np.abs(by_m['Actual_Sales']-by_m['pr']))/np.sum(by_m['Actual_Sales'])*100
print(f"  {'V8d (flat, p=0)':>20s} | {w:>5.1f}% | {b:>+5.1f}% | {r2:>.3f} | {aw:>4.1f}%")

# FC-only and FC-debiased
for label, col in [('FC-only', 'fc_only'), ('FC-only debiased', 'fc_debiased')]:
    ts = test[m2 & test[col].notna() & test['gsa_site'].isin(clean_sites)]
    y, p = ts['Actual_Sales'].values, ts[col].values
    w = np.sum(np.abs(y-p))/np.sum(y)*100
    b = np.mean((p-y)/y)*100
    r2 = 1 - np.sum((y-p)**2)/np.sum((y-np.mean(y))**2)
    tf12 = ts[(ts['Timeframe']==12)&(ts['Prediction_Lag']==1)]
    by_m = tf12.groupby('Reference_Month').agg({'Actual_Sales':'sum'})
    by_m['pr'] = tf12.groupby('Reference_Month').apply(lambda g: test.loc[g.index, col].sum()).values
    aw = np.sum(np.abs(by_m['Actual_Sales']-by_m['pr']))/np.sum(by_m['Actual_Sales'])*100
    print(f"  {label:>20s} | {w:>5.1f}% | {b:>+5.1f}% | {r2:>.3f} | {aw:>4.1f}%")

for pv in P_VALUES:
    for suffix, label_sfx in [('', ''), ('_vis', '+vis')]:
        col = f'v8_p{int(pv*10)}{suffix}'
        label = f'p={pv}{label_sfx}'
        ts = test[m2 & test[col].notna() & test['gsa_site'].isin(clean_sites)]
        y, p = ts['Actual_Sales'].values, ts[col].values
        w = np.sum(np.abs(y-p))/np.sum(y)*100
        b = np.mean((p-y)/y)*100
        r2 = 1 - np.sum((y-p)**2)/np.sum((y-np.mean(y))**2)
        tf12 = ts[(ts['Timeframe']==12)&(ts['Prediction_Lag']==1)]
        by_m = tf12.groupby('Reference_Month').agg({'Actual_Sales':'sum'})
        by_m['pr'] = tf12.groupby('Reference_Month').apply(lambda g: test.loc[g.index, col].sum()).values
        aw = np.sum(np.abs(by_m['Actual_Sales']-by_m['pr']))/np.sum(by_m['Actual_Sales'])*100
        print(f"  {label:>20s} | {w:>5.1f}% | {b:>+5.1f}% | {r2:>.3f} | {aw:>4.1f}%")

# Per-site: V8d vs FC-only vs FC-debiased
print(f"\n  Per-site (months 2+) — V8d vs FC-only vs FC-debiased:")
print(f"  {'Site':>20s} | {'V8d':>12s} | {'FC-only':>12s} | {'FC-debiased':>12s} | {'d-fc':>6s} | {'d-fcdb':>6s}")
print("  " + "-" * 80)
for gs in sorted(clean_sites):
    site = gs.split('|')[1]
    sub = test[m2 & (test['gsa_site']==gs) & (test['Actual_Sales']>0)]
    vals = {}
    for lbl, c in [('d','v8d'),('f','fc_only'),('fd','fc_debiased')]:
        v = sub[sub[c].notna()]
        if len(v) > 0:
            vals[lbl+'w'] = np.sum(np.abs(v['Actual_Sales']-v[c]))/np.sum(v['Actual_Sales'])*100
            vals[lbl+'b'] = np.mean((v[c]-v['Actual_Sales'])/v['Actual_Sales'])*100
    if all(k in vals for k in ['dw','fw','fdw']):
        print(f"  {site:>20s} | {vals['dw']:>5.1f}%{vals['db']:>+5.0f}% | {vals['fw']:>5.1f}%{vals['fb']:>+5.0f}% | {vals['fdw']:>5.1f}%{vals['fdb']:>+5.0f}% | {vals['fw']-vals['dw']:>+4.1f} | {vals['fdw']-vals['dw']:>+4.1f}")

# Per-lag: V8d vs FC-only vs FC-debiased
print(f"\n  WMAPE by lag — V8d vs FC-only vs FC-debiased:")
print(f"  {'Lag':>4s} | {'V8d':>8s} | {'FC-only':>8s} | {'FC-deb':>8s} | {'d-fc':>6s} | {'d-fcdb':>6s}")
print("  " + "-" * 50)
for lag in sorted(test['Prediction_Lag'].unique()):
    sub = test[m2 & (test['Prediction_Lag']==lag) & (test['Actual_Sales']>0) & test['gsa_site'].isin(clean_sites)]
    vals = {}
    for lbl, c in [('d','v8d'),('f','fc_only'),('fd','fc_debiased')]:
        v = sub[sub[c].notna()]
        if len(v) > 0:
            vals[lbl] = np.sum(np.abs(v['Actual_Sales']-v[c]))/np.sum(v['Actual_Sales'])*100
    if all(k in vals for k in ['d','f','fd']):
        print(f"  {lag:>4d} | {vals['d']:>6.1f}% | {vals['f']:>6.1f}% | {vals['fd']:>6.1f}% | {vals['f']-vals['d']:>+4.1f} | {vals['fd']-vals['d']:>+4.1f}")

# ============================================================
# K-FOLD RANDOM CV: Drift vs Noise Diagnostic
# ============================================================
N_FOLDS = 5
np.random.seed(42)

print(f"\n\n{'='*100}")
print(f"K-FOLD RANDOM CV ({N_FOLDS} folds) — Diagnosing drift vs noise")
print("=" * 100)

# Shuffle row indices
indices = np.arange(len(df))
np.random.shuffle(indices)
fold_size = len(df) // N_FOLDS

fold_metrics = {model: [] for model in ['V8d', 'FC-only', 'FC-debiased']}

for fold_i in range(N_FOLDS):
    fold_start = fold_i * fold_size
    fold_end = fold_start + fold_size if fold_i < N_FOLDS - 1 else len(df)
    test_idx = indices[fold_start:fold_end]
    train_idx = np.setdiff1d(indices, test_idx)

    f_train = df.iloc[train_idx].copy()
    f_test = df.iloc[test_idx].copy()

    # Build flat curves from this fold's training data
    oo_f, _ = build_conditioned_curves(f_train, 'oo_ratio', start='2023-01-01')
    cov_f, _ = build_conditioned_curves(f_train, 'fc_coverage', start='2023-01-01')
    bias_f, _ = build_conditioned_curves(f_train, 'fc_bias', start='2023-01-01')

    # Apply flat curves
    apply_flat_only(f_train, oo_f, cov_f, bias_f)
    apply_flat_only(f_test, oo_f, cov_f, bias_f)

    # Build FC debias curves for this fold
    fold_fc_debias = {}
    for (gs, tf, lag), grp in f_train.groupby(['gsa_site', 'Timeframe', 'Prediction_Lag']):
        v = grp[grp['fc_implied_flat'].notna() & (grp['fc_implied_flat'] > 0) & (grp['Actual_Sales'] > 0)].copy()
        if len(v) >= MIN_OBS_FLAT:
            ratio = v['Actual_Sales'].values / v['fc_implied_flat'].values
            mask = (ratio > 0.1) & (ratio < 10)
            if mask.sum() >= MIN_OBS_FLAT:
                v_m = v.iloc[mask]
                days = (v_m['Reference_Month'] - v_m['Reference_Month'].min()).dt.days / 365
                wts = (1 + days.values) ** RECENCY_POWER
                fold_fc_debias[(gs, tf, lag)] = np.average(ratio[mask], weights=wts)
    for (gs, lag), grp in f_train.groupby(['gsa_site', 'Prediction_Lag']):
        v = grp[grp['fc_implied_flat'].notna() & (grp['fc_implied_flat'] > 0) & (grp['Actual_Sales'] > 0)].copy()
        if len(v) >= MIN_OBS_FLAT:
            ratio = v['Actual_Sales'].values / v['fc_implied_flat'].values
            mask = (ratio > 0.1) & (ratio < 10)
            if mask.sum() >= MIN_OBS_FLAT:
                v_m = v.iloc[mask]
                days = (v_m['Reference_Month'] - v_m['Reference_Month'].min()).dt.days / 365
                wts = (1 + days.values) ** RECENCY_POWER
                fold_fc_debias[('FB', gs, lag)] = np.average(ratio[mask], weights=wts)

    # Apply debiased FC to both train and test
    for ds in [f_train, f_test]:
        fc_db = np.full(len(ds), np.nan)
        for i, (idx, row) in enumerate(ds.iterrows()):
            gs, tf, lag = row['gsa_site'], row['Timeframe'], row['Prediction_Lag']
            fc_val = row['fc_implied_flat']
            if np.isnan(fc_val) or fc_val <= 0:
                continue
            for key in [(gs, tf, lag), ('FB', gs, lag)]:
                if key in fold_fc_debias:
                    fc_db[i] = fc_val * fold_fc_debias[key]
                    break
        ds['fc_implied_debiased'] = np.maximum(fc_db, 0)
        ds['oo_none'] = np.nan

    # Run models
    f_test_months = sorted(f_test['Reference_Month'].unique())
    f_test['v8d'] = weighted_avg(f_test, f_train, f_test_months, 'oo_implied_flat', 'fc_implied_flat')
    f_test['fc_only'] = weighted_avg(f_test, f_train, f_test_months, 'oo_none', 'fc_implied_flat')
    f_test['fc_debiased'] = weighted_avg(f_test, f_train, f_test_months, 'oo_none', 'fc_implied_debiased')

    # Compute metrics for each model (clean sites only)
    ts = f_test[f_test['gsa_site'].isin(clean_sites) & (f_test['Actual_Sales'] > 0)]
    for model_name, col in [('V8d', 'v8d'), ('FC-only', 'fc_only'), ('FC-debiased', 'fc_debiased')]:
        v = ts[ts[col].notna()]
        if len(v) > 0:
            y, p = v['Actual_Sales'].values, v[col].values
            wmape = np.sum(np.abs(y - p)) / np.sum(y) * 100
            bias = np.mean((p - y) / y) * 100
            r2 = 1 - np.sum((y - p)**2) / np.sum((y - np.mean(y))**2)
            fold_metrics[model_name].append({'wmape': wmape, 'bias': bias, 'r2': r2})

    print(f"  Fold {fold_i+1}: V8d={fold_metrics['V8d'][-1]['wmape']:.1f}%  "
          f"FC-only={fold_metrics['FC-only'][-1]['wmape']:.1f}%  "
          f"FC-deb={fold_metrics['FC-debiased'][-1]['wmape']:.1f}%")

# Summary: Random CV vs Temporal
print(f"\n  {'':>20s} | {'Random CV (mean±std)':>25s} | {'Temporal':>10s} | {'Gap':>6s}")
print("  " + "-" * 70)

# Get temporal metrics for comparison
temporal_metrics = {}
ts_temp = test[m2 & test['gsa_site'].isin(clean_sites) & (test['Actual_Sales'] > 0)]
for model_name, col in [('V8d', 'v8d'), ('FC-only', 'fc_only'), ('FC-debiased', 'fc_debiased')]:
    v = ts_temp[ts_temp[col].notna()]
    if len(v) > 0:
        y, p = v['Actual_Sales'].values, v[col].values
        temporal_metrics[model_name] = np.sum(np.abs(y - p)) / np.sum(y) * 100

for model_name in ['V8d', 'FC-only', 'FC-debiased']:
    wmapes = [m['wmape'] for m in fold_metrics[model_name]]
    biases = [m['bias'] for m in fold_metrics[model_name]]
    r2s = [m['r2'] for m in fold_metrics[model_name]]
    t_wmape = temporal_metrics.get(model_name, float('nan'))
    gap = t_wmape - np.mean(wmapes)
    print(f"  {model_name:>20s} | {np.mean(wmapes):>5.1f}% ± {np.std(wmapes):>4.1f}% (bias {np.mean(biases):>+5.1f}%) | {t_wmape:>6.1f}% | {gap:>+4.1f}pp")

# Per-fold detail
print(f"\n  Per-fold WMAPE:")
print(f"  {'Fold':>6s} | {'V8d':>8s} | {'FC-only':>8s} | {'FC-deb':>8s} | {'V8d bias':>9s}")
print("  " + "-" * 50)
for i in range(N_FOLDS):
    v8d_m = fold_metrics['V8d'][i]
    fc_m = fold_metrics['FC-only'][i]
    fcd_m = fold_metrics['FC-debiased'][i]
    print(f"  {'F'+str(i+1):>6s} | {v8d_m['wmape']:>6.1f}% | {fc_m['wmape']:>6.1f}% | {fcd_m['wmape']:>6.1f}% | {v8d_m['bias']:>+7.1f}%")

# Per-lag comparison: random CV (pooled across folds) vs temporal
# Re-run one fold to get per-lag detail (use full random CV pool)
print(f"\n  WMAPE by lag — Random CV (fold 1) vs Temporal:")
# Recompute fold 1 for per-lag
fold_start = 0
fold_end = fold_size
t_idx = indices[fold_start:fold_end]
tr_idx = np.setdiff1d(indices, t_idx)
f1_test = df.iloc[t_idx].copy()
f1_train = df.iloc[tr_idx].copy()
oo_f1, _ = build_conditioned_curves(f1_train, 'oo_ratio', start='2023-01-01')
cov_f1, _ = build_conditioned_curves(f1_train, 'fc_coverage', start='2023-01-01')
bias_f1, _ = build_conditioned_curves(f1_train, 'fc_bias', start='2023-01-01')
apply_flat_only(f1_train, oo_f1, cov_f1, bias_f1)
apply_flat_only(f1_test, oo_f1, cov_f1, bias_f1)
f1_test['oo_none'] = np.nan
f1_train['oo_none'] = np.nan
f1_months = sorted(f1_test['Reference_Month'].unique())
f1_test['v8d'] = weighted_avg(f1_test, f1_train, f1_months, 'oo_implied_flat', 'fc_implied_flat')
f1_test['fc_only'] = weighted_avg(f1_test, f1_train, f1_months, 'oo_none', 'fc_implied_flat')

print(f"  {'Lag':>4s} | {'Rand V8d':>9s} | {'Temp V8d':>9s} | {'gap':>6s} | {'Rand FC':>9s} | {'Temp FC':>9s} | {'gap':>6s}")
print("  " + "-" * 65)
for lag in sorted(df['Prediction_Lag'].unique()):
    # Random fold 1
    sub_r = f1_test[(f1_test['Prediction_Lag']==lag) & (f1_test['Actual_Sales']>0) & f1_test['gsa_site'].isin(clean_sites)]
    # Temporal
    sub_t = test[m2 & (test['Prediction_Lag']==lag) & (test['Actual_Sales']>0) & test['gsa_site'].isin(clean_sites)]
    vals = {}
    for lbl, sub, col in [('rv', sub_r, 'v8d'), ('tv', sub_t, 'v8d'), ('rf', sub_r, 'fc_only'), ('tf', sub_t, 'fc_only')]:
        v = sub[sub[col].notna()]
        if len(v) > 0:
            vals[lbl] = np.sum(np.abs(v['Actual_Sales']-v[col]))/np.sum(v['Actual_Sales'])*100
    if all(k in vals for k in ['rv','tv','rf','tf']):
        print(f"  {lag:>4d} | {vals['rv']:>7.1f}% | {vals['tv']:>7.1f}% | {vals['tv']-vals['rv']:>+4.1f} | {vals['rf']:>7.1f}% | {vals['tf']:>7.1f}% | {vals['tf']-vals['rf']:>+4.1f}")

print(f"\n\nV7 reference: WMAPE=8.8%  bias=+0.5%  Acct=2.2%")
print("\nDone!")
