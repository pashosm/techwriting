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

# LT-aware kernel parameters
LT_BANDWIDTHS = [1.0, 1.5, 2.0, 3.0, 5.0]  # Bandwidth values to sweep
DEFAULT_LT_BW = 2.0                           # Default bandwidth
MIN_EFF_OBS_LT = 3.0                          # Minimum effective sample size for LT-aware

# OE-aware kernel parameters
OE_BANDWIDTHS = [1.0, 2.0, 3.0, 5.0]          # OE bandwidth values to sweep
DEFAULT_OE_BW = 3.0                             # Default OE bandwidth (conservative)

df = pd.read_csv('Dummy Training Data Customer 1 Try 2 11-Feb-2026.csv',
                 parse_dates=['Reference_Month', 'Target_Period_Start', 'Target_Period_End'])
if 'Unnamed: 0' in df.columns: df = df.drop(columns=['Unnamed: 0'])
if 'GSA' not in df.columns: df['GSA'] = 'Customer 1'
df['gsa_site'] = df['GSA'].astype(str) + '|' + df['Site'].astype(str)
df['Avg_Weighted_Order_Earliness'] = df['Avg_Weighted_Order_Earliness'].fillna(0)
df['Avg_Weighted_Lead_Time'] = df['Avg_Weighted_Lead_Time'].fillna(0)
df = df[~df['Site'].isin(['Site A'])].copy()

df['oo_ratio'] = np.where(df['Actual_Sales'] > 0, df['Open_Orders'] / df['Actual_Sales'], np.nan)
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

# ============================================================
# LT-AWARE KERNEL CURVES (V8h)
# ============================================================
def build_lt_aware_curves(train_df, ratio_col, start=None):
    """Store per-cell training data for LT+OE kernel weighting.
    Returns: flat dict, lt_data dict
    lt_data: {(gs, tf, lag): {'ratios': arr, 'lt': arr, 'oe': arr, 'recency_wts': arr}}
    """
    td = train_df if start is None else train_df[train_df['Reference_Month'] >= start]
    flat = {}
    lt_data = {}

    for (gs, tf, lag), grp in td.groupby(['gsa_site', 'Timeframe', 'Prediction_Lag']):
        valid = grp[grp[ratio_col].notna() & (grp[ratio_col] > 0) & (grp[ratio_col] < 10)].copy()
        if len(valid) < MIN_OBS_FLAT:
            continue
        days = (valid['Reference_Month'] - valid['Reference_Month'].min()).dt.days / 365
        wts = (1 + days.values) ** RECENCY_POWER
        flat_val = np.average(valid[ratio_col], weights=wts)
        flat[(gs, tf, lag)] = flat_val
        lt_data[(gs, tf, lag)] = {
            'ratios': valid[ratio_col].values,
            'lt': valid['Avg_Weighted_Lead_Time'].values,
            'oe': valid['Avg_Weighted_Order_Earliness'].values,
            'recency_wts': wts,
        }

    # Fallback curves pooled across TFs
    for (gs, lag), grp in td.groupby(['gsa_site', 'Prediction_Lag']):
        valid = grp[grp[ratio_col].notna() & (grp[ratio_col] > 0) & (grp[ratio_col] < 10)]
        if len(valid) >= MIN_OBS_FLAT:
            days = (valid['Reference_Month'] - valid['Reference_Month'].min()).dt.days / 365
            wts = (1 + days.values) ** RECENCY_POWER
            flat[('FB', gs, lag)] = np.average(valid[ratio_col], weights=wts)
            lt_data[('FB', gs, lag)] = {
                'ratios': valid[ratio_col].values,
                'lt': valid['Avg_Weighted_Lead_Time'].values,
                'oe': valid['Avg_Weighted_Order_Earliness'].values,
                'recency_wts': wts,
            }

    return flat, lt_data

def get_lt_aware_value(lt_data, flat, gs, tf, lag, lt_current, h,
                       oe_current=None, h_oe=None):
    """Get LT+OE kernel-weighted ratio.
    Falls back: 2D (LT+OE) -> 1D (LT-only) -> flat.
    """
    if np.isnan(lt_current) or lt_current == 0:
        for key in [(gs, tf, lag), ('FB', gs, lag)]:
            if key in flat:
                return flat[key], 'flat'
        return None, None

    use_oe = (h_oe is not None and oe_current is not None
              and not np.isnan(oe_current) and oe_current > 0)

    for key in [(gs, tf, lag), ('FB', gs, lag)]:
        if key in lt_data:
            d = lt_data[key]
            lt_kernel = np.exp(-((d['lt'] - lt_current) ** 2) / (2 * h * h))

            # Try 2D kernel (LT + OE)
            if use_oe:
                oe_kernel = np.exp(-((d['oe'] - oe_current) ** 2) / (2 * h_oe * h_oe))
                combined_2d = d['recency_wts'] * lt_kernel * oe_kernel
                wt_sum_2d = np.sum(combined_2d)
                if wt_sum_2d > 0:
                    n_eff_2d = wt_sum_2d ** 2 / np.sum(combined_2d ** 2)
                    if n_eff_2d >= MIN_EFF_OBS_LT:
                        return np.average(d['ratios'], weights=combined_2d), 'lt_oe_aware'

            # Fallback: 1D kernel (LT only)
            combined_wts = d['recency_wts'] * lt_kernel
            wt_sum = np.sum(combined_wts)
            if wt_sum > 0:
                n_eff = wt_sum ** 2 / np.sum(combined_wts ** 2)
                if n_eff >= MIN_EFF_OBS_LT:
                    return np.average(d['ratios'], weights=combined_wts), 'lt_aware'
        if key in flat:
            return flat[key], 'flat'
    return None, None

def apply_lt_aware(dset, lt_data, oo_flat, cov_f, bias_f, h, h_oe=None):
    """Apply LT+OE kernel OO curves + flat FC curves. Returns OO type counts."""
    oo_impl = np.full(len(dset), np.nan)
    fc_impl = np.full(len(dset), np.nan)
    oo_types = {}

    for i, (idx, row) in enumerate(dset.iterrows()):
        gs, tf, lag = row['gsa_site'], row['Timeframe'], row['Prediction_Lag']
        lt = row['Avg_Weighted_Lead_Time']
        oe = row['Avg_Weighted_Order_Earliness']

        # OO: LT+OE kernel (falls back to LT-only, then flat)
        f, ftype = get_lt_aware_value(lt_data, oo_flat, gs, tf, lag, lt, h,
                                       oe_current=oe, h_oe=h_oe)
        if f and f > 0.001:
            oo_impl[i] = row['Open_Orders'] / f
            if ftype: oo_types[ftype] = oo_types.get(ftype, 0) + 1

        # FC: flat (same as V8d)
        for key in [(gs, tf, lag), ('FB', gs, lag)]:
            cf = cov_f.get(key)
            bf = bias_f.get(key)
            if cf and bf and cf > 0.001 and bf > 0.001 and row['Forecast_Value'] > 0:
                fc_impl[i] = row['Forecast_Value'] / bf / cf
                break

    dset['oo_implied_lt'] = np.maximum(oo_impl, 0)
    dset['fc_implied_lt'] = np.maximum(fc_impl, 0)
    return oo_types

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

# Build LT-aware OO data store
oo_flat_lt, oo_lt_data = build_lt_aware_curves(train, 'oo_ratio', start='2023-01-01')

print(f"\n  OO curves:  {len(oo_flat)} flat, {len(oo_cond)} conditioned")
print(f"  OO LT-aware: {len(oo_lt_data)} cells with stored training data")
print(f"  FC cov:     {len(cov_flat)} flat, {len(cov_cond)} conditioned")
print(f"  FC bias:    {len(bias_flat)} flat, {len(bias_cond)} conditioned")

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

# Apply conditioned curves
print("\n--- Applying curves ---")
oo_t, fc_t = apply_curves(train, oo_flat, oo_cond, cov_flat, cov_cond, bias_flat, bias_cond)
oo_te, fc_te = apply_curves(test, oo_flat, oo_cond, cov_flat, cov_cond, bias_flat, bias_cond)
print(f"  Test OO: {oo_te}")
print(f"  Test FC: {fc_te}")

# Apply flat curves for baseline
apply_flat_only(train, oo_flat, cov_flat, bias_flat)
apply_flat_only(test, oo_flat, cov_flat, bias_flat)

# Apply LT-aware curves (OO only, FC uses flat)
print(f"  Applying LT+OE kernel (h_lt={DEFAULT_LT_BW:.1f}, h_oe={DEFAULT_OE_BW:.1f})...")
lt_types_train = apply_lt_aware(train, oo_lt_data, oo_flat_lt, cov_flat, bias_flat, DEFAULT_LT_BW, DEFAULT_OE_BW)
lt_types_test = apply_lt_aware(test, oo_lt_data, oo_flat_lt, cov_flat, bias_flat, DEFAULT_LT_BW, DEFAULT_OE_BW)
print(f"  Test OO kernel types: {lt_types_test}")

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
# SIGNAL QUALITY: LT-aware OO (V8h)
# ============================================================
print(f"\n\n{'='*100}")
print("SIGNAL QUALITY: OO LT+OE kernel vs Flat vs Conditioned")
print("=" * 100)

print(f"\n  {'Signal':>25s} | {'WMAPE':>7s} | {'Bias':>7s} | {'R²':>6s}")
print("  " + "-" * 55)
for label, col in [('OO flat', 'oo_implied_flat'), ('OO conditioned', 'oo_implied'),
                    ('OO lt+oe kernel', 'oo_implied_lt')]:
    v = test[test[col].notna() & (test[col]>0) & (test['Actual_Sales']>0) & test['gsa_site'].isin(clean_sites)]
    y, p = v['Actual_Sales'].values, v[col].values
    w = np.sum(np.abs(y-p))/np.sum(y)*100
    b = np.mean((p-y)/y)*100
    r2 = 1 - np.sum((y-p)**2)/np.sum((y-np.mean(y))**2)
    print(f"  {label:>25s} | {w:>5.1f}% | {b:>+5.1f}% | {r2:>.3f}")

print(f"\n  OO implied per-site (flat vs LT+OE kernel):")
print(f"  {'Site':>15s} | {'Flat':>12s} | {'LT+OE kern':>12s} | {'Delta':>8s}")
print("  " + "-" * 55)
for gs in top_sites:
    site = gs.split('|')[1]
    fw, fb, lw, lb = 0, 0, 0, 0
    for col, lbl in [('oo_implied_flat', 'flat'), ('oo_implied_lt', 'lt')]:
        v = test[(test['gsa_site']==gs) & test[col].notna() & (test[col]>0) & (test['Actual_Sales']>0)]
        if len(v) > 0:
            w = np.sum(np.abs(v['Actual_Sales']-v[col]))/np.sum(v['Actual_Sales'])*100
            b = np.mean((v[col]-v['Actual_Sales'])/v['Actual_Sales'])*100
            if lbl == 'flat': fw, fb = w, b
            else: lw, lb = w, b
    print(f"  {site:>15s} | {fw:>5.1f}%{fb:>+5.0f}% | {lw:>5.1f}%{lb:>+5.0f}% | {lw-fw:>+5.1f}pp")

# ============================================================
# BANDWIDTH SWEEP
# ============================================================
print(f"\n\n{'='*100}")
print("BANDWIDTH SWEEP: LT bandwidth (h_oe fixed at default)")
print("=" * 100)
print(f"\n  {'h_lt':>5s} | {'WMAPE':>7s} | {'Bias':>7s} | {'R²':>6s} | {'%lt_oe':>7s} | {'%lt':>5s} | {'%flat':>6s}")
print("  " + "-" * 55)
for h in LT_BANDWIDTHS:
    types = apply_lt_aware(test, oo_lt_data, oo_flat_lt, cov_flat, bias_flat, h, DEFAULT_OE_BW)
    v = test[test['oo_implied_lt'].notna() & (test['oo_implied_lt']>0) &
             (test['Actual_Sales']>0) & test['gsa_site'].isin(clean_sites)]
    y, p = v['Actual_Sales'].values, v['oo_implied_lt'].values
    wmape = np.sum(np.abs(y-p))/np.sum(y)*100
    bias = np.mean((p-y)/y)*100
    r2 = 1 - np.sum((y-p)**2)/np.sum((y-np.mean(y))**2)
    tot = max(sum(types.values()), 1)
    pct_2d = types.get('lt_oe_aware', 0) / tot * 100
    pct_lt = types.get('lt_aware', 0) / tot * 100
    pct_fl = types.get('flat', 0) / tot * 100
    print(f"  {h:>5.1f} | {wmape:>5.1f}% | {bias:>+5.1f}% | {r2:>.3f} | {pct_2d:>5.1f}% | {pct_lt:>4.1f}% | {pct_fl:>4.1f}%")

print(f"\n\n{'='*100}")
print("BANDWIDTH SWEEP: OE bandwidth (h_lt fixed at default)")
print("=" * 100)
print(f"\n  {'h_oe':>5s} | {'WMAPE':>7s} | {'Bias':>7s} | {'R²':>6s} | {'%lt_oe':>7s} | {'%lt':>5s} | {'%flat':>6s}")
print("  " + "-" * 55)
# Also test h_oe=None (LT-only) as reference
types = apply_lt_aware(test, oo_lt_data, oo_flat_lt, cov_flat, bias_flat, DEFAULT_LT_BW, None)
v = test[test['oo_implied_lt'].notna() & (test['oo_implied_lt']>0) &
         (test['Actual_Sales']>0) & test['gsa_site'].isin(clean_sites)]
y, p = v['Actual_Sales'].values, v['oo_implied_lt'].values
wmape = np.sum(np.abs(y-p))/np.sum(y)*100
bias = np.mean((p-y)/y)*100
r2 = 1 - np.sum((y-p)**2)/np.sum((y-np.mean(y))**2)
tot = max(sum(types.values()), 1)
pct_lt = types.get('lt_aware', 0) / tot * 100
pct_fl = types.get('flat', 0) / tot * 100
print(f"  {'none':>5s} | {wmape:>5.1f}% | {bias:>+5.1f}% | {r2:>.3f} | {'0.0':>5s}% | {pct_lt:>4.1f}% | {pct_fl:>4.1f}%  (LT-only ref)")
for h_oe in OE_BANDWIDTHS:
    types = apply_lt_aware(test, oo_lt_data, oo_flat_lt, cov_flat, bias_flat, DEFAULT_LT_BW, h_oe)
    v = test[test['oo_implied_lt'].notna() & (test['oo_implied_lt']>0) &
             (test['Actual_Sales']>0) & test['gsa_site'].isin(clean_sites)]
    y, p = v['Actual_Sales'].values, v['oo_implied_lt'].values
    wmape = np.sum(np.abs(y-p))/np.sum(y)*100
    bias = np.mean((p-y)/y)*100
    r2 = 1 - np.sum((y-p)**2)/np.sum((y-np.mean(y))**2)
    tot = max(sum(types.values()), 1)
    pct_2d = types.get('lt_oe_aware', 0) / tot * 100
    pct_lt = types.get('lt_aware', 0) / tot * 100
    pct_fl = types.get('flat', 0) / tot * 100
    print(f"  {h_oe:>5.1f} | {wmape:>5.1f}% | {bias:>+5.1f}% | {r2:>.3f} | {pct_2d:>5.1f}% | {pct_lt:>4.1f}% | {pct_fl:>4.1f}%")

# Restore default bandwidths for V8h model run
apply_lt_aware(train, oo_lt_data, oo_flat_lt, cov_flat, bias_flat, DEFAULT_LT_BW, DEFAULT_OE_BW)
apply_lt_aware(test, oo_lt_data, oo_flat_lt, cov_flat, bias_flat, DEFAULT_LT_BW, DEFAULT_OE_BW)

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
print("FULL MODEL: V8d (flat) vs V8g (conditioned) vs V8h (LT+OE kernel)")
print("=" * 100)

# V8d baseline
print("\n  Running V8d (flat curves)...")
test['v8d'] = weighted_avg(test, train, test_months, 'oo_implied_flat', 'fc_implied_flat')

# V8g conditioned
print("  Running V8g (conditioned curves)...")
test['v8g'] = weighted_avg(test, train, test_months, 'oo_implied', 'fc_implied')

# V8h LT-aware
print("  Running V8h (LT+OE kernel OO + flat FC)...")
test['v8h'] = weighted_avg(test, train, test_months, 'oo_implied_lt', 'fc_implied_lt')

m2 = test['Reference_Month'] > test_months[0]

print(f"\n  {'Model':>15s} | {'WMAPE':>7s} | {'Bias':>7s} | {'R²':>6s} | {'Acct':>6s}")
print("  " + "-" * 50)
for label, col in [('V8d (flat)', 'v8d'), ('V8g (cond)', 'v8g'), ('V8h (LT+OE)', 'v8h')]:
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
print(f"  {'Site':>20s} | {'V8d':>12s} | {'V8g':>12s} | {'V8h':>12s} | {'d-h':>8s}")
print("  " + "-" * 75)
for gs in sorted(clean_sites):
    site = gs.split('|')[1]
    sub = test[m2 & (test['gsa_site']==gs) & (test['Actual_Sales']>0)]
    bv = sub[sub['v8d'].notna()]
    cv = sub[sub['v8g'].notna()]
    hv = sub[sub['v8h'].notna()]
    if len(bv)>0 and len(cv)>0 and len(hv)>0:
        bw = np.sum(np.abs(bv['Actual_Sales']-bv['v8d']))/np.sum(bv['Actual_Sales'])*100
        cw = np.sum(np.abs(cv['Actual_Sales']-cv['v8g']))/np.sum(cv['Actual_Sales'])*100
        hw = np.sum(np.abs(hv['Actual_Sales']-hv['v8h']))/np.sum(hv['Actual_Sales'])*100
        bb = np.mean((bv['v8d']-bv['Actual_Sales'])/bv['Actual_Sales'])*100
        cb = np.mean((cv['v8g']-cv['Actual_Sales'])/cv['Actual_Sales'])*100
        hb = np.mean((hv['v8h']-hv['Actual_Sales'])/hv['Actual_Sales'])*100
        print(f"  {site:>20s} | {bw:>5.1f}%{bb:>+5.0f}% | {cw:>5.1f}%{cb:>+5.0f}% | {hw:>5.1f}%{hb:>+5.0f}% | {hw-bw:>+5.1f}pp")

# Per-lag
print(f"\n  WMAPE by lag:")
print(f"  {'Lag':>4s} | {'V8d':>8s} | {'V8g':>8s} | {'V8h':>8s} | {'d-h':>8s}")
print("  " + "-" * 45)
for lag in sorted(test['Prediction_Lag'].unique()):
    sub = test[m2 & (test['Prediction_Lag']==lag) & (test['Actual_Sales']>0) & test['gsa_site'].isin(clean_sites)]
    bv = sub[sub['v8d'].notna()]
    cv = sub[sub['v8g'].notna()]
    hv = sub[sub['v8h'].notna()]
    if len(bv)>0:
        bw = np.sum(np.abs(bv['Actual_Sales']-bv['v8d']))/np.sum(bv['Actual_Sales'])*100
        cw = np.sum(np.abs(cv['Actual_Sales']-cv['v8g']))/np.sum(cv['Actual_Sales'])*100
        hw = np.sum(np.abs(hv['Actual_Sales']-hv['v8h']))/np.sum(hv['Actual_Sales'])*100
        print(f"  {lag:>4d} | {bw:>6.1f}% | {cw:>6.1f}% | {hw:>6.1f}% | {hw-bw:>+6.1f}pp")

# Account monthly
print(f"\n  Account TF=12/L=1:")
print(f"  {'Month':>12s} | {'Actual':>8s} | {'V8d':>15s} | {'V8g':>15s} | {'V8h':>15s}")
print("  " + "-" * 80)
for ref in test_months[:7]:
    sub = test[(test['Reference_Month']==ref) & (test['Timeframe']==12) &
                (test['Prediction_Lag']==1) & test['gsa_site'].isin(clean_sites)]
    act = sub['Actual_Sales'].sum()
    bp = sub['v8d'].sum()
    cp = sub['v8g'].sum()
    hp = sub['v8h'].sum()
    if act > 0:
        print(f"  {ref.date()} | ${act/1e6:>5.0f}M | ${bp/1e6:>5.0f}M ({(bp-act)/act*100:>+5.1f}%) | ${cp/1e6:>5.0f}M ({(cp-act)/act*100:>+5.1f}%) | ${hp/1e6:>5.0f}M ({(hp-act)/act*100:>+5.1f}%)")

print(f"\n\nV7 reference: WMAPE=8.8%  bias=+0.5%  Acct=2.2%")
print("\nDone!")
