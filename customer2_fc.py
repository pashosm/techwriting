"""
Customer 2: FC-only model — Temporal vs Random CV
==================================================
Runs the same FC-only + historical-average weighted model from V8g
on Customer 2 data.  Compares temporal train/test split with 5-fold
random CV to check whether the FC signal is similarly stationary.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from numpy.linalg import lstsq

RECENCY_POWER = 2
MIN_OBS_FLAT = 3

# ---- Load & parse ----
df = pd.read_csv('Dummy_Training_Data_Customer_2_12-Feb-26.csv',
                 parse_dates=['Reference_Month', 'Target_Period_Start', 'Target_Period_End'])
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Parse dollar-formatted columns
def parse_dollar(s):
    if isinstance(s, str):
        s = s.replace('$', '').replace(',', '').strip()
        if s in ('', '-', 'N/A', 'n/a'):
            return np.nan
        if s.startswith('(') and s.endswith(')'):
            return -float(s[1:-1])
        return float(s)
    return s

for col in ['Actual_Sales', 'Open_Orders', 'Historical_Sales_Lag1', 'Historical_Sales_Lag12']:
    df[col] = df[col].apply(parse_dollar)

df['gsa_site'] = df['GSA'].astype(str) + '|' + df['Site'].astype(str)
df['Avg_Weighted_Order_Earliness'] = df['Avg_Weighted_Order_Earliness'].fillna(0)
df['Avg_Weighted_Lead_Time'] = df['Avg_Weighted_Lead_Time'].fillna(0)

# Compute ratios
df['oo_ratio'] = np.where(df['Actual_Sales'] > 0, df['Open_Orders'] / df['Actual_Sales'], np.nan)
df['fc_coverage'] = np.where(df['Actual_Sales'] > 0, df['Covered_Orders'] / df['Actual_Sales'], np.nan)
df['fc_bias'] = np.where(df['Covered_Orders'] > 0, df['Forecast_Value'] / df['Covered_Orders'], np.nan)

gsa_sites = sorted(df['gsa_site'].unique())
# Sites with meaningful forecast data
fc_sites = [gs for gs in gsa_sites if 'Other' not in gs and 'Site M' not in gs]

print(f"Customer 2 data: {len(df)} rows, {df['Reference_Month'].nunique()} months, {len(gsa_sites)} sites")
print(f"Forecast sites: {fc_sites}")
print(f"Rows with forecasts: {(df['Has_Forecast']==1).sum()} ({100*(df['Has_Forecast']==1).mean():.0f}%)")

cutoff = df['Reference_Month'].quantile(0.8)
train = df[df['Reference_Month'] <= cutoff].copy()
test = df[df['Reference_Month'] > cutoff].copy()
test_months = sorted(test['Reference_Month'].unique())
print(f"Train: {len(train)} rows ({train['Reference_Month'].min().date()} – {train['Reference_Month'].max().date()})")
print(f"Test:  {len(test)} rows ({test['Reference_Month'].min().date()} – {test['Reference_Month'].max().date()})")
print(f"Test months: {[str(m.date()) for m in test_months]}")

# ---- Curve building (flat only, same as V8g) ----
def build_flat_curves(train_df, ratio_col, start=None):
    td = train_df if start is None else train_df[train_df['Reference_Month'] >= start]
    flat = {}
    for (gs, tf, lag), grp in td.groupby(['gsa_site', 'Timeframe', 'Prediction_Lag']):
        valid = grp[grp[ratio_col].notna() & (grp[ratio_col] > 0) & (grp[ratio_col] < 10)]
        if len(valid) < MIN_OBS_FLAT:
            continue
        days = (valid['Reference_Month'] - valid['Reference_Month'].min()).dt.days / 365
        wts = (1 + days.values) ** RECENCY_POWER
        flat[(gs, tf, lag)] = np.average(valid[ratio_col], weights=wts)
    # Fallback: pooled across TFs
    for (gs, lag), grp in td.groupby(['gsa_site', 'Prediction_Lag']):
        valid = grp[grp[ratio_col].notna() & (grp[ratio_col] > 0) & (grp[ratio_col] < 10)]
        if len(valid) >= MIN_OBS_FLAT:
            days = (valid['Reference_Month'] - valid['Reference_Month'].min()).dt.days / 365
            wts = (1 + days.values) ** RECENCY_POWER
            flat[('FB', gs, lag)] = np.average(valid[ratio_col], weights=wts)
    return flat

def apply_fc_curves(dset, cov_f, bias_f):
    fc_impl = np.full(len(dset), np.nan)
    for i, (idx, row) in enumerate(dset.iterrows()):
        gs, tf, lag = row['gsa_site'], row['Timeframe'], row['Prediction_Lag']
        for key in [(gs, tf, lag), ('FB', gs, lag)]:
            cf = cov_f.get(key)
            bf = bias_f.get(key)
            if cf and bf and cf > 0.001 and bf > 0.001 and row['Forecast_Value'] > 0:
                fc_impl[i] = row['Forecast_Value'] / bf / cf
                break
    dset['fc_implied_flat'] = np.maximum(fc_impl, 0)
    dset['oo_none'] = np.nan

# ---- Multi-vintage forecast functions ----
VINTAGE_RECENCY_POWER = 1.5

def build_forecast_registry(data_df):
    """Index forecast-bearing rows by target period for O(1) lookup.
    Returns {(gsa_site, TP_Start, TP_End): [vintage_dicts sorted by lag asc]}"""
    registry = {}
    fc_rows = data_df[data_df['Has_Forecast'] == 1]
    for _, row in fc_rows.iterrows():
        key = (row['gsa_site'], row['Target_Period_Start'], row['Target_Period_End'])
        if key not in registry:
            registry[key] = []
        registry[key].append({
            'ref_month': row['Reference_Month'],
            'lag': row['Prediction_Lag'],
            'fc_value': row['Forecast_Value'],
            'covered_orders': row['Covered_Orders'],
            'timeframe': row['Timeframe'],
            'gsa_site': row['gsa_site'],
        })
    for key in registry:
        registry[key].sort(key=lambda v: v['lag'])
    return registry

def _vintage_to_estimate(vintage, cov_f, bias_f):
    """Convert one forecast vintage to an implied-actual estimate using curves
    at the vintage's original (site, tf, lag). Returns (estimate, success)."""
    gs = vintage['gsa_site']
    tf = vintage['timeframe']
    lag = vintage['lag']
    fc_val = vintage['fc_value']
    if fc_val <= 0:
        return np.nan, False
    for key in [(gs, tf, lag), ('FB', gs, lag)]:
        cf = cov_f.get(key)
        bf = bias_f.get(key)
        if cf and bf and cf > 0.001 and bf > 0.001:
            return fc_val / bf / cf, True
    return np.nan, False

def _combine_vintage_estimates(estimates_with_lags):
    """Combine multiple vintage estimates with inverse-lag power weighting.
    estimates_with_lags: list of (estimate, original_lag) tuples."""
    if not estimates_with_lags:
        return np.nan
    total_w, total_val = 0.0, 0.0
    for est, orig_lag in estimates_with_lags:
        w = 1.0 / (orig_lag ** VINTAGE_RECENCY_POWER)
        total_w += w
        total_val += w * est
    return total_val / total_w if total_w > 0 else np.nan

def apply_fc_curves_multi(dset, cov_f, bias_f, registry, causality='temporal'):
    """Apply multi-vintage FC curves. For each row, look up ALL causally-valid
    forecasts for its target period and combine them.

    causality='temporal': only use forecasts with ref_month <= row's ref_month
    causality='fold': registry already restricted to train fold, no extra filter
    """
    n = len(dset)
    fc_impl_multi = np.full(n, np.nan)
    fc_impl_flat = np.full(n, np.nan)
    n_vintages = np.zeros(n, dtype=int)
    best_lag = np.full(n, np.nan)

    for i, (idx, row) in enumerate(dset.iterrows()):
        gs = row['gsa_site']
        tf = row['Timeframe']
        lag = row['Prediction_Lag']
        tp_key = (gs, row['Target_Period_Start'], row['Target_Period_End'])

        # Single-vintage (current behavior, for comparison)
        if row['Forecast_Value'] > 0:
            est, ok = _vintage_to_estimate({
                'gsa_site': gs, 'timeframe': tf, 'lag': lag,
                'fc_value': row['Forecast_Value'],
            }, cov_f, bias_f)
            if ok:
                fc_impl_flat[i] = max(est, 0)

        # Multi-vintage
        if tp_key not in registry:
            continue
        valid_vintages = []
        for v in registry[tp_key]:
            if causality == 'temporal' and v['ref_month'] > row['Reference_Month']:
                continue
            est, ok = _vintage_to_estimate(v, cov_f, bias_f)
            if ok:
                valid_vintages.append((max(est, 0), v['lag']))
        if valid_vintages:
            fc_impl_multi[i] = max(_combine_vintage_estimates(valid_vintages), 0)
            n_vintages[i] = len(valid_vintages)
            best_lag[i] = min(vl for _, vl in valid_vintages)

    dset['fc_implied_multi'] = fc_impl_multi
    dset['fc_implied_flat'] = fc_impl_flat
    dset['n_vintages_used'] = n_vintages
    dset['best_vintage_lag'] = best_lag
    dset['oo_none'] = np.nan

def weighted_avg(test_df, train_df, t_months, fc_col='fc_implied_flat'):
    """FC-only weighted average (no OO signal). Falls back to historical avg."""
    results = test_df.copy()
    results['pred'] = np.nan
    # Historical averages
    hist = {}
    for gs in gsa_sites:
        for tf in train_df['Timeframe'].unique():
            sub = train_df[(train_df['gsa_site']==gs) & (train_df['Timeframe']==tf)]
            if len(sub) > 0:
                d = (sub['Reference_Month'] - sub['Reference_Month'].min()).dt.days / 365
                w = (1 + d.values) ** RECENCY_POWER
                hist[(gs, tf)] = np.average(sub['Actual_Sales'], weights=w)

    for mi, month in enumerate(t_months):
        if mi == 0:
            prior = train_df[train_df['Reference_Month'] >= train_df['Reference_Month'].quantile(0.8)]
        else:
            prior = results[results['Reference_Month'].isin(t_months[max(0, mi-3):mi])]

        cmask = results['Reference_Month'] == month
        for gs in results.loc[cmask, 'gsa_site'].unique():
            for tf in results['Timeframe'].unique():
                ps = prior[(prior['gsa_site']==gs) & (prior['Timeframe']==tf) & (prior['Actual_Sales']>0)]
                fc_v = ps[ps[fc_col].notna() & (ps[fc_col]>0)]
                fc_mape = np.mean(np.abs(fc_v[fc_col] - fc_v['Actual_Sales']) / fc_v['Actual_Sales']) if len(fc_v) >= 1 else 1.0

                rmask = cmask & (results['gsa_site']==gs) & (results['Timeframe']==tf)
                for idx in results[rmask].index:
                    row = results.loc[idx]
                    lag = row['Prediction_Lag']
                    has_fc = not np.isnan(row[fc_col]) and row[fc_col] > 0
                    fc_f = max(0.3, 1.0 - 0.06*(lag-1)) if has_fc else 0
                    w_fc = (1/max(fc_mape, 0.01)) * fc_f if has_fc else 0
                    hv = hist.get((gs, tf), 0)
                    w_h = (1/0.5) * 0.1 * lag/12 if hv > 0 else 0
                    tot = w_fc + w_h
                    if tot > 0:
                        results.at[idx, 'pred'] = max(0,
                            (w_fc/tot) * (row[fc_col] if has_fc else 0) +
                            (w_h/tot) * hv)
                    elif hv > 0:
                        results.at[idx, 'pred'] = hv
    return results['pred']


# ==================================================================
# TEMPORAL EVALUATION
# ==================================================================
print(f"\n\n{'='*100}")
print("TEMPORAL EVALUATION — Customer 2 FC-only (single vs multi-vintage)")
print("=" * 100)

cov_f = build_flat_curves(train, 'fc_coverage', start='2023-01-01')
bias_f = build_flat_curves(train, 'fc_bias', start='2023-01-01')
print(f"  FC curves built: {len(cov_f)} coverage, {len(bias_f)} bias")

# Build forecast registry from full df (causality enforced at lookup time)
registry_all = build_forecast_registry(df)
print(f"  Forecast registry: {len(registry_all)} target periods with forecasts")

# Apply multi-vintage (produces both fc_implied_flat and fc_implied_multi)
apply_fc_curves_multi(train, cov_f, bias_f, registry_all, causality='temporal')
apply_fc_curves_multi(test, cov_f, bias_f, registry_all, causality='temporal')

# Coverage comparison
n_single = test['fc_implied_flat'].notna().sum()
n_multi = test['fc_implied_multi'].notna().sum()
print(f"  Single-vintage FC coverage: {n_single}/{len(test)} ({100*n_single/len(test):.0f}%)")
print(f"  Multi-vintage FC coverage:  {n_multi}/{len(test)} ({100*n_multi/len(test):.0f}%)")
print(f"  Rows gained: {n_multi - n_single}")

# Vintage depth stats (FC sites only)
m_fc_all = test['gsa_site'].isin(fc_sites)
vdepth = test.loc[m_fc_all, 'n_vintages_used']
print(f"  Vintage depth (FC sites): mean={vdepth.mean():.1f}, median={vdepth.median():.0f}, max={vdepth.max():.0f}")

# Run both models
test['fc_single'] = weighted_avg(test, train, test_months, 'fc_implied_flat')
test['fc_multi'] = weighted_avg(test, train, test_months, 'fc_implied_multi')

# Overall metrics (FC sites only)
m_fc = test['gsa_site'].isin(fc_sites) & (test['Actual_Sales'] > 0)
ts = test[m_fc]

print(f"\n  {'Model':>20s} | {'WMAPE':>7s} | {'Bias':>7s} | {'N':>5s}")
print("  " + "-" * 50)
for label, col in [('Single-vintage', 'fc_single'), ('Multi-vintage', 'fc_multi')]:
    v = ts[ts[col].notna()]
    w = np.sum(np.abs(v['Actual_Sales'] - v[col])) / np.sum(v['Actual_Sales']) * 100
    b = np.mean((v[col] - v['Actual_Sales']) / v['Actual_Sales']) * 100
    print(f"  {label:>20s} | {w:>5.1f}% | {b:>+5.1f}% | {len(v):>5d}")

# Per-site comparison
print(f"\n  Per-site WMAPE — Single vs Multi-vintage (temporal):")
print(f"  {'Site':>25s} | {'Single':>8s} | {'Multi':>8s} | {'Diff':>6s} | {'FC1':>4s} | {'FCn':>4s} | {'N':>4s}")
print("  " + "-" * 75)
for gs in sorted(fc_sites):
    sub = test[(test['gsa_site']==gs) & (test['Actual_Sales']>0)]
    n_s = sub['fc_implied_flat'].notna().sum()
    n_m = sub['fc_implied_multi'].notna().sum()
    v_s = sub[sub['fc_single'].notna()]
    v_m = sub[sub['fc_multi'].notna()]
    if len(v_s) > 0 and len(v_m) > 0:
        ws = np.sum(np.abs(v_s['Actual_Sales'] - v_s['fc_single'])) / np.sum(v_s['Actual_Sales']) * 100
        wm = np.sum(np.abs(v_m['Actual_Sales'] - v_m['fc_multi'])) / np.sum(v_m['Actual_Sales']) * 100
        print(f"  {gs:>25s} | {ws:>6.1f}% | {wm:>6.1f}% | {wm-ws:>+4.1f}pp | {n_s:>4d} | {n_m:>4d} | {len(v_s):>4d}")

# Per-lag comparison
print(f"\n  Per-lag WMAPE — Single vs Multi-vintage (temporal, FC sites):")
print(f"  {'Lag':>4s} | {'Single':>8s} | {'Multi':>8s} | {'Diff':>6s} | {'FC1':>4s} | {'FCn':>4s}")
print("  " + "-" * 50)
for lag in sorted(test['Prediction_Lag'].unique()):
    sub = test[m_fc & (test['Prediction_Lag']==lag)]
    n_s = sub['fc_implied_flat'].notna().sum()
    n_m = sub['fc_implied_multi'].notna().sum()
    v_s = sub[sub['fc_single'].notna()]
    v_m = sub[sub['fc_multi'].notna()]
    if len(v_s) > 0 and len(v_m) > 0:
        ws = np.sum(np.abs(v_s['Actual_Sales'] - v_s['fc_single'])) / np.sum(v_s['Actual_Sales']) * 100
        wm = np.sum(np.abs(v_m['Actual_Sales'] - v_m['fc_multi'])) / np.sum(v_m['Actual_Sales']) * 100
        print(f"  {lag:>4d} | {ws:>6.1f}% | {wm:>6.1f}% | {wm-ws:>+4.1f}pp | {n_s:>4d} | {n_m:>4d}")


# ==================================================================
# K-FOLD RANDOM CV
# ==================================================================
N_FOLDS = 5
np.random.seed(42)

print(f"\n\n{'='*100}")
print(f"K-FOLD RANDOM CV ({N_FOLDS} folds) — Customer 2 FC-only (single vs multi-vintage)")
print("=" * 100)

indices = np.arange(len(df))
np.random.shuffle(indices)
fold_size = len(df) // N_FOLDS

fold_metrics_single = []
fold_metrics_multi = []

for fold_i in range(N_FOLDS):
    fold_start = fold_i * fold_size
    fold_end = fold_start + fold_size if fold_i < N_FOLDS - 1 else len(df)
    test_idx = indices[fold_start:fold_end]
    train_idx = np.setdiff1d(indices, test_idx)

    f_train = df.iloc[train_idx].copy()
    f_test = df.iloc[test_idx].copy()

    # Build curves from fold training data
    cov_f_fold = build_flat_curves(f_train, 'fc_coverage', start='2023-01-01')
    bias_f_fold = build_flat_curves(f_train, 'fc_bias', start='2023-01-01')

    # Build registry from TRAINING fold only (no leakage)
    registry_fold = build_forecast_registry(f_train)

    # Apply multi-vintage (produces both flat and multi columns)
    apply_fc_curves_multi(f_train, cov_f_fold, bias_f_fold, registry_fold, causality='fold')
    apply_fc_curves_multi(f_test, cov_f_fold, bias_f_fold, registry_fold, causality='fold')

    f_test_months = sorted(f_test['Reference_Month'].unique())
    f_test['fc_single'] = weighted_avg(f_test, f_train, f_test_months, 'fc_implied_flat')
    f_test['fc_multi'] = weighted_avg(f_test, f_train, f_test_months, 'fc_implied_multi')

    # Metrics (FC sites)
    ts = f_test[f_test['gsa_site'].isin(fc_sites) & (f_test['Actual_Sales'] > 0)]

    for label, col, metrics_list in [('Single', 'fc_single', fold_metrics_single),
                                      ('Multi', 'fc_multi', fold_metrics_multi)]:
        v = ts[ts[col].notna()]
        if len(v) > 0:
            y, p = v['Actual_Sales'].values, v[col].values
            fold_w = np.sum(np.abs(y - p)) / np.sum(y) * 100
            fold_b = np.mean((p - y) / y) * 100
            site_wmapes = {}
            for gs in fc_sites:
                sv = v[v['gsa_site']==gs]
                if len(sv) > 0:
                    site_wmapes[gs] = np.sum(np.abs(sv['Actual_Sales'] - sv[col])) / np.sum(sv['Actual_Sales']) * 100
            metrics_list.append({'wmape': fold_w, 'bias': fold_b, 'n': len(v), 'sites': site_wmapes})

    n_s = ts['fc_implied_flat'].notna().sum()
    n_m = ts['fc_implied_multi'].notna().sum()
    ws = fold_metrics_single[-1]['wmape'] if fold_metrics_single else 0
    wm = fold_metrics_multi[-1]['wmape'] if fold_metrics_multi else 0
    print(f"  Fold {fold_i+1}: Single={ws:.1f}%  Multi={wm:.1f}%  FC1={n_s} FCn={n_m}")

# Summary
# Capture temporal WMAPE for comparison (multi-vintage)
ts_temp = test[test['gsa_site'].isin(fc_sites) & (test['Actual_Sales'] > 0)]
v_temp_s = ts_temp[ts_temp['fc_single'].notna()]
v_temp_m = ts_temp[ts_temp['fc_multi'].notna()]
wmape_temp_s = np.sum(np.abs(v_temp_s['Actual_Sales'] - v_temp_s['fc_single'])) / np.sum(v_temp_s['Actual_Sales']) * 100
wmape_temp_m = np.sum(np.abs(v_temp_m['Actual_Sales'] - v_temp_m['fc_multi'])) / np.sum(v_temp_m['Actual_Sales']) * 100

print(f"\n  {'Model':>20s} | {'Random CV (mean±std)':>25s} | {'Temporal':>10s} | {'Gap':>6s}")
print("  " + "-" * 70)
for label, metrics, t_wmape in [('Single-vintage', fold_metrics_single, wmape_temp_s),
                                 ('Multi-vintage', fold_metrics_multi, wmape_temp_m)]:
    ws = [m['wmape'] for m in metrics]
    bs = [m['bias'] for m in metrics]
    print(f"  {label:>20s} | {np.mean(ws):>5.1f}% ± {np.std(ws):>4.1f}% (bias {np.mean(bs):>+5.1f}%) | {t_wmape:>6.1f}% | {t_wmape - np.mean(ws):>+4.1f}pp")

# Per-site: random vs temporal
print(f"\n  Per-site comparison — Random CV vs Temporal (multi-vintage):")
print(f"  {'Site':>25s} | {'Rand mean':>9s} | {'Rand std':>8s} | {'Temporal':>8s} | {'Gap':>6s}")
print("  " + "-" * 65)
for gs in sorted(fc_sites):
    rands = [m['sites'].get(gs) for m in fold_metrics_multi if gs in m['sites']]
    if len(rands) >= 3:
        sub_t = test[(test['gsa_site']==gs) & (test['Actual_Sales']>0)]
        vt = sub_t[sub_t['fc_multi'].notna()]
        if len(vt) > 0:
            tw = np.sum(np.abs(vt['Actual_Sales'] - vt['fc_multi'])) / np.sum(vt['Actual_Sales']) * 100
            print(f"  {gs:>25s} | {np.mean(rands):>7.1f}% | {np.std(rands):>6.1f}% | {tw:>6.1f}% | {tw - np.mean(rands):>+4.1f}")

print("\nDone!")
