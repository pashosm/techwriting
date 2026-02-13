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
print("TEMPORAL EVALUATION — Customer 2 FC-only")
print("=" * 100)

cov_f = build_flat_curves(train, 'fc_coverage', start='2023-01-01')
bias_f = build_flat_curves(train, 'fc_bias', start='2023-01-01')
print(f"  FC curves built: {len(cov_f)} coverage, {len(bias_f)} bias")

apply_fc_curves(train, cov_f, bias_f)
apply_fc_curves(test, cov_f, bias_f)

# Check how many test rows got FC implied values
n_fc = test['fc_implied_flat'].notna().sum()
print(f"  Test rows with FC implied: {n_fc}/{len(test)} ({100*n_fc/len(test):.0f}%)")

test['fc_only'] = weighted_avg(test, train, test_months, 'fc_implied_flat')

# Overall metrics (FC sites only)
m_fc = test['gsa_site'].isin(fc_sites) & (test['Actual_Sales'] > 0)
ts = test[m_fc]
v = ts[ts['fc_only'].notna()]
wmape = np.sum(np.abs(v['Actual_Sales'] - v['fc_only'])) / np.sum(v['Actual_Sales']) * 100
bias = np.mean((v['fc_only'] - v['Actual_Sales']) / v['Actual_Sales']) * 100
print(f"\n  OVERALL (FC sites): WMAPE={wmape:.1f}%  bias={bias:+.1f}%  N={len(v)}")

# Per-site
print(f"\n  Per-site WMAPE — FC-only (temporal):")
print(f"  {'Site':>25s} | {'WMAPE':>7s} | {'Bias':>7s} | {'N':>5s} | {'FC rows':>7s}")
print("  " + "-" * 65)
for gs in sorted(fc_sites):
    sub = test[(test['gsa_site']==gs) & (test['Actual_Sales']>0)]
    v = sub[sub['fc_only'].notna()]
    n_fc_rows = sub['fc_implied_flat'].notna().sum()
    if len(v) > 0:
        y, p = v['Actual_Sales'].values, v['fc_only'].values
        w = np.sum(np.abs(y - p)) / np.sum(y) * 100
        b = np.mean((p - y) / y) * 100
        print(f"  {gs:>25s} | {w:>5.1f}% | {b:>+5.1f}% | {len(v):>5d} | {n_fc_rows:>7d}")

# Per-lag
print(f"\n  Per-lag WMAPE — FC-only (temporal, FC sites):")
print(f"  {'Lag':>4s} | {'WMAPE':>7s} | {'Bias':>7s} | {'N':>5s}")
print("  " + "-" * 30)
for lag in sorted(test['Prediction_Lag'].unique()):
    sub = test[m_fc & (test['Prediction_Lag']==lag)]
    v = sub[sub['fc_only'].notna()]
    if len(v) > 0:
        y, p = v['Actual_Sales'].values, v['fc_only'].values
        w = np.sum(np.abs(y - p)) / np.sum(y) * 100
        b = np.mean((p - y) / y) * 100
        print(f"  {lag:>4d} | {w:>5.1f}% | {b:>+5.1f}% | {len(v):>5d}")


# ==================================================================
# K-FOLD RANDOM CV
# ==================================================================
N_FOLDS = 5
np.random.seed(42)

print(f"\n\n{'='*100}")
print(f"K-FOLD RANDOM CV ({N_FOLDS} folds) — Customer 2 FC-only")
print("=" * 100)

indices = np.arange(len(df))
np.random.shuffle(indices)
fold_size = len(df) // N_FOLDS

fold_metrics = []

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

    apply_fc_curves(f_train, cov_f_fold, bias_f_fold)
    apply_fc_curves(f_test, cov_f_fold, bias_f_fold)

    # Run FC-only model
    f_test_months = sorted(f_test['Reference_Month'].unique())
    f_test['fc_only'] = weighted_avg(f_test, f_train, f_test_months, 'fc_implied_flat')

    # Metrics (FC sites)
    ts = f_test[f_test['gsa_site'].isin(fc_sites) & (f_test['Actual_Sales'] > 0)]
    v = ts[ts['fc_only'].notna()]
    if len(v) > 0:
        y, p = v['Actual_Sales'].values, v['fc_only'].values
        fold_w = np.sum(np.abs(y - p)) / np.sum(y) * 100
        fold_b = np.mean((p - y) / y) * 100

        # Per-site for this fold
        site_wmapes = {}
        for gs in fc_sites:
            sv = v[v['gsa_site']==gs]
            if len(sv) > 0:
                site_wmapes[gs] = np.sum(np.abs(sv['Actual_Sales'] - sv['fc_only'])) / np.sum(sv['Actual_Sales']) * 100

        fold_metrics.append({'wmape': fold_w, 'bias': fold_b, 'n': len(v), 'sites': site_wmapes})
        print(f"  Fold {fold_i+1}: WMAPE={fold_w:.1f}%  bias={fold_b:+.1f}%  N={len(v)}")

# Summary
print(f"\n  {'Model':>20s} | {'Random CV (mean±std)':>25s} | {'Temporal':>10s} | {'Gap':>6s}")
print("  " + "-" * 70)
wmapes = [m['wmape'] for m in fold_metrics]
biases = [m['bias'] for m in fold_metrics]
print(f"  {'FC-only':>20s} | {np.mean(wmapes):>5.1f}% ± {np.std(wmapes):>4.1f}% (bias {np.mean(biases):>+5.1f}%) | {wmape:>6.1f}% | {wmape - np.mean(wmapes):>+4.1f}pp")

# Per-site: random vs temporal
print(f"\n  Per-site comparison — Random CV vs Temporal:")
print(f"  {'Site':>25s} | {'Rand mean':>9s} | {'Rand std':>8s} | {'Temporal':>8s} | {'Gap':>6s}")
print("  " + "-" * 65)
for gs in sorted(fc_sites):
    rands = [m['sites'].get(gs) for m in fold_metrics if gs in m['sites']]
    if len(rands) >= 3:
        # Temporal
        sub_t = test[(test['gsa_site']==gs) & (test['Actual_Sales']>0)]
        vt = sub_t[sub_t['fc_only'].notna()]
        if len(vt) > 0:
            tw = np.sum(np.abs(vt['Actual_Sales'] - vt['fc_only'])) / np.sum(vt['Actual_Sales']) * 100
            print(f"  {gs:>25s} | {np.mean(rands):>7.1f}% | {np.std(rands):>6.1f}% | {tw:>6.1f}% | {tw - np.mean(rands):>+4.1f}")

# Per-lag: random (fold 1) vs temporal
print(f"\n  Per-lag comparison — Random CV (fold 1) vs Temporal:")
# Recompute fold 1 for per-lag detail
t_idx = indices[:fold_size]
tr_idx = np.setdiff1d(indices, t_idx)
f1_train = df.iloc[tr_idx].copy()
f1_test = df.iloc[t_idx].copy()
cov_f1 = build_flat_curves(f1_train, 'fc_coverage', start='2023-01-01')
bias_f1 = build_flat_curves(f1_train, 'fc_bias', start='2023-01-01')
apply_fc_curves(f1_train, cov_f1, bias_f1)
apply_fc_curves(f1_test, cov_f1, bias_f1)
f1_months = sorted(f1_test['Reference_Month'].unique())
f1_test['fc_only'] = weighted_avg(f1_test, f1_train, f1_months, 'fc_implied_flat')

print(f"  {'Lag':>4s} | {'Rand FC':>9s} | {'Temp FC':>9s} | {'Gap':>6s}")
print("  " + "-" * 35)
for lag in sorted(df['Prediction_Lag'].unique()):
    sub_r = f1_test[(f1_test['Prediction_Lag']==lag) & (f1_test['Actual_Sales']>0) & f1_test['gsa_site'].isin(fc_sites)]
    sub_t = test[m_fc & (test['Prediction_Lag']==lag)]
    vr = sub_r[sub_r['fc_only'].notna()]
    vt = sub_t[sub_t['fc_only'].notna()]
    if len(vr) > 0 and len(vt) > 0:
        rw = np.sum(np.abs(vr['Actual_Sales'] - vr['fc_only'])) / np.sum(vr['Actual_Sales']) * 100
        tw = np.sum(np.abs(vt['Actual_Sales'] - vt['fc_only'])) / np.sum(vt['Actual_Sales']) * 100
        print(f"  {lag:>4d} | {rw:>7.1f}% | {tw:>7.1f}% | {tw - rw:>+4.1f}")

print("\nDone!")
