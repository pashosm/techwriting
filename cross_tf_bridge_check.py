#!/usr/bin/env python3
"""
Cross-Timeframe Bridge Feasibility Check
==========================================
Two questions:
1. Does a completed TF=3 curve predict the TF=12 curve shape?
   (i.e., is there a stable TF=3 → TF=12 relationship?)
2. Does in-flight TF=3 OO data add predictive power beyond the
   last completed TF=3?

If both hold, we can build: adjusted_TF12 = f(stale_TF12, recent_TF3, inflight_OO)
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 160)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')

# ── Load ───────────────────────────────────────────────────────────
CSV = "/home/user/techwriting/training_data_anonymized.csv"
df = pd.read_csv(CSV)
df['Reference_Month'] = pd.to_datetime(df['Reference_Month'])

# Compute fc_coverage for all rows (including zero-forecast as 0 coverage)
df['fc_coverage'] = np.where(
    df['Actual_Sales'] > 0,
    df['Covered_Orders'] / df['Actual_Sales'], np.nan)

# OO-based coverage (observable at forecast time)
# OO_Coverage = Covered_Open_Orders / Open_Orders (already in data)
# Also compute OO/Sales_Lag1 as a proxy
df['oo_sl1_ratio'] = np.where(
    df['Historical_Sales_Lag1'] > 0,
    df['Open_Orders'] / df['Historical_Sales_Lag1'], np.nan)
df['coo_sl1_ratio'] = np.where(
    df['Historical_Sales_Lag1'] > 0,
    df['Covered_Open_Orders'] / df['Historical_Sales_Lag1'], np.nan)

lags = sorted(df['Prediction_Lag'].unique())
sites = sorted(df['Site'].unique())
ref_months = sorted(df['Reference_Month'].unique())

print("=" * 90)
print("CROSS-TIMEFRAME BRIDGE FEASIBILITY CHECK")
print("=" * 90)

# ══════════════════════════════════════════════════════════════════
# CHECK 1: TF=3 → TF=12 CURVE RELATIONSHIP
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("CHECK 1: Does TF=3 fc_coverage curve predict TF=12 fc_coverage curve?")
print("=" * 90)

# For each site+ref_month, build the coverage curve (mean fc_coverage at each lag)
# for both TF=3 and TF=12. Then check correlation.

# Approach: at each lag position, correlate TF=3 coverage with TF=12 coverage
# across all site+ref_month combos.

tf3 = df[df['Timeframe'] == 3].copy()
tf12 = df[df['Timeframe'] == 12].copy()

# Build pivot: rows = (site, ref_month), columns = lag, values = fc_coverage
tf3_pivot = tf3.pivot_table(
    index=['Site', 'Reference_Month'],
    columns='Prediction_Lag',
    values='fc_coverage',
    aggfunc='mean'
)
tf12_pivot = tf12.pivot_table(
    index=['Site', 'Reference_Month'],
    columns='Prediction_Lag',
    values='fc_coverage',
    aggfunc='mean'
)

# Find shared (site, ref_month) pairs
shared_idx = tf3_pivot.index.intersection(tf12_pivot.index)
print(f"\nShared (site, ref_month) pairs: {len(shared_idx)}")

tf3_shared = tf3_pivot.loc[shared_idx]
tf12_shared = tf12_pivot.loc[shared_idx]

# Method 1: Lag-by-lag correlation
# At each lag, does TF=3 coverage correlate with TF=12 coverage?
print("\n--- Lag-by-lag: TF=3 coverage vs TF=12 coverage ---")
print(f"{'Lag':>5} {'Pearson r':>12} {'Spearman r':>12} {'N':>8} {'Signal'}")
print("-" * 50)
for lag in lags:
    if lag in tf3_shared.columns and lag in tf12_shared.columns:
        a = tf3_shared[lag].values
        b = tf12_shared[lag].values
        mask = ~np.isnan(a) & ~np.isnan(b)
        n = mask.sum()
        if n >= 20:
            pr, _ = stats.pearsonr(a[mask], b[mask])
            sr, sp = stats.spearmanr(a[mask], b[mask])
            sig = "STRONG" if sr > 0.6 else "moderate" if sr > 0.3 else "weak"
            print(f"{lag:>5} {pr:>12.4f} {sr:>12.4f} {n:>8} {sig}")
        else:
            print(f"{lag:>5} {'insuff.':>12} {'':>12} {n:>8}")

# Method 2: Curve shape correlation
# For each site+ref_month, compute correlation between the TF=3 curve
# and TF=12 curve across lags.
print("\n--- Curve shape: correlation between TF=3 and TF=12 curves ---")
print("For each (site, ref_month), Pearson r across all 12 lags")

shape_corrs = []
for idx in shared_idx:
    vec3 = tf3_shared.loc[idx].values
    vec12 = tf12_shared.loc[idx].values
    mask = ~np.isnan(vec3) & ~np.isnan(vec12)
    if mask.sum() >= 5 and vec3[mask].std() > 0 and vec12[mask].std() > 0:
        r, _ = stats.pearsonr(vec3[mask], vec12[mask])
        shape_corrs.append(r)

shape_corrs = np.array(shape_corrs)
print(f"N site-months with valid shape correlation: {len(shape_corrs)}")
print(f"  mean r = {shape_corrs.mean():.4f}")
print(f"  median r = {np.median(shape_corrs):.4f}")
print(f"  std = {shape_corrs.std():.4f}")
print(f"  % with r > 0.8: {100 * (shape_corrs > 0.8).mean():.1f}%")
print(f"  % with r > 0.5: {100 * (shape_corrs > 0.5).mean():.1f}%")
print(f"  % with r > 0.0: {100 * (shape_corrs > 0.0).mean():.1f}%")

# Method 3: Level relationship (not just shape)
# Does the MEAN TF=3 coverage predict the MEAN TF=12 coverage?
print("\n--- Level: mean TF=3 coverage vs mean TF=12 coverage ---")
tf3_mean = tf3_shared.mean(axis=1)
tf12_mean = tf12_shared.mean(axis=1)
mask = ~np.isnan(tf3_mean) & ~np.isnan(tf12_mean)
n = mask.sum()
if n >= 20:
    pr, _ = stats.pearsonr(tf3_mean[mask], tf12_mean[mask])
    sr, sp = stats.spearmanr(tf3_mean[mask], tf12_mean[mask])
    print(f"  Pearson r = {pr:.4f}, Spearman r = {sr:.4f}, N = {n}")

# Method 4: Ratio stability — TF12/TF3 at each lag
print("\n--- Ratio TF12/TF3 at each lag (stability of the multiplier) ---")
print(f"{'Lag':>5} {'mean ratio':>12} {'median ratio':>14} {'std':>8} {'CV':>8}")
print("-" * 55)
for lag in lags:
    if lag in tf3_shared.columns and lag in tf12_shared.columns:
        a = tf3_shared[lag].values
        b = tf12_shared[lag].values
        mask = ~np.isnan(a) & ~np.isnan(b) & (a > 0.01)  # avoid div by tiny
        if mask.sum() >= 20:
            ratio = b[mask] / a[mask]
            # Clip extreme ratios for summary
            ratio_clipped = ratio[(ratio > 0.01) & (ratio < 100)]
            if len(ratio_clipped) >= 10:
                print(f"{lag:>5} {ratio_clipped.mean():>12.4f} {np.median(ratio_clipped):>14.4f} "
                      f"{ratio_clipped.std():>8.4f} {ratio_clipped.std()/ratio_clipped.mean():>8.4f}")


# ══════════════════════════════════════════════════════════════════
# CHECK 1b: TEMPORAL STABILITY — does the TF3→TF12 relationship drift?
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("CHECK 1b: Is the TF=3 → TF=12 relationship stable over time?")
print("=" * 90)

# Split by year and check if the correlation holds in each period
print("\n--- Lag-1 correlation by year ---")
print(f"{'Year':>6} {'Pearson r':>12} {'Spearman r':>12} {'N':>8}")
print("-" * 45)

for idx in shared_idx:
    pass  # just to have the loop

tf3_lag1 = tf3[tf3['Prediction_Lag'] == 1].groupby(['Site', 'Reference_Month'])['fc_coverage'].mean()
tf12_lag1 = tf12[tf12['Prediction_Lag'] == 1].groupby(['Site', 'Reference_Month'])['fc_coverage'].mean()
shared_l1 = tf3_lag1.index.intersection(tf12_lag1.index)

for year in [2022, 2023, 2024, 2025]:
    mask_year = [idx for idx in shared_l1 if idx[1].year == year]
    if len(mask_year) >= 10:
        a = tf3_lag1.reindex(mask_year).values
        b = tf12_lag1.reindex(mask_year).values
        valid = ~np.isnan(a) & ~np.isnan(b)
        if valid.sum() >= 10:
            pr, _ = stats.pearsonr(a[valid], b[valid])
            sr, _ = stats.spearmanr(a[valid], b[valid])
            print(f"{year:>6} {pr:>12.4f} {sr:>12.4f} {valid.sum():>8}")


# ══════════════════════════════════════════════════════════════════
# CHECK 2: IN-FLIGHT OO DATA PREDICTIVENESS
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("CHECK 2: Does in-flight TF=3 OO data predict TF=12 coverage?")
print("=" * 90)

# The idea: at a given ref_month, we have:
# - Completed TF=3: the most recent ref_month where TF=3 target period is done
#   (for ref_month M, that's roughly M-3 or M-4)
# - In-flight TF=3 at ref_month M: OO data for the current TF=3 window
# - TF=12 coverage at ref_month M: what we're trying to predict
#
# We want to see if in-flight OO adds value beyond completed TF=3.

# For each (site, ref_month), gather:
# 1. TF=12 fc_coverage curve (ground truth to predict)
# 2. TF=3 fc_coverage at the SAME ref_month (concurrent, partially overlapping)
# 3. OO data at the same ref_month for TF=3

# Build OO-based features from TF=3
tf3_oo = tf3.pivot_table(
    index=['Site', 'Reference_Month'],
    columns='Prediction_Lag',
    values=['OO_Coverage', 'OO_Bias', 'oo_sl1_ratio', 'coo_sl1_ratio'],
    aggfunc='mean'
)

# For simplicity, take the Lag=1 values (most observable/fresh)
print("\n--- At Lag=1: Can TF=3 OO metrics predict TF=12 fc_coverage? ---")

# Build a combined frame
tf3_lag1_df = tf3[tf3['Prediction_Lag'] == 1][
    ['Site', 'Reference_Month', 'fc_coverage', 'OO_Coverage', 'OO_Bias',
     'oo_sl1_ratio', 'coo_sl1_ratio', 'Open_Orders', 'Covered_Open_Orders',
     'Forecast_Value', 'Covered_Orders']
].set_index(['Site', 'Reference_Month'])
tf3_lag1_df.columns = ['tf3_cov_' + c if c == 'fc_coverage' else 'tf3_' + c for c in
                        ['fc_coverage', 'OO_Coverage', 'OO_Bias',
                         'oo_sl1_ratio', 'coo_sl1_ratio', 'Open_Orders',
                         'Covered_Open_Orders', 'Forecast_Value', 'Covered_Orders']]

tf12_lag1_df = tf12[tf12['Prediction_Lag'] == 1][
    ['Site', 'Reference_Month', 'fc_coverage']
].set_index(['Site', 'Reference_Month'])
tf12_lag1_df.columns = ['tf12_fc_coverage']

combined = tf3_lag1_df.join(tf12_lag1_df, how='inner')
print(f"Combined rows (site+month with both TF=3 and TF=12 at Lag=1): {len(combined)}")

# Correlation of each TF=3 feature with TF=12 fc_coverage
print(f"\n{'TF=3 Feature':<30} {'Pearson r':>12} {'Spearman r':>12} {'N':>8} {'Signal'}")
print("-" * 70)

target = combined['tf12_fc_coverage']
for col in sorted(combined.columns):
    if col == 'tf12_fc_coverage':
        continue
    vals = combined[col]
    mask = ~np.isnan(vals) & ~np.isnan(target)
    n = mask.sum()
    if n >= 20:
        pr, _ = stats.pearsonr(vals[mask], target[mask])
        sr, sp = stats.spearmanr(vals[mask], target[mask])
        sig = "STRONG" if abs(sr) > 0.5 else "moderate" if abs(sr) > 0.3 else "weak"
        print(f"{col:<30} {pr:>12.4f} {sr:>12.4f} {n:>8} {sig}")
    else:
        print(f"{col:<30} {'insuff.':>12} {'':>12} {n:>8}")

# ── Key test: does OO add value BEYOND completed TF=3 coverage? ───
print("\n" + "=" * 90)
print("CHECK 2b: Incremental value of OO beyond TF=3 completed coverage")
print("=" * 90)
print("Partial correlation: TF=3 OO metrics vs TF=12 coverage,")
print("controlling for TF=3 completed coverage.\n")

# Simple approach: compute residuals of TF=12 coverage after regressing on TF=3 coverage,
# then check if OO metrics correlate with those residuals.

mask_base = ~np.isnan(combined['tf3_cov_fc_coverage']) & ~np.isnan(target)
if mask_base.sum() >= 50:
    from numpy.polynomial.polynomial import polyfit, polyval

    x_base = combined.loc[mask_base, 'tf3_cov_fc_coverage'].values
    y = target[mask_base].values

    # Fit simple linear model: TF12 ~ TF3_coverage
    slope, intercept, r_base, p_base, _ = stats.linregress(x_base, y)
    y_pred = intercept + slope * x_base
    residuals = y - y_pred

    print(f"Baseline: TF=12 ~ TF=3 completed coverage")
    print(f"  R² = {r_base**2:.4f}, r = {r_base:.4f}, N = {mask_base.sum()}")

    # Now check if OO features correlate with the residuals
    print(f"\n{'OO Feature':<30} {'r with residual':>16} {'p-value':>12} {'N':>8} {'Adds value?'}")
    print("-" * 75)

    residual_series = pd.Series(residuals, index=combined.index[mask_base])

    for col in ['tf3_OO_Coverage', 'tf3_OO_Bias', 'tf3_oo_sl1_ratio',
                'tf3_coo_sl1_ratio', 'tf3_Open_Orders', 'tf3_Covered_Open_Orders']:
        vals = combined.loc[mask_base, col]
        m = ~np.isnan(vals)
        n = m.sum()
        if n >= 20:
            sr, sp = stats.spearmanr(vals[m], residual_series[m])
            adds = "YES" if (abs(sr) > 0.15 and sp < 0.05) else "marginal" if abs(sr) > 0.1 else "no"
            print(f"{col:<30} {sr:>16.4f} {sp:>12.6f} {n:>8} {adds}")
        else:
            print(f"{col:<30} {'insuff.':>16} {'':>12} {n:>8}")

# ── Check across multiple lags ────────────────────────────────────
print("\n" + "=" * 90)
print("CHECK 2c: TF=3 OO_Coverage vs TF=12 coverage across all lags")
print("=" * 90)

print(f"\n{'Lag':>5} {'TF3_cov→TF12 r':>16} {'TF3_OO_cov→TF12 r':>20} {'TF3_oo_sl1→TF12 r':>20} {'N':>8}")
print("-" * 75)

for lag in lags:
    tf3_l = tf3[tf3['Prediction_Lag'] == lag].groupby(['Site', 'Reference_Month']).agg(
        fc_coverage=('fc_coverage', 'mean'),
        OO_Coverage=('OO_Coverage', 'mean'),
        oo_sl1_ratio=('oo_sl1_ratio', 'mean'),
    )
    tf12_l = tf12[tf12['Prediction_Lag'] == lag].groupby(['Site', 'Reference_Month'])['fc_coverage'].mean()

    shared = tf3_l.index.intersection(tf12_l.index)
    if len(shared) < 20:
        continue

    t12_cov = tf12_l.reindex(shared).values
    t3_cov = tf3_l.loc[shared, 'fc_coverage'].values
    t3_oo_cov = tf3_l.loc[shared, 'OO_Coverage'].values
    t3_oo_sl1 = tf3_l.loc[shared, 'oo_sl1_ratio'].values

    results = []
    for label, feature in [('TF3_cov', t3_cov), ('TF3_OO_cov', t3_oo_cov), ('TF3_oo_sl1', t3_oo_sl1)]:
        mask = ~np.isnan(feature) & ~np.isnan(t12_cov)
        n = mask.sum()
        if n >= 20:
            sr, _ = stats.spearmanr(feature[mask], t12_cov[mask])
            results.append(f"{sr:>16.4f}")
        else:
            results.append(f"{'insuff.':>16}")
    n_total = (~np.isnan(t3_cov) & ~np.isnan(t12_cov)).sum()
    print(f"{lag:>5} {results[0]:>16} {results[1]:>20} {results[2]:>20} {n_total:>8}")


# ══════════════════════════════════════════════════════════════════
# CHECK 3: DRIFT DETECTION — does TF=3 change predict TF=12 change?
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("CHECK 3: Does CHANGE in TF=3 predict CHANGE in TF=12?")
print("=" * 90)
print("If TF=3 coverage shifted from period t-1 to t, does TF=12 shift similarly?")

# For each site, compute period-over-period change in mean coverage
# at Lag=1 for both TF=3 and TF=12
tf3_lag1_ts = tf3[tf3['Prediction_Lag'] == 1].pivot_table(
    index='Reference_Month', columns='Site', values='fc_coverage', aggfunc='mean'
)
tf12_lag1_ts = tf12[tf12['Prediction_Lag'] == 1].pivot_table(
    index='Reference_Month', columns='Site', values='fc_coverage', aggfunc='mean'
)

# Compute period-over-period changes
tf3_diff = tf3_lag1_ts.diff()
tf12_diff = tf12_lag1_ts.diff()

# Align and correlate
shared_months = tf3_diff.index.intersection(tf12_diff.index)
shared_sites = tf3_diff.columns.intersection(tf12_diff.columns)

all_tf3_changes = []
all_tf12_changes = []
for site in shared_sites:
    for month in shared_months:
        v3 = tf3_diff.loc[month, site]
        v12 = tf12_diff.loc[month, site]
        if not np.isnan(v3) and not np.isnan(v12):
            all_tf3_changes.append(v3)
            all_tf12_changes.append(v12)

all_tf3_changes = np.array(all_tf3_changes)
all_tf12_changes = np.array(all_tf12_changes)
print(f"\nSite-month change pairs: {len(all_tf3_changes)}")

if len(all_tf3_changes) >= 50:
    pr, pp = stats.pearsonr(all_tf3_changes, all_tf12_changes)
    sr, sp = stats.spearmanr(all_tf3_changes, all_tf12_changes)
    print(f"Correlation of ΔTF3 with ΔTF12:")
    print(f"  Pearson r = {pr:.4f} (p={pp:.6f})")
    print(f"  Spearman r = {sr:.4f} (p={sp:.6f})")
    print(f"  {'STRONG' if sr > 0.3 else 'moderate' if sr > 0.15 else 'weak'} signal")

    # Also check with OO data
    tf3_oo_ts = tf3[tf3['Prediction_Lag'] == 1].pivot_table(
        index='Reference_Month', columns='Site', values='OO_Coverage', aggfunc='mean'
    )
    tf3_oo_diff = tf3_oo_ts.diff()

    oo_changes = []
    tf12_changes_oo = []
    for site in shared_sites:
        if site not in tf3_oo_diff.columns:
            continue
        for month in shared_months:
            if month not in tf3_oo_diff.index:
                continue
            v_oo = tf3_oo_diff.loc[month, site] if month in tf3_oo_diff.index else np.nan
            v12 = tf12_diff.loc[month, site]
            if not np.isnan(v_oo) and not np.isnan(v12):
                oo_changes.append(v_oo)
                tf12_changes_oo.append(v12)

    if len(oo_changes) >= 20:
        sr_oo, sp_oo = stats.spearmanr(oo_changes, tf12_changes_oo)
        print(f"\nCorrelation of ΔTF3_OO_Coverage with ΔTF12:")
        print(f"  Spearman r = {sr_oo:.4f} (p={sp_oo:.6f}), N={len(oo_changes)}")


# ══════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)
print("""
CHECK 1: TF=3 → TF=12 curve relationship
  - Lag-by-lag correlation: how well does TF=3 coverage at each lag
    predict TF=12 coverage at the same lag?
  - Curve shape: do TF=3 and TF=12 curves have similar shapes?
  - Ratio stability: is TF12/TF3 a stable multiplier?

CHECK 2: In-flight OO value
  - Direct correlation: do TF=3 OO metrics predict TF=12 coverage?
  - Incremental value: do OO metrics add info beyond completed TF=3?

CHECK 3: Drift detection
  - Does a CHANGE in TF=3 coverage predict a CHANGE in TF=12 coverage?
  - If yes, we can use recent TF=3 shifts to adjust stale TF=12 curves.
""")
print("=" * 90)
print("ANALYSIS COMPLETE")
print("=" * 90)
