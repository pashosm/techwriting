#!/usr/bin/env python3
"""
Cross-Timeframe Bridge Feasibility Check (v2 — realistic temporal offset)
==========================================================================
Previous version tested TF=3 vs TF=12 at the SAME Reference_Month.
That's not realistic: when making a Dec 2024 TF=12 forecast, the most
recent COMPLETED TF=3 is from ~Aug 2024 (4 months prior).

This version tests:
  PART A: Same-month baseline (for reference)
  PART B: Realistic offset — TF=3 at M-4 vs TF=12 at M
  PART C: In-flight OO from the CURRENT month's TF=3 (not yet completed)
  PART D: Drift detection with realistic offset

If the signal survives the 4-month gap, the bridge approach is viable.
"""

import pandas as pd
import numpy as np
from scipy import stats
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 160)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')

# ── Load ───────────────────────────────────────────────────────────
CSV = "/home/user/techwriting/training_data_anonymized.csv"
df = pd.read_csv(CSV)
df['Reference_Month'] = pd.to_datetime(df['Reference_Month'])

df['fc_coverage'] = np.where(
    df['Actual_Sales'] > 0,
    df['Covered_Orders'] / df['Actual_Sales'], np.nan)
df['oo_sl1_ratio'] = np.where(
    df['Historical_Sales_Lag1'] > 0,
    df['Open_Orders'] / df['Historical_Sales_Lag1'], np.nan)
df['coo_sl1_ratio'] = np.where(
    df['Historical_Sales_Lag1'] > 0,
    df['Covered_Open_Orders'] / df['Historical_Sales_Lag1'], np.nan)

lags = sorted(df['Prediction_Lag'].unique())
tf3 = df[df['Timeframe'] == 3].copy()
tf12 = df[df['Timeframe'] == 12].copy()

print("=" * 90)
print("CROSS-TIMEFRAME BRIDGE FEASIBILITY CHECK (v2 — realistic offset)")
print("=" * 90)

# ══════════════════════════════════════════════════════════════════
# HELPER: build lag-by-lag correlation table
# ══════════════════════════════════════════════════════════════════
def lag_by_lag_correlation(tf3_data, tf12_data, shared_keys):
    """Correlate TF=3 and TF=12 fc_coverage at each lag for shared keys."""
    results = []
    for lag in lags:
        t3 = tf3_data[tf3_data['Prediction_Lag'] == lag].groupby(
            ['Site', 'Reference_Month'])['fc_coverage'].mean()
        t12 = tf12_data[tf12_data['Prediction_Lag'] == lag].groupby(
            ['Site', 'Reference_Month'])['fc_coverage'].mean()

        # Align to shared keys
        common = t3.index.intersection(t12.index).intersection(shared_keys)
        if len(common) < 20:
            results.append((lag, np.nan, np.nan, 0))
            continue

        a = t3.reindex(common).values
        b = t12.reindex(common).values
        mask = ~np.isnan(a) & ~np.isnan(b)
        n = mask.sum()
        if n >= 20:
            pr, _ = stats.pearsonr(a[mask], b[mask])
            sr, _ = stats.spearmanr(a[mask], b[mask])
            results.append((lag, pr, sr, n))
        else:
            results.append((lag, np.nan, np.nan, n))
    return results


def print_lag_table(results, label=""):
    print(f"\n--- Lag-by-lag: {label} ---")
    print(f"{'Lag':>5} {'Pearson r':>12} {'Spearman r':>12} {'N':>8} {'Signal'}")
    print("-" * 50)
    for lag, pr, sr, n in results:
        if np.isnan(sr):
            print(f"{lag:>5} {'insuff.':>12} {'':>12} {n:>8}")
        else:
            sig = "STRONG" if sr > 0.6 else "moderate" if sr > 0.3 else "weak"
            print(f"{lag:>5} {pr:>12.4f} {sr:>12.4f} {n:>8} {sig}")


# ══════════════════════════════════════════════════════════════════
# PART A: SAME-MONTH BASELINE (reproduce previous results briefly)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("PART A: SAME-MONTH BASELINE (TF=3 and TF=12 from same Reference_Month)")
print("=" * 90)

# Build shared keys: (site, ref_month) present in both TF=3 and TF=12
tf3_keys = set(zip(tf3['Site'], tf3['Reference_Month']))
tf12_keys = set(zip(tf12['Site'], tf12['Reference_Month']))
same_month_keys = pd.MultiIndex.from_tuples(
    sorted(tf3_keys & tf12_keys), names=['Site', 'Reference_Month'])

print(f"Same-month pairs: {len(same_month_keys)}")

results_same = lag_by_lag_correlation(tf3, tf12, same_month_keys)
print_lag_table(results_same, "Same-month TF=3 vs TF=12")

# ══════════════════════════════════════════════════════════════════
# PART B: REALISTIC OFFSET — TF=3 at M-4 vs TF=12 at M
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("PART B: REALISTIC OFFSET (TF=3 from M-4 vs TF=12 at M)")
print("=" * 90)
print("For each TF=12 at Reference_Month M, pair with TF=3 from M-4")
print("(the most recent completed TF=3 window)\n")

# For TF=3 at Reference_Month M-4, Lag=1: target is M-3 to M-1 (3 months).
# By month M, that target period is fully complete. So M-4 is correct.
# Actually let's be precise: TF=3 at ref M-4, Lag=1 covers months M-3, M-2, M-1.
# By month M this is complete. For Lag=3, target is M-1 to M+1 — NOT complete.
# So only Lag=1 and Lag=2 from M-4 are certainly complete by M.
# But for the CURVE, we're looking at the historical pattern, not predicting from it.
# The fc_coverage at M-4 uses Actual_Sales which are known after the target period.
# So the full TF=3 curve from M-4 IS available by month M.

# Build offset pairs: for each (site, M) in TF=12, find (site, M-4) in TF=3
OFFSET_MONTHS = 4

# Create a shifted TF=3 dataset where Reference_Month is shifted forward by OFFSET
tf3_shifted = tf3.copy()
tf3_shifted['Reference_Month_Original'] = tf3_shifted['Reference_Month']
tf3_shifted['Reference_Month'] = tf3_shifted['Reference_Month'] + pd.DateOffset(months=OFFSET_MONTHS)

# Now keys align: tf3_shifted at "M" actually contains data from M-4
tf3_shifted_keys = set(zip(tf3_shifted['Site'], tf3_shifted['Reference_Month']))
offset_keys = pd.MultiIndex.from_tuples(
    sorted(tf3_shifted_keys & tf12_keys), names=['Site', 'Reference_Month'])

print(f"Offset pairs (TF=3 from M-4, TF=12 at M): {len(offset_keys)}")

results_offset = lag_by_lag_correlation(tf3_shifted, tf12, offset_keys)
print_lag_table(results_offset, "OFFSET: TF=3(M-4) vs TF=12(M)")

# ── Side-by-side comparison ───────────────────────────────────────
print("\n--- COMPARISON: Same-month vs Offset (Spearman r) ---")
print(f"{'Lag':>5} {'Same-month':>12} {'Offset(M-4)':>14} {'Degradation':>14}")
print("-" * 50)
for (lag, _, sr_same, _), (_, _, sr_off, _) in zip(results_same, results_offset):
    if not np.isnan(sr_same) and not np.isnan(sr_off):
        deg = sr_same - sr_off
        print(f"{lag:>5} {sr_same:>12.4f} {sr_off:>14.4f} {deg:>14.4f}")

# ── Level correlation with offset ─────────────────────────────────
print("\n--- Level: mean TF=3(M-4) coverage vs mean TF=12(M) coverage ---")
tf3_shifted_pivot = tf3_shifted.pivot_table(
    index=['Site', 'Reference_Month'], columns='Prediction_Lag',
    values='fc_coverage', aggfunc='mean')
tf12_pivot = tf12.pivot_table(
    index=['Site', 'Reference_Month'], columns='Prediction_Lag',
    values='fc_coverage', aggfunc='mean')

shared_offset = tf3_shifted_pivot.index.intersection(tf12_pivot.index)
tf3_off_shared = tf3_shifted_pivot.loc[shared_offset]
tf12_off_shared = tf12_pivot.loc[shared_offset]

tf3_off_mean = tf3_off_shared.mean(axis=1)
tf12_off_mean = tf12_off_shared.mean(axis=1)
mask = ~np.isnan(tf3_off_mean) & ~np.isnan(tf12_off_mean)
if mask.sum() >= 20:
    pr, _ = stats.pearsonr(tf3_off_mean[mask], tf12_off_mean[mask])
    sr, _ = stats.spearmanr(tf3_off_mean[mask], tf12_off_mean[mask])
    print(f"  Pearson r = {pr:.4f}, Spearman r = {sr:.4f}, N = {mask.sum()}")

# ── Temporal stability of offset relationship ─────────────────────
print("\n--- Offset relationship stability by year ---")
print(f"{'Year':>6} {'Pearson r':>12} {'Spearman r':>12} {'N':>8}")
print("-" * 45)

tf3_off_lag1 = tf3_shifted[tf3_shifted['Prediction_Lag'] == 1].groupby(
    ['Site', 'Reference_Month'])['fc_coverage'].mean()
tf12_lag1 = tf12[tf12['Prediction_Lag'] == 1].groupby(
    ['Site', 'Reference_Month'])['fc_coverage'].mean()
shared_off_l1 = tf3_off_lag1.index.intersection(tf12_lag1.index)

for year in [2022, 2023, 2024, 2025]:
    year_keys = [idx for idx in shared_off_l1 if idx[1].year == year]
    if len(year_keys) >= 10:
        a = tf3_off_lag1.reindex(year_keys).values
        b = tf12_lag1.reindex(year_keys).values
        valid = ~np.isnan(a) & ~np.isnan(b)
        if valid.sum() >= 10:
            pr, _ = stats.pearsonr(a[valid], b[valid])
            sr, _ = stats.spearmanr(a[valid], b[valid])
            print(f"{year:>6} {pr:>12.4f} {sr:>12.4f} {valid.sum():>8}")

# ── Also test other offsets to see the degradation curve ──────────
print("\n--- How does correlation degrade with offset? (Lag=1) ---")
print(f"{'Offset':>8} {'Spearman r':>12} {'N':>8}")
print("-" * 35)

for offset in [0, 1, 2, 3, 4, 5, 6, 8, 10, 12]:
    tf3_tmp = tf3[tf3['Prediction_Lag'] == 1].copy()
    tf3_tmp['Reference_Month'] = tf3_tmp['Reference_Month'] + pd.DateOffset(months=offset)
    tf3_tmp_grp = tf3_tmp.groupby(['Site', 'Reference_Month'])['fc_coverage'].mean()

    tf12_tmp = tf12[tf12['Prediction_Lag'] == 1].groupby(
        ['Site', 'Reference_Month'])['fc_coverage'].mean()

    common = tf3_tmp_grp.index.intersection(tf12_tmp.index)
    if len(common) < 20:
        continue
    a = tf3_tmp_grp.reindex(common).values
    b = tf12_tmp.reindex(common).values
    valid = ~np.isnan(a) & ~np.isnan(b)
    if valid.sum() >= 20:
        sr, _ = stats.spearmanr(a[valid], b[valid])
        print(f"{offset:>8} {sr:>12.4f} {valid.sum():>8}")


# ══════════════════════════════════════════════════════════════════
# PART C: IN-FLIGHT OO FROM CURRENT MONTH
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("PART C: IN-FLIGHT OO DATA (same month as TF=12 forecast)")
print("=" * 90)
print("At month M, the TF=3 window is in-flight (target not yet complete).")
print("But OO data IS observable. Does it help predict TF=12 coverage?")
print("And does it add value BEYOND the completed TF=3 from M-4?\n")

# Build combined frame: for each (site, M):
# - TF=12 fc_coverage at M (ground truth)
# - TF=3 fc_coverage from M-4 (completed, offset)
# - TF=3 OO data from M (in-flight, same month)

# TF=3 in-flight OO at Lag=1 (same month as TF=12)
tf3_inflight = tf3[tf3['Prediction_Lag'] == 1][
    ['Site', 'Reference_Month', 'OO_Coverage', 'OO_Bias',
     'Open_Orders', 'Covered_Open_Orders', 'oo_sl1_ratio', 'coo_sl1_ratio']
].groupby(['Site', 'Reference_Month']).mean()
tf3_inflight.columns = ['inflight_' + c for c in tf3_inflight.columns]

# TF=3 completed coverage from M-4
tf3_completed = tf3_shifted[tf3_shifted['Prediction_Lag'] == 1][
    ['Site', 'Reference_Month', 'fc_coverage']
].groupby(['Site', 'Reference_Month']).mean()
tf3_completed.columns = ['completed_tf3_coverage']

# TF=12 target at M
tf12_target = tf12[tf12['Prediction_Lag'] == 1][
    ['Site', 'Reference_Month', 'fc_coverage']
].groupby(['Site', 'Reference_Month']).mean()
tf12_target.columns = ['tf12_fc_coverage']

# Join all three
combined = tf12_target.join(tf3_completed, how='left').join(tf3_inflight, how='left')
print(f"Combined rows: {len(combined)}")
print(f"  With completed TF=3 (M-4): {combined['completed_tf3_coverage'].notna().sum()}")
print(f"  With in-flight OO_Coverage: {combined['inflight_OO_Coverage'].notna().sum()}")

# Direct correlations
print(f"\n{'Feature':<35} {'Spearman r':>12} {'N':>8} {'Signal'}")
print("-" * 65)

target = combined['tf12_fc_coverage']
for col in sorted(combined.columns):
    if col == 'tf12_fc_coverage':
        continue
    vals = combined[col]
    mask = ~np.isnan(vals) & ~np.isnan(target)
    n = mask.sum()
    if n >= 20:
        sr, sp = stats.spearmanr(vals[mask], target[mask])
        sig = "STRONG" if abs(sr) > 0.5 else "moderate" if abs(sr) > 0.3 else "weak"
        print(f"{col:<35} {sr:>12.4f} {n:>8} {sig}")

# Incremental value of in-flight OO beyond completed TF=3
print("\n--- Incremental value: residuals after completed TF=3(M-4) ---")
mask_base = combined['completed_tf3_coverage'].notna() & target.notna()
if mask_base.sum() >= 50:
    x = combined.loc[mask_base, 'completed_tf3_coverage'].values
    y = target[mask_base].values
    slope, intercept, r_base, _, _ = stats.linregress(x, y)
    resid = y - (intercept + slope * x)
    print(f"Baseline: TF=12 ~ completed_TF=3(M-4) coverage")
    print(f"  R² = {r_base**2:.4f}, r = {r_base:.4f}, N = {mask_base.sum()}")

    resid_s = pd.Series(resid, index=combined.index[mask_base])

    print(f"\n{'In-flight feature':<35} {'r with resid':>14} {'p-value':>12} {'N':>8} {'Adds?'}")
    print("-" * 75)
    for col in ['inflight_OO_Coverage', 'inflight_OO_Bias',
                'inflight_Open_Orders', 'inflight_Covered_Open_Orders',
                'inflight_oo_sl1_ratio', 'inflight_coo_sl1_ratio']:
        vals = combined.loc[mask_base, col]
        m = vals.notna()
        n = m.sum()
        if n >= 20:
            sr, sp = stats.spearmanr(vals[m], resid_s[m])
            adds = "YES" if (abs(sr) > 0.15 and sp < 0.05) else \
                   "marginal" if abs(sr) > 0.1 else "no"
            print(f"{col:<35} {sr:>14.4f} {sp:>12.6f} {n:>8} {adds}")
        else:
            print(f"{col:<35} {'insuff.':>14} {'':>12} {n:>8}")


# ══════════════════════════════════════════════════════════════════
# PART D: DRIFT DETECTION WITH OFFSET
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("PART D: DRIFT DETECTION (does Δ in lagged TF=3 predict Δ in TF=12?)")
print("=" * 90)
print("Using TF=3 from M-4 (completed) — does the change from M-5→M-4")
print("predict the change from M-1→M in TF=12?\n")

# TF=3 at M-4 over time, per site (Lag=1)
tf3_off_ts = tf3_shifted[tf3_shifted['Prediction_Lag'] == 1].pivot_table(
    index='Reference_Month', columns='Site', values='fc_coverage', aggfunc='mean')
tf12_ts = tf12[tf12['Prediction_Lag'] == 1].pivot_table(
    index='Reference_Month', columns='Site', values='fc_coverage', aggfunc='mean')

tf3_off_diff = tf3_off_ts.diff()
tf12_diff = tf12_ts.diff()

shared_months = tf3_off_diff.index.intersection(tf12_diff.index)
shared_sites = tf3_off_diff.columns.intersection(tf12_diff.columns)

changes_tf3 = []
changes_tf12 = []
for site in shared_sites:
    for month in shared_months:
        v3 = tf3_off_diff.loc[month, site]
        v12 = tf12_diff.loc[month, site]
        if not np.isnan(v3) and not np.isnan(v12):
            changes_tf3.append(v3)
            changes_tf12.append(v12)

changes_tf3 = np.array(changes_tf3)
changes_tf12 = np.array(changes_tf12)
print(f"Site-month change pairs: {len(changes_tf3)}")

if len(changes_tf3) >= 50:
    pr, pp = stats.pearsonr(changes_tf3, changes_tf12)
    sr, sp = stats.spearmanr(changes_tf3, changes_tf12)
    print(f"Δ completed_TF3(M-4) vs Δ TF12(M):")
    print(f"  Pearson r = {pr:.4f}, Spearman r = {sr:.4f}")
    print(f"  {'STRONG' if sr > 0.3 else 'moderate' if sr > 0.15 else 'weak'} signal")


# ══════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("SUMMARY & INTERPRETATION")
print("=" * 90)
print("""
PART A: Same-month baseline (for reference)
  TF=3 and TF=12 from same forecast month. Previously showed r=0.95-0.98.

PART B: Realistic offset (TF=3 from M-4, TF=12 at M)
  The actual use case. How much does the 4-month gap degrade the signal?
  Also shows degradation curve across multiple offsets.

PART C: In-flight OO from current month
  At month M, the TF=3 window is in-flight. Does OO data add value
  beyond the completed TF=3 from M-4?

PART D: Drift detection with offset
  Does the change in completed TF=3 predict changes in TF=12?

Key question: Is the offset correlation still strong enough to be useful,
or does the 4-month gap destroy the signal?
""")
print("=" * 90)
print("ANALYSIS COMPLETE")
print("=" * 90)
