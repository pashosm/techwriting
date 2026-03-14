#!/usr/bin/env python3
"""
COO Signal Integration Analysis
=================================
Understand the relationship between in-flight Covered Open Orders (COO)
and TF=12 fc_coverage to determine the best integration method.

Questions:
1. What is the functional form? Linear? Log? Saturating?
2. Is there systematic bias (over/under prediction)?
3. Should we adjust curves, or replace them?
4. Does the relationship hold across lags, sites, time?
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.width', 160)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')

# ── Load ───────────────────────────────────────────────────────────
CSV = "/home/user/techwriting/training_data_anonymized.csv"
df = pd.read_csv(CSV)
df['Reference_Month'] = pd.to_datetime(df['Reference_Month'])

df['fc_coverage'] = np.where(
    df['Actual_Sales'] > 0,
    df['Covered_Orders'] / df['Actual_Sales'], np.nan)
df['coo_sl1_ratio'] = np.where(
    df['Historical_Sales_Lag1'] > 0,
    df['Covered_Open_Orders'] / df['Historical_Sales_Lag1'], np.nan)

tf12 = df[df['Timeframe'] == 12].copy()
tf3 = df[df['Timeframe'] == 3].copy()

# Also build the "stale TF=12 curve" — for each (site, M), the historical
# average TF=12 fc_coverage from prior completed windows.
# And the completed TF=3 from M-4.

print("=" * 90)
print("COO SIGNAL INTEGRATION ANALYSIS")
print("=" * 90)

# ══════════════════════════════════════════════════════════════════
# SECTION 1: FUNCTIONAL FORM
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("SECTION 1: FUNCTIONAL FORM — COO/SL1 vs TF=12 fc_coverage")
print("=" * 90)

# At each lag, look at the COO/SL1 ratio vs fc_coverage
for lag in [1, 3, 6, 9, 12]:
    sub = tf12[tf12['Prediction_Lag'] == lag].copy()
    x = sub['coo_sl1_ratio'].values
    y = sub['fc_coverage'].values
    mask = ~np.isnan(x) & ~np.isnan(y) & (x < np.percentile(x[~np.isnan(x)], 99))

    if mask.sum() < 50:
        continue

    xm, ym = x[mask], y[mask]

    # Linear fit
    slope, intercept, r_lin, p, _ = stats.linregress(xm, ym)

    # Log fit (log(x+eps) vs y)
    eps = 0.001
    xlog = np.log(xm + eps)
    slope_log, int_log, r_log, _, _ = stats.linregress(xlog, ym)

    # Rank correlation (nonlinearity indicator: if Spearman >> Pearson, nonlinear)
    sr, _ = stats.spearmanr(xm, ym)

    print(f"\nLag={lag} (N={mask.sum()}):")
    print(f"  Linear:    y = {slope:.4f}*x + {intercept:.4f}, R²={r_lin**2:.4f}, r={r_lin:.4f}")
    print(f"  Log:       y = {slope_log:.4f}*log(x) + {int_log:.4f}, R²={r_log**2:.4f}, r={r_log:.4f}")
    print(f"  Spearman:  r = {sr:.4f}")
    print(f"  Spearman vs Pearson gap: {sr - r_lin:.4f} "
          f"{'(nonlinear)' if abs(sr - r_lin) > 0.05 else '(~linear)'}")

    # Quantile analysis: bin x into deciles, show mean y in each
    print(f"  {'Decile':<10} {'COO/SL1 range':>20} {'Mean fc_cov':>14} {'Median fc_cov':>16} {'N':>6}")
    print(f"  {'-'*70}")
    try:
        deciles = pd.qcut(xm, 10, duplicates='drop')
        for name, group_mask in zip(deciles.categories, [deciles == cat for cat in deciles.categories]):
            yg = ym[group_mask]
            if len(yg) > 0:
                print(f"  {str(name):<10} {'':>20} {yg.mean():>14.4f} {np.median(yg):>16.4f} {len(yg):>6}")
    except Exception:
        # Fallback: manual bins
        pcts = np.percentile(xm, np.arange(0, 101, 10))
        for i in range(len(pcts) - 1):
            bin_mask = (xm >= pcts[i]) & (xm < pcts[i+1]) if i < len(pcts)-2 else (xm >= pcts[i])
            yg = ym[bin_mask]
            if len(yg) > 0:
                print(f"  {i+1:<10} {pcts[i]:>8.4f}-{pcts[i+1]:>8.4f} {yg.mean():>14.4f} {np.median(yg):>16.4f} {len(yg):>6}")


# ══════════════════════════════════════════════════════════════════
# SECTION 2: BIAS ANALYSIS
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("SECTION 2: BIAS — Does COO/SL1 systematically over/under predict?")
print("=" * 90)

# If COO/SL1 = 0.3 and fc_coverage = 0.25, the ratio is 1.2x (over-predicts)
# Compare COO/SL1 to fc_coverage directly
for lag in [1, 3, 6, 9, 12]:
    sub = tf12[tf12['Prediction_Lag'] == lag]
    x = sub['coo_sl1_ratio'].values
    y = sub['fc_coverage'].values
    mask = ~np.isnan(x) & ~np.isnan(y) & (y > 0.001)  # need positive coverage

    if mask.sum() < 50:
        continue

    xm, ym = x[mask], y[mask]
    ratio = xm / ym  # >1 means COO/SL1 over-predicts coverage

    print(f"\nLag={lag} (N={mask.sum()}):")
    print(f"  COO/SL1 vs fc_coverage ratio (>1 = over-predict):")
    print(f"    mean={ratio.mean():.4f}, median={np.median(ratio):.4f}, "
          f"std={ratio.std():.4f}")
    print(f"    % over-predict (ratio > 1): {100*(ratio > 1).mean():.1f}%")
    print(f"    % under-predict (ratio < 1): {100*(ratio < 1).mean():.1f}%")

    # Bias by coverage level
    print(f"  Bias by fc_coverage level:")
    print(f"  {'Coverage bin':>15} {'Mean ratio':>12} {'Median ratio':>14} {'N':>6} {'Direction'}")
    print(f"  {'-'*55}")
    bins = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0), (1.0, 999)]
    for lo, hi in bins:
        bm = (ym >= lo) & (ym < hi)
        if bm.sum() >= 10:
            r = ratio[bm]
            direction = "OVER" if np.median(r) > 1.1 else "UNDER" if np.median(r) < 0.9 else "~fair"
            print(f"  {lo:.1f}-{hi:.1f}{'>':>10} {r.mean():>12.4f} {np.median(r):>14.4f} {bm.sum():>6} {direction}")


# ══════════════════════════════════════════════════════════════════
# SECTION 3: RELATIONSHIP ACROSS LAGS (the curve shape)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("SECTION 3: COO/SL1 vs fc_coverage ACROSS LAGS")
print("=" * 90)
print("How does the COO/SL1 → fc_coverage relationship change by lag?")
print("This tells us if we can use a single model or need lag-specific adjustment.\n")

lags = sorted(df['Prediction_Lag'].unique())

print(f"{'Lag':>5} {'Slope':>10} {'Intercept':>12} {'R²':>8} {'Mean COO/SL1':>14} {'Mean fc_cov':>14} {'Ratio':>10}")
print("-" * 80)

for lag in lags:
    sub = tf12[tf12['Prediction_Lag'] == lag]
    x = sub['coo_sl1_ratio'].values
    y = sub['fc_coverage'].values
    mask = ~np.isnan(x) & ~np.isnan(y) & (x < np.percentile(x[~np.isnan(x)], 99))
    if mask.sum() < 50:
        continue
    xm, ym = x[mask], y[mask]
    slope, intercept, r, _, _ = stats.linregress(xm, ym)
    mean_ratio = ym.mean() / xm.mean() if xm.mean() > 0 else np.nan
    print(f"{lag:>5} {slope:>10.4f} {intercept:>12.4f} {r**2:>8.4f} {xm.mean():>14.4f} {ym.mean():>14.4f} {mean_ratio:>10.4f}")


# ══════════════════════════════════════════════════════════════════
# SECTION 4: INTEGRATION APPROACHES — COMPARISON
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("SECTION 4: COMPARING INTEGRATION APPROACHES")
print("=" * 90)
print("For each (site, ref_month, lag), predict TF=12 fc_coverage using:")
print("  A) Stale TF=12 curve only (historical site average)")
print("  B) COO/SL1 alone (linear model)")
print("  C) Completed TF=3(M-4) alone")
print("  D) Stale curve + COO/SL1 adjustment")
print("  E) Stale curve + completed TF=3(M-4) + COO/SL1\n")

# Build stale TF=12 curve: for each (site, lag), the leave-one-out mean
# of fc_coverage across all reference months BEFORE the current one.
# Simplified: use expanding mean up to M-1.

tf12_sorted = tf12.sort_values(['Site', 'Prediction_Lag', 'Reference_Month'])

# For each row, compute the historical average fc_coverage for that site+lag
# using only prior months.
tf12_sorted['hist_avg_cov'] = tf12_sorted.groupby(['Site', 'Prediction_Lag'])['fc_coverage'].transform(
    lambda s: s.expanding().mean().shift(1)
)

# Completed TF=3 from M-4
tf3_shifted = tf3.copy()
tf3_shifted['Reference_Month'] = tf3_shifted['Reference_Month'] + pd.DateOffset(months=4)
tf3_completed = tf3_shifted.groupby(['Site', 'Reference_Month', 'Prediction_Lag'])['fc_coverage'].mean()
tf3_completed.name = 'tf3_m4_coverage'

# Merge
tf12_sorted = tf12_sorted.set_index(['Site', 'Reference_Month', 'Prediction_Lag'])
tf12_sorted = tf12_sorted.join(tf3_completed, how='left')
tf12_sorted = tf12_sorted.reset_index()

# Filter to rows where we have all signals
eval_mask = (
    tf12_sorted['fc_coverage'].notna() &
    tf12_sorted['coo_sl1_ratio'].notna() &
    tf12_sorted['hist_avg_cov'].notna()
)
eval_df = tf12_sorted[eval_mask].copy()
print(f"Evaluation rows (all signals available): {len(eval_df)}")
eval_with_tf3 = eval_df[eval_df['tf3_m4_coverage'].notna()].copy()
print(f"Evaluation rows (including TF=3 M-4): {len(eval_with_tf3)}")

# ── Approach A: Stale curve only ──────────────────────────────────
eval_df['pred_A'] = eval_df['hist_avg_cov']

# ── Approach B: COO/SL1 linear (per-lag) ──────────────────────────
# Fit linear model per lag on training data, predict
for lag in lags:
    lag_mask = eval_df['Prediction_Lag'] == lag
    if lag_mask.sum() < 50:
        continue
    x = eval_df.loc[lag_mask, 'coo_sl1_ratio'].values
    y = eval_df.loc[lag_mask, 'fc_coverage'].values
    slope, intercept, _, _, _ = stats.linregress(x, y)
    eval_df.loc[lag_mask, 'pred_B'] = intercept + slope * x

# ── Approach D: Stale + COO adjustment ────────────────────────────
# Fit: fc_coverage = a * hist_avg + b * COO/SL1 + c
from numpy.linalg import lstsq

for lag in lags:
    lm = eval_df['Prediction_Lag'] == lag
    if lm.sum() < 50:
        continue
    X = np.column_stack([
        eval_df.loc[lm, 'hist_avg_cov'].values,
        eval_df.loc[lm, 'coo_sl1_ratio'].values,
        np.ones(lm.sum())
    ])
    y = eval_df.loc[lm, 'fc_coverage'].values
    coefs, _, _, _ = lstsq(X, y, rcond=None)
    eval_df.loc[lm, 'pred_D'] = X @ coefs
    if lag == 1:
        print(f"\n  Approach D coefficients at Lag=1:")
        print(f"    fc_cov = {coefs[0]:.4f} * hist_avg + {coefs[1]:.4f} * COO/SL1 + {coefs[2]:.4f}")

# ── Approach E: Stale + TF3(M-4) + COO (where TF3 available) ─────
for lag in lags:
    lm = eval_with_tf3['Prediction_Lag'] == lag
    if lm.sum() < 50:
        continue
    X = np.column_stack([
        eval_with_tf3.loc[lm, 'hist_avg_cov'].values,
        eval_with_tf3.loc[lm, 'tf3_m4_coverage'].values,
        eval_with_tf3.loc[lm, 'coo_sl1_ratio'].values,
        np.ones(lm.sum())
    ])
    y = eval_with_tf3.loc[lm, 'fc_coverage'].values
    coefs, _, _, _ = lstsq(X, y, rcond=None)
    eval_with_tf3.loc[lm, 'pred_E'] = X @ coefs
    if lag == 1:
        print(f"\n  Approach E coefficients at Lag=1:")
        print(f"    fc_cov = {coefs[0]:.4f} * hist_avg + {coefs[1]:.4f} * TF3(M-4) "
              f"+ {coefs[2]:.4f} * COO/SL1 + {coefs[3]:.4f}")

# ── Compare MAE/RMSE across approaches ───────────────────────────
print("\n--- Overall prediction accuracy ---")
print(f"{'Approach':<45} {'MAE':>10} {'RMSE':>10} {'R²':>10} {'N':>8}")
print("-" * 90)

y_true = eval_df['fc_coverage'].values
for label, col in [('A: Stale curve only', 'pred_A'),
                    ('B: COO/SL1 linear (per-lag)', 'pred_B'),
                    ('D: Stale + COO/SL1 (per-lag)', 'pred_D')]:
    if col not in eval_df.columns:
        continue
    pred = eval_df[col].values
    mask = ~np.isnan(pred) & ~np.isnan(y_true)
    if mask.sum() < 20:
        continue
    mae = np.mean(np.abs(y_true[mask] - pred[mask]))
    rmse = np.sqrt(np.mean((y_true[mask] - pred[mask])**2))
    ss_res = np.sum((y_true[mask] - pred[mask])**2)
    ss_tot = np.sum((y_true[mask] - y_true[mask].mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    print(f"{label:<45} {mae:>10.4f} {rmse:>10.4f} {r2:>10.4f} {mask.sum():>8}")

# Approach E on the subset that has TF=3
if 'pred_E' in eval_with_tf3.columns:
    y_true_e = eval_with_tf3['fc_coverage'].values
    # Also compute A and D on same subset for fair comparison
    for label, source_df, col in [
        ('A: Stale curve only (TF3 subset)', eval_with_tf3, 'pred_A'),
        ('D: Stale + COO/SL1 (TF3 subset)', eval_with_tf3, 'pred_D'),
        ('E: Stale + TF3(M-4) + COO/SL1', eval_with_tf3, 'pred_E'),
    ]:
        if col not in source_df.columns:
            # Recompute A and B on this subset
            if col == 'pred_A':
                source_df[col] = source_df['hist_avg_cov']
            else:
                continue
        pred = source_df[col].values
        y_t = source_df['fc_coverage'].values
        mask = ~np.isnan(pred) & ~np.isnan(y_t)
        if mask.sum() < 20:
            continue
        mae = np.mean(np.abs(y_t[mask] - pred[mask]))
        rmse = np.sqrt(np.mean((y_t[mask] - pred[mask])**2))
        ss_res = np.sum((y_t[mask] - pred[mask])**2)
        ss_tot = np.sum((y_t[mask] - y_t[mask].mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        print(f"{label:<45} {mae:>10.4f} {rmse:>10.4f} {r2:>10.4f} {mask.sum():>8}")


# ── Per-lag breakdown ─────────────────────────────────────────────
print("\n--- Per-lag MAE comparison ---")
print(f"{'Lag':>5} {'A:Stale':>12} {'B:COO only':>12} {'D:Stale+COO':>14} {'D vs A improv':>16}")
print("-" * 65)

for lag in lags:
    lm = eval_df['Prediction_Lag'] == lag
    if lm.sum() < 50:
        continue
    yt = eval_df.loc[lm, 'fc_coverage'].values
    for label, col in [('A', 'pred_A'), ('B', 'pred_B'), ('D', 'pred_D')]:
        if col not in eval_df.columns:
            continue
        pred = eval_df.loc[lm, col].values
        mask = ~np.isnan(pred) & ~np.isnan(yt)
        if mask.sum() > 0:
            globals()[f'mae_{label}_{lag}'] = np.mean(np.abs(yt[mask] - pred[mask]))

    mae_a = globals().get(f'mae_A_{lag}', np.nan)
    mae_b = globals().get(f'mae_B_{lag}', np.nan)
    mae_d = globals().get(f'mae_D_{lag}', np.nan)
    improv = (mae_a - mae_d) / mae_a * 100 if mae_a > 0 and not np.isnan(mae_d) else np.nan
    print(f"{lag:>5} {mae_a:>12.4f} {mae_b:>12.4f} {mae_d:>14.4f} {improv:>15.1f}%")


# ══════════════════════════════════════════════════════════════════
# SECTION 5: RESIDUAL ANALYSIS ON APPROACH D
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("SECTION 5: RESIDUAL ANALYSIS (Approach D: Stale + COO)")
print("=" * 90)
print("Where does the combined model still fail?\n")

if 'pred_D' in eval_df.columns:
    eval_df['resid_D'] = eval_df['fc_coverage'] - eval_df['pred_D']
    valid = eval_df['resid_D'].notna()

    # Residuals by coverage level
    print("--- Residuals by actual fc_coverage level ---")
    print(f"{'Coverage bin':>15} {'Mean resid':>12} {'Median resid':>14} {'Std':>10} {'N':>8} {'Bias'}")
    print("-" * 70)
    bins = [(0, 0.01), (0.01, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0), (1.0, 999)]
    for lo, hi in bins:
        bm = valid & (eval_df['fc_coverage'] >= lo) & (eval_df['fc_coverage'] < hi)
        if bm.sum() >= 10:
            r = eval_df.loc[bm, 'resid_D']
            bias = "UNDER" if r.mean() > 0.02 else "OVER" if r.mean() < -0.02 else "~ok"
            print(f"  {lo:.2f}-{hi:.2f}{'>':>8} {r.mean():>12.4f} {r.median():>14.4f} {r.std():>10.4f} {bm.sum():>8} {bias}")

    # Residuals by site
    print("\n--- Top 10 sites by absolute residual ---")
    site_resid = eval_df[valid].groupby('Site')['resid_D'].agg(['mean', 'std', 'count'])
    site_resid['abs_mean'] = site_resid['mean'].abs()
    site_resid = site_resid.sort_values('abs_mean', ascending=False)
    print(f"{'Site':>12} {'Mean resid':>12} {'Std':>10} {'N':>8}")
    print("-" * 45)
    for site, row in site_resid.head(10).iterrows():
        print(f"{site:>12} {row['mean']:>12.4f} {row['std']:>10.4f} {int(row['count']):>8}")


# ══════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)
print("""
Key findings for integrating COO signal:

1. FUNCTIONAL FORM: Is COO/SL1 → fc_coverage linear, log, etc.?
2. BIAS: Does COO/SL1 systematically over- or under-predict?
3. BEST APPROACH: Which integration method wins?
   A = stale curve only
   B = COO/SL1 only
   D = stale curve + COO/SL1
   E = stale curve + TF=3(M-4) + COO/SL1
4. WHERE IT FAILS: Residual analysis of the best approach
""")
print("=" * 90)
print("ANALYSIS COMPLETE")
print("=" * 90)
