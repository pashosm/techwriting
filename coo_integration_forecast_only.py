#!/usr/bin/env python3
"""
COO Integration Analysis — Forecast-only subset
=================================================
Previous analysis was dominated by 79K zero-coverage rows.
This version filters to Has_Forecast=1 only (~10K rows) where
coverage is meaningful and prediction accuracy actually matters.
"""

import pandas as pd
import numpy as np
from scipy import stats
from numpy.linalg import lstsq
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.width', 160)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')

# ── Load & filter ──────────────────────────────────────────────────
CSV = "/home/user/techwriting/training_data_anonymized.csv"
df = pd.read_csv(CSV)
df['Reference_Month'] = pd.to_datetime(df['Reference_Month'])

df['fc_coverage'] = np.where(
    df['Actual_Sales'] > 0,
    df['Covered_Orders'] / df['Actual_Sales'], np.nan)
df['coo_sl1_ratio'] = np.where(
    df['Historical_Sales_Lag1'] > 0,
    df['Covered_Open_Orders'] / df['Historical_Sales_Lag1'], np.nan)
df['oo_sl1_ratio'] = np.where(
    df['Historical_Sales_Lag1'] > 0,
    df['Open_Orders'] / df['Historical_Sales_Lag1'], np.nan)
df['fc_sl1_ratio'] = np.where(
    df['Historical_Sales_Lag1'] > 0,
    df['Forecast_Value'] / df['Historical_Sales_Lag1'], np.nan)

# Filter to TF=12, Has_Forecast=1
tf12 = df[(df['Timeframe'] == 12) & (df['Has_Forecast'] == 1)].copy()
tf3 = df[df['Timeframe'] == 3].copy()
lags = sorted(tf12['Prediction_Lag'].unique())

print("=" * 90)
print("COO INTEGRATION — FORECAST-ONLY SUBSET")
print("=" * 90)
print(f"\nTF=12, Has_Forecast=1: {len(tf12)} rows")
print(f"Sites: {tf12['Site'].nunique()}, Ref months: {tf12['Reference_Month'].nunique()}")
print(f"Coverage: mean={tf12['fc_coverage'].mean():.3f}, "
      f"median={tf12['fc_coverage'].median():.3f}, "
      f"std={tf12['fc_coverage'].std():.3f}")


# ══════════════════════════════════════════════════════════════════
# SECTION 1: FUNCTIONAL FORM (forecast-only)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("SECTION 1: FUNCTIONAL FORM (Has_Forecast=1 only)")
print("=" * 90)

for lag in [1, 3, 6, 9, 12]:
    sub = tf12[tf12['Prediction_Lag'] == lag]
    x = sub['coo_sl1_ratio'].values
    y = sub['fc_coverage'].values
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 30:
        continue
    xm, ym = x[mask], y[mask]

    # Remove extreme outliers
    p99 = np.percentile(xm, 99)
    clip = xm < p99
    xm, ym = xm[clip], ym[clip]

    slope, intercept, r_lin, _, _ = stats.linregress(xm, ym)
    eps = 0.001
    slope_log, int_log, r_log, _, _ = stats.linregress(np.log(xm + eps), ym)
    sr, _ = stats.spearmanr(xm, ym)

    print(f"\nLag={lag} (N={len(xm)}):")
    print(f"  Linear:   R²={r_lin**2:.4f}, slope={slope:.4f}, intercept={intercept:.4f}")
    print(f"  Log:      R²={r_log**2:.4f}")
    print(f"  Spearman: {sr:.4f}  (gap={sr-r_lin:.4f} {'nonlinear' if abs(sr-r_lin)>0.05 else '~linear'})")

    # Decile analysis
    try:
        deciles = pd.qcut(xm, 5, duplicates='drop')
        print(f"  {'Quintile':<25} {'Mean cov':>10} {'Median cov':>12} {'N':>6}")
        print(f"  {'-'*58}")
        for cat in deciles.categories:
            g = ym[deciles == cat]
            print(f"  {str(cat):<25} {g.mean():>10.4f} {np.median(g):>12.4f} {len(g):>6}")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════
# SECTION 2: BIAS (forecast-only)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("SECTION 2: BIAS (Has_Forecast=1 only)")
print("=" * 90)

for lag in [1, 3, 6, 9, 12]:
    sub = tf12[tf12['Prediction_Lag'] == lag]
    x = sub['coo_sl1_ratio'].values
    y = sub['fc_coverage'].values
    mask = ~np.isnan(x) & ~np.isnan(y) & (y > 0.001)
    if mask.sum() < 30:
        continue
    xm, ym = x[mask], y[mask]
    ratio = xm / ym

    print(f"\nLag={lag} (N={mask.sum()}):")
    print(f"  COO/SL1 as fraction of fc_coverage:")
    print(f"    mean={ratio.mean():.4f}, median={np.median(ratio):.4f}")
    print(f"    Over-predict: {100*(ratio>1).mean():.1f}%, Under: {100*(ratio<1).mean():.1f}%")


# ══════════════════════════════════════════════════════════════════
# SECTION 3: SLOPE / MULTIPLIER ACROSS LAGS
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("SECTION 3: COO/SL1 → fc_coverage relationship by lag")
print("=" * 90)

print(f"\n{'Lag':>5} {'Slope':>8} {'Intercept':>10} {'R²(lin)':>10} {'R²(log)':>10} "
      f"{'Mean COO/SL1':>14} {'Mean cov':>10} {'Ratio':>8}")
print("-" * 85)

for lag in lags:
    sub = tf12[tf12['Prediction_Lag'] == lag]
    x = sub['coo_sl1_ratio'].values
    y = sub['fc_coverage'].values
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 30:
        continue
    xm, ym = x[mask], y[mask]
    p99 = np.percentile(xm, 99)
    clip = xm < p99
    xm, ym = xm[clip], ym[clip]

    slope, intercept, r_lin, _, _ = stats.linregress(xm, ym)
    _, _, r_log, _, _ = stats.linregress(np.log(xm + 0.001), ym)
    mean_ratio = ym.mean() / xm.mean() if xm.mean() > 0 else np.nan
    print(f"{lag:>5} {slope:>8.3f} {intercept:>10.4f} {r_lin**2:>10.4f} {r_log**2:>10.4f} "
          f"{xm.mean():>14.4f} {ym.mean():>14.4f} {mean_ratio:>8.2f}")


# ══════════════════════════════════════════════════════════════════
# SECTION 4: INTEGRATION APPROACHES (forecast-only)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("SECTION 4: COMPARING APPROACHES (Has_Forecast=1 only)")
print("=" * 90)

# Stale curve: expanding mean of fc_coverage for this site+lag, shifted by 1
tf12_sorted = tf12.sort_values(['Site', 'Prediction_Lag', 'Reference_Month'])
tf12_sorted['hist_avg'] = tf12_sorted.groupby(['Site', 'Prediction_Lag'])['fc_coverage'].transform(
    lambda s: s.expanding().mean().shift(1)
)

# Completed TF=3 from M-4
tf3_shifted = tf3.copy()
tf3_shifted['Reference_Month'] = tf3_shifted['Reference_Month'] + pd.DateOffset(months=4)
tf3_completed = tf3_shifted.groupby(['Site', 'Reference_Month', 'Prediction_Lag'])['fc_coverage'].mean()
tf3_completed.name = 'tf3_m4_cov'

tf12_sorted = tf12_sorted.set_index(['Site', 'Reference_Month', 'Prediction_Lag'])
tf12_sorted = tf12_sorted.join(tf3_completed, how='left')
tf12_sorted = tf12_sorted.reset_index()

# Eval set: rows with all signals
eval_df = tf12_sorted[
    tf12_sorted['fc_coverage'].notna() &
    tf12_sorted['coo_sl1_ratio'].notna() &
    tf12_sorted['hist_avg'].notna()
].copy()

print(f"\nEval rows (forecast-only, all signals): {len(eval_df)}")
print(f"  With TF=3(M-4): {eval_df['tf3_m4_cov'].notna().sum()}")

# ── Approach A: Stale curve only ──
eval_df['pred_A'] = eval_df['hist_avg']

# ── Approach B: COO/SL1 linear (per-lag) ──
for lag in lags:
    lm = eval_df['Prediction_Lag'] == lag
    if lm.sum() < 30:
        continue
    x = eval_df.loc[lm, 'coo_sl1_ratio'].values
    y = eval_df.loc[lm, 'fc_coverage'].values
    slope, intercept, _, _, _ = stats.linregress(x, y)
    eval_df.loc[lm, 'pred_B_lin'] = intercept + slope * x

# ── Approach B_log: COO/SL1 log (per-lag) ──
for lag in lags:
    lm = eval_df['Prediction_Lag'] == lag
    if lm.sum() < 30:
        continue
    x = eval_df.loc[lm, 'coo_sl1_ratio'].values
    y = eval_df.loc[lm, 'fc_coverage'].values
    xlog = np.log(x + 0.001)
    slope, intercept, _, _, _ = stats.linregress(xlog, y)
    eval_df.loc[lm, 'pred_B_log'] = intercept + slope * xlog

# ── Approach D: Stale + COO/SL1 (per-lag, linear) ──
for lag in lags:
    lm = eval_df['Prediction_Lag'] == lag
    if lm.sum() < 30:
        continue
    X = np.column_stack([
        eval_df.loc[lm, 'hist_avg'].values,
        eval_df.loc[lm, 'coo_sl1_ratio'].values,
        np.ones(lm.sum())
    ])
    y = eval_df.loc[lm, 'fc_coverage'].values
    coefs, _, _, _ = lstsq(X, y, rcond=None)
    eval_df.loc[lm, 'pred_D'] = X @ coefs
    if lag == 1:
        print(f"\n  D coefficients (Lag=1): {coefs[0]:.4f}*hist + {coefs[1]:.4f}*COO + {coefs[2]:.4f}")

# ── Approach D_log: Stale + log(COO/SL1) (per-lag) ──
for lag in lags:
    lm = eval_df['Prediction_Lag'] == lag
    if lm.sum() < 30:
        continue
    X = np.column_stack([
        eval_df.loc[lm, 'hist_avg'].values,
        np.log(eval_df.loc[lm, 'coo_sl1_ratio'].values + 0.001),
        np.ones(lm.sum())
    ])
    y = eval_df.loc[lm, 'fc_coverage'].values
    coefs, _, _, _ = lstsq(X, y, rcond=None)
    eval_df.loc[lm, 'pred_D_log'] = X @ coefs
    if lag == 1:
        print(f"  D_log coefficients (Lag=1): {coefs[0]:.4f}*hist + {coefs[1]:.4f}*log(COO) + {coefs[2]:.4f}")

# ── Approach E: Stale + TF3(M-4) + COO (per-lag) ──
eval_tf3 = eval_df[eval_df['tf3_m4_cov'].notna()].copy()
for lag in lags:
    lm = eval_tf3['Prediction_Lag'] == lag
    if lm.sum() < 30:
        continue
    X = np.column_stack([
        eval_tf3.loc[lm, 'hist_avg'].values,
        eval_tf3.loc[lm, 'tf3_m4_cov'].values,
        np.log(eval_tf3.loc[lm, 'coo_sl1_ratio'].values + 0.001),
        np.ones(lm.sum())
    ])
    y = eval_tf3.loc[lm, 'fc_coverage'].values
    coefs, _, _, _ = lstsq(X, y, rcond=None)
    eval_tf3.loc[lm, 'pred_E'] = X @ coefs
    if lag == 1:
        print(f"  E coefficients (Lag=1): {coefs[0]:.4f}*hist + {coefs[1]:.4f}*TF3 + {coefs[2]:.4f}*log(COO) + {coefs[3]:.4f}")

# ── Approach F: Stale + all available features (per-lag) ──
# Add order earliness and lead time
for lag in lags:
    lm = eval_df['Prediction_Lag'] == lag
    sub = eval_df[lm]
    feat_mask = (
        sub['hist_avg'].notna() &
        sub['coo_sl1_ratio'].notna() &
        sub['Avg_Weighted_Order_Earliness'].notna() &
        sub['Avg_Weighted_Lead_Time'].notna() &
        sub['oo_sl1_ratio'].notna()
    )
    if feat_mask.sum() < 30:
        continue
    idx = sub[feat_mask].index
    X = np.column_stack([
        eval_df.loc[idx, 'hist_avg'].values,
        np.log(eval_df.loc[idx, 'coo_sl1_ratio'].values + 0.001),
        eval_df.loc[idx, 'oo_sl1_ratio'].values,
        eval_df.loc[idx, 'Avg_Weighted_Order_Earliness'].values,
        eval_df.loc[idx, 'Avg_Weighted_Lead_Time'].values,
        np.ones(len(idx))
    ])
    y = eval_df.loc[idx, 'fc_coverage'].values
    coefs, _, _, _ = lstsq(X, y, rcond=None)
    eval_df.loc[idx, 'pred_F'] = X @ coefs
    if lag == 1:
        print(f"  F coefficients (Lag=1): {coefs[0]:.4f}*hist + {coefs[1]:.4f}*log(COO) "
              f"+ {coefs[2]:.4f}*OO/SL1 + {coefs[3]:.4f}*earliness + {coefs[4]:.4f}*leadtime + {coefs[5]:.4f}")

# ── Score all approaches ──────────────────────────────────────────
def score(y_true, y_pred, label, n_min=20):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    n = mask.sum()
    if n < n_min:
        return None
    yt, yp = y_true[mask], y_pred[mask]
    mae = np.mean(np.abs(yt - yp))
    rmse = np.sqrt(np.mean((yt - yp)**2))
    ss_res = np.sum((yt - yp)**2)
    ss_tot = np.sum((yt - yt.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    median_ae = np.median(np.abs(yt - yp))
    return {'label': label, 'MAE': mae, 'MedAE': median_ae, 'RMSE': rmse, 'R²': r2, 'N': n}


print("\n--- Overall prediction accuracy (Has_Forecast=1 only) ---")
print(f"{'Approach':<45} {'MAE':>8} {'MedAE':>8} {'RMSE':>8} {'R²':>8} {'N':>8}")
print("-" * 85)

y_true = eval_df['fc_coverage'].values
approaches = [
    ('A: Stale curve only', 'pred_A'),
    ('B: COO/SL1 linear', 'pred_B_lin'),
    ('B_log: COO/SL1 log', 'pred_B_log'),
    ('D: Stale + COO linear', 'pred_D'),
    ('D_log: Stale + log(COO)', 'pred_D_log'),
    ('F: Stale + log(COO) + OO + timing', 'pred_F'),
]
for label, col in approaches:
    if col not in eval_df.columns:
        continue
    s = score(y_true, eval_df[col].values, label)
    if s:
        print(f"{s['label']:<45} {s['MAE']:>8.4f} {s['MedAE']:>8.4f} {s['RMSE']:>8.4f} {s['R²']:>8.4f} {s['N']:>8}")

# Approach E on TF3 subset
if 'pred_E' in eval_tf3.columns:
    print(f"\n  --- On TF3-available subset ({len(eval_tf3)} rows) ---")
    y_t3 = eval_tf3['fc_coverage'].values
    for label, col in [('A: Stale (TF3 subset)', 'pred_A'),
                        ('D_log: Stale+log(COO) (TF3 subset)', 'pred_D_log'),
                        ('E: Stale+TF3(M-4)+log(COO)', 'pred_E')]:
        if col not in eval_tf3.columns:
            continue
        s = score(y_t3, eval_tf3[col].values, label)
        if s:
            print(f"  {s['label']:<43} {s['MAE']:>8.4f} {s['MedAE']:>8.4f} {s['RMSE']:>8.4f} {s['R²']:>8.4f} {s['N']:>8}")


# ── Per-lag breakdown ─────────────────────────────────────────────
print("\n--- Per-lag MAE comparison ---")
print(f"{'Lag':>5} {'A:Stale':>10} {'B_log:COO':>10} {'D_log:St+COO':>14} {'F:Full':>10} {'D_log vs A':>12}")
print("-" * 70)

for lag in lags:
    lm = eval_df['Prediction_Lag'] == lag
    if lm.sum() < 30:
        continue
    yt = eval_df.loc[lm, 'fc_coverage'].values

    maes = {}
    for key, col in [('A', 'pred_A'), ('B_log', 'pred_B_log'), ('D_log', 'pred_D_log'), ('F', 'pred_F')]:
        if col in eval_df.columns:
            pred = eval_df.loc[lm, col].values
            m = ~np.isnan(pred) & ~np.isnan(yt)
            if m.sum() > 0:
                maes[key] = np.mean(np.abs(yt[m] - pred[m]))

    a = maes.get('A', np.nan)
    b = maes.get('B_log', np.nan)
    d = maes.get('D_log', np.nan)
    f = maes.get('F', np.nan)
    improv = (a - d) / a * 100 if a > 0 and not np.isnan(d) else np.nan
    print(f"{lag:>5} {a:>10.4f} {b:>10.4f} {d:>14.4f} {f:>10.4f} {improv:>11.1f}%")


# ══════════════════════════════════════════════════════════════════
# SECTION 5: ERROR PREDICTION — can we predict stale curve error?
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("SECTION 5: PREDICTING STALE CURVE ERROR")
print("=" * 90)
print("Can COO/OO features predict HOW WRONG the stale curve will be?\n")

eval_df['stale_error'] = eval_df['fc_coverage'] - eval_df['hist_avg']

print(f"Stale curve error stats (actual - predicted):")
print(f"  mean={eval_df['stale_error'].mean():.4f} (positive = stale under-predicts)")
print(f"  std={eval_df['stale_error'].std():.4f}")
print(f"  median={eval_df['stale_error'].median():.4f}")

# What correlates with the error?
error = eval_df['stale_error']
print(f"\n{'Feature':<35} {'Spearman r':>12} {'p-value':>12} {'N':>8}")
print("-" * 70)

for col in ['coo_sl1_ratio', 'oo_sl1_ratio', 'fc_sl1_ratio',
            'OO_Coverage', 'OO_Bias',
            'Avg_Weighted_Order_Earliness', 'Avg_Weighted_Lead_Time',
            'Open_Orders', 'Covered_Open_Orders', 'Forecast_Value']:
    if col not in eval_df.columns:
        continue
    vals = eval_df[col]
    mask = vals.notna() & error.notna()
    n = mask.sum()
    if n >= 20:
        sr, sp = stats.spearmanr(vals[mask], error[mask])
        print(f"{col:<35} {sr:>12.4f} {sp:>12.6f} {n:>8}")

# Also: does the CHANGE in COO from previous month predict the error?
eval_df_sorted = eval_df.sort_values(['Site', 'Prediction_Lag', 'Reference_Month'])
eval_df_sorted['coo_sl1_change'] = eval_df_sorted.groupby(
    ['Site', 'Prediction_Lag'])['coo_sl1_ratio'].diff()

mask = eval_df_sorted['coo_sl1_change'].notna() & eval_df_sorted['stale_error'].notna()
if mask.sum() >= 20:
    sr, sp = stats.spearmanr(eval_df_sorted.loc[mask, 'coo_sl1_change'],
                              eval_df_sorted.loc[mask, 'stale_error'])
    print(f"{'Δ COO/SL1 (month-over-month)':<35} {sr:>12.4f} {sp:>12.6f} {mask.sum():>8}")


# ══════════════════════════════════════════════════════════════════
# SECTION 6: RESIDUAL ANALYSIS ON BEST APPROACH
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("SECTION 6: RESIDUAL ANALYSIS (best approach)")
print("=" * 90)

best_col = 'pred_D_log'
eval_df['resid'] = eval_df['fc_coverage'] - eval_df[best_col]
valid = eval_df['resid'].notna()

print(f"\nResiduals for D_log (Stale + log(COO)):")
print(f"  mean={eval_df.loc[valid, 'resid'].mean():.4f}, "
      f"std={eval_df.loc[valid, 'resid'].std():.4f}")

print(f"\n--- By coverage level ---")
print(f"{'Coverage bin':>15} {'Mean resid':>12} {'Std':>8} {'N':>8} {'Bias'}")
print("-" * 55)
bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, 5)]
for lo, hi in bins:
    bm = valid & (eval_df['fc_coverage'] >= lo) & (eval_df['fc_coverage'] < hi)
    if bm.sum() >= 10:
        r = eval_df.loc[bm, 'resid']
        bias = "UNDER" if r.mean() > 0.03 else "OVER" if r.mean() < -0.03 else "~ok"
        print(f"  {lo:.1f}-{hi:.1f}{'>':>8} {r.mean():>12.4f} {r.std():>8.4f} {bm.sum():>8} {bias}")

print(f"\n--- By lag ---")
print(f"{'Lag':>5} {'Mean resid':>12} {'Std':>8} {'N':>8}")
print("-" * 40)
for lag in lags:
    lm = valid & (eval_df['Prediction_Lag'] == lag)
    if lm.sum() >= 10:
        r = eval_df.loc[lm, 'resid']
        print(f"{lag:>5} {r.mean():>12.4f} {r.std():>8.4f} {lm.sum():>8}")


print("\n" + "=" * 90)
print("ANALYSIS COMPLETE")
print("=" * 90)
