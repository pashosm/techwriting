#!/usr/bin/env python3
"""
Residual Bias Analysis
=======================
The D_log model (stale + log(COO)) over-predicts low-coverage sites
and under-predicts high-coverage sites. Why?

Investigate:
1. Is this regression to the mean? (model shrinks toward center)
2. Is it a feature distribution issue? (low-coverage sites have different COO patterns)
3. Is the relationship actually nonlinear in a way log doesn't capture?
4. Can we identify which rows will be over/under-predicted?
5. Does a nonlinear correction fix it?
"""

import pandas as pd
import numpy as np
from scipy import stats
from numpy.linalg import lstsq
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.width', 160)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')

# ── Load & prep (same as coo_integration_forecast_only.py) ────────
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

tf12 = df[(df['Timeframe'] == 12) & (df['Has_Forecast'] == 1)].copy()
lags = sorted(tf12['Prediction_Lag'].unique())

# Build stale curve
tf12 = tf12.sort_values(['Site', 'Prediction_Lag', 'Reference_Month'])
tf12['hist_avg'] = tf12.groupby(['Site', 'Prediction_Lag'])['fc_coverage'].transform(
    lambda s: s.expanding().mean().shift(1)
)

# Build D_log predictions
eval_df = tf12[
    tf12['fc_coverage'].notna() &
    tf12['coo_sl1_ratio'].notna() &
    tf12['hist_avg'].notna()
].copy()

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

eval_df['resid'] = eval_df['fc_coverage'] - eval_df['pred_D_log']

print("=" * 90)
print("RESIDUAL BIAS ANALYSIS")
print("=" * 90)
print(f"Eval rows: {len(eval_df)}")


# ══════════════════════════════════════════════════════════════════
# 1. IS THIS REGRESSION TO THE MEAN?
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("1. REGRESSION TO THE MEAN CHECK")
print("=" * 90)
print("If the model simply predicts closer to the mean than reality,")
print("it will over-predict lows and under-predict highs.\n")

overall_mean = eval_df['fc_coverage'].mean()
print(f"Overall mean coverage: {overall_mean:.4f}")
print(f"Prediction range: {eval_df['pred_D_log'].min():.4f} to {eval_df['pred_D_log'].max():.4f}")
print(f"Actual range:     {eval_df['fc_coverage'].min():.4f} to {eval_df['fc_coverage'].max():.4f}")
print(f"Prediction std:   {eval_df['pred_D_log'].std():.4f}")
print(f"Actual std:       {eval_df['fc_coverage'].std():.4f}")

# The telltale sign: predicted values have less variance than actuals
shrinkage = 1 - eval_df['pred_D_log'].std() / eval_df['fc_coverage'].std()
print(f"\nShrinkage: {shrinkage:.1%} (predicted std is {shrinkage:.1%} smaller than actual)")
print("If positive → model is shrinking toward the mean → explains over/under pattern")

# Calibration plot: bin predictions, compare mean prediction vs mean actual
print("\n--- Calibration: binned predictions vs actuals ---")
print(f"{'Pred bin':>15} {'Mean pred':>10} {'Mean actual':>12} {'Gap':>8} {'N':>8} {'Direction'}")
print("-" * 65)

pred_bins = [(0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6),
             (0.6, 0.7), (0.7, 0.8), (0.8, 1.0), (1.0, 5.0)]
for lo, hi in pred_bins:
    bm = (eval_df['pred_D_log'] >= lo) & (eval_df['pred_D_log'] < hi)
    if bm.sum() >= 10:
        mp = eval_df.loc[bm, 'pred_D_log'].mean()
        ma = eval_df.loc[bm, 'fc_coverage'].mean()
        gap = ma - mp
        direction = "under-pred" if gap > 0.02 else "over-pred" if gap < -0.02 else "~calibrated"
        print(f"  {lo:.1f}-{hi:.1f}{'>':>8} {mp:>10.4f} {ma:>12.4f} {gap:>8.4f} {bm.sum():>8} {direction}")


# ══════════════════════════════════════════════════════════════════
# 2. FEATURE DISTRIBUTION BY COVERAGE LEVEL
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("2. FEATURE DISTRIBUTIONS BY COVERAGE LEVEL")
print("=" * 90)
print("Do low-coverage and high-coverage sites look different on inputs?\n")

eval_df['cov_bin'] = pd.cut(eval_df['fc_coverage'],
                             bins=[0, 0.3, 0.5, 0.7, 0.9, 5],
                             labels=['0-0.3', '0.3-0.5', '0.5-0.7', '0.7-0.9', '0.9+'])

print(f"{'Coverage':>10} {'hist_avg':>10} {'COO/SL1':>10} {'log(COO)':>10} {'OO/SL1':>10} "
      f"{'Earliness':>10} {'LeadTime':>10} {'N':>8}")
print("-" * 80)

for cat in ['0-0.3', '0.3-0.5', '0.5-0.7', '0.7-0.9', '0.9+']:
    sub = eval_df[eval_df['cov_bin'] == cat]
    if len(sub) < 10:
        continue
    print(f"{cat:>10} {sub['hist_avg'].mean():>10.4f} {sub['coo_sl1_ratio'].mean():>10.4f} "
          f"{np.log(sub['coo_sl1_ratio'] + 0.001).mean():>10.4f} "
          f"{sub['oo_sl1_ratio'].mean():>10.4f} "
          f"{sub['Avg_Weighted_Order_Earliness'].mean():>10.4f} "
          f"{sub['Avg_Weighted_Lead_Time'].mean():>10.4f} "
          f"{len(sub):>8}")


# ══════════════════════════════════════════════════════════════════
# 3. NONLINEARITY IN HIST_AVG → ACTUAL
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("3. NONLINEARITY: hist_avg vs actual coverage")
print("=" * 90)
print("Is the stale curve itself biased differently at different levels?\n")

print(f"{'hist_avg bin':>15} {'Mean hist':>10} {'Mean actual':>12} {'Gap':>8} {'N':>8}")
print("-" * 55)

for lo, hi in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, 5.0)]:
    bm = (eval_df['hist_avg'] >= lo) & (eval_df['hist_avg'] < hi)
    if bm.sum() >= 10:
        mh = eval_df.loc[bm, 'hist_avg'].mean()
        ma = eval_df.loc[bm, 'fc_coverage'].mean()
        print(f"  {lo:.1f}-{hi:.1f}{'>':>8} {mh:>10.4f} {ma:>12.4f} {ma-mh:>8.4f} {bm.sum():>8}")


# ══════════════════════════════════════════════════════════════════
# 4. WHAT PREDICTS THE RESIDUAL?
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("4. WHAT PREDICTS THE RESIDUAL?")
print("=" * 90)
print("If we can predict the residual, we can correct for it.\n")

resid = eval_df['resid']

# Test many features
candidates = {
    'pred_D_log': eval_df['pred_D_log'],
    'pred_D_log²': eval_df['pred_D_log'] ** 2,
    'hist_avg': eval_df['hist_avg'],
    'hist_avg²': eval_df['hist_avg'] ** 2,
    'coo_sl1_ratio': eval_df['coo_sl1_ratio'],
    'log(COO)': np.log(eval_df['coo_sl1_ratio'] + 0.001),
    'log(COO)²': np.log(eval_df['coo_sl1_ratio'] + 0.001) ** 2,
    'oo_sl1_ratio': eval_df['oo_sl1_ratio'],
    'OO_Coverage': eval_df['OO_Coverage'],
    'OO_Bias': eval_df['OO_Bias'],
    'Earliness': eval_df['Avg_Weighted_Order_Earliness'],
    'Lead_Time': eval_df['Avg_Weighted_Lead_Time'],
    'hist_avg * log(COO)': eval_df['hist_avg'] * np.log(eval_df['coo_sl1_ratio'] + 0.001),
    'pred - 0.5 (distance from center)': (eval_df['pred_D_log'] - 0.5).abs(),
}

print(f"{'Feature':<35} {'Spearman r':>12} {'p-value':>12} {'N':>8}")
print("-" * 70)

results = []
for name, vals in candidates.items():
    mask = vals.notna() & resid.notna()
    n = mask.sum()
    if n >= 20:
        sr, sp = stats.spearmanr(vals[mask], resid[mask])
        results.append((name, sr, sp, n))

results.sort(key=lambda x: abs(x[1]), reverse=True)
for name, sr, sp, n in results:
    print(f"{name:<35} {sr:>12.4f} {sp:>12.6f} {n:>8}")


# ══════════════════════════════════════════════════════════════════
# 5. CORRECTED MODELS
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("5. CORRECTED MODELS — can we fix the bias?")
print("=" * 90)

def evaluate(y_true, y_pred, label):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    mae = np.mean(np.abs(yt - yp))
    medae = np.median(np.abs(yt - yp))
    rmse = np.sqrt(np.mean((yt - yp)**2))
    ss_res = np.sum((yt - yp)**2)
    ss_tot = np.sum((yt - yt.mean())**2)
    r2 = 1 - ss_res / ss_tot
    return {'label': label, 'MAE': mae, 'MedAE': medae, 'RMSE': rmse, 'R²': r2, 'N': mask.sum()}

# Approach D_log (baseline for this analysis)
base = evaluate(eval_df['fc_coverage'].values, eval_df['pred_D_log'].values, 'D_log (baseline)')

# ── Correction 1: Add quadratic term on prediction ──
print("\n--- Correction 1: Add pred² (quadratic calibration) ---")
for lag in lags:
    lm = eval_df['Prediction_Lag'] == lag
    if lm.sum() < 30:
        continue
    X = np.column_stack([
        eval_df.loc[lm, 'hist_avg'].values,
        np.log(eval_df.loc[lm, 'coo_sl1_ratio'].values + 0.001),
        eval_df.loc[lm, 'hist_avg'].values ** 2,
        (np.log(eval_df.loc[lm, 'coo_sl1_ratio'].values + 0.001)) ** 2,
        np.ones(lm.sum())
    ])
    y = eval_df.loc[lm, 'fc_coverage'].values
    coefs, _, _, _ = lstsq(X, y, rcond=None)
    eval_df.loc[lm, 'pred_quad'] = X @ coefs
    if lag == 1:
        print(f"  Lag=1: {coefs[0]:.4f}*hist + {coefs[1]:.4f}*log(COO) "
              f"+ {coefs[2]:.4f}*hist² + {coefs[3]:.4f}*log(COO)² + {coefs[4]:.4f}")

# ── Correction 2: Interaction term ──
print("\n--- Correction 2: Add hist_avg * log(COO) interaction ---")
for lag in lags:
    lm = eval_df['Prediction_Lag'] == lag
    if lm.sum() < 30:
        continue
    ha = eval_df.loc[lm, 'hist_avg'].values
    lc = np.log(eval_df.loc[lm, 'coo_sl1_ratio'].values + 0.001)
    X = np.column_stack([ha, lc, ha * lc, np.ones(lm.sum())])
    y = eval_df.loc[lm, 'fc_coverage'].values
    coefs, _, _, _ = lstsq(X, y, rcond=None)
    eval_df.loc[lm, 'pred_interact'] = X @ coefs
    if lag == 1:
        print(f"  Lag=1: {coefs[0]:.4f}*hist + {coefs[1]:.4f}*log(COO) "
              f"+ {coefs[2]:.4f}*hist*log(COO) + {coefs[3]:.4f}")

# ── Correction 3: Quadratic + interaction ──
print("\n--- Correction 3: Quadratic + interaction ---")
for lag in lags:
    lm = eval_df['Prediction_Lag'] == lag
    if lm.sum() < 30:
        continue
    ha = eval_df.loc[lm, 'hist_avg'].values
    lc = np.log(eval_df.loc[lm, 'coo_sl1_ratio'].values + 0.001)
    X = np.column_stack([ha, lc, ha**2, lc**2, ha * lc, np.ones(lm.sum())])
    y = eval_df.loc[lm, 'fc_coverage'].values
    coefs, _, _, _ = lstsq(X, y, rcond=None)
    eval_df.loc[lm, 'pred_full'] = X @ coefs
    if lag == 1:
        print(f"  Lag=1: {coefs[0]:.4f}*hist + {coefs[1]:.4f}*log(COO) "
              f"+ {coefs[2]:.4f}*hist² + {coefs[3]:.4f}*log(COO)² "
              f"+ {coefs[4]:.4f}*hist*log(COO) + {coefs[5]:.4f}")

# ── Correction 4: Full model F with quadratic terms ──
print("\n--- Correction 4: Full F model + quadratic + interaction ---")
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
    ha = eval_df.loc[idx, 'hist_avg'].values
    lc = np.log(eval_df.loc[idx, 'coo_sl1_ratio'].values + 0.001)
    X = np.column_stack([
        ha, lc, ha**2, lc**2, ha * lc,
        eval_df.loc[idx, 'oo_sl1_ratio'].values,
        eval_df.loc[idx, 'Avg_Weighted_Order_Earliness'].values,
        eval_df.loc[idx, 'Avg_Weighted_Lead_Time'].values,
        np.ones(len(idx))
    ])
    y = eval_df.loc[idx, 'fc_coverage'].values
    coefs, _, _, _ = lstsq(X, y, rcond=None)
    eval_df.loc[idx, 'pred_F_quad'] = X @ coefs

# ── Compare all ───────────────────────────────────────────────────
print("\n--- Overall comparison ---")
print(f"{'Approach':<50} {'MAE':>8} {'MedAE':>8} {'RMSE':>8} {'R²':>8} {'N':>8}")
print("-" * 90)

y_true = eval_df['fc_coverage'].values
for label, col in [
    ('A: Stale curve only', 'hist_avg'),
    ('D_log: Stale + log(COO)', 'pred_D_log'),
    ('+ Quadratic terms', 'pred_quad'),
    ('+ Interaction term', 'pred_interact'),
    ('+ Quad + interaction', 'pred_full'),
    ('F_quad: Full + quad + interaction', 'pred_F_quad'),
]:
    if col not in eval_df.columns:
        continue
    s = evaluate(y_true, eval_df[col].values, label)
    if s:
        print(f"{s['label']:<50} {s['MAE']:>8.4f} {s['MedAE']:>8.4f} "
              f"{s['RMSE']:>8.4f} {s['R²']:>8.4f} {s['N']:>8}")

# ── Per-lag breakdown of best vs baseline ─────────────────────────
print("\n--- Per-lag: D_log vs best corrected model ---")
print(f"{'Lag':>5} {'A:Stale':>10} {'D_log':>10} {'Quad+Int':>10} {'F_quad':>10} {'F_quad vs A':>14}")
print("-" * 65)

for lag in lags:
    lm = eval_df['Prediction_Lag'] == lag
    if lm.sum() < 30:
        continue
    yt = eval_df.loc[lm, 'fc_coverage'].values
    maes = {}
    for key, col in [('A', 'hist_avg'), ('D_log', 'pred_D_log'),
                      ('QI', 'pred_full'), ('Fq', 'pred_F_quad')]:
        if col in eval_df.columns:
            p = eval_df.loc[lm, col].values
            m = ~np.isnan(p) & ~np.isnan(yt)
            if m.sum() > 0:
                maes[key] = np.mean(np.abs(yt[m] - p[m]))
    a = maes.get('A', np.nan)
    d = maes.get('D_log', np.nan)
    qi = maes.get('QI', np.nan)
    fq = maes.get('Fq', np.nan)
    improv = (a - fq) / a * 100 if a > 0 and not np.isnan(fq) else np.nan
    print(f"{lag:>5} {a:>10.4f} {d:>10.4f} {qi:>10.4f} {fq:>10.4f} {improv:>13.1f}%")


# ── Calibration of best model ─────────────────────────────────────
print("\n--- Calibration of F_quad (best model) ---")
print(f"{'Pred bin':>15} {'Mean pred':>10} {'Mean actual':>12} {'Gap':>8} {'N':>8}")
print("-" * 55)

best_col = 'pred_F_quad'
if best_col in eval_df.columns:
    for lo, hi in pred_bins:
        bm = eval_df[best_col].notna() & (eval_df[best_col] >= lo) & (eval_df[best_col] < hi)
        if bm.sum() >= 10:
            mp = eval_df.loc[bm, best_col].mean()
            ma = eval_df.loc[bm, 'fc_coverage'].mean()
            gap = ma - mp
            direction = "under" if gap > 0.02 else "over" if gap < -0.02 else "~ok"
            print(f"  {lo:.1f}-{hi:.1f}{'>':>8} {mp:>10.4f} {ma:>12.4f} {gap:>8.4f} {bm.sum():>8} {direction}")

# ── Residual by coverage level for best model ─────────────────────
print(f"\n--- Residual by actual coverage level (F_quad) ---")
print(f"{'Coverage bin':>15} {'Mean resid':>12} {'Std':>8} {'N':>8} {'Bias'}")
print("-" * 55)

if best_col in eval_df.columns:
    eval_df['resid_best'] = eval_df['fc_coverage'] - eval_df[best_col]
    valid = eval_df['resid_best'].notna()
    bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, 5)]
    for lo, hi in bins:
        bm = valid & (eval_df['fc_coverage'] >= lo) & (eval_df['fc_coverage'] < hi)
        if bm.sum() >= 10:
            r = eval_df.loc[bm, 'resid_best']
            bias = "UNDER" if r.mean() > 0.03 else "OVER" if r.mean() < -0.03 else "~ok"
            print(f"  {lo:.1f}-{hi:.1f}{'>':>8} {r.mean():>12.4f} {r.std():>8.4f} {bm.sum():>8} {bias}")


print("\n" + "=" * 90)
print("ANALYSIS COMPLETE")
print("=" * 90)
