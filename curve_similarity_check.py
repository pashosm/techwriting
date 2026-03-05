#!/usr/bin/env python3
"""
Curve Similarity Feasibility Check (v2)
========================================
Question: Can proxy features OBSERVABLE AT FORECAST TIME predict which sites
have similar forecast_coverage and forecast_bias aging curves?

Key constraint: NO features that require Actual_Sales (only known after the fact).
Observable fields: Historical_Sales_Lag1, Historical_Sales_Lag12, Open_Orders,
                   Forecast_Value, Covered_Orders — plus ratios of these.

Filtered to TF=12 only. Curves are 12-point vectors across Lags 1–12.
"""

import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 140)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')

# ── 1. Load & filter ──────────────────────────────────────────────
CSV = "/home/user/techwriting/Dummy Training Data Customer 1 Try 2 11-Feb-2026.csv"
df = pd.read_csv(CSV)
df['Reference_Month'] = pd.to_datetime(df['Reference_Month'])

# Filter to TF=12 only
df = df[df['Timeframe'] == 12].copy()

# Ground truth metrics (use Actual_Sales ONLY for the curve we're trying to predict)
df['fc_coverage'] = np.where(
    df['Actual_Sales'] > 0,
    df['Covered_Orders'] / df['Actual_Sales'], np.nan)
df['fc_bias'] = np.where(
    df['Covered_Orders'] > 0,
    df['Forecast_Value'] / df['Covered_Orders'], np.nan)

sites = sorted(df['Site'].unique())
lags = sorted(df['Prediction_Lag'].unique())

print("=" * 90)
print("CURVE SIMILARITY FEASIBILITY CHECK (v2)")
print("Filtered to TF=12 | Observable-at-forecast-time proxies only")
print("=" * 90)
print(f"\nSites: {sites}")
print(f"Curve: {len(lags)} lags (1–12) at TF=12")

# ── 2. Build aging curve vectors per site ──────────────────────────
cov_curves = {}
bias_curves = {}
for site in sites:
    sd = df[df['Site'] == site]
    cov_vec = []
    bias_vec = []
    for lag in lags:
        cell = sd[sd['Prediction_Lag'] == lag]
        cov_vec.append(cell['fc_coverage'].mean())
        bias_vec.append(cell['fc_bias'].mean())
    cov_curves[site] = np.array(cov_vec)
    bias_curves[site] = np.array(bias_vec)

print("\n--- Curve vector completeness ---")
print(f"{'Site':<25} {'Coverage pts':>14} {'Bias pts':>14}")
print("-" * 55)
for site in sites:
    cov_ok = np.sum(~np.isnan(cov_curves[site]))
    bias_ok = np.sum(~np.isnan(bias_curves[site]))
    print(f"{site:<25} {cov_ok:>10}/{len(lags):<4} {bias_ok:>10}/{len(lags):<4}")


# ── 3. Pairwise curve similarity ──────────────────────────────────
def curve_similarity(vec_a, vec_b):
    """Pearson r between two curve vectors, using shared non-NaN points."""
    mask = ~np.isnan(vec_a) & ~np.isnan(vec_b)
    n = mask.sum()
    if n < 5:
        return np.nan, n
    a, b = vec_a[mask], vec_b[mask]
    if a.std() == 0 or b.std() == 0:
        return np.nan, n
    r, _ = stats.pearsonr(a, b)
    return r, n


print("\n" + "=" * 90)
print("SECTION 1: PAIRWISE CURVE SIMILARITY (Ground Truth)")
print("=" * 90)

print("\n--- Coverage curve similarity (Pearson r) ---")
print(f"{'Pair':<50} {'r':>8} {'N pts':>8}")
print("-" * 70)
cov_sim_pairs = []
for s1, s2 in combinations(sites, 2):
    r, n = curve_similarity(cov_curves[s1], cov_curves[s2])
    cov_sim_pairs.append((s1, s2, r, n))
    rstr = f"{r:.4f}" if not np.isnan(r) else "   NaN"
    print(f"{s1} — {s2:<25} {rstr:>8} {n:>8}")

print("\n--- Bias curve similarity (Pearson r) ---")
print(f"{'Pair':<50} {'r':>8} {'N pts':>8}")
print("-" * 70)
bias_sim_pairs = []
for s1, s2 in combinations(sites, 2):
    r, n = curve_similarity(bias_curves[s1], bias_curves[s2])
    bias_sim_pairs.append((s1, s2, r, n))
    rstr = f"{r:.4f}" if not np.isnan(r) else "   NaN"
    print(f"{s1} — {s2:<25} {rstr:>8} {n:>8}")

# Combined
combined_sim = {}
for (s1, s2, cr, cn), (_, _, br, bn) in zip(cov_sim_pairs, bias_sim_pairs):
    if not np.isnan(cr) and not np.isnan(br):
        combo = (cr + br) / 2
    elif not np.isnan(cr):
        combo = cr
    else:
        combo = np.nan
    combined_sim[(s1, s2)] = combo

# ── 4. Observable-at-forecast-time proxy features ──────────────────
# Raw fields: Historical_Sales_Lag1, Historical_Sales_Lag12, Open_Orders,
#             Forecast_Value, Covered_Orders
# Derived ratios (all from observable fields — NO Actual_Sales):
#   OO_coverage  = Covered_Orders / Sales_Lag1
#   OO_bias      = Forecast_Value / Covered_Orders
#   OO_FC_ratio  = Open_Orders / Forecast_Value
#   FC_Sales     = Forecast_Value / Sales_Lag1
#   OO_Sales     = Open_Orders / Sales_Lag1
#   CO_Sales     = Covered_Orders / Sales_Lag1

print("\n" + "=" * 90)
print("SECTION 2: OBSERVABLE PROXY FEATURES (no Actual_Sales)")
print("=" * 90)

proxy_features = {}
for site in sites:
    sd = df[df['Site'] == site]

    sales1 = sd['Historical_Sales_Lag1']
    sales12 = sd['Historical_Sales_Lag12']
    oo = sd['Open_Orders']
    fc = sd['Forecast_Value']
    co = sd['Covered_Orders']

    def safe_ratio(num, denom):
        """Mean of element-wise ratio where denom > 0."""
        mask = denom > 0
        if mask.sum() == 0:
            return np.nan
        return (num[mask] / denom[mask]).mean()

    proxy_features[site] = {
        # Raw means
        'sales_lag1': sales1.mean(),
        'sales_lag12': sales12.mean(),
        'open_orders': oo.mean(),
        'forecast_value': fc.mean(),
        'covered_orders': co.mean(),
        # Ratios (all observable at forecast time)
        'OO_cov (CO/SL1)': safe_ratio(co, sales1),
        'OO_bias (FC/CO)': safe_ratio(fc, co),
        'OO/FC': safe_ratio(oo, fc),
        'FC/SL1': safe_ratio(fc, sales1),
        'OO/SL1': safe_ratio(oo, sales1),
        'CO/SL12': safe_ratio(co, sales12),
    }

proxy_df = pd.DataFrame(proxy_features).T
print("\nProxy feature summary per site:")
print(proxy_df.to_string())

# ── 5. Correlate each proxy with curve similarity ──────────────────
print("\n" + "=" * 90)
print("SECTION 3: DOES PROXY SIMILARITY PREDICT CURVE SIMILARITY?")
print("=" * 90)
print("Spearman r between |proxy_A - proxy_B| and curve similarity(A, B)")
print("Negative r = sites with similar proxy have similar curves (good)")

# Build pairwise vectors
cov_sim_vec = []
combined_sim_vec = []
for s1, s2 in combinations(sites, 2):
    match_cov = [x[2] for x in cov_sim_pairs if x[0] == s1 and x[1] == s2]
    cov_sim_vec.append(match_cov[0] if match_cov else np.nan)
    combined_sim_vec.append(combined_sim.get((s1, s2), np.nan))
cov_sim_vec = np.array(cov_sim_vec)
combined_sim_vec = np.array(combined_sim_vec)

for target_label, sim_vec in [("COVERAGE curve", cov_sim_vec),
                                ("COMBINED curve", combined_sim_vec)]:
    print(f"\n--- vs {target_label} similarity ---")
    print(f"{'Proxy feature':<25} {'Spearman r':>12} {'p-value':>10} {'N pairs':>10} {'Signal?'}")
    print("-" * 75)

    for col in proxy_df.columns:
        proxy_dist = []
        for s1, s2 in combinations(sites, 2):
            v1 = proxy_df.loc[s1, col]
            v2 = proxy_df.loc[s2, col]
            proxy_dist.append(abs(v1 - v2))
        proxy_dist = np.array(proxy_dist)

        mask = ~np.isnan(proxy_dist) & ~np.isnan(sim_vec)
        n = mask.sum()
        if n >= 5:
            r, p = stats.spearmanr(proxy_dist[mask], sim_vec[mask])
            if r < -0.3 and p < 0.1:
                sig = "PROMISING"
            elif r < -0.2:
                sig = "weak"
            else:
                sig = "—"
            print(f"{col:<25} {r:>12.4f} {p:>10.4f} {n:>10} {sig}")
        else:
            print(f"{col:<25} {'insuff.':>12} {'':>10} {n:>10}")

# ── 6. Multi-feature: ratio-only composite ─────────────────────────
print("\n" + "=" * 90)
print("SECTION 4: COMPOSITE RATIO PROXY (all ratios combined)")
print("=" * 90)

ratio_cols = ['OO_cov (CO/SL1)', 'OO_bias (FC/CO)', 'OO/FC', 'FC/SL1', 'OO/SL1', 'CO/SL12']
usable_ratio = [c for c in ratio_cols if proxy_df[c].notna().all()]
print(f"Usable ratio features: {usable_ratio}")

if len(usable_ratio) >= 2:
    proxy_std = proxy_df[usable_ratio].copy()
    for c in usable_ratio:
        mu, sigma = proxy_std[c].mean(), proxy_std[c].std()
        if sigma > 0:
            proxy_std[c] = (proxy_std[c] - mu) / sigma

    ratio_dist = []
    for s1, s2 in combinations(sites, 2):
        d = np.sqrt(((proxy_std.loc[s1] - proxy_std.loc[s2]) ** 2).sum())
        ratio_dist.append(d)
    ratio_dist = np.array(ratio_dist)

    for label, sim_vec in [("Coverage", cov_sim_vec), ("Combined", combined_sim_vec)]:
        mask = ~np.isnan(ratio_dist) & ~np.isnan(sim_vec)
        n = mask.sum()
        if n >= 5:
            r, p = stats.spearmanr(ratio_dist[mask], sim_vec[mask])
            print(f"\n  All-ratio composite vs {label} curve similarity:")
            print(f"    Spearman r = {r:.4f}, p = {p:.4f}, N = {n}")
            print(f"    {'PROMISING' if (r < -0.3 and p < 0.1) else 'weak/no signal'}")

# ── 7. Pair-level detail ──────────────────────────────────────────
print("\n" + "=" * 90)
print("SECTION 5: PAIR-LEVEL DETAIL")
print("=" * 90)
print(f"{'Pair':<42} {'Cov r':>7} {'Bias r':>7} {'OO_cov diff':>12} {'OO_bias diff':>13} {'OO/FC diff':>11}")
print("-" * 95)

ranked = sorted(
    [(s1, s2) for s1, s2 in combinations(sites, 2)],
    key=lambda x: combined_sim.get(x, -99) if not np.isnan(combined_sim.get(x, np.nan)) else -99,
    reverse=True
)

for s1, s2 in ranked:
    cr = [x[2] for x in cov_sim_pairs if x[0] == s1 and x[1] == s2][0]
    br = [x[2] for x in bias_sim_pairs if x[0] == s1 and x[1] == s2][0]

    oo_cov_d = abs(proxy_df.loc[s1, 'OO_cov (CO/SL1)'] - proxy_df.loc[s2, 'OO_cov (CO/SL1)'])
    oo_bias_d = abs(proxy_df.loc[s1, 'OO_bias (FC/CO)'] - proxy_df.loc[s2, 'OO_bias (FC/CO)'])
    oo_fc_d = abs(proxy_df.loc[s1, 'OO/FC'] - proxy_df.loc[s2, 'OO/FC'])

    def fmt(v):
        return f"{v:.4f}" if not np.isnan(v) else "   NaN"

    print(f"{s1} — {s2:<20} {fmt(cr):>7} {fmt(br):>7} {fmt(oo_cov_d):>12} {fmt(oo_bias_d):>13} {fmt(oo_fc_d):>11}")

print("\n" + "=" * 90)
print("INTERPRETATION")
print("=" * 90)
print("""
ALL proxy features use only data observable at forecast time:
  - Historical_Sales_Lag1/12 (prior month sales)
  - Open_Orders, Forecast_Value, Covered_Orders (current forecast data)
  - Ratios derived from the above (NO Actual_Sales)

The ground truth (fc_coverage, fc_bias curves) uses Actual_Sales — that's
what we're trying to predict. The question is whether the observable proxies
can identify which sites will have similar curves.

Negative Spearman r (< -0.3, p < 0.1) = sites with similar proxy values
tend to have similar aging curves → approach is viable.
""")
print("=" * 90)
print("ANALYSIS COMPLETE")
print("=" * 90)
