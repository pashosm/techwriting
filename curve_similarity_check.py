#!/usr/bin/env python3
"""
Curve Similarity Feasibility Check
====================================
Question: Can proxy features (available at receipt time) predict which sites
have similar forecast_coverage and forecast_bias aging curves?

Step 1: Build each site's aging curve as a vector over (Timeframe, Lag).
Step 2: Compute pairwise site similarity on those curves (ground truth).
Step 3: Compute pairwise site similarity on proxy features (summary stats).
Step 4: Check whether proxy similarity predicts curve similarity.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from itertools import combinations

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 140)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')

# ── 1. Load & compute derived columns ─────────────────────────────
CSV = "/home/user/techwriting/Dummy Training Data Customer 1 Try 2 11-Feb-2026.csv"
df = pd.read_csv(CSV)
df['Reference_Month'] = pd.to_datetime(df['Reference_Month'])

df['fc_coverage'] = np.where(
    df['Actual_Sales'] > 0,
    df['Covered_Orders'] / df['Actual_Sales'], np.nan)
df['fc_bias'] = np.where(
    df['Covered_Orders'] > 0,
    df['Forecast_Value'] / df['Covered_Orders'], np.nan)
df['oo_ratio'] = np.where(
    df['Actual_Sales'] > 0,
    df['Open_Orders'] / df['Actual_Sales'], np.nan)

print("=" * 90)
print("CURVE SIMILARITY FEASIBILITY CHECK")
print("=" * 90)

# ── 2. Build aging curve vectors per site ──────────────────────────
# Each site's "curve" = mean fc_coverage and fc_bias at each (Timeframe, Lag)

# Grid of all (TF, Lag) combos
tfs = sorted(df['Timeframe'].unique())
lags = sorted(df['Prediction_Lag'].unique())
grid = [(tf, lag) for tf in tfs for lag in lags]

sites = sorted(df['Site'].unique())

print(f"\nSites: {sites}")
print(f"Grid: {len(tfs)} timeframes x {len(lags)} lags = {len(grid)} points per curve")

# Build coverage curves
cov_curves = {}
bias_curves = {}
for site in sites:
    sd = df[df['Site'] == site]
    cov_vec = []
    bias_vec = []
    for tf, lag in grid:
        cell = sd[(sd['Timeframe'] == tf) & (sd['Prediction_Lag'] == lag)]
        cov_vec.append(cell['fc_coverage'].mean())  # NaN if no data
        bias_vec.append(cell['fc_bias'].mean())
    cov_curves[site] = np.array(cov_vec)
    bias_curves[site] = np.array(bias_vec)

# Report coverage of each site's curve vector
print("\n--- Curve vector completeness ---")
print(f"{'Site':<25} {'Coverage pts':>14} {'Bias pts':>14} {'Total':>8}")
print("-" * 65)
for site in sites:
    cov_ok = np.sum(~np.isnan(cov_curves[site]))
    bias_ok = np.sum(~np.isnan(bias_curves[site]))
    print(f"{site:<25} {cov_ok:>10}/{len(grid):<4} {bias_ok:>10}/{len(grid):<4} {len(grid):>6}")

# ── 3. Pairwise curve similarity ──────────────────────────────────
# Use correlation on the curve vectors (handles scale differences).
# For each pair, compute correlation over the grid points where BOTH
# sites have non-NaN values.

print("\n" + "=" * 90)
print("SECTION 1: PAIRWISE CURVE SIMILARITY (Ground Truth)")
print("=" * 90)


def curve_similarity(vec_a, vec_b, metric='correlation'):
    """Correlation between two curve vectors, using only shared non-NaN points."""
    mask = ~np.isnan(vec_a) & ~np.isnan(vec_b)
    n = mask.sum()
    if n < 5:
        return np.nan, n
    a, b = vec_a[mask], vec_b[mask]
    if metric == 'correlation':
        r, _ = stats.pearsonr(a, b)
        return r, n
    elif metric == 'euclidean':
        # Normalize first so scale doesn't dominate
        a_n = (a - a.mean()) / (a.std() + 1e-9)
        b_n = (b - b.mean()) / (b.std() + 1e-9)
        return -np.sqrt(np.mean((a_n - b_n) ** 2)), n  # negative so higher = more similar


# Coverage similarity
print("\n--- Coverage curve similarity (Pearson r) ---")
print(f"{'Pair':<50} {'r':>8} {'N pts':>8}")
print("-" * 70)
cov_sim_pairs = []
for s1, s2 in combinations(sites, 2):
    r, n = curve_similarity(cov_curves[s1], cov_curves[s2])
    cov_sim_pairs.append((s1, s2, r, n))
    rstr = f"{r:.4f}" if not np.isnan(r) else "   NaN"
    print(f"{s1} — {s2:<25} {rstr:>8} {n:>8}")

# Bias similarity
print("\n--- Bias curve similarity (Pearson r) ---")
print(f"{'Pair':<50} {'r':>8} {'N pts':>8}")
print("-" * 70)
bias_sim_pairs = []
for s1, s2 in combinations(sites, 2):
    r, n = curve_similarity(bias_curves[s1], bias_curves[s2])
    bias_sim_pairs.append((s1, s2, r, n))
    rstr = f"{r:.4f}" if not np.isnan(r) else "   NaN"
    print(f"{s1} — {s2:<25} {rstr:>8} {n:>8}")

# Combined similarity (average of coverage and bias correlations)
print("\n--- Combined curve similarity (avg of coverage r + bias r) ---")
print(f"{'Pair':<50} {'Cov r':>8} {'Bias r':>8} {'Combined':>10}")
print("-" * 80)
combined_sim = {}
for (s1, s2, cr, cn), (_, _, br, bn) in zip(cov_sim_pairs, bias_sim_pairs):
    if not np.isnan(cr) and not np.isnan(br):
        combo = (cr + br) / 2
    elif not np.isnan(cr):
        combo = cr  # coverage only
    else:
        combo = np.nan
    combined_sim[(s1, s2)] = combo
    crstr = f"{cr:.4f}" if not np.isnan(cr) else "   NaN"
    brstr = f"{br:.4f}" if not np.isnan(br) else "   NaN"
    costr = f"{combo:.4f}" if not np.isnan(combo) else "   NaN"
    print(f"{s1} — {s2:<25} {crstr:>8} {brstr:>8} {costr:>10}")

# ── 4. Proxy feature summary stats per site ────────────────────────
print("\n" + "=" * 90)
print("SECTION 2: PROXY FEATURE SUMMARY STATISTICS")
print("=" * 90)

proxy_features = {}
for site in sites:
    sd = df[df['Site'] == site]
    proxy_features[site] = {
        'mean_sales_lag1': sd['Historical_Sales_Lag1'].mean(),
        'mean_sales_lag12': sd['Historical_Sales_Lag12'].mean(),
        'mean_forecast_value': sd['Forecast_Value'].mean(),
        'mean_open_orders': sd['Open_Orders'].mean(),
        'mean_oo_ratio': sd['oo_ratio'].mean(),
        'mean_fc_coverage': sd['fc_coverage'].mean(),
        'mean_fc_bias': sd['fc_bias'].mean() if sd['fc_bias'].notna().sum() > 0 else np.nan,
        # Also try ratios/normalized features
        'oo_to_sales_ratio': sd['Open_Orders'].sum() / sd['Historical_Sales_Lag1'].sum()
            if sd['Historical_Sales_Lag1'].sum() > 0 else np.nan,
        'fc_to_sales_ratio': sd['Forecast_Value'].sum() / sd['Historical_Sales_Lag1'].sum()
            if sd['Historical_Sales_Lag1'].sum() > 0 else np.nan,
    }

proxy_df = pd.DataFrame(proxy_features).T
print("\nProxy feature summary per site:")
print(proxy_df.to_string())

# ── 5. Pairwise proxy similarity ──────────────────────────────────
print("\n" + "=" * 90)
print("SECTION 3: DOES PROXY SIMILARITY PREDICT CURVE SIMILARITY?")
print("=" * 90)

# For each proxy feature, compute pairwise distance (absolute difference
# of the summary stat), then correlate with curve similarity.
# Use rank correlation (Spearman) since we care about ordering, not linearity.

proxy_cols = list(proxy_df.columns)

# Build vectors: one entry per site pair, aligned
pair_labels = []
curve_sim_vec = []      # combined curve similarity
cov_sim_vec = []        # coverage-only
for s1, s2 in combinations(sites, 2):
    pair_labels.append(f"{s1} — {s2}")
    curve_sim_vec.append(combined_sim.get((s1, s2), np.nan))
    # Also extract coverage-only
    match = [x for x in cov_sim_pairs if x[0] == s1 and x[1] == s2]
    cov_sim_vec.append(match[0][2] if match else np.nan)

curve_sim_vec = np.array(curve_sim_vec)
cov_sim_vec = np.array(cov_sim_vec)

print("\n--- Correlation between proxy distance and COVERAGE curve similarity ---")
print(f"{'Proxy feature':<25} {'Spearman r':>12} {'p-value':>10} {'N pairs':>10} {'Interpretation'}")
print("-" * 85)

for col in proxy_cols:
    # Pairwise absolute difference for this proxy
    proxy_dist = []
    for s1, s2 in combinations(sites, 2):
        v1 = proxy_df.loc[s1, col]
        v2 = proxy_df.loc[s2, col]
        proxy_dist.append(abs(v1 - v2))
    proxy_dist = np.array(proxy_dist)

    # We want NEGATIVE correlation: smaller proxy distance → higher curve similarity
    # But let's just report the raw correlation and interpret
    mask = ~np.isnan(proxy_dist) & ~np.isnan(cov_sim_vec)
    n = mask.sum()
    if n >= 5:
        r, p = stats.spearmanr(proxy_dist[mask], cov_sim_vec[mask])
        # Negative r means: closer proxy values → more similar curves (good!)
        interp = "PROMISING" if (r < -0.3 and p < 0.1) else \
                 "weak signal" if (r < -0.2) else "no signal"
        print(f"{col:<25} {r:>12.4f} {p:>10.4f} {n:>10} {interp}")
    else:
        print(f"{col:<25} {'insufficient data':>12} {'':>10} {n:>10}")

print("\n--- Correlation between proxy distance and COMBINED curve similarity ---")
print(f"{'Proxy feature':<25} {'Spearman r':>12} {'p-value':>10} {'N pairs':>10} {'Interpretation'}")
print("-" * 85)

for col in proxy_cols:
    proxy_dist = []
    for s1, s2 in combinations(sites, 2):
        v1 = proxy_df.loc[s1, col]
        v2 = proxy_df.loc[s2, col]
        proxy_dist.append(abs(v1 - v2))
    proxy_dist = np.array(proxy_dist)

    mask = ~np.isnan(proxy_dist) & ~np.isnan(curve_sim_vec)
    n = mask.sum()
    if n >= 5:
        r, p = stats.spearmanr(proxy_dist[mask], curve_sim_vec[mask])
        interp = "PROMISING" if (r < -0.3 and p < 0.1) else \
                 "weak signal" if (r < -0.2) else "no signal"
        print(f"{col:<25} {r:>12.4f} {p:>10.4f} {n:>10} {interp}")
    else:
        print(f"{col:<25} {'insufficient data':>12} {'':>10} {n:>10}")

# ── 6. Multi-feature proxy similarity ──────────────────────────────
print("\n" + "=" * 90)
print("SECTION 4: MULTI-FEATURE PROXY SIMILARITY")
print("=" * 90)
print("Using normalized Euclidean distance across all proxy features simultaneously.")

# Use only features with full coverage across sites
usable_cols = [c for c in proxy_cols if proxy_df[c].notna().all()]
print(f"Usable proxy features (non-NaN for all sites): {usable_cols}")

if len(usable_cols) >= 2:
    # Standardize
    proxy_std = proxy_df[usable_cols].copy()
    for c in usable_cols:
        mu, sigma = proxy_std[c].mean(), proxy_std[c].std()
        if sigma > 0:
            proxy_std[c] = (proxy_std[c] - mu) / sigma

    # Pairwise Euclidean distance
    multi_proxy_dist = []
    for s1, s2 in combinations(sites, 2):
        d = np.sqrt(((proxy_std.loc[s1] - proxy_std.loc[s2]) ** 2).sum())
        multi_proxy_dist.append(d)
    multi_proxy_dist = np.array(multi_proxy_dist)

    # Correlate with curve similarity
    for label, sim_vec in [("Coverage", cov_sim_vec), ("Combined", curve_sim_vec)]:
        mask = ~np.isnan(multi_proxy_dist) & ~np.isnan(sim_vec)
        n = mask.sum()
        if n >= 5:
            r, p = stats.spearmanr(multi_proxy_dist[mask], sim_vec[mask])
            print(f"\n  Multi-proxy distance vs {label} curve similarity:")
            print(f"    Spearman r = {r:.4f}, p = {p:.4f}, N = {n}")
            print(f"    {'PROMISING' if (r < -0.3 and p < 0.1) else 'weak/no signal'}")

# ── 7. Ratio-only proxy (scale-invariant) ──────────────────────────
print("\n" + "=" * 90)
print("SECTION 5: RATIO-ONLY PROXY SIMILARITY (scale-invariant)")
print("=" * 90)
print("Testing whether ratio-based features work better than absolute values.")

ratio_cols = [c for c in ['mean_oo_ratio', 'mean_fc_coverage', 'oo_to_sales_ratio',
                           'fc_to_sales_ratio'] if c in usable_cols]
print(f"Ratio features: {ratio_cols}")

if len(ratio_cols) >= 1:
    proxy_ratio_std = proxy_df[ratio_cols].copy()
    for c in ratio_cols:
        mu, sigma = proxy_ratio_std[c].mean(), proxy_ratio_std[c].std()
        if sigma > 0:
            proxy_ratio_std[c] = (proxy_ratio_std[c] - mu) / sigma

    ratio_dist = []
    for s1, s2 in combinations(sites, 2):
        d = np.sqrt(((proxy_ratio_std.loc[s1] - proxy_ratio_std.loc[s2]) ** 2).sum())
        ratio_dist.append(d)
    ratio_dist = np.array(ratio_dist)

    for label, sim_vec in [("Coverage", cov_sim_vec), ("Combined", curve_sim_vec)]:
        mask = ~np.isnan(ratio_dist) & ~np.isnan(sim_vec)
        n = mask.sum()
        if n >= 5:
            r, p = stats.spearmanr(ratio_dist[mask], sim_vec[mask])
            print(f"\n  Ratio-proxy distance vs {label} curve similarity:")
            print(f"    Spearman r = {r:.4f}, p = {p:.4f}, N = {n}")
            print(f"    {'PROMISING' if (r < -0.3 and p < 0.1) else 'weak/no signal'}")

# ── 8. Scatter data for visual inspection ──────────────────────────
print("\n" + "=" * 90)
print("SECTION 6: PAIR-LEVEL DETAIL (for visual inspection)")
print("=" * 90)

# Show the most and least similar site pairs on curves, and their proxy distances
print("\n--- All pairs ranked by coverage curve similarity ---")
print(f"{'Pair':<45} {'Cov r':>8} {'Bias r':>8} {'OO_ratio diff':>15} {'FC_cov diff':>14}")
print("-" * 95)

ranked = sorted(zip(
    [f"{s1} — {s2}" for s1, s2 in combinations(sites, 2)],
    cov_sim_vec,
    [combined_sim.get((s1, s2), np.nan) for s1, s2 in combinations(sites, 2)],
    combinations(sites, 2)
), key=lambda x: x[1] if not np.isnan(x[1]) else -99, reverse=True)

for label, cov_r, combo_r, (s1, s2) in ranked:
    oo_diff = abs(proxy_df.loc[s1, 'mean_oo_ratio'] - proxy_df.loc[s2, 'mean_oo_ratio'])
    fc_diff = abs(proxy_df.loc[s1, 'mean_fc_coverage'] - proxy_df.loc[s2, 'mean_fc_coverage'])
    bias_r_val = [x[2] for x in bias_sim_pairs if x[0] == s1 and x[1] == s2]
    br = bias_r_val[0] if bias_r_val else np.nan
    crstr = f"{cov_r:.4f}" if not np.isnan(cov_r) else "   NaN"
    brstr = f"{br:.4f}" if not np.isnan(br) else "   NaN"
    print(f"{label:<45} {crstr:>8} {brstr:>8} {oo_diff:>15.4f} {fc_diff:>14.4f}")

print("\n" + "=" * 90)
print("INTERPRETATION")
print("=" * 90)
print("""
If Spearman r is strongly NEGATIVE (< -0.3) and significant (p < 0.1):
  → Sites with similar proxy features tend to have similar aging curves.
  → This supports the idea that proxy features can identify "similar" sites
     at the point of receipt.

If Spearman r is near zero or positive:
  → Proxy feature similarity does NOT predict curve similarity.
  → The similar-site approach may not be viable with these features.

Note: With only 9 sites (36 pairs), statistical power is limited.
This is a directional check, not a definitive proof.
""")

print("=" * 90)
print("ANALYSIS COMPLETE")
print("=" * 90)
