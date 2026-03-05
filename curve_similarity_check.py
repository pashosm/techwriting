#!/usr/bin/env python3
"""
Curve Similarity Feasibility Check (v3 — full anonymized dataset)
==================================================================
Question: Can proxy features OBSERVABLE AT FORECAST TIME predict which sites
have similar forecast_coverage and forecast_bias aging curves?

Dataset: training_data_anonymized.csv (160 sites, 24 customers, 420K rows)
Filter:  TF=12 only. Curves are 12-point vectors across Lags 1–12.

Observable fields (no Actual_Sales):
  Raw:    Historical_Sales_Lag1, Historical_Sales_Lag12, Open_Orders,
          Forecast_Value, Covered_Orders, Covered_Open_Orders
  Pre-computed: OO_Coverage (= Covered_Open_Orders / Open_Orders),
                OO_Bias (= Forecast_Value / Covered_Open_Orders)
  Derived ratios: FC/SL1, OO/SL1, CO/SL1, etc.

Ground truth: Coverage (= Covered_Orders / Actual_Sales) and Bias curves.
"""

import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 160)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')

# ── 1. Load & filter ──────────────────────────────────────────────
CSV = "/home/user/techwriting/training_data_anonymized.csv"
df = pd.read_csv(CSV)
df = df[df['Timeframe'] == 12].copy()

sites = sorted(df['Site'].unique())
lags = sorted(df['Prediction_Lag'].unique())
customers = sorted(df['GSA'].unique())

print("=" * 90)
print("CURVE SIMILARITY FEASIBILITY CHECK (v3 — full dataset)")
print("Filtered to TF=12 | Observable-at-forecast-time proxies only")
print("=" * 90)
print(f"\n{len(sites)} sites, {len(customers)} customers, {len(df)} rows at TF=12")
print(f"Curve: {len(lags)} lags (1–12)")

# ── 2. Build aging curve vectors per site ──────────────────────────
# Ground truth: Coverage and Bias (these use Actual_Sales)
# Coverage is NaN when Covered_Orders == 0 (no forecast), and Bias similarly.
# We compute fc_coverage ourselves to include zero-coverage cases.

df['fc_coverage'] = np.where(
    df['Actual_Sales'] > 0,
    df['Covered_Orders'] / df['Actual_Sales'], np.nan)
df['fc_bias'] = df['Bias']  # already pre-computed

cov_curves = {}
bias_curves = {}
site_obs_count = {}

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
    site_obs_count[site] = len(sd)

# Filter to sites with enough curve data
MIN_COV_PTS = 8
MIN_BIAS_PTS = 5
valid_cov_sites = [s for s in sites if np.sum(~np.isnan(cov_curves[s])) >= MIN_COV_PTS]
valid_bias_sites = [s for s in sites if np.sum(~np.isnan(bias_curves[s])) >= MIN_BIAS_PTS]

print(f"\nSites with >= {MIN_COV_PTS} coverage curve points: {len(valid_cov_sites)}")
print(f"Sites with >= {MIN_BIAS_PTS} bias curve points: {len(valid_bias_sites)}")


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


# Only compute within-customer pairs (the use case is: new site for a customer,
# borrow from existing sites of that customer) AND cross-customer pairs.
# But with 160 sites, all-pairs is 12,720 — manageable.

print("\n" + "=" * 90)
print("SECTION 1: COMPUTING PAIRWISE CURVE SIMILARITY")
print("=" * 90)

# Coverage similarity for all valid pairs
cov_sim = {}
for s1, s2 in combinations(valid_cov_sites, 2):
    r, n = curve_similarity(cov_curves[s1], cov_curves[s2])
    if not np.isnan(r):
        cov_sim[(s1, s2)] = r

# Bias similarity for all valid pairs
bias_sim = {}
for s1, s2 in combinations(valid_bias_sites, 2):
    r, n = curve_similarity(bias_curves[s1], bias_curves[s2])
    if not np.isnan(r):
        bias_sim[(s1, s2)] = r

print(f"Coverage: {len(cov_sim)} valid pairs")
print(f"Bias: {len(bias_sim)} valid pairs")

# Distribution of similarities
if cov_sim:
    cov_vals = np.array(list(cov_sim.values()))
    print(f"\nCoverage similarity distribution:")
    print(f"  mean={cov_vals.mean():.3f}  median={np.median(cov_vals):.3f}  "
          f"std={cov_vals.std():.3f}  min={cov_vals.min():.3f}  max={cov_vals.max():.3f}")

if bias_sim:
    bias_vals = np.array(list(bias_sim.values()))
    print(f"Bias similarity distribution:")
    print(f"  mean={bias_vals.mean():.3f}  median={np.median(bias_vals):.3f}  "
          f"std={bias_vals.std():.3f}  min={bias_vals.min():.3f}  max={bias_vals.max():.3f}")

# Combined similarity
combined_sim = {}
all_pairs = set(list(cov_sim.keys()) + list(bias_sim.keys()))
for pair in all_pairs:
    cr = cov_sim.get(pair, np.nan)
    br = bias_sim.get(pair, np.nan)
    if not np.isnan(cr) and not np.isnan(br):
        combined_sim[pair] = (cr + br) / 2
    elif not np.isnan(cr):
        combined_sim[pair] = cr

print(f"Combined: {len(combined_sim)} valid pairs")

# ── 4. Observable proxy features per site ──────────────────────────
print("\n" + "=" * 90)
print("SECTION 2: OBSERVABLE PROXY FEATURES")
print("=" * 90)

proxy_features = {}
for site in sites:
    sd = df[df['Site'] == site]

    sl1 = sd['Historical_Sales_Lag1']
    sl12 = sd['Historical_Sales_Lag12']
    oo = sd['Open_Orders']
    fc = sd['Forecast_Value']
    co = sd['Covered_Orders']
    coo = sd['Covered_Open_Orders']

    def safe_ratio(num, denom):
        mask = denom > 0
        if mask.sum() == 0:
            return np.nan
        return (num[mask] / denom[mask]).mean()

    proxy_features[site] = {
        # Raw means
        'sales_lag1': sl1.mean(),
        'sales_lag12': sl12.mean(),
        'open_orders': oo.mean(),
        'forecast_value': fc.mean(),
        'covered_orders': co.mean(),
        'covered_oo': coo.mean(),
        # Pre-computed observable ratios (no Actual_Sales)
        'OO_Coverage': sd['OO_Coverage'].mean(),  # Covered_OO / Open_Orders
        'OO_Bias': sd['OO_Bias'].mean(),          # Forecast / Covered_OO
        # Derived ratios (all observable at forecast time)
        'FC/SL1': safe_ratio(fc, sl1),
        'OO/SL1': safe_ratio(oo, sl1),
        'CO/SL1': safe_ratio(co, sl1),
        'COO/SL1': safe_ratio(coo, sl1),
        'OO/FC': safe_ratio(oo, fc),
        'COO/OO': safe_ratio(coo, oo),
        'FC/CO': safe_ratio(fc, co),
    }

proxy_df = pd.DataFrame(proxy_features).T

# Show summary stats for proxy features
print("\nProxy feature summary (across all sites):")
print(proxy_df.describe().to_string())

# ── 5. Correlate each proxy with curve similarity ──────────────────
print("\n" + "=" * 90)
print("SECTION 3: DOES PROXY SIMILARITY PREDICT CURVE SIMILARITY?")
print("=" * 90)
print("Spearman r between |proxy_A - proxy_B| and curve_similarity(A, B)")
print("Negative r = sites with similar proxy have similar curves (good)\n")

proxy_cols = list(proxy_df.columns)

for target_label, sim_dict in [("COVERAGE curve", cov_sim),
                                 ("BIAS curve", bias_sim),
                                 ("COMBINED curve", combined_sim)]:
    if not sim_dict:
        continue

    pairs = list(sim_dict.keys())
    sim_vec = np.array([sim_dict[p] for p in pairs])

    print(f"--- vs {target_label} similarity ({len(pairs)} pairs) ---")
    print(f"{'Proxy feature':<20} {'Spearman r':>12} {'p-value':>12} {'N pairs':>10} {'Signal?'}")
    print("-" * 70)

    results = []
    for col in proxy_cols:
        proxy_dist = []
        valid_mask = []
        for s1, s2 in pairs:
            v1 = proxy_df.loc[s1, col]
            v2 = proxy_df.loc[s2, col]
            d = abs(v1 - v2)
            proxy_dist.append(d)
            valid_mask.append(not np.isnan(d))
        proxy_dist = np.array(proxy_dist)
        valid_mask = np.array(valid_mask)

        n = valid_mask.sum()
        if n >= 20:
            r, p = stats.spearmanr(proxy_dist[valid_mask], sim_vec[valid_mask])
            results.append((col, r, p, n))
        else:
            results.append((col, np.nan, np.nan, n))

    # Sort by Spearman r (most negative = best)
    results.sort(key=lambda x: x[1] if not np.isnan(x[1]) else 999)
    for col, r, p, n in results:
        if np.isnan(r):
            print(f"{col:<20} {'insuff.':>12} {'':>12} {n:>10}")
        else:
            sig = "PROMISING" if (r < -0.3 and p < 0.05) else \
                  "weak" if r < -0.15 else "—"
            print(f"{col:<20} {r:>12.4f} {p:>12.6f} {n:>10} {sig}")
    print()

# ── 6. Within-customer vs cross-customer analysis ─────────────────
print("=" * 90)
print("SECTION 4: WITHIN-CUSTOMER vs CROSS-CUSTOMER")
print("=" * 90)
print("Does similarity work better when comparing sites of the same customer?\n")

# Map site → customer
site_to_cust = df.groupby('Site')['GSA'].first().to_dict()

for target_label, sim_dict in [("COVERAGE", cov_sim), ("BIAS", bias_sim)]:
    if not sim_dict:
        continue

    within = [(p, v) for p, v in sim_dict.items()
              if site_to_cust.get(p[0]) == site_to_cust.get(p[1])]
    across = [(p, v) for p, v in sim_dict.items()
              if site_to_cust.get(p[0]) != site_to_cust.get(p[1])]

    if within and across:
        w_vals = [v for _, v in within]
        a_vals = [v for _, v in across]
        print(f"{target_label}:")
        print(f"  Within-customer pairs: {len(within)}, "
              f"mean similarity = {np.mean(w_vals):.3f}")
        print(f"  Cross-customer pairs:  {len(across)}, "
              f"mean similarity = {np.mean(a_vals):.3f}")
        t, p = stats.mannwhitneyu(w_vals, a_vals, alternative='greater')
        print(f"  Mann-Whitney U test (within > across): p = {p:.6f}")

        # Now test proxies within-customer only
        print(f"\n  --- Proxy correlations (WITHIN-customer pairs only) ---")
        within_pairs = [p for p, _ in within]
        within_sim = np.array([v for _, v in within])

        best_features = []
        for col in proxy_cols:
            proxy_dist = []
            valid_mask = []
            for s1, s2 in within_pairs:
                v1 = proxy_df.loc[s1, col]
                v2 = proxy_df.loc[s2, col]
                d = abs(v1 - v2)
                proxy_dist.append(d)
                valid_mask.append(not np.isnan(d))
            proxy_dist = np.array(proxy_dist)
            valid_mask = np.array(valid_mask)
            n = valid_mask.sum()
            if n >= 10:
                r, p = stats.spearmanr(proxy_dist[valid_mask], within_sim[valid_mask])
                best_features.append((col, r, p, n))

        best_features.sort(key=lambda x: x[1])
        print(f"  {'Proxy':<20} {'Spearman r':>12} {'p-value':>12} {'N':>6}")
        print(f"  {'-'*55}")
        for col, r, p, n in best_features[:8]:
            sig = " <<<" if (r < -0.3 and p < 0.05) else ""
            print(f"  {col:<20} {r:>12.4f} {p:>12.6f} {n:>6}{sig}")
        print()

# ── 7. Summary ─────────────────────────────────────────────────────
print("=" * 90)
print("SUMMARY")
print("=" * 90)
print("""
Observable-at-forecast-time proxy features tested:
  Raw:     sales_lag1, sales_lag12, open_orders, forecast_value,
           covered_orders, covered_oo
  Pre-computed: OO_Coverage (Covered_OO / Open_Orders),
                OO_Bias (Forecast / Covered_OO)
  Ratios:  FC/SL1, OO/SL1, CO/SL1, COO/SL1, OO/FC, COO/OO, FC/CO

Ground truth: Coverage (Covered_Orders / Actual_Sales) and Bias curves
at TF=12, across Lags 1–12.

Negative Spearman r (< -0.3, p < 0.05) = proxy distance predicts curve
dissimilarity → approach viable.
""")
print("=" * 90)
print("ANALYSIS COMPLETE")
print("=" * 90)
