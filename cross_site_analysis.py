#!/usr/bin/env python3
"""
Cross-Site Demand Correlation Feasibility Analysis
===================================================
Customer 1 data — assess whether cross-site correlations are strong
enough and predictive enough to justify adding cross-site features
to the forecasting model.
"""

import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 140)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')

# ── 1. Load & filter ────────────────────────────────────────────────
CSV = "/home/user/techwriting/Dummy Training Data Customer 1 Try 2 11-Feb-2026.csv"
df = pd.read_csv(CSV)
df['Reference_Month'] = pd.to_datetime(df['Reference_Month'])

# Filter: TF=3, Prediction_Lag=1  →  one row per (Site, Reference_Month)
filt = df[(df['Timeframe'] == 3) & (df['Prediction_Lag'] == 1)].copy()
filt['OO_ratio'] = filt['Open_Orders'] / filt['Actual_Sales']

print("=" * 80)
print("SECTION 1: DATA OVERVIEW")
print("=" * 80)
print(f"Rows after filter (TF=3, Lag=1): {len(filt)}")
print(f"Unique sites: {sorted(filt['Site'].unique())}")
print(f"Number of unique sites: {filt['Site'].nunique()}")
print(f"Unique Reference_Months: {filt['Reference_Month'].nunique()}")
print(f"Date range: {filt['Reference_Month'].min()} to {filt['Reference_Month'].max()}")

# Per-site coverage
coverage = filt.groupby('Site')['Reference_Month'].nunique()
total_months = filt['Reference_Month'].nunique()
print(f"\nPer-site month coverage (out of {total_months} total months):")
for site, n in coverage.items():
    flag = " *** FULL ***" if n == total_months else f"  (missing {total_months - n})"
    print(f"  {site}: {n} months{flag}")

full_sites = coverage[coverage == total_months].index.tolist()
print(f"\nSites with FULL coverage across all {total_months} months: {full_sites}")
print(f"  → {len(full_sites)} of {filt['Site'].nunique()} sites")

# ── 2. Pivot tables ─────────────────────────────────────────────────
pivot_sales = filt.pivot_table(
    index='Reference_Month', columns='Site', values='Actual_Sales'
)
pivot_oo = filt.pivot_table(
    index='Reference_Month', columns='Site', values='Open_Orders'
)
pivot_oo_ratio = filt.pivot_table(
    index='Reference_Month', columns='Site', values='OO_ratio'
)

# Use only full-coverage sites for correlation analysis
if len(full_sites) >= 2:
    analysis_sites = full_sites
else:
    # Fall back to sites with at least 90% coverage
    min_months = int(total_months * 0.9)
    analysis_sites = coverage[coverage >= min_months].index.tolist()
    print(f"\n  (Not enough full-coverage sites; using sites with >= {min_months} months: {analysis_sites})")

pivot_sales_full = pivot_sales[analysis_sites].sort_index()
pivot_oo_ratio_full = pivot_oo_ratio[analysis_sites].sort_index()

print(f"\nUsing {len(analysis_sites)} sites for correlation analysis: {analysis_sites}")

# Quick summary stats
print("\n--- Actual_Sales summary per site (TF=3, Lag=1) ---")
print(pivot_sales_full.describe().to_string())

print("\n--- OO_ratio summary per site ---")
print(pivot_oo_ratio_full.describe().to_string())

# ── 3. Contemporaneous correlations (lag=0): LEVELS ─────────────────
print("\n" + "=" * 80)
print("SECTION 2: CONTEMPORANEOUS CORRELATIONS — Actual_Sales LEVELS")
print("=" * 80)
corr_levels = pivot_sales_full.corr(method='pearson')
print("\nPearson correlation matrix (Actual_Sales levels):")
print(corr_levels.to_string())

# ── 4. Contemporaneous correlations (lag=0): CHANGES ────────────────
print("\n" + "=" * 80)
print("SECTION 3: CONTEMPORANEOUS CORRELATIONS — Actual_Sales CHANGES (MoM diff)")
print("=" * 80)
diff_sales = pivot_sales_full.diff().dropna()
n_diff = len(diff_sales)
print(f"Number of observations after differencing: {n_diff}")

corr_changes = diff_sales.corr(method='pearson')
print("\nPearson correlation matrix (Actual_Sales month-to-month changes):")
print(corr_changes.to_string())

# ── 5. Identify top correlated pairs ────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 4: TOP CORRELATED PAIRS (by |r| of Actual_Sales changes)")
print("=" * 80)

pairs = []
sites = corr_changes.columns.tolist()
for i, s1 in enumerate(sites):
    for s2 in sites[i+1:]:
        r = corr_changes.loc[s1, s2]
        pairs.append((s1, s2, r, abs(r)))

pairs.sort(key=lambda x: x[3], reverse=True)
print("\nAll site pairs ranked by |correlation| of sales changes:")
print(f"{'Pair':<30} {'r':>8} {'|r|':>8} {'r²':>8}")
print("-" * 56)
for s1, s2, r, ar in pairs:
    print(f"{s1} — {s2:<20} {r:>8.4f} {ar:>8.4f} {r**2:>8.4f}")

top_pairs = pairs[:4]  # top 4
print(f"\nTop {len(top_pairs)} pairs selected for lagged analysis.")

# ── 6. Lagged cross-correlations (Actual_Sales changes) ─────────────
print("\n" + "=" * 80)
print("SECTION 5: LAGGED CROSS-CORRELATIONS — Actual_Sales CHANGES")
print("=" * 80)
print("Question: Does Site A's sales change at time t predict Site B's change at t+lag?")

def lagged_corr(series_a, series_b, lag):
    """Correlation of series_a[:-lag] with series_b[lag:]."""
    if lag == 0:
        a, b = series_a.align(series_b, join='inner')
        a = a.dropna()
        b = b.loc[a.index]
        b = b.dropna()
        a = a.loc[b.index]
        if len(a) < 5:
            return np.nan, np.nan, len(a)
        r, p = stats.pearsonr(a, b)
        return r, p, len(a)
    # lag > 0: a at time t vs b at time t+lag
    a = series_a.iloc[:-lag]
    b = series_b.iloc[lag:]
    # Align by resetting index
    a_vals = a.values
    b_vals = b.values
    mask = ~(np.isnan(a_vals) | np.isnan(b_vals))
    if mask.sum() < 5:
        return np.nan, np.nan, int(mask.sum())
    r, p = stats.pearsonr(a_vals[mask], b_vals[mask])
    return r, p, int(mask.sum())

for s1, s2, r0, _ in top_pairs:
    print(f"\n--- {s1} → {s2} ---")
    print(f"  {'Lag':>4}  {'Direction':<25} {'r':>8} {'p-value':>10} {'N':>5} {'Signif?':>8}")
    print(f"  " + "-" * 65)
    a = diff_sales[s1]
    b = diff_sales[s2]
    # lag=0
    r, p, n = lagged_corr(a, b, 0)
    sig = "YES" if (not np.isnan(r) and abs(r) > 0.297) else "no"
    print(f"  {0:>4}  {'contemporaneous':<25} {r:>8.4f} {p:>10.4f} {n:>5} {sig:>8}")
    # A leads B (lags 1-3)
    for lag in [1, 2, 3]:
        r, p, n = lagged_corr(a, b, lag)
        rstr = f"{r:>8.4f}" if not np.isnan(r) else "     NaN"
        pstr = f"{p:>10.4f}" if not np.isnan(p) else "       NaN"
        sig = "YES" if (not np.isnan(r) and abs(r) > 0.297) else "no"
        print(f"  {lag:>4}  {s1+' leads':<25} {rstr} {pstr} {n:>5} {sig:>8}")
    # B leads A (lags 1-3)
    for lag in [1, 2, 3]:
        r, p, n = lagged_corr(b, a, lag)
        rstr = f"{r:>8.4f}" if not np.isnan(r) else "     NaN"
        pstr = f"{p:>10.4f}" if not np.isnan(p) else "       NaN"
        sig = "YES" if (not np.isnan(r) and abs(r) > 0.297) else "no"
        print(f"  {lag:>4}  {s2+' leads':<25} {rstr} {pstr} {n:>5} {sig:>8}")

# ── 7. OO_ratio changes ─────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 6: OO_ratio CHANGE CORRELATIONS")
print("=" * 80)

diff_oo = pivot_oo_ratio_full.diff().dropna()
n_diff_oo = len(diff_oo)
print(f"Number of observations after differencing OO_ratio: {n_diff_oo}")

corr_oo_changes = diff_oo.corr(method='pearson')
print("\nPearson correlation matrix (OO_ratio month-to-month changes):")
print(corr_oo_changes.to_string())

# Same top pairs — lagged
print("\nLagged cross-correlations of OO_ratio changes (same top pairs):")
for s1, s2, _, _ in top_pairs:
    print(f"\n--- {s1} → {s2} (OO_ratio changes) ---")
    print(f"  {'Lag':>4}  {'Direction':<25} {'r':>8} {'p-value':>10} {'N':>5} {'Signif?':>8}")
    print(f"  " + "-" * 65)
    a = diff_oo[s1]
    b = diff_oo[s2]
    # lag=0
    r, p, n = lagged_corr(a, b, 0)
    sig = "YES" if (not np.isnan(r) and abs(r) > 0.297) else "no"
    print(f"  {0:>4}  {'contemporaneous':<25} {r:>8.4f} {p:>10.4f} {n:>5} {sig:>8}")
    for lag in [1, 2, 3]:
        r, p, n = lagged_corr(a, b, lag)
        rstr = f"{r:>8.4f}" if not np.isnan(r) else "     NaN"
        pstr = f"{p:>10.4f}" if not np.isnan(p) else "       NaN"
        sig = "YES" if (not np.isnan(r) and abs(r) > 0.297) else "no"
        print(f"  {lag:>4}  {s1+' leads':<25} {rstr} {pstr} {n:>5} {sig:>8}")
    for lag in [1, 2, 3]:
        r, p, n = lagged_corr(b, a, lag)
        rstr = f"{r:>8.4f}" if not np.isnan(r) else "     NaN"
        pstr = f"{p:>10.4f}" if not np.isnan(p) else "       NaN"
        sig = "YES" if (not np.isnan(r) and abs(r) > 0.297) else "no"
        print(f"  {lag:>4}  {s2+' leads':<25} {rstr} {pstr} {n:>5} {sig:>8}")

# ── 8. Statistical significance assessment ───────────────────────────
print("\n" + "=" * 80)
print("SECTION 7: STATISTICAL SIGNIFICANCE ASSESSMENT")
print("=" * 80)

for label, n_obs in [("Levels (N=months)", len(pivot_sales_full)),
                      ("Changes (N=months-1)", n_diff)]:
    # Critical r for two-tailed t-test at p<0.05
    df_stat = n_obs - 2
    if df_stat > 0:
        t_crit = stats.t.ppf(0.975, df_stat)
        r_crit = t_crit / np.sqrt(t_crit**2 + df_stat)
    else:
        r_crit = np.nan
    print(f"\n{label}: N = {n_obs}")
    print(f"  Degrees of freedom: {df_stat}")
    print(f"  Critical |r| for p < 0.05 (two-tailed): {r_crit:.4f}")
    print(f"  That means r² > {r_crit**2:.4f} — the cross-site signal must explain")
    print(f"  >{r_crit**2*100:.1f}% of variance to be statistically significant.")

# Multiple comparisons warning
n_pairs = len(pairs)
bonf = 0.05 / n_pairs
for label, n_obs in [("Changes", n_diff)]:
    df_stat = n_obs - 2
    t_crit_bonf = stats.t.ppf(1 - bonf / 2, df_stat)
    r_crit_bonf = t_crit_bonf / np.sqrt(t_crit_bonf**2 + df_stat)
    print(f"\nBonferroni correction for {n_pairs} pairs:")
    print(f"  Adjusted alpha: {bonf:.5f}")
    print(f"  Bonferroni-corrected critical |r|: {r_crit_bonf:.4f}")

# Count how many pairs are significant
print("\n--- Summary of significant contemporaneous sales-change correlations ---")
sig_count = 0
for s1, s2, r, ar in pairs:
    n_obs = n_diff
    df_stat = n_obs - 2
    t_val = r * np.sqrt(df_stat) / np.sqrt(1 - r**2) if abs(r) < 1 else np.inf
    p_val = 2 * (1 - stats.t.cdf(abs(t_val), df_stat))
    if p_val < 0.05:
        sig_count += 1
        print(f"  {s1} — {s2}: r={r:.4f}, p={p_val:.4f}, r²={r**2:.4f} *")
print(f"\n  {sig_count} of {n_pairs} pairs significant at p<0.05 (uncorrected)")

# ── 9. Pipeline integration analysis ─────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 8: PIPELINE INTEGRATION ANALYSIS")
print("=" * 80)
print("""
Current weighted_avg function (pipeline.py lines 375-469):
  - Iterates: for each (month, gsa_site, timeframe)
  - For each row: blends OO_implied, FC_implied, and historical_avg
  - Weights based on: inverse MAPE * lag_decay_factor
  - Each site predicted INDEPENDENTLY — no cross-site information used

WHERE cross-site info COULD be injected:

  Option A: Cross-site adjustment to historical average (hist dict, line 399-406)
    - Instead of just this site's history, blend in correlated sites' recent actuals
    - Pro: Simple, only changes the 'hist' computation
    - Con: hist is a fallback signal (low weight); limited impact

  Option B: Additional signal term in the weighted blend (lines 460-465)
    - Add a 4th term: w_cross * cross_site_prediction
    - cross_site_prediction = weighted avg of correlated sites' recent actuals/changes
    - Pro: Direct, principled
    - Con: Adds complexity; needs enough correlation to justify a weight

  Option C: Prior/regularizer when OO & FC are missing (lines 439-440, 466-467)
    - When has_oo=False and has_fc=False, currently falls back to hist
    - Could instead use correlated sites' current OO signals as a proxy
    - Pro: Helps exactly where the model is weakest
    - Con: Only helps when signals are missing

PRACTICAL FEASIBILITY DEPENDS ON:
  1. Correlation magnitude — need |r| > 0.3 to explain >9% of variance
  2. Lagged correlations — contemporaneous helps only if you have one site's
     actual before predicting another; LAGGED correlations are truly predictive
  3. Stability — correlations must hold out-of-sample
  4. Sample size — with N~44, even r=0.3 is barely significant; overfitting risk is high
""")

print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
