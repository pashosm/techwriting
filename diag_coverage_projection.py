"""Project FC_Corrected coverage into the future.

For each cell (gs, tf, lag, regime), count how many observations exist as of
the latest snapshot and how quickly they're accumulating.  Project when each
cell will cross MIN_OBS_FLAT=3 and estimate dollar coverage over time.
"""
import aging_v9 as v9
import numpy as np
import pandas as pd

df = v9.load_data(v9.DEFAULT_DATA_PATH)
latest_snap = '2025-03-01'
snap_ts = pd.Timestamp(latest_snap)

# --- Current state at latest snapshot ---
train = df[df['Target_Period_End'] < snap_ts].copy()
test = df[(df['Target_Period_Start'] >= snap_ts) &
          (df['Reference_Month'] < snap_ts) &
          (df['Prediction_Lag'].isin(v9.TEST_LAGS))].copy()
curve_train = train[train['Has_Forecast'] == 1]

# Build current curves to see which cells qualify
cov_curve = v9.build_flat_curves(curve_train, 'fc_coverage')
bias_curve = v9.build_flat_curves(curve_train, 'fc_bias')

# All test-eligible sites and their latest actual sales
hist = v9.build_historical(train)

# Get all unique sites from test rows and their sales
all_sites = test.groupby('gsa_site').agg(
    Actual_Sales=('Actual_Sales', 'sum'),
    n_rows=('Actual_Sales', 'count'),
).reset_index()

# ============================================================
# For each site, check observation count per cell at latest snapshot
# ============================================================
print("=" * 100)
print(f"FC_Corrected Coverage Projection (from {latest_snap})")
print("=" * 100)

# Count observations per cell in curve_train
cell_obs = curve_train.groupby(
    ['gsa_site', 'Timeframe', 'Prediction_Lag', 'site_regime']
).agg(
    n_bias=('fc_bias', lambda x: x.notna().sum()),
    n_cov=('fc_coverage', lambda x: x.notna().sum()),
    first_obs=('Reference_Month', 'min'),
    last_obs=('Reference_Month', 'max'),
).reset_index()

# A cell qualifies if BOTH bias and coverage have >= MIN_OBS_FLAT valid obs
cell_obs['qualifies'] = (cell_obs['n_bias'] >= v9.MIN_OBS_FLAT) & (cell_obs['n_cov'] >= v9.MIN_OBS_FLAT)
cell_obs['min_obs'] = cell_obs[['n_bias', 'n_cov']].min(axis=1)
cell_obs['obs_needed'] = (v9.MIN_OBS_FLAT - cell_obs['min_obs']).clip(lower=0)

# Focus on lag=1 cells (primary prediction lag)
lag1 = cell_obs[cell_obs['Prediction_Lag'] == 1].copy()

# Estimate observation accumulation rate per cell
# Use the span between first and last obs to estimate monthly rate
lag1['span_months'] = (lag1['last_obs'] - lag1['first_obs']).dt.days / 30.0
lag1['obs_rate'] = np.where(lag1['span_months'] > 0,
                            lag1['min_obs'] / lag1['span_months'],
                            0.0)

# For cells with only 1-2 obs, assume ~1 obs per month (conservative)
lag1['obs_rate'] = lag1['obs_rate'].clip(lower=0.3, upper=2.0)
lag1.loc[lag1['min_obs'] <= 1, 'obs_rate'] = 0.5  # conservative for brand new

# Months until qualification
lag1['months_to_qualify'] = np.where(
    lag1['qualifies'], 0,
    np.ceil(lag1['obs_needed'] / lag1['obs_rate'])
)

# Get actual sales per site for dollar weighting
site_sales = test.groupby('gsa_site')['Actual_Sales'].sum().to_dict()

# Also get the mean actual sales per site (latest available)
site_mean_sales = train.groupby('gsa_site')['Actual_Sales'].mean().to_dict()

lag1['site_sales'] = lag1['gsa_site'].map(site_sales).fillna(0)

# ============================================================
# Sites that currently have FC coverage vs those that don't
# ============================================================
# A site has FC coverage if at least one of its lag-1 cells qualifies
site_has_fc = lag1[lag1['qualifies']].groupby('gsa_site')['site_sales'].first()
site_no_fc = lag1[~lag1['qualifies'] & ~lag1['gsa_site'].isin(site_has_fc.index)].copy()

# Sites with NO forecast data at all
all_test_sites = set(test['gsa_site'].unique())
sites_with_any_fc_data = set(lag1['gsa_site'].unique())
sites_no_fc_data = all_test_sites - sites_with_any_fc_data

print(f"\nSite counts (lag-1 cells):")
print(f"  Sites with FC_C coverage:       {len(site_has_fc)}")
print(f"  Sites with FC data but not yet qualifying: {len(site_no_fc['gsa_site'].unique())}")
print(f"  Sites with NO forecast data at all:        {len(sites_no_fc_data)}")

# ============================================================
# Largest missing sites
# ============================================================
print(f"\n{'='*100}")
print("Largest Sites WITHOUT FC_Corrected Coverage")
print(f"{'='*100}")

# For sites with some FC data but not qualifying
almost_there = lag1[~lag1['qualifies'] & ~lag1['gsa_site'].isin(site_has_fc.index)].copy()
almost_summary = almost_there.groupby('gsa_site').agg(
    max_obs=('min_obs', 'max'),
    min_months_to_qualify=('months_to_qualify', 'min'),
    site_sales=('site_sales', 'first'),
).sort_values('site_sales', ascending=False)

# Sites with no FC data at all
no_fc_sites = []
for gs in sites_no_fc_data:
    sales = site_sales.get(gs, 0)
    no_fc_sites.append({'gsa_site': gs, 'max_obs': 0, 'min_months_to_qualify': np.inf, 'site_sales': sales})
no_fc_df = pd.DataFrame(no_fc_sites)

# Combine
all_missing = pd.concat([almost_summary.reset_index(), no_fc_df], ignore_index=True)
all_missing = all_missing.sort_values('site_sales', ascending=False)

total_test_sales = test['Actual_Sales'].sum()
fc_covered_sales = sum(site_sales.get(gs, 0) for gs in site_has_fc.index)

print(f"\nCurrently covered: ${fc_covered_sales:,.0f} ({fc_covered_sales/total_test_sales*100:.1f}% of ${total_test_sales:,.0f})")
print(f"Missing:           ${total_test_sales - fc_covered_sales:,.0f} ({(total_test_sales - fc_covered_sales)/total_test_sales*100:.1f}%)")

print(f"\n  {'Site':<35} {'Sales ($)':>16} {'% of Total':>10} {'Obs':>4} {'Mo to Qualify':>13} {'Status'}")
print(f"  {'-'*35} {'-'*16} {'-'*10} {'-'*4} {'-'*13} {'-'*20}")
cum_pct = 0
for _, r in all_missing.head(25).iterrows():
    pct = r['site_sales'] / total_test_sales * 100
    cum_pct += pct
    obs = int(r['max_obs'])
    mtq = r['min_months_to_qualify']
    if mtq == np.inf:
        status = "No FC data"
        mtq_str = "N/A"
    elif mtq <= 3:
        status = f"Need {v9.MIN_OBS_FLAT - obs} more obs"
        mtq_str = f"~{int(mtq)}mo"
    else:
        status = f"Need {v9.MIN_OBS_FLAT - obs} more obs"
        mtq_str = f"~{int(mtq)}mo"
    print(f"  {r['gsa_site']:<35} ${r['site_sales']:>14,.0f} {pct:>9.1f}% {obs:>4} {mtq_str:>13} {status}")

print(f"\n  Top 25 missing sites = {cum_pct:.1f}% of total test sales")

# ============================================================
# Month-by-month projection
# ============================================================
print(f"\n{'='*100}")
print("Projected Coverage Over Time (dollar-weighted)")
print(f"{'='*100}")

# For each future month, estimate which cells will have crossed MIN_OBS_FLAT
# and compute cumulative dollar coverage

# Current FC-covered sites and their sales
covered_sites = set(site_has_fc.index)
covered_sales = fc_covered_sales

# Build a list of (site, months_to_qualify, sales) for sites not yet covered
pipeline = []
for gs in all_missing['gsa_site'].unique():
    row = all_missing[all_missing['gsa_site'] == gs].iloc[0]
    if row['min_months_to_qualify'] < np.inf:
        pipeline.append({
            'gsa_site': gs,
            'months_to_qualify': row['min_months_to_qualify'],
            'sales': row['site_sales'],
        })

# Sort by months_to_qualify
pipeline.sort(key=lambda x: x['months_to_qualify'])

print(f"\n  {'Month':>3} {'Date':>12} | {'FC_C Sites':>10} {'FC_C $':>18} {'FC_C $%':>8} | {'New Sites':>10}")
print(f"  {'-'*3} {'-'*12} | {'-'*10} {'-'*18} {'-'*8} | {'-'*10}")

# Current state
n_fc_sites = len(covered_sites)
print(f"  {'0':>3} {latest_snap:>12} | {n_fc_sites:>10} ${covered_sales:>16,.0f} {covered_sales/total_test_sales*100:>7.1f}% | {'(current)':>10}")

hit_50 = None
for month_ahead in range(1, 25):
    new_this_month = [p for p in pipeline
                      if p['months_to_qualify'] <= month_ahead
                      and p['gsa_site'] not in covered_sites]
    new_count = 0
    for p in new_this_month:
        covered_sites.add(p['gsa_site'])
        covered_sales += p['sales']
        new_count += 1

    proj_date = (snap_ts + pd.DateOffset(months=month_ahead)).strftime('%Y-%m-%d')
    pct = covered_sales / total_test_sales * 100
    print(f"  {month_ahead:>3} {proj_date:>12} | {len(covered_sites):>10} ${covered_sales:>16,.0f} {pct:>7.1f}% | {new_count:>10}")

    if hit_50 is None and pct >= 50:
        hit_50 = (month_ahead, proj_date, pct)

# Never-qualifiable sites
never_qualify_sales = sum(r['site_sales'] for _, r in all_missing.iterrows()
                          if r['min_months_to_qualify'] == np.inf)
max_possible = total_test_sales - never_qualify_sales
max_possible_pct = max_possible / total_test_sales * 100

print(f"\n  Ceiling (sites with no FC data cannot qualify): ${max_possible:,.0f} ({max_possible_pct:.1f}%)")
if never_qualify_sales > 0:
    print(f"  Sites with NO forecast data account for ${never_qualify_sales:,.0f} ({never_qualify_sales/total_test_sales*100:.1f}%)")

if hit_50:
    print(f"\n  >>> 50% dollar coverage projected at: {hit_50[1]} (+{hit_50[0]} months, {hit_50[2]:.1f}%)")
elif max_possible_pct < 50:
    print(f"\n  >>> 50% dollar coverage NOT reachable — max possible is {max_possible_pct:.1f}%")
    print(f"      {len(sites_no_fc_data)} sites have no forecast data at all")
else:
    print(f"\n  >>> 50% dollar coverage not reached within 24 months — pipeline too slow")
