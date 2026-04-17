"""Project FC_Corrected coverage into the future.

Instead of only looking at training-eligible rows, examine ALL forecast rows
(including those with future Target_Period_End dates) to determine exactly
when each site's cell will cross MIN_OBS_FLAT=3.  For each future snapshot T,
a forecast row enters training when Target_Period_End < T.
"""
import aging_v9 as v9
import numpy as np
import pandas as pd

df = v9.load_data(v9.DEFAULT_DATA_PATH)
latest_snap = '2025-03-01'
snap_ts = pd.Timestamp(latest_snap)

# All forecast rows in the dataset (not just training-eligible)
all_fc = df[df['Has_Forecast'] == 1].copy()

# Use latest available actual sales per site for dollar weighting
# (from all completed periods)
completed = df[df['Target_Period_End'] < snap_ts]
site_sales = completed.groupby('gsa_site')['Actual_Sales'].mean().to_dict()
all_test_sites = sorted(df['gsa_site'].unique())
total_sales = sum(site_sales.get(gs, 0) for gs in all_test_sites)

# ============================================================
# For each cell, find the Target_Period_End dates of all FC rows
# so we know exactly when each observation enters training
# ============================================================
print("=" * 100)
print(f"FC_Corrected Coverage Projection (all forecasts, from {latest_snap})")
print("=" * 100)

# For each cell (gs, tf, lag, regime), list the Target_Period_End dates
# of rows with valid fc_bias and fc_coverage
valid_fc = all_fc[
    all_fc['fc_bias'].notna() & (all_fc['fc_bias'] > 0) & (all_fc['fc_bias'] < 10) &
    all_fc['fc_coverage'].notna() & (all_fc['fc_coverage'] > 0) & (all_fc['fc_coverage'] < 10)
].copy()

cell_completion_dates = {}
for (gs, tf, lag, regime), grp in valid_fc.groupby(
        ['gsa_site', 'Timeframe', 'Prediction_Lag', 'site_regime']):
    dates = sorted(grp['Target_Period_End'].unique())
    cell_completion_dates[(gs, tf, lag, regime)] = dates

# ============================================================
# Simulate forward: for each future snapshot, which cells qualify?
# ============================================================
# Generate monthly snapshots from current through +24 months
future_snaps = pd.date_range(snap_ts, periods=25, freq='MS')

# For each snapshot, determine which lag-1 cells have >= MIN_OBS_FLAT
# completed observations, and thus which sites have FC_C coverage
projection = []

for snap_t in future_snaps:
    qualified_sites = set()
    qualified_sales = 0

    for (gs, tf, lag, regime), dates in cell_completion_dates.items():
        if lag > v9.MAX_VINTAGE_LAG:
            continue
        # Count how many target periods have completed by this snapshot
        n_completed = sum(1 for d in dates if d < snap_t)
        if n_completed >= v9.MIN_OBS_FLAT:
            if gs not in qualified_sites:
                qualified_sites.add(gs)
                qualified_sales += site_sales.get(gs, 0)

    projection.append({
        'snapshot': snap_t,
        'n_sites': len(qualified_sites),
        'fc_sales': qualified_sales,
        'fc_pct': qualified_sales / total_sales * 100 if total_sales > 0 else 0,
        'sites': qualified_sites,
    })

# ============================================================
# Print projection table
# ============================================================
print(f"\n  {'Mo':>3} {'Snapshot':>12} | {'FC_C Sites':>10} {'FC_C $':>18} {'FC_C $%':>8} | {'New Sites':>10}")
print(f"  {'-'*3} {'-'*12} | {'-'*10} {'-'*18} {'-'*8} | {'-'*10}")

hit_50 = None
prev_sites = set()
for i, p in enumerate(projection):
    new_sites = p['sites'] - prev_sites
    snap_str = p['snapshot'].strftime('%Y-%m-%d')
    label = "(current)" if i == 0 else str(len(new_sites))
    print(f"  {i:>3} {snap_str:>12} | {p['n_sites']:>10} ${p['fc_sales']:>16,.0f} {p['fc_pct']:>7.1f}% | {label:>10}")

    if hit_50 is None and p['fc_pct'] >= 50:
        hit_50 = (i, snap_str, p['fc_pct'])

    prev_sites = p['sites'].copy()

# ============================================================
# Ceiling: sites that have ANY forecast data vs none at all
# ============================================================
sites_with_any_fc = set(valid_fc['gsa_site'].unique())
sites_no_fc_ever = set(all_test_sites) - sites_with_any_fc
never_qualify_sales = sum(site_sales.get(gs, 0) for gs in sites_no_fc_ever)
max_possible = total_sales - never_qualify_sales
max_pct = max_possible / total_sales * 100 if total_sales > 0 else 0

# Also check: sites with FC data but fewer than MIN_OBS_FLAT even looking at
# ALL future dates (i.e., structurally can't qualify even with infinite time)
sites_too_few_total = set()
for gs in sites_with_any_fc:
    # Check if any lag<=MAX_VINTAGE_LAG cell for this site has >= MIN_OBS_FLAT total rows
    has_enough = False
    for (g, tf, lag, regime), dates in cell_completion_dates.items():
        if g == gs and lag <= v9.MAX_VINTAGE_LAG and len(dates) >= v9.MIN_OBS_FLAT:
            has_enough = True
            break
    if not has_enough:
        sites_too_few_total.add(gs)

too_few_sales = sum(site_sales.get(gs, 0) for gs in sites_too_few_total)

print(f"\n  Ceiling analysis:")
print(f"    Sites with valid FC data in dataset:     {len(sites_with_any_fc)}")
print(f"    Sites with NO FC data anywhere:          {len(sites_no_fc_ever)} (${never_qualify_sales:,.0f}, {never_qualify_sales/total_sales*100:.1f}%)")
print(f"    Sites with FC but < {v9.MIN_OBS_FLAT} total obs:       {len(sites_too_few_total)} (${too_few_sales:,.0f}, {too_few_sales/total_sales*100:.1f}%)")
print(f"    Max reachable coverage:                  ${max_possible - too_few_sales:,.0f} ({(max_possible - too_few_sales)/total_sales*100:.1f}%)")

if hit_50:
    print(f"\n  >>> 50% dollar coverage projected at: {hit_50[1]} (+{hit_50[0]} months, {hit_50[2]:.1f}%)")
elif max_pct < 50:
    print(f"\n  >>> 50% dollar coverage NOT reachable with current forecast data")
else:
    print(f"\n  >>> 50% dollar coverage not reached within 24 months")

# ============================================================
# Largest missing sites — with actual Target_Period_End info
# ============================================================
print(f"\n{'='*100}")
print("Largest Sites NOT YET Covered (with pipeline status)")
print(f"{'='*100}")

# For the final projection snapshot, which sites are still missing?
final_covered = projection[-1]['sites']
current_covered = projection[0]['sites']

missing_sites = []
for gs in all_test_sites:
    if gs in current_covered:
        continue
    sales = site_sales.get(gs, 0)

    # Find earliest snapshot where this site qualifies
    qualifies_at = None
    for p in projection:
        if gs in p['sites']:
            qualifies_at = p['snapshot'].strftime('%Y-%m-%d')
            break

    # Current obs count (best lag<=3 cell)
    best_obs = 0
    total_obs = 0
    for (g, tf, lag, regime), dates in cell_completion_dates.items():
        if g == gs and lag <= v9.MAX_VINTAGE_LAG:
            n_now = sum(1 for d in dates if d < snap_ts)
            n_total = len(dates)
            if n_now > best_obs:
                best_obs = n_now
            total_obs = max(total_obs, n_total)

    has_fc = gs in sites_with_any_fc

    missing_sites.append({
        'gsa_site': gs,
        'sales': sales,
        'has_fc': has_fc,
        'obs_now': best_obs,
        'obs_total': total_obs,
        'qualifies_at': qualifies_at,
    })

missing_df = pd.DataFrame(missing_sites).sort_values('sales', ascending=False)

print(f"\n  {'Site':<35} {'Avg Sales ($)':>16} {'% of Total':>10} {'Obs Now':>8} {'Obs Total':>10} {'Qualifies':>14} {'Status'}")
print(f"  {'-'*35} {'-'*16} {'-'*10} {'-'*8} {'-'*10} {'-'*14} {'-'*20}")

cum_pct = 0
for _, r in missing_df.head(30).iterrows():
    pct = r['sales'] / total_sales * 100
    cum_pct += pct
    if not r['has_fc']:
        status = "No FC data"
        qual = "Never"
    elif r['qualifies_at']:
        status = f"{r['obs_now']}/{v9.MIN_OBS_FLAT} obs"
        qual = r['qualifies_at']
    else:
        status = f"{r['obs_now']}/{v9.MIN_OBS_FLAT} obs (insuf)"
        qual = "Never*"
    print(f"  {r['gsa_site']:<35} ${r['sales']:>14,.0f} {pct:>9.1f}% {r['obs_now']:>8} {r['obs_total']:>10} {qual:>14} {status}")

print(f"\n  Top 30 missing sites = {cum_pct:.1f}% of total avg sales")
print(f"  * 'Never' with FC data = fewer than {v9.MIN_OBS_FLAT} total FC rows exist for any lag<={v9.MAX_VINTAGE_LAG} cell")
