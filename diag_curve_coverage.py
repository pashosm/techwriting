"""Per-(customer, site) diagnostic: at what calendar month does the
(site, Timeframe=12, Prediction_Lag=1) cell first accumulate 3 completed
FC-bearing observations? That is the first snapshot at which V9 would
use a curve-corrected forecast for this cell."""
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from aging_v9 import load_data, MIN_OBS_FLAT

df = load_data('training_data_anonymized.csv')  # tf=12 already applied

# Eligibility: lag=1, FC-bearing, both ratios computable
eligible = df[
    (df['Prediction_Lag'] == 1) &
    (df['Has_Forecast'] == 1) &
    (df['fc_coverage'].notna()) & (df['fc_coverage'] > 0) & (df['fc_coverage'] < 10) &
    (df['fc_bias'].notna())     & (df['fc_bias'] > 0)     & (df['fc_bias'] < 10)
].copy()

def first_qualifying_snapshot(tpe_series):
    """Given the Target_Period_End dates of eligible rows (sorted asc), return
    the first calendar-month snapshot at which >=3 of them are completed
    (i.e., TPE < snap). That is the month-start AFTER the 3rd observation."""
    s = sorted(tpe_series)
    if len(s) < MIN_OBS_FLAT:
        return None, len(s)
    third = s[MIN_OBS_FLAT - 1]
    # First month-start strictly after `third`
    snap = (pd.Timestamp(third) + pd.DateOffset(days=1))
    return pd.Timestamp(snap.year, snap.month, 1) + (
        pd.DateOffset(months=1) if snap.day != 1 else pd.DateOffset(0)
    ), len(s)

rows = []
for (cust, site), grp in eligible.groupby(['GSA', 'Site']):
    snap, n = first_qualifying_snapshot(grp['Target_Period_End'])
    rows.append({
        'customer': cust,
        'site': site,
        'n_eligible_lag1_tf12': n,
        'qualifies_from': snap,
        'earliest_tpe': grp['Target_Period_End'].min() if n > 0 else None,
        'latest_tpe': grp['Target_Period_End'].max() if n > 0 else None,
    })

# Also include sites that have NO lag-1 FC rows at all (exist in data but not in `eligible`)
all_sites = set(df[['GSA', 'Site']].apply(tuple, axis=1))
seen = {(r['customer'], r['site']) for r in rows}
for cust, site in sorted(all_sites - seen):
    rows.append({
        'customer': cust, 'site': site,
        'n_eligible_lag1_tf12': 0, 'qualifies_from': None,
        'earliest_tpe': None, 'latest_tpe': None,
    })

out = pd.DataFrame(rows).sort_values(
    ['customer', 'qualifies_from', 'site'],
    na_position='last'
).reset_index(drop=True)
out['qualifies_from'] = out['qualifies_from'].apply(
    lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else '— never —')
out['earliest_tpe'] = out['earliest_tpe'].apply(
    lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else '')
out['latest_tpe'] = out['latest_tpe'].apply(
    lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else '')

# ---------- Per-site detail ----------
print("=" * 90)
print("Per-site: first snapshot at which (site, tf=12, lag=1) cell has >=3 completed FC obs")
print("=" * 90)
print(f"{'Customer':<14} {'Site':<14} {'Qualifies From':<16} {'n_eligible':>10} "
      f"{'earliest':<12} {'latest':<12}")
print("-" * 90)
for _, r in out.iterrows():
    print(f"{r['customer']:<14} {r['site']:<14} {r['qualifies_from']:<16} "
          f"{r['n_eligible_lag1_tf12']:>10,} {r['earliest_tpe']:<12} {r['latest_tpe']:<12}")

# ---------- Per-customer rollup ----------
print("\n" + "=" * 90)
print("Per-customer coverage rollup: sites qualifying by snapshot date")
print("=" * 90)
snapshots_to_show = pd.date_range('2024-01-01', '2025-04-01', freq='MS')

per_cust = []
for cust, grp in out.groupby('customer'):
    total_sites = len(grp)
    qualifying_dates = pd.to_datetime(grp['qualifies_from'], errors='coerce').dropna()
    row = {'customer': cust, 'total_sites': total_sites, 'ever_qualify': len(qualifying_dates)}
    for snap in snapshots_to_show:
        row[snap.strftime('%Y-%m')] = int((qualifying_dates <= snap).sum())
    per_cust.append(row)

per_cust_df = pd.DataFrame(per_cust)
header = f"{'Customer':<14} {'TotSites':>8} {'EverQual':>8}"
for snap in snapshots_to_show:
    header += f" {snap.strftime('%Y-%m'):>7}"
print(header)
print("-" * len(header))
for r in per_cust:
    line = f"{r['customer']:<14} {r['total_sites']:>8,} {r['ever_qualify']:>8,}"
    for snap in snapshots_to_show:
        line += f" {r[snap.strftime('%Y-%m')]:>7,}"
    print(line)

# Grand total line
total = {'customer': 'ALL', 'total_sites': len(out),
         'ever_qualify': int(pd.to_datetime(out['qualifies_from'], errors='coerce').notna().sum())}
all_dates = pd.to_datetime(out['qualifies_from'], errors='coerce').dropna()
line = f"{'ALL':<14} {total['total_sites']:>8,} {total['ever_qualify']:>8,}"
for snap in snapshots_to_show:
    line += f" {int((all_dates <= snap).sum()):>7,}"
print("-" * len(header))
print(line)
