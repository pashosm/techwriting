"""Diagnostic: MIN_OBS_FLAT=2 vs 3 — coverage vs accuracy tradeoff."""
import aging_v9 as v9
import numpy as np
import pandas as pd

df = v9.load_data(v9.DEFAULT_DATA_PATH)
snaps = [d.strftime('%Y-%m-%d') for d in v9.monthly_grid('2024-06-01', '2025-03-01')]

print("=== MIN_OBS_FLAT sweep: 2 vs 3 vs 4 ===\n")
print(f"  {'MIN_OBS':>7} | {'Ovrl WMAPE':>10} {'Ovrl Bias':>9} | {'FC_C n':>6} {'FC_C WMAPE':>10} {'FC_C Bias':>9} | {'HIST n':>6} {'HIST WMAPE':>10} {'HIST Bias':>9}")
print(f"  {'-'*7} | {'-'*10} {'-'*9} | {'-'*6} {'-'*10} {'-'*9} | {'-'*6} {'-'*10} {'-'*9}")

for min_obs in [2, 3, 4, 5]:
    orig = v9.MIN_OBS_FLAT
    v9.MIN_OBS_FLAT = min_obs
    all_preds = []
    for snap in snaps:
        _, pred = v9.run_snapshot(df, snap, 3, verbose=False)
        if pred is not None:
            pred['Snapshot'] = snap
            all_preds.append(pred)
    v9.MIN_OBS_FLAT = orig

    combined = pd.concat(all_preds, ignore_index=True)
    fc_c = combined[combined['Prediction_Source'].isin(v9.SOURCES_FC_CORRECTED)]
    hist = combined[combined['Prediction_Source'] == 'HISTORICAL']

    ow, _ = v9._wmape(combined['Actual_Sales'], combined['Prediction'])
    ob = v9._bias(combined['Actual_Sales'], combined['Prediction'])
    fw, fn = v9._wmape(fc_c['Actual_Sales'], fc_c['Prediction'])
    fb = v9._bias(fc_c['Actual_Sales'], fc_c['Prediction'])
    hw, hn = v9._wmape(hist['Actual_Sales'], hist['Prediction'])
    hb = v9._bias(hist['Actual_Sales'], hist['Prediction'])

    print(f"  {min_obs:>7} | {ow*100:>9.1f}% {ob*100:>+8.1f}% | {fn:>6} {fw*100:>9.1f}% {fb*100:>+8.1f}% | {hn:>6} {hw*100:>9.1f}% {hb*100:>+8.1f}%")

# Now detailed: what do the NEW rows (min=2 but not min=3) look like?
print("\n\n=== What rows does MIN_OBS=2 add that MIN_OBS=3 doesn't? ===\n")

all_m2 = []
all_m3 = []
for snap in snaps:
    v9.MIN_OBS_FLAT = 2
    _, p2 = v9.run_snapshot(df, snap, 3, verbose=False)
    v9.MIN_OBS_FLAT = 3
    _, p3 = v9.run_snapshot(df, snap, 3, verbose=False)

    p2['Snapshot'] = snap
    p3['Snapshot'] = snap
    all_m2.append(p2)
    all_m3.append(p3)

v9.MIN_OBS_FLAT = 3  # restore

c2 = pd.concat(all_m2, ignore_index=True)
c3 = pd.concat(all_m3, ignore_index=True)

fc2 = c2[c2['Prediction_Source'].isin(v9.SOURCES_FC_CORRECTED)]
fc3 = c3[c3['Prediction_Source'].isin(v9.SOURCES_FC_CORRECTED)]

# Find rows in fc2 that are NOT fc_corrected in c3
fc3_keys = set(zip(fc3['gsa_site'], fc3['Snapshot'], fc3['Target_Period_Start']))
new_mask = ~pd.Series(list(zip(fc2['gsa_site'], fc2['Snapshot'], fc2['Target_Period_Start']))).isin(fc3_keys)
new_rows = fc2[new_mask.values]
kept_rows = fc2[~new_mask.values]

print(f"FC_C with MIN=3: {len(fc3)} rows")
print(f"FC_C with MIN=2: {len(fc2)} rows")
print(f"NEW rows (gained by relaxing to 2): {len(new_rows)}")
print(f"Kept rows (FC_C in both): {len(kept_rows)}")
print()

if len(new_rows) > 0:
    nw, nn = v9._wmape(new_rows['Actual_Sales'], new_rows['Prediction'])
    nb = v9._bias(new_rows['Actual_Sales'], new_rows['Prediction'])
    print(f"NEW rows quality:  WMAPE={nw*100:.1f}%  Bias={nb*100:+.1f}%  n={nn}")
    print(f"                   Actual=${new_rows['Actual_Sales'].sum():,.0f}  Pred=${new_rows['Prediction'].sum():,.0f}")

    kw, kn = v9._wmape(kept_rows['Actual_Sales'], kept_rows['Prediction'])
    kb = v9._bias(kept_rows['Actual_Sales'], kept_rows['Prediction'])
    print(f"KEPT rows quality: WMAPE={kw*100:.1f}%  Bias={kb*100:+.1f}%  n={kn}")
    print()

    # Confidence flags on new rows
    for flag in ['HIGH_RISK', 'NORMAL']:
        sub = new_rows[new_rows['Confidence_Flag'] == flag]
        if len(sub) == 0:
            continue
        sw, sn = v9._wmape(sub['Actual_Sales'], sub['Prediction'])
        sb = v9._bias(sub['Actual_Sales'], sub['Prediction'])
        print(f"  NEW + {flag:<10}: n={len(sub):>4}  WMAPE={sw*100:.1f}%  Bias={sb*100:+.1f}%")

    # Per-site breakdown of new rows
    print(f"\n  Per-site breakdown of NEW rows (sorted by |dollar error|):")
    print(f"  {'Site':<30} {'Count':>5} {'Actual':>14} {'Predicted':>14} {'WMAPE':>8} {'Bias':>8}")
    print(f"  {'-'*30} {'-'*5} {'-'*14} {'-'*14} {'-'*8} {'-'*8}")
    site_rows = []
    for gs, grp in new_rows.groupby('gsa_site'):
        w, n = v9._wmape(grp['Actual_Sales'], grp['Prediction'])
        b = v9._bias(grp['Actual_Sales'], grp['Prediction'])
        derr = abs(grp['Prediction'].sum() - grp['Actual_Sales'].sum())
        site_rows.append((gs, len(grp), grp['Actual_Sales'].sum(), grp['Prediction'].sum(), w, b, derr))
    site_rows.sort(key=lambda x: -x[6])
    for gs, cnt, act, pred, w, b, _ in site_rows:
        print(f"  {gs:<30} {cnt:>5} {act:>14,.0f} {pred:>14,.0f} {w*100:>7.1f}% {b*100:>+7.1f}%")

# Also check: do existing MIN=3 rows get DIFFERENT predictions with MIN=2?
# (They shouldn't, since curves only get more keys, not different values for existing keys)
# But let's verify
print("\n=== Sanity check: do existing FC_C rows change predictions? ===")
merged = kept_rows[['gsa_site', 'Snapshot', 'Target_Period_Start', 'Prediction']].rename(
    columns={'Prediction': 'Pred_MIN2'})
m3_slim = fc3[['gsa_site', 'Snapshot', 'Target_Period_Start', 'Prediction']].rename(
    columns={'Prediction': 'Pred_MIN3'})
check = merged.merge(m3_slim, on=['gsa_site', 'Snapshot', 'Target_Period_Start'])
check['diff'] = (check['Pred_MIN2'] - check['Pred_MIN3']).abs()
print(f"Max absolute difference: ${check['diff'].max():,.2f}")
print(f"Rows with any difference: {(check['diff'] > 0.01).sum()}")
