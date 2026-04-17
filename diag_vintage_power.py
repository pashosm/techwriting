"""Diagnostic: optimal vintage recency power for multi-vintage combination.

Current: weight = 1 / lag^1.5
Test powers from 0 (equal weight) to 4.0 (heavily favor recent) across
the full 10-snapshot grid. Also test lag^inf (most recent vintage only).
"""
import aging_v9 as v9
import numpy as np
import pandas as pd

df = v9.load_data(v9.DEFAULT_DATA_PATH)
snaps = [d.strftime('%Y-%m-%d') for d in v9.monthly_grid('2024-06-01', '2025-03-01')]

POWERS_TO_TEST = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 'most_recent']


def predict_with_power(test_df, cov_curve, bias_curve, hist, registry, curve_stats, power):
    """Like v9.predict but with configurable vintage recency power."""
    n = len(test_df)
    preds = np.full(n, np.nan)
    sources = np.empty(n, dtype=object)

    for i, (_, row) in enumerate(test_df.iterrows()):
        gs, tf, ref_m = row['gsa_site'], row['Timeframe'], row['Reference_Month']
        tp_key = (gs, row['Target_Period_Start'], row['Target_Period_End'])

        if tp_key in registry:
            cands = []
            for v in registry[tp_key]:
                if v['ref_month'] > ref_m:
                    continue
                if v['lag'] > v9.MAX_VINTAGE_LAG:
                    continue
                if v['forecast_error']:
                    continue
                cands.append(v)
            cands.sort(key=lambda v: v['lag'])

            valid = []
            for v in cands:
                est, ok, corrected, _ = v9._vintage_to_estimate(v, cov_curve, bias_curve, curve_stats)
                if ok:
                    valid.append((max(est, 0), v['lag'], corrected))

            if valid:
                if power == 'most_recent':
                    combined = valid[0][0]
                else:
                    estimates = [(e, l) for e, l, _ in valid]
                    if power == 0.0:
                        combined = np.mean([e for e, l in estimates])
                    else:
                        weights = [1.0 / (l ** power) for e, l in estimates]
                        total_w = sum(weights)
                        combined = sum(e * w for (e, l), w in zip(estimates, weights)) / total_w
                combined = max(combined, 0)
                preds[i] = combined
                any_c = any(c for _, _, c in valid)
                if len(valid) >= 2:
                    sources[i] = 'FC_MULTI_CORRECTED' if any_c else 'FC_MULTI_RAW'
                else:
                    sources[i] = 'FC_SINGLE_CORRECTED' if any_c else 'FC_SINGLE_RAW'
                continue

        h = hist.get((gs, tf), hist.get(('FB', gs)))
        if h is not None and h > 0:
            preds[i] = h
            sources[i] = 'HISTORICAL'

    out = test_df.copy()
    out['Prediction'] = preds
    out['Prediction_Source'] = sources
    return out


# ============================================================
# Run all powers across all snapshots
# ============================================================
print("=" * 110)
print(f"Vintage Recency Power Sweep (MAX_VINTAGE_LAG={v9.MAX_VINTAGE_LAG})")
print("=" * 110)

power_results = {p: [] for p in POWERS_TO_TEST}

for snap in snaps:
    snap_ts = pd.Timestamp(snap)
    horizon_end = snap_ts + pd.DateOffset(months=3)
    train = df[df['Target_Period_End'] < snap_ts].copy()
    test = df[(df['Target_Period_Start'] >= snap_ts) & (df['Target_Period_Start'] < horizon_end) &
              (df['Reference_Month'] < snap_ts) & (df['Prediction_Lag'].isin(v9.TEST_LAGS))].copy()
    visible = df[df['Reference_Month'] < snap_ts]
    curve_train = train[train['Has_Forecast'] == 1]
    if len(train) == 0 or len(test) == 0:
        continue

    cov_curve = v9.build_flat_curves(curve_train, 'fc_coverage')
    bias_curve = v9.build_flat_curves(curve_train, 'fc_bias')
    hist = v9.build_historical(train)
    registry = v9.build_forecast_registry(visible)
    c_stats = v9.build_curve_stats(curve_train)

    for power in POWERS_TO_TEST:
        pred = predict_with_power(test, cov_curve, bias_curve, hist, registry, c_stats, power)
        pred['Snapshot'] = snap
        power_results[power].append(pred)

# ============================================================
# Summary table
# ============================================================
print(f"\n{'--- FC_Corrected Only ---':^110}")
print(f"\n  {'Power':<14} | {'Rows':>6} {'WMAPE':>8} {'Bias':>8} {'$ Error':>16} | {'vs 1.5':>10}")
print(f"  {'-'*14} | {'-'*6} {'-'*8} {'-'*8} {'-'*16} | {'-'*10}")

baseline_err = None
for power in POWERS_TO_TEST:
    all_pred = pd.concat(power_results[power], ignore_index=True)
    fc = all_pred[all_pred['Prediction_Source'].isin(v9.SOURCES_FC_CORRECTED)]
    if len(fc) == 0:
        continue
    w, _ = v9._wmape(fc['Actual_Sales'], fc['Prediction'])
    b = v9._bias(fc['Actual_Sales'], fc['Prediction'])
    err = np.abs(fc['Prediction'] - fc['Actual_Sales']).sum()

    if power == 1.5:
        baseline_err = err

    diff_str = ''
    if baseline_err is not None:
        diff = err - baseline_err
        diff_str = f"${diff:+,.0f}"

    p_label = 'most_recent' if power == 'most_recent' else f'{power:.1f}'
    marker = ' <-- current' if power == 1.5 else ''
    print(f"  {p_label:<14} | {len(fc):>6} {w*100:>7.1f}% {b*100:>+7.1f}% ${err:>14,.0f} | {diff_str:>10}{marker}")

# ============================================================
# Per-snapshot detail for key powers
# ============================================================
key_powers = [0.0, 1.0, 1.5, 2.0, 3.0, 'most_recent']
print(f"\n\n{'--- Per-Snapshot WMAPE (FC_Corrected) ---':^110}")
header = f"  {'Snapshot':<12} |"
for p in key_powers:
    p_label = 'most_rec' if p == 'most_recent' else f'p={p}'
    header += f" {p_label:>8}"
print(header)
print(f"  {'-'*12} |" + f" {'-'*8}" * len(key_powers))

for snap in snaps:
    line = f"  {snap:<12} |"
    for power in key_powers:
        all_pred = pd.concat(power_results[power], ignore_index=True)
        fc = all_pred[(all_pred['Snapshot'] == snap) & all_pred['Prediction_Source'].isin(v9.SOURCES_FC_CORRECTED)]
        if len(fc) == 0:
            line += f" {'N/A':>8}"
            continue
        w, _ = v9._wmape(fc['Actual_Sales'], fc['Prediction'])
        line += f" {w*100:>7.1f}%"
    print(line)

# Overall row
line = f"  {'OVERALL':<12} |"
for power in key_powers:
    all_pred = pd.concat(power_results[power], ignore_index=True)
    fc = all_pred[all_pred['Prediction_Source'].isin(v9.SOURCES_FC_CORRECTED)]
    w, _ = v9._wmape(fc['Actual_Sales'], fc['Prediction'])
    line += f" {w*100:>7.1f}%"
print(line)

# ============================================================
# Per-snapshot BIAS
# ============================================================
print(f"\n\n{'--- Per-Snapshot Bias (FC_Corrected) ---':^110}")
header = f"  {'Snapshot':<12} |"
for p in key_powers:
    p_label = 'most_rec' if p == 'most_recent' else f'p={p}'
    header += f" {p_label:>8}"
print(header)
print(f"  {'-'*12} |" + f" {'-'*8}" * len(key_powers))

for snap in snaps:
    line = f"  {snap:<12} |"
    for power in key_powers:
        all_pred = pd.concat(power_results[power], ignore_index=True)
        fc = all_pred[(all_pred['Snapshot'] == snap) & all_pred['Prediction_Source'].isin(v9.SOURCES_FC_CORRECTED)]
        if len(fc) == 0:
            line += f" {'N/A':>8}"
            continue
        b = v9._bias(fc['Actual_Sales'], fc['Prediction'])
        line += f" {b*100:>+7.1f}%"
    print(line)

line = f"  {'OVERALL':<12} |"
for power in key_powers:
    all_pred = pd.concat(power_results[power], ignore_index=True)
    fc = all_pred[all_pred['Prediction_Source'].isin(v9.SOURCES_FC_CORRECTED)]
    b = v9._bias(fc['Actual_Sales'], fc['Prediction'])
    line += f" {b*100:>+7.1f}%"
print(line)

# ============================================================
# Overall (all sources) comparison
# ============================================================
print(f"\n\n{'--- Overall (All Sources) ---':^110}")
print(f"\n  {'Power':<14} | {'Rows':>6} {'WMAPE':>8} {'Bias':>8}")
print(f"  {'-'*14} | {'-'*6} {'-'*8} {'-'*8}")

for power in POWERS_TO_TEST:
    all_pred = pd.concat(power_results[power], ignore_index=True)
    scored = all_pred[all_pred['Prediction'].notna()]
    w, _ = v9._wmape(scored['Actual_Sales'], scored['Prediction'])
    b = v9._bias(scored['Actual_Sales'], scored['Prediction'])
    p_label = 'most_recent' if power == 'most_recent' else f'{power:.1f}'
    marker = ' <-- current' if power == 1.5 else ''
    print(f"  {p_label:<14} | {len(scored):>6} {w*100:>7.1f}% {b*100:>+7.1f}%{marker}")
