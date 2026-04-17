"""Diagnostic: does multi-vintage combination add value at MAX_LAG=3?"""
import aging_v9 as v9
import numpy as np
import pandas as pd

df = v9.load_data(v9.DEFAULT_DATA_PATH)
snaps = [d.strftime('%Y-%m-%d') for d in v9.monthly_grid('2024-06-01', '2025-03-01')]

def run_variant(df, snaps, max_lag, top_n):
    all_preds = []
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
        c_stats = v9.build_curve_stats(curve_train)
        hist = v9.build_historical(train)
        registry = v9.build_forecast_registry(visible)

        n = len(test)
        preds = np.full(n, np.nan)
        sources = np.empty(n, dtype=object)
        n_vint = np.zeros(n, dtype=int)

        for i, (_, row) in enumerate(test.iterrows()):
            gs, tf, ref_m = row['gsa_site'], row['Timeframe'], row['Reference_Month']
            tp_key = (gs, row['Target_Period_Start'], row['Target_Period_End'])
            if tp_key in registry:
                cands = []
                for v in registry[tp_key]:
                    if v['ref_month'] > ref_m: continue
                    if v['lag'] > max_lag: continue
                    if v['forecast_error']: continue
                    cands.append(v)
                cands.sort(key=lambda v: v['lag'])
                cands = cands[:top_n]
                valid = []
                for v in cands:
                    est, ok, corrected, _ = v9._vintage_to_estimate(v, cov_curve, bias_curve, c_stats)
                    if ok:
                        valid.append((max(est, 0), v['lag'], corrected))
                if valid:
                    n_vint[i] = len(valid)
                    combined = max(v9._combine_vintage_estimates([(e, l) for e, l, _ in valid]), 0)
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
        out = test.copy()
        out['Prediction'] = preds
        out['Prediction_Source'] = sources
        out['N_Vintages'] = n_vint
        out['Snapshot'] = snap
        all_preds.append(out)
    return pd.concat(all_preds, ignore_index=True)


multi = run_variant(df, snaps, max_lag=3, top_n=99)
single = run_variant(df, snaps, max_lag=3, top_n=1)

print("=== MAX_LAG=3: Multi-vintage vs Most-recent-only, per snapshot ===")
hdr = f"  {'Snapshot':<12} | {'Multi n':>7} {'M WMAPE':>8} {'M Bias':>8} | {'Single n':>8} {'S WMAPE':>8} {'S Bias':>8} | {'Winner':>8}"
print(hdr)
print(f"  {'-'*12} | {'-'*7} {'-'*8} {'-'*8} | {'-'*8} {'-'*8} {'-'*8} | {'-'*8}")

for snap in snaps:
    m = multi[(multi['Snapshot'] == snap) & multi['Prediction_Source'].isin(v9.SOURCES_FC_CORRECTED)]
    s = single[(single['Snapshot'] == snap) & single['Prediction_Source'].isin(v9.SOURCES_FC_CORRECTED)]
    mw, _ = v9._wmape(m['Actual_Sales'], m['Prediction'])
    mb = v9._bias(m['Actual_Sales'], m['Prediction'])
    sw, _ = v9._wmape(s['Actual_Sales'], s['Prediction'])
    sb = v9._bias(s['Actual_Sales'], s['Prediction'])
    winner = 'Multi' if mw < sw else 'Single'
    print(f"  {snap:<12} | {len(m):>7} {mw*100:>7.1f}% {mb*100:>+7.1f}% | {len(s):>8} {sw*100:>7.1f}% {sb*100:>+7.1f}% | {winner:>8}")

mc = multi[multi['Prediction_Source'].isin(v9.SOURCES_FC_CORRECTED)]
sc = single[single['Prediction_Source'].isin(v9.SOURCES_FC_CORRECTED)]
mw, _ = v9._wmape(mc['Actual_Sales'], mc['Prediction'])
mb = v9._bias(mc['Actual_Sales'], mc['Prediction'])
sw, _ = v9._wmape(sc['Actual_Sales'], sc['Prediction'])
sb = v9._bias(sc['Actual_Sales'], sc['Prediction'])
print(f"  {'-'*12} | {'-'*7} {'-'*8} {'-'*8} | {'-'*8} {'-'*8} {'-'*8} |")
print(f"  {'OVERALL':<12} | {len(mc):>7} {mw*100:>7.1f}% {mb*100:>+7.1f}% | {len(sc):>8} {sw*100:>7.1f}% {sb*100:>+7.1f}% |")
print()

# Head-to-head: for rows that exist in both, did combination help?
merged = multi[multi['Prediction_Source'].isin(v9.SOURCES_FC_CORRECTED)][
    ['gsa_site', 'Snapshot', 'Target_Period_Start', 'Actual_Sales', 'Prediction', 'N_Vintages']].copy()
merged = merged.rename(columns={'Prediction': 'Multi_Pred', 'N_Vintages': 'Multi_NV'})
sing_slim = single[single['Prediction_Source'].isin(v9.SOURCES_FC_CORRECTED)][
    ['gsa_site', 'Snapshot', 'Target_Period_Start', 'Prediction']].copy()
sing_slim = sing_slim.rename(columns={'Prediction': 'Single_Pred'})
joined = merged.merge(sing_slim, on=['gsa_site', 'Snapshot', 'Target_Period_Start'], how='inner')
joined['multi_err'] = np.abs(joined['Multi_Pred'] - joined['Actual_Sales'])
joined['single_err'] = np.abs(joined['Single_Pred'] - joined['Actual_Sales'])
joined['multi_better'] = joined['multi_err'] < joined['single_err']

# All rows that exist in both
print(f"Head-to-head (all {len(joined)} common rows):")
print(f"  Multi wins:  {joined['multi_better'].sum()} ({joined['multi_better'].mean()*100:.0f}%)")
print(f"  Single wins: {(~joined['multi_better']).sum()} ({(~joined['multi_better']).mean()*100:.0f}%)")
print(f"  Multi total |err|:  ${joined['multi_err'].sum():,.0f}")
print(f"  Single total |err|: ${joined['single_err'].sum():,.0f}")
dollar_diff = joined['multi_err'].sum() - joined['single_err'].sum()
print(f"  Dollar diff:        ${dollar_diff:+,.0f} ({'multi worse' if dollar_diff > 0 else 'multi better'})")
print()

# Subset: only rows where multi actually used >1 vintage
multi_only = joined[joined['Multi_NV'] > 1]
print(f"Head-to-head (only {len(multi_only)} rows where multi used >1 vintage):")
print(f"  Multi wins:  {multi_only['multi_better'].sum()} ({multi_only['multi_better'].mean()*100:.0f}%)")
print(f"  Single wins: {(~multi_only['multi_better']).sum()} ({(~multi_only['multi_better']).mean()*100:.0f}%)")
print(f"  Multi total |err|:  ${multi_only['multi_err'].sum():,.0f}")
print(f"  Single total |err|: ${multi_only['single_err'].sum():,.0f}")
dollar_diff2 = multi_only['multi_err'].sum() - multi_only['single_err'].sum()
print(f"  Dollar diff:        ${dollar_diff2:+,.0f} ({'multi worse' if dollar_diff2 > 0 else 'multi better'})")
print()

# Breakdown: which vintages does multi add?
print("Vintage count distribution (multi, MAX_LAG=3):")
vc = mc['N_Vintages'].value_counts().sort_index()
for nv, cnt in vc.items():
    print(f"  {nv} vintage(s): {cnt} rows")
