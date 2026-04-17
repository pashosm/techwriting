"""Diagnostic: is curve drift detectable and correctable?

For each cell (gs, tf, lag, regime), measure:
1. Trend slope of bias*coverage product over time
2. R² of that trend (systematic drift vs noise)
3. Whether a trend-adjusted correction improves predictions
"""
import aging_v9 as v9
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

df = v9.load_data(v9.DEFAULT_DATA_PATH)
snaps = [d.strftime('%Y-%m-%d') for d in v9.monthly_grid('2024-06-01', '2025-03-01')]

# ============================================================
# Part 1: Measure drift across all cells at a representative snapshot
# ============================================================
print("=" * 90)
print("Part 1: How much drift exists in bias*coverage across cells?")
print("=" * 90)

snap = '2024-12-01'
snap_ts = pd.Timestamp(snap)
train = df[(df['Target_Period_End'] < snap_ts) & (df['Has_Forecast'] == 1)].copy()
train['bc_product'] = train['fc_bias'] * train['fc_coverage']
train = train[train['bc_product'].notna() & (train['bc_product'] > 0) & (train['bc_product'] < 10)]

cell_drift = []
for (gs, tf, lag, regime), grp in train.groupby(
        ['gsa_site', 'Timeframe', 'Prediction_Lag', 'site_regime']):
    if lag > v9.MAX_VINTAGE_LAG:
        continue
    if len(grp) < v9.MIN_OBS_FLAT:
        continue
    # Time axis in months from earliest observation
    t = (grp['Reference_Month'] - grp['Reference_Month'].min()).dt.days.values / 30.0
    y = grp['bc_product'].values

    if t.max() - t.min() < 3:  # need at least 3 months span
        continue

    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(t, y)
    r2 = r_value ** 2

    # Recency-weighted mean (what the current model uses)
    wts = v9._recency_weights(grp['Reference_Month'])
    wmean = np.average(y, weights=wts)

    # Latest value vs mean
    latest = y[-3:].mean() if len(y) >= 3 else y[-1]

    cell_drift.append({
        'gs': gs, 'tf': tf, 'lag': lag, 'regime': regime,
        'n': len(grp), 'span_months': t.max(),
        'slope_per_month': slope,
        'r2': r2, 'p_value': p_value,
        'wmean': wmean, 'latest_3': latest,
        'first_3': y[:3].mean() if len(y) >= 3 else y[0],
        'total_drift': (latest - (y[:3].mean() if len(y) >= 3 else y[0])),
        'pct_drift': (latest - (y[:3].mean() if len(y) >= 3 else y[0])) / (y[:3].mean() if len(y) >= 3 else y[0]) * 100,
    })

cdf = pd.DataFrame(cell_drift)
print(f"\nCells with >= {v9.MIN_OBS_FLAT} obs and >= 3 months span: {len(cdf)}")
print(f"  Lag 1: {len(cdf[cdf['lag']==1])},  Lag 2: {len(cdf[cdf['lag']==2])},  Lag 3: {len(cdf[cdf['lag']==3])}")

print(f"\nSlope distribution (per month):")
for p in [10, 25, 50, 75, 90]:
    print(f"  P{p:2d}: {cdf['slope_per_month'].quantile(p/100):+.4f}")

print(f"\nR² distribution (how systematic is the drift):")
for p in [10, 25, 50, 75, 90]:
    print(f"  P{p:2d}: {cdf['r2'].quantile(p/100):.3f}")

print(f"\nStatistically significant drift (p < 0.05): {(cdf['p_value'] < 0.05).sum()} of {len(cdf)} ({(cdf['p_value'] < 0.05).mean()*100:.0f}%)")
print(f"Significant + R² > 0.3: {((cdf['p_value'] < 0.05) & (cdf['r2'] > 0.3)).sum()}")
print(f"Significant + R² > 0.5: {((cdf['p_value'] < 0.05) & (cdf['r2'] > 0.5)).sum()}")

print(f"\nPct drift (latest 3 obs vs first 3 obs):")
for p in [10, 25, 50, 75, 90]:
    print(f"  P{p:2d}: {cdf['pct_drift'].quantile(p/100):+.1f}%")

# Show the high-drift lag-1 cells
print(f"\nLag-1 cells with |pct_drift| > 20% and R² > 0.3:")
high_drift = cdf[(cdf['lag'] == 1) & (cdf['pct_drift'].abs() > 20) & (cdf['r2'] > 0.3)]
high_drift = high_drift.sort_values('pct_drift')
print(f"  {'Site':<30} {'n':>4} {'Span':>5} {'First3':>7} {'Last3':>7} {'Drift%':>7} {'R²':>5} {'Slope/mo':>9}")
for _, r in high_drift.iterrows():
    print(f"  {r['gs']:<30} {r['n']:>4} {r['span_months']:>4.0f}m {r['first_3']:>7.3f} {r['latest_3']:>7.3f} {r['pct_drift']:>+6.1f}% {r['r2']:>5.3f} {r['slope_per_month']:>+9.5f}")


# ============================================================
# Part 2: Does a trend-adjusted curve improve predictions?
# ============================================================
print("\n\n" + "=" * 90)
print("Part 2: Trend-adjusted correction vs current (recency-weighted mean)")
print("=" * 90)


def build_trend_curves(train_df, ratio_col):
    """Like build_flat_curves but extrapolates the trend to the prediction date."""
    curve = {}
    for (gs, tf, lag, regime), grp in train_df.groupby(
            ['gsa_site', 'Timeframe', 'Prediction_Lag', 'site_regime']):
        valid = grp[grp[ratio_col].notna() & (grp[ratio_col] > 0) & (grp[ratio_col] < 10)]
        if len(valid) < v9.MIN_OBS_FLAT:
            continue

        t = (valid['Reference_Month'] - valid['Reference_Month'].min()).dt.days.values / 30.0
        y = valid[ratio_col].values

        # Always store the recency-weighted mean as fallback
        wts = v9._recency_weights(valid['Reference_Month'])
        wmean = np.average(y, weights=wts)

        if len(valid) >= 4 and (t.max() - t.min()) >= 3:
            slope, intercept, r_value, p_value, _ = sp_stats.linregress(t, y)
            r2 = r_value ** 2
            if p_value < 0.10 and r2 > 0.20:
                # Extrapolate to 1 month beyond last observation
                t_pred = t.max() + 1.0
                trend_val = intercept + slope * t_pred
                # Clamp to reasonable range (don't let it go negative or explode)
                trend_val = max(trend_val, wmean * 0.3)
                trend_val = min(trend_val, wmean * 3.0)
                curve[(gs, tf, lag, regime)] = trend_val
                continue

        curve[(gs, tf, lag, regime)] = wmean
    return curve


# Run both approaches across all snapshots
results_current = []
results_trend = []

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

    # Current approach
    cov_c = v9.build_flat_curves(curve_train, 'fc_coverage')
    bias_c = v9.build_flat_curves(curve_train, 'fc_bias')

    # Trend approach
    cov_t = build_trend_curves(curve_train, 'fc_coverage')
    bias_t = build_trend_curves(curve_train, 'fc_bias')

    hist = v9.build_historical(train)
    registry = v9.build_forecast_registry(visible)
    c_stats = v9.build_curve_stats(curve_train)

    pred_c = v9.predict(test, cov_c, bias_c, hist, registry, c_stats)
    pred_t = v9.predict(test, cov_t, bias_t, hist, registry, c_stats)

    pred_c['Snapshot'] = snap
    pred_t['Snapshot'] = snap
    results_current.append(pred_c)
    results_trend.append(pred_t)

rc = pd.concat(results_current, ignore_index=True)
rt = pd.concat(results_trend, ignore_index=True)

fcc = rc[rc['Prediction_Source'].isin(v9.SOURCES_FC_CORRECTED)]
fct = rt[rt['Prediction_Source'].isin(v9.SOURCES_FC_CORRECTED)]

print(f"\nPer-snapshot comparison (FC_Corrected only):")
print(f"  {'Snapshot':<12} | {'Current n':>9} {'Cur WMAPE':>9} {'Cur Bias':>9} | {'Trend n':>9} {'Trd WMAPE':>9} {'Trd Bias':>9} | {'WMAPE diff':>10}")
print(f"  {'-'*12} | {'-'*9} {'-'*9} {'-'*9} | {'-'*9} {'-'*9} {'-'*9} | {'-'*10}")

for snap in snaps:
    sc = fcc[fcc['Snapshot'] == snap]
    st = fct[fct['Snapshot'] == snap]
    if len(sc) == 0:
        continue
    cw, _ = v9._wmape(sc['Actual_Sales'], sc['Prediction'])
    cb = v9._bias(sc['Actual_Sales'], sc['Prediction'])
    tw, _ = v9._wmape(st['Actual_Sales'], st['Prediction'])
    tb = v9._bias(st['Actual_Sales'], st['Prediction'])
    diff = tw - cw
    print(f"  {snap:<12} | {len(sc):>9} {cw*100:>8.1f}% {cb*100:>+8.1f}% | {len(st):>9} {tw*100:>8.1f}% {tb*100:>+8.1f}% | {diff*100:>+9.1f}pp")

cw_all, _ = v9._wmape(fcc['Actual_Sales'], fcc['Prediction'])
cb_all = v9._bias(fcc['Actual_Sales'], fcc['Prediction'])
tw_all, _ = v9._wmape(fct['Actual_Sales'], fct['Prediction'])
tb_all = v9._bias(fct['Actual_Sales'], fct['Prediction'])
print(f"  {'-'*12} | {'-'*9} {'-'*9} {'-'*9} | {'-'*9} {'-'*9} {'-'*9} |")
print(f"  {'OVERALL':<12} | {len(fcc):>9} {cw_all*100:>8.1f}% {cb_all*100:>+8.1f}% | {len(fct):>9} {tw_all*100:>8.1f}% {tb_all*100:>+8.1f}% | {(tw_all-cw_all)*100:>+9.1f}pp")

# Head-to-head per row
merged = fcc[['gsa_site', 'Snapshot', 'Target_Period_Start', 'Actual_Sales', 'Prediction']].rename(
    columns={'Prediction': 'Pred_Current'})
trend_slim = fct[['gsa_site', 'Snapshot', 'Target_Period_Start', 'Prediction']].rename(
    columns={'Prediction': 'Pred_Trend'})
hh = merged.merge(trend_slim, on=['gsa_site', 'Snapshot', 'Target_Period_Start'])
hh['cur_err'] = np.abs(hh['Pred_Current'] - hh['Actual_Sales'])
hh['trd_err'] = np.abs(hh['Pred_Trend'] - hh['Actual_Sales'])
hh['trend_better'] = hh['trd_err'] < hh['cur_err']
hh['changed'] = np.abs(hh['Pred_Current'] - hh['Pred_Trend']) > 1.0

changed = hh[hh['changed']]
print(f"\nHead-to-head ({len(changed)} rows where trend changed the prediction):")
print(f"  Trend wins:   {changed['trend_better'].sum()} ({changed['trend_better'].mean()*100:.0f}%)")
print(f"  Current wins: {(~changed['trend_better']).sum()} ({(~changed['trend_better']).mean()*100:.0f}%)")
print(f"  Trend total |err|:   ${changed['trd_err'].sum():,.0f}")
print(f"  Current total |err|: ${changed['cur_err'].sum():,.0f}")
dollar_diff = changed['trd_err'].sum() - changed['cur_err'].sum()
print(f"  Dollar diff:         ${dollar_diff:+,.0f} ({'trend worse' if dollar_diff > 0 else 'trend better'})")
