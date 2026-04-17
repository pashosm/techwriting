"""Diagnostic: regression-based FC correction vs ratio-based correction.

At the Dec 2024 snapshot, compare:
1. Current approach: estimate = FC_value / bias / coverage (per-cell ratio)
2. Regression: Actual = a + b * FC_value (per-cell OLS with intercept)

Per-cell = (gsa_site, Timeframe, Prediction_Lag, site_regime)
"""
import aging_v9 as v9
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

df = v9.load_data(v9.DEFAULT_DATA_PATH)
snap = '2024-12-01'
snap_ts = pd.Timestamp(snap)
horizon_end = snap_ts + pd.DateOffset(months=3)

train = df[df['Target_Period_End'] < snap_ts].copy()
test = df[(df['Target_Period_Start'] >= snap_ts) & (df['Target_Period_Start'] < horizon_end) &
          (df['Reference_Month'] < snap_ts) & (df['Prediction_Lag'].isin(v9.TEST_LAGS))].copy()
visible = df[df['Reference_Month'] < snap_ts]
curve_train = train[train['Has_Forecast'] == 1]

# ============================================================
# Current approach (ratio-based)
# ============================================================
cov_curve = v9.build_flat_curves(curve_train, 'fc_coverage')
bias_curve = v9.build_flat_curves(curve_train, 'fc_bias')
hist = v9.build_historical(train)
registry = v9.build_forecast_registry(visible)
c_stats = v9.build_curve_stats(curve_train)

pred_current = v9.predict(test, cov_curve, bias_curve, hist, registry, curve_stats=c_stats)

# ============================================================
# Regression approach (per-cell OLS: Actual = a + b * FC_value)
# ============================================================

# Build regression models per cell
# Training data: rows with Has_Forecast=1 AND known actuals AND valid FC
reg_train = curve_train[
    curve_train['Forecast_Value'].notna() & (curve_train['Forecast_Value'] > 0) &
    curve_train['Actual_Sales'].notna() & (curve_train['Actual_Sales'] > 0)
].copy()

reg_models = {}
reg_stats = []
for (gs, tf, lag, regime), grp in reg_train.groupby(
        ['gsa_site', 'Timeframe', 'Prediction_Lag', 'site_regime']):
    if lag > v9.MAX_VINTAGE_LAG:
        continue
    if len(grp) < v9.MIN_OBS_FLAT:
        continue

    x = grp['Forecast_Value'].values
    y = grp['Actual_Sales'].values

    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(x, y)
    r2 = r_value ** 2

    reg_models[(gs, tf, lag, regime)] = {
        'slope': slope, 'intercept': intercept,
        'r2': r2, 'n': len(grp), 'p_value': p_value,
    }
    reg_stats.append({
        'gs': gs, 'tf': tf, 'lag': lag, 'regime': regime,
        'n': len(grp), 'slope': slope, 'intercept': intercept,
        'r2': r2, 'p_value': p_value,
        'x_mean': x.mean(), 'y_mean': y.mean(),
    })

rdf = pd.DataFrame(reg_stats)

print("=" * 90)
print(f"Regression Diagnostics (snapshot {snap})")
print("=" * 90)
print(f"\nCells with regression models: {len(reg_models)}")
print(f"  vs cells with ratio curves: {len(cov_curve)}")

print(f"\nRegression R² distribution:")
for p in [10, 25, 50, 75, 90]:
    print(f"  P{p:2d}: {rdf['r2'].quantile(p/100):.3f}")

print(f"\nSlope distribution (ideally near 1.0 if FC ~ Actual):")
for p in [10, 25, 50, 75, 90]:
    print(f"  P{p:2d}: {rdf['slope'].quantile(p/100):.3f}")

print(f"\nIntercept distribution:")
for p in [10, 25, 50, 75, 90]:
    print(f"  P{p:2d}: {rdf['intercept'].quantile(p/100):+,.0f}")

print(f"\nSignificant regressions (p < 0.05): {(rdf['p_value'] < 0.05).sum()} of {len(rdf)} ({(rdf['p_value'] < 0.05).mean()*100:.0f}%)")

# Show a few example regressions
print(f"\nExample regression models (lag=1, largest by n):")
lag1_reg = rdf[rdf['lag'] == 1].sort_values('n', ascending=False).head(10)
print(f"  {'Site':<30} {'n':>4} {'Slope':>7} {'Intercept':>14} {'R²':>6} {'p':>8}")
for _, r in lag1_reg.iterrows():
    print(f"  {r['gs']:<30} {r['n']:>4} {r['slope']:>7.3f} {r['intercept']:>+13,.0f} {r['r2']:>6.3f} {r['p_value']:>8.4f}")

# ============================================================
# Generate predictions using regression
# ============================================================
# Mirror the predict() logic but use regression instead of ratio correction

n = len(test)
reg_preds = np.full(n, np.nan)
reg_sources = np.empty(n, dtype=object)

for i, (_, row) in enumerate(test.iterrows()):
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

        valid_ests = []
        for v in cands:
            key = (v['gsa_site'], v['timeframe'], v['lag'], v['site_regime'])
            model = reg_models.get(key)
            if model is None:
                continue
            fc_val = v['fc_value']
            if fc_val <= 0:
                continue
            est = model['intercept'] + model['slope'] * fc_val
            est = max(est, 0)
            valid_ests.append((est, v['lag']))

        if valid_ests:
            combined = v9._combine_vintage_estimates(valid_ests)
            combined = max(combined, 0)
            reg_preds[i] = combined
            if len(valid_ests) >= 2:
                reg_sources[i] = 'FC_MULTI_CORRECTED'
            else:
                reg_sources[i] = 'FC_SINGLE_CORRECTED'
            continue

    h = hist.get((gs, tf), hist.get(('FB', gs)))
    if h is not None and h > 0:
        reg_preds[i] = h
        reg_sources[i] = 'HISTORICAL'

pred_reg = test.copy()
pred_reg['Prediction'] = reg_preds
pred_reg['Prediction_Source'] = reg_sources

# ============================================================
# Compare results
# ============================================================
print(f"\n\n{'='*90}")
print(f"Results Comparison (snapshot {snap}, 3-month horizon)")
print(f"{'='*90}")

for label, pdf in [("CURRENT (ratio)", pred_current), ("REGRESSION (OLS)", pred_reg)]:
    scored = pdf[pdf['Prediction'].notna()]
    fc_c = scored[scored['Prediction_Source'].isin(v9.SOURCES_FC_CORRECTED)]
    hi = scored[scored['Prediction_Source'] == 'HISTORICAL']

    print(f"\n  {label}:")
    if len(fc_c) > 0:
        w, _ = v9._wmape(fc_c['Actual_Sales'], fc_c['Prediction'])
        b = v9._bias(fc_c['Actual_Sales'], fc_c['Prediction'])
        print(f"    FC_Corrected: {len(fc_c):>4} rows, WMAPE {w*100:>5.1f}%, Bias {b*100:>+5.1f}%")
    if len(hi) > 0:
        w, _ = v9._wmape(hi['Actual_Sales'], hi['Prediction'])
        b = v9._bias(hi['Actual_Sales'], hi['Prediction'])
        print(f"    Historical:   {len(hi):>4} rows, WMAPE {w*100:>5.1f}%, Bias {b*100:>+5.1f}%")
    if len(scored) > 0:
        w, _ = v9._wmape(scored['Actual_Sales'], scored['Prediction'])
        b = v9._bias(scored['Actual_Sales'], scored['Prediction'])
        print(f"    Overall:      {len(scored):>4} rows, WMAPE {w*100:>5.1f}%, Bias {b*100:>+5.1f}%")

# ============================================================
# Head-to-head on FC_Corrected rows
# ============================================================
print(f"\n{'='*90}")
print("Head-to-Head: FC_Corrected rows")
print(f"{'='*90}")

cur_fc = pred_current[pred_current['Prediction_Source'].isin(v9.SOURCES_FC_CORRECTED)][
    ['gsa_site', 'Target_Period_Start', 'Actual_Sales', 'Prediction']].rename(
    columns={'Prediction': 'Pred_Current'})
reg_fc = pred_reg[pred_reg['Prediction_Source'].isin(v9.SOURCES_FC_CORRECTED)][
    ['gsa_site', 'Target_Period_Start', 'Prediction']].rename(
    columns={'Prediction': 'Pred_Reg'})

hh = cur_fc.merge(reg_fc, on=['gsa_site', 'Target_Period_Start'], how='inner')
hh['cur_err'] = np.abs(hh['Pred_Current'] - hh['Actual_Sales'])
hh['reg_err'] = np.abs(hh['Pred_Reg'] - hh['Actual_Sales'])
hh['reg_better'] = hh['reg_err'] < hh['cur_err']

print(f"\n  Common FC_C rows: {len(hh)}")
print(f"  Regression wins: {hh['reg_better'].sum()} ({hh['reg_better'].mean()*100:.0f}%)")
print(f"  Current wins:    {(~hh['reg_better']).sum()} ({(~hh['reg_better']).mean()*100:.0f}%)")
print(f"  Regression total |err|: ${hh['reg_err'].sum():,.0f}")
print(f"  Current total |err|:    ${hh['cur_err'].sum():,.0f}")
dollar_diff = hh['reg_err'].sum() - hh['cur_err'].sum()
print(f"  Dollar diff:             ${dollar_diff:+,.0f} ({'regression worse' if dollar_diff > 0 else 'regression better'})")

# Also check: rows that ONLY regression covers (or only current covers)
reg_only = pred_reg[pred_reg['Prediction_Source'].isin(v9.SOURCES_FC_CORRECTED) &
                    ~pred_reg.index.isin(pred_current[pred_current['Prediction_Source'].isin(v9.SOURCES_FC_CORRECTED)].index)]
cur_only = pred_current[pred_current['Prediction_Source'].isin(v9.SOURCES_FC_CORRECTED) &
                        ~pred_current.index.isin(pred_reg[pred_reg['Prediction_Source'].isin(v9.SOURCES_FC_CORRECTED)].index)]

if len(reg_only) > 0:
    print(f"\n  Rows only regression covers: {len(reg_only)}")
if len(cur_only) > 0:
    print(f"  Rows only current covers: {len(cur_only)}")

# ============================================================
# Per-site breakdown for FC_C rows
# ============================================================
print(f"\n{'='*90}")
print("Per-Site Comparison (FC_Corrected, sorted by dollar error difference)")
print(f"{'='*90}")

if len(hh) > 0:
    site_comp = hh.groupby('gsa_site').agg(
        n=('Actual_Sales', 'count'),
        actual=('Actual_Sales', 'sum'),
        cur_err=('cur_err', 'sum'),
        reg_err=('reg_err', 'sum'),
        cur_pred=('Pred_Current', 'sum'),
        reg_pred=('Pred_Reg', 'sum'),
    )
    site_comp['cur_wmape'] = site_comp['cur_err'] / site_comp['actual'] * 100
    site_comp['reg_wmape'] = site_comp['reg_err'] / site_comp['actual'] * 100
    site_comp['err_diff'] = site_comp['reg_err'] - site_comp['cur_err']
    site_comp = site_comp.sort_values('err_diff')

    print(f"\n  {'Site':<30} {'n':>3} {'Actual':>14} {'Cur WMAPE':>10} {'Reg WMAPE':>10} {'Err Diff ($)':>14} {'Winner':>10}")
    print(f"  {'-'*30} {'-'*3} {'-'*14} {'-'*10} {'-'*10} {'-'*14} {'-'*10}")
    for gs, r in site_comp.iterrows():
        winner = 'Reg' if r['err_diff'] < 0 else 'Current'
        print(f"  {gs:<30} {r['n']:>3} ${r['actual']:>12,.0f} {r['cur_wmape']:>9.1f}% {r['reg_wmape']:>9.1f}% ${r['err_diff']:>+12,.0f} {winner:>10}")
