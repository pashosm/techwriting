"""Diagnostic: customer-pooled fallback with guardrails across all snapshots.

Guardrails:
- N_sites >= 2 in the customer pool
- Within-customer CV < 0.30

Run across full 10-snapshot grid to validate.
"""
import aging_v9 as v9
import numpy as np
import pandas as pd

df = v9.load_data(v9.DEFAULT_DATA_PATH)
snaps = [d.strftime('%Y-%m-%d') for d in v9.monthly_grid('2024-06-01', '2025-03-01')]

POOL_MIN_SITES = 2
POOL_MAX_CV = 0.30

def build_customer_pool(cov_curve, bias_curve, curve_stats):
    """Build customer-level pooled bias/coverage curves with metadata."""
    pool_rows = []
    for (gs, tf, lag, regime), cov in cov_curve.items():
        bias = bias_curve.get((gs, tf, lag, regime))
        if bias is None:
            continue
        customer = gs.split('|')[0]
        stats = curve_stats.get((gs, tf, lag, regime))
        n = stats['n'] if stats else v9.MIN_OBS_FLAT
        pool_rows.append({
            'customer': customer, 'site': gs, 'tf': tf, 'lag': lag, 'regime': regime,
            'bias': bias, 'coverage': cov, 'n': n,
        })

    if not pool_rows:
        return {}, {}, {}

    pool_df = pd.DataFrame(pool_rows)
    cust_cov, cust_bias, cust_meta = {}, {}, {}

    for (cust, tf, lag, regime), grp in pool_df.groupby(['customer', 'tf', 'lag', 'regime']):
        if len(grp) < 1:
            continue
        weights = grp['n'].values.astype(float)
        w_bias = np.average(grp['bias'].values, weights=weights)
        w_cov = np.average(grp['coverage'].values, weights=weights)
        bias_cv = grp['bias'].std() / grp['bias'].mean() if grp['bias'].mean() > 0 and len(grp) > 1 else 0.0
        cov_cv = grp['coverage'].std() / grp['coverage'].mean() if grp['coverage'].mean() > 0 and len(grp) > 1 else 0.0

        cust_cov[(cust, tf, lag, regime)] = w_cov
        cust_bias[(cust, tf, lag, regime)] = w_bias
        cust_meta[(cust, tf, lag, regime)] = {
            'n_sites': grp['site'].nunique(),
            'bc_cv': max(bias_cv, cov_cv),
            'total_obs': grp['n'].sum(),
        }
    return cust_cov, cust_bias, cust_meta


def predict_with_pool(test_df, cov_curve, bias_curve, hist, registry, curve_stats,
                      cust_cov, cust_bias, cust_meta):
    """Like v9.predict but falls back to customer-pooled correction with guardrails."""
    n = len(test_df)
    preds = np.full(n, np.nan)
    sources = np.empty(n, dtype=object)

    for i, (_, row) in enumerate(test_df.iterrows()):
        gs, tf, ref_m = row['gsa_site'], row['Timeframe'], row['Reference_Month']
        customer = gs.split('|')[0]
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
            used_pool = False

            for v in cands:
                fc_val = v['fc_value']
                if fc_val <= 0:
                    continue
                key = (v['gsa_site'], v['timeframe'], v['lag'], v['site_regime'])
                cf = cov_curve.get(key)
                bf = bias_curve.get(key)

                if cf and bf and cf > 0.001 and bf > 0.001:
                    est = fc_val / bf / cf
                    valid_ests.append((max(est, 0), v['lag']))
                else:
                    cust_key = (customer, v['timeframe'], v['lag'], v['site_regime'])
                    ccf = cust_cov.get(cust_key)
                    cbf = cust_bias.get(cust_key)
                    meta = cust_meta.get(cust_key)
                    if (ccf and cbf and ccf > 0.001 and cbf > 0.001 and meta
                            and meta['n_sites'] >= POOL_MIN_SITES
                            and meta['bc_cv'] < POOL_MAX_CV):
                        est = fc_val / cbf / ccf
                        valid_ests.append((max(est, 0), v['lag']))
                        used_pool = True

            if valid_ests:
                combined = max(v9._combine_vintage_estimates(valid_ests), 0)
                preds[i] = combined
                if used_pool and not any(
                    cov_curve.get((v['gsa_site'], v['timeframe'], v['lag'], v['site_regime']))
                    for v in cands if not v['forecast_error'] and v['lag'] <= v9.MAX_VINTAGE_LAG
                ):
                    sources[i] = 'FC_CUST_POOLED'
                else:
                    sources[i] = 'FC_SITE_CORRECTED'
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
# Run across all snapshots
# ============================================================
print("=" * 110)
print(f"Customer-Pooled Fallback (guardrails: N_sites >= {POOL_MIN_SITES}, CV < {POOL_MAX_CV})")
print("=" * 110)

results_current = []
results_pooled = []

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

    cust_cov, cust_bias, cust_meta = build_customer_pool(cov_curve, bias_curve, c_stats)

    pred_cur = v9.predict(test, cov_curve, bias_curve, hist, registry, curve_stats=c_stats)
    pred_pool = predict_with_pool(test, cov_curve, bias_curve, hist, registry, c_stats,
                                  cust_cov, cust_bias, cust_meta)

    pred_cur['Snapshot'] = snap
    pred_pool['Snapshot'] = snap
    results_current.append(pred_cur)
    results_pooled.append(pred_pool)

rc = pd.concat(results_current, ignore_index=True)
rp = pd.concat(results_pooled, ignore_index=True)

# ============================================================
# Per-snapshot comparison
# ============================================================
print(f"\n  {'Snap':<12} | {'--- CURRENT ---':^35} | {'--- WITH POOL ---':^45} |")
print(f"  {'':12} | {'FC_C n':>7} {'FC_C W':>7} {'FC_C B':>7} {'H W':>7} | {'FC_C n':>7} {'Pool n':>7} {'FC_C W':>7} {'Pool W':>7} {'H n':>5} {'H W':>7} |")
print(f"  {'-'*12} | {'-'*7} {'-'*7} {'-'*7} {'-'*7} | {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*5} {'-'*7} |")

for snap in snaps:
    sc = rc[rc['Snapshot'] == snap]
    sp = rp[rp['Snapshot'] == snap]

    c_fc = sc[sc['Prediction_Source'].isin(v9.SOURCES_FC_CORRECTED)]
    c_hi = sc[sc['Prediction_Source'] == 'HISTORICAL']

    p_fc = sp[sp['Prediction_Source'] == 'FC_SITE_CORRECTED']
    p_pool = sp[sp['Prediction_Source'] == 'FC_CUST_POOLED']
    p_hi = sp[sp['Prediction_Source'] == 'HISTORICAL']

    cw, _ = v9._wmape(c_fc['Actual_Sales'], c_fc['Prediction']) if len(c_fc) > 0 else (float('nan'), 0)
    cb = v9._bias(c_fc['Actual_Sales'], c_fc['Prediction']) if len(c_fc) > 0 else float('nan')
    chw, _ = v9._wmape(c_hi['Actual_Sales'], c_hi['Prediction']) if len(c_hi) > 0 else (float('nan'), 0)

    pw, _ = v9._wmape(p_fc['Actual_Sales'], p_fc['Prediction']) if len(p_fc) > 0 else (float('nan'), 0)
    ppw, _ = v9._wmape(p_pool['Actual_Sales'], p_pool['Prediction']) if len(p_pool) > 0 else (float('nan'), 0)
    phw, _ = v9._wmape(p_hi['Actual_Sales'], p_hi['Prediction']) if len(p_hi) > 0 else (float('nan'), 0)

    ppw_str = f"{ppw*100:>6.1f}%" if not np.isnan(ppw) else "   N/A"
    print(f"  {snap:<12} | {len(c_fc):>7} {cw*100:>6.1f}% {cb*100:>+6.1f}% {chw*100:>6.1f}% | {len(p_fc):>7} {len(p_pool):>7} {pw*100:>6.1f}% {ppw_str} {len(p_hi):>5} {phw*100:>6.1f}% |")

# ============================================================
# Overall comparison
# ============================================================
print(f"\n{'='*110}")
print("Overall Comparison (all snapshots combined)")
print(f"{'='*110}")

for label, pdf in [("CURRENT", rc), ("WITH POOL", rp)]:
    scored = pdf[pdf['Prediction'].notna()]
    print(f"\n  {label}:")
    for src_label, mask in [
        ('FC_Site_Corrected', scored['Prediction_Source'].isin(v9.SOURCES_FC_CORRECTED if label == 'CURRENT' else ['FC_SITE_CORRECTED'])),
        ('FC_Cust_Pooled', scored['Prediction_Source'] == 'FC_CUST_POOLED'),
        ('Historical', scored['Prediction_Source'] == 'HISTORICAL'),
        ('Overall', pd.Series(True, index=scored.index)),
    ]:
        sub = scored[mask]
        if len(sub) == 0:
            continue
        w, _ = v9._wmape(sub['Actual_Sales'], sub['Prediction'])
        b = v9._bias(sub['Actual_Sales'], sub['Prediction'])
        d = sub['Actual_Sales'].sum()
        print(f"    {src_label:<20} {len(sub):>5} rows  ${d:>16,.0f}  WMAPE {w*100:>5.1f}%  Bias {b*100:>+5.1f}%")

# ============================================================
# Head-to-head: rows that changed from Historical to FC_Cust_Pooled
# ============================================================
print(f"\n{'='*110}")
print("Head-to-Head: Rows Upgraded from Historical to FC_Cust_Pooled")
print(f"{'='*110}")

# Find rows that are HISTORICAL in current but FC_CUST_POOLED in pooled
merged = rc[['gsa_site', 'Snapshot', 'Target_Period_Start', 'Actual_Sales',
             'Prediction', 'Prediction_Source']].rename(
    columns={'Prediction': 'Cur_Pred', 'Prediction_Source': 'Cur_Src'})
pool_slim = rp[['gsa_site', 'Snapshot', 'Target_Period_Start',
                'Prediction', 'Prediction_Source']].rename(
    columns={'Prediction': 'Pool_Pred', 'Prediction_Source': 'Pool_Src'})

hh = merged.merge(pool_slim, on=['gsa_site', 'Snapshot', 'Target_Period_Start'])
upgraded = hh[hh['Pool_Src'] == 'FC_CUST_POOLED'].copy()

if len(upgraded) > 0:
    upgraded['Pool_Err'] = np.abs(upgraded['Pool_Pred'] - upgraded['Actual_Sales'])
    upgraded['Cur_Err'] = np.abs(upgraded['Cur_Pred'] - upgraded['Actual_Sales'])
    upgraded['Pool_Better'] = upgraded['Pool_Err'] < upgraded['Cur_Err']

    print(f"\n  Total upgraded rows: {len(upgraded)}")
    print(f"  Total actual sales:  ${upgraded['Actual_Sales'].sum():,.0f}")

    pw, _ = v9._wmape(upgraded['Actual_Sales'], upgraded['Pool_Pred'])
    pb = v9._bias(upgraded['Actual_Sales'], upgraded['Pool_Pred'])
    hw, _ = v9._wmape(upgraded['Actual_Sales'], upgraded['Cur_Pred'])
    hb = v9._bias(upgraded['Actual_Sales'], upgraded['Cur_Pred'])

    print(f"\n  FC_Cust_Pooled: WMAPE {pw*100:.1f}%, Bias {pb*100:+.1f}%")
    print(f"  Historical:     WMAPE {hw*100:.1f}%, Bias {hb*100:+.1f}%")

    print(f"\n  Head-to-head:")
    print(f"    Pooled wins:     {upgraded['Pool_Better'].sum()} ({upgraded['Pool_Better'].mean()*100:.0f}%)")
    print(f"    Historical wins: {(~upgraded['Pool_Better']).sum()} ({(~upgraded['Pool_Better']).mean()*100:.0f}%)")
    print(f"    Pooled total |err|:     ${upgraded['Pool_Err'].sum():,.0f}")
    print(f"    Historical total |err|: ${upgraded['Cur_Err'].sum():,.0f}")
    dd = upgraded['Pool_Err'].sum() - upgraded['Cur_Err'].sum()
    print(f"    Dollar diff:            ${dd:+,.0f} ({'pooled worse' if dd > 0 else 'pooled better'})")

    # Per-snapshot breakdown of upgraded rows
    print(f"\n  Per-snapshot:")
    print(f"  {'Snap':<12} {'n':>4} {'Pool WMAPE':>11} {'Hist WMAPE':>11} {'Pool $Err':>14} {'Hist $Err':>14} {'Winner':>8}")
    print(f"  {'-'*12} {'-'*4} {'-'*11} {'-'*11} {'-'*14} {'-'*14} {'-'*8}")
    for snap in snaps:
        sub = upgraded[upgraded['Snapshot'] == snap]
        if len(sub) == 0:
            continue
        pw2, _ = v9._wmape(sub['Actual_Sales'], sub['Pool_Pred'])
        hw2, _ = v9._wmape(sub['Actual_Sales'], sub['Cur_Pred'])
        pe = sub['Pool_Err'].sum()
        he = sub['Cur_Err'].sum()
        winner = 'Pool' if pe < he else 'Hist'
        print(f"  {snap:<12} {len(sub):>4} {pw2*100:>10.1f}% {hw2*100:>10.1f}% ${pe:>12,.0f} ${he:>12,.0f} {winner:>8}")

    # Per-site detail
    print(f"\n  Per-site (across all snapshots):")
    upgraded['Customer'] = upgraded['gsa_site'].str.split('|').str[0]
    site_agg = upgraded.groupby('gsa_site').agg(
        n=('Actual_Sales', 'count'),
        actual=('Actual_Sales', 'sum'),
        pool_err=('Pool_Err', 'sum'),
        cur_err=('Cur_Err', 'sum'),
    )
    site_agg['pool_wmape'] = site_agg['pool_err'] / site_agg['actual'] * 100
    site_agg['hist_wmape'] = site_agg['cur_err'] / site_agg['actual'] * 100
    site_agg['diff'] = site_agg['pool_err'] - site_agg['cur_err']
    site_agg = site_agg.sort_values('diff')

    print(f"  {'Site':<30} {'n':>3} {'Actual':>14} {'Pool WMAPE':>11} {'Hist WMAPE':>11} {'$Diff':>14} {'Winner':>8}")
    print(f"  {'-'*30} {'-'*3} {'-'*14} {'-'*11} {'-'*11} {'-'*14} {'-'*8}")
    for gs, r in site_agg.iterrows():
        winner = 'Pool' if r['diff'] < 0 else 'Hist'
        print(f"  {gs:<30} {r['n']:>3} ${r['actual']:>12,.0f} {r['pool_wmape']:>10.1f}% {r['hist_wmape']:>10.1f}% ${r['diff']:>+12,.0f} {winner:>8}")
else:
    print("\n  No rows were upgraded across any snapshot.")
