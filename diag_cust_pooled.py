"""Diagnostic: customer-level pooled correction as fallback for new sites.

At Dec 2024 snapshot, test using customer-average bias/coverage curves
when site-level curves aren't available. Compare to Historical fallback.
Also analyze what predicts success vs failure of the pooled correction.
"""
import aging_v9 as v9
import numpy as np
import pandas as pd

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
# Build site-level curves (current approach)
# ============================================================
cov_curve = v9.build_flat_curves(curve_train, 'fc_coverage')
bias_curve = v9.build_flat_curves(curve_train, 'fc_bias')
hist = v9.build_historical(train)
registry = v9.build_forecast_registry(visible)
c_stats = v9.build_curve_stats(curve_train)

# ============================================================
# Build customer-level pooled curves
# ============================================================
# For each (customer, tf, lag, regime), compute weighted-average bias and coverage
# across all sites in that customer that have curves

pool_rows = []
for (gs, tf, lag, regime), cov in cov_curve.items():
    bias = bias_curve.get((gs, tf, lag, regime))
    if bias is None:
        continue
    customer = gs.split('|')[0]
    # Get the observation count for weighting (more obs = more reliable)
    stats = c_stats.get((gs, tf, lag, regime))
    n = stats['n'] if stats else v9.MIN_OBS_FLAT
    pool_rows.append({
        'customer': customer, 'site': gs, 'tf': tf, 'lag': lag, 'regime': regime,
        'bias': bias, 'coverage': cov, 'n': n,
    })

pool_df = pd.DataFrame(pool_rows)

# Compute customer-level averages weighted by observation count
cust_cov = {}
cust_bias = {}
cust_meta = {}  # Store metadata for analysis: n_sites, CV, etc.

for (cust, tf, lag, regime), grp in pool_df.groupby(['customer', 'tf', 'lag', 'regime']):
    if len(grp) < 1:
        continue
    weights = grp['n'].values.astype(float)
    w_bias = np.average(grp['bias'].values, weights=weights)
    w_cov = np.average(grp['coverage'].values, weights=weights)

    # Within-customer spread (for predicting success/failure)
    bias_cv = grp['bias'].std() / grp['bias'].mean() if grp['bias'].mean() > 0 and len(grp) > 1 else 0.0
    cov_cv = grp['coverage'].std() / grp['coverage'].mean() if grp['coverage'].mean() > 0 and len(grp) > 1 else 0.0

    cust_cov[(cust, tf, lag, regime)] = w_cov
    cust_bias[(cust, tf, lag, regime)] = w_bias
    cust_meta[(cust, tf, lag, regime)] = {
        'n_sites': len(grp),
        'bias_cv': bias_cv,
        'cov_cv': cov_cv,
        'bc_cv': max(bias_cv, cov_cv),
        'total_obs': grp['n'].sum(),
    }

print("=" * 90)
print(f"Customer-Pooled Correction Fallback (snapshot {snap})")
print("=" * 90)
print(f"\nSite-level curve cells: {len(cov_curve)}")
print(f"Customer-level pooled cells: {len(cust_cov)}")

# ============================================================
# Generate predictions with customer-pooled fallback
# ============================================================
n = len(test)
pooled_preds = np.full(n, np.nan)
pooled_sources = np.empty(n, dtype=object)
pooled_method = np.empty(n, dtype=object)  # 'site_curve' | 'cust_pool' | 'historical'
pooled_cust_cv = np.full(n, np.nan)
pooled_cust_nsites = np.zeros(n, dtype=int)

for i, (_, row) in enumerate(test.iterrows()):
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
        method_used = None
        cust_cv_val = np.nan
        cust_ns = 0

        for v in cands:
            key = (v['gsa_site'], v['timeframe'], v['lag'], v['site_regime'])
            cf = cov_curve.get(key)
            bf = bias_curve.get(key)
            fc_val = v['fc_value']
            if fc_val <= 0:
                continue

            if cf and bf and cf > 0.001 and bf > 0.001:
                # Site-level correction available
                est = fc_val / bf / cf
                valid_ests.append((max(est, 0), v['lag']))
                method_used = 'site_curve'
            else:
                # Try customer-level pooled correction
                cust_key = (customer, v['timeframe'], v['lag'], v['site_regime'])
                ccf = cust_cov.get(cust_key)
                cbf = cust_bias.get(cust_key)
                if ccf and cbf and ccf > 0.001 and cbf > 0.001:
                    est = fc_val / cbf / ccf
                    valid_ests.append((max(est, 0), v['lag']))
                    meta = cust_meta.get(cust_key, {})
                    cust_cv_val = meta.get('bc_cv', np.nan)
                    cust_ns = meta.get('n_sites', 0)
                    if method_used != 'site_curve':
                        method_used = 'cust_pool'

        if valid_ests:
            combined = max(v9._combine_vintage_estimates(valid_ests), 0)
            pooled_preds[i] = combined
            if method_used == 'cust_pool':
                pooled_sources[i] = 'FC_CUST_POOLED'
                pooled_cust_cv[i] = cust_cv_val
                pooled_cust_nsites[i] = cust_ns
            else:
                pooled_sources[i] = 'FC_SITE_CORRECTED'
            pooled_method[i] = method_used
            continue

    h = hist.get((gs, tf), hist.get(('FB', gs)))
    if h is not None and h > 0:
        pooled_preds[i] = h
        pooled_sources[i] = 'HISTORICAL'
        pooled_method[i] = 'historical'

pred_pool = test.copy()
pred_pool['Prediction'] = pooled_preds
pred_pool['Prediction_Source'] = pooled_sources
pred_pool['Method'] = pooled_method
pred_pool['Cust_Pool_CV'] = pooled_cust_cv
pred_pool['Cust_Pool_NSites'] = pooled_cust_nsites

# Also get current model predictions for comparison
pred_current = v9.predict(test, cov_curve, bias_curve, hist, registry, curve_stats=c_stats)

# ============================================================
# Results comparison
# ============================================================
print(f"\n{'='*90}")
print(f"Results Comparison (snapshot {snap})")
print(f"{'='*90}")

# Current model breakdown
cur_fc = pred_current[pred_current['Prediction_Source'].isin(v9.SOURCES_FC_CORRECTED)]
cur_hist = pred_current[pred_current['Prediction_Source'] == 'HISTORICAL']
cur_all = pred_current[pred_current['Prediction'].notna()]

print(f"\n  CURRENT MODEL:")
for label, sub in [('FC_Corrected', cur_fc), ('Historical', cur_hist), ('Overall', cur_all)]:
    if len(sub) == 0:
        continue
    w, _ = v9._wmape(sub['Actual_Sales'], sub['Prediction'])
    b = v9._bias(sub['Actual_Sales'], sub['Prediction'])
    d = sub['Actual_Sales'].sum()
    print(f"    {label:<20} {len(sub):>4} rows  ${d:>14,.0f}  WMAPE {w*100:>5.1f}%  Bias {b*100:>+5.1f}%")

# Pooled model breakdown
pool_site = pred_pool[pred_pool['Prediction_Source'] == 'FC_SITE_CORRECTED']
pool_cust = pred_pool[pred_pool['Prediction_Source'] == 'FC_CUST_POOLED']
pool_hist = pred_pool[pred_pool['Prediction_Source'] == 'HISTORICAL']
pool_all = pred_pool[pred_pool['Prediction'].notna()]

print(f"\n  WITH CUSTOMER-POOLED FALLBACK:")
for label, sub in [('FC_Site_Corrected', pool_site), ('FC_Cust_Pooled', pool_cust),
                    ('Historical', pool_hist), ('Overall', pool_all)]:
    if len(sub) == 0:
        continue
    w, _ = v9._wmape(sub['Actual_Sales'], sub['Prediction'])
    b = v9._bias(sub['Actual_Sales'], sub['Prediction'])
    d = sub['Actual_Sales'].sum()
    print(f"    {label:<20} {len(sub):>4} rows  ${d:>14,.0f}  WMAPE {w*100:>5.1f}%  Bias {b*100:>+5.1f}%")

# ============================================================
# Focus on the NEW rows: cust_pooled vs what they would have been (historical)
# ============================================================
print(f"\n{'='*90}")
print(f"New Rows: FC_Cust_Pooled vs Historical Fallback")
print(f"{'='*90}")

# These are rows that got FC_CUST_POOLED — what would they have gotten under current model?
new_rows = pred_pool[pred_pool['Prediction_Source'] == 'FC_CUST_POOLED'].copy()

if len(new_rows) > 0:
    # Match to current model predictions for same rows
    cur_match = pred_current.loc[new_rows.index]

    print(f"\n  Rows upgraded from Historical to FC_Cust_Pooled: {len(new_rows)}")
    print(f"  Total actual sales in these rows: ${new_rows['Actual_Sales'].sum():,.0f}")

    pw, _ = v9._wmape(new_rows['Actual_Sales'], new_rows['Prediction'])
    pb = v9._bias(new_rows['Actual_Sales'], new_rows['Prediction'])
    print(f"\n  FC_Cust_Pooled:  WMAPE {pw*100:.1f}%, Bias {pb*100:+.1f}%")

    hw, _ = v9._wmape(cur_match['Actual_Sales'], cur_match['Prediction'])
    hb = v9._bias(cur_match['Actual_Sales'], cur_match['Prediction'])
    print(f"  Historical:      WMAPE {hw*100:.1f}%, Bias {hb*100:+.1f}%")

    # Head-to-head
    new_rows['Pool_Err'] = np.abs(new_rows['Prediction'] - new_rows['Actual_Sales'])
    new_rows['Hist_Pred'] = cur_match['Prediction'].values
    new_rows['Hist_Err'] = np.abs(new_rows['Hist_Pred'] - new_rows['Actual_Sales'])
    new_rows['Pool_Better'] = new_rows['Pool_Err'] < new_rows['Hist_Err']

    print(f"\n  Head-to-head:")
    print(f"    Pooled wins:     {new_rows['Pool_Better'].sum()} ({new_rows['Pool_Better'].mean()*100:.0f}%)")
    print(f"    Historical wins: {(~new_rows['Pool_Better']).sum()} ({(~new_rows['Pool_Better']).mean()*100:.0f}%)")
    print(f"    Pooled total |err|:     ${new_rows['Pool_Err'].sum():,.0f}")
    print(f"    Historical total |err|: ${new_rows['Hist_Err'].sum():,.0f}")
    dd = new_rows['Pool_Err'].sum() - new_rows['Hist_Err'].sum()
    print(f"    Dollar diff:            ${dd:+,.0f} ({'pooled worse' if dd > 0 else 'pooled better'})")

    # ============================================================
    # Per-row detail
    # ============================================================
    print(f"\n{'='*90}")
    print(f"Per-Row Detail (FC_Cust_Pooled rows)")
    print(f"{'='*90}")
    print(f"\n  {'Site':<30} {'Actual':>14} {'Pool Pred':>14} {'Hist Pred':>14} {'Pool Err%':>10} {'Hist Err%':>10} {'Winner':>8} {'CustCV':>7} {'NSites':>7}")
    print(f"  {'-'*30} {'-'*14} {'-'*14} {'-'*14} {'-'*10} {'-'*10} {'-'*8} {'-'*7} {'-'*7}")

    for _, r in new_rows.sort_values('Actual_Sales', ascending=False).iterrows():
        pool_pct = r['Pool_Err'] / r['Actual_Sales'] * 100 if r['Actual_Sales'] > 0 else np.nan
        hist_pct = r['Hist_Err'] / r['Actual_Sales'] * 100 if r['Actual_Sales'] > 0 else np.nan
        winner = 'Pool' if r['Pool_Better'] else 'Hist'
        cv_str = f"{r['Cust_Pool_CV']:.2f}" if not np.isnan(r['Cust_Pool_CV']) else 'N/A'
        print(f"  {r['gsa_site']:<30} ${r['Actual_Sales']:>12,.0f} ${r['Prediction']:>12,.0f} ${r['Hist_Pred']:>12,.0f} {pool_pct:>9.1f}% {hist_pct:>9.1f}% {winner:>8} {cv_str:>7} {r['Cust_Pool_NSites']:>7.0f}")

    # ============================================================
    # What predicts success? Analyze by customer CV and n_sites
    # ============================================================
    print(f"\n{'='*90}")
    print(f"What Predicts Success of Customer-Pooled Correction?")
    print(f"{'='*90}")

    # By customer CV bucket
    new_rows['CV_Bucket'] = pd.cut(new_rows['Cust_Pool_CV'], bins=[0, 0.15, 0.30, 0.50, 1.0, 99],
                                    labels=['<0.15', '0.15-0.30', '0.30-0.50', '0.50-1.0', '>1.0'])
    print(f"\n  By within-customer CV of correction factors:")
    print(f"  {'CV Bucket':<12} {'n':>4} {'Pool WMAPE':>11} {'Hist WMAPE':>11} {'Pool Wins':>10} {'Verdict':>10}")
    print(f"  {'-'*12} {'-'*4} {'-'*11} {'-'*11} {'-'*10} {'-'*10}")
    for bucket in ['<0.15', '0.15-0.30', '0.30-0.50', '0.50-1.0', '>1.0']:
        sub = new_rows[new_rows['CV_Bucket'] == bucket]
        if len(sub) == 0:
            continue
        pw2, _ = v9._wmape(sub['Actual_Sales'], sub['Prediction'])
        hw2, _ = v9._wmape(sub['Actual_Sales'], sub['Hist_Pred'])
        pwin = sub['Pool_Better'].sum()
        verdict = 'POOL' if pw2 < hw2 else 'HIST'
        print(f"  {bucket:<12} {len(sub):>4} {pw2*100:>10.1f}% {hw2*100:>10.1f}% {pwin:>6}/{len(sub):<3} {verdict:>10}")

    # By number of sites contributing to the pool
    print(f"\n  By number of sites in customer pool:")
    print(f"  {'N Sites':<10} {'n':>4} {'Pool WMAPE':>11} {'Hist WMAPE':>11} {'Pool Wins':>10} {'Verdict':>10}")
    print(f"  {'-'*10} {'-'*4} {'-'*11} {'-'*11} {'-'*10} {'-'*10}")
    for ns in sorted(new_rows['Cust_Pool_NSites'].unique()):
        sub = new_rows[new_rows['Cust_Pool_NSites'] == ns]
        if len(sub) == 0:
            continue
        pw2, _ = v9._wmape(sub['Actual_Sales'], sub['Prediction'])
        hw2, _ = v9._wmape(sub['Actual_Sales'], sub['Hist_Pred'])
        pwin = sub['Pool_Better'].sum()
        verdict = 'POOL' if pw2 < hw2 else 'HIST'
        print(f"  {int(ns):<10} {len(sub):>4} {pw2*100:>10.1f}% {hw2*100:>10.1f}% {pwin:>6}/{len(sub):<3} {verdict:>10}")

    # By customer
    print(f"\n  By customer:")
    print(f"  {'Customer':<15} {'n':>4} {'Pool WMAPE':>11} {'Hist WMAPE':>11} {'Pool $Err':>14} {'Hist $Err':>14} {'Verdict':>10}")
    print(f"  {'-'*15} {'-'*4} {'-'*11} {'-'*11} {'-'*14} {'-'*14} {'-'*10}")
    new_rows['Customer'] = new_rows['gsa_site'].str.split('|').str[0]
    for cust, sub in new_rows.groupby('Customer'):
        if len(sub) == 0:
            continue
        pw2, _ = v9._wmape(sub['Actual_Sales'], sub['Prediction'])
        hw2, _ = v9._wmape(sub['Actual_Sales'], sub['Hist_Pred'])
        pe = sub['Pool_Err'].sum()
        he = sub['Hist_Err'].sum()
        verdict = 'POOL' if pe < he else 'HIST'
        print(f"  {cust:<15} {len(sub):>4} {pw2*100:>10.1f}% {hw2*100:>10.1f}% ${pe:>12,.0f} ${he:>12,.0f} {verdict:>10}")
else:
    print("\n  No rows were upgraded to FC_Cust_Pooled (all sites already have curves or no FC)")
