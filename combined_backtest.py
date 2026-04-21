#!/usr/bin/env python3
"""Combined backtest: Relaxed +3mo base + CrossTF for periods beyond that.

Layer 1 (freshest):  Relaxed — real data for periods ending within 3mo of snap
Layer 2 (inference): CrossTF — regression-inferred fc_product for periods
                     ending 3-12mo after snap

Comparison:
  A) Current (strict, 2-curve)
  B) Relaxed +3mo (2-curve)
  C) CrossTF only (product curve, strict base + synthetic)
  D) Combined (product curve, relaxed base + synthetic beyond 3mo)
  E) Oracle (2-curve, all published)
"""
import sys
sys.path.insert(0, '/home/user/techwriting')

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import aging_v9 as v9
import cross_tf_backtest as xtf

DATA_PATH = '/home/user/techwriting/training_data_anonymized.csv'
SNAP_START = '2024-06-01'
SNAP_END = '2025-03-01'
HORIZON = 3
RELAX_MONTHS = 3


def infer_beyond_relaxed(all_df, snapshot, regressions, relax_months):
    """Like xtf.infer_synthetic_observations but only for periods ending
    BEYOND snap + relax_months (the relaxed filter already covers the rest)."""
    snap = pd.Timestamp(snapshot)
    relaxed_end = snap + pd.DateOffset(months=relax_months)

    incomplete_12 = all_df[
        (all_df['Timeframe'] == 12) &
        (all_df['Target_Period_End'] >= relaxed_end) &
        (all_df['Target_Period_Start'] < snap) &
        (all_df['Has_Forecast'] == 1) &
        (all_df['Reference_Month'] < snap) &
        (all_df['Prediction_Lag'] <= v9.MAX_VINTAGE_LAG)
    ]

    completed_short = all_df[
        (all_df['Timeframe'].isin(xtf.CROSS_TF_SHORT)) &
        (all_df['Target_Period_End'] < snap) &
        (all_df['Has_Forecast'] == 1)
    ]
    short_lookup = completed_short.set_index(
        ['gsa_site', 'Target_Period_Start', 'Prediction_Lag', 'Timeframe'])

    rows = []
    for _, r12 in incomplete_12.iterrows():
        site = r12['gsa_site']
        start = r12['Target_Period_Start']
        lag = r12['Prediction_Lag']

        for short_tf in xtf.CROSS_TF_SHORT:
            key = (site, start, lag, short_tf)
            if key not in short_lookup.index:
                continue
            model = regressions.get((site, short_tf))
            if model is None:
                continue

            match = short_lookup.loc[key]
            if isinstance(match, pd.DataFrame):
                match = match.iloc[0]
            fp_short = match['fc_product']
            if pd.isna(fp_short) or fp_short <= 0 or fp_short >= xtf.PRODUCT_FILTER_HI:
                continue

            inferred = model['slope'] * fp_short + model['intercept']
            inferred = np.clip(inferred, 0.01, xtf.PRODUCT_FILTER_HI)

            rows.append({
                'gsa_site': site,
                'Timeframe': 12,
                'Prediction_Lag': lag,
                'site_regime': r12['site_regime'],
                'Reference_Month': r12['Reference_Month'],
                'fc_product': inferred,
                'is_synthetic': True,
                'source_tf': short_tf,
            })
            break

    if not rows:
        return pd.DataFrame(columns=[
            'gsa_site', 'Timeframe', 'Prediction_Lag', 'site_regime',
            'Reference_Month', 'fc_product', 'is_synthetic', 'source_tf'])
    return pd.DataFrame(rows)


def run_snapshot(all_df, snapshot):
    snap = pd.Timestamp(snapshot)
    horizon_end = snap + pd.DateOffset(months=HORIZON)
    relaxed_end = snap + pd.DateOffset(months=RELAX_MONTHS)

    df12 = all_df[all_df['Timeframe'] == 12].copy()
    test_12 = df12[
        (df12['Target_Period_Start'] >= snap) &
        (df12['Target_Period_Start'] < horizon_end) &
        (df12['Reference_Month'] < snap) &
        (df12['Prediction_Lag'].isin(v9.TEST_LAGS))
    ].copy()
    visible_12 = df12[df12['Reference_Month'] < snap]

    if len(test_12) == 0:
        return None

    # Shared registry
    registry = v9.build_forecast_registry(visible_12)

    # --- A) Current: strict, 2-curve ---
    train_strict = df12[df12['Target_Period_End'] < snap]
    ct_strict = train_strict[train_strict['Has_Forecast'] == 1].copy()
    hist_strict = v9.build_historical(train_strict)
    cov_a = v9.build_flat_curves(ct_strict, 'fc_coverage')
    bias_a = v9.build_flat_curves(ct_strict, 'fc_bias')
    stats_a = v9.build_curve_stats(ct_strict)
    pred_a = v9.predict(test_12, cov_a, bias_a, hist_strict, registry, stats_a)

    # --- B) Relaxed +3mo: 2-curve ---
    train_relaxed = df12[df12['Target_Period_End'] < relaxed_end]
    ct_relaxed = train_relaxed[train_relaxed['Has_Forecast'] == 1].copy()
    hist_relaxed = v9.build_historical(train_relaxed)
    cov_b = v9.build_flat_curves(ct_relaxed, 'fc_coverage')
    bias_b = v9.build_flat_curves(ct_relaxed, 'fc_bias')
    stats_b = v9.build_curve_stats(ct_relaxed)
    pred_b = v9.predict(test_12, cov_b, bias_b, hist_relaxed, registry, stats_b)

    # --- C) CrossTF only: product curve, strict base + all synthetic ---
    ct_strict_p = ct_strict.copy()
    ct_strict_p['fc_product'] = ct_strict_p['fc_bias'] * ct_strict_p['fc_coverage']
    regressions = xtf.build_cross_tf_regressions(all_df, snap)
    synthetic_all = xtf.infer_synthetic_observations(all_df, snap, regressions)
    prod_c = xtf.build_product_curve(ct_strict_p, synthetic_all)
    pred_c = xtf.predict_product(test_12, prod_c, hist_strict, registry)

    # --- D) Combined: relaxed base (product curve) + synthetic beyond 3mo ---
    ct_relaxed_p = ct_relaxed.copy()
    ct_relaxed_p['fc_product'] = ct_relaxed_p['fc_bias'] * ct_relaxed_p['fc_coverage']
    synthetic_beyond = infer_beyond_relaxed(all_df, snap, regressions, RELAX_MONTHS)
    prod_d = xtf.build_product_curve(ct_relaxed_p, synthetic_beyond)
    pred_d = xtf.predict_product(test_12, prod_d, hist_relaxed, registry)

    # --- E) Oracle: 2-curve, all published ---
    oracle_pool = df12[
        (df12['Has_Forecast'] == 1) & (df12['Reference_Month'] < snap)
    ].copy()
    oracle_pool = oracle_pool[~oracle_pool.index.isin(test_12.index)]
    oracle_train = df12[
        (df12['Reference_Month'] < snap) & (~df12.index.isin(test_12.index))
    ]
    hist_oracle = v9.build_historical(oracle_train)
    cov_e = v9.build_flat_curves(oracle_pool, 'fc_coverage')
    bias_e = v9.build_flat_curves(oracle_pool, 'fc_bias')
    stats_e = v9.build_curve_stats(oracle_pool)
    pred_e = v9.predict(test_12, cov_e, bias_e, hist_oracle, registry, stats_e)

    return {
        'snapshot': snapshot,
        'current': xtf.score_variant(pred_a),
        'relaxed': xtf.score_variant(pred_b),
        'crosstf': xtf.score_variant(pred_c),
        'combined': xtf.score_variant(pred_d),
        'oracle': xtf.score_variant(pred_e),
        'pred_current': pred_a,
        'pred_relaxed': pred_b,
        'pred_crosstf': pred_c,
        'pred_combined': pred_d,
        'pred_oracle': pred_e,
        'n_synthetic_all': len(synthetic_all),
        'n_synthetic_beyond': len(synthetic_beyond),
        'n_curve_strict': len(ct_strict),
        'n_curve_relaxed': len(ct_relaxed),
    }


def _fp(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '   n/a'
    return f'{x*100:+.1f}%'

def _fw(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '  n/a'
    return f'{x*100:.1f}%'


def main():
    print('=' * 100)
    print('Combined Backtest: Relaxed +3mo + CrossTF Beyond')
    print('=' * 100)

    all_df = xtf.load_all_timeframes(DATA_PATH)
    print(f'Loaded {len(all_df):,} rows')

    snapshots = [d.strftime('%Y-%m-%d')
                 for d in v9.monthly_grid(SNAP_START, SNAP_END)]

    variants = [
        ('current', 'Current'),
        ('relaxed', 'Relaxed +3mo'),
        ('crosstf', 'CrossTF'),
        ('combined', 'Combined'),
        ('oracle', 'Oracle'),
    ]

    results = []
    for snap in snapshots:
        print(f'  {snap}...', end='')
        r = run_snapshot(all_df, snap)
        if r is None:
            print(' skipped')
            continue
        print(f' synth_beyond={r["n_synthetic_beyond"]}  '
              f'curve: {r["n_curve_strict"]}→{r["n_curve_relaxed"]}(+relax)')
        results.append(r)

    if not results:
        return

    # ---- Overall table ----
    print('\n' + '=' * 145)
    print('FIVE-WAY COMPARISON: Per-Snapshot Overall')
    print('=' * 145)
    hdr = '  ' + ' | '.join(
        [f'{"Snap":<12}'] +
        [f'{label:^24}' for _, label in variants])
    sub = '  ' + ' | '.join(
        [f'{"":12}'] +
        [f'{"FC_C":>5} {"HIST":>5} {"WMAPE":>6} {"Bias":>7}'] * len(variants))
    print(hdr)
    print(sub)
    print('  ' + '-' * 143)

    for r in results:
        parts = [f'{r["snapshot"]:<12}']
        for vkey, _ in variants:
            m = r[vkey]
            parts.append(f'{m["FC_C"]["rows"]:>5} {m["HIST"]["rows"]:>5} '
                         f'{_fw(m["overall"]["wmape"]):>6} '
                         f'{_fp(m["overall"]["bias"]):>7}')
        print('  ' + ' | '.join(parts))

    # Means
    print('  ' + '-' * 143)
    means = {}
    mean_parts = [f'{"MEAN":<12}']
    for vkey, _ in variants:
        wv = [r[vkey]['overall']['wmape'] for r in results
              if not np.isnan(r[vkey]['overall']['wmape'] or np.nan)]
        bv = [r[vkey]['overall']['bias'] for r in results
              if not np.isnan(r[vkey]['overall']['bias'] or np.nan)]
        fc = [r[vkey]['FC_C']['rows'] for r in results]
        hr = [r[vkey]['HIST']['rows'] for r in results]
        means[vkey] = {
            'wmape': np.mean(wv), 'bias': np.mean(bv),
            'fc_c': np.mean(fc), 'hist': np.mean(hr)}
        mean_parts.append(
            f'{means[vkey]["fc_c"]:>5.0f} {means[vkey]["hist"]:>5.0f} '
            f'{_fw(means[vkey]["wmape"]):>6} {_fp(means[vkey]["bias"]):>7}')
    print('  ' + ' | '.join(mean_parts))

    # ---- FC_C detail ----
    print('\n' + '=' * 130)
    print('FC_CORRECTED Detail')
    print('=' * 130)
    for r in results:
        parts = [f'{r["snapshot"]:<12}']
        for vkey, _ in variants:
            m = r[vkey]['FC_C']
            parts.append(f'{m["rows"]:>4} {_fw(m["wmape"]):>6} {_fp(m["bias"]):>7}')
        print('  ' + ' | '.join(parts))

    # ---- Per-customer ----
    print('\n' + '=' * 130)
    print('PER-CUSTOMER BIAS (all sources, all snapshots)')
    print('=' * 130)

    pred_keys = {
        'current': 'pred_current', 'relaxed': 'pred_relaxed',
        'crosstf': 'pred_crosstf', 'combined': 'pred_combined',
        'oracle': 'pred_oracle'}
    combined_preds = {
        vkey: pd.concat([r[pk] for r in results], ignore_index=True)
        for vkey, pk in pred_keys.items()}

    print(f'  {"Customer":<16} {"Rows":>5}', end='')
    for _, label in variants:
        print(f' | {label:>10} {"Bias":>7}', end='')
    print()
    print('  ' + '-' * 128)

    for cust in sorted(combined_preds['current']['GSA'].unique()):
        base = combined_preds['current']
        bc = base[base['GSA'] == cust]
        if bc['Actual_Sales'].sum() == 0:
            continue
        print(f'  {cust:<16} {len(bc):>5}', end='')
        for vkey, _ in variants:
            cc = combined_preds[vkey]
            cc_c = cc[cc['GSA'] == cust]
            w, _ = v9._wmape(cc_c['Actual_Sales'], cc_c['Prediction'])
            b = v9._bias(cc_c['Actual_Sales'], cc_c['Prediction'])
            print(f' | {_fw(w):>10} {_fp(b):>7}', end='')
        print()

    # ---- Summary ----
    print('\n' + '=' * 100)
    print('SUMMARY')
    print('=' * 100)
    for vkey, label in variants:
        m = means[vkey]
        print(f'  {label:<22} WMAPE={_fw(m["wmape"])}  Bias={_fp(m["bias"])}  '
              f'FC_C={m["fc_c"]:.0f}  HIST={m["hist"]:.0f}')

    # Delta table
    print('\n  Incremental value of combining:')
    for key, label in [('relaxed', 'Relaxed+3 alone'),
                       ('combined', 'Relaxed+3 + CrossTF')]:
        dw = (means[key]['wmape'] - means['current']['wmape']) * 100
        db = (means[key]['bias'] - means['current']['bias']) * 100
        dfc = means[key]['fc_c'] - means['current']['fc_c']
        print(f'    {label:<26} dWMAPE={dw:+.1f}pp  dBias={db:+.1f}pp  '
              f'dFC_C={dfc:+.0f}')


if __name__ == '__main__':
    main()
