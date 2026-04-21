#!/usr/bin/env python3
"""Relaxed causality backtest: extend training visibility by +2 and +3 months.

Validates the idea that periods within 2-3 months of completion can be treated
as "effectively completed" (all orders are in due to 3-month lead times).

Comparison:
  A) Current:    Target_Period_End < snapshot (strict causality)
  B) Relaxed+2:  Target_Period_End < snapshot + 2 months
  C) Relaxed+3:  Target_Period_End < snapshot + 3 months
  D) CrossTF:    fc_product curve with cross-TF synthetic observations
  E) Oracle:     All published TF=12 (no causality restriction on training)

Relaxation applies to ALL training (curves + historical).
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
RELAXATION_MONTHS = [0, 2, 3]  # 0 = current (strict)


def run_relaxed_variant(df12, test_12, visible_12, snap, relax_months):
    """Run one variant with relaxed training filter."""
    relaxed_end = snap + pd.DateOffset(months=relax_months)
    train = df12[df12['Target_Period_End'] < relaxed_end].copy()
    curve_train = train[train['Has_Forecast'] == 1].copy()

    if len(train) == 0:
        return None, None

    hist = v9.build_historical(train)
    registry = v9.build_forecast_registry(visible_12)
    cov_curve = v9.build_flat_curves(curve_train, 'fc_coverage')
    bias_curve = v9.build_flat_curves(curve_train, 'fc_bias')
    c_stats = v9.build_curve_stats(curve_train)

    pred = v9.predict(test_12, cov_curve, bias_curve, hist, registry, c_stats)
    return pred, len(curve_train)


def run_snapshot(all_df, snapshot):
    snap = pd.Timestamp(snapshot)
    horizon_end = snap + pd.DateOffset(months=HORIZON)

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

    results = {'snapshot': snapshot}

    # A/B/C) Relaxed variants (0=current, 2, 3 months)
    for rm in RELAXATION_MONTHS:
        label = f'relax_{rm}'
        pred, n_curve = run_relaxed_variant(df12, test_12, visible_12, snap, rm)
        if pred is None:
            return None
        results[label] = xtf.score_variant(pred)
        results[f'{label}_pred'] = pred
        results[f'{label}_n_curve'] = n_curve

    # D) CrossTF (from existing prototype)
    train_12 = df12[df12['Target_Period_End'] < snap]
    curve_train_12 = train_12[train_12['Has_Forecast'] == 1].copy()
    ct12 = curve_train_12.copy()
    ct12['fc_product'] = ct12['fc_bias'] * ct12['fc_coverage']

    regressions = xtf.build_cross_tf_regressions(all_df, snap)
    synthetic = xtf.infer_synthetic_observations(all_df, snap, regressions)
    prod_curve = xtf.build_product_curve(ct12, synthetic)

    hist_strict = v9.build_historical(train_12)
    registry = v9.build_forecast_registry(visible_12)
    pred_xtf = xtf.predict_product(test_12, prod_curve, hist_strict, registry)
    results['crosstf'] = xtf.score_variant(pred_xtf)
    results['crosstf_pred'] = pred_xtf

    # E) Oracle
    oracle_pool = df12[
        (df12['Has_Forecast'] == 1) & (df12['Reference_Month'] < snap)
    ].copy()
    oracle_pool = oracle_pool[~oracle_pool.index.isin(test_12.index)]
    cov_o = v9.build_flat_curves(oracle_pool, 'fc_coverage')
    bias_o = v9.build_flat_curves(oracle_pool, 'fc_bias')
    stats_o = v9.build_curve_stats(oracle_pool)
    oracle_train = df12[
        (df12['Reference_Month'] < snap) &
        (~df12.index.isin(test_12.index))
    ]
    hist_oracle = v9.build_historical(oracle_train)
    pred_oracle = v9.predict(test_12, cov_o, bias_o, hist_oracle, registry, stats_o)
    results['oracle'] = xtf.score_variant(pred_oracle)
    results['oracle_pred'] = pred_oracle

    return results


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
    print('Relaxed Causality Backtest: +2mo and +3mo training visibility')
    print('=' * 100)

    print('\nLoading all-timeframe data...')
    all_df = xtf.load_all_timeframes(DATA_PATH)
    n12 = (all_df['Timeframe'] == 12).sum()
    print(f'  {len(all_df):,} total rows, {n12:,} TF=12')

    snapshots = [d.strftime('%Y-%m-%d')
                 for d in v9.monthly_grid(SNAP_START, SNAP_END)]

    all_results = []
    for snap in snapshots:
        print(f'  Processing {snap}...')
        r = run_snapshot(all_df, snap)
        if r is None:
            print(f'    Skipped')
            continue
        n0 = r['relax_0_n_curve']
        n2 = r['relax_2_n_curve']
        n3 = r['relax_3_n_curve']
        print(f'    Curve train rows: strict={n0}  +2mo={n2}  +3mo={n3}  '
              f'(+{n2-n0}/{n3-n0} new)')
        all_results.append(r)

    if not all_results:
        print('No valid snapshots.')
        return

    # ---- Overall table ----
    variants = [
        ('relax_0', 'Current (strict)'),
        ('relax_2', 'Relaxed +2mo'),
        ('relax_3', 'Relaxed +3mo'),
        ('crosstf', 'CrossTF'),
        ('oracle', 'Oracle'),
    ]

    print('\n' + '=' * 140)
    print('FIVE-WAY COMPARISON: Per-Snapshot Overall')
    print('=' * 140)
    hdr_parts = [f'{"Snap":<12}']
    sub_parts = [f'{"":12}']
    for _, label in variants:
        hdr_parts.append(f'{label:^24}')
        sub_parts.append(f'{"FC_C":>5} {"HIST":>5} {"WMAPE":>6} {"Bias":>7}')
    print('  ' + ' | '.join(hdr_parts))
    print('  ' + ' | '.join(sub_parts))
    print('  ' + '-' * 138)

    for r in all_results:
        parts = [f'{r["snapshot"]:<12}']
        for vkey, _ in variants:
            m = r[vkey]
            fc_n = m['FC_C']['rows']
            h_n = m['HIST']['rows']
            w = _fw(m['overall']['wmape'])
            b = _fp(m['overall']['bias'])
            parts.append(f'{fc_n:>5} {h_n:>5} {w:>6} {b:>7}')
        print('  ' + ' | '.join(parts))

    # Means
    print('  ' + '-' * 138)
    mean_parts = [f'{"MEAN":<12}']
    means = {}
    for vkey, _ in variants:
        wvals = [r[vkey]['overall']['wmape'] for r in all_results
                 if r[vkey]['overall']['wmape'] is not None
                 and not np.isnan(r[vkey]['overall']['wmape'])]
        bvals = [r[vkey]['overall']['bias'] for r in all_results
                 if r[vkey]['overall']['bias'] is not None
                 and not np.isnan(r[vkey]['overall']['bias'])]
        fc_rows = [r[vkey]['FC_C']['rows'] for r in all_results]
        h_rows = [r[vkey]['HIST']['rows'] for r in all_results]
        m_w = np.mean(wvals) if wvals else np.nan
        m_b = np.mean(bvals) if bvals else np.nan
        m_fc = np.mean(fc_rows)
        m_h = np.mean(h_rows)
        means[vkey] = {'wmape': m_w, 'bias': m_b, 'fc_c': m_fc, 'hist': m_h}
        mean_parts.append(f'{m_fc:>5.0f} {m_h:>5.0f} {_fw(m_w):>6} {_fp(m_b):>7}')
    print('  ' + ' | '.join(mean_parts))

    # ---- FC_CORRECTED detail ----
    print('\n' + '=' * 140)
    print('FC_CORRECTED Detail')
    print('=' * 140)
    hdr2 = [f'{"Snap":<12}']
    sub2 = [f'{"":12}']
    for _, label in variants:
        hdr2.append(f'{label:^20}')
        sub2.append(f'{"n":>4} {"WMAPE":>6} {"Bias":>7}')
    print('  ' + ' | '.join(hdr2))
    print('  ' + ' | '.join(sub2))
    print('  ' + '-' * 130)

    for r in all_results:
        parts = [f'{r["snapshot"]:<12}']
        for vkey, _ in variants:
            m = r[vkey]['FC_C']
            parts.append(f'{m["rows"]:>4} {_fw(m["wmape"]):>6} {_fp(m["bias"]):>7}')
        print('  ' + ' | '.join(parts))

    # ---- Per-customer ----
    print('\n' + '=' * 130)
    print('PER-CUSTOMER BIAS COMPARISON (all sources, all snapshots)')
    print('=' * 130)
    hdr3 = f'  {"Customer":<16} {"Rows":>5}'
    for _, label in variants:
        hdr3 += f' | {label:>12} {"Bias":>7}'
    print(hdr3)
    print('  ' + '-' * 128)

    # Collect all predictions
    all_preds = {vkey: [] for vkey, _ in variants}
    pred_keys = {
        'relax_0': 'relax_0_pred', 'relax_2': 'relax_2_pred',
        'relax_3': 'relax_3_pred', 'crosstf': 'crosstf_pred',
        'oracle': 'oracle_pred',
    }
    for r in all_results:
        for vkey, _ in variants:
            all_preds[vkey].append(r[pred_keys[vkey]])

    combined = {vkey: pd.concat(preds, ignore_index=True)
                for vkey, preds in all_preds.items()}

    customers = sorted(combined['relax_0']['GSA'].unique())
    for cust in customers:
        c0 = combined['relax_0']
        c0c = c0[c0['GSA'] == cust]
        if c0c['Actual_Sales'].sum() == 0:
            continue
        row_str = f'  {cust:<16} {len(c0c):>5}'
        for vkey, _ in variants:
            cc = combined[vkey]
            cc_cust = cc[cc['GSA'] == cust]
            b = v9._bias(cc_cust['Actual_Sales'], cc_cust['Prediction'])
            w, _ = v9._wmape(cc_cust['Actual_Sales'], cc_cust['Prediction'])
            row_str += f' | {_fw(w):>12} {_fp(b):>7}'
        print(row_str)

    # ---- Summary ----
    print('\n' + '=' * 100)
    print('SUMMARY')
    print('=' * 100)
    for vkey, label in variants:
        m = means[vkey]
        print(f'  {label:<22} WMAPE={_fw(m["wmape"])}  Bias={_fp(m["bias"])}  '
              f'FC_C={m["fc_c"]:.0f}  HIST={m["hist"]:.0f}')


if __name__ == '__main__':
    main()
