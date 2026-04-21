#!/usr/bin/env python3
"""Cross-TF Backtest: infer fresh TF=12 fc_product from shorter timeframes.

Four-way comparison per snapshot:
  A) Current:      two-curve (bias*cov), completed TF=12 only
  B) Product-Only: fc_product curve, completed TF=12 only
  C) CrossTF:      fc_product curve, completed TF=12 + synthetic from shorter TFs
  D) Oracle:       two-curve (bias*cov), all published TF=12
"""
import sys
sys.path.insert(0, '/home/user/techwriting')

import pandas as pd
import numpy as np
from scipy import stats as sp_stats
import warnings
warnings.filterwarnings('ignore')

import aging_v9 as v9

# ============================================================
# Constants
# ============================================================
CROSS_TF_SHORT = [9, 6, 3]
CROSS_TF_MIN_PAIRS = 5
CROSS_TF_MIN_R2 = 0.10
SYNTHETIC_DISCOUNT = 0.80
PRODUCT_FILTER_HI = 10.0

DATA_PATH = '/home/user/techwriting/training_data_anonymized.csv'
SNAP_START = '2024-06-01'
SNAP_END = '2025-03-01'
HORIZON = 3


# ============================================================
# Data loading (all timeframes, not just TF=12)
# ============================================================
def load_all_timeframes(path):
    df = pd.read_csv(
        path,
        parse_dates=['Reference_Month', 'Target_Period_Start', 'Target_Period_End'],
        encoding='utf-8-sig',
    )
    df['gsa_site'] = df['GSA'].astype(str) + '|' + df['Site'].astype(str)
    df['fc_coverage'] = np.where(
        df['Actual_Sales'] > 0, df['Covered_Orders'] / df['Actual_Sales'], np.nan)
    df['fc_bias'] = np.where(
        df['Covered_Orders'] > 0, df['Forecast_Value'] / df['Covered_Orders'], np.nan)
    df['fc_product'] = df['fc_coverage'] * df['fc_bias']
    if 'site_regime' in df.columns:
        df['site_regime'] = df['site_regime'].fillna(0.0)
    if 'forecast_error' in df.columns:
        df['forecast_error'] = df['forecast_error'].map(
            {True: True, False: False, 'True': True, 'False': False}
        ).fillna(False).astype(bool)
    return df


# ============================================================
# Cross-TF regression building
# ============================================================
def build_cross_tf_regressions(completed_df, snapshot):
    """Per-(site, short_tf) OLS: short fc_product -> TF=12 fc_product.
    Pairs matched on (site, start_date, lag), pooled across lags."""
    snap = pd.Timestamp(snapshot)
    regressions = {}

    completed = completed_df[
        (completed_df['Target_Period_End'] < snap) &
        (completed_df['Has_Forecast'] == 1)
    ].copy()

    tf12 = completed[completed['Timeframe'] == 12]
    tf12_lookup = tf12.set_index(['gsa_site', 'Target_Period_Start', 'Prediction_Lag'])

    for short_tf in CROSS_TF_SHORT:
        short = completed[completed['Timeframe'] == short_tf]
        short_lookup = short.set_index(['gsa_site', 'Target_Period_Start', 'Prediction_Lag'])

        common_idx = tf12_lookup.index.intersection(short_lookup.index)
        if len(common_idx) == 0:
            continue

        pairs = pd.DataFrame({
            'site': [idx[0] for idx in common_idx],
            'fp12': tf12_lookup.loc[common_idx, 'fc_product'].values,
            'fp_short': short_lookup.loc[common_idx, 'fc_product'].values,
        })
        pairs = pairs.dropna(subset=['fp12', 'fp_short'])
        pairs = pairs[(pairs['fp12'] > 0) & (pairs['fp12'] < PRODUCT_FILTER_HI) &
                       (pairs['fp_short'] > 0) & (pairs['fp_short'] < PRODUCT_FILTER_HI)]

        for site, grp in pairs.groupby('site'):
            if len(grp) < CROSS_TF_MIN_PAIRS:
                continue
            x = grp['fp_short'].values
            y = grp['fp12'].values
            if np.std(x) < 1e-9 or np.std(y) < 1e-9:
                continue
            slope, intercept, r, p, se = sp_stats.linregress(x, y)
            r2 = r ** 2
            if r2 >= CROSS_TF_MIN_R2:
                regressions[(site, short_tf)] = {
                    'slope': slope, 'intercept': intercept,
                    'r2': r2, 'n': len(grp), 'p_value': p,
                }
    return regressions


# ============================================================
# Synthetic observation inference
# ============================================================
def infer_synthetic_observations(all_df, snapshot, regressions):
    """For incomplete TF=12 periods, infer fc_product from completed shorter TFs."""
    snap = pd.Timestamp(snapshot)

    incomplete_12 = all_df[
        (all_df['Timeframe'] == 12) &
        (all_df['Target_Period_Start'] < snap) &
        (all_df['Target_Period_End'] >= snap) &
        (all_df['Has_Forecast'] == 1) &
        (all_df['Reference_Month'] < snap) &
        (all_df['Prediction_Lag'] <= v9.MAX_VINTAGE_LAG)
    ]

    completed_short = all_df[
        (all_df['Timeframe'].isin(CROSS_TF_SHORT)) &
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

        for short_tf in CROSS_TF_SHORT:
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
            if pd.isna(fp_short) or fp_short <= 0 or fp_short >= PRODUCT_FILTER_HI:
                continue

            inferred = model['slope'] * fp_short + model['intercept']
            inferred = np.clip(inferred, 0.01, PRODUCT_FILTER_HI)

            rows.append({
                'gsa_site': site,
                'Timeframe': 12,
                'Prediction_Lag': lag,
                'site_regime': r12['site_regime'],
                'Reference_Month': r12['Reference_Month'],
                'fc_product': inferred,
                'is_synthetic': True,
                'source_tf': short_tf,
                'regression_r2': model['r2'],
            })
            break  # use best available (longest TF first)

    if not rows:
        return pd.DataFrame(columns=[
            'gsa_site', 'Timeframe', 'Prediction_Lag', 'site_regime',
            'Reference_Month', 'fc_product', 'is_synthetic', 'source_tf',
            'regression_r2'])
    return pd.DataFrame(rows)


# ============================================================
# Product curve building (with synthetic discount)
# ============================================================
def build_product_curve(curve_train, synthetic_df=None):
    """{(gs, 12, lag, regime): recency_weighted_mean_fc_product}."""
    real = curve_train[['gsa_site', 'Timeframe', 'Prediction_Lag',
                         'site_regime', 'Reference_Month', 'fc_product']].copy()
    real['is_synthetic'] = False

    if synthetic_df is not None and len(synthetic_df) > 0:
        syn = synthetic_df[['gsa_site', 'Timeframe', 'Prediction_Lag',
                             'site_regime', 'Reference_Month', 'fc_product',
                             'is_synthetic']].copy()
        combined = pd.concat([real, syn], ignore_index=True)
    else:
        combined = real

    combined = combined[
        combined['fc_product'].notna() &
        (combined['fc_product'] > 0) &
        (combined['fc_product'] < PRODUCT_FILTER_HI)
    ]

    curve = {}
    for (gs, tf, lag, regime), grp in combined.groupby(
            ['gsa_site', 'Timeframe', 'Prediction_Lag', 'site_regime']):
        if len(grp) < v9.MIN_OBS_FLAT:
            continue
        wts = v9._recency_weights(grp['Reference_Month'])
        discount = np.where(grp['is_synthetic'].values, SYNTHETIC_DISCOUNT, 1.0)
        wts = wts * discount
        curve[(gs, tf, lag, regime)] = np.average(grp['fc_product'], weights=wts)
    return curve


# ============================================================
# Prediction with single fc_product curve
# ============================================================
def predict_product(test_df, product_curve, hist, registry):
    """Like v9.predict but uses estimate = fc_value / product_curve[cell]."""
    n = len(test_df)
    preds = np.full(n, np.nan)
    sources = np.empty(n, dtype=object)
    n_vint = np.zeros(n, dtype=int)
    n_corr = np.zeros(n, dtype=int)

    for i, (_, row) in enumerate(test_df.iterrows()):
        gs = row['gsa_site']
        tf = row['Timeframe']
        ref_m = row['Reference_Month']
        tp_key = (gs, row['Target_Period_Start'], row['Target_Period_End'])

        if tp_key in registry:
            valid = []
            for vint in registry[tp_key]:
                if vint['ref_month'] > ref_m:
                    continue
                if vint['lag'] > v9.MAX_VINTAGE_LAG:
                    continue
                if vint['forecast_error']:
                    continue
                fc_val = vint['fc_value']
                if fc_val <= 0:
                    continue
                key = (vint['gsa_site'], vint['timeframe'],
                       vint['lag'], vint['site_regime'])
                fp = product_curve.get(key)
                if fp and fp > 0.001:
                    valid.append((max(fc_val / fp, 0), vint['lag']))

            if valid:
                n_vint[i] = len(valid)
                n_corr[i] = len(valid)
                combined = max(v9._combine_vintage_estimates(valid), 0)
                preds[i] = combined
                sources[i] = ('FC_MULTI_CORRECTED' if len(valid) >= 2
                              else 'FC_SINGLE_CORRECTED')
                continue

        h = hist.get((gs, tf), hist.get(('FB', gs)))
        if h is not None and h > 0:
            preds[i] = h
            sources[i] = 'HISTORICAL'

    out = test_df.copy()
    out['Prediction'] = preds
    out['Prediction_Source'] = sources
    out['N_Vintages'] = n_vint
    out['N_Corrected_Vintages'] = n_corr
    out['Curve_Corrected'] = n_corr > 0
    return out


# ============================================================
# Scoring helper
# ============================================================
def score_variant(pred_df):
    fc_c = pred_df[pred_df['Prediction_Source'].isin(v9.SOURCES_FC_CORRECTED)]
    hist = pred_df[pred_df['Prediction_Source'] == 'HISTORICAL']
    return {
        'overall': v9._metric_block(pred_df),
        'FC_C': v9._metric_block(fc_c),
        'HIST': v9._metric_block(hist),
    }


# ============================================================
# Snapshot comparison (4 variants)
# ============================================================
def run_snapshot_comparison(all_df, snapshot, horizon_months):
    snap = pd.Timestamp(snapshot)
    horizon_end = snap + pd.DateOffset(months=horizon_months)

    df12 = all_df[all_df['Timeframe'] == 12].copy()

    train_12 = df12[df12['Target_Period_End'] < snap]
    test_12 = df12[
        (df12['Target_Period_Start'] >= snap) &
        (df12['Target_Period_Start'] < horizon_end) &
        (df12['Reference_Month'] < snap) &
        (df12['Prediction_Lag'].isin(v9.TEST_LAGS))
    ].copy()
    visible_12 = df12[df12['Reference_Month'] < snap]
    curve_train_12 = train_12[train_12['Has_Forecast'] == 1].copy()

    if len(train_12) == 0 or len(test_12) == 0:
        return None

    hist = v9.build_historical(train_12)
    registry = v9.build_forecast_registry(visible_12)

    # --- A) Current: two-curve, completed TF=12 ---
    cov_c = v9.build_flat_curves(curve_train_12, 'fc_coverage')
    bias_c = v9.build_flat_curves(curve_train_12, 'fc_bias')
    stats_c = v9.build_curve_stats(curve_train_12)
    pred_current = v9.predict(test_12, cov_c, bias_c, hist, registry, stats_c)

    # --- B) Product-Only: fc_product curve, completed TF=12 ---
    ct12 = curve_train_12.copy()
    ct12['fc_product'] = ct12['fc_bias'] * ct12['fc_coverage']
    prod_curve_b = build_product_curve(ct12)
    pred_product = predict_product(test_12, prod_curve_b, hist, registry)

    # --- C) CrossTF: fc_product curve, completed + synthetic ---
    regressions = build_cross_tf_regressions(all_df, snap)
    synthetic = infer_synthetic_observations(all_df, snap, regressions)
    prod_curve_c = build_product_curve(ct12, synthetic)
    pred_crosstf = predict_product(test_12, prod_curve_c, hist, registry)

    # --- D) Oracle: two-curve, all published TF=12 ---
    oracle_pool = df12[
        (df12['Has_Forecast'] == 1) & (df12['Reference_Month'] < snap)
    ].copy()
    oracle_pool = oracle_pool[~oracle_pool.index.isin(test_12.index)]
    cov_o = v9.build_flat_curves(oracle_pool, 'fc_coverage')
    bias_o = v9.build_flat_curves(oracle_pool, 'fc_bias')
    stats_o = v9.build_curve_stats(oracle_pool)
    pred_oracle = v9.predict(test_12, cov_o, bias_o, hist, registry, stats_o)

    # Regression diagnostics
    reg_summary = {}
    for stf in CROSS_TF_SHORT:
        models = {k: v for k, v in regressions.items() if k[1] == stf}
        r2s = [m['r2'] for m in models.values()]
        reg_summary[stf] = {
            'n_models': len(models),
            'median_r2': np.median(r2s) if r2s else np.nan,
            'mean_r2': np.mean(r2s) if r2s else np.nan,
        }
    syn_by_tf = {}
    if len(synthetic) > 0:
        for stf in CROSS_TF_SHORT:
            syn_by_tf[stf] = int((synthetic['source_tf'] == stf).sum())

    return {
        'snapshot': snapshot,
        'current': score_variant(pred_current),
        'product': score_variant(pred_product),
        'crosstf': score_variant(pred_crosstf),
        'oracle': score_variant(pred_oracle),
        'pred_current': pred_current,
        'pred_crosstf': pred_crosstf,
        'pred_oracle': pred_oracle,
        'reg_summary': reg_summary,
        'n_synthetic': len(synthetic),
        'syn_by_tf': syn_by_tf,
        'n_regressions': len(regressions),
    }


# ============================================================
# Reporting
# ============================================================
def _fp(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '   n/a'
    return f'{x*100:+.1f}%'

def _fw(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '  n/a'
    return f'{x*100:.1f}%'


def print_snapshot_table(results):
    print('\n' + '=' * 130)
    print('FOUR-WAY COMPARISON: Per-Snapshot Summary')
    print('=' * 130)
    hdr = (f'  {"Snap":<12} | {"--- CURRENT (2-crv) ---":^24} '
           f'| {"--- PRODUCT-ONLY ------":^24} '
           f'| {"--- CROSS-TF ----------":^24} '
           f'| {"--- ORACLE (2-crv) ----":^24}')
    sub = (f'  {"":12} | {"FC_C":>5} {"HIST":>5} {"WMAPE":>6} {"Bias":>7} '
           f'| {"FC_C":>5} {"HIST":>5} {"WMAPE":>6} {"Bias":>7} '
           f'| {"FC_C":>5} {"HIST":>5} {"WMAPE":>6} {"Bias":>7} '
           f'| {"FC_C":>5} {"HIST":>5} {"WMAPE":>6} {"Bias":>7}')
    print(hdr)
    print(sub)
    print('  ' + '-' * 128)

    for r in results:
        parts = []
        for var in ['current', 'product', 'crosstf', 'oracle']:
            m = r[var]
            fc_n = m['FC_C']['rows']
            h_n = m['HIST']['rows']
            w = _fw(m['overall']['wmape'])
            b = _fp(m['overall']['bias'])
            parts.append(f'{fc_n:>5} {h_n:>5} {w:>6} {b:>7}')
        print(f'  {r["snapshot"]:<12} | {" | ".join(parts)}')

    # Means
    print('  ' + '-' * 128)
    means = {}
    for var in ['current', 'product', 'crosstf', 'oracle']:
        wvals = [r[var]['overall']['wmape'] for r in results
                 if r[var]['overall']['wmape'] is not None
                 and not np.isnan(r[var]['overall']['wmape'])]
        bvals = [r[var]['overall']['bias'] for r in results
                 if r[var]['overall']['bias'] is not None
                 and not np.isnan(r[var]['overall']['bias'])]
        fc_rows = [r[var]['FC_C']['rows'] for r in results]
        h_rows = [r[var]['HIST']['rows'] for r in results]
        means[var] = {
            'wmape': np.mean(wvals) if wvals else np.nan,
            'bias': np.mean(bvals) if bvals else np.nan,
            'fc_c': np.mean(fc_rows),
            'hist': np.mean(h_rows),
        }
        parts_m = []
    for var in ['current', 'product', 'crosstf', 'oracle']:
        m = means[var]
        parts_m.append(f'{m["fc_c"]:>5.0f} {m["hist"]:>5.0f} '
                       f'{_fw(m["wmape"]):>6} {_fp(m["bias"]):>7}')
    print(f'  {"MEAN":<12} | {" | ".join(parts_m)}')
    return means


def print_fc_corrected_detail(results):
    print('\n' + '=' * 130)
    print('FC_CORRECTED Detail (the rows where FC correction was applied)')
    print('=' * 130)
    hdr = (f'  {"Snap":<12} | {"--- CURRENT ---":^18} '
           f'| {"--- PRODUCT ----":^18} '
           f'| {"--- CROSS-TF ---":^18} '
           f'| {"--- ORACLE -----":^18} '
           f'| {"Synth":>5}')
    sub = (f'  {"":12} | {"n":>4} {"WMAPE":>6} {"Bias":>7} '
           f'| {"n":>4} {"WMAPE":>6} {"Bias":>7} '
           f'| {"n":>4} {"WMAPE":>6} {"Bias":>7} '
           f'| {"n":>4} {"WMAPE":>6} {"Bias":>7} '
           f'| {"":>5}')
    print(hdr)
    print(sub)
    print('  ' + '-' * 110)

    for r in results:
        parts = []
        for var in ['current', 'product', 'crosstf', 'oracle']:
            m = r[var]['FC_C']
            parts.append(f'{m["rows"]:>4} {_fw(m["wmape"]):>6} {_fp(m["bias"]):>7}')
        parts.append(f'{r["n_synthetic"]:>5}')
        print(f'  {r["snapshot"]:<12} | {" | ".join(parts)}')


def print_regression_diagnostics(results):
    print('\n' + '=' * 90)
    print('REGRESSION DIAGNOSTICS')
    print('=' * 90)
    for r in results:
        syn_parts = []
        for stf in CROSS_TF_SHORT:
            rs = r['reg_summary'].get(stf, {})
            n_mod = rs.get('n_models', 0)
            med_r2 = rs.get('median_r2', np.nan)
            n_syn = r['syn_by_tf'].get(stf, 0)
            r2_str = f'{med_r2:.3f}' if not np.isnan(med_r2) else '  n/a'
            syn_parts.append(f'TF={stf}: {n_mod} models, R²={r2_str}, {n_syn} synth')
        print(f'  {r["snapshot"]}: {r["n_synthetic"]} synthetic total  |  '
              f'{" | ".join(syn_parts)}')


def print_per_customer(results):
    """Aggregate across snapshots, show per-customer bias comparison."""
    all_current = pd.concat([r['pred_current'] for r in results], ignore_index=True)
    all_crosstf = pd.concat([r['pred_crosstf'] for r in results], ignore_index=True)
    all_oracle = pd.concat([r['pred_oracle'] for r in results], ignore_index=True)

    print('\n' + '=' * 120)
    print('PER-CUSTOMER BIAS COMPARISON (all sources, across all snapshots)')
    print('=' * 120)
    print(f'  {"Customer":<16} {"Rows":>5} '
          f'| {"Curr WMAPE":>10} {"Curr Bias":>10} '
          f'| {"XTF WMAPE":>10} {"XTF Bias":>10} '
          f'| {"Orac WMAPE":>10} {"Orac Bias":>10} '
          f'| {"dBias C→X":>10}')
    print('  ' + '-' * 118)

    customers = sorted(all_current['GSA'].unique())
    for cust in customers:
        c = all_current[all_current['GSA'] == cust]
        x = all_crosstf[all_crosstf['GSA'] == cust]
        o = all_oracle[all_oracle['GSA'] == cust]
        if c['Actual_Sales'].sum() == 0:
            continue

        cw, _ = v9._wmape(c['Actual_Sales'], c['Prediction'])
        cb = v9._bias(c['Actual_Sales'], c['Prediction'])
        xw, _ = v9._wmape(x['Actual_Sales'], x['Prediction'])
        xb = v9._bias(x['Actual_Sales'], x['Prediction'])
        ow, _ = v9._wmape(o['Actual_Sales'], o['Prediction'])
        ob = v9._bias(o['Actual_Sales'], o['Prediction'])

        db = (xb - cb) if (xb is not None and cb is not None
              and not np.isnan(xb) and not np.isnan(cb)) else np.nan
        db_str = f'{db*100:+.1f}pp' if not np.isnan(db) else '   n/a'

        print(f'  {cust:<16} {len(c):>5} '
              f'| {_fw(cw):>10} {_fp(cb):>10} '
              f'| {_fw(xw):>10} {_fp(xb):>10} '
              f'| {_fw(ow):>10} {_fp(ob):>10} '
              f'| {db_str:>10}')


def print_head_to_head(results):
    """Rows that changed from HISTORICAL to FC_CORRECTED in CrossTF."""
    print('\n' + '=' * 100)
    print('HEAD-TO-HEAD: Rows upgraded HISTORICAL → FC_CORRECTED by CrossTF')
    print('=' * 100)
    print(f'  {"Snap":<12} {"Upgraded":>8} '
          f'{"Curr WMAPE":>11} {"XTF WMAPE":>10} '
          f'{"Curr Bias":>10} {"XTF Bias":>10}')
    print('  ' + '-' * 68)

    for r in results:
        pc = r['pred_current']
        px = r['pred_crosstf']
        curr_hist = pc[pc['Prediction_Source'] == 'HISTORICAL']
        if len(curr_hist) == 0:
            continue
        xtf_for_those = px.loc[curr_hist.index]
        upgraded = xtf_for_those[
            xtf_for_those['Prediction_Source'].isin(v9.SOURCES_FC_CORRECTED)]
        if len(upgraded) == 0:
            print(f'  {r["snapshot"]:<12} {0:>8}')
            continue

        curr_sub = curr_hist.loc[upgraded.index]
        cw, _ = v9._wmape(curr_sub['Actual_Sales'], curr_sub['Prediction'])
        xw, _ = v9._wmape(upgraded['Actual_Sales'], upgraded['Prediction'])
        cb = v9._bias(curr_sub['Actual_Sales'], curr_sub['Prediction'])
        xb = v9._bias(upgraded['Actual_Sales'], upgraded['Prediction'])
        print(f'  {r["snapshot"]:<12} {len(upgraded):>8} '
              f'{_fw(cw):>11} {_fw(xw):>10} '
              f'{_fp(cb):>10} {_fp(xb):>10}')


# ============================================================
# Main
# ============================================================
def main():
    print('=' * 90)
    print('Cross-TF Backtest: Infer Fresh TF=12 Curves from Shorter Timeframes')
    print('=' * 90)
    print(f'Constants: MIN_PAIRS={CROSS_TF_MIN_PAIRS}  MIN_R2={CROSS_TF_MIN_R2}  '
          f'DISCOUNT={SYNTHETIC_DISCOUNT}  HORIZON={HORIZON}mo')
    print(f'Snapshot grid: {SNAP_START} to {SNAP_END}')

    print('\nLoading all-timeframe data...')
    all_df = load_all_timeframes(DATA_PATH)
    print(f'  Loaded {len(all_df):,} rows across TF={sorted(all_df["Timeframe"].unique())}')
    print(f'  TF=12: {(all_df["Timeframe"]==12).sum():,}  '
          f'TF=9: {(all_df["Timeframe"]==9).sum():,}  '
          f'TF=6: {(all_df["Timeframe"]==6).sum():,}  '
          f'TF=3: {(all_df["Timeframe"]==3).sum():,}')

    snapshots = [d.strftime('%Y-%m-%d')
                 for d in v9.monthly_grid(SNAP_START, SNAP_END)]

    results = []
    for snap in snapshots:
        print(f'\n  Processing snapshot {snap}...')
        r = run_snapshot_comparison(all_df, snap, HORIZON)
        if r is None:
            print(f'    Skipped (empty train/test)')
            continue
        print(f'    Regressions: {r["n_regressions"]}  '
              f'Synthetic rows: {r["n_synthetic"]}')
        results.append(r)

    if not results:
        print('No valid snapshots.')
        return

    means = print_snapshot_table(results)
    print_fc_corrected_detail(results)
    print_regression_diagnostics(results)
    print_per_customer(results)
    print_head_to_head(results)

    # Final summary
    print('\n' + '=' * 90)
    print('SUMMARY')
    print('=' * 90)
    for var, label in [('current', 'Current (2-curve)'),
                       ('product', 'Product-Only'),
                       ('crosstf', 'CrossTF'),
                       ('oracle', 'Oracle (2-curve)')]:
        m = means[var]
        print(f'  {label:<22} WMAPE={_fw(m["wmape"])}  Bias={_fp(m["bias"])}  '
              f'FC_C={m["fc_c"]:.0f}  HIST={m["hist"]:.0f}')


if __name__ == '__main__':
    main()
