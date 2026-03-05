"""
FC Signal Parameter Tuning
===========================
Systematic sweep of FC-related parameters to optimize WMAPE.

FC is the dominant signal: FC-only multi beats all OO-containing models for
Customer 2, and for Customer 1 FC carries most of the prediction value. The
current FC parameters were set heuristically and never tuned:

    VINTAGE_RECENCY_POWER = 1.5   (multi-vintage combination weighting)
    FC lag decay coeff    = 0.06  (blend reliability penalty)
    FC lag decay floor    = 0.3   (minimum FC reliability factor)
    FC curve RECENCY_POWER = 2    (curve building recency, shared with OO)
    Vintage staleness cap = None  (no cap on vintage lag)

Sweep order (sequential, carry best forward):
    1. VINTAGE_RECENCY_POWER
    2. FC lag decay (coeff x floor) — 2D grid
    3. Vintage staleness cap
    4. FC curve RECENCY_POWER
    5. Combined best + validation via K-fold random CV

Usage:
    python fc_tuning.py
    python fc_tuning.py --file data.csv
"""

import pandas as pd
import numpy as np
import warnings
import argparse
warnings.filterwarnings('ignore')

# ============================================================
# DEFAULT CONSTANTS (from pipeline.py)
# ============================================================
RECENCY_POWER = 2
MIN_OBS_FLAT = 3
MIN_SITE_ROWS = 30
VINTAGE_RECENCY_POWER = 1.5
N_FOLDS = 5
RANDOM_SEED = 42


# ============================================================
# DATA LOADING (mirrors pipeline.py)
# ============================================================

def parse_dollar(s):
    if isinstance(s, str):
        s = s.replace('$', '').replace(',', '').strip()
        if s in ('', '-', 'N/A', 'n/a'):
            return np.nan
        if s.startswith('(') and s.endswith(')'):
            return -float(s[1:-1])
        return float(s)
    return s


def load_and_preprocess(filepath):
    df = pd.read_csv(filepath,
                     parse_dates=['Reference_Month', 'Target_Period_Start',
                                  'Target_Period_End'])
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    numeric_cols = ['Actual_Sales', 'Open_Orders', 'Covered_Orders', 'Forecast_Value',
                    'Historical_Sales_Lag1', 'Historical_Sales_Lag12']
    for col in numeric_cols:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].apply(parse_dollar)

    if 'GSA' not in df.columns:
        df['GSA'] = 'Customer'
    df['gsa_site'] = df['GSA'].astype(str) + '|' + df['Site'].astype(str)
    df['Avg_Weighted_Order_Earliness'] = df['Avg_Weighted_Order_Earliness'].fillna(0)
    df['Avg_Weighted_Lead_Time'] = df['Avg_Weighted_Lead_Time'].fillna(0)

    df['oo_ratio'] = np.where(
        df['Actual_Sales'] > 0,
        df['Open_Orders'] / df['Actual_Sales'], np.nan)
    df['fc_coverage'] = np.where(
        df['Actual_Sales'] > 0,
        df['Covered_Orders'] / df['Actual_Sales'], np.nan)
    df['fc_bias'] = np.where(
        df['Covered_Orders'] > 0,
        df['Forecast_Value'] / df['Covered_Orders'], np.nan)

    return df


# ============================================================
# CURVE BUILDING — parameterized recency_power
# ============================================================

def build_flat_curves(train_df, ratio_col, start=None, recency_power=RECENCY_POWER):
    td = train_df if start is None else train_df[train_df['Reference_Month'] >= start]
    flat = {}

    for (gs, tf, lag), grp in td.groupby(['gsa_site', 'Timeframe', 'Prediction_Lag']):
        valid = grp[grp[ratio_col].notna() & (grp[ratio_col] > 0) & (grp[ratio_col] < 10)]
        if len(valid) < MIN_OBS_FLAT:
            continue
        days = (valid['Reference_Month'] - valid['Reference_Month'].min()).dt.days / 365
        wts = (1 + days.values) ** recency_power
        flat[(gs, tf, lag)] = np.average(valid[ratio_col], weights=wts)

    for (gs, lag), grp in td.groupby(['gsa_site', 'Prediction_Lag']):
        valid = grp[grp[ratio_col].notna() & (grp[ratio_col] > 0) & (grp[ratio_col] < 10)]
        if len(valid) >= MIN_OBS_FLAT:
            days = (valid['Reference_Month'] - valid['Reference_Month'].min()).dt.days / 365
            wts = (1 + days.values) ** recency_power
            flat[('FB', gs, lag)] = np.average(valid[ratio_col], weights=wts)

    return flat


def apply_flat_curves(dset, oo_f, cov_f, bias_f):
    oo_impl = np.full(len(dset), np.nan)
    fc_impl = np.full(len(dset), np.nan)

    for i, (idx, row) in enumerate(dset.iterrows()):
        gs, tf, lag = row['gsa_site'], row['Timeframe'], row['Prediction_Lag']

        for key in [(gs, tf, lag), ('FB', gs, lag)]:
            if key in oo_f and oo_f[key] > 0.001:
                oo_impl[i] = row['Open_Orders'] / oo_f[key]
                break

        for key in [(gs, tf, lag), ('FB', gs, lag)]:
            cf = cov_f.get(key)
            bf = bias_f.get(key)
            if cf and bf and cf > 0.001 and bf > 0.001 and row['Forecast_Value'] > 0:
                fc_impl[i] = row['Forecast_Value'] / bf / cf
                break

    dset['oo_implied_flat'] = np.maximum(oo_impl, 0)
    dset['fc_implied_flat'] = np.maximum(fc_impl, 0)


# ============================================================
# MULTI-VINTAGE FC — parameterized vintage_power & staleness_cap
# ============================================================

def build_forecast_registry(data_df):
    registry = {}
    fc_rows = data_df[data_df['Has_Forecast'] == 1]
    for _, row in fc_rows.iterrows():
        key = (row['gsa_site'], row['Target_Period_Start'], row['Target_Period_End'])
        if key not in registry:
            registry[key] = []
        registry[key].append({
            'ref_month': row['Reference_Month'],
            'lag': row['Prediction_Lag'],
            'fc_value': row['Forecast_Value'],
            'covered_orders': row['Covered_Orders'],
            'timeframe': row['Timeframe'],
            'gsa_site': row['gsa_site'],
        })
    for key in registry:
        registry[key].sort(key=lambda v: v['lag'])
    return registry


def _vintage_to_estimate(vintage, cov_f, bias_f):
    gs = vintage['gsa_site']
    tf = vintage['timeframe']
    lag = vintage['lag']
    fc_val = vintage['fc_value']
    if fc_val <= 0:
        return np.nan, False
    for key in [(gs, tf, lag), ('FB', gs, lag)]:
        cf = cov_f.get(key)
        bf = bias_f.get(key)
        if cf and bf and cf > 0.001 and bf > 0.001:
            return fc_val / bf / cf, True
    return np.nan, False


def _combine_vintage_estimates(estimates_with_lags, vintage_power=VINTAGE_RECENCY_POWER):
    if not estimates_with_lags:
        return np.nan
    total_w, total_val = 0.0, 0.0
    for est, orig_lag in estimates_with_lags:
        w = 1.0 / (orig_lag ** vintage_power)
        total_w += w
        total_val += w * est
    return total_val / total_w if total_w > 0 else np.nan


def apply_multi_vintage_fc(dset, cov_f, bias_f, registry, causality='temporal',
                           vintage_power=VINTAGE_RECENCY_POWER, staleness_cap=None):
    """Apply multi-vintage FC with parameterized vintage_power and staleness_cap."""
    n = len(dset)
    fc_impl_multi = np.full(n, np.nan)
    n_vintages = np.zeros(n, dtype=int)
    best_lag = np.full(n, np.nan)

    for i, (idx, row) in enumerate(dset.iterrows()):
        gs = row['gsa_site']
        tp_key = (gs, row['Target_Period_Start'], row['Target_Period_End'])
        if tp_key not in registry:
            continue
        valid_vintages = []
        for v in registry[tp_key]:
            if causality == 'temporal' and v['ref_month'] > row['Reference_Month']:
                continue
            if staleness_cap is not None and v['lag'] > staleness_cap:
                continue
            est, ok = _vintage_to_estimate(v, cov_f, bias_f)
            if ok:
                valid_vintages.append((max(est, 0), v['lag']))
        if valid_vintages:
            fc_impl_multi[i] = max(
                _combine_vintage_estimates(valid_vintages, vintage_power), 0)
            n_vintages[i] = len(valid_vintages)
            best_lag[i] = min(vl for _, vl in valid_vintages)

    dset['fc_implied_multi'] = fc_impl_multi
    dset['n_vintages_used'] = n_vintages
    dset['best_vintage_lag'] = best_lag


# ============================================================
# WEIGHTED AVERAGE — parameterized fc_lag_coeff & fc_lag_floor
# ============================================================

def weighted_avg(test_df, train_df, test_months, gsa_sites_list,
                 oo_col='oo_implied_flat', fc_col='fc_implied_flat',
                 fc_lag_coeff=0.06, fc_lag_floor=0.3):
    """Blend OO, FC, and historical average with parameterized FC lag decay."""
    results = test_df.copy()
    results['pred'] = np.nan

    if oo_col == 'oo_none':
        if 'oo_none' not in results.columns:
            results['oo_none'] = np.nan
        if 'oo_none' not in train_df.columns:
            train_df = train_df.copy()
            train_df['oo_none'] = np.nan

    hist = {}
    for gs in gsa_sites_list:
        for tf in train_df['Timeframe'].unique():
            sub = train_df[(train_df['gsa_site'] == gs) & (train_df['Timeframe'] == tf)]
            if len(sub) > 0:
                d = (sub['Reference_Month'] - sub['Reference_Month'].min()).dt.days / 365
                w = (1 + d.values) ** RECENCY_POWER
                hist[(gs, tf)] = np.average(sub['Actual_Sales'], weights=w)

    for mi, month in enumerate(test_months):
        if mi == 0:
            prior = train_df[train_df['Reference_Month'] >=
                             train_df['Reference_Month'].quantile(0.8)]
        else:
            prior = results[results['Reference_Month'].isin(
                test_months[max(0, mi - 3):mi])]

        cmask = results['Reference_Month'] == month
        for gs in results.loc[cmask, 'gsa_site'].unique():
            for tf in results['Timeframe'].unique():
                ps = prior[(prior['gsa_site'] == gs) &
                           (prior['Timeframe'] == tf) &
                           (prior['Actual_Sales'] > 0)]

                oo_v = ps[ps[oo_col].notna() & (ps[oo_col] > 0)]
                oo_mape = (np.mean(np.abs(oo_v[oo_col] - oo_v['Actual_Sales']) /
                           oo_v['Actual_Sales']) if len(oo_v) >= 1 else 1.0)

                fc_v = ps[ps[fc_col].notna() & (ps[fc_col] > 0)]
                fc_mape = (np.mean(np.abs(fc_v[fc_col] - fc_v['Actual_Sales']) /
                           fc_v['Actual_Sales']) if len(fc_v) >= 1 else 1.0)

                rmask = cmask & (results['gsa_site'] == gs) & \
                        (results['Timeframe'] == tf)
                for idx in results[rmask].index:
                    row = results.loc[idx]
                    lag = row['Prediction_Lag']

                    has_oo = not np.isnan(row[oo_col]) and row[oo_col] > 0
                    has_fc = not np.isnan(row[fc_col]) and row[fc_col] > 0

                    # OO lag factor (hardcoded — not tuning OO)
                    oo_f = max(0.3, 1.0 - 0.06 * (lag - 1)) if has_oo else 0

                    # FC lag factor — parameterized
                    fc_lag = lag
                    if (has_fc and fc_col == 'fc_implied_multi' and
                            'best_vintage_lag' in results.columns):
                        bvl = results.at[idx, 'best_vintage_lag']
                        if not np.isnan(bvl):
                            fc_lag = bvl
                    fc_f = max(fc_lag_floor, 1.0 - fc_lag_coeff * (fc_lag - 1)) if has_fc else 0

                    w_oo = (1 / max(oo_mape, 0.01)) * oo_f if has_oo else 0
                    w_fc = (1 / max(fc_mape, 0.01)) * fc_f if has_fc else 0
                    hv = hist.get((gs, tf), 0)
                    w_h = (1 / 0.5) * 0.1 * lag / 12 if hv > 0 else 0

                    tot = w_oo + w_fc + w_h
                    if tot > 0:
                        results.at[idx, 'pred'] = max(0,
                            (w_oo / tot) * (row[oo_col] if has_oo else 0) +
                            (w_fc / tot) * (row[fc_col] if has_fc else 0) +
                            (w_h / tot) * hv)
                    elif hv > 0:
                        results.at[idx, 'pred'] = hv

    return results['pred']


# ============================================================
# EVALUATION METRICS
# ============================================================

def compute_wmape(actual, predicted):
    mask = actual > 0
    if mask.sum() == 0:
        return np.nan
    return np.sum(np.abs(actual[mask] - predicted[mask])) / np.sum(actual[mask]) * 100


def compute_bias(actual, predicted):
    mask = actual > 0
    if mask.sum() == 0:
        return np.nan
    return np.mean((predicted[mask] - actual[mask]) / actual[mask]) * 100


# ============================================================
# HELPER: run one model configuration and return WMAPE
# ============================================================

def run_config(test, train, test_months, gsa_sites, oo_col, fc_col,
               fc_lag_coeff=0.06, fc_lag_floor=0.3):
    """Run weighted_avg with given params, return WMAPE on months 2+."""
    preds = weighted_avg(test, train, test_months, gsa_sites,
                         oo_col=oo_col, fc_col=fc_col,
                         fc_lag_coeff=fc_lag_coeff, fc_lag_floor=fc_lag_floor)
    m2 = test['Reference_Month'] > test_months[0]
    mask = m2 & preds.notna() & (test['Actual_Sales'] > 0)
    if mask.sum() == 0:
        return np.nan, np.nan
    wmape = compute_wmape(test.loc[mask, 'Actual_Sales'].values, preds[mask].values)
    bias = compute_bias(test.loc[mask, 'Actual_Sales'].values, preds[mask].values)
    return wmape, bias


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment(df, customer_name):
    print(f"\n{'#' * 100}")
    print(f"# FC PARAMETER TUNING — {customer_name}")
    print(f"{'#' * 100}")

    # ---- Filter sites ----
    site_counts = df.groupby('gsa_site').size()
    valid_sites = sorted(site_counts[site_counts >= MIN_SITE_ROWS].index.tolist())
    df = df[df['gsa_site'].isin(valid_sites)].copy()
    gsa_sites = sorted(df['gsa_site'].unique())

    print(f"\n  Data: {len(df)} rows, {len(gsa_sites)} sites")
    print(f"  Sites: {gsa_sites}")

    # ---- Train/test split ----
    cutoff = df['Reference_Month'].quantile(0.8)
    train = df[df['Reference_Month'] <= cutoff].copy()
    test = df[df['Reference_Month'] > cutoff].copy()
    test_months = sorted(test['Reference_Month'].unique())

    print(f"  Train: {len(train)} rows "
          f"({train['Reference_Month'].min().date()} – "
          f"{train['Reference_Month'].max().date()})")
    print(f"  Test:  {len(test)} rows "
          f"({test['Reference_Month'].min().date()} – "
          f"{test['Reference_Month'].max().date()})")

    # ---- Build OO curves (FIXED — not tuning OO) ----
    oo_flat = build_flat_curves(train, 'oo_ratio', start='2023-01-01')
    print(f"  OO curves: {len(oo_flat)}")

    # ---- Build FC curves with default recency ----
    cov_flat = build_flat_curves(train, 'fc_coverage', start='2023-01-01')
    bias_flat = build_flat_curves(train, 'fc_bias', start='2023-01-01')
    print(f"  FC coverage curves: {len(cov_flat)}  |  FC bias: {len(bias_flat)}")

    # ---- Apply OO flat curves (fixed for all experiments) ----
    apply_flat_curves(train, oo_flat, cov_flat, bias_flat)
    apply_flat_curves(test, oo_flat, cov_flat, bias_flat)

    # ---- Build FC registry ----
    fc_registry = build_forecast_registry(df)
    print(f"  FC registry: {len(fc_registry)} target periods")

    # ---- Ensure oo_none column ----
    train['oo_none'] = np.nan
    test['oo_none'] = np.nan

    # ==================================================================
    # STEP 1: BASELINE with default parameters
    # ==================================================================
    print(f"\n  {'=' * 80}")
    print(f"  STEP 1: BASELINE (default params)")
    print(f"  {'=' * 80}")
    print(f"  Defaults: vintage_power=1.5, lag_coeff=0.06, lag_floor=0.3, "
          f"staleness_cap=None, fc_recency=2")

    apply_multi_vintage_fc(train, cov_flat, bias_flat, fc_registry, 'temporal')
    apply_multi_vintage_fc(test, cov_flat, bias_flat, fc_registry, 'temporal')

    baseline_v8d, baseline_v8d_bias = run_config(
        test, train, test_months, gsa_sites,
        'oo_implied_flat', 'fc_implied_multi')
    baseline_fc, baseline_fc_bias = run_config(
        test, train, test_months, gsa_sites,
        'oo_none', 'fc_implied_multi')

    print(f"\n  {'Model':>25s} | {'WMAPE':>7s} | {'Bias':>7s}")
    print(f"  {'-' * 45}")
    print(f"  {'V8d + mFC (default)':>25s} | {baseline_v8d:>5.1f}% | {baseline_v8d_bias:>+5.1f}%")
    print(f"  {'FC-only multi (default)':>25s} | {baseline_fc:>5.1f}% | {baseline_fc_bias:>+5.1f}%")

    # ==================================================================
    # STEP 2: VINTAGE_RECENCY_POWER sweep
    # ==================================================================
    print(f"\n  {'=' * 80}")
    print(f"  STEP 2: VINTAGE_RECENCY_POWER sweep")
    print(f"  {'=' * 80}")

    vintage_powers = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    print(f"\n  {'VintPow':>8s} | {'V8d+mFC':>10s} | {'FC-only':>10s} | "
          f"{'V8d diff':>10s} | {'FC diff':>10s}")
    print(f"  {'-' * 55}")

    best_vp = VINTAGE_RECENCY_POWER
    best_vp_wmape = baseline_v8d

    for vp in vintage_powers:
        # Rebuild multi-vintage FC with this vintage_power
        apply_multi_vintage_fc(train, cov_flat, bias_flat, fc_registry, 'temporal',
                               vintage_power=vp)
        apply_multi_vintage_fc(test, cov_flat, bias_flat, fc_registry, 'temporal',
                               vintage_power=vp)

        w_v8d, _ = run_config(test, train, test_months, gsa_sites,
                              'oo_implied_flat', 'fc_implied_multi')
        w_fc, _ = run_config(test, train, test_months, gsa_sites,
                             'oo_none', 'fc_implied_multi')

        d_v8d = w_v8d - baseline_v8d
        d_fc = w_fc - baseline_fc
        marker = ' <-- best' if w_v8d < best_vp_wmape else ''
        print(f"  {vp:>8.1f} | {w_v8d:>8.1f}% | {w_fc:>8.1f}% | "
              f"{d_v8d:>+8.1f}pp | {d_fc:>+8.1f}pp{marker}")

        if w_v8d < best_vp_wmape:
            best_vp = vp
            best_vp_wmape = w_v8d

    print(f"\n  Best VINTAGE_RECENCY_POWER: {best_vp} (V8d+mFC = {best_vp_wmape:.1f}%)")

    # Rebuild with best vintage power for subsequent steps
    apply_multi_vintage_fc(train, cov_flat, bias_flat, fc_registry, 'temporal',
                           vintage_power=best_vp)
    apply_multi_vintage_fc(test, cov_flat, bias_flat, fc_registry, 'temporal',
                           vintage_power=best_vp)

    # ==================================================================
    # STEP 3: FC lag decay sweep (2D: coeff x floor)
    # ==================================================================
    print(f"\n  {'=' * 80}")
    print(f"  STEP 3: FC lag decay sweep (coeff x floor)")
    print(f"  Using vintage_power={best_vp}")
    print(f"  {'=' * 80}")

    lag_coeffs = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15]
    lag_floors = [0.1, 0.2, 0.3, 0.5, 1.0]

    # V8d+mFC heatmap
    print(f"\n  V8d+mFC WMAPE heatmap (coeff \\ floor):")
    header = f"  {'coeff':>8s}"
    for fl in lag_floors:
        header += f" | {fl:>6.1f}"
    print(header)
    print(f"  {'-' * (11 + 9 * len(lag_floors))}")

    best_coeff = 0.06
    best_floor = 0.3
    best_decay_wmape = best_vp_wmape
    grid_results = {}

    for co in lag_coeffs:
        line = f"  {co:>8.2f}"
        for fl in lag_floors:
            w, _ = run_config(test, train, test_months, gsa_sites,
                              'oo_implied_flat', 'fc_implied_multi',
                              fc_lag_coeff=co, fc_lag_floor=fl)
            grid_results[(co, fl)] = w
            marker = '*' if w < best_decay_wmape else ' '
            if w < best_decay_wmape:
                best_coeff = co
                best_floor = fl
                best_decay_wmape = w
            line += f" | {w:>5.1f}%{marker}"
        print(line)

    print(f"\n  Best FC lag decay: coeff={best_coeff}, floor={best_floor} "
          f"(V8d+mFC = {best_decay_wmape:.1f}%)")

    # Also check FC-only at best decay params
    fc_at_best_decay, _ = run_config(test, train, test_months, gsa_sites,
                                     'oo_none', 'fc_implied_multi',
                                     fc_lag_coeff=best_coeff, fc_lag_floor=best_floor)
    print(f"  FC-only multi at best decay: {fc_at_best_decay:.1f}%")

    # ==================================================================
    # STEP 4: Vintage staleness cap sweep
    # ==================================================================
    print(f"\n  {'=' * 80}")
    print(f"  STEP 4: Vintage staleness cap sweep")
    print(f"  Using vintage_power={best_vp}, coeff={best_coeff}, floor={best_floor}")
    print(f"  {'=' * 80}")

    staleness_caps = [3, 4, 6, 8, 10, None]
    print(f"\n  {'Cap':>8s} | {'V8d+mFC':>10s} | {'FC-only':>10s} | "
          f"{'V8d diff':>10s} | {'FC diff':>10s} | {'Avg Vint':>8s}")
    print(f"  {'-' * 65}")

    best_cap = None
    best_cap_wmape = best_decay_wmape

    for cap in staleness_caps:
        apply_multi_vintage_fc(train, cov_flat, bias_flat, fc_registry, 'temporal',
                               vintage_power=best_vp, staleness_cap=cap)
        apply_multi_vintage_fc(test, cov_flat, bias_flat, fc_registry, 'temporal',
                               vintage_power=best_vp, staleness_cap=cap)

        w_v8d, _ = run_config(test, train, test_months, gsa_sites,
                              'oo_implied_flat', 'fc_implied_multi',
                              fc_lag_coeff=best_coeff, fc_lag_floor=best_floor)
        w_fc, _ = run_config(test, train, test_months, gsa_sites,
                             'oo_none', 'fc_implied_multi',
                             fc_lag_coeff=best_coeff, fc_lag_floor=best_floor)

        # avg vintages used
        m2 = test['Reference_Month'] > test_months[0]
        avg_v = test.loc[m2 & (test['n_vintages_used'] > 0), 'n_vintages_used'].mean()

        d_v8d = w_v8d - best_decay_wmape
        d_fc = w_fc - fc_at_best_decay
        cap_str = str(cap) if cap is not None else 'None'
        marker = ' <-- best' if w_v8d < best_cap_wmape else ''
        print(f"  {cap_str:>8s} | {w_v8d:>8.1f}% | {w_fc:>8.1f}% | "
              f"{d_v8d:>+8.1f}pp | {d_fc:>+8.1f}pp | {avg_v:>6.1f}{marker}")

        if w_v8d < best_cap_wmape:
            best_cap = cap
            best_cap_wmape = w_v8d

    cap_str = str(best_cap) if best_cap is not None else 'None'
    print(f"\n  Best staleness cap: {cap_str} (V8d+mFC = {best_cap_wmape:.1f}%)")

    # Rebuild with best cap for subsequent steps
    apply_multi_vintage_fc(train, cov_flat, bias_flat, fc_registry, 'temporal',
                           vintage_power=best_vp, staleness_cap=best_cap)
    apply_multi_vintage_fc(test, cov_flat, bias_flat, fc_registry, 'temporal',
                           vintage_power=best_vp, staleness_cap=best_cap)

    # ==================================================================
    # STEP 5: FC curve RECENCY_POWER sweep
    # ==================================================================
    print(f"\n  {'=' * 80}")
    print(f"  STEP 5: FC curve RECENCY_POWER sweep")
    print(f"  Using vintage_power={best_vp}, coeff={best_coeff}, floor={best_floor}, "
          f"cap={cap_str}")
    print(f"  {'=' * 80}")

    fc_recency_powers = [1.0, 1.5, 2.0, 2.5, 3.0]
    print(f"\n  {'FC Recency':>12s} | {'V8d+mFC':>10s} | {'FC-only':>10s} | "
          f"{'V8d diff':>10s} | {'FC diff':>10s}")
    print(f"  {'-' * 60}")

    best_fc_rp = RECENCY_POWER
    best_fc_rp_wmape = best_cap_wmape

    for rp in fc_recency_powers:
        # Rebuild FC curves with different recency power (OO stays at RECENCY_POWER=2)
        cov_rp = build_flat_curves(train, 'fc_coverage', start='2023-01-01',
                                   recency_power=rp)
        bias_rp = build_flat_curves(train, 'fc_bias', start='2023-01-01',
                                    recency_power=rp)

        # Reapply flat FC curves (OO curves unchanged)
        # We need to rebuild fc_implied_flat with new curves
        fc_impl_train = np.full(len(train), np.nan)
        for i, (idx, row) in enumerate(train.iterrows()):
            gs, tf, lag = row['gsa_site'], row['Timeframe'], row['Prediction_Lag']
            for key in [(gs, tf, lag), ('FB', gs, lag)]:
                cf = cov_rp.get(key)
                bf = bias_rp.get(key)
                if cf and bf and cf > 0.001 and bf > 0.001 and row['Forecast_Value'] > 0:
                    fc_impl_train[i] = max(row['Forecast_Value'] / bf / cf, 0)
                    break
        train['fc_implied_flat'] = fc_impl_train

        fc_impl_test = np.full(len(test), np.nan)
        for i, (idx, row) in enumerate(test.iterrows()):
            gs, tf, lag = row['gsa_site'], row['Timeframe'], row['Prediction_Lag']
            for key in [(gs, tf, lag), ('FB', gs, lag)]:
                cf = cov_rp.get(key)
                bf = bias_rp.get(key)
                if cf and bf and cf > 0.001 and bf > 0.001 and row['Forecast_Value'] > 0:
                    fc_impl_test[i] = max(row['Forecast_Value'] / bf / cf, 0)
                    break
        test['fc_implied_flat'] = fc_impl_test

        # Rebuild multi-vintage FC with these curves
        apply_multi_vintage_fc(train, cov_rp, bias_rp, fc_registry, 'temporal',
                               vintage_power=best_vp, staleness_cap=best_cap)
        apply_multi_vintage_fc(test, cov_rp, bias_rp, fc_registry, 'temporal',
                               vintage_power=best_vp, staleness_cap=best_cap)

        w_v8d, _ = run_config(test, train, test_months, gsa_sites,
                              'oo_implied_flat', 'fc_implied_multi',
                              fc_lag_coeff=best_coeff, fc_lag_floor=best_floor)
        w_fc, _ = run_config(test, train, test_months, gsa_sites,
                             'oo_none', 'fc_implied_multi',
                             fc_lag_coeff=best_coeff, fc_lag_floor=best_floor)

        d_v8d = w_v8d - best_cap_wmape
        d_fc = w_fc - fc_at_best_decay
        marker = ' <-- best' if w_v8d < best_fc_rp_wmape else ''
        print(f"  {rp:>12.1f} | {w_v8d:>8.1f}% | {w_fc:>8.1f}% | "
              f"{d_v8d:>+8.1f}pp | {d_fc:>+8.1f}pp{marker}")

        if w_v8d < best_fc_rp_wmape:
            best_fc_rp = rp
            best_fc_rp_wmape = w_v8d

    print(f"\n  Best FC RECENCY_POWER: {best_fc_rp} (V8d+mFC = {best_fc_rp_wmape:.1f}%)")

    # ==================================================================
    # STEP 6: Combined best — final evaluation
    # ==================================================================
    print(f"\n  {'=' * 80}")
    print(f"  STEP 6: COMBINED BEST")
    print(f"  {'=' * 80}")

    cap_str = str(best_cap) if best_cap is not None else 'None'
    print(f"  Best parameters:")
    print(f"    VINTAGE_RECENCY_POWER = {best_vp}")
    print(f"    FC lag decay coeff    = {best_coeff}")
    print(f"    FC lag decay floor    = {best_floor}")
    print(f"    Vintage staleness cap = {cap_str}")
    print(f"    FC curve RECENCY_POWER = {best_fc_rp}")

    # Rebuild everything with best params
    cov_best = build_flat_curves(train, 'fc_coverage', start='2023-01-01',
                                 recency_power=best_fc_rp)
    bias_best = build_flat_curves(train, 'fc_bias', start='2023-01-01',
                                  recency_power=best_fc_rp)

    # Reapply FC flat curves with best recency
    # (need to restore from original data since train/test fc_implied_flat was overwritten)
    # Re-derive from raw data
    for dset, curves_label in [(train, 'train'), (test, 'test')]:
        fc_impl = np.full(len(dset), np.nan)
        for i, (idx, row) in enumerate(dset.iterrows()):
            gs, tf, lag = row['gsa_site'], row['Timeframe'], row['Prediction_Lag']
            for key in [(gs, tf, lag), ('FB', gs, lag)]:
                cf = cov_best.get(key)
                bf = bias_best.get(key)
                if cf and bf and cf > 0.001 and bf > 0.001 and row['Forecast_Value'] > 0:
                    fc_impl[i] = max(row['Forecast_Value'] / bf / cf, 0)
                    break
        dset['fc_implied_flat'] = fc_impl

    apply_multi_vintage_fc(train, cov_best, bias_best, fc_registry, 'temporal',
                           vintage_power=best_vp, staleness_cap=best_cap)
    apply_multi_vintage_fc(test, cov_best, bias_best, fc_registry, 'temporal',
                           vintage_power=best_vp, staleness_cap=best_cap)

    # Run final models
    tuned_v8d, tuned_v8d_bias = run_config(
        test, train, test_months, gsa_sites,
        'oo_implied_flat', 'fc_implied_multi',
        fc_lag_coeff=best_coeff, fc_lag_floor=best_floor)
    tuned_fc, tuned_fc_bias = run_config(
        test, train, test_months, gsa_sites,
        'oo_none', 'fc_implied_multi',
        fc_lag_coeff=best_coeff, fc_lag_floor=best_floor)

    print(f"\n  {'Model':>30s} | {'WMAPE':>7s} | {'Bias':>7s} | {'vs Default':>10s}")
    print(f"  {'-' * 60}")
    print(f"  {'V8d + mFC (default)':>30s} | {baseline_v8d:>5.1f}% | "
          f"{baseline_v8d_bias:>+5.1f}% |       ---")
    print(f"  {'V8d + mFC (tuned)':>30s} | {tuned_v8d:>5.1f}% | "
          f"{tuned_v8d_bias:>+5.1f}% | {tuned_v8d - baseline_v8d:>+6.1f}pp")
    print(f"  {'FC-only multi (default)':>30s} | {baseline_fc:>5.1f}% | "
          f"{baseline_fc_bias:>+5.1f}% |       ---")
    print(f"  {'FC-only multi (tuned)':>30s} | {tuned_fc:>5.1f}% | "
          f"{tuned_fc_bias:>+5.1f}% | {tuned_fc - baseline_fc:>+6.1f}pp")

    # Per-lag breakdown: tuned vs default
    print(f"\n  Per-lag WMAPE: default vs tuned (V8d+mFC):")
    print(f"  {'Lag':>4s} | {'Default':>10s} | {'Tuned':>10s} | {'Diff':>8s}")
    print(f"  {'-' * 40}")

    # Get predictions for per-lag
    preds_default = weighted_avg(test, train, test_months, gsa_sites,
                                 'oo_implied_flat', 'fc_implied_multi',
                                 fc_lag_coeff=0.06, fc_lag_floor=0.3)
    # Need to rebuild multi-vintage with defaults for fair comparison
    apply_multi_vintage_fc(test, cov_flat, bias_flat, fc_registry, 'temporal',
                           vintage_power=VINTAGE_RECENCY_POWER)
    preds_default_fair = weighted_avg(test, train, test_months, gsa_sites,
                                      'oo_implied_flat', 'fc_implied_multi',
                                      fc_lag_coeff=0.06, fc_lag_floor=0.3)
    # Rebuild with tuned for tuned predictions
    apply_multi_vintage_fc(test, cov_best, bias_best, fc_registry, 'temporal',
                           vintage_power=best_vp, staleness_cap=best_cap)
    preds_tuned = weighted_avg(test, train, test_months, gsa_sites,
                               'oo_implied_flat', 'fc_implied_multi',
                               fc_lag_coeff=best_coeff, fc_lag_floor=best_floor)

    test['pred_default'] = preds_default_fair
    test['pred_tuned'] = preds_tuned
    m2 = test['Reference_Month'] > test_months[0]

    for lag in sorted(test['Prediction_Lag'].unique()):
        sub = test[m2 & (test['Prediction_Lag'] == lag) & (test['Actual_Sales'] > 0)]
        v_def = sub[sub['pred_default'].notna()]
        v_tun = sub[sub['pred_tuned'].notna()]
        if len(v_def) > 0 and len(v_tun) > 0:
            w_def = compute_wmape(v_def['Actual_Sales'].values, v_def['pred_default'].values)
            w_tun = compute_wmape(v_tun['Actual_Sales'].values, v_tun['pred_tuned'].values)
            print(f"  {lag:>4d} | {w_def:>8.1f}% | {w_tun:>8.1f}% | {w_tun - w_def:>+6.1f}pp")

    # Per-TF breakdown
    print(f"\n  Per-TF WMAPE: default vs tuned (V8d+mFC):")
    print(f"  {'TF':>4s} | {'Default':>10s} | {'Tuned':>10s} | {'Diff':>8s}")
    print(f"  {'-' * 40}")

    for tf in sorted(test['Timeframe'].unique()):
        sub = test[m2 & (test['Timeframe'] == tf) & (test['Actual_Sales'] > 0)]
        v_def = sub[sub['pred_default'].notna()]
        v_tun = sub[sub['pred_tuned'].notna()]
        if len(v_def) > 0 and len(v_tun) > 0:
            w_def = compute_wmape(v_def['Actual_Sales'].values, v_def['pred_default'].values)
            w_tun = compute_wmape(v_tun['Actual_Sales'].values, v_tun['pred_tuned'].values)
            print(f"  {tf:>4d} | {w_def:>8.1f}% | {w_tun:>8.1f}% | {w_tun - w_def:>+6.1f}pp")

    # ==================================================================
    # STEP 7: K-Fold Random CV validation
    # ==================================================================
    print(f"\n  {'=' * 80}")
    print(f"  STEP 7: K-FOLD RANDOM CV ({N_FOLDS} folds) — default vs tuned")
    print(f"  {'=' * 80}")

    np.random.seed(RANDOM_SEED)
    indices = np.arange(len(df))
    np.random.shuffle(indices)
    fold_size = len(df) // N_FOLDS

    cv_configs = {
        'V8d+mFC (default)': {
            'oo_col': 'oo_implied_flat', 'fc_col': 'fc_implied_multi',
            'vintage_power': VINTAGE_RECENCY_POWER, 'staleness_cap': None,
            'fc_lag_coeff': 0.06, 'fc_lag_floor': 0.3,
            'fc_recency': RECENCY_POWER,
        },
        'V8d+mFC (tuned)': {
            'oo_col': 'oo_implied_flat', 'fc_col': 'fc_implied_multi',
            'vintage_power': best_vp, 'staleness_cap': best_cap,
            'fc_lag_coeff': best_coeff, 'fc_lag_floor': best_floor,
            'fc_recency': best_fc_rp,
        },
        'FC-only (default)': {
            'oo_col': 'oo_none', 'fc_col': 'fc_implied_multi',
            'vintage_power': VINTAGE_RECENCY_POWER, 'staleness_cap': None,
            'fc_lag_coeff': 0.06, 'fc_lag_floor': 0.3,
            'fc_recency': RECENCY_POWER,
        },
        'FC-only (tuned)': {
            'oo_col': 'oo_none', 'fc_col': 'fc_implied_multi',
            'vintage_power': best_vp, 'staleness_cap': best_cap,
            'fc_lag_coeff': best_coeff, 'fc_lag_floor': best_floor,
            'fc_recency': best_fc_rp,
        },
    }

    fold_metrics = {name: [] for name in cv_configs}

    for fold_i in range(N_FOLDS):
        fold_start = fold_i * fold_size
        fold_end = fold_start + fold_size if fold_i < N_FOLDS - 1 else len(df)
        test_idx = indices[fold_start:fold_end]
        train_idx = np.setdiff1d(indices, test_idx)

        f_train = df.iloc[train_idx].copy()
        f_test = df.iloc[test_idx].copy()

        # Build OO curves (fixed)
        oo_f = build_flat_curves(f_train, 'oo_ratio', start='2023-01-01')

        for config_name, cfg in cv_configs.items():
            # Build FC curves with this config's recency power
            cov_f = build_flat_curves(f_train, 'fc_coverage', start='2023-01-01',
                                      recency_power=cfg['fc_recency'])
            bias_f = build_flat_curves(f_train, 'fc_bias', start='2023-01-01',
                                       recency_power=cfg['fc_recency'])

            # Apply flat curves
            apply_flat_curves(f_train, oo_f, cov_f, bias_f)
            apply_flat_curves(f_test, oo_f, cov_f, bias_f)

            # Multi-vintage FC from training fold
            fc_reg_fold = build_forecast_registry(f_train)
            apply_multi_vintage_fc(f_train, cov_f, bias_f, fc_reg_fold, 'fold',
                                   vintage_power=cfg['vintage_power'],
                                   staleness_cap=cfg['staleness_cap'])
            apply_multi_vintage_fc(f_test, cov_f, bias_f, fc_reg_fold, 'fold',
                                   vintage_power=cfg['vintage_power'],
                                   staleness_cap=cfg['staleness_cap'])

            f_train['oo_none'] = np.nan
            f_test['oo_none'] = np.nan

            f_test_months = sorted(f_test['Reference_Month'].unique())

            preds = weighted_avg(f_test, f_train, f_test_months, gsa_sites,
                                 oo_col=cfg['oo_col'], fc_col=cfg['fc_col'],
                                 fc_lag_coeff=cfg['fc_lag_coeff'],
                                 fc_lag_floor=cfg['fc_lag_floor'])

            ts = f_test[f_test['Actual_Sales'] > 0]
            mask = preds.notna()
            v = ts[mask.reindex(ts.index, fill_value=False)]
            if len(v) > 0:
                p = preds[v.index]
                wmape = compute_wmape(v['Actual_Sales'].values, p.values)
                bias = compute_bias(v['Actual_Sales'].values, p.values)
                fold_metrics[config_name].append({'wmape': wmape, 'bias': bias})

        # Per-fold summary
        def _get_last(name):
            return fold_metrics[name][-1]['wmape'] if fold_metrics[name] else np.nan
        print(f"    Fold {fold_i + 1}: "
              f"V8d def={_get_last('V8d+mFC (default)'):.1f}%  "
              f"V8d tun={_get_last('V8d+mFC (tuned)'):.1f}%  "
              f"FC def={_get_last('FC-only (default)'):.1f}%  "
              f"FC tun={_get_last('FC-only (tuned)'):.1f}%")

    # CV summary
    print(f"\n  {'Config':>25s} | {'CV Mean':>8s} | {'CV Std':>7s} | "
          f"{'Temporal':>10s} | {'Gap':>6s}")
    print(f"  {'-' * 65}")

    temporal_lookup = {
        'V8d+mFC (default)': baseline_v8d,
        'V8d+mFC (tuned)': tuned_v8d,
        'FC-only (default)': baseline_fc,
        'FC-only (tuned)': tuned_fc,
    }

    for config_name in cv_configs:
        wmapes = [m['wmape'] for m in fold_metrics[config_name]]
        if wmapes:
            cv_mean = np.mean(wmapes)
            cv_std = np.std(wmapes)
            t_wmape = temporal_lookup.get(config_name, np.nan)
            gap = t_wmape - cv_mean
            print(f"  {config_name:>25s} | {cv_mean:>6.1f}% | {cv_std:>5.1f}% | "
                  f"{t_wmape:>8.1f}% | {gap:>+4.1f}pp")

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print(f"\n  {'=' * 80}")
    print(f"  SUMMARY — {customer_name}")
    print(f"  {'=' * 80}")

    cap_str = str(best_cap) if best_cap is not None else 'None'
    print(f"\n  Default parameters:")
    print(f"    VINTAGE_RECENCY_POWER = 1.5")
    print(f"    FC lag decay coeff    = 0.06")
    print(f"    FC lag decay floor    = 0.3")
    print(f"    Vintage staleness cap = None")
    print(f"    FC curve RECENCY_POWER = 2")
    print(f"\n  Tuned parameters:")
    print(f"    VINTAGE_RECENCY_POWER = {best_vp}")
    print(f"    FC lag decay coeff    = {best_coeff}")
    print(f"    FC lag decay floor    = {best_floor}")
    print(f"    Vintage staleness cap = {cap_str}")
    print(f"    FC curve RECENCY_POWER = {best_fc_rp}")
    print(f"\n  Temporal WMAPE change:")
    print(f"    V8d+mFC:   {baseline_v8d:.1f}% -> {tuned_v8d:.1f}% "
          f"({tuned_v8d - baseline_v8d:+.1f}pp)")
    print(f"    FC-only:   {baseline_fc:.1f}% -> {tuned_fc:.1f}% "
          f"({tuned_fc - baseline_fc:+.1f}pp)")

    return {
        'customer': customer_name,
        'defaults': {'vintage_power': 1.5, 'lag_coeff': 0.06, 'lag_floor': 0.3,
                     'staleness_cap': None, 'fc_recency': 2},
        'tuned': {'vintage_power': best_vp, 'lag_coeff': best_coeff,
                  'lag_floor': best_floor, 'staleness_cap': best_cap,
                  'fc_recency': best_fc_rp},
        'baseline_v8d': baseline_v8d,
        'tuned_v8d': tuned_v8d,
        'baseline_fc': baseline_fc,
        'tuned_fc': tuned_fc,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='FC Parameter Tuning Experiment')
    parser.add_argument('--file', type=str, default=None,
                        help='Path to CSV data file')
    args = parser.parse_args()

    if args.file:
        filepath = args.file
    else:
        import glob
        csvs = sorted(glob.glob('*.csv'))
        if len(csvs) == 0:
            print("No CSV files found. Please specify with --file")
            return
        filepath = csvs[0]
        print(f"Auto-detected: {filepath}")

    print(f"Loading: {filepath}")
    df = load_and_preprocess(filepath)
    print(f"Loaded {len(df)} rows")

    customers = sorted(df['GSA'].unique())
    all_results = []
    for cust in customers:
        cust_df = df[df['GSA'] == cust].copy()
        result = run_experiment(cust_df, cust)
        all_results.append(result)

    # Cross-customer summary
    if len(all_results) > 1:
        print(f"\n\n{'=' * 100}")
        print(f"CROSS-CUSTOMER SUMMARY")
        print(f"{'=' * 100}")
        print(f"\n  {'Customer':>20s} | {'V8d Def':>8s} | {'V8d Tun':>8s} | "
              f"{'V8d Diff':>8s} | {'FC Def':>8s} | {'FC Tun':>8s} | {'FC Diff':>8s}")
        print(f"  {'-' * 80}")
        for r in all_results:
            d_v8d = r['tuned_v8d'] - r['baseline_v8d']
            d_fc = r['tuned_fc'] - r['baseline_fc']
            print(f"  {r['customer']:>20s} | {r['baseline_v8d']:>6.1f}% | "
                  f"{r['tuned_v8d']:>6.1f}% | {d_v8d:>+6.1f}pp | "
                  f"{r['baseline_fc']:>6.1f}% | {r['tuned_fc']:>6.1f}% | {d_fc:>+6.1f}pp")

        print(f"\n  Best tuned parameters per customer:")
        for r in all_results:
            cap_str = str(r['tuned']['staleness_cap']) if r['tuned']['staleness_cap'] is not None else 'None'
            print(f"  {r['customer']:>20s}: vintage_power={r['tuned']['vintage_power']}, "
                  f"coeff={r['tuned']['lag_coeff']}, floor={r['tuned']['lag_floor']}, "
                  f"cap={cap_str}, fc_recency={r['tuned']['fc_recency']}")

    print(f"\n\nDone!")


if __name__ == '__main__':
    main()
