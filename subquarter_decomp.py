"""
Sub-Quarter Decomposition Experiment
======================================
Tests whether predicting TF=6/9/12 targets by decomposing them into TF=3
sub-quarters and summing the sub-quarter predictions improves accuracy.

Core idea:
    A TF=12 target can be decomposed into four TF=3 sub-quarters.
    Each sub-quarter exists as its own TF=3 row at the same Reference_Month
    but at effective_lag = parent_lag + 3*k (k=0,1,...).

    predicted_TF12 = sum(predicted_TF3_Qk for k in range(4))

    TF=3 curves may be better calibrated, and the decomposition naturally
    captures front-loaded vs back-loaded ordering patterns.

Data verification confirms perfect additivity:
    TF=12 Actual_Sales == sum of four TF=3 Actual_Sales
    TF=12 Open_Orders == sum of four TF=3 Open_Orders

Usage:
    python subquarter_decomp.py
    python subquarter_decomp.py --file "Dummy Training Data Customer 1 Try 2 11-Feb-2026.csv"
"""

import pandas as pd
import numpy as np
import warnings
import argparse
warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTS
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
# CURVE BUILDING (from pipeline.py)
# ============================================================

def build_flat_curves(train_df, ratio_col, start=None):
    td = train_df if start is None else train_df[train_df['Reference_Month'] >= start]
    flat = {}

    for (gs, tf, lag), grp in td.groupby(['gsa_site', 'Timeframe', 'Prediction_Lag']):
        valid = grp[grp[ratio_col].notna() & (grp[ratio_col] > 0) & (grp[ratio_col] < 10)]
        if len(valid) < MIN_OBS_FLAT:
            continue
        days = (valid['Reference_Month'] - valid['Reference_Month'].min()).dt.days / 365
        wts = (1 + days.values) ** RECENCY_POWER
        flat[(gs, tf, lag)] = np.average(valid[ratio_col], weights=wts)

    for (gs, lag), grp in td.groupby(['gsa_site', 'Prediction_Lag']):
        valid = grp[grp[ratio_col].notna() & (grp[ratio_col] > 0) & (grp[ratio_col] < 10)]
        if len(valid) >= MIN_OBS_FLAT:
            days = (valid['Reference_Month'] - valid['Reference_Month'].min()).dt.days / 365
            wts = (1 + days.values) ** RECENCY_POWER
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
# MULTI-VINTAGE FC (from pipeline.py)
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


def _combine_vintage_estimates(estimates_with_lags):
    if not estimates_with_lags:
        return np.nan
    total_w, total_val = 0.0, 0.0
    for est, orig_lag in estimates_with_lags:
        w = 1.0 / (orig_lag ** VINTAGE_RECENCY_POWER)
        total_w += w
        total_val += w * est
    return total_val / total_w if total_w > 0 else np.nan


def apply_multi_vintage_fc(dset, cov_f, bias_f, registry, causality='temporal'):
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
            est, ok = _vintage_to_estimate(v, cov_f, bias_f)
            if ok:
                valid_vintages.append((max(est, 0), v['lag']))
        if valid_vintages:
            fc_impl_multi[i] = max(_combine_vintage_estimates(valid_vintages), 0)
            n_vintages[i] = len(valid_vintages)
            best_lag[i] = min(vl for _, vl in valid_vintages)

    dset['fc_implied_multi'] = fc_impl_multi
    dset['n_vintages_used'] = n_vintages
    dset['best_vintage_lag'] = best_lag


# ============================================================
# WEIGHTED AVERAGE BLENDING (from pipeline.py)
# ============================================================

def weighted_avg(test_df, train_df, test_months, gsa_sites_list,
                 oo_col='oo_implied_flat', fc_col='fc_implied_flat'):
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

                    oo_f = max(0.3, 1.0 - 0.06 * (lag - 1)) if has_oo else 0

                    fc_lag = lag
                    if (has_fc and fc_col == 'fc_implied_multi' and
                            'best_vintage_lag' in results.columns):
                        bvl = results.at[idx, 'best_vintage_lag']
                        if not np.isnan(bvl):
                            fc_lag = bvl
                    fc_f = max(0.3, 1.0 - 0.06 * (fc_lag - 1)) if has_fc else 0

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
# SUB-QUARTER DECOMPOSITION — NEW FUNCTIONS
# ============================================================

def build_tf3_lookup(df):
    """Build O(1) lookup: (gsa_site, Reference_Month, TP_Start, TP_End) -> row data.

    Only includes TF=3 rows.
    """
    tf3 = df[df['Timeframe'] == 3]
    lookup = {}
    for _, row in tf3.iterrows():
        key = (row['gsa_site'], row['Reference_Month'],
               row['Target_Period_Start'], row['Target_Period_End'])
        lookup[key] = {
            'oo': row['Open_Orders'],
            'actual': row['Actual_Sales'],
            'fc_value': row['Forecast_Value'],
            'covered_orders': row['Covered_Orders'],
            'has_fc': row['Has_Forecast'] == 1,
            'oo_ratio': row.get('oo_ratio', np.nan),
            'fc_coverage': row.get('fc_coverage', np.nan),
            'fc_bias': row.get('fc_bias', np.nan),
        }
    return lookup


def verify_additivity(df):
    """Verify that TF>3 Actual_Sales and Open_Orders equal sum of TF=3 sub-quarters."""
    tf3_lookup = build_tf3_lookup(df)

    print(f"\n  {'TF':>4s} | {'Total':>6s} | {'Full match':>10s} | "
          f"{'Partial':>8s} | {'No match':>8s} | "
          f"{'AS exact':>8s} | {'OO exact':>8s} | {'FC exact':>8s}")
    print(f"  {'-' * 80}")

    for parent_tf in [6, 9, 12]:
        n_sq = parent_tf // 3
        parent_rows = df[df['Timeframe'] == parent_tf]
        n_total = 0
        n_full = 0
        n_partial = 0
        n_none = 0
        n_as_exact = 0
        n_oo_exact = 0
        n_fc_exact = 0

        for _, row in parent_rows.iterrows():
            n_total += 1
            found = 0
            as_sum = 0.0
            oo_sum = 0.0
            fc_sum = 0.0
            all_found = True

            for k in range(n_sq):
                sq_start = row['Target_Period_Start'] + pd.DateOffset(months=3 * k)
                sq_end = row['Target_Period_Start'] + pd.DateOffset(months=3 * (k + 1)) \
                         - pd.Timedelta(days=1)
                key = (row['gsa_site'], row['Reference_Month'], sq_start, sq_end)
                if key in tf3_lookup:
                    found += 1
                    sq = tf3_lookup[key]
                    as_sum += sq['actual'] if not np.isnan(sq['actual']) else 0
                    oo_sum += sq['oo'] if not np.isnan(sq['oo']) else 0
                    fc_sum += sq['fc_value'] if not np.isnan(sq['fc_value']) else 0
                else:
                    all_found = False

            if found == n_sq:
                n_full += 1
                if abs(as_sum - row['Actual_Sales']) < 0.01 * max(abs(row['Actual_Sales']), 1):
                    n_as_exact += 1
                if abs(oo_sum - row['Open_Orders']) < 0.01 * max(abs(row['Open_Orders']), 1):
                    n_oo_exact += 1
                fc_parent = row['Forecast_Value'] if not np.isnan(row['Forecast_Value']) else 0
                if abs(fc_sum - fc_parent) < 0.01 * max(abs(fc_parent), 1):
                    n_fc_exact += 1
            elif found > 0:
                n_partial += 1
            else:
                n_none += 1

        pct_full = 100 * n_full / n_total if n_total > 0 else 0
        pct_as = 100 * n_as_exact / n_full if n_full > 0 else 0
        pct_oo = 100 * n_oo_exact / n_full if n_full > 0 else 0
        pct_fc = 100 * n_fc_exact / n_full if n_full > 0 else 0
        print(f"  {parent_tf:>4d} | {n_total:>6d} | {n_full:>5d} ({pct_full:>3.0f}%) | "
              f"{n_partial:>8d} | {n_none:>8d} | "
              f"{pct_as:>6.0f}% | {pct_oo:>6.0f}% | {pct_fc:>6.0f}%")


def build_subquarter_map(df, tf3_lookup):
    """For each TF>3 row, find its TF=3 sub-quarter data.

    Returns: {df_index: [{'sq_index': k, 'eff_lag': int,
                           'oo': float, 'fc_value': float,
                           'has_fc': bool, 'actual': float}, ...]}
    """
    sq_map = {}
    parent = df[df['Timeframe'] > 3]

    for idx, row in parent.iterrows():
        n_sq = int(row['Timeframe']) // 3
        subs = []
        for k in range(n_sq):
            eff_lag = int(row['Prediction_Lag']) + 3 * k
            sq_start = row['Target_Period_Start'] + pd.DateOffset(months=3 * k)
            sq_end = row['Target_Period_Start'] + pd.DateOffset(months=3 * (k + 1)) \
                     - pd.Timedelta(days=1)
            key = (row['gsa_site'], row['Reference_Month'], sq_start, sq_end)

            if key in tf3_lookup:
                sq = tf3_lookup[key]
                subs.append({
                    'sq_index': k,
                    'eff_lag': eff_lag,
                    'oo': sq['oo'],
                    'fc_value': sq['fc_value'],
                    'has_fc': sq['has_fc'],
                    'actual': sq['actual'],
                })
            else:
                subs.append({
                    'sq_index': k,
                    'eff_lag': eff_lag,
                    'oo': np.nan,
                    'fc_value': np.nan,
                    'has_fc': False,
                    'actual': np.nan,
                })

        sq_map[idx] = subs

    return sq_map


def compute_historical_shares(train_df, tf3_lookup):
    """Compute average share each sub-quarter contributes to the parent total.

    Returns: {(gsa_site, parent_tf, sq_index): share}
    """
    share_data = {}  # {(gs, parent_tf, sq_idx): [share_values]}

    parent = train_df[train_df['Timeframe'] > 3]
    for _, row in parent.iterrows():
        if row['Actual_Sales'] <= 0 or np.isnan(row['Actual_Sales']):
            continue
        n_sq = int(row['Timeframe']) // 3
        gs = row['gsa_site']
        parent_tf = int(row['Timeframe'])

        for k in range(n_sq):
            sq_start = row['Target_Period_Start'] + pd.DateOffset(months=3 * k)
            sq_end = row['Target_Period_Start'] + pd.DateOffset(months=3 * (k + 1)) \
                     - pd.Timedelta(days=1)
            key = (gs, row['Reference_Month'], sq_start, sq_end)
            if key in tf3_lookup:
                sq = tf3_lookup[key]
                if not np.isnan(sq['actual']) and sq['actual'] > 0:
                    share = sq['actual'] / row['Actual_Sales']
                    skey = (gs, parent_tf, k)
                    if skey not in share_data:
                        share_data[skey] = []
                    share_data[skey].append(share)

    shares = {}
    for skey, vals in share_data.items():
        shares[skey] = np.mean(vals)

    return shares


def apply_bottom_up(dset, sq_map, oo_curves, cov_curves, bias_curves,
                    shares=None, mode='all_or_nothing'):
    """Compute bottom-up OO and FC predictions for TF>3 rows.

    For TF=3 rows, copies oo_implied_flat and fc_implied_flat unchanged.

    mode='all_or_nothing': only predict when ALL sub-quarters have data + curves
    mode='shares': use historical shares to scale when sub-quarters are missing

    Adds columns: oo_implied_bu, fc_implied_bu, n_sq_available, n_sq_total
    """
    n = len(dset)
    oo_bu = np.full(n, np.nan)
    fc_bu = np.full(n, np.nan)
    n_avail = np.zeros(n, dtype=int)
    n_total = np.zeros(n, dtype=int)

    for i, (idx, row) in enumerate(dset.iterrows()):
        gs = row['gsa_site']
        tf = int(row['Timeframe'])

        if tf == 3:
            # TF=3: use standard flat curves
            oo_bu[i] = row.get('oo_implied_flat', np.nan)
            fc_bu[i] = row.get('fc_implied_flat', np.nan)
            n_avail[i] = 1
            n_total[i] = 1
            continue

        if idx not in sq_map:
            continue

        subs = sq_map[idx]
        n_sq = len(subs)
        n_total[i] = n_sq

        # OO bottom-up
        oo_estimates = []
        oo_available_share = 0.0
        for sq in subs:
            oo_val = sq['oo']
            eff_lag = sq['eff_lag']
            if np.isnan(oo_val) or oo_val < 0:
                continue
            # Look up TF=3 OO curve at effective lag
            curve_val = None
            for key in [(gs, 3, eff_lag), ('FB', gs, eff_lag)]:
                if key in oo_curves and oo_curves[key] > 0.001:
                    curve_val = oo_curves[key]
                    break
            if curve_val is not None:
                implied = max(oo_val / curve_val, 0)
                oo_estimates.append((sq['sq_index'], implied))
                if shares:
                    sh = shares.get((gs, tf, sq['sq_index']), 1.0 / n_sq)
                    oo_available_share += sh

        n_avail[i] = len(oo_estimates)

        if mode == 'all_or_nothing':
            if len(oo_estimates) == n_sq:
                oo_bu[i] = sum(est for _, est in oo_estimates)
        elif mode == 'shares' and shares:
            if len(oo_estimates) > 0:
                raw_sum = sum(est for _, est in oo_estimates)
                if oo_available_share > 0.01:
                    oo_bu[i] = raw_sum / oo_available_share
                else:
                    oo_bu[i] = raw_sum * (n_sq / len(oo_estimates))

        # FC bottom-up
        fc_estimates = []
        fc_available_share = 0.0
        for sq in subs:
            fc_val = sq['fc_value']
            eff_lag = sq['eff_lag']
            if not sq['has_fc'] or np.isnan(fc_val) or fc_val <= 0:
                continue
            # Look up TF=3 FC curves at effective lag
            cf = None
            bf = None
            for key in [(gs, 3, eff_lag), ('FB', gs, eff_lag)]:
                c = cov_curves.get(key)
                b = bias_curves.get(key)
                if c and b and c > 0.001 and b > 0.001:
                    cf, bf = c, b
                    break
            if cf is not None:
                implied = max(fc_val / bf / cf, 0)
                fc_estimates.append((sq['sq_index'], implied))
                if shares:
                    sh = shares.get((gs, tf, sq['sq_index']), 1.0 / n_sq)
                    fc_available_share += sh

        if mode == 'all_or_nothing':
            if len(fc_estimates) == n_sq:
                fc_bu[i] = sum(est for _, est in fc_estimates)
        elif mode == 'shares' and shares:
            if len(fc_estimates) > 0:
                raw_sum = sum(est for _, est in fc_estimates)
                if fc_available_share > 0.01:
                    fc_bu[i] = raw_sum / fc_available_share
                else:
                    fc_bu[i] = raw_sum * (n_sq / len(fc_estimates))

    dset['oo_implied_bu'] = oo_bu
    dset['fc_implied_bu'] = fc_bu
    dset['n_sq_available'] = n_avail
    dset['n_sq_total'] = n_total


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment(df, customer_name):
    print(f"\n{'#' * 100}")
    print(f"# SUB-QUARTER DECOMPOSITION EXPERIMENT — {customer_name}")
    print(f"{'#' * 100}")

    # ---- Filter sites ----
    site_counts = df.groupby('gsa_site').size()
    valid_sites = sorted(site_counts[site_counts >= MIN_SITE_ROWS].index.tolist())
    df = df[df['gsa_site'].isin(valid_sites)].copy()
    gsa_sites = sorted(df['gsa_site'].unique())

    print(f"\n  Data: {len(df)} rows, {len(gsa_sites)} sites")
    print(f"  Sites: {gsa_sites}")
    print(f"  TF distribution: {dict(df['Timeframe'].value_counts().sort_index())}")

    # ============================================================
    # STEP 0: Verify additivity
    # ============================================================
    print(f"\n  {'=' * 80}")
    print(f"  ADDITIVITY VERIFICATION")
    print(f"  {'=' * 80}")

    tf3_lookup = build_tf3_lookup(df)
    verify_additivity(df)

    # ============================================================
    # STEP 1: Train/test split
    # ============================================================
    cutoff = df['Reference_Month'].quantile(0.8)
    train = df[df['Reference_Month'] <= cutoff].copy()
    test = df[df['Reference_Month'] > cutoff].copy()
    test_months = sorted(test['Reference_Month'].unique())

    print(f"\n  Train: {len(train)} rows "
          f"({train['Reference_Month'].min().date()} – "
          f"{train['Reference_Month'].max().date()})")
    print(f"  Test:  {len(test)} rows "
          f"({test['Reference_Month'].min().date()} – "
          f"{test['Reference_Month'].max().date()})")

    # ============================================================
    # STEP 2: Build curves
    # ============================================================
    print(f"\n  {'=' * 80}")
    print(f"  BUILDING CURVES")
    print(f"  {'=' * 80}")

    # All-TF flat curves (V8d baseline)
    oo_flat = build_flat_curves(train, 'oo_ratio', start='2023-01-01')
    cov_flat = build_flat_curves(train, 'fc_coverage', start='2023-01-01')
    bias_flat = build_flat_curves(train, 'fc_bias', start='2023-01-01')

    # TF=3-only curves for bottom-up
    tf3_train = train[train['Timeframe'] == 3]
    oo_tf3 = build_flat_curves(tf3_train, 'oo_ratio', start='2023-01-01')
    cov_tf3 = build_flat_curves(tf3_train, 'fc_coverage', start='2023-01-01')
    bias_tf3 = build_flat_curves(tf3_train, 'fc_bias', start='2023-01-01')

    print(f"  All-TF flat OO curves: {len(oo_flat)}")
    print(f"  TF=3 OO curves: {len(oo_tf3)}")
    print(f"  TF=3 FC cov curves: {len(cov_tf3)}  |  FC bias: {len(bias_tf3)}")

    # Check TF=3 curve coverage at various lags
    print(f"\n  TF=3 OO curve availability by lag:")
    for lag in range(1, 16):
        n_sites = sum(1 for gs in gsa_sites
                      if (gs, 3, lag) in oo_tf3 or ('FB', gs, lag) in oo_tf3)
        if n_sites > 0:
            print(f"    Lag {lag:>2d}: {n_sites}/{len(gsa_sites)} sites")

    # ============================================================
    # STEP 3: Apply standard curves
    # ============================================================
    print(f"\n  {'=' * 80}")
    print(f"  APPLYING CURVES")
    print(f"  {'=' * 80}")

    apply_flat_curves(train, oo_flat, cov_flat, bias_flat)
    apply_flat_curves(test, oo_flat, cov_flat, bias_flat)

    # Multi-vintage FC
    fc_registry = build_forecast_registry(df)
    apply_multi_vintage_fc(train, cov_flat, bias_flat, fc_registry, 'temporal')
    apply_multi_vintage_fc(test, cov_flat, bias_flat, fc_registry, 'temporal')
    n_multi = test['fc_implied_multi'].notna().sum()
    print(f"  Multi-vintage FC coverage: {n_multi}/{len(test)} "
          f"({100 * n_multi / len(test):.0f}%)")

    # ============================================================
    # STEP 4: Sub-quarter decomposition
    # ============================================================
    print(f"\n  {'=' * 80}")
    print(f"  SUB-QUARTER DECOMPOSITION")
    print(f"  {'=' * 80}")

    # Build lookup and map
    tf3_lookup_all = build_tf3_lookup(df)
    sq_map_train = build_subquarter_map(train, tf3_lookup_all)
    sq_map_test = build_subquarter_map(test, tf3_lookup_all)

    # Historical shares from training data
    shares = compute_historical_shares(train, tf3_lookup_all)

    # Print shares
    print(f"\n  Historical sub-quarter shares (training data):")
    for parent_tf in [6, 9, 12]:
        n_sq = parent_tf // 3
        print(f"\n  TF={parent_tf}:")
        header = f"    {'Site':>25s}"
        for k in range(n_sq):
            header += f" | {'Q' + str(k + 1):>6s}"
        print(header)
        print(f"    {'-' * (28 + 9 * n_sq)}")
        for gs in gsa_sites:
            line = f"    {gs.split('|')[1] if '|' in gs else gs:>25s}"
            for k in range(n_sq):
                sh = shares.get((gs, parent_tf, k), np.nan)
                if not np.isnan(sh):
                    line += f" | {sh:>5.1%}"
                else:
                    line += f" |   n/a"
            print(line)

    # Apply bottom-up: Strategy A (all-or-nothing)
    apply_bottom_up(train, sq_map_train, oo_tf3, cov_tf3, bias_tf3,
                    mode='all_or_nothing')
    apply_bottom_up(test, sq_map_test, oo_tf3, cov_tf3, bias_tf3,
                    mode='all_or_nothing')

    # Stats on coverage
    test_tfgt3 = test[test['Timeframe'] > 3]
    n_oo_bu = test_tfgt3['oo_implied_bu'].notna().sum()
    n_fc_bu = test_tfgt3['fc_implied_bu'].notna().sum()
    print(f"\n  Bottom-up coverage (all-or-nothing, TF>3 test rows):")
    print(f"    OO bottom-up: {n_oo_bu}/{len(test_tfgt3)} "
          f"({100 * n_oo_bu / len(test_tfgt3):.0f}%)")
    print(f"    FC bottom-up: {n_fc_bu}/{len(test_tfgt3)} "
          f"({100 * n_fc_bu / len(test_tfgt3):.0f}%)")

    # Per-TF coverage
    for parent_tf in [6, 9, 12]:
        sub = test_tfgt3[test_tfgt3['Timeframe'] == parent_tf]
        if len(sub) > 0:
            n_oo = sub['oo_implied_bu'].notna().sum()
            n_fc = sub['fc_implied_bu'].notna().sum()
            print(f"    TF={parent_tf}: OO={n_oo}/{len(sub)} ({100 * n_oo / len(sub):.0f}%), "
                  f"FC={n_fc}/{len(sub)} ({100 * n_fc / len(sub):.0f}%)")

    # Strategy B (shares): store as separate columns
    apply_bottom_up(train, sq_map_train, oo_tf3, cov_tf3, bias_tf3,
                    shares=shares, mode='shares')
    train.rename(columns={'oo_implied_bu': 'oo_implied_bu_shares',
                          'fc_implied_bu': 'fc_implied_bu_shares'}, inplace=True)
    # Re-run strategy A for train (was overwritten)
    apply_bottom_up(train, sq_map_train, oo_tf3, cov_tf3, bias_tf3,
                    mode='all_or_nothing')

    apply_bottom_up(test, sq_map_test, oo_tf3, cov_tf3, bias_tf3,
                    shares=shares, mode='shares')
    test.rename(columns={'oo_implied_bu': 'oo_implied_bu_shares',
                         'fc_implied_bu': 'fc_implied_bu_shares'}, inplace=True)
    # Re-run strategy A for test (was overwritten)
    apply_bottom_up(test, sq_map_test, oo_tf3, cov_tf3, bias_tf3,
                    mode='all_or_nothing')

    test_tfgt3_upd = test[test['Timeframe'] > 3]
    n_oo_sh = test_tfgt3_upd['oo_implied_bu_shares'].notna().sum()
    print(f"\n  Bottom-up coverage (shares mode, TF>3 test rows):")
    print(f"    OO bottom-up (shares): {n_oo_sh}/{len(test_tfgt3_upd)} "
          f"({100 * n_oo_sh / len(test_tfgt3_upd):.0f}%)")

    # oo_none column
    train['oo_none'] = np.nan
    test['oo_none'] = np.nan

    # ============================================================
    # STEP 5: Full model evaluation
    # ============================================================
    print(f"\n  {'=' * 80}")
    print(f"  FULL MODEL EVALUATION (blended OO + FC)")
    print(f"  {'=' * 80}")

    models = {
        'V8d (OO+FC)':       ('oo_implied_flat',       'fc_implied_flat'),
        'V8d + mFC':         ('oo_implied_flat',       'fc_implied_multi'),
        'BU-OO + flat FC':   ('oo_implied_bu',         'fc_implied_flat'),
        'BU-OO + mFC':       ('oo_implied_bu',         'fc_implied_multi'),
        'BU-OO(sh) + mFC':   ('oo_implied_bu_shares',  'fc_implied_multi'),
        'BU-all + mFC':      ('oo_implied_bu',         'fc_implied_bu'),
        'FC-only multi':     ('oo_none',               'fc_implied_multi'),
    }

    for model_name, (oo_c, fc_c) in models.items():
        test[model_name] = weighted_avg(test, train, test_months, gsa_sites,
                                        oo_col=oo_c, fc_col=fc_c)

    # Skip first test month
    m2 = test['Reference_Month'] > test_months[0]

    # Overall results
    print(f"\n  Overall (temporal, months 2+):")
    print(f"  {'Model':>22s} | {'WMAPE':>7s} | {'Bias':>7s} | {'N':>5s}")
    print(f"  {'-' * 50}")
    temporal_results = {}
    for model_name in models:
        ts = test[m2 & test[model_name].notna() & (test['Actual_Sales'] > 0)]
        if len(ts) == 0:
            continue
        w = compute_wmape(ts['Actual_Sales'].values, ts[model_name].values)
        b = compute_bias(ts['Actual_Sales'].values, ts[model_name].values)
        temporal_results[model_name] = w
        print(f"  {model_name:>22s} | {w:>5.1f}% | {b:>+5.1f}% | {len(ts):>5d}")

    # ---- Per-TF breakdown ----
    key_models = list(models.keys())
    print(f"\n  Per-TF WMAPE (temporal, months 2+):")
    header = f"  {'TF':>4s}"
    for mn in key_models:
        short = mn[:12]
        header += f" | {short:>12s}"
    print(header)
    print(f"  {'-' * (7 + 15 * len(key_models))}")
    for tf in sorted(test['Timeframe'].unique()):
        sub = test[m2 & (test['Timeframe'] == tf) & (test['Actual_Sales'] > 0)]
        line = f"  {tf:>4d}"
        for mn in key_models:
            v = sub[sub[mn].notna()]
            if len(v) > 0:
                w = compute_wmape(v['Actual_Sales'].values, v[mn].values)
                line += f" | {w:>10.1f}%"
            else:
                line += f" | {'n/a':>11s}"
        print(line)

    # ---- Per-site breakdown ----
    print(f"\n  Per-site WMAPE (temporal, months 2+):")
    key_short = ['V8d + mFC', 'BU-OO + mFC', 'BU-OO(sh) + mFC', 'FC-only multi']
    header = f"  {'Site':>25s}"
    for mn in key_short:
        short = mn[:14]
        header += f" | {short:>14s}"
    print(header)
    print(f"  {'-' * (28 + 17 * len(key_short))}")
    for gs in gsa_sites:
        sub = test[m2 & (test['gsa_site'] == gs) & (test['Actual_Sales'] > 0)]
        if len(sub) == 0:
            continue
        site = gs.split('|')[1] if '|' in gs else gs
        line = f"  {site:>25s}"
        for mn in key_short:
            v = sub[sub[mn].notna()]
            if len(v) > 0:
                w = compute_wmape(v['Actual_Sales'].values, v[mn].values)
                line += f" | {w:>12.1f}%"
            else:
                line += f" | {'n/a':>13s}"
        print(line)

    # ---- Per-lag breakdown ----
    print(f"\n  Per-lag WMAPE (temporal, months 2+):")
    header = f"  {'Lag':>4s}"
    for mn in key_short:
        short = mn[:14]
        header += f" | {short:>14s}"
    print(header)
    print(f"  {'-' * (7 + 17 * len(key_short))}")
    for lag in sorted(test['Prediction_Lag'].unique()):
        sub = test[m2 & (test['Prediction_Lag'] == lag) & (test['Actual_Sales'] > 0)]
        line = f"  {lag:>4d}"
        for mn in key_short:
            v = sub[sub[mn].notna()]
            if len(v) > 0:
                w = compute_wmape(v['Actual_Sales'].values, v[mn].values)
                line += f" | {w:>12.1f}%"
            else:
                line += f" | {'n/a':>13s}"
        print(line)

    # ---- Per (TF, lag) detail for BU-OO+mFC vs V8d+mFC ----
    print(f"\n  Per (TF, lag) WMAPE comparison: V8d+mFC vs BU-OO+mFC (temporal, months 2+):")
    print(f"  {'TF':>4s} | {'Lag':>4s} | {'V8d+mFC':>10s} | {'BU-OO+mFC':>10s} | "
          f"{'Diff':>8s} | {'N':>5s} | {'BU avail':>8s}")
    print(f"  {'-' * 60}")
    for tf in sorted(test['Timeframe'].unique()):
        for lag in sorted(test['Prediction_Lag'].unique()):
            sub = test[m2 & (test['Timeframe'] == tf) &
                       (test['Prediction_Lag'] == lag) &
                       (test['Actual_Sales'] > 0)]
            if len(sub) == 0:
                continue
            v1 = sub[sub['V8d + mFC'].notna()]
            v2 = sub[sub['BU-OO + mFC'].notna()]
            if len(v1) > 0:
                w1 = compute_wmape(v1['Actual_Sales'].values, v1['V8d + mFC'].values)
                w2 = compute_wmape(v2['Actual_Sales'].values,
                                   v2['BU-OO + mFC'].values) if len(v2) > 0 else np.nan
                diff = w2 - w1 if not np.isnan(w2) else np.nan
                diff_s = f"{diff:>+6.1f}pp" if not np.isnan(diff) else "    n/a"
                w2_s = f"{w2:>8.1f}%" if not np.isnan(w2) else "     n/a"
                n_bu = sub['oo_implied_bu'].notna().sum()
                print(f"  {tf:>4d} | {lag:>4d} | {w1:>8.1f}% | {w2_s:>10s} | "
                      f"{diff_s:>8s} | {len(sub):>5d} | {n_bu:>4d}/{len(sub)}")

    # ============================================================
    # STEP 6: Sub-quarter prediction diagnostics
    # ============================================================
    print(f"\n  {'=' * 80}")
    print(f"  DIAGNOSTICS")
    print(f"  {'=' * 80}")

    # OO signal quality: bottom-up implied vs actual (for TF>3 only)
    print(f"\n  OO signal accuracy (TF>3 test rows, months 2+):")
    print(f"  {'Signal':>25s} | {'WMAPE':>7s} | {'Bias':>7s} | {'N':>5s}")
    print(f"  {'-' * 50}")
    for label, col in [('OO flat (per-TF)', 'oo_implied_flat'),
                        ('OO bottom-up (all)', 'oo_implied_bu'),
                        ('OO bottom-up (shares)', 'oo_implied_bu_shares')]:
        sub = test[m2 & (test['Timeframe'] > 3) & test[col].notna() &
                   (test[col] > 0) & (test['Actual_Sales'] > 0)]
        if len(sub) > 0:
            w = compute_wmape(sub['Actual_Sales'].values, sub[col].values)
            b = compute_bias(sub['Actual_Sales'].values, sub[col].values)
            print(f"  {label:>25s} | {w:>5.1f}% | {b:>+5.1f}% | {len(sub):>5d}")

    # Per-TF OO signal quality
    print(f"\n  Per-TF OO signal accuracy (test rows, months 2+):")
    print(f"  {'TF':>4s} | {'Flat WMAPE':>10s} | {'BU WMAPE':>10s} | "
          f"{'BU(sh) WMAPE':>12s} | {'Diff (BU-Flat)':>14s}")
    print(f"  {'-' * 60}")
    for tf in sorted(test['Timeframe'].unique()):
        sub = test[m2 & (test['Timeframe'] == tf) & (test['Actual_Sales'] > 0)]
        v_flat = sub[sub['oo_implied_flat'].notna() & (sub['oo_implied_flat'] > 0)]
        v_bu = sub[sub['oo_implied_bu'].notna() & (sub['oo_implied_bu'] > 0)]
        v_sh = sub[sub['oo_implied_bu_shares'].notna() & (sub['oo_implied_bu_shares'] > 0)]

        w_flat = compute_wmape(v_flat['Actual_Sales'].values,
                               v_flat['oo_implied_flat'].values) if len(v_flat) > 0 else np.nan
        w_bu = compute_wmape(v_bu['Actual_Sales'].values,
                             v_bu['oo_implied_bu'].values) if len(v_bu) > 0 else np.nan
        w_sh = compute_wmape(v_sh['Actual_Sales'].values,
                             v_sh['oo_implied_bu_shares'].values) if len(v_sh) > 0 else np.nan

        flat_s = f"{w_flat:>8.1f}%" if not np.isnan(w_flat) else "     n/a"
        bu_s = f"{w_bu:>8.1f}%" if not np.isnan(w_bu) else "     n/a"
        sh_s = f"{w_sh:>10.1f}%" if not np.isnan(w_sh) else "       n/a"
        diff = w_bu - w_flat if not (np.isnan(w_bu) or np.isnan(w_flat)) else np.nan
        diff_s = f"{diff:>+12.1f}pp" if not np.isnan(diff) else "          n/a"
        print(f"  {tf:>4d} | {flat_s:>10s} | {bu_s:>10s} | {sh_s:>12s} | {diff_s:>14s}")

    # ============================================================
    # STEP 7: K-Fold Random CV
    # ============================================================
    print(f"\n  {'=' * 80}")
    print(f"  K-FOLD RANDOM CV ({N_FOLDS} folds)")
    print(f"  {'=' * 80}")

    np.random.seed(RANDOM_SEED)
    indices = np.arange(len(df))
    np.random.shuffle(indices)
    fold_size = len(df) // N_FOLDS

    cv_models = {
        'V8d + mFC':       ('oo_implied_flat',      'fc_implied_multi'),
        'BU-OO + mFC':     ('oo_implied_bu',        'fc_implied_multi'),
        'BU-OO(sh) + mFC': ('oo_implied_bu_shares', 'fc_implied_multi'),
        'FC-only multi':   ('oo_none',              'fc_implied_multi'),
    }
    fold_metrics = {mn: [] for mn in cv_models}

    for fold_i in range(N_FOLDS):
        fold_start = fold_i * fold_size
        fold_end = fold_start + fold_size if fold_i < N_FOLDS - 1 else len(df)
        test_idx = indices[fold_start:fold_end]
        train_idx = np.setdiff1d(indices, test_idx)

        f_train = df.iloc[train_idx].copy()
        f_test = df.iloc[test_idx].copy()

        # Build curves from fold training data
        oo_f = build_flat_curves(f_train, 'oo_ratio', start='2023-01-01')
        cov_f = build_flat_curves(f_train, 'fc_coverage', start='2023-01-01')
        bias_f = build_flat_curves(f_train, 'fc_bias', start='2023-01-01')

        # TF=3 curves
        tf3_f = f_train[f_train['Timeframe'] == 3]
        oo_f_tf3 = build_flat_curves(tf3_f, 'oo_ratio', start='2023-01-01')
        cov_f_tf3 = build_flat_curves(tf3_f, 'fc_coverage', start='2023-01-01')
        bias_f_tf3 = build_flat_curves(tf3_f, 'fc_bias', start='2023-01-01')

        # Apply flat curves
        apply_flat_curves(f_train, oo_f, cov_f, bias_f)
        apply_flat_curves(f_test, oo_f, cov_f, bias_f)

        # Multi-vintage FC from training fold
        fc_reg_fold = build_forecast_registry(f_train)
        apply_multi_vintage_fc(f_train, cov_f, bias_f, fc_reg_fold, 'fold')
        apply_multi_vintage_fc(f_test, cov_f, bias_f, fc_reg_fold, 'fold')

        # Sub-quarter decomposition
        tf3_lk = build_tf3_lookup(df)  # structural lookup from full data
        sq_map_ft = build_subquarter_map(f_train, tf3_lk)
        sq_map_fte = build_subquarter_map(f_test, tf3_lk)
        f_shares = compute_historical_shares(f_train, tf3_lk)

        # Strategy A
        apply_bottom_up(f_train, sq_map_ft, oo_f_tf3, cov_f_tf3, bias_f_tf3,
                        mode='all_or_nothing')
        apply_bottom_up(f_test, sq_map_fte, oo_f_tf3, cov_f_tf3, bias_f_tf3,
                        mode='all_or_nothing')

        # Strategy B (shares)
        apply_bottom_up(f_train, sq_map_ft, oo_f_tf3, cov_f_tf3, bias_f_tf3,
                        shares=f_shares, mode='shares')
        f_train.rename(columns={'oo_implied_bu': 'oo_implied_bu_shares',
                                'fc_implied_bu': 'fc_implied_bu_shares'}, inplace=True)
        apply_bottom_up(f_train, sq_map_ft, oo_f_tf3, cov_f_tf3, bias_f_tf3,
                        mode='all_or_nothing')

        apply_bottom_up(f_test, sq_map_fte, oo_f_tf3, cov_f_tf3, bias_f_tf3,
                        shares=f_shares, mode='shares')
        f_test.rename(columns={'oo_implied_bu': 'oo_implied_bu_shares',
                               'fc_implied_bu': 'fc_implied_bu_shares'}, inplace=True)
        apply_bottom_up(f_test, sq_map_fte, oo_f_tf3, cov_f_tf3, bias_f_tf3,
                        mode='all_or_nothing')

        f_train['oo_none'] = np.nan
        f_test['oo_none'] = np.nan

        f_test_months = sorted(f_test['Reference_Month'].unique())

        for model_name, (oo_c, fc_c) in cv_models.items():
            f_test[model_name] = weighted_avg(f_test, f_train, f_test_months,
                                              gsa_sites, oo_col=oo_c, fc_col=fc_c)

        ts = f_test[f_test['Actual_Sales'] > 0]
        for model_name in cv_models:
            v = ts[ts[model_name].notna()]
            if len(v) > 0:
                wmape = compute_wmape(v['Actual_Sales'].values, v[model_name].values)
                bias = compute_bias(v['Actual_Sales'].values, v[model_name].values)
                fold_metrics[model_name].append({'wmape': wmape, 'bias': bias})

        v8d_w = fold_metrics['V8d + mFC'][-1]['wmape'] if fold_metrics['V8d + mFC'] else 0
        bu_w = fold_metrics['BU-OO + mFC'][-1]['wmape'] if fold_metrics['BU-OO + mFC'] else 0
        print(f"    Fold {fold_i + 1}: V8d+mFC={v8d_w:.1f}%  BU-OO+mFC={bu_w:.1f}%")

    # Random CV summary
    print(f"\n  {'Model':>22s} | {'Random CV':>20s} | {'Temporal':>10s} | {'Gap':>6s}")
    print(f"  {'-' * 65}")
    for model_name in cv_models:
        wmapes = [m['wmape'] for m in fold_metrics[model_name]]
        biases = [m['bias'] for m in fold_metrics[model_name]]
        t_wmape = temporal_results.get(model_name, float('nan'))
        gap = t_wmape - np.mean(wmapes) if wmapes else float('nan')
        if wmapes:
            print(f"  {model_name:>22s} | {np.mean(wmapes):>5.1f}% +/- {np.std(wmapes):>4.1f}% "
                  f"(b{np.mean(biases):>+5.1f}%) | {t_wmape:>6.1f}% | {gap:>+4.1f}pp")

    print(f"\n\nDone!")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Sub-Quarter Decomposition Experiment')
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
    for cust in customers:
        cust_df = df[df['GSA'] == cust].copy()
        run_experiment(cust_df, cust)


if __name__ == '__main__':
    main()
