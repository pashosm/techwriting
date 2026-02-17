"""
Unified Aging Curve Pipeline
=============================
Processes all customers from a single dataset.
Auto-detects signal quality and selects appropriate model variants per customer.

Usage:
    python pipeline.py                          # runs on all customers in dataset
    python pipeline.py --customer "Customer 1"  # runs on one customer only

Input:  Single CSV with all customers, shared columns:
    Reference_Month, Target_Period_Start, Target_Period_End, Timeframe,
    Prediction_Lag, GSA, Site, Actual_Sales, Open_Orders, Covered_Orders,
    Forecast_Value, Has_Forecast, Avg_Weighted_Lead_Time,
    Avg_Weighted_Order_Earliness, Historical_Sales_Lag1, Historical_Sales_Lag12

Output: Per-customer summary tables printed to stdout.
"""

import pandas as pd
import numpy as np
import warnings
import argparse
warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTS (shared across all customers)
# ============================================================
RECENCY_POWER = 2          # Power for recency weighting in curve building
MIN_OBS_FLAT = 3           # Minimum observations to build a flat curve
MIN_OBS_COND = 8           # Minimum observations for conditioned regression
MIN_R2_IMPROVE = 0.05      # Conditioned curve must improve R² by at least this
MIN_SITE_ROWS = 30         # Auto-exclude sites with fewer rows than this
VINTAGE_RECENCY_POWER = 1.5  # Power for inverse-lag weighting in multi-vintage
N_FOLDS = 5                # Number of folds for random CV
RANDOM_SEED = 42
OO_NONSTATIONARY_GAP = 20  # If random CV WMAPE exceeds temporal by this many pp, flag OO
LAG_BUCKETS = {'short': (1, 3), 'medium': (4, 7), 'long': (8, 12)}
MIN_FOLDS_FOR_GAP = 3      # Need at least this many folds with observations to trust gap
CI_LEVEL = 0.80            # Prediction interval coverage target
MIN_OBS_CI = 5             # Minimum observations to compute quantiles at a granularity level

# ============================================================
# CORE FUNCTIONS
# ============================================================

def parse_dollar(s):
    """Parse dollar-formatted strings like '$1,234.56' or '($500.00)' to float."""
    if isinstance(s, str):
        s = s.replace('$', '').replace(',', '').strip()
        if s in ('', '-', 'N/A', 'n/a'):
            return np.nan
        if s.startswith('(') and s.endswith(')'):
            return -float(s[1:-1])
        return float(s)
    return s


def load_and_preprocess(filepath, dollar_cols=None):
    """Load CSV and compute derived columns shared across all customers.

    Parameters
    ----------
    filepath : str
        Path to the unified CSV.
    dollar_cols : list[str] or None
        Columns that need dollar-string parsing. If None, auto-detected.

    Returns
    -------
    pd.DataFrame with derived columns added.
    """
    df = pd.read_csv(filepath,
                     parse_dates=['Reference_Month', 'Target_Period_Start',
                                  'Target_Period_End'])
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # Auto-detect dollar columns: any column that should be numeric but loaded as string
    numeric_cols = ['Actual_Sales', 'Open_Orders', 'Covered_Orders', 'Forecast_Value',
                    'Historical_Sales_Lag1', 'Historical_Sales_Lag12']
    if dollar_cols is None:
        dollar_cols = []
        for col in numeric_cols:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                dollar_cols.append(col)
    if dollar_cols:
        for col in dollar_cols:
            if col in df.columns:
                df[col] = df[col].apply(parse_dollar)

    # Derived columns
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


# ---- Curve Building ----

def build_flat_curves(train_df, ratio_col, start=None):
    """Build recency-weighted flat aging curves.

    Returns dict: {(gsa_site, Timeframe, Prediction_Lag): value}
    plus fallback keys {('FB', gsa_site, Prediction_Lag): value}.
    """
    td = train_df if start is None else train_df[train_df['Reference_Month'] >= start]
    flat = {}

    for (gs, tf, lag), grp in td.groupby(['gsa_site', 'Timeframe', 'Prediction_Lag']):
        valid = grp[grp[ratio_col].notna() & (grp[ratio_col] > 0) & (grp[ratio_col] < 10)]
        if len(valid) < MIN_OBS_FLAT:
            continue
        days = (valid['Reference_Month'] - valid['Reference_Month'].min()).dt.days / 365
        wts = (1 + days.values) ** RECENCY_POWER
        flat[(gs, tf, lag)] = np.average(valid[ratio_col], weights=wts)

    # Fallback: pooled across timeframes
    for (gs, lag), grp in td.groupby(['gsa_site', 'Prediction_Lag']):
        valid = grp[grp[ratio_col].notna() & (grp[ratio_col] > 0) & (grp[ratio_col] < 10)]
        if len(valid) >= MIN_OBS_FLAT:
            days = (valid['Reference_Month'] - valid['Reference_Month'].min()).dt.days / 365
            wts = (1 + days.values) ** RECENCY_POWER
            flat[('FB', gs, lag)] = np.average(valid[ratio_col], weights=wts)

    return flat


def apply_flat_curves(dset, oo_f, cov_f, bias_f):
    """Apply flat OO and FC curves to produce oo_implied_flat and fc_implied_flat."""
    oo_impl = np.full(len(dset), np.nan)
    fc_impl = np.full(len(dset), np.nan)

    for i, (idx, row) in enumerate(dset.iterrows()):
        gs, tf, lag = row['gsa_site'], row['Timeframe'], row['Prediction_Lag']

        # OO implied
        for key in [(gs, tf, lag), ('FB', gs, lag)]:
            if key in oo_f and oo_f[key] > 0.001:
                oo_impl[i] = row['Open_Orders'] / oo_f[key]
                break

        # FC implied
        for key in [(gs, tf, lag), ('FB', gs, lag)]:
            cf = cov_f.get(key)
            bf = bias_f.get(key)
            if cf and bf and cf > 0.001 and bf > 0.001 and row['Forecast_Value'] > 0:
                fc_impl[i] = row['Forecast_Value'] / bf / cf
                break

    dset['oo_implied_flat'] = np.maximum(oo_impl, 0)
    dset['fc_implied_flat'] = np.maximum(fc_impl, 0)


# ---- Multi-Vintage Forecast Functions ----

def build_forecast_registry(data_df):
    """Index forecast-bearing rows by target period for O(1) lookup.

    Returns {(gsa_site, TP_Start, TP_End): [vintage_dicts sorted by lag asc]}
    """
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
    """Convert one forecast vintage to an implied-actual estimate using flat curves
    at the vintage's original (site, tf, lag)."""
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
    """Combine multiple vintage estimates with inverse-lag power weighting.
    w = 1 / (lag ^ VINTAGE_RECENCY_POWER)"""
    if not estimates_with_lags:
        return np.nan
    total_w, total_val = 0.0, 0.0
    for est, orig_lag in estimates_with_lags:
        w = 1.0 / (orig_lag ** VINTAGE_RECENCY_POWER)
        total_w += w
        total_val += w * est
    return total_val / total_w if total_w > 0 else np.nan


def apply_multi_vintage_fc(dset, cov_f, bias_f, registry, causality='temporal'):
    """Apply multi-vintage FC curves. For each row, look up ALL causally-valid
    forecasts for its target period and combine them.

    causality='temporal': only use forecasts with ref_month <= row's ref_month
    causality='fold': registry already restricted to train fold, no extra filter

    Adds columns: fc_implied_multi, n_vintages_used, best_vintage_lag
    """
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


# ---- Multi-Vintage OO ----

def build_oo_registry(data_df):
    """Index OO observations by target period for multi-vintage OO.

    Returns {(gsa_site, TP_Start, TP_End): [dicts sorted by lag asc]}
    """
    registry = {}
    for _, row in data_df.iterrows():
        if row['Open_Orders'] <= 0 or pd.isna(row['Actual_Sales']) or row['Actual_Sales'] <= 0:
            continue
        key = (row['gsa_site'], row['Target_Period_Start'], row['Target_Period_End'])
        if key not in registry:
            registry[key] = []
        registry[key].append({
            'ref_month': row['Reference_Month'],
            'lag': row['Prediction_Lag'],
            'oo': row['Open_Orders'],
            'tf': row['Timeframe'],
            'gsa_site': row['gsa_site'],
        })
    for key in registry:
        registry[key].sort(key=lambda v: v['lag'])
    return registry


def _get_oo_curve(oo_f, gs, tf, lag):
    """Look up flat OO curve value with fallback."""
    for key in [(gs, tf, lag), ('FB', gs, lag)]:
        if key in oo_f and oo_f[key] > 0.001:
            return oo_f[key]
    return None


def apply_multi_vintage_oo(dset, oo_f, registry, causality='temporal'):
    """Multi-vintage OO: combine OO/curve(L) across all causally-valid lags.

    Adds column: oo_implied_multi
    """
    n = len(dset)
    oo_multi = np.full(n, np.nan)

    for i, (idx, row) in enumerate(dset.iterrows()):
        gs = row['gsa_site']
        tp_key = (gs, row['Target_Period_Start'], row['Target_Period_End'])

        if tp_key not in registry:
            continue

        estimates = []
        for v in registry[tp_key]:
            if causality == 'temporal' and v['ref_month'] > row['Reference_Month']:
                continue
            cv = _get_oo_curve(oo_f, v['gsa_site'], v['tf'], v['lag'])
            if cv is not None and v['oo'] > 0:
                implied = v['oo'] / cv
                estimates.append((max(implied, 0), v['lag']))

        if estimates:
            oo_multi[i] = max(_combine_vintage_estimates(estimates), 0)

    dset['oo_implied_multi'] = oo_multi


# ---- FC Debiasing ----

def build_fc_debias_curves(train_df):
    """Learn correction factors per (site, tf, lag) from training data.
    fc_debias[(gs,tf,lag)] = recency-weighted mean of (Actual_Sales / fc_implied_flat)
    """
    fc_debias = {}
    for (gs, tf, lag), grp in train_df.groupby(['gsa_site', 'Timeframe', 'Prediction_Lag']):
        v = grp[grp['fc_implied_flat'].notna() & (grp['fc_implied_flat'] > 0) &
                (grp['Actual_Sales'] > 0)].copy()
        if len(v) >= MIN_OBS_FLAT:
            ratio = v['Actual_Sales'].values / v['fc_implied_flat'].values
            mask = (ratio > 0.1) & (ratio < 10)
            if mask.sum() >= MIN_OBS_FLAT:
                v_m = v.iloc[mask]
                days = (v_m['Reference_Month'] - v_m['Reference_Month'].min()).dt.days / 365
                wts = (1 + days.values) ** RECENCY_POWER
                fc_debias[(gs, tf, lag)] = np.average(ratio[mask], weights=wts)

    # Fallback: pooled across TFs
    for (gs, lag), grp in train_df.groupby(['gsa_site', 'Prediction_Lag']):
        v = grp[grp['fc_implied_flat'].notna() & (grp['fc_implied_flat'] > 0) &
                (grp['Actual_Sales'] > 0)].copy()
        if len(v) >= MIN_OBS_FLAT:
            ratio = v['Actual_Sales'].values / v['fc_implied_flat'].values
            mask = (ratio > 0.1) & (ratio < 10)
            if mask.sum() >= MIN_OBS_FLAT:
                v_m = v.iloc[mask]
                days = (v_m['Reference_Month'] - v_m['Reference_Month'].min()).dt.days / 365
                wts = (1 + days.values) ** RECENCY_POWER
                fc_debias[('FB', gs, lag)] = np.average(ratio[mask], weights=wts)

    return fc_debias


def apply_fc_debias(dset, fc_debias):
    """Apply learned FC debiasing correction factors."""
    fc_db = np.full(len(dset), np.nan)
    for i, (idx, row) in enumerate(dset.iterrows()):
        gs, tf, lag = row['gsa_site'], row['Timeframe'], row['Prediction_Lag']
        fc_val = row['fc_implied_flat']
        if np.isnan(fc_val) or fc_val <= 0:
            continue
        for key in [(gs, tf, lag), ('FB', gs, lag)]:
            if key in fc_debias:
                fc_db[i] = fc_val * fc_debias[key]
                break
    dset['fc_implied_debiased'] = np.maximum(fc_db, 0)


# ---- Weighted Average Blending ----

def weighted_avg(test_df, train_df, test_months, gsa_sites_list,
                 oo_col='oo_implied_flat', fc_col='fc_implied_flat'):
    """Blend OO signal, FC signal, and historical average into a final prediction.

    Parameters
    ----------
    test_df, train_df : DataFrames
    test_months : sorted list of test Reference_Months
    gsa_sites_list : list of gsa_site values for this customer
    oo_col : column name for OO implied actual (use 'oo_none' to disable OO)
    fc_col : column name for FC implied actual
    """
    results = test_df.copy()
    results['pred'] = np.nan

    # Ensure oo_none column exists
    if oo_col == 'oo_none':
        if 'oo_none' not in results.columns:
            results['oo_none'] = np.nan
        if 'oo_none' not in train_df.columns:
            train_df = train_df.copy()
            train_df['oo_none'] = np.nan

    # Precompute historical averages
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

                # OO MAPE from prior
                oo_v = ps[ps[oo_col].notna() & (ps[oo_col] > 0)]
                oo_mape = (np.mean(np.abs(oo_v[oo_col] - oo_v['Actual_Sales']) /
                           oo_v['Actual_Sales']) if len(oo_v) >= 1 else 1.0)

                # FC MAPE from prior
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

                    # OO lag factor
                    oo_f = max(0.3, 1.0 - 0.06 * (lag - 1)) if has_oo else 0

                    # FC lag factor — use best_vintage_lag when multi-vintage
                    fc_lag = lag
                    if (has_fc and fc_col == 'fc_implied_multi' and
                            'best_vintage_lag' in results.columns):
                        bvl = results.at[idx, 'best_vintage_lag']
                        if not np.isnan(bvl):
                            fc_lag = bvl
                    fc_f = max(0.3, 1.0 - 0.06 * (fc_lag - 1)) if has_fc else 0

                    # Weights
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


# ---- Evaluation Metrics ----

def compute_wmape(actual, predicted):
    """Weighted Mean Absolute Percentage Error."""
    mask = actual > 0
    if mask.sum() == 0:
        return np.nan
    return np.sum(np.abs(actual[mask] - predicted[mask])) / np.sum(actual[mask]) * 100


def compute_bias(actual, predicted):
    """Mean signed percentage error."""
    mask = actual > 0
    if mask.sum() == 0:
        return np.nan
    return np.mean((predicted[mask] - actual[mask]) / actual[mask]) * 100


def lag_to_bucket(lag):
    """Map a prediction lag to its bucket name."""
    for name, (lo, hi) in LAG_BUCKETS.items():
        if lo <= lag <= hi:
            return name
    return 'long'


def build_model_selection_table(gsa_sites, models,
                                temporal_results, temporal_site, temporal_site_lag,
                                cv_results, fold_site_metrics, fold_site_lag_metrics,
                                customer_recommended):
    """Hierarchical cascade model selection.

    For each (site, lag_bucket), select the best model using the finest
    granularity where we have enough CV folds to measure the temporal/random gap.

    Cascade levels:
        1. (site, lag_bucket) — most granular
        2. (site)             — pooled across lags
        3. customer-level     — existing global default

    Parameters
    ----------
    gsa_sites : list of site identifiers
    models : dict of {model_name: (oo_col, fc_col)}
    temporal_results : dict {model_name: wmape} — customer-level temporal
    temporal_site : dict {(gs, model_name): wmape}
    temporal_site_lag : dict {(gs, lag_bucket, model_name): wmape}
    cv_results : dict {model_name: {cv_mean, gap, ...}} — customer-level CV
    fold_site_metrics : dict {(gs, model_name): [wmape_per_fold]}
    fold_site_lag_metrics : dict {(gs, lb, model_name): [wmape_per_fold]}
    customer_recommended : str — fallback model name

    Returns
    -------
    selection : dict {(gsa_site, lag_bucket): (model_name, level, gap)}
    """
    # The two key contenders: OO-inclusive vs FC-only
    oo_model = 'mOO + mFC'
    fc_model = 'FC-only multi'
    oo_probe = 'V8d (OO+FC)'  # Used to measure OO stationarity

    selection = {}

    for gs in gsa_sites:
        for lb in LAG_BUCKETS:
            # ---- Level 1: (site, lag_bucket) ----
            probe_key = (gs, lb, oo_probe)
            if (probe_key in temporal_site_lag
                    and probe_key in fold_site_lag_metrics
                    and len(fold_site_lag_metrics[probe_key]) >= MIN_FOLDS_FOR_GAP):
                t_w = temporal_site_lag[probe_key]
                cv_w = np.mean(fold_site_lag_metrics[probe_key])
                gap = t_w - cv_w
                if abs(gap) > OO_NONSTATIONARY_GAP:
                    selection[(gs, lb)] = (fc_model, 'site+lag', gap)
                else:
                    # OO is stationary at this level — pick the better contender
                    oo_tw = temporal_site_lag.get((gs, lb, oo_model), float('inf'))
                    fc_tw = temporal_site_lag.get((gs, lb, fc_model), float('inf'))
                    best = oo_model if oo_tw <= fc_tw else fc_model
                    selection[(gs, lb)] = (best, 'site+lag', gap)
                continue

            # ---- Level 2: (site) ----
            probe_skey = (gs, oo_probe)
            if (probe_skey in temporal_site
                    and probe_skey in fold_site_metrics
                    and len(fold_site_metrics[probe_skey]) >= MIN_FOLDS_FOR_GAP):
                t_w = temporal_site[probe_skey]
                cv_w = np.mean(fold_site_metrics[probe_skey])
                gap = t_w - cv_w
                if abs(gap) > OO_NONSTATIONARY_GAP:
                    selection[(gs, lb)] = (fc_model, 'site', gap)
                else:
                    oo_tw = temporal_site.get((gs, oo_model), float('inf'))
                    fc_tw = temporal_site.get((gs, fc_model), float('inf'))
                    best = oo_model if oo_tw <= fc_tw else fc_model
                    selection[(gs, lb)] = (best, 'site', gap)
                continue

            # ---- Level 3: customer-level default ----
            cust_gap = cv_results.get(oo_probe, {}).get('gap', 0)
            selection[(gs, lb)] = (customer_recommended, 'customer', cust_gap)

    return selection


def build_prediction_intervals(actuals, preds, sites, timeframes, lags):
    """Build parametric prediction interval ratios from out-of-sample data.

    Models log(actual/predicted) as normal per (site, tf, lag) group.
    Uses a prediction interval with small-sample correction:
        margin = z * sigma * sqrt(1 + 1/n)
    which widens intervals when data is sparse.

    Hierarchical fallback: (site,tf,lag) → (site,lag) → (site) → global.

    Parameters
    ----------
    actuals, preds : array-like of actual and predicted values
    sites, timeframes, lags : array-like of corresponding metadata

    Returns
    -------
    ci_lookup : dict mapping group keys to (q_lower, q_upper) ratio pairs.
    """
    Z_80 = 1.2816  # norm.ppf(0.90) for 80% two-sided interval

    a = np.asarray(actuals, dtype=float)
    p = np.asarray(preds, dtype=float)
    valid = (p > 0) & (a > 0) & np.isfinite(p) & np.isfinite(a)
    if valid.sum() < MIN_OBS_CI:
        return {}

    df = pd.DataFrame({
        'log_r': np.log(a[valid] / p[valid]),
        's': np.asarray(sites)[valid],
        't': np.asarray(timeframes)[valid],
        'l': np.asarray(lags)[valid],
    })

    def _interval(grp):
        n = len(grp)
        mu = grp['log_r'].mean()
        sigma = grp['log_r'].std(ddof=1)  # Bessel-corrected
        margin = Z_80 * sigma * np.sqrt(1 + 1 / n)
        return (np.exp(mu - margin), np.exp(mu + margin))

    ci = {}

    # Level 1: (site, tf, lag)
    for (sv, tv, lv), grp in df.groupby(['s', 't', 'l']):
        if len(grp) >= MIN_OBS_CI:
            ci[(sv, tv, lv)] = _interval(grp)

    # Level 2: (site, lag) — pooled across timeframes
    for (sv, lv), grp in df.groupby(['s', 'l']):
        if len(grp) >= MIN_OBS_CI:
            ci[('FB_lag', sv, lv)] = _interval(grp)

    # Level 3: (site) — pooled across tf and lag
    for sv, grp in df.groupby('s'):
        if len(grp) >= MIN_OBS_CI:
            ci[('FB_site', sv)] = _interval(grp)

    # Level 4: global
    ci['FB_global'] = _interval(df)

    return ci


def lookup_ci(ci_lookup, site, tf, lag):
    """Look up CI ratio pair with hierarchical fallback.

    Returns (q_lower, q_upper) — multiply prediction by these to get interval.
    """
    for key in [(site, tf, lag), ('FB_lag', site, lag),
                ('FB_site', site), 'FB_global']:
        if key in ci_lookup:
            return ci_lookup[key]
    return (np.nan, np.nan)


# ============================================================
# PER-CUSTOMER PIPELINE
# ============================================================

def run_customer_pipeline(cust_df, customer_name):
    """Run the full pipeline for one customer.

    Parameters
    ----------
    cust_df : DataFrame — all rows for this customer (already preprocessed)
    customer_name : str — for display

    Returns
    -------
    dict with results summary
    """
    print(f"\n{'#' * 100}")
    print(f"# CUSTOMER: {customer_name}")
    print(f"{'#' * 100}")

    # ---- Auto-detect usable sites ----
    site_counts = cust_df.groupby('gsa_site').size()
    valid_sites = sorted(site_counts[site_counts >= MIN_SITE_ROWS].index.tolist())
    dropped_sites = sorted(site_counts[site_counts < MIN_SITE_ROWS].index.tolist())
    df = cust_df[cust_df['gsa_site'].isin(valid_sites)].copy()

    gsa_sites = sorted(df['gsa_site'].unique())
    fc_pct = (df['Has_Forecast'] == 1).mean() * 100
    oo_pct = (df['Open_Orders'] > 0).mean() * 100

    print(f"\n  Data: {len(df)} rows, {df['Reference_Month'].nunique()} months, "
          f"{len(gsa_sites)} sites")
    if dropped_sites:
        print(f"  Dropped {len(dropped_sites)} sites with < {MIN_SITE_ROWS} rows: "
              f"{dropped_sites[:5]}{'...' if len(dropped_sites) > 5 else ''}")
    print(f"  Sites: {gsa_sites}")
    print(f"  Forecast coverage: {fc_pct:.0f}%  |  OO coverage: {oo_pct:.0f}%")

    # ---- Train / test split ----
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

    # ---- Build flat curves ----
    print(f"\n  Building curves...")
    oo_flat = build_flat_curves(train, 'oo_ratio', start='2023-01-01')
    cov_flat = build_flat_curves(train, 'fc_coverage', start='2023-01-01')
    bias_flat = build_flat_curves(train, 'fc_bias', start='2023-01-01')
    print(f"    OO curves: {len(oo_flat)}  |  FC coverage: {len(cov_flat)}  |  "
          f"FC bias: {len(bias_flat)}")

    # ---- Apply flat curves ----
    apply_flat_curves(train, oo_flat, cov_flat, bias_flat)
    apply_flat_curves(test, oo_flat, cov_flat, bias_flat)

    # ---- Build and apply multi-vintage FC ----
    fc_registry = build_forecast_registry(df)
    print(f"    FC registry: {len(fc_registry)} target periods with forecasts")
    apply_multi_vintage_fc(train, cov_flat, bias_flat, fc_registry, causality='temporal')
    apply_multi_vintage_fc(test, cov_flat, bias_flat, fc_registry, causality='temporal')

    n_single = test['fc_implied_flat'].notna().sum()
    n_multi = test['fc_implied_multi'].notna().sum()
    print(f"    Single-vintage FC coverage: {n_single}/{len(test)} "
          f"({100 * n_single / len(test):.0f}%)")
    print(f"    Multi-vintage FC coverage:  {n_multi}/{len(test)} "
          f"({100 * n_multi / len(test):.0f}%)")

    # ---- Build and apply multi-vintage OO ----
    oo_registry = build_oo_registry(df)
    print(f"    OO registry: {len(oo_registry)} target periods with OO data")
    apply_multi_vintage_oo(train, oo_flat, oo_registry, causality='temporal')
    apply_multi_vintage_oo(test, oo_flat, oo_registry, causality='temporal')

    n_oo_s = test['oo_implied_flat'].notna().sum()
    n_oo_m = test['oo_implied_multi'].notna().sum()
    print(f"    Single-vintage OO coverage: {n_oo_s}/{len(test)} "
          f"({100 * n_oo_s / len(test):.0f}%)")
    print(f"    Multi-vintage OO coverage:  {n_oo_m}/{len(test)} "
          f"({100 * n_oo_m / len(test):.0f}%)")

    # ---- FC debiasing ----
    fc_debias = build_fc_debias_curves(train)
    apply_fc_debias(train, fc_debias)
    apply_fc_debias(test, fc_debias)

    # ---- Ensure oo_none column ----
    train['oo_none'] = np.nan
    test['oo_none'] = np.nan

    # ---- Temporal evaluation ----
    print(f"\n  {'=' * 80}")
    print(f"  TEMPORAL EVALUATION — {customer_name}")
    print(f"  {'=' * 80}")

    models = {
        'V8d (OO+FC)':      ('oo_implied_flat', 'fc_implied_flat'),
        'FC-only':           ('oo_none',         'fc_implied_flat'),
        'FC-only debiased':  ('oo_none',         'fc_implied_debiased'),
        'V8d + multi-FC':    ('oo_implied_flat', 'fc_implied_multi'),
        'FC-only multi':     ('oo_none',         'fc_implied_multi'),
        'mOO + mFC':         ('oo_implied_multi', 'fc_implied_multi'),
        'mOO + FC flat':     ('oo_implied_multi', 'fc_implied_flat'),
    }

    temporal_results = {}
    for model_name, (oo_c, fc_c) in models.items():
        test[model_name] = weighted_avg(test, train, test_months, gsa_sites,
                                        oo_col=oo_c, fc_col=fc_c)

    # Skip first test month for fair comparison (cold start)
    m2 = test['Reference_Month'] > test_months[0]

    print(f"\n  {'Model':>22s} | {'WMAPE':>7s} | {'Bias':>7s} | {'N':>5s}")
    print(f"  {'-' * 50}")
    for model_name in models:
        ts = test[m2 & test[model_name].notna() & (test['Actual_Sales'] > 0)]
        if len(ts) == 0:
            continue
        w = compute_wmape(ts['Actual_Sales'].values, ts[model_name].values)
        b = compute_bias(ts['Actual_Sales'].values, ts[model_name].values)
        temporal_results[model_name] = w
        print(f"  {model_name:>22s} | {w:>5.1f}% | {b:>+5.1f}% | {len(ts):>5d}")

    # Per-site breakdown for best models
    print(f"\n  Per-site WMAPE (temporal, months 2+):")
    print(f"  {'Site':>25s} | {'V8d':>8s} | {'V8d+mFC':>8s} | {'FC-multi':>8s} | {'N':>4s}")
    print(f"  {'-' * 65}")
    for gs in gsa_sites:
        sub = test[m2 & (test['gsa_site'] == gs) & (test['Actual_Sales'] > 0)]
        if len(sub) == 0:
            continue
        vals = {}
        for lbl, mn in [('d', 'V8d (OO+FC)'), ('dm', 'V8d + multi-FC'),
                        ('fm', 'FC-only multi')]:
            v = sub[sub[mn].notna()]
            if len(v) > 0:
                vals[lbl] = compute_wmape(v['Actual_Sales'].values, v[mn].values)
        if vals:
            d_s = f"{vals.get('d', np.nan):>6.1f}%" if 'd' in vals else "    n/a"
            dm_s = f"{vals.get('dm', np.nan):>6.1f}%" if 'dm' in vals else "    n/a"
            fm_s = f"{vals.get('fm', np.nan):>6.1f}%" if 'fm' in vals else "    n/a"
            print(f"  {gs:>25s} | {d_s:>8s} | {dm_s:>8s} | {fm_s:>8s} | {len(sub):>4d}")

    # Per-lag breakdown
    print(f"\n  WMAPE by lag (temporal, months 2+):")
    print(f"  {'Lag':>4s} | {'V8d':>8s} | {'V8d+mFC':>8s} | {'FC-multi':>8s}")
    print(f"  {'-' * 40}")
    for lag in sorted(test['Prediction_Lag'].unique()):
        sub = test[m2 & (test['Prediction_Lag'] == lag) & (test['Actual_Sales'] > 0)]
        vals = {}
        for lbl, mn in [('d', 'V8d (OO+FC)'), ('dm', 'V8d + multi-FC'),
                        ('fm', 'FC-only multi')]:
            v = sub[sub[mn].notna()]
            if len(v) > 0:
                vals[lbl] = compute_wmape(v['Actual_Sales'].values, v[mn].values)
        if vals:
            d_s = f"{vals.get('d', np.nan):>6.1f}%" if 'd' in vals else "    n/a"
            dm_s = f"{vals.get('dm', np.nan):>6.1f}%" if 'dm' in vals else "    n/a"
            fm_s = f"{vals.get('fm', np.nan):>6.1f}%" if 'fm' in vals else "    n/a"
            print(f"  {lag:>4d} | {d_s:>8s} | {dm_s:>8s} | {fm_s:>8s}")

    # ---- Per-(site) and per-(site, lag_bucket) temporal metrics ----
    temporal_site = {}       # {(gs, model_name): wmape}
    temporal_site_lag = {}   # {(gs, lag_bucket, model_name): wmape}

    for gs in gsa_sites:
        sub_gs = test[m2 & (test['gsa_site'] == gs) & (test['Actual_Sales'] > 0)]
        if len(sub_gs) == 0:
            continue
        for model_name in models:
            v = sub_gs[sub_gs[model_name].notna()]
            if len(v) > 0:
                temporal_site[(gs, model_name)] = compute_wmape(
                    v['Actual_Sales'].values, v[model_name].values)
        for lb, (lo, hi) in LAG_BUCKETS.items():
            sub_lb = sub_gs[(sub_gs['Prediction_Lag'] >= lo) &
                            (sub_gs['Prediction_Lag'] <= hi)]
            if len(sub_lb) == 0:
                continue
            for model_name in models:
                v = sub_lb[sub_lb[model_name].notna()]
                if len(v) > 0:
                    temporal_site_lag[(gs, lb, model_name)] = compute_wmape(
                        v['Actual_Sales'].values, v[model_name].values)

    # ---- K-Fold Random CV ----
    print(f"\n  {'=' * 80}")
    print(f"  K-FOLD RANDOM CV ({N_FOLDS} folds) — {customer_name}")
    print(f"  {'=' * 80}")

    np.random.seed(RANDOM_SEED)
    indices = np.arange(len(df))
    np.random.shuffle(indices)
    fold_size = len(df) // N_FOLDS

    fold_metrics = {mn: [] for mn in models}
    fold_site_metrics = {}       # {(gs, model_name): [wmape_per_fold]}
    fold_site_lag_metrics = {}   # {(gs, lb, model_name): [wmape_per_fold]}
    cv_oof_dfs = []              # out-of-fold predictions for CI calibration

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

        # Apply flat curves
        apply_flat_curves(f_train, oo_f, cov_f, bias_f)
        apply_flat_curves(f_test, oo_f, cov_f, bias_f)

        # FC debiasing for this fold
        fold_debias = build_fc_debias_curves(f_train)
        apply_fc_debias(f_train, fold_debias)
        apply_fc_debias(f_test, fold_debias)

        # Multi-vintage FC from TRAINING fold only
        fc_registry_fold = build_forecast_registry(f_train)
        apply_multi_vintage_fc(f_train, cov_f, bias_f, fc_registry_fold, causality='fold')
        apply_multi_vintage_fc(f_test, cov_f, bias_f, fc_registry_fold, causality='fold')

        # Multi-vintage OO from TRAINING fold only
        oo_registry_fold = build_oo_registry(f_train)
        apply_multi_vintage_oo(f_train, oo_f, oo_registry_fold, causality='fold')
        apply_multi_vintage_oo(f_test, oo_f, oo_registry_fold, causality='fold')

        f_train['oo_none'] = np.nan
        f_test['oo_none'] = np.nan

        f_test_months = sorted(f_test['Reference_Month'].unique())

        # Run all models
        for model_name, (oo_c, fc_c) in models.items():
            f_test[model_name] = weighted_avg(f_test, f_train, f_test_months,
                                              gsa_sites, oo_col=oo_c, fc_col=fc_c)

        # Metrics
        ts = f_test[(f_test['Actual_Sales'] > 0)]
        for model_name in models:
            v = ts[ts[model_name].notna()]
            if len(v) > 0:
                wmape = compute_wmape(v['Actual_Sales'].values, v[model_name].values)
                bias = compute_bias(v['Actual_Sales'].values, v[model_name].values)
                fold_metrics[model_name].append({'wmape': wmape, 'bias': bias})

        # Per-(site) and per-(site, lag_bucket) fold metrics
        for gs in gsa_sites:
            sub_gs = ts[ts['gsa_site'] == gs]
            if len(sub_gs) == 0:
                continue
            for model_name in models:
                v = sub_gs[sub_gs[model_name].notna()]
                site_key = (gs, model_name)
                if site_key not in fold_site_metrics:
                    fold_site_metrics[site_key] = []
                if len(v) > 0:
                    fold_site_metrics[site_key].append(
                        compute_wmape(v['Actual_Sales'].values, v[model_name].values))
            for lb, (lo, hi) in LAG_BUCKETS.items():
                sub_lb = sub_gs[(sub_gs['Prediction_Lag'] >= lo) &
                                (sub_gs['Prediction_Lag'] <= hi)]
                if len(sub_lb) == 0:
                    continue
                for model_name in models:
                    v = sub_lb[sub_lb[model_name].notna()]
                    sl_key = (gs, lb, model_name)
                    if sl_key not in fold_site_lag_metrics:
                        fold_site_lag_metrics[sl_key] = []
                    if len(v) > 0:
                        fold_site_lag_metrics[sl_key].append(
                            compute_wmape(v['Actual_Sales'].values,
                                          v[model_name].values))

        # Collect out-of-fold predictions for CI calibration
        oof_cols = (['gsa_site', 'Timeframe', 'Prediction_Lag', 'Actual_Sales']
                    + list(models.keys()))
        cv_oof_dfs.append(f_test[oof_cols].copy())

        # Per-fold summary
        v8d_w = fold_metrics['V8d (OO+FC)'][-1]['wmape'] if fold_metrics['V8d (OO+FC)'] else 0
        fm_w = fold_metrics['FC-only multi'][-1]['wmape'] if fold_metrics['FC-only multi'] else 0
        mm_w = fold_metrics['mOO + mFC'][-1]['wmape'] if fold_metrics['mOO + mFC'] else 0
        print(f"    Fold {fold_i + 1}: V8d={v8d_w:.1f}%  mOO+mFC={mm_w:.1f}%  FC-multi={fm_w:.1f}%")

    # Random CV summary
    print(f"\n  {'Model':>22s} | {'Random CV (mean +/- std)':>26s} | {'Temporal':>10s} | {'Gap':>6s}")
    print(f"  {'-' * 72}")

    cv_results = {}
    for model_name in models:
        wmapes = [m['wmape'] for m in fold_metrics[model_name]]
        biases = [m['bias'] for m in fold_metrics[model_name]]
        t_wmape = temporal_results.get(model_name, float('nan'))
        gap = t_wmape - np.mean(wmapes) if wmapes else float('nan')
        cv_results[model_name] = {
            'cv_mean': np.mean(wmapes) if wmapes else float('nan'),
            'cv_std': np.std(wmapes) if wmapes else float('nan'),
            'temporal': t_wmape,
            'gap': gap,
        }
        if wmapes:
            print(f"  {model_name:>22s} | {np.mean(wmapes):>5.1f}% +/- {np.std(wmapes):>4.1f}% "
                  f"(bias {np.mean(biases):>+5.1f}%) | {t_wmape:>6.1f}% | {gap:>+4.1f}pp")

    # ---- Customer-level OO stationarity (used as fallback) ----
    oo_gap = cv_results.get('V8d (OO+FC)', {}).get('gap', 0)
    fc_gap = cv_results.get('FC-only multi', {}).get('gap', 0)
    oo_nonstationary = abs(oo_gap) > OO_NONSTATIONARY_GAP

    if oo_nonstationary:
        customer_recommended = 'FC-only multi'
    else:
        best_model = min(temporal_results, key=temporal_results.get)
        customer_recommended = best_model

    # ---- Hierarchical model selection ----
    selection_table = build_model_selection_table(
        gsa_sites, models,
        temporal_results, temporal_site, temporal_site_lag,
        cv_results, fold_site_metrics, fold_site_lag_metrics,
        customer_recommended)

    print(f"\n  {'=' * 80}")
    print(f"  HIERARCHICAL MODEL SELECTION — {customer_name}")
    print(f"  {'=' * 80}")
    print(f"  Customer-level OO gap: {oo_gap:+.1f}pp "
          f"({'NON-STATIONARY' if oo_nonstationary else 'stationary'}) "
          f"-> default: {customer_recommended}")

    print(f"\n  {'Site':>25s} | {'Lag':>6s} | {'Model':>15s} | {'Level':>8s} | {'Gap':>8s}")
    print(f"  {'-' * 72}")
    for gs in gsa_sites:
        for lb in LAG_BUCKETS:
            model_name, level, gap = selection_table[(gs, lb)]
            print(f"  {gs:>25s} | {lb:>6s} | {model_name:>15s} | {level:>8s} | {gap:>+6.1f}pp")

    # ---- Apply selection to test set ----
    test['selected'] = np.nan
    for idx in test.index:
        row = test.loc[idx]
        gs = row['gsa_site']
        lb = lag_to_bucket(row['Prediction_Lag'])
        sel_model = selection_table.get((gs, lb), (customer_recommended, '', 0))[0]
        if sel_model in test.columns and not np.isnan(row[sel_model]):
            test.at[idx, 'selected'] = row[sel_model]

    # Compute selected-model metrics
    ts_sel = test[m2 & test['selected'].notna() & (test['Actual_Sales'] > 0)]
    if len(ts_sel) > 0:
        sel_wmape = compute_wmape(ts_sel['Actual_Sales'].values, ts_sel['selected'].values)
        sel_bias = compute_bias(ts_sel['Actual_Sales'].values, ts_sel['selected'].values)
        print(f"\n  Hierarchical-selected WMAPE: {sel_wmape:.1f}%  "
              f"bias: {sel_bias:+.1f}%  N={len(ts_sel)}")

        # Compare against uniform models
        for lbl in ['mOO + mFC', 'FC-only multi']:
            if lbl in temporal_results:
                diff = sel_wmape - temporal_results[lbl]
                print(f"    vs uniform {lbl}: {diff:+.1f}pp")

    # Per-site selected breakdown
    print(f"\n  Per-site WMAPE (hierarchical selected vs uniform models):")
    print(f"  {'Site':>25s} | {'Selected':>10s} | {'mOO+mFC':>10s} | {'FC-multi':>10s} | {'N':>4s}")
    print(f"  {'-' * 70}")
    for gs in gsa_sites:
        sub = test[m2 & (test['gsa_site'] == gs) & (test['Actual_Sales'] > 0)]
        if len(sub) == 0:
            continue
        vals = {}
        for lbl, mn in [('sel', 'selected'), ('mm', 'mOO + mFC'), ('fm', 'FC-only multi')]:
            v = sub[sub[mn].notna()]
            if len(v) > 0:
                vals[lbl] = compute_wmape(v['Actual_Sales'].values, v[mn].values)
        if vals:
            s_s = f"{vals.get('sel', np.nan):>6.1f}%" if 'sel' in vals else "     n/a"
            mm_s = f"{vals.get('mm', np.nan):>6.1f}%" if 'mm' in vals else "     n/a"
            fm_s = f"{vals.get('fm', np.nan):>6.1f}%" if 'fm' in vals else "     n/a"
            print(f"  {gs:>25s} | {s_s:>10s} | {mm_s:>10s} | {fm_s:>10s} | {len(sub):>4d}")

    # Per-fold detail
    print(f"\n  Per-fold WMAPE:")
    print(f"  {'Fold':>6s} | {'V8d':>8s} | {'mOO+mFC':>8s} | {'V8d+mFC':>8s} | {'FC-multi':>8s}")
    print(f"  {'-' * 50}")
    for i in range(N_FOLDS):
        v8d_w = fold_metrics['V8d (OO+FC)'][i]['wmape'] if i < len(fold_metrics['V8d (OO+FC)']) else np.nan
        mm_w = fold_metrics['mOO + mFC'][i]['wmape'] if i < len(fold_metrics['mOO + mFC']) else np.nan
        dm_w = fold_metrics['V8d + multi-FC'][i]['wmape'] if i < len(fold_metrics['V8d + multi-FC']) else np.nan
        fm_w = fold_metrics['FC-only multi'][i]['wmape'] if i < len(fold_metrics['FC-only multi']) else np.nan
        print(f"  {'F' + str(i + 1):>6s} | {v8d_w:>6.1f}% | {mm_w:>6.1f}% | {dm_w:>6.1f}% | {fm_w:>6.1f}%")

    # ---- 80% Prediction Intervals (calibrated from CV out-of-fold data) ----
    print(f"\n  {'=' * 80}")
    print(f"  80% PREDICTION INTERVALS — {customer_name}")
    print(f"  {'=' * 80}")

    # Calibrate from temporal test predictions (same source as production:
    # recent actuals vs predictions, specific to site/tf/lag)
    ci_data = test[m2 & test['selected'].notna() & (test['Actual_Sales'] > 0)]
    print(f"  Calibration: {len(ci_data)} temporal test predictions (months 2+)")

    ci_lookup = build_prediction_intervals(
        ci_data['Actual_Sales'].values, ci_data['selected'].values,
        ci_data['gsa_site'].values, ci_data['Timeframe'].values,
        ci_data['Prediction_Lag'].values)

    # Count keys at each level
    n_stl = sum(1 for k in ci_lookup if isinstance(k, tuple) and len(k) == 3
                and k[0] != 'FB_lag' and k[0] != 'FB_site')
    n_sl = sum(1 for k in ci_lookup if isinstance(k, tuple) and len(k) == 3
               and k[0] == 'FB_lag')
    n_s = sum(1 for k in ci_lookup if isinstance(k, tuple) and len(k) == 2
              and k[0] == 'FB_site')
    print(f"  CI curves: {n_stl} (site,tf,lag) + {n_sl} (site,lag) + "
          f"{n_s} (site) + 1 global fallback")

    # Apply to temporal test set
    test['ci_lower'] = np.nan
    test['ci_upper'] = np.nan
    for idx in test.index:
        pred = test.at[idx, 'selected']
        if pd.isna(pred) or pred <= 0:
            continue
        q_lo, q_hi = lookup_ci(ci_lookup, test.at[idx, 'gsa_site'],
                               test.at[idx, 'Timeframe'],
                               test.at[idx, 'Prediction_Lag'])
        test.at[idx, 'ci_lower'] = pred * q_lo
        test.at[idx, 'ci_upper'] = pred * q_hi

    # Coverage and width statistics
    ci_rows = test[m2 & test['ci_lower'].notna() & (test['Actual_Sales'] > 0)]
    if len(ci_rows) > 0:
        in_ci = ((ci_rows['Actual_Sales'] >= ci_rows['ci_lower']) &
                 (ci_rows['Actual_Sales'] <= ci_rows['ci_upper']))
        coverage = in_ci.mean() * 100
        rel_width = ((ci_rows['ci_upper'] - ci_rows['ci_lower']) /
                     ci_rows['selected']).mean() * 100
        print(f"\n  Overall coverage: {coverage:.1f}% (target: {CI_LEVEL * 100:.0f}%)")
        print(f"  Average relative CI width: {rel_width:.0f}% of point prediction")

        # Per-site breakdown
        print(f"\n  {'Site':>25s} | {'Coverage':>9s} | {'Avg Width':>10s} | "
              f"{'Median CI':>18s} | {'N':>4s}")
        print(f"  {'-' * 78}")
        for gs in gsa_sites:
            sub = ci_rows[ci_rows['gsa_site'] == gs]
            if len(sub) == 0:
                continue
            s_in = ((sub['Actual_Sales'] >= sub['ci_lower']) &
                    (sub['Actual_Sales'] <= sub['ci_upper']))
            s_cov = s_in.mean() * 100
            s_wid = ((sub['ci_upper'] - sub['ci_lower']) /
                     sub['selected']).mean() * 100
            med_lo = sub['ci_lower'].median()
            med_hi = sub['ci_upper'].median()
            print(f"  {gs:>25s} | {s_cov:>7.0f}% | {s_wid:>8.0f}% | "
                  f"[{med_lo:>7,.0f} – {med_hi:>7,.0f}] | {len(sub):>4d}")

        # Per-lag breakdown
        print(f"\n  {'Lag':>4s} | {'Coverage':>9s} | {'Avg Width':>10s} | {'N':>4s}")
        print(f"  {'-' * 35}")
        for lag in sorted(ci_rows['Prediction_Lag'].unique()):
            sub = ci_rows[ci_rows['Prediction_Lag'] == lag]
            s_in = ((sub['Actual_Sales'] >= sub['ci_lower']) &
                    (sub['Actual_Sales'] <= sub['ci_upper']))
            s_cov = s_in.mean() * 100
            s_wid = ((sub['ci_upper'] - sub['ci_lower']) /
                     sub['selected']).mean() * 100
            print(f"  {lag:>4d} | {s_cov:>7.0f}% | {s_wid:>8.0f}% | {len(sub):>4d}")

        # Per-timeframe breakdown
        print(f"\n  {'Timeframe':>12s} | {'Coverage':>9s} | {'Avg Width':>10s} | {'N':>4s}")
        print(f"  {'-' * 43}")
        for tf in sorted(ci_rows['Timeframe'].unique()):
            sub = ci_rows[ci_rows['Timeframe'] == tf]
            s_in = ((sub['Actual_Sales'] >= sub['ci_lower']) &
                    (sub['Actual_Sales'] <= sub['ci_upper']))
            s_cov = s_in.mean() * 100
            s_wid = ((sub['ci_upper'] - sub['ci_lower']) /
                     sub['selected']).mean() * 100
            print(f"  {str(tf):>12s} | {s_cov:>7.0f}% | {s_wid:>8.0f}% | {len(sub):>4d}")

    return {
        'customer': customer_name,
        'n_rows': len(df),
        'n_sites': len(gsa_sites),
        'sites': gsa_sites,
        'fc_pct': fc_pct,
        'oo_pct': oo_pct,
        'temporal': temporal_results,
        'cv': cv_results,
        'oo_nonstationary': oo_nonstationary,
        'customer_recommended': customer_recommended,
        'selection_table': selection_table,
        'ci_lookup': ci_lookup,
    }


# ============================================================
# MAIN: Load data, loop over customers, print cross-customer summary
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Unified Aging Curve Pipeline')
    parser.add_argument('--file', type=str, default=None,
                        help='Path to unified CSV data file')
    parser.add_argument('--customer', type=str, default=None,
                        help='Run for a single customer (GSA name). '
                             'If omitted, runs all customers.')
    args = parser.parse_args()

    # ---- Load data ----
    if args.file:
        filepath = args.file
    else:
        # Auto-detect: look for CSV files in current directory
        import glob
        csvs = sorted(glob.glob('*.csv'))
        if len(csvs) == 1:
            filepath = csvs[0]
            print(f"Auto-detected data file: {filepath}")
        elif len(csvs) > 1:
            print(f"Multiple CSV files found: {csvs}")
            print("Please specify one with --file")
            return
        else:
            print("No CSV files found. Please specify with --file")
            return

    print(f"Loading data from: {filepath}")
    df = load_and_preprocess(filepath)
    print(f"Loaded {len(df)} rows, {df['GSA'].nunique()} customers")

    # ---- Determine which customers to run ----
    if args.customer:
        customers = [args.customer]
        if args.customer not in df['GSA'].unique():
            print(f"Customer '{args.customer}' not found. "
                  f"Available: {sorted(df['GSA'].unique())}")
            return
    else:
        customers = sorted(df['GSA'].unique())

    print(f"Running pipeline for: {customers}")

    # ---- Run per-customer pipelines ----
    all_results = []
    for cust in customers:
        cust_df = df[df['GSA'] == cust].copy()
        result = run_customer_pipeline(cust_df, cust)
        all_results.append(result)

    # ---- Cross-customer summary ----
    if len(all_results) > 1:
        print(f"\n\n{'=' * 100}")
        print(f"CROSS-CUSTOMER SUMMARY")
        print(f"{'=' * 100}")

        print(f"\n  {'Customer':>20s} | {'Rows':>6s} | {'Sites':>5s} | {'FC%':>4s} | "
              f"{'Best Temporal':>14s} | {'OO ok?':>6s} | {'Default':>15s} | {'Cells':>5s}")
        print(f"  {'-' * 100}")

        for r in all_results:
            best_temp_name = min(r['temporal'], key=r['temporal'].get) if r['temporal'] else 'n/a'
            best_temp_val = r['temporal'].get(best_temp_name, float('nan'))
            oo_ok = 'YES' if not r['oo_nonstationary'] else 'NO'
            st = r.get('selection_table', {})
            n_cells = len(st)
            n_oo = sum(1 for m, _, _ in st.values() if m == 'mOO + mFC')

            print(f"  {r['customer']:>20s} | {r['n_rows']:>6d} | {r['n_sites']:>5d} | "
                  f"{r['fc_pct']:>3.0f}% | {best_temp_val:>5.1f}% ({best_temp_name[:8]:>8s}) | "
                  f"{oo_ok:>6s} | {r['customer_recommended']:>15s} | "
                  f"{n_oo}/{n_cells}")

    print(f"\n\nDone!")


if __name__ == '__main__':
    main()
