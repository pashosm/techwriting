"""
OO Power-CDF Normalization Experiment
======================================
Tests whether normalizing OO_ratio by progress^alpha (site-specific power law)
produces a more stationary OO signal than flat curves (V8d) or uniform-CDF
correction (V8k).

Core idea:
    progress = max(ε, 1 - lag / (LT + OE))
    OO_ratio_norm = OO_ratio / progress^alpha
    implied_actual = OO / (curve_norm × progress^alpha)

Alpha is estimated per-site and per-(site, timeframe) from training data.

Usage:
    python oo_normalization.py                                         # auto-detect CSV
    python oo_normalization.py --file "Dummy Training Data Customer 1 Try 2 11-Feb-2026.csv"
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
MIN_OBS_ALPHA = 10       # Minimum observations to estimate alpha
ALPHA_MIN = 0.3          # Floor for alpha
ALPHA_MAX = 4.0          # Cap for alpha
PROGRESS_EPSILON = 0.02  # Floor for progress to avoid log(0)
MIN_SITE_ROWS = 30
VINTAGE_RECENCY_POWER = 1.5
N_FOLDS = 5
RANDOM_SEED = 42

# V8k constants for comparison
V8K_MIN_PROGRESS = 0.05
V8K_MAX_CORRECTION = 3.0
V8K_MIN_CORRECTION = 0.2

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
    df['lt_oe'] = df['Avg_Weighted_Lead_Time'] + df['Avg_Weighted_Order_Earliness']

    df['oo_ratio'] = np.where(
        df['Actual_Sales'] > 0,
        df['Open_Orders'] / df['Actual_Sales'], np.nan)
    df['fc_coverage'] = np.where(
        df['Actual_Sales'] > 0,
        df['Covered_Orders'] / df['Actual_Sales'], np.nan)
    df['fc_bias'] = np.where(
        df['Covered_Orders'] > 0,
        df['Forecast_Value'] / df['Covered_Orders'], np.nan)

    # Progress through ordering window
    df['progress'] = np.where(
        df['lt_oe'] > 0.5,
        np.maximum(PROGRESS_EPSILON, 1.0 - df['Prediction_Lag'] / df['lt_oe']),
        np.nan)

    return df


# ============================================================
# ALPHA ESTIMATION
# ============================================================

def estimate_alpha(train_df, groupby_cols, min_obs=MIN_OBS_ALPHA):
    """Estimate power-law exponent: OO_ratio ~ progress^alpha.

    Uses weighted log-log OLS: log(OO_ratio) = alpha * log(progress) + c

    Parameters
    ----------
    train_df : DataFrame with oo_ratio, progress, Reference_Month columns
    groupby_cols : list of columns to group by (e.g., ['gsa_site'] or
                   ['gsa_site', 'Timeframe'])
    min_obs : minimum observations per group

    Returns
    -------
    alphas : dict {group_key: alpha}
    alpha_meta : dict {group_key: {alpha, intercept, r2, n_obs, ...}}
    """
    alphas = {}
    alpha_meta = {}

    for group_key, grp in train_df.groupby(groupby_cols):
        valid = grp[
            grp['oo_ratio'].notna() &
            (grp['oo_ratio'] > 0) & (grp['oo_ratio'] < 10) &
            grp['progress'].notna() &
            (grp['progress'] > PROGRESS_EPSILON)
        ].copy()

        if len(valid) < min_obs:
            continue

        log_prog = np.log(valid['progress'].values)
        log_ratio = np.log(valid['oo_ratio'].values)

        # Recency weighting
        days = (valid['Reference_Month'] - valid['Reference_Month'].min()).dt.days / 365
        wts = (1 + days.values) ** RECENCY_POWER

        # Weighted OLS: log_ratio = c + alpha * log_prog
        X = np.column_stack([np.ones(len(log_prog)), log_prog])
        W = np.sqrt(wts)
        Xw = X * W[:, None]
        yw = log_ratio * W

        try:
            coef = np.linalg.lstsq(Xw, yw, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue

        intercept, alpha = coef[0], coef[1]
        alpha = np.clip(alpha, ALPHA_MIN, ALPHA_MAX)

        # R² in log space
        pred = X @ coef
        ss_res = np.sum(wts * (log_ratio - pred) ** 2)
        ss_tot = np.sum(wts * (log_ratio - np.average(log_ratio, weights=wts)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Convert group_key to consistent format
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        alphas[group_key] = alpha
        alpha_meta[group_key] = {
            'alpha': alpha,
            'intercept': intercept,
            'r2': r2,
            'n_obs': len(valid),
            'mean_progress': valid['progress'].mean(),
            'mean_oo_ratio': valid['oo_ratio'].mean(),
        }

    return alphas, alpha_meta


def get_alpha(alphas, gs, tf=None):
    """Look up alpha with fallback. Try (gs, tf) first, then (gs,)."""
    if tf is not None:
        key = (gs, tf)
        if key in alphas:
            return alphas[key]
    key = (gs,)
    if key in alphas:
        return alphas[key]
    # Global fallback: median of all alphas
    if alphas:
        return np.median(list(alphas.values()))
    return 1.0  # ultimate fallback: uniform


# ============================================================
# CURVE BUILDING
# ============================================================

def build_flat_curves(train_df, ratio_col, start=None):
    """Build recency-weighted flat aging curves (same as pipeline.py)."""
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


def build_normalized_curves(train_df, alphas, alpha_mode='site', start=None):
    """Build curves on OO_ratio / progress^alpha (the normalized ratio).

    Parameters
    ----------
    train_df : DataFrame with oo_ratio, progress, gsa_site, Timeframe,
               Prediction_Lag, Reference_Month
    alphas : dict from estimate_alpha()
    alpha_mode : 'site' (keys are (gs,)) or 'site_tf' (keys are (gs, tf))
    start : optional date filter

    Returns
    -------
    norm_curves : dict {(gs, tf, lag): avg_normalized_ratio}
    """
    td = train_df if start is None else train_df[train_df['Reference_Month'] >= start]
    norm_curves = {}

    for (gs, tf, lag), grp in td.groupby(['gsa_site', 'Timeframe', 'Prediction_Lag']):
        valid = grp[
            grp['oo_ratio'].notna() & (grp['oo_ratio'] > 0) & (grp['oo_ratio'] < 10) &
            grp['progress'].notna() & (grp['progress'] > PROGRESS_EPSILON)
        ]
        if len(valid) < MIN_OBS_FLAT:
            continue

        # Look up alpha
        if alpha_mode == 'site_tf':
            alpha = get_alpha(alphas, gs, tf)
        else:
            alpha = get_alpha(alphas, gs)

        # Normalized ratio
        norm_ratio = valid['oo_ratio'].values / (valid['progress'].values ** alpha)

        # Filter outliers
        mask = (norm_ratio > 0) & (norm_ratio < 50)
        if mask.sum() < MIN_OBS_FLAT:
            continue

        days = (valid.loc[mask, 'Reference_Month'] if isinstance(mask, pd.Series)
                else valid.iloc[mask]['Reference_Month'])
        # Use boolean indexing consistently
        valid_masked = valid[mask] if isinstance(mask, np.ndarray) else valid.loc[mask]
        days_vals = (valid_masked['Reference_Month'] -
                     valid_masked['Reference_Month'].min()).dt.days / 365
        wts = (1 + days_vals.values) ** RECENCY_POWER
        norm_curves[(gs, tf, lag)] = np.average(norm_ratio[mask], weights=wts)

    # Fallback: pooled across timeframes
    for (gs, lag), grp in td.groupby(['gsa_site', 'Prediction_Lag']):
        valid = grp[
            grp['oo_ratio'].notna() & (grp['oo_ratio'] > 0) & (grp['oo_ratio'] < 10) &
            grp['progress'].notna() & (grp['progress'] > PROGRESS_EPSILON)
        ]
        if len(valid) < MIN_OBS_FLAT:
            continue

        # For fallback, use site-level alpha
        alpha = get_alpha(alphas, gs)
        norm_ratio = valid['oo_ratio'].values / (valid['progress'].values ** alpha)
        mask = (norm_ratio > 0) & (norm_ratio < 50)
        if mask.sum() < MIN_OBS_FLAT:
            continue

        valid_masked = valid[mask] if isinstance(mask, np.ndarray) else valid.loc[mask]
        days_vals = (valid_masked['Reference_Month'] -
                     valid_masked['Reference_Month'].min()).dt.days / 365
        wts = (1 + days_vals.values) ** RECENCY_POWER
        norm_curves[('FB', gs, lag)] = np.average(norm_ratio[mask], weights=wts)

    return norm_curves


def apply_normalized_curves(dset, norm_curves, alphas, flat_fallback,
                            alpha_mode='site', out_col='oo_implied_norm'):
    """Apply normalized OO curves to produce implied actuals.

    implied = OO / (curve_norm × progress^alpha)
    Falls back to flat curve when progress is unavailable.
    """
    oo_impl = np.full(len(dset), np.nan)
    n_norm = 0
    n_flat = 0

    for i, (idx, row) in enumerate(dset.iterrows()):
        gs, tf, lag = row['gsa_site'], row['Timeframe'], row['Prediction_Lag']
        progress = row.get('progress', np.nan)

        if not np.isnan(progress) and progress > PROGRESS_EPSILON:
            if alpha_mode == 'site_tf':
                alpha = get_alpha(alphas, gs, tf)
            else:
                alpha = get_alpha(alphas, gs)

            for key in [(gs, tf, lag), ('FB', gs, lag)]:
                if key in norm_curves and norm_curves[key] > 0.001:
                    oo_impl[i] = row['Open_Orders'] / (
                        norm_curves[key] * (progress ** alpha))
                    n_norm += 1
                    break
            else:
                # Normalized curve missing — try flat fallback
                for key in [(gs, tf, lag), ('FB', gs, lag)]:
                    if key in flat_fallback and flat_fallback[key] > 0.001:
                        oo_impl[i] = row['Open_Orders'] / flat_fallback[key]
                        n_flat += 1
                        break
        else:
            # No progress data — use flat fallback
            for key in [(gs, tf, lag), ('FB', gs, lag)]:
                if key in flat_fallback and flat_fallback[key] > 0.001:
                    oo_impl[i] = row['Open_Orders'] / flat_fallback[key]
                    n_flat += 1
                    break

    dset[out_col] = np.maximum(oo_impl, 0)
    return n_norm, n_flat


def apply_flat_curves(dset, oo_f, cov_f, bias_f):
    """Apply flat OO and FC curves (same as pipeline.py)."""
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


# ---- V8k: Progress correction for comparison ----

def build_ltoe_reference(train_df):
    """Compute training-era LT+OE per cell."""
    ref_ltoe = {}
    for (gs, tf, lag), grp in train_df.groupby(['gsa_site', 'Timeframe', 'Prediction_Lag']):
        valid = grp[grp['lt_oe'] > 0.5]
        if len(valid) >= MIN_OBS_FLAT:
            days = (valid['Reference_Month'] - valid['Reference_Month'].min()).dt.days / 365
            wts = (1 + days.values) ** RECENCY_POWER
            ref_ltoe[(gs, tf, lag)] = np.average(valid['lt_oe'], weights=wts)
    for (gs, lag), grp in train_df.groupby(['gsa_site', 'Prediction_Lag']):
        valid = grp[grp['lt_oe'] > 0.5]
        if len(valid) >= MIN_OBS_FLAT:
            days = (valid['Reference_Month'] - valid['Reference_Month'].min()).dt.days / 365
            wts = (1 + days.values) ** RECENCY_POWER
            ref_ltoe[('FB', gs, lag)] = np.average(valid['lt_oe'], weights=wts)
    return ref_ltoe


def apply_v8k_correction(dset, oo_flat, ref_ltoe, out_col='oo_implied_v8k'):
    """V8k: progress-based multiplicative correction with uniform CDF (alpha=1)."""
    oo_impl = np.full(len(dset), np.nan)

    for i, (idx, row) in enumerate(dset.iterrows()):
        gs, tf, lag = row['gsa_site'], row['Timeframe'], row['Prediction_Lag']
        lt_oe_cur = row['lt_oe']

        ref = None
        for key in [(gs, tf, lag), ('FB', gs, lag)]:
            if key in ref_ltoe:
                ref = ref_ltoe[key]
                break

        flat_val = None
        for k in [(gs, tf, lag), ('FB', gs, lag)]:
            if k in oo_flat and oo_flat[k] > 0.001:
                flat_val = oo_flat[k]
                break

        if flat_val is not None and ref is not None and ref > 0.5 and lt_oe_cur > 0.5:
            prog_ref = max(V8K_MIN_PROGRESS, 1.0 - lag / ref)
            prog_cur = max(V8K_MIN_PROGRESS, 1.0 - lag / lt_oe_cur)
            correction = np.clip(prog_cur / prog_ref, V8K_MIN_CORRECTION, V8K_MAX_CORRECTION)
            corrected_ratio = flat_val * correction
            oo_impl[i] = row['Open_Orders'] / corrected_ratio
        elif flat_val is not None:
            oo_impl[i] = row['Open_Orders'] / flat_val

    dset[out_col] = np.maximum(oo_impl, 0)


# ---- Damped Power-CDF Correction ----

DAMPED_PROGRESS_THRESHOLD = 0.35  # Below this progress, blend toward flat

def apply_damped_power_correction(dset, oo_flat, ref_ltoe, alphas,
                                  alpha_mode='site',
                                  out_col='oo_implied_damped'):
    """Apply power-CDF correction to flat curves with progress-dependent damping.

    At high progress (short lags): full power-CDF correction using site-specific alpha
    At low progress (long lags): smoothly fade back to flat curve

    The correction is:
        correction = (progress_cur / progress_ref)^alpha
        damping = smoothstep(progress_cur, threshold=0.35)
        effective_correction = 1 + damping * (correction - 1)

    This preserves the flat curve where it's reliable (high lags) and applies
    the LT+OE correction where progress is well-defined (short lags).
    """
    oo_impl = np.full(len(dset), np.nan)
    stats = {'damped': 0, 'full': 0, 'flat_only': 0}

    for i, (idx, row) in enumerate(dset.iterrows()):
        gs, tf, lag = row['gsa_site'], row['Timeframe'], row['Prediction_Lag']
        lt_oe_cur = row['lt_oe']

        ref = None
        for key in [(gs, tf, lag), ('FB', gs, lag)]:
            if key in ref_ltoe:
                ref = ref_ltoe[key]
                break

        flat_val = None
        for k in [(gs, tf, lag), ('FB', gs, lag)]:
            if k in oo_flat and oo_flat[k] > 0.001:
                flat_val = oo_flat[k]
                break

        if flat_val is None:
            continue

        if ref is not None and ref > 0.5 and lt_oe_cur > 0.5:
            prog_ref = max(PROGRESS_EPSILON, 1.0 - lag / ref)
            prog_cur = max(PROGRESS_EPSILON, 1.0 - lag / lt_oe_cur)

            # Site-specific alpha
            if alpha_mode == 'site_tf':
                alpha = get_alpha(alphas, gs, tf)
            else:
                alpha = get_alpha(alphas, gs)

            # Power-CDF correction
            ratio = prog_cur / prog_ref
            ratio = np.clip(ratio, V8K_MIN_CORRECTION, V8K_MAX_CORRECTION)
            correction = ratio ** alpha

            # Damping: smoothstep based on current progress
            # At prog_cur >= threshold: full correction (damping = 1)
            # At prog_cur < threshold: linear fade to 0
            if prog_cur >= DAMPED_PROGRESS_THRESHOLD:
                damping = 1.0
                stats['full'] += 1
            else:
                damping = prog_cur / DAMPED_PROGRESS_THRESHOLD
                stats['damped'] += 1

            effective = 1.0 + damping * (correction - 1.0)
            corrected_ratio = flat_val * effective
            oo_impl[i] = row['Open_Orders'] / corrected_ratio
        else:
            oo_impl[i] = row['Open_Orders'] / flat_val
            stats['flat_only'] += 1

    dset[out_col] = np.maximum(oo_impl, 0)
    return stats


# ---- Multi-vintage FC (same as pipeline.py) ----

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
            fc_val = v['fc_value']
            if fc_val <= 0:
                continue
            vgs, vtf, vlag = v['gsa_site'], v['timeframe'], v['lag']
            for key in [(vgs, vtf, vlag), ('FB', vgs, vlag)]:
                cf = cov_f.get(key)
                bf = bias_f.get(key)
                if cf and bf and cf > 0.001 and bf > 0.001:
                    est = fc_val / bf / cf
                    valid_vintages.append((max(est, 0), vlag))
                    break
        if valid_vintages:
            total_w, total_val = 0.0, 0.0
            for est, orig_lag in valid_vintages:
                w = 1.0 / (orig_lag ** VINTAGE_RECENCY_POWER)
                total_w += w
                total_val += w * est
            fc_impl_multi[i] = max(total_val / total_w, 0) if total_w > 0 else np.nan
            n_vintages[i] = len(valid_vintages)
            best_lag[i] = min(vl for _, vl in valid_vintages)

    dset['fc_implied_multi'] = fc_impl_multi
    dset['n_vintages_used'] = n_vintages
    dset['best_vintage_lag'] = best_lag


# ---- Weighted Average Blending ----

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


# ---- Evaluation Metrics ----

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
# MAIN PIPELINE
# ============================================================

def run_experiment(df, customer_name):
    print(f"\n{'#' * 100}")
    print(f"# OO NORMALIZATION EXPERIMENT — {customer_name}")
    print(f"{'#' * 100}")

    # ---- Filter sites ----
    site_counts = df.groupby('gsa_site').size()
    valid_sites = sorted(site_counts[site_counts >= MIN_SITE_ROWS].index.tolist())
    df = df[df['gsa_site'].isin(valid_sites)].copy()
    gsa_sites = sorted(df['gsa_site'].unique())

    print(f"\n  Data: {len(df)} rows, {len(gsa_sites)} sites")
    print(f"  Sites: {gsa_sites}")

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

    # ============================================================
    # STEP 1: Estimate alpha
    # ============================================================
    print(f"\n  {'=' * 80}")
    print(f"  ALPHA ESTIMATION")
    print(f"  {'=' * 80}")

    # Per-site alpha
    alpha_site, alpha_site_meta = estimate_alpha(train, ['gsa_site'])
    print(f"\n  Per-site alphas ({len(alpha_site)} sites):")
    print(f"  {'Site':>25s} | {'Alpha':>6s} | {'R²':>6s} | {'N':>5s} | {'Mean prog':>9s}")
    print(f"  {'-' * 60}")
    for key in sorted(alpha_site_meta.keys()):
        m = alpha_site_meta[key]
        gs = key[0]
        print(f"  {gs:>25s} | {m['alpha']:>6.2f} | {m['r2']:>6.3f} | "
              f"{m['n_obs']:>5d} | {m['mean_progress']:>9.3f}")

    # Per-(site, timeframe) alpha
    alpha_site_tf, alpha_site_tf_meta = estimate_alpha(
        train, ['gsa_site', 'Timeframe'])
    print(f"\n  Per-(site, timeframe) alphas ({len(alpha_site_tf)} groups):")
    print(f"  {'Site':>25s} | {'TF':>3s} | {'Alpha':>6s} | {'R²':>6s} | {'N':>5s}")
    print(f"  {'-' * 55}")
    for key in sorted(alpha_site_tf_meta.keys()):
        m = alpha_site_tf_meta[key]
        gs, tf = key[0], key[1]
        print(f"  {gs:>25s} | {tf:>3d} | {m['alpha']:>6.2f} | {m['r2']:>6.3f} | "
              f"{m['n_obs']:>5d}")

    # ============================================================
    # STEP 2: Build curves
    # ============================================================
    print(f"\n  {'=' * 80}")
    print(f"  BUILDING CURVES")
    print(f"  {'=' * 80}")

    # Flat curves (V8d baseline)
    oo_flat = build_flat_curves(train, 'oo_ratio', start='2023-01-01')
    cov_flat = build_flat_curves(train, 'fc_coverage', start='2023-01-01')
    bias_flat = build_flat_curves(train, 'fc_bias', start='2023-01-01')
    print(f"  Flat OO curves: {len(oo_flat)}")
    print(f"  Flat FC coverage: {len(cov_flat)}  |  FC bias: {len(bias_flat)}")

    # Normalized curves (per-site alpha)
    norm_site = build_normalized_curves(train, alpha_site, alpha_mode='site',
                                        start='2023-01-01')
    print(f"  Normalized OO curves (per-site alpha): {len(norm_site)}")

    # Normalized curves (per-site-tf alpha)
    norm_site_tf = build_normalized_curves(train, alpha_site_tf, alpha_mode='site_tf',
                                           start='2023-01-01')
    print(f"  Normalized OO curves (per-site-tf alpha): {len(norm_site_tf)}")

    # LT+OE reference for V8k
    ref_ltoe = build_ltoe_reference(train)

    # ============================================================
    # STEP 3: Apply curves
    # ============================================================
    print(f"\n  {'=' * 80}")
    print(f"  APPLYING CURVES")
    print(f"  {'=' * 80}")

    # Apply flat curves
    apply_flat_curves(train, oo_flat, cov_flat, bias_flat)
    apply_flat_curves(test, oo_flat, cov_flat, bias_flat)

    # Apply V8k correction
    apply_v8k_correction(train, oo_flat, ref_ltoe)
    apply_v8k_correction(test, oo_flat, ref_ltoe)

    # Apply normalized (per-site alpha)
    n_norm_tr, n_flat_tr = apply_normalized_curves(
        train, norm_site, alpha_site, oo_flat, alpha_mode='site',
        out_col='oo_implied_norm_site')
    n_norm_te, n_flat_te = apply_normalized_curves(
        test, norm_site, alpha_site, oo_flat, alpha_mode='site',
        out_col='oo_implied_norm_site')
    print(f"  Norm (per-site): test coverage = {n_norm_te} normalized + "
          f"{n_flat_te} flat fallback")

    # Apply normalized (per-site-tf alpha)
    n_norm_tr2, n_flat_tr2 = apply_normalized_curves(
        train, norm_site_tf, alpha_site_tf, oo_flat, alpha_mode='site_tf',
        out_col='oo_implied_norm_site_tf')
    n_norm_te2, n_flat_te2 = apply_normalized_curves(
        test, norm_site_tf, alpha_site_tf, oo_flat, alpha_mode='site_tf',
        out_col='oo_implied_norm_site_tf')
    print(f"  Norm (per-site-tf): test coverage = {n_norm_te2} normalized + "
          f"{n_flat_te2} flat fallback")

    # Apply damped power-CDF correction (per-site alpha)
    damp_stats_tr = apply_damped_power_correction(
        train, oo_flat, ref_ltoe, alpha_site, alpha_mode='site',
        out_col='oo_implied_damped_s')
    damp_stats_te = apply_damped_power_correction(
        test, oo_flat, ref_ltoe, alpha_site, alpha_mode='site',
        out_col='oo_implied_damped_s')
    print(f"  Damped (per-site): full={damp_stats_te['full']}, "
          f"damped={damp_stats_te['damped']}, flat_only={damp_stats_te['flat_only']}")

    # Apply damped power-CDF correction (per-site-tf alpha)
    apply_damped_power_correction(
        train, oo_flat, ref_ltoe, alpha_site_tf, alpha_mode='site_tf',
        out_col='oo_implied_damped_st')
    damp_stats_te2 = apply_damped_power_correction(
        test, oo_flat, ref_ltoe, alpha_site_tf, alpha_mode='site_tf',
        out_col='oo_implied_damped_st')
    print(f"  Damped (per-site-tf): full={damp_stats_te2['full']}, "
          f"damped={damp_stats_te2['damped']}, flat_only={damp_stats_te2['flat_only']}")

    # Multi-vintage FC
    fc_registry = build_forecast_registry(df)
    apply_multi_vintage_fc(train, cov_flat, bias_flat, fc_registry, 'temporal')
    apply_multi_vintage_fc(test, cov_flat, bias_flat, fc_registry, 'temporal')
    n_multi = test['fc_implied_multi'].notna().sum()
    print(f"  Multi-vintage FC coverage: {n_multi}/{len(test)} "
          f"({100 * n_multi / len(test):.0f}%)")

    # oo_none column for FC-only models
    train['oo_none'] = np.nan
    test['oo_none'] = np.nan

    # ============================================================
    # STEP 4: Signal quality (OO implied vs actual)
    # ============================================================
    print(f"\n  {'=' * 80}")
    print(f"  OO SIGNAL QUALITY (implied actual vs actual)")
    print(f"  {'=' * 80}")

    print(f"\n  {'Signal':>30s} | {'WMAPE':>7s} | {'Bias':>7s} | {'N':>5s}")
    print(f"  {'-' * 55}")
    for label, col in [('OO flat (V8d)', 'oo_implied_flat'),
                        ('OO V8k (alpha=1)', 'oo_implied_v8k'),
                        ('OO norm (per-site)', 'oo_implied_norm_site'),
                        ('OO norm (per-site-tf)', 'oo_implied_norm_site_tf'),
                        ('OO damped (per-site)', 'oo_implied_damped_s'),
                        ('OO damped (per-site-tf)', 'oo_implied_damped_st')]:
        v = test[test[col].notna() & (test[col] > 0) & (test['Actual_Sales'] > 0)]
        if len(v) > 0:
            w = compute_wmape(v['Actual_Sales'].values, v[col].values)
            b = compute_bias(v['Actual_Sales'].values, v[col].values)
            print(f"  {label:>30s} | {w:>5.1f}% | {b:>+5.1f}% | {len(v):>5d}")

    # Per-site signal quality
    print(f"\n  Per-site OO signal WMAPE:")
    print(f"  {'Site':>25s} | {'Flat':>8s} | {'V8k':>8s} | "
          f"{'Norm-S':>8s} | {'DampS':>8s} | {'DampST':>8s}")
    print(f"  {'-' * 75}")
    for gs in gsa_sites:
        sub = test[(test['gsa_site'] == gs) & (test['Actual_Sales'] > 0)]
        vals = {}
        for lbl, col in [('flat', 'oo_implied_flat'), ('v8k', 'oo_implied_v8k'),
                          ('ns', 'oo_implied_norm_site'),
                          ('ds', 'oo_implied_damped_s'),
                          ('dst', 'oo_implied_damped_st')]:
            v = sub[sub[col].notna() & (sub[col] > 0)]
            if len(v) > 0:
                vals[lbl] = compute_wmape(v['Actual_Sales'].values, v[col].values)
        if vals:
            site = gs.split('|')[1] if '|' in gs else gs
            f_s = f"{vals.get('flat', np.nan):>6.1f}%" if 'flat' in vals else "    n/a"
            k_s = f"{vals.get('v8k', np.nan):>6.1f}%" if 'v8k' in vals else "    n/a"
            ns_s = f"{vals.get('ns', np.nan):>6.1f}%" if 'ns' in vals else "    n/a"
            ds_s = f"{vals.get('ds', np.nan):>6.1f}%" if 'ds' in vals else "    n/a"
            dst_s = f"{vals.get('dst', np.nan):>6.1f}%" if 'dst' in vals else "    n/a"
            print(f"  {site:>25s} | {f_s:>8s} | {k_s:>8s} | {ns_s:>8s} | {ds_s:>8s} | {dst_s:>8s}")

    # Per-lag signal quality
    print(f"\n  Per-lag OO signal WMAPE:")
    print(f"  {'Lag':>4s} | {'Flat':>8s} | {'V8k':>8s} | "
          f"{'Norm-S':>8s} | {'DampS':>8s} | {'DampST':>8s}")
    print(f"  {'-' * 55}")
    for lag in sorted(test['Prediction_Lag'].unique()):
        sub = test[(test['Prediction_Lag'] == lag) & (test['Actual_Sales'] > 0)]
        vals = {}
        for lbl, col in [('flat', 'oo_implied_flat'), ('v8k', 'oo_implied_v8k'),
                          ('ns', 'oo_implied_norm_site'),
                          ('ds', 'oo_implied_damped_s'),
                          ('dst', 'oo_implied_damped_st')]:
            v = sub[sub[col].notna() & (sub[col] > 0)]
            if len(v) > 0:
                vals[lbl] = compute_wmape(v['Actual_Sales'].values, v[col].values)
        if vals:
            f_s = f"{vals.get('flat', np.nan):>6.1f}%" if 'flat' in vals else "    n/a"
            k_s = f"{vals.get('v8k', np.nan):>6.1f}%" if 'v8k' in vals else "    n/a"
            ns_s = f"{vals.get('ns', np.nan):>6.1f}%" if 'ns' in vals else "    n/a"
            ds_s = f"{vals.get('ds', np.nan):>6.1f}%" if 'ds' in vals else "    n/a"
            dst_s = f"{vals.get('dst', np.nan):>6.1f}%" if 'dst' in vals else "    n/a"
            print(f"  {lag:>4d} | {f_s:>8s} | {k_s:>8s} | {ns_s:>8s} | {ds_s:>8s} | {dst_s:>8s}")

    # ============================================================
    # STEP 5: Full model evaluation (blended)
    # ============================================================
    print(f"\n  {'=' * 80}")
    print(f"  FULL MODEL EVALUATION (blended OO + FC)")
    print(f"  {'=' * 80}")

    models = {
        'V8d (OO+FC)':           ('oo_implied_flat',        'fc_implied_flat'),
        'V8k + FC flat':         ('oo_implied_v8k',         'fc_implied_flat'),
        'NormS + FC flat':       ('oo_implied_norm_site',   'fc_implied_flat'),
        'DampS + FC flat':       ('oo_implied_damped_s',    'fc_implied_flat'),
        'DampST + FC flat':      ('oo_implied_damped_st',   'fc_implied_flat'),
        'V8d + mFC':             ('oo_implied_flat',        'fc_implied_multi'),
        'V8k + mFC':             ('oo_implied_v8k',         'fc_implied_multi'),
        'NormS + mFC':           ('oo_implied_norm_site',   'fc_implied_multi'),
        'DampS + mFC':           ('oo_implied_damped_s',    'fc_implied_multi'),
        'DampST + mFC':          ('oo_implied_damped_st',   'fc_implied_multi'),
        'FC-only multi':         ('oo_none',                'fc_implied_multi'),
    }

    for model_name, (oo_c, fc_c) in models.items():
        test[model_name] = weighted_avg(test, train, test_months, gsa_sites,
                                        oo_col=oo_c, fc_col=fc_c)

    # Skip first test month
    m2 = test['Reference_Month'] > test_months[0]

    print(f"\n  {'Model':>22s} | {'WMAPE':>7s} | {'Bias':>7s} | {'N':>5s}")
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

    # Per-site breakdown for key models
    key_models = ['V8d (OO+FC)', 'V8d + mFC', 'DampS + mFC', 'DampST + mFC',
                  'FC-only multi']
    print(f"\n  Per-site WMAPE (temporal, months 2+):")
    header = f"  {'Site':>25s}"
    for mn in key_models:
        short = mn[:10]
        header += f" | {short:>10s}"
    print(header)
    print(f"  {'-' * (28 + 13 * len(key_models))}")
    for gs in gsa_sites:
        sub = test[m2 & (test['gsa_site'] == gs) & (test['Actual_Sales'] > 0)]
        if len(sub) == 0:
            continue
        site = gs.split('|')[1] if '|' in gs else gs
        line = f"  {site:>25s}"
        for mn in key_models:
            v = sub[sub[mn].notna()]
            if len(v) > 0:
                w = compute_wmape(v['Actual_Sales'].values, v[mn].values)
                line += f" | {w:>8.1f}%"
            else:
                line += f" | {'n/a':>9s}"
        print(line)

    # Per-lag breakdown
    print(f"\n  Per-lag WMAPE (temporal, months 2+):")
    header = f"  {'Lag':>4s}"
    for mn in key_models:
        short = mn[:10]
        header += f" | {short:>10s}"
    print(header)
    print(f"  {'-' * (7 + 13 * len(key_models))}")
    for lag in sorted(test['Prediction_Lag'].unique()):
        sub = test[m2 & (test['Prediction_Lag'] == lag) & (test['Actual_Sales'] > 0)]
        line = f"  {lag:>4d}"
        for mn in key_models:
            v = sub[sub[mn].notna()]
            if len(v) > 0:
                w = compute_wmape(v['Actual_Sales'].values, v[mn].values)
                line += f" | {w:>8.1f}%"
            else:
                line += f" | {'n/a':>9s}"
        print(line)

    # ============================================================
    # STEP 6: K-Fold Random CV
    # ============================================================
    print(f"\n  {'=' * 80}")
    print(f"  K-FOLD RANDOM CV ({N_FOLDS} folds)")
    print(f"  {'=' * 80}")

    np.random.seed(RANDOM_SEED)
    indices = np.arange(len(df))
    np.random.shuffle(indices)
    fold_size = len(df) // N_FOLDS

    fold_metrics = {mn: [] for mn in models}

    for fold_i in range(N_FOLDS):
        fold_start = fold_i * fold_size
        fold_end = fold_start + fold_size if fold_i < N_FOLDS - 1 else len(df)
        test_idx = indices[fold_start:fold_end]
        train_idx = np.setdiff1d(indices, test_idx)

        f_train = df.iloc[train_idx].copy()
        f_test = df.iloc[test_idx].copy()

        # Build all curves from fold training data
        oo_f = build_flat_curves(f_train, 'oo_ratio', start='2023-01-01')
        cov_f = build_flat_curves(f_train, 'fc_coverage', start='2023-01-01')
        bias_f = build_flat_curves(f_train, 'fc_bias', start='2023-01-01')

        # Estimate alphas from fold training data
        a_site, _ = estimate_alpha(f_train, ['gsa_site'])
        a_site_tf, _ = estimate_alpha(f_train, ['gsa_site', 'Timeframe'])

        # Build normalized curves
        nc_site = build_normalized_curves(f_train, a_site, 'site', start='2023-01-01')
        nc_site_tf = build_normalized_curves(f_train, a_site_tf, 'site_tf',
                                              start='2023-01-01')

        # LT+OE reference for V8k
        f_ref_ltoe = build_ltoe_reference(f_train)

        # Apply curves to train and test
        apply_flat_curves(f_train, oo_f, cov_f, bias_f)
        apply_flat_curves(f_test, oo_f, cov_f, bias_f)
        apply_v8k_correction(f_train, oo_f, f_ref_ltoe)
        apply_v8k_correction(f_test, oo_f, f_ref_ltoe)
        apply_normalized_curves(f_train, nc_site, a_site, oo_f, 'site',
                                'oo_implied_norm_site')
        apply_normalized_curves(f_test, nc_site, a_site, oo_f, 'site',
                                'oo_implied_norm_site')
        apply_normalized_curves(f_train, nc_site_tf, a_site_tf, oo_f, 'site_tf',
                                'oo_implied_norm_site_tf')
        apply_normalized_curves(f_test, nc_site_tf, a_site_tf, oo_f, 'site_tf',
                                'oo_implied_norm_site_tf')

        # Apply damped correction in CV
        apply_damped_power_correction(f_train, oo_f, f_ref_ltoe, a_site,
                                      'site', 'oo_implied_damped_s')
        apply_damped_power_correction(f_test, oo_f, f_ref_ltoe, a_site,
                                      'site', 'oo_implied_damped_s')
        apply_damped_power_correction(f_train, oo_f, f_ref_ltoe, a_site_tf,
                                      'site_tf', 'oo_implied_damped_st')
        apply_damped_power_correction(f_test, oo_f, f_ref_ltoe, a_site_tf,
                                      'site_tf', 'oo_implied_damped_st')

        # Multi-vintage FC from training fold
        fc_reg_fold = build_forecast_registry(f_train)
        apply_multi_vintage_fc(f_train, cov_f, bias_f, fc_reg_fold, 'fold')
        apply_multi_vintage_fc(f_test, cov_f, bias_f, fc_reg_fold, 'fold')

        f_train['oo_none'] = np.nan
        f_test['oo_none'] = np.nan

        f_test_months = sorted(f_test['Reference_Month'].unique())

        for model_name, (oo_c, fc_c) in models.items():
            f_test[model_name] = weighted_avg(f_test, f_train, f_test_months,
                                              gsa_sites, oo_col=oo_c, fc_col=fc_c)

        ts = f_test[f_test['Actual_Sales'] > 0]
        for model_name in models:
            v = ts[ts[model_name].notna()]
            if len(v) > 0:
                wmape = compute_wmape(v['Actual_Sales'].values, v[model_name].values)
                bias = compute_bias(v['Actual_Sales'].values, v[model_name].values)
                fold_metrics[model_name].append({'wmape': wmape, 'bias': bias})

        # Per-fold summary
        v8d_w = fold_metrics['V8d (OO+FC)'][-1]['wmape'] if fold_metrics['V8d (OO+FC)'] else 0
        ds_w = fold_metrics['DampS + mFC'][-1]['wmape'] if fold_metrics['DampS + mFC'] else 0
        dst_w = fold_metrics['DampST + mFC'][-1]['wmape'] if fold_metrics['DampST + mFC'] else 0
        print(f"    Fold {fold_i + 1}: V8d={v8d_w:.1f}%  DampS+mFC={ds_w:.1f}%  "
              f"DampST+mFC={dst_w:.1f}%")

    # Random CV summary
    print(f"\n  {'Model':>22s} | {'Random CV':>20s} | {'Temporal':>10s} | {'Gap':>6s}")
    print(f"  {'-' * 65}")
    for model_name in models:
        wmapes = [m['wmape'] for m in fold_metrics[model_name]]
        biases = [m['bias'] for m in fold_metrics[model_name]]
        t_wmape = temporal_results.get(model_name, float('nan'))
        gap = t_wmape - np.mean(wmapes) if wmapes else float('nan')
        if wmapes:
            print(f"  {model_name:>22s} | {np.mean(wmapes):>5.1f}% ± {np.std(wmapes):>4.1f}% "
                  f"(b{np.mean(biases):>+5.1f}%) | {t_wmape:>6.1f}% | {gap:>+4.1f}pp")

    # ============================================================
    # STEP 7: Normalization quality diagnostic
    # ============================================================
    print(f"\n  {'=' * 80}")
    print(f"  NORMALIZATION QUALITY DIAGNOSTIC")
    print(f"  {'=' * 80}")

    # Compare CV of raw vs normalized OO ratio within progress bins
    print(f"\n  CV of OO_ratio in progress bins (raw vs normalized, training data):")
    prog_bins = np.arange(0, 1.01, 0.1)
    t_valid = train[
        train['oo_ratio'].notna() & (train['oo_ratio'] > 0) &
        (train['oo_ratio'] < 10) & train['progress'].notna() &
        (train['progress'] > PROGRESS_EPSILON)
    ].copy()

    if len(t_valid) > 0:
        # Add normalized ratios to training data for this diagnostic
        for i, (idx, row) in enumerate(t_valid.iterrows()):
            gs = row['gsa_site']
            tf = row['Timeframe']
            prog = row['progress']
            a_s = get_alpha(alpha_site, gs)
            a_st = get_alpha(alpha_site_tf, gs, tf)
            t_valid.at[idx, 'oo_ratio_norm_s'] = row['oo_ratio'] / (prog ** a_s)
            t_valid.at[idx, 'oo_ratio_norm_st'] = row['oo_ratio'] / (prog ** a_st)

        t_valid['prog_bin'] = pd.cut(t_valid['progress'], bins=prog_bins)

        print(f"\n  {'Bin':>12s} | {'N':>5s} | {'Raw CV':>8s} | {'NormS CV':>9s} | "
              f"{'NormST CV':>10s} | {'S improve':>9s} | {'ST improve':>10s}")
        print(f"  {'-' * 80}")

        total_raw_cv = []
        total_ns_cv = []
        total_nst_cv = []
        for bin_label in sorted(t_valid['prog_bin'].unique()):
            sub = t_valid[t_valid['prog_bin'] == bin_label]
            if len(sub) < 5:
                continue
            raw_cv = sub['oo_ratio'].std() / sub['oo_ratio'].mean()
            ns_cv = sub['oo_ratio_norm_s'].std() / sub['oo_ratio_norm_s'].mean()
            nst_cv = sub['oo_ratio_norm_st'].std() / sub['oo_ratio_norm_st'].mean()
            s_impr = (1 - ns_cv / raw_cv) * 100 if raw_cv > 0 else 0
            st_impr = (1 - nst_cv / raw_cv) * 100 if raw_cv > 0 else 0
            total_raw_cv.append(raw_cv)
            total_ns_cv.append(ns_cv)
            total_nst_cv.append(nst_cv)
            print(f"  {str(bin_label):>12s} | {len(sub):>5d} | {raw_cv:>8.3f} | "
                  f"{ns_cv:>9.3f} | {nst_cv:>10.3f} | {s_impr:>+7.1f}% | {st_impr:>+8.1f}%")

        if total_raw_cv:
            avg_raw = np.mean(total_raw_cv)
            avg_ns = np.mean(total_ns_cv)
            avg_nst = np.mean(total_nst_cv)
            print(f"  {'AVERAGE':>12s} | {'':>5s} | {avg_raw:>8.3f} | "
                  f"{avg_ns:>9.3f} | {avg_nst:>10.3f} | "
                  f"{(1 - avg_ns/avg_raw)*100:>+7.1f}% | "
                  f"{(1 - avg_nst/avg_raw)*100:>+8.1f}%")

    print(f"\n\nDone!")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='OO Power-CDF Normalization Experiment')
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
