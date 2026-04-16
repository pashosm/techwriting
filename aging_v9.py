"""
V9 (Phase 1): Pure FC Model with Point-in-Time Evaluation
==========================================================
Simplified architecture — four principles:
  1. FC is the only prediction signal
  2. Fall back to historical sales when no FC, label the source
  3. Built on FC bias x FC coverage curves (no LT/OE conditioning)
  4. Regime detection deferred to Phase 2

Point-in-time evaluation: at snapshot date T, the model only sees rows whose
outcome was already known at T (Target_Period_End < T). Recent FC vintages
(Reference_Month < T) remain available even when the cell has no history.

Evaluation scope (Phase 1 narrowed):
  - Universe:   Timeframe == 12  (applied globally on load)
  - Test rows:  Prediction_Lag == 1  (applied when selecting test window)
  - Registry:   NOT lag-filtered; keeps all vintages of tf=12 target periods so
                lag-1 test rows can combine their own vintage with earlier-
                published lag-2/3/.../N vintages for the same target period.

Bias/coverage correction is only applied when the specific
(gsa_site, Timeframe, Prediction_Lag) cell has >= MIN_OBS_FLAT=3 observations
in completed, FC-bearing training data. Otherwise the vintage contributes
raw (uncorrected) Forecast_Value to the multi-vintage combination.

Usage:
  python aging_v9.py --snapshot 2024-12-01 --horizon 3
  python aging_v9.py --snapshot 2024-12-01 --horizon 3 --data path/to/file.csv
  python aging_v9.py --grid 2024-06-01:2025-03-01 --horizon 3
"""
import argparse
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Constants (mirror V8g for fair comparison)
# ============================================================
RECENCY_POWER = 2
MIN_OBS_FLAT = 3
VINTAGE_RECENCY_POWER = 1.5

# Universe filter: applied in load_data. Every train/test/registry row must
# satisfy this. tf=12 means we care only about 12-month target periods.
UNIVERSE_TIMEFRAMES = {12}

# Test filter: applied when selecting test rows inside run_snapshot. Registry
# is NOT filtered by this - earlier-lag vintages of the same target period
# must remain available to lag-1 test rows via multi-vintage combination.
TEST_LAGS = {1}

DEFAULT_DATA_PATH = 'training_data_anonymized.csv'

# ============================================================
# Data loading
# ============================================================
def load_data(path):
    df = pd.read_csv(
        path,
        parse_dates=['Reference_Month', 'Target_Period_Start', 'Target_Period_End'],
        encoding='utf-8-sig',
    )
    df['gsa_site'] = df['GSA'].astype(str) + '|' + df['Site'].astype(str)

    # Universe filter
    df = df[df['Timeframe'].isin(UNIVERSE_TIMEFRAMES)].copy()

    # Derived ratios for curve building
    df['fc_coverage'] = np.where(df['Actual_Sales'] > 0,
                                 df['Covered_Orders'] / df['Actual_Sales'], np.nan)
    df['fc_bias'] = np.where(df['Covered_Orders'] > 0,
                             df['Forecast_Value'] / df['Covered_Orders'], np.nan)
    return df

# ============================================================
# Curve building (flat only, no LT/OE conditioning)
# Primary cell (gs, tf, lag) only. No pooled ('FB', gs, lag) fallback.
# ============================================================
def _recency_weights(ref_months):
    days = (ref_months - ref_months.min()).dt.days / 365
    return (1 + days.values) ** RECENCY_POWER

def build_flat_curves(train_df, ratio_col):
    """{(gs, tf, lag): val} at MIN_OBS_FLAT=3. No site-level fallback."""
    curve = {}
    for (gs, tf, lag), grp in train_df.groupby(['gsa_site', 'Timeframe', 'Prediction_Lag']):
        valid = grp[grp[ratio_col].notna() & (grp[ratio_col] > 0) & (grp[ratio_col] < 10)]
        if len(valid) < MIN_OBS_FLAT:
            continue
        wts = _recency_weights(valid['Reference_Month'])
        curve[(gs, tf, lag)] = np.average(valid[ratio_col], weights=wts)
    return curve

def build_historical(train_df):
    """{(gs, tf): recency-weighted mean Actual_Sales} + {('FB', gs): pooled}.
    Historical fallback keeps its site-level pool: it's an independent signal,
    not a correction curve."""
    hist = {}
    for (gs, tf), grp in train_df.groupby(['gsa_site', 'Timeframe']):
        valid = grp[grp['Actual_Sales'].notna() & (grp['Actual_Sales'] > 0)]
        if len(valid) < MIN_OBS_FLAT:
            continue
        wts = _recency_weights(valid['Reference_Month'])
        hist[(gs, tf)] = np.average(valid['Actual_Sales'], weights=wts)
    for gs, grp in train_df.groupby('gsa_site'):
        valid = grp[grp['Actual_Sales'].notna() & (grp['Actual_Sales'] > 0)]
        if len(valid) < MIN_OBS_FLAT:
            continue
        wts = _recency_weights(valid['Reference_Month'])
        hist[('FB', gs)] = np.average(valid['Actual_Sales'], weights=wts)
    return hist

# ============================================================
# Multi-vintage forecast registry (mirrors V8g)
# ============================================================
def build_forecast_registry(data_df):
    """{(gsa_site, TP_Start, TP_End): [vintage_dicts sorted by lag asc]}.
    Built from ALL FC-bearing rows visible at snapshot, not just training rows."""
    registry = {}
    fc_rows = data_df[data_df['Has_Forecast'] == 1]
    for _, row in fc_rows.iterrows():
        key = (row['gsa_site'], row['Target_Period_Start'], row['Target_Period_End'])
        registry.setdefault(key, []).append({
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

def _vintage_to_estimate(vintage, cov_curve, bias_curve):
    """Returns (estimate, valid, corrected).
    corrected=True iff the specific (gs, tf, lag) cell had a curve (>= MIN_OBS_FLAT
    completed+FC-bearing training obs). No fallback key is consulted.
    When no cell curve exists, returns raw fc_value with corrected=False."""
    gs, tf, lag = vintage['gsa_site'], vintage['timeframe'], vintage['lag']
    fc_val = vintage['fc_value']
    if fc_val <= 0:
        return np.nan, False, False
    key = (gs, tf, lag)
    cf = cov_curve.get(key)
    bf = bias_curve.get(key)
    if cf and bf and cf > 0.001 and bf > 0.001:
        return fc_val / bf / cf, True, True
    return fc_val, True, False

def _combine_vintage_estimates(estimates_with_lags):
    if not estimates_with_lags:
        return np.nan
    total_w, total_val = 0.0, 0.0
    for est, orig_lag in estimates_with_lags:
        w = 1.0 / (orig_lag ** VINTAGE_RECENCY_POWER)
        total_w += w
        total_val += w * est
    return total_val / total_w if total_w > 0 else np.nan

# ============================================================
# Prediction
# ============================================================
SOURCES_FC_CORRECTED = {'FC_MULTI_CORRECTED', 'FC_SINGLE_CORRECTED'}
SOURCES_FC_RAW = {'FC_MULTI_RAW', 'FC_SINGLE_RAW'}
SOURCES_FC_ANY = SOURCES_FC_CORRECTED | SOURCES_FC_RAW
ALL_SOURCES = ['FC_MULTI_CORRECTED', 'FC_MULTI_RAW',
               'FC_SINGLE_CORRECTED', 'FC_SINGLE_RAW', 'HISTORICAL']

def predict(test_df, cov_curve, bias_curve, hist, registry):
    """For each test row: combine all available vintages of its target period
    (with ref_month <= test.Reference_Month and fc_value > 0). Label source by
    (multi vs single) x (at least one corrected vs all raw), else HISTORICAL."""
    n = len(test_df)
    preds = np.full(n, np.nan)
    sources = np.empty(n, dtype=object)
    n_vint = np.zeros(n, dtype=int)
    n_corrected_vint = np.zeros(n, dtype=int)

    for i, (_, row) in enumerate(test_df.iterrows()):
        gs = row['gsa_site']
        tf = row['Timeframe']
        ref_m = row['Reference_Month']
        tp_key = (gs, row['Target_Period_Start'], row['Target_Period_End'])

        if tp_key in registry:
            valid = []
            for v in registry[tp_key]:
                if v['ref_month'] > ref_m:  # temporal causality
                    continue
                est, ok, corrected = _vintage_to_estimate(v, cov_curve, bias_curve)
                if ok:
                    valid.append((max(est, 0), v['lag'], corrected))

            if valid:
                n_vint[i] = len(valid)
                n_corrected_vint[i] = sum(1 for _, _, c in valid if c)
                combined = max(_combine_vintage_estimates(
                    [(e, l) for e, l, _ in valid]), 0)
                preds[i] = combined
                any_corrected = n_corrected_vint[i] > 0
                if len(valid) >= 2:
                    sources[i] = 'FC_MULTI_CORRECTED' if any_corrected else 'FC_MULTI_RAW'
                else:
                    sources[i] = 'FC_SINGLE_CORRECTED' if any_corrected else 'FC_SINGLE_RAW'
                continue

        # Historical fallback
        h = hist.get((gs, tf), hist.get(('FB', gs)))
        if h is not None and h > 0:
            preds[i] = h
            sources[i] = 'HISTORICAL'

    out = test_df.copy()
    out['Prediction'] = preds
    out['Prediction_Source'] = sources
    out['N_Vintages'] = n_vint
    out['N_Corrected_Vintages'] = n_corrected_vint
    out['Curve_Corrected'] = n_corrected_vint > 0
    return out

# ============================================================
# Scoring
# ============================================================
def _wmape(actual, pred):
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(p)) & (a > 0)
    if not mask.any():
        return np.nan, 0
    return float(np.sum(np.abs(a[mask] - p[mask])) / np.sum(a[mask])), int(mask.sum())

def _bias(actual, pred):
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(p)) & (a > 0)
    if not mask.any():
        return np.nan
    return float((np.sum(p[mask]) - np.sum(a[mask])) / np.sum(a[mask]))

def _metric_block(sub):
    w, n = _wmape(sub['Actual_Sales'], sub['Prediction'])
    return {
        'wmape': w,
        'bias': _bias(sub['Actual_Sales'], sub['Prediction']),
        'n': n,
        'rows': len(sub),
    }

def score(pred_df):
    metrics = {'overall': _metric_block(pred_df)}
    # 5-level source detail
    for src in ALL_SOURCES:
        metrics[src] = _metric_block(pred_df[pred_df['Prediction_Source'] == src])
    # 3-level rollup (FC_CORRECTED / FC_RAW / HIST)
    metrics['FC_CORRECTED'] = _metric_block(
        pred_df[pred_df['Prediction_Source'].isin(SOURCES_FC_CORRECTED)])
    metrics['FC_RAW'] = _metric_block(
        pred_df[pred_df['Prediction_Source'].isin(SOURCES_FC_RAW)])
    metrics['FC_ANY'] = _metric_block(
        pred_df[pred_df['Prediction_Source'].isin(SOURCES_FC_ANY)])
    metrics['HIST'] = _metric_block(
        pred_df[pred_df['Prediction_Source'] == 'HISTORICAL'])
    metrics['NO_PREDICTION'] = {'rows': int(pred_df['Prediction'].isna().sum())}
    return metrics

# ============================================================
# Snapshot harness
# ============================================================
def run_snapshot(df, snapshot_date, horizon_months, verbose=True):
    """Train on rows known at snapshot, predict rows landing in next horizon_months."""
    snap = pd.Timestamp(snapshot_date)
    horizon_end = snap + pd.DateOffset(months=horizon_months)

    # Train: outcome already known by snapshot
    train = df[df['Target_Period_End'] < snap].copy()

    # Test: target window starts in [snap, snap+horizon), ref_month strictly
    # before snap, and only the lag(s) we care about.
    test = df[
        (df['Target_Period_Start'] >= snap) &
        (df['Target_Period_Start'] < horizon_end) &
        (df['Reference_Month'] < snap) &
        (df['Prediction_Lag'].isin(TEST_LAGS))
    ].copy()

    # Registry: any FC vintage with ref_month < snap (all lags).
    visible = df[df['Reference_Month'] < snap]

    # Curves: completed + FC-bearing training data only.
    curve_train = train[train['Has_Forecast'] == 1]

    if verbose:
        print(f"\n  Snapshot {snap.date()} (horizon {horizon_months}mo):")
        print(f"    Train rows (outcomes known):      {len(train):,}")
        print(f"    Curve train rows (Has_Forecast=1): {len(curve_train):,}")
        print(f"    Test rows (tf={sorted(UNIVERSE_TIMEFRAMES)}, lag={sorted(TEST_LAGS)}): "
              f"{len(test):,}")
        print(f"    Visible FC-bearing rows (all lags): "
              f"{int((visible['Has_Forecast']==1).sum()):,}")

    if len(train) == 0 or len(test) == 0:
        return None, None

    cov_curve = build_flat_curves(curve_train, 'fc_coverage')
    bias_curve = build_flat_curves(curve_train, 'fc_bias')
    hist = build_historical(train)
    registry = build_forecast_registry(visible)

    if verbose:
        print(f"    Curves: cov={len(cov_curve):,}  bias={len(bias_curve):,}  hist={len(hist):,}")
        print(f"    Registry target-period keys: {len(registry):,}")

    pred_df = predict(test, cov_curve, bias_curve, hist, registry)
    metrics = score(pred_df)
    return metrics, pred_df

# ============================================================
# Reporting
# ============================================================
def _fmt_pct(x):
    return f"{x*100:+.1f}%" if x is not None and not (isinstance(x, float) and np.isnan(x)) else "  n/a"

def _fmt_wmape(x):
    return f"{x*100:.1f}%" if x is not None and not (isinstance(x, float) and np.isnan(x)) else " n/a"

def print_metrics(metrics, snapshot_date):
    print(f"\n  Results for snapshot {snapshot_date}:")
    print(f"    {'Source':<24} {'Rows':>8} {'WMAPE':>8} {'Bias':>10}")
    print(f"    {'-'*24} {'-'*8} {'-'*8} {'-'*10}")
    # 3-level rollup first
    for src in ['overall', 'FC_CORRECTED', 'FC_RAW', 'FC_ANY', 'HIST']:
        m = metrics[src]
        print(f"    {src:<24} {m.get('rows',0):>8,} "
              f"{_fmt_wmape(m.get('wmape')):>8} {_fmt_pct(m.get('bias')):>10}")
    print(f"    {'-'*24}")
    # 5-level detail
    for src in ALL_SOURCES:
        m = metrics[src]
        print(f"    {src:<24} {m.get('rows',0):>8,} "
              f"{_fmt_wmape(m.get('wmape')):>8} {_fmt_pct(m.get('bias')):>10}")
    print(f"    {'NO_PREDICTION':<24} {metrics['NO_PREDICTION']['rows']:>8,}")

def print_per_site(pred_df, top=15):
    print(f"\n  Per-site WMAPE (top {top} by row count):")
    rows = []
    for gs, grp in pred_df.groupby('gsa_site'):
        w, n = _wmape(grp['Actual_Sales'], grp['Prediction'])
        rows.append((gs, len(grp), n, w, _bias(grp['Actual_Sales'], grp['Prediction'])))
    rows.sort(key=lambda r: -r[1])
    print(f"    {'Site':<28} {'Rows':>8} {'Scored':>8} {'WMAPE':>8} {'Bias':>10}")
    for gs, r, n, w, b in rows[:top]:
        print(f"    {gs:<28} {r:>8,} {n:>8,} {_fmt_wmape(w):>8} {_fmt_pct(b):>10}")

def print_per_lag(pred_df):
    print(f"\n  Per-Lag WMAPE:")
    print(f"    {'Lag':>4} {'Rows':>8} {'WMAPE':>8} {'Bias':>10}")
    for lag in sorted(pred_df['Prediction_Lag'].unique()):
        grp = pred_df[pred_df['Prediction_Lag'] == lag]
        w, n = _wmape(grp['Actual_Sales'], grp['Prediction'])
        print(f"    {lag:>4} {len(grp):>8,} {_fmt_wmape(w):>8} {_fmt_pct(_bias(grp['Actual_Sales'], grp['Prediction'])):>10}")

def print_per_customer(pred_df, sort_key='fc_corr_bias'):
    """Aggregate predictions by customer (GSA) across all snapshots.
    Shows overall + FC_CORRECTED + FC_RAW + HIST per customer."""
    print(f"\n  Per-Customer WMAPE / Bias  (sorted by |FC_Corrected bias|, largest first)")
    hdr = (f"    {'Customer':<14} {'Rows':>6} {'Overall':>10} "
           f"| {'FC_C n':>6} {'FC_C WMAPE':>11} {'FC_C Bias':>10} "
           f"| {'FC_R n':>6} {'FC_R WMAPE':>11} {'FC_R Bias':>10} "
           f"| {'HIST n':>6} {'HIST WMAPE':>11} {'HIST Bias':>10}")
    print(hdr)
    print(f"    {'-'*14} {'-'*6} {'-'*10} | {'-'*6} {'-'*11} {'-'*10} "
          f"| {'-'*6} {'-'*11} {'-'*10} | {'-'*6} {'-'*11} {'-'*10}")

    rows = []
    for cust, grp in pred_df.groupby('GSA'):
        fc_c = grp[grp['Prediction_Source'].isin(SOURCES_FC_CORRECTED)]
        fc_r = grp[grp['Prediction_Source'].isin(SOURCES_FC_RAW)]
        hist = grp[grp['Prediction_Source'] == 'HISTORICAL']
        overall_w, _ = _wmape(grp['Actual_Sales'], grp['Prediction'])
        overall_b = _bias(grp['Actual_Sales'], grp['Prediction'])
        fc_c_w, _ = _wmape(fc_c['Actual_Sales'], fc_c['Prediction'])
        fc_c_b = _bias(fc_c['Actual_Sales'], fc_c['Prediction'])
        fc_r_w, _ = _wmape(fc_r['Actual_Sales'], fc_r['Prediction'])
        fc_r_b = _bias(fc_r['Actual_Sales'], fc_r['Prediction'])
        h_w, _ = _wmape(hist['Actual_Sales'], hist['Prediction'])
        h_b = _bias(hist['Actual_Sales'], hist['Prediction'])
        rows.append({
            'cust': cust, 'rows': len(grp),
            'overall_w': overall_w, 'overall_b': overall_b,
            'fc_c_n': len(fc_c), 'fc_c_w': fc_c_w, 'fc_c_b': fc_c_b,
            'fc_r_n': len(fc_r), 'fc_r_w': fc_r_w, 'fc_r_b': fc_r_b,
            'h_n': len(hist), 'h_w': h_w, 'h_b': h_b,
        })

    # Sort by |FC_Corrected bias| desc, putting n/a at end
    def sort_k(r):
        v = r['fc_c_b']
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return -1.0
        return abs(v)
    rows.sort(key=sort_k, reverse=True)

    for r in rows:
        print(f"    {r['cust']:<14} {r['rows']:>6,} "
              f"{_fmt_wmape(r['overall_w']):>4}/{_fmt_pct(r['overall_b']):<5} | "
              f"{r['fc_c_n']:>6,} {_fmt_wmape(r['fc_c_w']):>11} {_fmt_pct(r['fc_c_b']):>10} | "
              f"{r['fc_r_n']:>6,} {_fmt_wmape(r['fc_r_w']):>11} {_fmt_pct(r['fc_r_b']):>10} | "
              f"{r['h_n']:>6,} {_fmt_wmape(r['h_w']):>11} {_fmt_pct(r['h_b']):>10}")

# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="V9 Phase 1: pure FC model, point-in-time evaluation")
    p.add_argument('--data', default=DEFAULT_DATA_PATH, help='CSV path')
    p.add_argument('--snapshot', default=None, help='Single snapshot date YYYY-MM-DD')
    p.add_argument('--grid', default=None,
                   help='Grid of monthly snapshots: START:END (e.g. 2024-06-01:2025-03-01)')
    p.add_argument('--horizon', type=int, default=3, help='Months of test window after snapshot')
    p.add_argument('--save-predictions', default=None, help='Optional path to write per-row predictions')
    p.add_argument('--quiet', action='store_true')
    return p.parse_args()

def monthly_grid(start, end):
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    out = []
    cur = s
    while cur <= e:
        out.append(cur)
        cur = cur + pd.DateOffset(months=1)
    return out

def main():
    args = parse_args()
    if not args.snapshot and not args.grid:
        print("Error: provide --snapshot or --grid", file=sys.stderr)
        sys.exit(2)

    print("=" * 90)
    print("V9 Phase 1: Pure FC Model - Point-in-Time Evaluation")
    print("=" * 90)
    print(f"Data: {args.data}")
    print(f"Universe: Timeframe in {sorted(UNIVERSE_TIMEFRAMES)}   "
          f"Test: Prediction_Lag in {sorted(TEST_LAGS)}")

    df = load_data(args.data)
    print(f"Loaded {len(df):,} rows after filters across {df['gsa_site'].nunique()} sites")
    print(f"Date range: {df['Reference_Month'].min().date()} to {df['Reference_Month'].max().date()}")

    snapshots = [args.snapshot] if args.snapshot else \
        [d.strftime('%Y-%m-%d') for d in monthly_grid(*args.grid.split(':'))]

    grid_summary = []
    all_preds = []
    for snap in snapshots:
        metrics, pred_df = run_snapshot(df, snap, args.horizon, verbose=not args.quiet)
        if metrics is None:
            print(f"  Snapshot {snap}: empty train or test, skipping")
            continue
        if not args.quiet:
            print_metrics(metrics, snap)
            if args.snapshot:  # detailed breakdowns only for single snapshot
                print_per_site(pred_df)
                print_per_lag(pred_df)

        grid_summary.append({
            'snapshot': snap,
            'wmape': metrics['overall']['wmape'],
            'bias': metrics['overall']['bias'],
            'rows': metrics['overall']['rows'],
            'fc_corrected_rows': metrics['FC_CORRECTED']['rows'],
            'fc_corrected_wmape': metrics['FC_CORRECTED']['wmape'],
            'fc_corrected_bias': metrics['FC_CORRECTED']['bias'],
            'fc_raw_rows': metrics['FC_RAW']['rows'],
            'fc_raw_wmape': metrics['FC_RAW']['wmape'],
            'fc_raw_bias': metrics['FC_RAW']['bias'],
            'hist_rows': metrics['HIST']['rows'],
            'hist_wmape': metrics['HIST']['wmape'],
            'hist_bias': metrics['HIST']['bias'],
            'no_pred_n': metrics['NO_PREDICTION']['rows'],
        })
        # Keep a slim copy of predictions for cross-snapshot aggregations
        slim = pred_df[['GSA', 'gsa_site', 'Prediction_Lag', 'Timeframe',
                        'Actual_Sales', 'Prediction', 'Prediction_Source']].copy()
        slim['Snapshot'] = snap
        all_preds.append(slim)

    if len(grid_summary) > 1:
        print("\n" + "=" * 100)
        print("Snapshot grid summary (overall row counts)")
        print("=" * 100)
        print(f"  {'Snapshot':<12} {'Rows':>8} {'WMAPE':>8} {'Bias':>10}  "
              f"{'FC_Corr':>8} {'FC_Raw':>7} {'HIST':>7} {'NoPred':>7}")
        for r in grid_summary:
            print(f"  {r['snapshot']:<12} {r['rows']:>8,} "
                  f"{_fmt_wmape(r['wmape']):>8} {_fmt_pct(r['bias']):>10}  "
                  f"{r['fc_corrected_rows']:>8,} {r['fc_raw_rows']:>7,} "
                  f"{r['hist_rows']:>7,} {r['no_pred_n']:>7,}")

        print("\n" + "=" * 100)
        print("Snapshot grid summary (signal basis: FC_Corrected vs FC_Raw vs HIST)")
        print("=" * 100)
        print(f"  {'Snapshot':<12} | {'FC_C Rows':>9} {'FC_C WMAPE':>10} {'FC_C Bias':>10} "
              f"| {'FC_R Rows':>9} {'FC_R WMAPE':>10} {'FC_R Bias':>10} "
              f"| {'HIST Rows':>9} {'HIST WMAPE':>10} {'HIST Bias':>10}")
        for r in grid_summary:
            print(f"  {r['snapshot']:<12} | "
                  f"{r['fc_corrected_rows']:>9,} "
                  f"{_fmt_wmape(r['fc_corrected_wmape']):>10} "
                  f"{_fmt_pct(r['fc_corrected_bias']):>10} | "
                  f"{r['fc_raw_rows']:>9,} "
                  f"{_fmt_wmape(r['fc_raw_wmape']):>10} "
                  f"{_fmt_pct(r['fc_raw_bias']):>10} | "
                  f"{r['hist_rows']:>9,} "
                  f"{_fmt_wmape(r['hist_wmape']):>10} "
                  f"{_fmt_pct(r['hist_bias']):>10}")

        # Aggregate across snapshots (simple mean of per-snapshot rates)
        def _mean(key):
            vals = [r[key] for r in grid_summary
                    if r[key] is not None and not np.isnan(r[key])]
            return np.mean(vals) if vals else np.nan

        print(f"\n  Mean WMAPE  overall={_mean('wmape')*100:.2f}%   "
              f"FC_Corrected={_mean('fc_corrected_wmape')*100:.2f}%   "
              f"FC_Raw={_mean('fc_raw_wmape')*100:.2f}%   "
              f"HIST={_mean('hist_wmape')*100:.2f}%")
        print(f"  Mean Bias   overall={_mean('bias')*100:+.2f}%   "
              f"FC_Corrected={_mean('fc_corrected_bias')*100:+.2f}%   "
              f"FC_Raw={_mean('fc_raw_bias')*100:+.2f}%   "
              f"HIST={_mean('hist_bias')*100:+.2f}%")

    # Per-customer aggregation across all snapshots
    if all_preds:
        combined = pd.concat(all_preds, ignore_index=True)
        print("\n" + "=" * 100)
        print("Per-customer summary across all snapshots")
        print("=" * 100)
        print_per_customer(combined)

        if args.save_predictions:
            combined.to_csv(args.save_predictions, index=False)
            print(f"\nWrote per-row predictions to {args.save_predictions}")

if __name__ == '__main__':
    main()
