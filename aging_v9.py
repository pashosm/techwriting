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

ALLOWED_LAGS = {1, 2, 3}
ALLOWED_TIMEFRAMES = {3, 6, 9, 12}

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

    # Phase 1 scope filters
    df = df[df['Prediction_Lag'].isin(ALLOWED_LAGS)].copy()
    df = df[df['Timeframe'].isin(ALLOWED_TIMEFRAMES)].copy()

    # Derived ratios for curve building
    df['fc_coverage'] = np.where(df['Actual_Sales'] > 0,
                                 df['Covered_Orders'] / df['Actual_Sales'], np.nan)
    df['fc_bias'] = np.where(df['Covered_Orders'] > 0,
                             df['Forecast_Value'] / df['Covered_Orders'], np.nan)
    return df

# ============================================================
# Curve building (flat only, no LT/OE conditioning)
# ============================================================
def _recency_weights(ref_months):
    days = (ref_months - ref_months.min()).dt.days / 365
    return (1 + days.values) ** RECENCY_POWER

def build_flat_curves(train_df, ratio_col):
    """{(gs, tf, lag): val} primary + {('FB', gs, lag): val} pooled fallback."""
    curve = {}
    for (gs, tf, lag), grp in train_df.groupby(['gsa_site', 'Timeframe', 'Prediction_Lag']):
        valid = grp[grp[ratio_col].notna() & (grp[ratio_col] > 0) & (grp[ratio_col] < 10)]
        if len(valid) < MIN_OBS_FLAT:
            continue
        wts = _recency_weights(valid['Reference_Month'])
        curve[(gs, tf, lag)] = np.average(valid[ratio_col], weights=wts)

    for (gs, lag), grp in train_df.groupby(['gsa_site', 'Prediction_Lag']):
        valid = grp[grp[ratio_col].notna() & (grp[ratio_col] > 0) & (grp[ratio_col] < 10)]
        if len(valid) < MIN_OBS_FLAT:
            continue
        wts = _recency_weights(valid['Reference_Month'])
        curve[('FB', gs, lag)] = np.average(valid[ratio_col], weights=wts)
    return curve

def build_historical(train_df):
    """{(gs, tf): recency-weighted mean Actual_Sales} + {('FB', gs): pooled}."""
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
    gs, tf, lag = vintage['gsa_site'], vintage['timeframe'], vintage['lag']
    fc_val = vintage['fc_value']
    if fc_val <= 0:
        return np.nan, False
    for key in [(gs, tf, lag), ('FB', gs, lag)]:
        cf = cov_curve.get(key)
        bf = bias_curve.get(key)
        if cf and bf and cf > 0.001 and bf > 0.001:
            return fc_val / bf / cf, True
    # No curve available -> raw FC (treated as identity correction)
    return fc_val, True

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
def predict(test_df, cov_curve, bias_curve, hist, registry):
    """For each test row, pick first available source:
       FC_MULTI -> FC_SINGLE -> HISTORICAL.
       Returns test_df with [Prediction, Prediction_Source, N_Vintages]."""
    n = len(test_df)
    preds = np.full(n, np.nan)
    sources = np.empty(n, dtype=object)
    n_vint = np.zeros(n, dtype=int)
    has_curve_correction = np.zeros(n, dtype=bool)

    for i, (_, row) in enumerate(test_df.iterrows()):
        gs = row['gsa_site']
        tf = row['Timeframe']
        ref_m = row['Reference_Month']
        tp_key = (gs, row['Target_Period_Start'], row['Target_Period_End'])

        # Try FC (multi or single vintage)
        if tp_key in registry:
            valid = []
            for v in registry[tp_key]:
                if v['ref_month'] > ref_m:  # temporal causality
                    continue
                est, ok = _vintage_to_estimate(v, cov_curve, bias_curve)
                if ok:
                    # Track whether we used a curve (vs raw FC)
                    used_curve = (gs, v['timeframe'], v['lag']) in cov_curve or \
                                 ('FB', gs, v['lag']) in cov_curve
                    valid.append((max(est, 0), v['lag'], used_curve))

            if len(valid) >= 2:
                preds[i] = max(_combine_vintage_estimates(
                    [(e, l) for e, l, _ in valid]), 0)
                sources[i] = 'FC_MULTI'
                n_vint[i] = len(valid)
                has_curve_correction[i] = any(uc for _, _, uc in valid)
                continue
            if len(valid) == 1:
                preds[i] = valid[0][0]
                sources[i] = 'FC_SINGLE'
                n_vint[i] = 1
                has_curve_correction[i] = valid[0][2]
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
    out['Curve_Corrected'] = has_curve_correction
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

def score(pred_df):
    metrics = {}
    overall_w, overall_n = _wmape(pred_df['Actual_Sales'], pred_df['Prediction'])
    metrics['overall'] = {
        'wmape': overall_w,
        'bias': _bias(pred_df['Actual_Sales'], pred_df['Prediction']),
        'n': overall_n,
        'rows': len(pred_df),
    }
    for src in ['FC_MULTI', 'FC_SINGLE', 'HISTORICAL']:
        sub = pred_df[pred_df['Prediction_Source'] == src]
        w, n = _wmape(sub['Actual_Sales'], sub['Prediction'])
        metrics[src] = {
            'wmape': w,
            'bias': _bias(sub['Actual_Sales'], sub['Prediction']),
            'n': n,
            'rows': len(sub),
        }
    metrics['NO_PREDICTION'] = {
        'rows': int(pred_df['Prediction'].isna().sum()),
    }
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

    # Test: target window starts in [snap, snap+horizon), ref_month strictly before snap
    test = df[
        (df['Target_Period_Start'] >= snap) &
        (df['Target_Period_Start'] < horizon_end) &
        (df['Reference_Month'] < snap)
    ].copy()

    # Registry: any FC vintage with ref_month < snap (recent vintages OK even with no history)
    visible = df[df['Reference_Month'] < snap]

    if verbose:
        print(f"\n  Snapshot {snap.date()} (horizon {horizon_months}mo):")
        print(f"    Train rows (outcomes known): {len(train):,}")
        print(f"    Test rows (in horizon):      {len(test):,}")
        print(f"    Visible FC-bearing rows:     {int((visible['Has_Forecast']==1).sum()):,}")

    if len(train) == 0 or len(test) == 0:
        return None, None

    cov_curve = build_flat_curves(train, 'fc_coverage')
    bias_curve = build_flat_curves(train, 'fc_bias')
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
    print(f"    {'Source':<14} {'Rows':>8} {'WMAPE':>8} {'Bias':>10}")
    print(f"    {'-'*14} {'-'*8} {'-'*8} {'-'*10}")
    for src in ['overall', 'FC_MULTI', 'FC_SINGLE', 'HISTORICAL']:
        m = metrics[src]
        rows = m.get('rows', 0)
        print(f"    {src:<14} {rows:>8,} {_fmt_wmape(m.get('wmape')):>8} {_fmt_pct(m.get('bias')):>10}")
    print(f"    {'NO_PREDICTION':<14} {metrics['NO_PREDICTION']['rows']:>8,}")

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
    print(f"Filters: Prediction_Lag in {sorted(ALLOWED_LAGS)}, Timeframe in {sorted(ALLOWED_TIMEFRAMES)}")

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
            'fc_multi_n': metrics['FC_MULTI']['n'],
            'fc_single_n': metrics['FC_SINGLE']['n'],
            'historical_n': metrics['HISTORICAL']['n'],
            'no_pred_n': metrics['NO_PREDICTION']['rows'],
        })
        if args.save_predictions:
            pred_df = pred_df.copy()
            pred_df['Snapshot'] = snap
            all_preds.append(pred_df)

    if len(grid_summary) > 1:
        print("\n" + "=" * 90)
        print("Snapshot grid summary")
        print("=" * 90)
        print(f"  {'Snapshot':<12} {'Rows':>8} {'WMAPE':>8} {'Bias':>10}  "
              f"{'FC_MULTI':>9} {'FC_SINGLE':>9} {'HIST':>8} {'NoPred':>7}")
        for r in grid_summary:
            print(f"  {r['snapshot']:<12} {r['rows']:>8,} {_fmt_wmape(r['wmape']):>8} "
                  f"{_fmt_pct(r['bias']):>10}  {r['fc_multi_n']:>9,} {r['fc_single_n']:>9,} "
                  f"{r['historical_n']:>8,} {r['no_pred_n']:>7,}")

        # Aggregate (volume-weighted across snapshots)
        wmapes = [r['wmape'] for r in grid_summary if r['wmape'] is not None and not np.isnan(r['wmape'])]
        biases = [r['bias'] for r in grid_summary if r['bias'] is not None and not np.isnan(r['bias'])]
        if wmapes:
            print(f"\n  Mean WMAPE across snapshots: {np.mean(wmapes)*100:.2f}%")
            print(f"  Mean Bias  across snapshots: {np.mean(biases)*100:+.2f}%")

    if args.save_predictions and all_preds:
        out = pd.concat(all_preds, ignore_index=True)
        out.to_csv(args.save_predictions, index=False)
        print(f"\nWrote per-row predictions to {args.save_predictions}")

if __name__ == '__main__':
    main()
