"""
OO Velocity Signal Diagnostic
==============================
For each target period, we observe OO at multiple lags. The question:
does the *rate of change* in OO (relative to curve expectations) predict
actual sales better than OO level alone?

Three signals tested:
  A. Single-lag OO:  implied_actual = OO(lag) / curve(lag)           [baseline]
  B. Multi-lag OO:   weighted avg of OO(L)/curve(L) across lags     [multi-vintage OO]
  C. OO velocity:    is OO arriving faster/slower than the curve expects?
                     velocity_ratio = actual_delta / expected_delta
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

RECENCY_POWER = 2
MIN_OBS_FLAT = 3

df = pd.read_csv('Dummy Training Data Customer 1 Try 2 11-Feb-2026.csv',
                 parse_dates=['Reference_Month', 'Target_Period_Start', 'Target_Period_End'])
if 'Unnamed: 0' in df.columns: df = df.drop(columns=['Unnamed: 0'])
if 'GSA' not in df.columns: df['GSA'] = 'Customer 1'
df['gsa_site'] = df['GSA'].astype(str) + '|' + df['Site'].astype(str)
df['oo_ratio'] = np.where(df['Actual_Sales'] > 0, df['Open_Orders'] / df['Actual_Sales'], np.nan)

clean_sites = [gs for gs in sorted(df['gsa_site'].unique()) if 'Site E' not in gs]
top_sites = ['Customer 1|Site W', 'Customer 1|Site J', 'Customer 1|Site S', 'Customer 1|Site T']

cutoff = df['Reference_Month'].quantile(0.8)
train = df[df['Reference_Month'] <= cutoff].copy()
test = df[df['Reference_Month'] > cutoff].copy()

# ---- Build flat OO curves ----
def build_flat_curves(train_df, ratio_col, start=None):
    td = train_df if start is None else train_df[train_df['Reference_Month'] >= start]
    flat = {}
    for (gs, tf, lag), grp in td.groupby(['gsa_site', 'Timeframe', 'Prediction_Lag']):
        valid = grp[grp[ratio_col].notna() & (grp[ratio_col] > 0) & (grp[ratio_col] < 10)]
        if len(valid) < MIN_OBS_FLAT: continue
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

def get_curve(flat, gs, tf, lag):
    for key in [(gs, tf, lag), ('FB', gs, lag)]:
        if key in flat and flat[key] > 0.001:
            return flat[key]
    return None

oo_flat = build_flat_curves(train, 'oo_ratio', start='2023-01-01')

# ============================================================
# BUILD OO TARGET-PERIOD REGISTRY
# ============================================================
# For each (gsa_site, TP_Start, TP_End), collect all OO observations across lags
print("=" * 100)
print("OO VELOCITY SIGNAL DIAGNOSTIC")
print("=" * 100)

def build_oo_registry(data_df):
    """Index OO observations by target period.
    Returns {(gs, TP_Start, TP_End): [{'lag': L, 'oo': OO, 'ref_month': RM, 'tf': TF, 'actual': A}, ...]}
    sorted by lag ascending (lag=1 is closest to target).
    """
    registry = {}
    for _, row in data_df.iterrows():
        if row['Open_Orders'] <= 0 or pd.isna(row['Actual_Sales']) or row['Actual_Sales'] <= 0:
            continue
        key = (row['gsa_site'], row['Target_Period_Start'], row['Target_Period_End'])
        if key not in registry:
            registry[key] = []
        registry[key].append({
            'lag': row['Prediction_Lag'],
            'oo': row['Open_Orders'],
            'ref_month': row['Reference_Month'],
            'tf': row['Timeframe'],
            'actual': row['Actual_Sales'],
            'gsa_site': row['gsa_site'],
        })
    for key in registry:
        registry[key].sort(key=lambda v: v['lag'])
    return registry

# Build from full dataset (causality enforced at lookup time)
oo_registry = build_oo_registry(df)

# Stats on vintage depth
depths = [len(v) for v in oo_registry.values()]
print(f"\n  OO registry: {len(oo_registry)} target periods")
print(f"  Vintage depth: mean={np.mean(depths):.1f}, median={np.median(depths):.0f}, "
      f"max={max(depths)}, 1-vintage={sum(1 for d in depths if d==1)}")

# ============================================================
# SIGNAL A: Single-lag OO (baseline) - already have this
# ============================================================

# ============================================================
# SIGNAL B: Multi-vintage OO
# ============================================================
# For each test row, look up all causally-valid OO observations for its
# target period, convert each to an implied actual, and combine.

VINTAGE_POWER = 1.5  # inverse-lag weighting

def apply_oo_multi_vintage(dset, oo_f, registry, causality='temporal'):
    """Multi-vintage OO: combine OO/curve(L) across all available lags."""
    n = len(dset)
    oo_single = np.full(n, np.nan)   # current lag only
    oo_multi = np.full(n, np.nan)    # all available lags combined
    n_vintages = np.zeros(n, dtype=int)

    for i, (idx, row) in enumerate(dset.iterrows()):
        gs, tf, lag = row['gsa_site'], row['Timeframe'], row['Prediction_Lag']
        tp_key = (gs, row['Target_Period_Start'], row['Target_Period_End'])

        # Single-vintage (baseline)
        c = get_curve(oo_f, gs, tf, lag)
        if c is not None and row['Open_Orders'] > 0:
            oo_single[i] = row['Open_Orders'] / c

        # Multi-vintage
        if tp_key not in registry:
            continue

        estimates = []
        for v in registry[tp_key]:
            # Causality: only use observations from before this reference month
            if causality == 'temporal' and v['ref_month'] > row['Reference_Month']:
                continue
            # Also skip observations at the same or later lag (redundant/future)
            # Actually we want ALL lags that are causally valid (observed before this ref_month)
            cv = get_curve(oo_f, v['gsa_site'], v['tf'], v['lag'])
            if cv is not None and v['oo'] > 0:
                implied = v['oo'] / cv
                estimates.append((max(implied, 0), v['lag']))

        if estimates:
            # Inverse-lag weighting: closer lags get more weight
            total_w, total_v = 0.0, 0.0
            for est, orig_lag in estimates:
                w = 1.0 / (orig_lag ** VINTAGE_POWER)
                total_w += w
                total_v += w * est
            oo_multi[i] = total_v / total_w if total_w > 0 else np.nan
            n_vintages[i] = len(estimates)

    dset['oo_implied_single'] = np.maximum(oo_single, 0)
    dset['oo_implied_multi'] = np.maximum(oo_multi, 0)
    dset['oo_n_vintages'] = n_vintages

print("\n--- Applying multi-vintage OO ---")
apply_oo_multi_vintage(train, oo_flat, oo_registry, 'temporal')
apply_oo_multi_vintage(test, oo_flat, oo_registry, 'temporal')

n_s = test['oo_implied_single'].notna().sum()
n_m = test['oo_implied_multi'].notna().sum()
print(f"  Single-vintage OO coverage: {n_s}/{len(test)} ({100*n_s/len(test):.0f}%)")
print(f"  Multi-vintage OO coverage:  {n_m}/{len(test)} ({100*n_m/len(test):.0f}%)")

vdepth = test.loc[test['oo_n_vintages'] > 0, 'oo_n_vintages']
print(f"  Vintage depth: mean={vdepth.mean():.1f}, median={vdepth.median():.0f}, "
      f"max={vdepth.max():.0f}")

# ============================================================
# SIGNAL C: OO Velocity
# ============================================================
# For each test row at lag L, find the OO at lag L+1 for the same target period.
# Compute:
#   expected_delta = curve(L) - curve(L+1)   [fraction of sales expected to arrive]
#   actual_delta   = (OO(L) - OO(L+1)) / OO_implied_single(L)  [actual fraction that arrived]
#   velocity_ratio = actual_delta / expected_delta
# If velocity_ratio > 1: orders arriving faster than expected → bullish

def compute_oo_velocity(dset, oo_f, registry, causality='temporal'):
    """Compute OO velocity signal for each row."""
    n = len(dset)
    velocity_ratio = np.full(n, np.nan)
    delta_implied = np.full(n, np.nan)  # OO-implied revision: IA(L) / IA(L+1)

    for i, (idx, row) in enumerate(dset.iterrows()):
        gs, tf, lag = row['gsa_site'], row['Timeframe'], row['Prediction_Lag']
        tp_key = (gs, row['Target_Period_Start'], row['Target_Period_End'])

        if tp_key not in registry or lag >= 12:
            continue

        # Current OO and curve
        oo_cur = row['Open_Orders']
        c_cur = get_curve(oo_f, gs, tf, lag)
        if c_cur is None or oo_cur <= 0:
            continue

        # Find the previous observation (lag+1) for this target period
        prev_obs = None
        for v in registry[tp_key]:
            if v['lag'] == lag + 1:
                if causality == 'temporal' and v['ref_month'] > row['Reference_Month']:
                    continue
                prev_obs = v
                break

        if prev_obs is None:
            continue

        oo_prev = prev_obs['oo']
        c_prev = get_curve(oo_f, gs, prev_obs['tf'], lag + 1)
        if c_prev is None or oo_prev <= 0:
            continue

        # Expected delta in OO ratio
        expected_delta = c_cur - c_prev  # should be positive (OO grows as lag shrinks)
        # Actual delta in OO (as fraction of implied actual)
        ia_cur = oo_cur / c_cur
        ia_prev = oo_prev / c_prev
        actual_delta = (oo_cur - oo_prev) / ia_cur if ia_cur > 0 else 0

        if abs(expected_delta) > 0.001:
            velocity_ratio[i] = actual_delta / expected_delta

        # Simpler signal: revision ratio between consecutive implied actuals
        if ia_prev > 0:
            delta_implied[i] = ia_cur / ia_prev

    dset['oo_velocity_ratio'] = velocity_ratio
    dset['oo_revision_ratio'] = delta_implied

print("\n--- Computing OO velocity ---")
compute_oo_velocity(train, oo_flat, oo_registry, 'temporal')
compute_oo_velocity(test, oo_flat, oo_registry, 'temporal')

n_vel = test['oo_velocity_ratio'].notna().sum()
n_rev = test['oo_revision_ratio'].notna().sum()
print(f"  Velocity ratio coverage: {n_vel}/{len(test)} ({100*n_vel/len(test):.0f}%)")
print(f"  Revision ratio coverage: {n_rev}/{len(test)} ({100*n_rev/len(test):.0f}%)")

# ============================================================
# SIGNAL QUALITY COMPARISON
# ============================================================
print(f"\n\n{'='*100}")
print("SIGNAL QUALITY: Single vs Multi-vintage OO")
print("=" * 100)

t = test[test['gsa_site'].isin(clean_sites) & (test['Actual_Sales'] > 0)]

print(f"\n  {'Signal':>25s} | {'WMAPE':>7s} | {'Bias':>7s} | {'R²':>6s} | {'N':>5s}")
print("  " + "-" * 60)
for label, col in [('OO single-lag', 'oo_implied_single'),
                    ('OO multi-vintage', 'oo_implied_multi')]:
    v = t[t[col].notna() & (t[col] > 0)]
    y, p = v['Actual_Sales'].values, v[col].values
    w = np.sum(np.abs(y - p)) / np.sum(y) * 100
    b = np.mean((p - y) / y) * 100
    r2 = 1 - np.sum((y - p)**2) / np.sum((y - np.mean(y))**2)
    print(f"  {label:>25s} | {w:>5.1f}% | {b:>+5.1f}% | {r2:>.3f} | {len(v):>5d}")

# Per-site
print(f"\n  Per-site WMAPE:")
print(f"  {'Site':>15s} | {'Single':>12s} | {'Multi':>12s} | {'Delta':>8s} | {'Avg depth':>9s}")
print("  " + "-" * 65)
for gs in top_sites:
    site = gs.split('|')[1]
    sub = t[t['gsa_site'] == gs]
    sw, sb, mw, mb = 0, 0, 0, 0
    for col, lbl in [('oo_implied_single', 's'), ('oo_implied_multi', 'm')]:
        v = sub[sub[col].notna() & (sub[col] > 0)]
        if len(v) > 0:
            w = np.sum(np.abs(v['Actual_Sales'] - v[col])) / np.sum(v['Actual_Sales']) * 100
            b = np.mean((v[col] - v['Actual_Sales']) / v['Actual_Sales']) * 100
            if lbl == 's': sw, sb = w, b
            else: mw, mb = w, b
    depth = sub.loc[sub['oo_n_vintages'] > 0, 'oo_n_vintages'].mean()
    print(f"  {site:>15s} | {sw:>5.1f}%{sb:>+5.0f}% | {mw:>5.1f}%{mb:>+5.0f}% | "
          f"{mw-sw:>+5.1f}pp | {depth:>7.1f}")

# Per-lag
print(f"\n  Per-lag WMAPE:")
print(f"  {'Lag':>4s} | {'Single':>8s} | {'Multi':>8s} | {'Delta':>6s} | {'Depth':>5s}")
print("  " + "-" * 42)
for lag in sorted(t['Prediction_Lag'].unique()):
    sub = t[t['Prediction_Lag'] == lag]
    vals = {}
    for lbl, col in [('s', 'oo_implied_single'), ('m', 'oo_implied_multi')]:
        v = sub[sub[col].notna() & (sub[col] > 0)]
        if len(v) > 0:
            vals[lbl] = np.sum(np.abs(v['Actual_Sales'] - v[col])) / np.sum(v['Actual_Sales']) * 100
    depth = sub.loc[sub['oo_n_vintages'] > 0, 'oo_n_vintages'].mean()
    if 's' in vals and 'm' in vals:
        print(f"  {lag:>4d} | {vals['s']:>6.1f}% | {vals['m']:>6.1f}% | "
              f"{vals['m']-vals['s']:>+4.1f} | {depth:>5.1f}")


# ============================================================
# VELOCITY SIGNAL ANALYSIS
# ============================================================
print(f"\n\n{'='*100}")
print("VELOCITY SIGNAL: Does OO arrival rate predict error?")
print("=" * 100)

# Revision ratio analysis
tv = t[t['oo_revision_ratio'].notna() & t['oo_implied_single'].notna() & (t['oo_implied_single'] > 0)].copy()
tv['oo_error'] = (tv['oo_implied_single'] - tv['Actual_Sales']) / tv['Actual_Sales']

print(f"\n  Revision ratio stats (IA_current / IA_previous, N={len(tv)}):")
rr = tv['oo_revision_ratio']
print(f"    Mean: {rr.mean():.3f}  Median: {rr.median():.3f}  Std: {rr.std():.3f}")
print(f"    p25: {rr.quantile(0.25):.3f}  p75: {rr.quantile(0.75):.3f}")

# Correlation: does revision ratio predict error?
corr = tv['oo_revision_ratio'].corr(tv['oo_error'])
print(f"\n  Correlation(revision_ratio, OO_error): {corr:.3f}")
corr_abs = tv['oo_revision_ratio'].corr(tv['oo_error'].abs())
print(f"  Correlation(revision_ratio, |OO_error|): {corr_abs:.3f}")

# Correlation with actual
corr_actual = tv['oo_revision_ratio'].corr(tv['Actual_Sales'])
print(f"  Correlation(revision_ratio, Actual_Sales): {corr_actual:.3f}")

# Bucket analysis: split revision ratio into quintiles, see WMAPE in each
print(f"\n  WMAPE by revision ratio quintile:")
print(f"  {'Quintile':>10s} | {'Range':>18s} | {'WMAPE':>7s} | {'Bias':>7s} | {'N':>4s}")
print("  " + "-" * 55)
tv['rr_quintile'] = pd.qcut(tv['oo_revision_ratio'], 5, labels=False, duplicates='drop')
for q in sorted(tv['rr_quintile'].unique()):
    sub = tv[tv['rr_quintile'] == q]
    rng = f"[{sub['oo_revision_ratio'].min():.2f}, {sub['oo_revision_ratio'].max():.2f}]"
    w = np.sum(np.abs(sub['Actual_Sales'] - sub['oo_implied_single'])) / np.sum(sub['Actual_Sales']) * 100
    b = np.mean(sub['oo_error']) * 100
    print(f"  {'Q'+str(q+1):>10s} | {rng:>18s} | {w:>5.1f}% | {b:>+5.1f}% | {len(sub):>4d}")

# Velocity ratio analysis
print(f"\n  Velocity ratio stats (actual_delta / expected_delta, "
      f"N={t['oo_velocity_ratio'].notna().sum()}):")
vr = t.loc[t['oo_velocity_ratio'].notna(), 'oo_velocity_ratio']
# Clip extreme outliers for stats
vr_clip = vr.clip(-10, 10)
print(f"    Mean: {vr_clip.mean():.3f}  Median: {vr_clip.median():.3f}  Std: {vr_clip.std():.3f}")

# Per-lag: does revision signal improve with more history?
print(f"\n  Revision ratio predictive power by lag:")
print(f"  {'Lag':>4s} | {'corr(RR,err)':>12s} | {'RR mean':>7s} | {'N':>4s}")
print("  " + "-" * 35)
for lag in sorted(tv['Prediction_Lag'].unique()):
    sub = tv[tv['Prediction_Lag'] == lag]
    if len(sub) >= 10:
        c = sub['oo_revision_ratio'].corr(sub['oo_error'])
        print(f"  {lag:>4d} | {c:>+10.3f}   | {sub['oo_revision_ratio'].mean():>7.3f} | {len(sub):>4d}")

# Per-site: is the signal consistent?
print(f"\n  Revision ratio predictive power by site:")
print(f"  {'Site':>15s} | {'corr(RR,err)':>12s} | {'RR mean':>7s} | {'N':>4s}")
print("  " + "-" * 45)
for gs in top_sites:
    site = gs.split('|')[1]
    sub = tv[tv['gsa_site'] == gs]
    if len(sub) >= 10:
        c = sub['oo_revision_ratio'].corr(sub['oo_error'])
        print(f"  {site:>15s} | {c:>+10.3f}   | {sub['oo_revision_ratio'].mean():>7.3f} | {len(sub):>4d}")


# ============================================================
# ADJUSTED OO SIGNAL: Use velocity to correct single-lag OO
# ============================================================
print(f"\n\n{'='*100}")
print("ADJUSTED OO: Correcting single-lag OO with velocity")
print("=" * 100)

# Simple correction: oo_implied_adjusted = oo_implied_single * revision_ratio
# (If revision ratio > 1, recent data says actual is higher than single-lag estimate)
t2 = t.copy()
t2['oo_implied_adjusted'] = np.where(
    t2['oo_revision_ratio'].notna() & t2['oo_implied_single'].notna(),
    t2['oo_implied_single'] * t2['oo_revision_ratio'],
    t2['oo_implied_single'])
t2['oo_implied_adjusted'] = np.maximum(t2['oo_implied_adjusted'], 0)

# Damped version: oo_implied_damped = oo_implied_single * (revision_ratio ^ 0.5)
t2['oo_implied_damped'] = np.where(
    t2['oo_revision_ratio'].notna() & t2['oo_implied_single'].notna() & (t2['oo_revision_ratio'] > 0),
    t2['oo_implied_single'] * (t2['oo_revision_ratio'] ** 0.5),
    t2['oo_implied_single'])
t2['oo_implied_damped'] = np.maximum(t2['oo_implied_damped'], 0)

print(f"\n  {'Signal':>25s} | {'WMAPE':>7s} | {'Bias':>7s} | {'R²':>6s} | {'N':>5s}")
print("  " + "-" * 60)
for label, col in [('OO single-lag', 'oo_implied_single'),
                    ('OO multi-vintage', 'oo_implied_multi'),
                    ('OO adjusted (RR^1.0)', 'oo_implied_adjusted'),
                    ('OO damped (RR^0.5)', 'oo_implied_damped')]:
    v = t2[t2[col].notna() & (t2[col] > 0)]
    y, p = v['Actual_Sales'].values, v[col].values
    w = np.sum(np.abs(y - p)) / np.sum(y) * 100
    b = np.mean((p - y) / y) * 100
    r2 = 1 - np.sum((y - p)**2) / np.sum((y - np.mean(y))**2)
    print(f"  {label:>25s} | {w:>5.1f}% | {b:>+5.1f}% | {r2:>.3f} | {len(v):>5d}")

# Per-lag for the best variants
print(f"\n  Per-lag WMAPE (single vs multi-vintage vs damped):")
print(f"  {'Lag':>4s} | {'Single':>8s} | {'Multi':>8s} | {'Damped':>8s} | {'S→M':>6s} | {'S→D':>6s}")
print("  " + "-" * 55)
for lag in sorted(t2['Prediction_Lag'].unique()):
    sub = t2[t2['Prediction_Lag'] == lag]
    vals = {}
    for lbl, col in [('s', 'oo_implied_single'), ('m', 'oo_implied_multi'),
                     ('d', 'oo_implied_damped')]:
        v = sub[sub[col].notna() & (sub[col] > 0)]
        if len(v) > 0:
            vals[lbl] = np.sum(np.abs(v['Actual_Sales'] - v[col])) / np.sum(v['Actual_Sales']) * 100
    if all(k in vals for k in ['s', 'm', 'd']):
        print(f"  {lag:>4d} | {vals['s']:>6.1f}% | {vals['m']:>6.1f}% | {vals['d']:>6.1f}% | "
              f"{vals['m']-vals['s']:>+4.1f} | {vals['d']-vals['s']:>+4.1f}")

print("\nDone!")
