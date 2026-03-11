"""
V8h: Pre-Curve LT Normalization (Site-Specific)
================================================
Instead of conditioning curves on LT (V8g -- overfits with per-cell regression),
normalize Open Orders by LT BEFORE applying flat aging curves.

    OO_norm = OO * (LT_ref / LT_current) ^ beta_site

Then flat curves operate on LT-neutral signals.
FC curves left flat (stable per development log).

Falls back to beta=0 (pure V8d) when LT signal is weak for a site.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from numpy.linalg import lstsq

RECENCY_POWER = 2
MIN_OBS_FLAT = 3
MIN_OBS_ELASTICITY = 50   # Need substantial data to learn site-level elasticity
MIN_R2_ELASTICITY = 0.02  # Minimum R² to trust the elasticity

df = pd.read_excel('/mnt/user-data/uploads/Dummy_Training_Data_Try_2_11-Feb-2026.xlsx')

if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

if 'GSA' not in df.columns:
    df['GSA'] = 'Customer 1'

df['gsa_site'] = df['GSA'].astype(str) + '|' + df['Site'].astype(str)
df['Avg_Weighted_Order_Earliness'] = df['Avg_Weighted_Order_Earliness'].fillna(0)
df['Avg_Weighted_Lead_Time'] = df['Avg_Weighted_Lead_Time'].fillna(0)
df = df[~df['Site'].isin(['Site A'])].copy()

df['oo_ratio'] = np.where(df['Actual_Sales'] > 0, df['Open_Orders'] / df['Actual_Sales'], np.nan)
df['fc_coverage'] = np.where(df['Actual_Sales'] > 0, df['Covered_Orders'] / df['Actual_Sales'], np.nan)
df['fc_bias'] = np.where(df['Covered_Orders'] > 0, df['Forecast_Value'] / df['Covered_Orders'], np.nan)

clean_sites = [gs for gs in sorted(df['gsa_site'].unique()) if 'Site E' not in gs]
top_sites = ['Customer 1|Site W', 'Customer 1|Site J', 'Customer 1|Site S', 'Customer 1|Site T']

cutoff = df['Reference_Month'].quantile(0.8)
train = df[df['Reference_Month'] <= cutoff].copy()
test = df[df['Reference_Month'] > cutoff].copy()
test_months = sorted(test['Reference_Month'].unique())
gsa_sites = sorted(df['gsa_site'].unique())


# ============================================================
# FLAT CURVE BUILDER (shared by V8d baseline and V8h)
# ============================================================

def build_flat_curves(train_df, ratio_col, start=None):
    """Build recency-weighted flat curves per (site, TF, lag) with TF-pooled fallbacks.
    Returns: {(gs, tf, lag): value, ('FB', gs, lag): value}
    """
    td = train_df if start is None else train_df[train_df['Reference_Month'] >= start]
    flat = {}

    # Per-cell curves
    for (gs, tf, lag), grp in td.groupby(['gsa_site', 'Timeframe', 'Prediction_Lag']):
        valid = grp[grp[ratio_col].notna() & (grp[ratio_col] > 0) & (grp[ratio_col] < 10)]
        if len(valid) < MIN_OBS_FLAT:
            continue
        days = (valid['Reference_Month'] - valid['Reference_Month'].min()).dt.days / 365
        wts = (1 + days.values) ** RECENCY_POWER
        flat[(gs, tf, lag)] = np.average(valid[ratio_col], weights=wts)

    # Fallback curves (pooled across TFs)
    for (gs, lag), grp in td.groupby(['gsa_site', 'Prediction_Lag']):
        valid = grp[grp[ratio_col].notna() & (grp[ratio_col] > 0) & (grp[ratio_col] < 10)]
        if len(valid) >= MIN_OBS_FLAT:
            days = (valid['Reference_Month'] - valid['Reference_Month'].min()).dt.days / 365
            wts = (1 + days.values) ** RECENCY_POWER
            flat[('FB', gs, lag)] = np.average(valid[ratio_col], weights=wts)

    return flat


def get_flat_value(flat, gs, tf, lag):
    """Look up flat curve: cell-level first, then TF-pooled fallback."""
    for key in [(gs, tf, lag), ('FB', gs, lag)]:
        if key in flat:
            return flat[key]
    return None


# ============================================================
# STEP 1: LEARN SITE-SPECIFIC LT ELASTICITY
# ============================================================

def learn_lt_elasticity(train_df, flat_curves, start=None):
    """Learn per-site power-law elasticity: how OO ratio scales with LT.

    For each site, pools across all (TF, lag) cells:
        deviation = oo_ratio / flat_curve_value   (removes cell base rate)
        log(deviation) ~ beta_lt * log(LT / LT_ref)

    Returns: {gsa_site: {'beta_lt', 'lt_ref', 'r2', 'n_obs'}}
    """
    td = train_df if start is None else train_df[train_df['Reference_Month'] >= start]
    elasticities = {}

    for gs in sorted(td['gsa_site'].unique()):
        site_data = td[td['gsa_site'] == gs].copy()

        # For each row, get the flat curve value and compute deviation
        devs = []
        lts = []
        wts = []

        ref_month_min = site_data['Reference_Month'].min()

        for _, row in site_data.iterrows():
            if np.isnan(row['oo_ratio']) or row['oo_ratio'] <= 0 or row['oo_ratio'] >= 10:
                continue
            flat_val = get_flat_value(flat_curves, gs, row['Timeframe'], row['Prediction_Lag'])
            if flat_val is None or flat_val <= 0.001:
                continue
            lt = row['Avg_Weighted_Lead_Time']
            if lt <= 0.1:
                continue

            dev = row['oo_ratio'] / flat_val
            days = (row['Reference_Month'] - ref_month_min).days / 365
            w = (1 + days) ** RECENCY_POWER

            devs.append(dev)
            lts.append(lt)
            wts.append(w)

        devs = np.array(devs)
        lts = np.array(lts)
        wts = np.array(wts)

        if len(devs) < MIN_OBS_ELASTICITY:
            elasticities[gs] = {'beta_lt': 0.0, 'lt_ref': 1.0, 'r2': 0.0, 'n_obs': len(devs)}
            continue

        # LT reference = recency-weighted mean
        lt_ref = np.average(lts, weights=wts)

        # Fit in log space: log(deviation) ~ beta * log(LT / LT_ref)
        log_dev = np.log(devs)
        log_lt_ratio = np.log(lts / lt_ref)

        # Weighted OLS (single predictor, no intercept -- deviation should be 1.0 at LT_ref)
        W = np.sqrt(wts)
        Xw = W * log_lt_ratio
        yw = W * log_dev

        # beta = sum(Xw * yw) / sum(Xw * Xw)
        denom = np.sum(Xw * Xw)
        if denom < 1e-10:
            elasticities[gs] = {'beta_lt': 0.0, 'lt_ref': lt_ref, 'r2': 0.0, 'n_obs': len(devs)}
            continue

        beta_lt = np.sum(Xw * yw) / denom

        # R² (weighted)
        pred = beta_lt * log_lt_ratio
        ss_res = np.sum(wts * (log_dev - pred) ** 2)
        ss_tot = np.sum(wts * (log_dev - np.average(log_dev, weights=wts)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Gate: if R² too low or beta physically wrong (negative = higher LT means lower ratio??)
        if r2 < MIN_R2_ELASTICITY or beta_lt < 0:
            elasticities[gs] = {'beta_lt': 0.0, 'lt_ref': lt_ref, 'r2': r2, 'n_obs': len(devs)}
        else:
            elasticities[gs] = {'beta_lt': beta_lt, 'lt_ref': lt_ref, 'r2': r2, 'n_obs': len(devs)}

    return elasticities


# ============================================================
# STEP 2: NORMALIZE OO BY LT
# ============================================================

def normalize_oo_column(df, elasticities, oo_col='Open_Orders', actual_col='Actual_Sales'):
    """Normalize Open Orders by LT: OO_norm = OO * (LT_ref / LT)^beta.
    Also computes oo_ratio_norm = OO_norm / Actual (for training data curve building).
    """
    oo_norm = df[oo_col].values.copy().astype(float)
    oo_ratio_norm = np.full(len(df), np.nan)

    for i, (idx, row) in enumerate(df.iterrows()):
        gs = row['gsa_site']
        lt = row['Avg_Weighted_Lead_Time']

        if gs not in elasticities:
            continue

        e = elasticities[gs]
        beta = e['beta_lt']
        lt_ref = e['lt_ref']

        if beta == 0.0 or lt <= 0.1:
            continue

        # Power-law normalization
        adjustment = (lt_ref / lt) ** beta
        oo_norm[i] = row[oo_col] * adjustment

        if row[actual_col] > 0:
            oo_ratio_norm[i] = oo_norm[i] / row[actual_col]

    df['OO_norm'] = oo_norm
    df['oo_ratio_norm'] = oo_ratio_norm


# ============================================================
# APPLY CURVES (V8d flat baseline and V8h LT-normalized)
# ============================================================

def apply_flat_curves(dset, oo_flat, cov_flat, bias_flat, oo_col='Open_Orders',
                      oo_out='oo_implied_flat', fc_out='fc_implied_flat'):
    """Apply flat curves to compute implied actuals (V8d baseline path)."""
    oo_impl = np.full(len(dset), np.nan)
    fc_impl = np.full(len(dset), np.nan)

    for i, (idx, row) in enumerate(dset.iterrows()):
        gs, tf, lag = row['gsa_site'], row['Timeframe'], row['Prediction_Lag']

        # OO implied
        oo_curve = get_flat_value(oo_flat, gs, tf, lag)
        if oo_curve and oo_curve > 0.001:
            oo_impl[i] = row[oo_col] / oo_curve

        # FC implied
        cf = get_flat_value(cov_flat, gs, tf, lag)
        bf = get_flat_value(bias_flat, gs, tf, lag)
        if cf and bf and cf > 0.001 and bf > 0.001 and row['Forecast_Value'] > 0:
            fc_impl[i] = row['Forecast_Value'] / bf / cf

    dset[oo_out] = np.maximum(oo_impl, 0)
    dset[fc_out] = np.maximum(fc_impl, 0)


def apply_v8h_curves(dset, oo_norm_flat, cov_flat, bias_flat, elasticities):
    """Apply V8h: LT-normalize OO, then use normalized flat curves. FC unchanged."""
    oo_impl = np.full(len(dset), np.nan)
    fc_impl = np.full(len(dset), np.nan)

    for i, (idx, row) in enumerate(dset.iterrows()):
        gs, tf, lag = row['gsa_site'], row['Timeframe'], row['Prediction_Lag']
        lt = row['Avg_Weighted_Lead_Time']

        # OO implied (LT-normalized)
        oo_curve = get_flat_value(oo_norm_flat, gs, tf, lag)
        if oo_curve and oo_curve > 0.001:
            e = elasticities.get(gs, {'beta_lt': 0.0, 'lt_ref': 1.0})
            beta = e['beta_lt']
            lt_ref = e['lt_ref']

            lt_safe = max(lt, 0.1)
            oo_norm = row['Open_Orders'] * (lt_ref / lt_safe) ** beta
            oo_impl[i] = oo_norm / oo_curve

        # FC implied (unchanged -- flat curves on raw data)
        cf = get_flat_value(cov_flat, gs, tf, lag)
        bf = get_flat_value(bias_flat, gs, tf, lag)
        if cf and bf and cf > 0.001 and bf > 0.001 and row['Forecast_Value'] > 0:
            fc_impl[i] = row['Forecast_Value'] / bf / cf

    dset['oo_implied_v8h'] = np.maximum(oo_impl, 0)
    dset['fc_implied_v8h'] = np.maximum(fc_impl, 0)


# ============================================================
# WEIGHTED AVERAGE COMBINATION (unchanged from V8d)
# ============================================================

def weighted_avg(test_df, train_df, test_months, oo_col='oo_implied', fc_col='fc_implied'):
    results = test_df.copy()
    results['pred'] = np.nan

    hist = {}
    for gs in gsa_sites:
        for tf in train_df['Timeframe'].unique():
            sub = train_df[(train_df['gsa_site']==gs)&(train_df['Timeframe']==tf)]
            if len(sub)>0:
                d = (sub['Reference_Month']-sub['Reference_Month'].min()).dt.days/365
                w = (1+d.values)**RECENCY_POWER
                hist[(gs,tf)] = np.average(sub['Actual_Sales'], weights=w)

    for mi, month in enumerate(test_months):
        if mi==0:
            prior = train_df[train_df['Reference_Month'] >= train_df['Reference_Month'].quantile(0.8)]
        else:
            prior = results[results['Reference_Month'].isin(test_months[max(0,mi-3):mi])]

        cmask = results['Reference_Month']==month
        for gs in results.loc[cmask, 'gsa_site'].unique():
            for tf in results['Timeframe'].unique():
                ps = prior[(prior['gsa_site']==gs)&(prior['Timeframe']==tf)&(prior['Actual_Sales']>0)]
                oo_v = ps[ps[oo_col].notna()&(ps[oo_col]>0)]
                oo_mape = np.mean(np.abs(oo_v[oo_col]-oo_v['Actual_Sales'])/oo_v['Actual_Sales']) if len(oo_v)>=1 else 1.0
                fc_v = ps[ps[fc_col].notna()&(ps[fc_col]>0)]
                fc_mape = np.mean(np.abs(fc_v[fc_col]-fc_v['Actual_Sales'])/fc_v['Actual_Sales']) if len(fc_v)>=1 else 1.0

                rmask = cmask&(results['gsa_site']==gs)&(results['Timeframe']==tf)
                for idx in results[rmask].index:
                    row = results.loc[idx]
                    lag = row['Prediction_Lag']
                    has_oo = not np.isnan(row[oo_col]) and row[oo_col]>0
                    has_fc = not np.isnan(row[fc_col]) and row[fc_col]>0
                    oo_f = max(0.3, 1.0-0.06*(lag-1)) if has_oo else 0
                    fc_f = max(0.3, 1.0-0.06*(lag-1)) if has_fc else 0
                    w_oo = (1/max(oo_mape,0.01))*oo_f if has_oo else 0
                    w_fc = (1/max(fc_mape,0.01))*fc_f if has_fc else 0
                    hv = hist.get((gs,tf), 0)
                    w_h = (1/0.5)*0.1*lag/12 if hv>0 else 0
                    tot = w_oo+w_fc+w_h
                    if tot>0:
                        results.at[idx, 'pred'] = max(0,
                            (w_oo/tot)*(row[oo_col] if has_oo else 0)+
                            (w_fc/tot)*(row[fc_col] if has_fc else 0)+
                            (w_h/tot)*hv)
                    elif hv>0:
                        results.at[idx, 'pred'] = hv

    return results['pred']


# ============================================================
# MAIN PIPELINE
# ============================================================

print("=" * 100)
print("V8h: PRE-CURVE LT NORMALIZATION (SITE-SPECIFIC)")
print("=" * 100)

# --- V8d baseline: flat curves on raw data ---
print("\n--- Building V8d baseline (flat curves on raw OO) ---")
oo_flat_raw = build_flat_curves(train, 'oo_ratio', start='2023-01-01')
cov_flat = build_flat_curves(train, 'fc_coverage')
bias_flat = build_flat_curves(train, 'fc_bias')
print(f"  OO flat curves: {len(oo_flat_raw)}")
print(f"  FC cov curves:  {len(cov_flat)}")
print(f"  FC bias curves: {len(bias_flat)}")

# --- Step 1: Learn LT elasticity per site ---
print("\n--- Step 1: Learning site-specific LT elasticity ---")
elasticities = learn_lt_elasticity(train, oo_flat_raw, start='2023-01-01')

print(f"\n  {'Site':>25s} | {'β_LT':>7s} | {'LT_ref':>7s} | {'R²':>6s} | {'N obs':>6s} | {'Status':>10s}")
print("  " + "-" * 75)
for gs in sorted(elasticities.keys()):
    e = elasticities[gs]
    site = gs.split('|')[1] if '|' in gs else gs
    status = "ACTIVE" if e['beta_lt'] > 0 else "flat (β=0)"
    print(f"  {site:>25s} | {e['beta_lt']:>+6.3f} | {e['lt_ref']:>6.2f} | {e['r2']:>.3f} | {e['n_obs']:>5d} | {status:>10s}")

# --- Step 2: Normalize OO in training data ---
print("\n--- Step 2: Normalizing OO by LT in training data ---")
normalize_oo_column(train, elasticities)

# Show effect of normalization on ratio variance
print(f"\n  Ratio variance reduction (top sites):")
print(f"  {'Site':>15s} | {'Raw std':>9s} | {'Norm std':>9s} | {'Reduction':>10s}")
print("  " + "-" * 50)
for gs in top_sites:
    site = gs.split('|')[1]
    raw = train[(train['gsa_site']==gs) & train['oo_ratio'].notna() & (train['oo_ratio']>0) & (train['oo_ratio']<10)]
    norm = train[(train['gsa_site']==gs) & train['oo_ratio_norm'].notna() & (train['oo_ratio_norm']>0) & (train['oo_ratio_norm']<10)]
    if len(raw) > 0 and len(norm) > 0:
        raw_std = np.std(raw['oo_ratio'])
        norm_std = np.std(norm['oo_ratio_norm'])
        reduction = (1 - norm_std / raw_std) * 100 if raw_std > 0 else 0
        print(f"  {site:>15s} | {raw_std:>8.4f} | {norm_std:>8.4f} | {reduction:>+8.1f}%")

# --- Step 3: Build flat curves on normalized OO ratios ---
print("\n--- Step 3: Building flat curves on LT-normalized OO ratios ---")
oo_flat_norm = build_flat_curves(train, 'oo_ratio_norm', start='2023-01-01')
print(f"  OO normalized flat curves: {len(oo_flat_norm)}")

# --- Step 4: Apply curves ---
print("\n--- Step 4: Applying curves ---")

# V8d baseline
apply_flat_curves(train, oo_flat_raw, cov_flat, bias_flat)
apply_flat_curves(test, oo_flat_raw, cov_flat, bias_flat)

# V8h LT-normalized
apply_v8h_curves(train, oo_flat_norm, cov_flat, bias_flat, elasticities)
apply_v8h_curves(test, oo_flat_norm, cov_flat, bias_flat, elasticities)


# ============================================================
# SIGNAL QUALITY: V8d flat vs V8h LT-normalized
# ============================================================

print(f"\n\n{'='*100}")
print("SIGNAL QUALITY: OO flat vs OO LT-normalized")
print("=" * 100)

print(f"\n  {'Signal':>25s} | {'WMAPE':>7s} | {'Bias':>7s} | {'R²':>6s}")
print("  " + "-" * 55)
for label, col in [('OO flat (V8d)', 'oo_implied_flat'), ('OO LT-norm (V8h)', 'oo_implied_v8h'),
                    ('FC flat (shared)', 'fc_implied_flat'), ('FC flat (V8h)', 'fc_implied_v8h')]:
    v = test[test[col].notna() & (test[col]>0) & (test['Actual_Sales']>0) & test['gsa_site'].isin(clean_sites)]
    if len(v) == 0:
        continue
    y, p = v['Actual_Sales'].values, v[col].values
    w = np.sum(np.abs(y-p))/np.sum(y)*100
    b = np.mean((p-y)/y)*100
    r2 = 1 - np.sum((y-p)**2)/np.sum((y-np.mean(y))**2)
    print(f"  {label:>25s} | {w:>5.1f}% | {b:>+5.1f}% | {r2:>.3f}")

# Per-site OO signal quality
print(f"\n  OO implied per-site:")
print(f"  {'Site':>15s} | {'V8d flat':>12s} | {'V8h LT-norm':>12s} | {'Delta':>8s}")
print("  " + "-" * 55)
for gs in top_sites:
    site = gs.split('|')[1]
    fw, fb, cw, cb = 0, 0, 0, 0
    for col, lbl in [('oo_implied_flat', 'flat'), ('oo_implied_v8h', 'v8h')]:
        v = test[(test['gsa_site']==gs) & test[col].notna() & (test[col]>0) & (test['Actual_Sales']>0)]
        if len(v) > 0:
            w = np.sum(np.abs(v['Actual_Sales']-v[col]))/np.sum(v['Actual_Sales'])*100
            b = np.mean((v[col]-v['Actual_Sales'])/v['Actual_Sales'])*100
            if lbl == 'flat': fw, fb = w, b
            else: cw, cb = w, b
    print(f"  {site:>15s} | {fw:>5.1f}%{fb:>+5.0f}% | {cw:>5.1f}%{cb:>+5.0f}% | {cw-fw:>+5.1f}pp")


# ============================================================
# FULL MODEL: V8d vs V8h
# ============================================================

print(f"\n\n{'='*100}")
print("FULL MODEL: V8d (flat) vs V8h (LT-normalized)")
print("=" * 100)

# V8d baseline
print("\n  Running V8d (flat curves)...")
test['v8d'] = weighted_avg(test, train, test_months, 'oo_implied_flat', 'fc_implied_flat')

# V8h LT-normalized
print("  Running V8h (LT-normalized OO + flat FC)...")
test['v8h'] = weighted_avg(test, train, test_months, 'oo_implied_v8h', 'fc_implied_v8h')

m2 = test['Reference_Month'] > test_months[0]

print(f"\n  {'Model':>15s} | {'WMAPE':>7s} | {'Bias':>7s} | {'R²':>6s} | {'Acct':>6s}")
print("  " + "-" * 50)
for label, col in [('V8d (flat)', 'v8d'), ('V8h (LT-norm)', 'v8h')]:
    ts = test[m2 & test[col].notna() & test['gsa_site'].isin(clean_sites)]
    y, p = ts['Actual_Sales'].values, ts[col].values
    w = np.sum(np.abs(y-p))/np.sum(y)*100
    b = np.mean((p-y)/y)*100
    r2 = 1 - np.sum((y-p)**2)/np.sum((y-np.mean(y))**2)
    tf12 = ts[(ts['Timeframe']==12)&(ts['Prediction_Lag']==1)]
    by_m = tf12.groupby('Reference_Month').agg({'Actual_Sales':'sum'})
    by_m['pr'] = tf12.groupby('Reference_Month').apply(lambda g: test.loc[g.index, col].sum()).values
    aw = np.sum(np.abs(by_m['Actual_Sales']-by_m['pr']))/np.sum(by_m['Actual_Sales'])*100
    print(f"  {label:>15s} | {w:>5.1f}% | {b:>+5.1f}% | {r2:>.3f} | {aw:>4.1f}%")

# Per-site
print(f"\n  Per-site (months 2+):")
print(f"  {'Site':>20s} | {'V8d':>12s} | {'V8h':>12s} | {'Delta':>8s}")
print("  " + "-" * 60)
for gs in sorted(clean_sites):
    site = gs.split('|')[1]
    sub = test[m2 & (test['gsa_site']==gs) & (test['Actual_Sales']>0)]
    bv = sub[sub['v8d'].notna()]
    cv = sub[sub['v8h'].notna()]
    if len(bv)>0 and len(cv)>0:
        bw = np.sum(np.abs(bv['Actual_Sales']-bv['v8d']))/np.sum(bv['Actual_Sales'])*100
        cw = np.sum(np.abs(cv['Actual_Sales']-cv['v8h']))/np.sum(cv['Actual_Sales'])*100
        bb = np.mean((bv['v8d']-bv['Actual_Sales'])/bv['Actual_Sales'])*100
        cb = np.mean((cv['v8h']-cv['Actual_Sales'])/cv['Actual_Sales'])*100
        print(f"  {site:>20s} | {bw:>5.1f}%{bb:>+5.0f}% | {cw:>5.1f}%{cb:>+5.0f}% | {cw-bw:>+5.1f}pp")

# Per-lag
print(f"\n  WMAPE by lag:")
print(f"  {'Lag':>4s} | {'V8d':>8s} | {'V8h':>8s} | {'Delta':>8s}")
print("  " + "-" * 35)
for lag in sorted(test['Prediction_Lag'].unique()):
    sub = test[m2 & (test['Prediction_Lag']==lag) & (test['Actual_Sales']>0) & test['gsa_site'].isin(clean_sites)]
    bv = sub[sub['v8d'].notna()]
    cv = sub[sub['v8h'].notna()]
    if len(bv)>0 and len(cv)>0:
        bw = np.sum(np.abs(bv['Actual_Sales']-bv['v8d']))/np.sum(bv['Actual_Sales'])*100
        cw = np.sum(np.abs(cv['Actual_Sales']-cv['v8h']))/np.sum(cv['Actual_Sales'])*100
        print(f"  {lag:>4d} | {bw:>6.1f}% | {cw:>6.1f}% | {cw-bw:>+6.1f}pp")

# Account monthly
print(f"\n  Account TF=12/L=1:")
print(f"  {'Month':>12s} | {'Actual':>8s} | {'V8d':>15s} | {'V8h':>15s}")
print("  " + "-" * 60)
for ref in test_months[:7]:
    sub = test[(test['Reference_Month']==ref) & (test['Timeframe']==12) &
                (test['Prediction_Lag']==1) & test['gsa_site'].isin(clean_sites)]
    act = sub['Actual_Sales'].sum()
    bp = sub['v8d'].sum()
    cp = sub['v8h'].sum()
    if act > 0:
        print(f"  {ref.date()} | ${act/1e6:>5.0f}M | ${bp/1e6:>5.0f}M ({(bp-act)/act*100:>+5.1f}%) | ${cp/1e6:>5.0f}M ({(cp-act)/act*100:>+5.1f}%)")


print(f"\n\nV7 reference: WMAPE=8.8%  bias=+0.5%  Acct=2.2%")
print("\nDone!")
