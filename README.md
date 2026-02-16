# V8 Model Series — Demand Forecasting (Customer 1 & Customer 2)

## Overview

This project builds demand forecasting models that combine two signals — **Open Orders (OO)** and **Customer Forecasts (FC)** — to predict actual sales for future target periods. The models are evaluated on two customers with different data characteristics and forecast availability patterns.

### Key files

| File | Purpose |
|------|---------|
| `aging_v8g.py` | Customer 1 full pipeline: curve building, signal quality, V8d–V8k model variants, multi-vintage FC, random CV |
| `customer2_fc.py` | Customer 2 FC-only pipeline: multi-vintage FC, temporal eval, random CV |
| `Dummy Training Data Customer 1 Try 2 11-Feb-2026.csv` | Customer 1 data (~13,500 rows) |
| `Dummy_Training_Data_Customer_2_12-Feb-26.csv` | Customer 2 data (~19,000 rows) |

### V7 Reference (target to beat)
WMAPE = 8.8%, bias = +0.5%, account error = 2.2%

---

## Definitions

- **Lag**: Months from reference month (when you are looking) to the first month of the timeframe of interest. If looking at June–August 2026 from February 2026, lag = 4.
- **Timeframe (TF)**: The length of the target period in months (3, 6, 9, or 12).
- **Target period**: A specific `(Site, Target_Period_Start, Target_Period_End)` — the future sales window we're predicting.
- **Reference Month**: The month from which we make the prediction. Each target period is observed from multiple Reference_Months at different lags.
- **LT (Lead Time)**: Published time for an order to be fulfilled. A 90-day lead time means ~3 months from order placement to delivery. This is a supply-side parameter set by the supplier.
- **OE (Order Earliness)**: `order promise date - order creation date - published lead time`. Measures how far ahead of the minimum the customer placed their order. LT and OE tend to move in the same direction.
- **OO ratio**: `Open_Orders / Actual_Sales` — the fraction of eventual demand already visible as orders at a given lag.
- **FC coverage**: `Covered_Orders / Actual_Sales` — the fraction of actual sales covered by the customer's forecast.
- **FC bias**: `Forecast_Value / Covered_Orders` — the ratio of the forecast to the orders it covers.
- **Visibility threshold**: An order appears in Open Orders when `LT + OE > lag`. At large lags only early-placed orders are visible; at small lags most orders are in.
- **Vintage**: A specific forecast for a target period, identified by the Reference_Month when it was issued. The same target period may have forecasts from multiple Reference_Months (multiple vintages).

---

## Data Structure: How Target Periods Work

A critical structural feature of the data: **each target period is observed from multiple Reference_Months.** For example, Site B's Jul 2023–Jun 2024 target period (TF=12) appears as 12 rows — one for each Reference_Month from Jul 2022 through Jun 2023, at lags 12 down to 1.

Across these rows:
- **Actual_Sales is identical** — it's the realized outcome, known only after the period ends
- **Open_Orders increases** as lag decreases (orders accumulate: $475K at lag=12 → $1.58M at lag=1)
- **Forecast_Value is dynamic** across vintages — forecasts improve closer to the target ($1.44M at lag=9 → $3.08M at lag=4 → $1.91M at lag=1)
- **Has_Forecast varies** — Customer 2 forecasts quarterly (only 3 of 12 rows have forecasts), Customer 1 forecasts more frequently but still not every month

This means **each target period contains more forecast information than any single row reveals.**

---

## Key Insight: Order Visibility as Progress Through the Ordering Window

Open Orders at a given lag traces a **cumulative ordering curve** — the fraction of total demand that has been ordered as a function of where we are relative to the ordering window.

The ordering window length is approximately `LT + OE`. The "progress" through this window is:

```
progress = max(0, 1 - lag / (LT + OE))
```

- `progress ≈ 0` → beginning of cycle, few orders placed
- `progress ≈ 1` → end of cycle, most orders placed

This explains why LT changes break flat curves: V8d's flat curves key only on lag, not on how much of the ordering window has elapsed. When LT+OE drops from train to test, the same lag corresponds to less progress through a shorter window, but the flat curve still returns the old (higher) ratio.

---

## What We Tried: OO Curve Corrections (V8d–V8k)

### V8d — Flat Curves (Baseline)
- Per-cell `(site, timeframe, lag)` recency-weighted average of OO ratio
- **Results**: 16.5% WMAPE, -2.9% bias, 2.5% account
- Works well within the training LT regime, but cannot adapt when LT+OE shifts

### V8g — Conditioned Regression
- `OO_ratio = α + β₁·LT + β₂·OE` per cell
- **Results**: 35.8% WMAPE, +36.0% bias
- Failed: regression coefficients estimated on training-era LT range extrapolate badly to test-era LT values

### V8h — LT-Aware Gaussian Kernel (then LT+OE 2D Kernel)
- Gaussian kernel weighting: upweight training points with similar LT
- **Results**: 18.5–18.6% WMAPE
- Failed: when test LT is systematically lower than all training LT, even the "most similar" training points are far away. Degraded Sites S (+6.2pp) and T (+5.5pp)

### V8i — LT-Normalized Curves
- Normalize: `OO_ratio_norm = OO / (Actual_Sales × LT)`, then multiply back by current LT
- **Results**: 25.1% WMAPE, +35.5% bias
- Failed: assumes `OO_ratio ∝ LT` (proportional, no intercept). Actual relationship has a non-zero intercept (`OO_ratio ≈ a + b·LT`), so dividing by LT over-corrects

### V8j — Additive Effective Lag Shift
- `effective_lag = lag + (ref_LT_OE - current_LT_OE)`, look up flat curve at effective_lag
- **Results**: 26.9% WMAPE, +9.5% bias
- Failed: the shift is **lag-independent** but the LT+OE effect is **lag-dependent**. At short lags (well within the ordering window), almost no correction is needed. At lags near the visibility boundary, large correction is needed. The uniform shift over-corrects at short lags.

### V8k — Progress-Based Multiplicative Correction
- `correction = progress(current_LT_OE, lag) / progress(ref_LT_OE, lag)`
- `corrected_ratio = flat_ratio × correction`
- Assumes uniform ordering distribution within the window
- **Results**: 18.4% WMAPE, +1.6% bias, 2.3% account
- First approach to improve bias (-2.9% → +1.6%) and account (2.5% → 2.3%)
- Helps Sites D (-3.1pp), Americas (-3.7pp), J (-0.2pp), W (flat)
- Still hurts Sites S (+4.2pp) and T (+8.5pp) where LT+OE drops were largest

---

## Random CV Diagnostic: OO is Non-Stationary, FC is Stationary

To test whether the OO and FC signals are stationary (i.e., can be treated as random events), we ran **5-fold random cross-validation** alongside the standard temporal train/test split. In random CV, rows are shuffled randomly across all time periods — both train and test contain data from every month. This destroys temporal structure and reveals whether a model depends on time-stability of its signal.

### Customer 1 Results (before multi-vintage)

| Model | Random CV (5-fold mean ± std) | Temporal Split | Gap |
|-------|-------------------------------|---------------|-----|
| V8d (OO+FC) | 94.9% ± 3.3% (bias +55%) | 16.5% | **-78pp** |
| FC-only | 19.9% ± 0.4% (bias +21%) | 21.3% | +1.4pp |
| FC-debiased | 21.0% ± 0.5% (bias +28%) | 25.1% | +4.1pp |

Per-lag detail (fold 1 vs temporal):

| Lag | Rand V8d | Temp V8d | Rand FC | Temp FC |
|-----|----------|----------|---------|---------|
| 1 | 38% | 11% | 16% | 16% |
| 3 | 51% | 15% | 20% | 19% |
| 6 | 94% | 19% | 21% | 25% |
| 9 | 101% | 26% | 20% | 31% |
| 12 | 285% | 34% | 28% | 36% |

### Interpretation

**OO is deeply non-stationary.** V8d explodes from 16.5% to 95% under random CV. The OO-to-sales relationship changes so dramatically over time that a time-averaged curve is wrong for every individual period. The +55% systematic bias means OO_implied massively overstates actual sales on average.

**FC is nearly stationary.** FC-only barely moves (21.3% → 19.9%, slightly *better* under random CV). This confirms FC's relationship to actuals is stable over time — it genuinely can be treated as approximately random.

**FC debiasing works when there's no drift.** FC-debiased improves from 25.1% (temporal) to 21.0% (random). With random sampling there's no systematic difference between train and test distributions, so the correction factors are valid. In the temporal split, FC's bias subtly shifted, making historical correction factors wrong.

### Key takeaway

The V8d model's 16.5% WMAPE is **almost entirely carried by FC**. OO adds value only because the temporal split keeps curve drift manageable. In random CV, OO has *negative* value. This fundamentally changes the strategy: **FC should be the primary signal, with OO used carefully (if at all) and with temporal safeguards.**

---

## Multi-Vintage FC Architecture

### The problem: wasted forecast information

The original model treats each row `(Site, Reference_Month, Timeframe, Lag)` independently. When a row has `Has_Forecast=0`, it falls back to historical average — even though the **same target period** likely has a forecast from a different Reference_Month (a different "vintage" issued at a different lag).

Customer 2 issues forecasts quarterly (~13 of 46 months). At the row level, only 16% have forecasts. But at the **target period** level, 95% have at least one forecast vintage. Customer 1 has 46% row-level coverage but similarly benefits from looking across vintages.

### The fix: target-period-centric forecast lookup

Instead of asking "does this row have a forecast?", the model now asks "what forecasts are available for this target period?"

**Step 1 — Build a forecast registry.** Scan all rows where `Has_Forecast=1` and index them by target period `(Site, Target_Period_Start, Target_Period_End)`. Each entry stores the forecast's Reference_Month, Prediction_Lag, Forecast_Value, and Covered_Orders.

**Step 2 — For each row, collect causally-valid vintages.** Look up the row's target period in the registry. Filter to vintages where the forecast's `Reference_Month ≤` the row's `Reference_Month` (we can't use forecasts that haven't been issued yet).

**Step 3 — Convert each vintage to an implied-actual estimate.** Each vintage's forecast went through a coverage and bias process at its **original lag**:

```
implied_actual = Forecast_Value / bias(site, tf, original_lag) / coverage(site, tf, original_lag)
```

The curves are looked up at the vintage's own lag, not the current row's lag. This is critical — a forecast issued at lag=4 has lag=4 coverage/bias characteristics regardless of when we're reading it.

**Step 4 — Combine vintages with recency weighting.** Multiple estimates are combined as a weighted average:

```
weight = 1 / original_lag^1.5
```

| Original Lag | Weight | Interpretation |
|-------------|--------|----------------|
| 1 | 1.000 | Most recent, fully trusted |
| 3 | 0.192 | ~5x less than lag=1 |
| 6 | 0.068 | ~15x less than lag=1 |
| 9 | 0.037 | ~27x less than lag=1 |

This naturally emphasizes the most recent forecast while still incorporating older vintages as stabilizers.

**Step 5 — Use `best_vintage_lag` for the FC weight in the final blend.** The `weighted_avg` function blends FC with a historical average fallback, penalizing FC at high lags: `fc_f = max(0.3, 1 - 0.06 × (lag - 1))`. With multi-vintage, `lag` is replaced by the best (lowest) vintage lag available. A row at lag=8 with a lag=1 vintage forecast gets `fc_f = 1.0` instead of `fc_f = 0.58`.

### Design choices

- **Temporal causality**: In temporal eval, only forecasts issued before the row's Reference_Month are used. In random CV, the registry is built from training fold only (no leakage from test fold).
- **Forecast values are dynamic**: Different vintages have genuinely different Forecast_Values — forecasts evolve as more information becomes available. This is real signal, not duplication.
- **Actual_Sales is constant**: All rows for a given target period share the same realized outcome. The different vintages give different predictions of this single truth.
- **Fallback**: If no vintage exists for a target period (or none pass the causality filter), `fc_implied_multi = NaN` and the model falls back to historical average, same as before.

### Customer 1 Results

| Model | Temporal | Random CV | Gap |
|-------|----------|-----------|-----|
| V8d (OO+FC flat) | 16.5% | 93.9% | -77.5pp |
| **V8d + multi-FC** | **15.3%** | 62.3% | -47.0pp |
| FC-only flat | 21.3% | 20.2% | +1.1pp |
| **FC-only multi** | 19.9% | **13.4%** | +6.5pp |

**V8d + multi-FC (15.3%)** is the first model variant to improve on V8d's 16.5% through a structural change alone — no new signals, no parameter tuning, just using forecast information that was already in the data but being ignored.

Per-lag improvement (V8d+multi-FC vs V8d):

| Lag | V8d | V8d+mFC | Gain | FC coverage |
|-----|-----|---------|------|-------------|
| 1 | 11.1% | 10.1% | -1.0pp | 210 rows |
| 4 | 16.3% | 14.4% | -1.9pp | 150 rows |
| 7 | 22.1% | 19.3% | -2.8pp | 90 rows |
| 8 | 24.5% | 20.4% | **-4.1pp** | 75 rows |

The improvement is largest at high lags (7-8) where the original model had no FC and relied entirely on OO or historical average. Multi-vintage provides real FC signal at these lags from forecasts issued in earlier months.

Per-site: Site W drops from 12.4% to **9.8%** (-2.6pp), Site S from 15.8% to 14.6% (-1.2pp).

**FC-only multi (13.4% random CV)** is the best unbiased FC result — better than temporal V8d. This demonstrates that the FC signal, when fully exploited across vintages, is extremely powerful.

### Customer 2 Results

| Model | Temporal | Random CV | Gap |
|-------|----------|-----------|-----|
| Single-vintage | 33.1% | 41.3% ± 0.7% | -8.2pp |
| **Multi-vintage** | 37.0% | **37.7% ± 0.8%** | **-0.7pp** |

The temporal/random gap collapses from **-8.2pp to -0.7pp**. This proves the single-vintage temporal result (33.1%) was artificially good — 78% of test rows fell through to historical averages, which leak future information in temporal splits. Multi-vintage replaces that leaked signal with real FC data, and the gap vanishes.

- FC coverage jumps from **14% to 53%** of test rows (513 → 2,010 rows)
- Vintage depth averages 2.0 forecasts per target period, max 4
- Random CV bias improves from +44.6% to +32.2% — less reliance on the biased historical average fallback

### Key takeaway

**Every customer's forecast data contains more signal than we were using.** The target-period-centric architecture extracts this signal by looking across Reference_Months for each target period. The approach is general — it works for both Customer 1 (monthly forecasts, 46% coverage) and Customer 2 (quarterly forecasts, 16% coverage) with the same code and hyperparameters.

---

## 80% Prediction Intervals

### Motivation

A point prediction alone doesn't convey how much confidence we have in it. A $10M prediction for Site W at lag 1 carries very different certainty than a $200K prediction for Site E at lag 12. Prediction intervals quantify this: for each `(site, timeframe, lag)` combination, what range covers 80% of expected outcomes?

### Method: Parametric Log-Normal Prediction Intervals

For each `(site, timeframe, lag)` group, the model computes the ratio `actual / predicted` from historical predictions and models `log(actual / predicted)` as approximately normal. This gives a natural multiplicative interval:

```
log_ratios = log(actual / predicted)    for each group
mu         = mean(log_ratios)           captures systematic bias
sigma      = std(log_ratios, ddof=1)    captures spread (Bessel-corrected)
margin     = 1.2816 × sigma × sqrt(1 + 1/n)

CI_lower   = prediction × exp(mu - margin)
CI_upper   = prediction × exp(mu + margin)
```

Key design choices:

- **Log-space modeling**: Sales ratios are naturally multiplicative (a 50% miss matters equally whether the prediction is $1M or $10M). Working in log-space makes the interval symmetric in relative terms.
- **z = 1.2816**: The normal quantile for a two-sided 80% interval (`norm.ppf(0.90)`).
- **`sqrt(1 + 1/n)` correction**: The critical difference between a *confidence interval* (for the mean) and a *prediction interval* (for a new observation). This factor widens the interval to account for both estimation uncertainty in (mu, sigma) and the inherent randomness of a new outcome. With n=5 observations it inflates by 10%; with n=100, less than 1%.
- **Bessel correction (`ddof=1`)**: Uses n-1 in the denominator of sigma to correct for small-sample bias in variance estimation.

### Hierarchical Fallback

Not every `(site, timeframe, lag)` cell has enough historical predictions to estimate reliable intervals. The model applies the same hierarchical fallback pattern used in model selection:

| Level | Key | Example | When used |
|-------|-----|---------|-----------|
| 1 (finest) | `(site, tf, lag)` | Site W, TF=3, Lag=2 | >= 5 observations in cell |
| 2 | `(site, lag)` | Site W, Lag=2 (pooled across TF) | Level 1 has < 5 obs |
| 3 | `(site)` | Site W (pooled across TF and lag) | Level 2 has < 5 obs |
| 4 (coarsest) | global | Customer-wide | Level 3 has < 5 obs |

This ensures every prediction gets an interval, with granularity adapted to available data.

### Calibration Source

Intervals are calibrated from **temporal test predictions** — the same out-of-sample evaluation window used for model assessment. This mirrors production usage: calibrate on the most recent actuals-vs-predictions history, then apply to new predictions going forward.

### Results

#### Customer 1

| Metric | Value |
|--------|-------|
| Overall coverage | 85.9% (target: 80%) |
| Average relative width | 225% of point prediction |
| CI curves fitted | 208 (site,tf,lag) + 105 (site,lag) + 9 (site) + 1 global |

Per-site intervals reflect each site's predictability:

| Site | Coverage | Avg Width | Median CI |
|------|----------|-----------|-----------|
| Site W | 85% | 40% | [$30.0M – $44.5M] |
| Site S | 87% | 45% | [$15.6M – $21.7M] |
| Site T | 83% | 30% | [$10.0M – $13.7M] |
| Site J | 85% | 72% | [$15.3M – $29.5M] |
| Site A | 85% | 57% | [$5.6M – $11.2M] |
| Site D | 88% | 161% | [$0.9M – $1.9M] |
| Site E | 77% | 3817% | [$0.1M – $87.9M] |

Site E's extreme width (3817%) reflects its fundamental unpredictability (89% WMAPE) — the interval honestly communicates that predictions for this site are unreliable.

Width grows with lag (more uncertainty further out):

| Lag | Coverage | Avg Width |
|-----|----------|-----------|
| 1 | 82% | 149% |
| 4 | 86% | 183% |
| 8 | 90% | 362% |
| 12 | 88% | 180% |

Width varies by timeframe:

| TF | Coverage | Avg Width |
|----|----------|-----------|
| 3 | 84% | 87% |
| 6 | 85% | 325% |
| 9 | 88% | 330% |
| 12 | 92% | 326% |

Quarterly (TF=3) is tightest — shorter aggregation windows are more predictable.

#### Customer 2

| Metric | Value |
|--------|-------|
| Overall coverage | 86.6% (target: 80%) |
| Average relative width | 221% of point prediction |
| CI curves fitted | 330 (site,tf,lag) + 132 (site,lag) + 11 (site) + 1 global |

### Interpretation

The intervals are slightly conservative (~86% vs 80% target). This overcoverage comes from the `sqrt(1 + 1/n)` prediction interval correction, which is theoretically correct and preferable to undercoverage in practice — a decision-maker relying on these intervals will find actual outcomes inside the stated range at least 80% of the time.

The intervals communicate two things simultaneously:
1. **Where we're confident**: Site W at lag 1 with TF=3 has a 40% relative width — the prediction is meaningful and actionable.
2. **Where we're not**: Site E at lag 12 has a 3817% relative width — the point prediction is essentially noise, and any downstream decision should treat it as such.

---

## Current Understanding & Open Questions

### Why V8k over-corrects for Sites S and T
The progress model assumes orders are **uniformly distributed** across the ordering window. In reality, orders concentrate earlier in the window, especially for longer timeframes. This means the correction at lags near the boundary may be too aggressive.

### LT+OE shifts by site (train → test)
| Site | LT+OE train | LT+OE test | Shift |
|------|------------|-----------|-------|
| Site S | 17.3 | 10.8 | -6.5 |
| Site T | 14.1 | 9.2 | -4.9 |
| Site D | 9.8 | 5.1 | -4.7 |
| Site W | 13.3 | 9.4 | -3.9 |
| Site J | 13.0 | 10.2 | -2.8 |

### Ideas to explore
1. **Non-uniform ordering CDF**: A different CDF shape (e.g. beta distribution, or empirically estimated) could moderate the correction for lags near the boundary.
2. **LT+OE affecting OO vs FC weighting**: Use the LT+OE shift to adjust how much weight we give OO vs FC. When LT+OE drops significantly, lean more on FC.
3. **FC-primary architecture**: Models where FC is the dominant signal. OO could play a supporting role only at short lags (1–3) where its temporal drift is smallest.
4. **Older vintage analysis**: The multi-vintage model currently weights all vintages by `1/lag^1.5`. Older vintages (high original lag) may carry different signal quality — worth examining whether a staleness cap or different weighting at very high lags improves results.
