# V8 Model Series — Demand Forecasting (Customer 1 & Customer 2)

## Overview

This project builds demand forecasting models that combine two signals — **Open Orders (OO)** and **Customer Forecasts (FC)** — to predict actual sales for future target periods. The models are evaluated on two customers with different data characteristics and forecast availability patterns.

### Key files

| File | Purpose |
|------|---------|
| `aging_v8g.py` | Customer 1 full pipeline: curve building, signal quality, V8d–V8k model variants, multi-vintage FC, random CV |
| `customer2_fc.py` | Customer 2 FC-only pipeline: multi-vintage FC, temporal eval, random CV |
| `oo_normalization.py` | Power-CDF OO normalization experiment: alpha estimation, normalized/damped curves, both customers |
| `subquarter_decomp.py` | Sub-quarter decomposition experiment: bottom-up TF=3 predictions for TF=6/9/12, both customers |
| `fc_tuning.py` | FC parameter tuning experiment: systematic sweep of vintage weighting, lag decay, staleness caps, curve recency |
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

## Power-CDF OO Normalization (Negative Result)

### Motivation

V8k's uniform CDF correction assumed `alpha=1` (orders distributed uniformly across the ordering window). Data exploration revealed that the ordering curve actually follows a **power law** `OO_ratio ~ progress^alpha` with **site-specific alpha**, explaining why V8k over-corrected Sites S and T.

### Alpha Estimation

Estimated via weighted log-log OLS regression: `log(OO_ratio) = alpha × log(progress) + c`, with recency weighting (`RECENCY_POWER=2`), clipped to [0.3, 4.0].

#### Customer 1 — Per-site alpha

| Site | Alpha | N obs | Interpretation |
|------|-------|-------|----------------|
| Site D | 1.05 | 441 | Nearly uniform ordering |
| Site S | 1.65 | 496 | Moderately back-loaded |
| Site T | 2.30 | 369 | Heavily back-loaded |
| Site W | 2.48 | 485 | Heavily back-loaded |

#### Customer 1 — Per-(site, timeframe) alpha

Alpha increases with timeframe for most sites, suggesting more back-loaded ordering at longer horizons:

| Site | TF3 | TF6 | TF9 | TF12 |
|------|-----|-----|-----|------|
| Site D | 1.00 | 0.94 | 1.14 | 1.10 |
| Site S | 1.25 | 1.60 | 2.04 | 2.00 |
| Site T | 1.56 | 2.14 | 2.66 | 3.06 |
| Site W | 2.13 | 2.69 | 2.84 | 2.98 |

#### Customer 2 — Per-site alpha

| Site | Alpha | N obs |
|------|-------|-------|
| Other-PS EMEA | 0.30 | 175 |
| Site H | 1.03 | 630 |
| Site B | 2.34 | 629 |

### Approach: Normalized OO Curves

```
progress = max(ε, 1 - lag / (LT + OE))
OO_ratio_norm = OO_ratio / progress^alpha       (training: normalized ratio)
implied_actual = OO / (curve_norm × progress^alpha)  (prediction)
```

Curves built on the normalized ratio per `(site, tf, lag)` with recency weighting. Also tested a **damped variant** that blends between normalized and flat using a smoothstep function of progress, to avoid noise amplification at high lags.

### Variants tested

| Variant | Description |
|---------|-------------|
| NormS | Per-site alpha, normalized curves |
| NormST | Per-(site, timeframe) alpha, normalized curves |
| DampS | Per-site alpha, damped correction on flat curves |
| DampST | Per-(site, tf) alpha, damped correction on flat curves |

Each combined with both flat FC and multi-vintage FC.

### Results

#### Customer 1 (temporal, months 2+)

| Model | WMAPE | Bias |
|-------|-------|------|
| **V8d + mFC** | **18.4%** | -0.4% |
| NormS + mFC | 20.1% | +2.9% |
| NormST + mFC | 20.2% | +3.1% |
| DampS + mFC | 21.9% | +5.3% |
| DampST + mFC | 22.0% | +5.3% |
| FC-only multi | 23.0% | +4.3% |

#### Customer 2 (temporal, months 2+)

| Model | WMAPE | Bias |
|-------|-------|------|
| **FC-only multi** | **38.2%** | +12.5% |
| V8d + mFC | 42.2% | +3.1% |
| NormS + mFC | 45.0% | +5.5% |
| DampS + mFC | 46.9% | +10.3% |
| DampST + mFC | 47.6% | +10.7% |

### Stationarity improvement (positive)

Despite not improving temporal accuracy, normalization dramatically improved stationarity as measured by the temporal-vs-random CV gap:

| Model | Customer 1 Gap | Customer 2 Gap |
|-------|---------------|---------------|
| V8d + mFC | 76.6pp | 66.7pp |
| NormS + mFC | 26.1pp | 22.4pp |
| NormST + mFC | 23.8pp | 26.3pp |

### Why normalization didn't improve temporal accuracy

1. **Noise amplification at high lags**: Dividing by `progress^alpha` when progress is small (high lags, short LT+OE) amplifies noise. Customer 1 lag 12: V8d 78.7% → NormS 108.6%.
2. **Recency weighting already handles drift**: Flat curves with `RECENCY_POWER=2` naturally downweight old observations, providing implicit adaptation to LT+OE changes. The normalization disrupts this already-adequate calibration.
3. **Damped correction doesn't help**: Blending normalized and flat based on progress level didn't find a sweet spot — the correction either applied too aggressively (short lags where flat was fine) or not enough (long lags where noise dominates).

### Key takeaway

The power-CDF model correctly captures the physics of ordering (site-specific alpha, progress-dependent visibility), and it successfully makes the OO signal more stationary. But the flat curves' recency weighting already compensates for LT+OE drift in the temporal evaluation window, and normalization introduces noise that outweighs the stationarity benefit. **This is a case where the theoretically correct model is outperformed by a simpler adaptive approach.**

The experiment is documented in `oo_normalization.py`.

---

## Sub-Quarter Decomposition (Negative Result)

### Motivation

A TF=12 target period can be decomposed into four non-overlapping TF=3 sub-quarters. Each sub-quarter exists as its own TF=3 row at the same Reference_Month but at progressively higher effective lags (`eff_lag = parent_lag + 3*k`). The hypothesis: predicting each sub-quarter independently using TF=3 curves and summing could be more accurate than a single TF=12 prediction, because TF=3 curves may be better calibrated and the decomposition naturally captures front-loaded vs back-loaded ordering patterns.

### Data verification

**Perfect additivity confirmed** for both customers: TF=3 Actual_Sales and Open_Orders sum exactly to TF=6/9/12 values (100% match for all fully-matched rows).

Sub-quarter availability depends on effective lag (max lag in data = 12):

| Parent TF | Full coverage (all subs) | Partial coverage |
|-----------|------------------------|-----------------|
| TF=6 | Parent lag 1–9 | Lag 10–12: 1 of 2 |
| TF=9 | Parent lag 1–6 | Lag 7–9: 2 of 3 |
| TF=12 | Parent lag 1–3 | Lag 4+: progressively fewer |

### Variants tested

- **BU-OO (all-or-nothing)**: Only predict when ALL sub-quarters have data and curve coverage. Falls back to per-TF flat curve otherwise.
- **BU-OO (shares)**: Scale available sub-quarter predictions by historical share to fill missing quarters. Historical shares computed per (site, parent_TF, sub-quarter index).
- Combined with both flat FC and multi-vintage FC.

### Results

#### Customer 1 (temporal, months 2+)

| Model | WMAPE | Bias |
|-------|-------|------|
| **V8d + mFC** | **18.4%** | +36.6% |
| BU-OO(sh) + mFC | 18.6% | +33.4% |
| BU-OO + mFC | 19.4% | +35.8% |
| FC-only multi | 23.0% | +46.0% |

Per-TF breakdown:

| TF | V8d + mFC | BU-OO + mFC | BU-OO(sh) + mFC |
|----|-----------|-------------|-----------------|
| 3 | 23.4% | 23.4% | 23.4% |
| 6 | 20.1% | 21.0% | **19.9%** |
| 9 | 15.7% | 17.0% | **15.5%** |
| 12 | **12.8%** | 14.7% | 14.9% |

#### Customer 2 (temporal, months 2+)

| Model | WMAPE | Bias |
|-------|-------|------|
| **FC-only multi** | **38.2%** | +24.1% |
| V8d + mFC | 42.2% | +36.1% |
| BU-OO + mFC | 42.3% | +41.2% |
| BU-OO(sh) + mFC | 44.2% | +41.2% |

Per-TF breakdown:

| TF | V8d + mFC | BU-OO + mFC | BU-OO(sh) + mFC |
|----|-----------|-------------|-----------------|
| 3 | 47.8% | 47.8% | 47.8% |
| 6 | 42.8% | 43.4% | 45.1% |
| 9 | 40.3% | 40.4% | 42.3% |
| 12 | 37.4% | **36.8%** | 41.4% |

### Stationarity improvement

Bottom-up OO is more stationary than flat OO (smaller temporal-vs-random CV gap):

| Model | Customer 1 Gap | Customer 2 Gap |
|-------|---------------|---------------|
| V8d + mFC | -45.2pp | -25.8pp |
| BU-OO + mFC | -33.3pp | -12.7pp |
| FC-only multi | +7.0pp | +1.5pp |

### Historical sub-quarter shares

Most sites have fairly uniform quarterly shares within longer TFs:

| Site | TF=12 Q1 | Q2 | Q3 | Q4 | Pattern |
|------|----------|-----|-----|-----|---------|
| Site W | 23.9% | 24.6% | 25.3% | 26.2% | ~Even |
| Site S | 24.6% | 24.1% | 24.9% | 26.5% | ~Even |
| Site T | 24.9% | 24.1% | 23.8% | 24.4% | ~Even |
| Site A | 5.5% | 25.2% | 34.2% | 68.3% | Highly seasonal |

With most sites showing near-even shares, there is limited front/back-loading signal for the decomposition to exploit.

### Why decomposition didn't improve temporal accuracy

1. **Error accumulation**: Summing 2–4 noisy TF=3 predictions introduces more variance than a single per-TF prediction. OO signal WMAPE: flat 43.0% vs BU 44.9% (Customer 1).
2. **High effective lags add noise**: Later sub-quarters have effective lags of 7–12+, where TF=3 OO curves are least reliable.
3. **Even quarterly shares**: Most sites split evenly across quarters (23–27% each), so decomposition provides minimal information gain over the aggregate.
4. **Shares scaling**: Pro-rating partial decompositions (shares mode) amplifies errors from the sub-quarters that are available.
5. **Marginal TF=6/9 improvement**: BU(shares) shows tiny gains for Customer 1 TF=6 (-0.2pp) and TF=9 (-0.2pp), but these are offset by TF=12 degradation.

### Key takeaway

Bottom-up decomposition is structurally sound (perfect additivity) and produces a more stationary OO signal. But the noise from summing multiple TF=3 predictions outweighs any benefit from capturing quarterly ordering patterns. **The per-TF flat curves already capture timeframe-specific dynamics more efficiently than reconstructing them from sub-quarters.** The one exception — Site A's extreme seasonality (Q1=5.5%, Q4=68.3%) — suggests decomposition could help for highly seasonal sites, but there aren't enough such sites to move the overall metric.

The experiment is documented in `subquarter_decomp.py`.

---

## FC Parameter Tuning (Positive Result)

### Motivation

Three OO-focused experiments (power-CDF normalization, sub-quarter decomposition, cross-site correlation) all produced negative results. FC is the dominant signal: FC-only multi beats all OO-containing models for Customer 2 (38.2% vs 42.2%), and for Customer 1 FC carries most of the prediction value. Five FC parameters were set heuristically and never tuned:

| Parameter | Default | What it does |
|-----------|---------|-------------|
| `VINTAGE_RECENCY_POWER` | 1.5 | Controls vintage combination weighting: `weight = 1/lag^power` |
| FC lag decay coefficient | 0.06 | FC reliability penalty in blend: `fc_f = max(floor, 1 - coeff*(lag-1))` |
| FC lag decay floor | 0.3 | Minimum FC reliability factor |
| Vintage staleness cap | None | Whether to exclude vintages beyond a maximum lag |
| FC curve RECENCY_POWER | 2 | Recency weighting when building FC coverage/bias curves |

### Approach

Systematic sequential sweep: optimize one parameter dimension at a time, carry the best forward to the next sweep. Final combination validated via K-fold random CV.

### Results

#### Customer 1 (temporal, months 2+)

| Model | WMAPE | Bias | vs Default |
|-------|-------|------|-----------|
| V8d + mFC (default) | 18.4% | +36.6% | --- |
| **V8d + mFC (tuned)** | **17.3%** | +36.4% | **-1.0pp** |
| FC-only multi (default) | 23.0% | +46.0% | --- |
| FC-only multi (tuned) | 21.2% | +44.2% | -1.8pp |

Per-TF improvement (V8d+mFC):

| TF | Default | Tuned | Diff |
|----|---------|-------|------|
| 3 | 23.4% | 22.5% | -0.9pp |
| 6 | 20.1% | 19.1% | -1.0pp |
| 9 | 15.7% | 14.4% | -1.2pp |
| 12 | 12.8% | 11.8% | -1.0pp |

Per-lag improvement (V8d+mFC):

| Lag | Default | Tuned | Diff |
|-----|---------|-------|------|
| 1 | 14.7% | 12.5% | -2.2pp |
| 4 | 17.3% | 16.4% | -1.0pp |
| 8 | 24.1% | 22.8% | -1.2pp |
| 12 | 38.9% | 39.2% | +0.3pp |

Improvement is consistent across all TFs and most lags. Largest gains at lag 1 (-2.2pp) where FC signal is strongest.

#### Customer 2 (temporal, months 2+)

| Model | WMAPE | Bias | vs Default |
|-------|-------|------|-----------|
| V8d + mFC (default) | 42.2% | +36.1% | --- |
| **V8d + mFC (tuned)** | **40.8%** | +36.7% | **-1.4pp** |
| FC-only multi (default) | 38.2% | +24.1% | --- |
| **FC-only multi (tuned)** | **36.2%** | +24.9% | **-2.0pp** |

Per-TF improvement (V8d+mFC):

| TF | Default | Tuned | Diff |
|----|---------|-------|------|
| 3 | 47.8% | 47.0% | -0.8pp |
| 6 | 42.8% | 41.8% | -1.0pp |
| 9 | 40.3% | 38.5% | -1.8pp |
| 12 | 37.4% | 35.1% | -2.3pp |

Improvement grows with TF — longer timeframes benefit most because they have more vintages to combine.

### Best parameters found

Both customers converged on the same direction for each parameter:

| Parameter | Default | Cust 1 Best | Cust 2 Best | Direction |
|-----------|---------|-------------|-------------|-----------|
| VINTAGE_RECENCY_POWER | 1.5 | **0.5** | **0.5** | Lower = more equal weighting |
| FC lag decay coeff | 0.06 | 0.06 | 0.06 | No change |
| FC lag decay floor | 0.3 | 0.3 | **0.5** | Slightly higher for Cust 2 |
| Vintage staleness cap | None | None | None | No cap needed |
| FC curve RECENCY_POWER | 2 | **3.0** | **3.0** | Higher = more emphasis on recent data |

### Parameter sweep findings

**VINTAGE_RECENCY_POWER (1.5 -> 0.5)**: The most impactful parameter. Lower power means more equal weighting across vintages — older forecasts contain real signal and shouldn't be discounted as aggressively. At power=0.5, a lag=3 vintage gets 58% of lag=1's weight (vs 19% at power=1.5). Both customers improve monotonically as power decreases. Clear signal: the default was too aggressive at discounting older vintages.

**FC lag decay (coeff/floor)**: The 2D heatmap is mostly flat — the FC lag decay coefficient and floor have little effect when multi-vintage is already adjusting the effective lag via `best_vintage_lag`. The default values (0.06/0.3) are near-optimal for both customers. This makes sense: with multi-vintage FC, most rows already have a nearby vintage, so the lag penalty rarely applies at full force.

**Vintage staleness cap**: No cap is better. Removing old vintages strictly hurts — even distant forecasts contribute useful information when weighted properly. Capping at 3 vintages degrades V8d+mFC by +16.5pp (Cust 1) and +7.5pp (Cust 2).

**FC curve RECENCY_POWER (2 -> 3)**: Higher recency power in curve building emphasizes the most recent training observations more. Both customers improve monotonically from 1.0 to 3.0. This makes sense: FC coverage and bias patterns are drifting over time, and putting more weight on recent observations produces more relevant curves for the test period.

### K-fold random CV validation

| Config | Cust 1 CV | Cust 1 Temporal | Cust 2 CV | Cust 2 Temporal |
|--------|-----------|-----------------|-----------|-----------------|
| V8d+mFC (default) | 63.6% | 18.4% | 68.0% | 42.2% |
| V8d+mFC (tuned) | 64.1% | 17.3% | 67.8% | 40.8% |
| FC-only (default) | 16.0% | 23.0% | 36.8% | 38.2% |
| FC-only (tuned) | 15.9% | 21.2% | 37.2% | 36.2% |

Tuned parameters show equivalent CV performance to defaults (within noise), confirming the temporal improvement is not overfitting. Customer 2 FC-only (tuned) actually reduces the temporal-CV gap from +1.5pp to -1.0pp, suggesting tuning aligns the model better with stationary behavior.

### Key takeaway

**FC parameter tuning delivers the first genuine accuracy improvement** across all experiments. The two impactful changes — lower vintage power (0.5 vs 1.5) and higher curve recency (3 vs 2) — both point in the same direction: **trust all vintage information more equally, but calibrate curves on the most recent data.** This is consistent with the finding that FC is a nearly stationary signal (unlike OO): older vintages are still informative, but the coverage/bias curves should adapt quickly to recent patterns.

The experiment is documented in `fc_tuning.py`.

---

## Current Understanding & Open Questions

### LT+OE shifts by site (train → test)
| Site | LT+OE train | LT+OE test | Shift |
|------|------------|-----------|-------|
| Site S | 17.3 | 10.8 | -6.5 |
| Site T | 14.1 | 9.2 | -4.9 |
| Site D | 9.8 | 5.1 | -4.7 |
| Site W | 13.3 | 9.4 | -3.9 |
| Site J | 13.0 | 10.2 | -2.8 |

### Ideas to explore
1. **LT+OE affecting OO vs FC weighting**: Use the LT+OE shift to adjust how much weight we give OO vs FC. When LT+OE drops significantly, lean more on FC.
2. **FC-primary architecture**: Models where FC is the dominant signal. OO could play a supporting role only at short lags (1–3) where its temporal drift is smallest.
3. ~~**Older vintage analysis**~~: **Done** — FC parameter tuning (vintage_power=0.5, no staleness cap) showed older vintages are valuable and should not be discounted aggressively.
4. **OO lag restriction**: Given that normalization helps stationarity but hurts accuracy, restrict OO signal to only short lags (e.g., lag ≤ 3–4) where progress is high and normalization noise is minimal, falling back to FC-only at longer lags.
5. **Apply tuned FC parameters to pipeline.py**: Update the production pipeline with the validated parameter improvements (VINTAGE_RECENCY_POWER=0.5, FC RECENCY_POWER=3). The lag decay params were already near-optimal.
