# Customer Forecasting Presentation Outline

## V8 Model Series — Demand Forecasting

---

## Slide 1: Title

**Customer Demand Forecasting: V8 Model Series**

- Combining Open Orders & Customer Forecasts to predict future sales
- Customers 1 & 2
- February 2026

---

## Slide 2: Agenda

1. Problem Statement & Goals
2. Two Signals: Open Orders (OO) & Customer Forecasts (FC)
3. Key Definitions
4. Baseline Model (V8d) & Its Math
5. What We Tried: V8g–V8k Variants
6. Multi-Vintage FC Architecture
7. Weighted Blending Formula
8. 80% Prediction Intervals
9. Key Findings — Customer 1
10. Key Findings — Customer 2
11. Signal Stationarity Diagnostic
12. Summary & Next Steps

---

## Slide 3: Problem Statement

**Goal**: Predict actual sales for future target periods using signals available today.

- Each target period is defined by (Site, Start Date, End Date)
- Predictions made at various "lags" — months before the target period begins
- Two available signals: Open Orders (OO) and Customer Forecasts (FC)
- Challenge: signals drift over time, especially when lead times change

**Reference to beat (V7)**: WMAPE = 8.8%, Bias = +0.5%

---

## Slide 4: Two Signals

### Open Orders (OO)

- Orders already placed for the target period
- Accumulate as delivery date approaches (more orders at shorter lags)
- Affected by Lead Time (LT) and Order Earliness (OE) — when these shift, the OO-to-sales relationship breaks

### Customer Forecasts (FC)

- Customer-provided demand projections
- Not always available every month (Customer 2 forecasts quarterly)
- More stable over time than OO (stationary signal)

---

## Slide 5: Key Definitions

| Term | Definition |
|------|-----------|
| **Lag** | Months from reference month to target period start. Looking at Jun–Aug 2026 from Feb 2026 → lag = 4 |
| **Timeframe (TF)** | Length of target period: 3, 6, 9, or 12 months |
| **Lead Time (LT)** | Published order fulfillment time (supplier-side) |
| **Order Earliness (OE)** | How far ahead of minimum the customer places orders: `promise date − creation date − LT` |
| **OO Ratio** | `Open_Orders / Actual_Sales` — fraction of demand visible as orders |
| **FC Coverage** | `Covered_Orders / Actual_Sales` — fraction of sales covered by forecast |
| **FC Bias** | `Forecast_Value / Covered_Orders` — how much the forecast over/under-states |
| **Vintage** | A specific forecast for a target period, identified by the Reference_Month it was issued |

---

## Slide 6: Data Structure — Target Periods

**Key structural insight**: Each target period is observed from multiple Reference_Months.

Example: Site B, Jul 2023–Jun 2024 (TF=12) appears as 12 rows — one per Reference_Month at lags 12 down to 1.

Across rows for the same target period:
- **Actual_Sales** — identical (single realized outcome)
- **Open_Orders** — increases as lag decreases ($475K at lag=12 → $1.58M at lag=1)
- **Forecast_Value** — dynamic across vintages (forecasts evolve with new info)
- **Has_Forecast** — varies (not every Reference_Month has a forecast)

**Implication**: Each target period contains more forecast information than any single row reveals.

---

## Slide 7: V8d Baseline — Flat Aging Curves

**Approach**: For each (site, timeframe, lag) cell, compute the recency-weighted average of the OO ratio from training data.

**Math**:

```
OO_ratio(site, tf, lag) = weighted_mean( Open_Orders / Actual_Sales )
                          using recency weights (rolling 3-month window)

OO_implied_actual = Open_Orders / OO_ratio(site, tf, lag)
```

**Results**: WMAPE = 16.5%, Bias = −2.9%, Account Error = 2.5%

**Limitation**: Curves are keyed only on lag, not on how much of the ordering window has elapsed. When LT+OE shifts between training and test, flat curves return wrong ratios.

---

## Slide 8: Order Visibility as Progress Through the Ordering Window

**Core concept**: Orders become visible when `LT + OE > lag`.

```
progress = max(0, 1 − lag / (LT + OE))
```

- `progress ≈ 0` → beginning of ordering cycle, few orders placed
- `progress ≈ 1` → end of cycle, most orders in

**Why LT changes break flat curves**: When LT+OE drops from train to test, the same lag corresponds to *less* progress through a *shorter* window, but V8d's flat curve still returns the old (higher) ratio.

---

## Slide 9: What We Tried — V8g through V8j (Failed Approaches)

| Model | Approach | WMAPE | Bias | Why It Failed |
|-------|----------|-------|------|---------------|
| **V8g** | Conditioned regression: `OO_ratio = α + β₁·LT + β₂·OE` | 35.8% | +36.0% | Regression extrapolates badly outside training LT range |
| **V8h** | Gaussian kernel weighting by LT similarity | 18.5% | — | When test LT is systematically lower, even "most similar" training points are far away |
| **V8i** | LT-normalized: `OO / (Sales × LT)` × current LT | 25.1% | +35.5% | Assumes `OO_ratio ∝ LT` (proportional). Actual relationship has non-zero intercept |
| **V8j** | Additive lag shift: `effective_lag = lag + (ref_LT_OE − current_LT_OE)` | 26.9% | +9.5% | Shift is lag-independent but LT+OE effect is lag-dependent. Over-corrects at short lags |

---

## Slide 10: V8k — Progress-Based Multiplicative Correction

**Approach**: Adjust the flat curve ratio by the ratio of "progress" at current vs reference LT+OE.

```
progress(LT_OE, lag) = max(0, 1 − lag / LT_OE)

correction = progress(current_LT_OE, lag) / progress(ref_LT_OE, lag)

corrected_ratio = flat_ratio × correction
```

**Results**: WMAPE = 18.4%, Bias = +1.6%, Account Error = 2.3%

- First variant to **improve bias** (−2.9% → +1.6%) and account error (2.5% → 2.3%)
- Helps Sites D (−3.1pp), Americas (−3.7pp), J (−0.2pp), W (flat)
- Still hurts Sites S (+4.2pp) and T (+8.5pp) where LT+OE drops were largest
- Assumes uniform order distribution; real orders may concentrate earlier in window

---

## Slide 11: Multi-Vintage FC — The Problem

**Wasted information**: Original model checks each row independently for Has_Forecast.

- If `Has_Forecast = 0` → falls back to historical average
- But the **same target period** likely has a forecast from a different Reference_Month

**Coverage at row level vs target-period level**:

| Customer | Row-Level FC Coverage | Target-Period-Level FC Coverage |
|----------|----------------------|--------------------------------|
| Customer 1 | 46% | ~95% |
| Customer 2 | 16% (quarterly forecasts) | ~95% |

---

## Slide 12: Multi-Vintage FC — The Algorithm

**Step 1 — Build forecast registry**
Scan all rows with `Has_Forecast = 1`. Index by target period `(Site, Start, End)`. Store each vintage's Reference_Month, Lag, Forecast_Value, Covered_Orders.

**Step 2 — Collect causally-valid vintages**
For each row, look up its target period. Filter to vintages where `forecast Reference_Month ≤ row's Reference_Month` (can't use future forecasts).

**Step 3 — Convert each vintage to implied actual**

```
implied_actual = Forecast_Value / bias(site, tf, original_lag) / coverage(site, tf, original_lag)
```

Curves are looked up at the **vintage's own lag**, not the current row's lag.

**Step 4 — Combine with recency weighting**

```
weight = 1 / original_lag^1.5
```

| Original Lag | Weight | Relative to Lag 1 |
|-------------|--------|-------------------|
| 1 | 1.000 | 1× |
| 3 | 0.192 | ~5× less |
| 6 | 0.068 | ~15× less |
| 9 | 0.037 | ~27× less |

**Step 5 — Use best_vintage_lag for weighting in final blend**
A row at lag=8 with a lag=1 vintage gets treated as if FC is at lag=1 for weighting purposes.

---

## Slide 13: Weighted Average Blending Formula

The final prediction blends three signals:

```
prediction = weighted_average(OO_implied, FC_implied, historical_avg)
```

**Weight calculation**:

```
weight_OO = (1 / OO_MAPE) × OO_factor(lag)
weight_FC = (1 / FC_MAPE) × FC_factor(best_vintage_lag)
weight_hist = (1 / 0.5) × 0.1 × (lag / 12)
```

**Lag-dependent factors**:

```
OO_factor = max(0.3, 1.0 − 0.06 × (lag − 1))     penalizes OO at high lags
FC_factor = max(0.3, 1.0 − 0.06 × (lag − 1))      uses best_vintage_lag with multi-vintage
```

Historical average gets increasing weight at higher lags as a stabilizer/fallback.

---

## Slide 14: 80% Prediction Intervals

**Motivation**: A point prediction doesn't convey confidence. $10M for Site W at lag 1 is very different from $200K for Site E at lag 12.

**Method — Log-Normal intervals**:

```
log_ratios = log(actual / predicted)        for each (site, tf, lag) group
mu         = mean(log_ratios)               captures systematic bias
sigma      = std(log_ratios, ddof=1)        Bessel-corrected spread

margin     = 1.2816 × sigma × sqrt(1 + 1/n)

CI_lower   = prediction × exp(mu − margin)
CI_upper   = prediction × exp(mu + margin)
```

**Key design choices**:
- **Log-space**: Makes intervals symmetric in relative terms (a 50% miss matters equally at $1M or $10M)
- **z = 1.2816**: Normal quantile for two-sided 80% interval
- **sqrt(1 + 1/n)**: Prediction interval correction — accounts for both estimation uncertainty and new-observation randomness

**Hierarchical fallback**: (site, tf, lag) → (site, lag) → (site) → global. Minimum 5 observations per cell.

---

## Slide 15: Customer 1 — Overall Results

| Model | WMAPE | Bias | Account Error |
|-------|-------|------|---------------|
| V7 (target to beat) | 8.8% | +0.5% | 2.2% |
| V8d (flat baseline) | 16.5% | −2.9% | 2.5% |
| V8k (progress correction) | 18.4% | +1.6% | 2.3% |
| FC-only flat | 21.3% | +21.0% | — |
| FC-only multi-vintage | 19.9% | +18.0% | — |
| **V8d + multi-vintage FC** | **15.3%** | **+1.5%** | **2.3%** |

**V8d + multi-vintage FC** is the best V8 variant — improves on baseline through structural change alone (no new signals, no tuning).

---

## Slide 16: Customer 1 — Improvement by Lag

Multi-vintage FC gains increase at higher lags (where original model had no FC):

| Lag | V8d | V8d + multi-FC | Improvement |
|-----|-----|----------------|-------------|
| 1 | 11.1% | 10.1% | −1.0pp |
| 4 | 16.3% | 14.4% | −1.9pp |
| 7 | 22.1% | 19.3% | −2.8pp |
| 8 | 24.5% | 20.4% | **−4.1pp** |

Largest gains at high lags — multi-vintage provides real FC signal from forecasts issued in earlier months.

---

## Slide 17: Customer 1 — Results by Site

| Site | WMAPE (V8d+mFC) | Prediction Interval Width | Interpretation |
|------|-----------------|--------------------------|----------------|
| Site W | 9.8% | 40% | Highly predictable |
| Site S | 14.6% | 45% | Solid |
| Site T | ~10% | 30% | Good |
| Site J | ~10% | 72% | Good accuracy, wider intervals |
| Site A | — | 57% | Moderate |
| Site D | — | 161% | Harder to predict |
| Site E | 89% | 3,817% | Essentially unpredictable — intervals communicate this honestly |

---

## Slide 18: Customer 2 — Multi-Vintage Impact

| Metric | Single-Vintage | Multi-Vintage |
|--------|---------------|---------------|
| Temporal WMAPE | 33.1% | 37.0% |
| Random CV WMAPE | 41.3% ± 0.7% | 37.7% ± 0.8% |
| **Temporal/Random Gap** | **−8.2pp** | **−0.7pp** |
| FC coverage (test rows) | 14% (513 rows) | 53% (2,010 rows) |
| Bias | +44.6% | +32.2% |
| Avg vintages per target period | 1.0 | 2.0 (max 4) |

**Why temporal WMAPE went up**: Single-vintage's 33.1% was artificially good — 78% of rows fell through to historical averages, which leak future information in temporal splits. Multi-vintage replaces that leaked signal with real FC data. The collapsed gap (8.2pp → 0.7pp) proves the result is now honest.

---

## Slide 19: Signal Stationarity Diagnostic

**Method**: Compare 5-fold random CV (shuffled across time) vs temporal train/test split. If a signal is stationary, both should give similar results.

### Customer 1

| Model | Random CV | Temporal | Gap |
|-------|-----------|----------|-----|
| V8d (OO+FC) | 94.9% ± 3.3% | 16.5% | **−78pp** |
| FC-only | 19.9% ± 0.4% | 21.3% | +1.4pp |

### Per-Lag Detail

| Lag | Random V8d | Temporal V8d | Random FC | Temporal FC |
|-----|-----------|-------------|-----------|-------------|
| 1 | 38% | 11% | 16% | 16% |
| 3 | 51% | 15% | 20% | 19% |
| 6 | 94% | 19% | 21% | 25% |
| 9 | 101% | 26% | 20% | 31% |
| 12 | 285% | 34% | 28% | 36% |

**OO is deeply non-stationary** — explodes from 16.5% to 95% under random CV.
**FC is nearly stationary** — barely moves (21.3% → 19.9%).

---

## Slide 20: Lead Time Shifts — Train to Test

LT+OE dropped significantly across most sites between training and test periods:

| Site | LT+OE (Train) | LT+OE (Test) | Shift |
|------|---------------|--------------|-------|
| Site S | 17.3 | 10.8 | −6.5 |
| Site T | 14.1 | 9.2 | −4.9 |
| Site D | 9.8 | 5.1 | −4.7 |
| Site W | 13.3 | 9.4 | −3.9 |
| Site J | 13.0 | 10.2 | −2.8 |

Sites with largest shifts (S, T) are hardest for OO-based corrections. This is why FC — which is unaffected by LT changes — is the more reliable signal.

---

## Slide 21: Prediction Interval Results

### Customer 1

| Metric | Value |
|--------|-------|
| Overall coverage | 85.9% (target: 80%) |
| Avg relative width | 225% of point prediction |

**By Lag**:

| Lag | Coverage | Avg Width |
|-----|----------|-----------|
| 1 | 82% | 149% |
| 4 | 86% | 183% |
| 8 | 90% | 362% |
| 12 | 88% | 180% |

**By Timeframe**:

| TF | Coverage | Avg Width |
|----|----------|-----------|
| 3-month | 84% | 87% |
| 6-month | 85% | 325% |
| 12-month | 92% | 326% |

Slightly conservative (~86% vs 80% target) — preferable to undercoverage for decision-makers.

### Customer 2

| Metric | Value |
|--------|-------|
| Overall coverage | 86.6% (target: 80%) |
| Avg relative width | 221% of point prediction |

---

## Slide 22: Key Findings Summary

1. **Best model: V8d + multi-vintage FC (15.3% WMAPE)**
   - Improves on flat baseline through structural change alone — no new signals, no parameter tuning
   - Uses forecast information already in the data but previously ignored

2. **Multi-vintage is transformative for sparse forecasters**
   - Customer 2 FC coverage: 14% → 53% of test rows
   - Temporal/random gap collapses from 8.2pp → 0.7pp (proves result is honest)

3. **OO is non-stationary; FC is the reliable signal**
   - Random CV diagnostic: 78pp gap for OO vs 1pp for FC
   - FC should be the primary signal; OO useful only at short lags with safeguards

4. **Lead time shifts are the core challenge**
   - Sites with largest LT+OE drops (S: −6.5, T: −4.9) are hardest to predict
   - Progress-based correction (V8k) helps conceptually but over-corrects in practice

5. **Prediction intervals are honest and calibrated**
   - 85–87% coverage (slightly conservative, which is good)
   - Width reflects real predictability: 40% for Site W vs 3,817% for Site E

---

## Slide 23: Open Questions & Future Directions

1. **Non-uniform ordering CDF** — Orders may concentrate earlier in the window (not uniform). Beta distribution or empirical CDF could moderate V8k's correction.

2. **Dynamic OO vs FC weighting** — When LT+OE drops significantly, automatically lean more on FC vs OO.

3. **FC-primary architecture** — Make FC the dominant signal. OO plays supporting role only at short lags (1–3) where temporal drift is smallest.

4. **Vintage staleness** — Current `1/lag^1.5` weighting treats all vintages the same. Very old vintages may need a staleness cap or different weighting.

---

## Appendix A: Full Formula Reference

### OO Implied Actual
```
OO_implied = Open_Orders / OO_ratio(site, tf, lag)
```

### FC Implied Actual (single vintage)
```
FC_implied = Forecast_Value / FC_bias(site, tf, lag) / FC_coverage(site, tf, lag)
```

### FC Implied Actual (multi-vintage)
```
For each vintage v with original_lag_v:
  implied_v = Forecast_Value_v / FC_bias(site, tf, original_lag_v) / FC_coverage(site, tf, original_lag_v)
  weight_v  = 1 / original_lag_v^1.5

FC_implied_multi = Σ(weight_v × implied_v) / Σ(weight_v)
```

### Progress Correction (V8k)
```
progress(LT_OE, lag) = max(0, 1 − lag / LT_OE)
correction = progress(current_LT_OE, lag) / progress(ref_LT_OE, lag)
corrected_ratio = flat_ratio × correction
```

### Final Blend
```
w_OO   = (1 / OO_MAPE) × max(0.3, 1.0 − 0.06 × (lag − 1))
w_FC   = (1 / FC_MAPE) × max(0.3, 1.0 − 0.06 × (best_vintage_lag − 1))
w_hist = (1 / 0.5) × 0.1 × (lag / 12)

prediction = (w_OO × OO_implied + w_FC × FC_implied + w_hist × hist_avg) / (w_OO + w_FC + w_hist)
```

### 80% Prediction Interval
```
log_ratios = log(actual / predicted)
mu    = mean(log_ratios)
sigma = std(log_ratios, ddof=1)
margin = 1.2816 × sigma × sqrt(1 + 1/n)

CI = [prediction × exp(mu − margin), prediction × exp(mu + margin)]
```

---

## Appendix B: Model Comparison Summary

| Model | Approach | WMAPE | Bias | Status |
|-------|----------|-------|------|--------|
| V8d | Flat aging curves | 16.5% | −2.9% | Baseline |
| V8g | Conditioned regression (LT+OE) | 35.8% | +36.0% | Failed |
| V8h | Gaussian kernel (LT-aware) | 18.5% | — | Failed |
| V8i | LT-normalized curves | 25.1% | +35.5% | Failed |
| V8j | Additive lag shift | 26.9% | +9.5% | Failed |
| V8k | Progress-based correction | 18.4% | +1.6% | Improved bias |
| V8d + multi-FC | Multi-vintage forecast architecture | **15.3%** | +1.5% | **Best** |
