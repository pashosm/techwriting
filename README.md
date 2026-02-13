# V8 Model Series — Customer 1 Demand Forecasting

## Definitions

- **Lag**: Months from reference month (when you are looking) to the first month of the timeframe of interest. If looking at June–August 2026 from February 2026, lag = 4.
- **LT (Lead Time)**: Published time for an order to be fulfilled. A 90-day lead time means ~3 months from order placement to delivery. This is a supply-side parameter set by the supplier.
- **OE (Order Earliness)**: `order promise date - order creation date - published lead time`. Measures how far ahead of the minimum the customer placed their order. If LT=3 months and the customer orders 5 months before the promise date, OE = 2 months. LT and OE tend to move in the same direction (when LT increases, customers start ordering earlier out of concern).
- **OO ratio**: `Open_Orders / Actual_Sales` — the fraction of eventual demand already visible as orders at a given lag.
- **Visibility threshold**: An order appears in Open Orders when `LT + OE > lag`. At large lags only early-placed orders are visible; at small lags most orders are in.

## Key Insight: Order Visibility as Progress Through the Ordering Window

Open Orders at a given lag traces a **cumulative ordering curve** — the fraction of total demand that has been ordered as a function of where we are relative to the ordering window.

The ordering window length is approximately `LT + OE`. The "progress" through this window is:

```
progress = max(0, 1 - lag / (LT + OE))
```

- `progress ≈ 0` → beginning of cycle, few orders placed
- `progress ≈ 1` → end of cycle, most orders placed

This explains why LT changes break flat curves: V8d's flat curves key only on lag, not on how much of the ordering window has elapsed. When LT+OE drops from train to test, the same lag corresponds to less progress through a shorter window, but the flat curve still returns the old (higher) ratio.

## What We Tried

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

## Random CV Diagnostic: Drift vs Noise

To test whether the OO and FC signals are stationary (i.e., can be treated as random events), we ran **5-fold random cross-validation** alongside the standard temporal train/test split. In random CV, rows are shuffled randomly across all time periods — both train and test contain data from every month. This destroys temporal structure and reveals whether a model depends on time-stability of its signal.

### Results

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

**OO is deeply non-stationary.** V8d explodes from 16.5% to 95% under random CV. When you build OO curves from a mix of all time periods and apply them to randomly sampled rows, the signal is catastrophically wrong at every lag. The OO-to-sales relationship changes so dramatically over time that a time-averaged curve is wrong for every individual period. The +55% systematic bias means OO_implied massively overstates actual sales on average.

**FC is nearly stationary.** FC-only barely moves (21.3% → 19.9%, slightly *better* under random CV). This confirms FC's relationship to actuals is stable over time — it genuinely can be treated as approximately random. The slight improvement comes from having more diverse training data and no train→test distribution shift.

**FC debiasing works when there's no drift.** FC-debiased improves from 25.1% (temporal) to 21.0% (random). With random sampling there's no systematic difference between train and test distributions, so the correction factors are valid. In the temporal split, FC's bias subtly shifted between train and test periods, making the historical correction factors wrong — hence debiasing backfired.

### Key takeaway

The V8d model's 16.5% WMAPE in temporal evaluation is **almost entirely carried by FC**. OO adds value only because the temporal split happens to keep curve drift manageable (a few months of extrapolation from recent training data). In a setting that strips away this temporal crutch, OO has *negative* value — it makes everything dramatically worse. This fundamentally changes the modeling strategy: **FC should be the primary signal, with OO used carefully (if at all) and with temporal safeguards.**

## Current Understanding & Open Questions

### Why V8k over-corrects for Sites S and T
The progress model assumes orders are **uniformly distributed** across the ordering window. In reality, orders concentrate earlier in the window, especially for longer timeframes. This means the correction at lags near the boundary may be too aggressive — fewer orders should be expected at the tail of the window than the uniform model predicts.

### LT+OE shifts by site (train → test)
| Site | LT+OE train | LT+OE test | Shift |
|------|------------|-----------|-------|
| Site S | 17.3 | 10.8 | -6.5 |
| Site T | 14.1 | 9.2 | -4.9 |
| Site D | 9.8 | 5.1 | -4.7 |
| Site W | 13.3 | 9.4 | -3.9 |
| Site J | 13.0 | 10.2 | -2.8 |

### Ideas to explore
1. **Non-uniform ordering CDF**: Orders concentrate at the beginning of the window. A different CDF shape (e.g. beta distribution, or empirically estimated) could moderate the correction for lags near the boundary.
2. **LT+OE affecting OO vs FC weighting**: Rather than correcting the OO curve itself, use the LT+OE shift to adjust how much weight we give OO vs FC in the final combination. When LT+OE drops significantly, the OO signal becomes less reliable at higher lags — the model should lean more on FC.
3. **FC-primary architecture**: Given the random CV findings, explore models where FC is the dominant signal. OO could play a supporting role — e.g., used only at short lags (1–3) where its temporal drift is smallest, or used as a secondary input whose weight decays with OO-ratio instability.

## V7 Reference
WMAPE = 8.8%, bias = +0.5%, account = 2.2%
