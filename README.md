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

## V7 Reference
WMAPE = 8.8%, bias = +0.5%, account = 2.2%
