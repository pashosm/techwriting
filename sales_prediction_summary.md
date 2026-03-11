# Sales Prediction Model -- Project Summary

## Objective

Predict `Actual_Sales` for a supply chain planning system where forecasts and open orders arrive at varying prediction lags (1--12 months) across multiple timeframes (3, 6, 9, 12 months) and manufacturing sites.

**Core production challenge:** At long timeframes (e.g., TF=12), the most recent completed actual may be 12+ months old, while shorter timeframes have current data. Models must handle this staleness gracefully.

---

## Data Overview

- **~12,000 rows** for Customer 1 across **7 active sites** (Site E excluded for low volume).
- **Target:** `Actual_Sales`
- **Primary predictors:** `Open_Orders`, `Forecast_Value`
- **Leakage column:** `Covered_Orders` (requires knowing actuals -- must not be used)
- **Conditioning variables:** `Avg_Weighted_Lead_Time`, `Avg_Weighted_Order_Earliness`
- **Historical lags:** `Historical_Sales_Lag1` and `Lag12` are identical in this dataset (YoY = 1.0 everywhere)
- **Site concentration:** Site W (40%), Site J (24%), Site S (19%), Site T (12%) = 95% of sales

---

## Model Evolution

### Phase 1 -- EDA

- Strong signal in Open Orders and Forecast Value, but relationships vary by site/TF/lag.
- Massive regime shift during dataset period: lead times contracted ~45%, order book coverage dropped from ~70% to ~25% (post-COVID normalization).

### Phase 2 -- Per-Cell Ridge Regression (V1--V7)

Architecture: separate Ridge model per (site x TF x lag) cell (~205 models).

| Version | WMAPE | Bias | Key change |
|---------|-------|------|------------|
| V5d | 9.2% | -5% | Feature selection + recency weighting |
| V6 | 8.5% | -2% | + post-hoc multiplicative bias correction |
| **V7** | **8.8%** | **+0.5%** | Production model: GSA-aware keys, confidence intervals, time-series CV |

**V7 limitation:** Per-cell models cannot handle production staleness -- TF=12 predictions require 12-month-old training data while TF=3 has current data.

### Phase 3 -- Aging Curve Architecture (V8 series)

Three-layer design: (1) normalize signals to implied actuals via aging curves, (2) weighted-average combination, (3) optional bias correction.

| Version | WMAPE | Key result |
|---------|-------|------------|
| V8a--V8c | 18--27% | Ridge combination and bias corrections overfit/amplify noise |
| **V8d** | **16.5%** | Weighted average, no corrections -- best aging-curve variant |
| V8f | 17--19% | Cross-TF calibration adds noise in backtest; kept as staleness insurance |
| V8g | 35.8% | LT/Earliness-conditioned curves overfit; needs refinement |

**V8d site breakdown:** Site W 12.4%, Site J 21.0% (-17% bias), Site S 15.8%, Site T 17.8%.

### Phase 4 -- Cross-TF Calibration (V8e--V8f)

- OO cross-TF ratios drift heavily with staleness (30--51% error at 12 months stale).
- FC coverage/bias ratios are much more stable (8--23% at 12 months).
- **Decision:** Calibrate OO only; use FC raw. Apply calibration only when ratio is fresh (< 6 months).

### Phase 5 -- Feature Interactions

- **OO/FC Surprise:** Real signal (correlation with error up to -0.61 at long lags) but too noisy for post-hoc correction. Net neutral.
- **Historical Sales Trajectory:** Zero incremental value -- anchor weight is small and trends are already in recency-weighted curves.
- **Covered Orders as signal:** OO + CO achieved 10.0% WMAPE but is **leakage** -- discarded.

### Phase 6 -- Lead Time / Earliness Conditioning (V8g, in progress)

**The smoking gun:** Lead time contraction (8 -> 4 months) directly drives OO ratio instability (correlations 0.70--0.81). This explains Site J's persistent -17% bias.

- Short/medium lags: Total visibility (LT + Earliness) best predictor (R^2 0.83--0.94).
- Long lags: LT alone dominates.
- First implementation (per-cell regression) overfits due to small samples + distributional shift.
- **Next steps:** pooled regression, multiplicative adjustments, or damped regression shrinking toward flat curves.

---

## Current State

| Model | WMAPE | Bias | Production Ready? |
|-------|-------|------|-------------------|
| V7 (Ridge) | **8.8%** | +0.5% | Yes (fragile to staleness) |
| V8d (Aging Curves) | 16.5% | -2.9% | Yes (robust to mismatched lags) |
| V8g (LT-conditioned) | 35.8% | +36% | No (overfits, needs work) |

**Gap:** V7 vs V8d = 7.7pp. This is the cost of production robustness. LT/Earliness conditioning targets this gap but implementation must handle extrapolation to shifted conditions.

---

## Key Lessons

1. **Covered_Orders is leakage** -- requires knowing actuals. Never use directly.
2. **Historical lags are identical** in this dataset -- YoY features give zero signal. May change with real production data.
3. **Bias correction layers amplify noise** in small-sample per-cell settings. Every correction tested was neutral or harmful.
4. **Lead time drives OO ratio instability** -- the regime shift is not random, it tracks a 45% LT contraction. Conditioning on LT is the path forward.
5. **Simpler combinations beat complex ones** -- weighted average outperformed Ridge combination with limited samples per cell.
6. **Cross-TF calibration is insurance, not improvement** -- only valuable when curves go stale beyond ~6 months.
