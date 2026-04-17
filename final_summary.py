import aging_v9 as v9
import numpy as np
import pandas as pd

df = v9.load_data(v9.DEFAULT_DATA_PATH)
snaps = [d.strftime('%Y-%m-%d') for d in v9.monthly_grid('2024-06-01', '2025-03-01')]

print("=" * 140)
print("V9 Final Summary: FC_Corrected vs Historical by Snapshot")
print("=" * 140)

hdr1 = (f"  {'Snapshot':<12} | {'FC_C':>5} {'FC_C %':>7} {'FC_C $Value':>16} {'FC_C $%':>7} "
        f"{'FC_C WMAPE':>10} {'FC_C Bias':>10} | "
        f"{'HIST':>5} {'HIST %':>7} {'HIST $Value':>16} {'HIST $%':>7} "
        f"{'HIST WMAPE':>10} {'HIST Bias':>10}")
print(hdr1)
print(f"  {'-'*12} | {'-'*5} {'-'*7} {'-'*16} {'-'*7} {'-'*10} {'-'*10} | "
      f"{'-'*5} {'-'*7} {'-'*16} {'-'*7} {'-'*10} {'-'*10}")

tot = {'fc_n': 0, 'fc_act': 0, 'fc_pred': 0, 'fc_err': 0,
       'h_n': 0, 'h_act': 0, 'h_pred': 0, 'h_err': 0}

for snap in snaps:
    _, pred = v9.run_snapshot(df, snap, 3, verbose=False)
    if pred is None:
        continue

    fc = pred[pred['Prediction_Source'].isin(v9.SOURCES_FC_CORRECTED)]
    hi = pred[pred['Prediction_Source'] == 'HISTORICAL']
    scored = pred[pred['Prediction'].notna()]

    total_rows = len(fc) + len(hi)
    total_act = fc['Actual_Sales'].sum() + hi['Actual_Sales'].sum()

    fc_act = fc['Actual_Sales'].sum()
    fc_pred = fc['Prediction'].sum()
    fc_err = np.abs(fc['Prediction'] - fc['Actual_Sales']).sum()
    h_act = hi['Actual_Sales'].sum()
    h_pred = hi['Prediction'].sum()
    h_err = np.abs(hi['Prediction'] - hi['Actual_Sales']).sum()

    fc_pct_rows = len(fc) / total_rows * 100 if total_rows > 0 else 0
    fc_pct_val = fc_act / total_act * 100 if total_act > 0 else 0
    h_pct_rows = len(hi) / total_rows * 100 if total_rows > 0 else 0
    h_pct_val = h_act / total_act * 100 if total_act > 0 else 0

    fc_wmape = fc_err / fc_act * 100 if fc_act > 0 else float('nan')
    fc_bias = (fc_pred - fc_act) / fc_act * 100 if fc_act > 0 else float('nan')
    h_wmape = h_err / h_act * 100 if h_act > 0 else float('nan')
    h_bias = (h_pred - h_act) / h_act * 100 if h_act > 0 else float('nan')

    tot['fc_n'] += len(fc); tot['fc_act'] += fc_act; tot['fc_pred'] += fc_pred; tot['fc_err'] += fc_err
    tot['h_n'] += len(hi); tot['h_act'] += h_act; tot['h_pred'] += h_pred; tot['h_err'] += h_err

    print(f"  {snap:<12} | {len(fc):>5} {fc_pct_rows:>6.1f}% ${fc_act:>14,.0f} {fc_pct_val:>6.1f}% "
          f"{fc_wmape:>9.1f}% {fc_bias:>+9.1f}% | "
          f"{len(hi):>5} {h_pct_rows:>6.1f}% ${h_act:>14,.0f} {h_pct_val:>6.1f}% "
          f"{h_wmape:>9.1f}% {h_bias:>+9.1f}%")

t = tot
total_n = t['fc_n'] + t['h_n']
total_act = t['fc_act'] + t['h_act']
print(f"  {'-'*12} | {'-'*5} {'-'*7} {'-'*16} {'-'*7} {'-'*10} {'-'*10} | "
      f"{'-'*5} {'-'*7} {'-'*16} {'-'*7} {'-'*10} {'-'*10}")
print(f"  {'TOTAL':<12} | {t['fc_n']:>5} {t['fc_n']/total_n*100:>6.1f}% ${t['fc_act']:>14,.0f} {t['fc_act']/total_act*100:>6.1f}% "
      f"{t['fc_err']/t['fc_act']*100:>9.1f}% {(t['fc_pred']-t['fc_act'])/t['fc_act']*100:>+9.1f}% | "
      f"{t['h_n']:>5} {t['h_n']/total_n*100:>6.1f}% ${t['h_act']:>14,.0f} {t['h_act']/total_act*100:>6.1f}% "
      f"{t['h_err']/t['h_act']*100:>9.1f}% {(t['h_pred']-t['h_act'])/t['h_act']*100:>+9.1f}%")
