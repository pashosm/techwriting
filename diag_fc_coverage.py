"""One-off diagnostic: why does FC row count not grow over time?"""
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from aging_v9 import load_data, build_forecast_registry

df = load_data('training_data_anonymized.csv')

print("=" * 100)
print("FC coverage diagnostic across snapshots")
print("=" * 100)
print(f"{'Snapshot':<12} {'Test':>6} {'TestHasFC':>10} {'TPKeyHit':>9} {'VintAvail':>10} "
      f"{'AnyValid':>9} {'AvgVint':>8}")

snapshots = pd.date_range('2024-06-01', '2025-03-01', freq='MS')
for snap in snapshots:
    horizon_end = snap + pd.DateOffset(months=3)
    test = df[
        (df['Target_Period_Start'] >= snap) &
        (df['Target_Period_Start'] < horizon_end) &
        (df['Reference_Month'] < snap)
    ]
    visible = df[df['Reference_Month'] < snap]
    registry = build_forecast_registry(visible)

    # Break down test coverage
    test_has_fc = int((test['Has_Forecast'] == 1).sum())

    tp_key_hit = 0           # target period has ANY entry in registry
    any_vintage_available = 0  # tp_key_hit AND at least one vintage exists
    any_valid = 0            # vintage passes temporal + fc_value>0
    total_vint = 0
    n_with_valid = 0

    for _, row in test.iterrows():
        tp_key = (row['gsa_site'], row['Target_Period_Start'], row['Target_Period_End'])
        if tp_key not in registry:
            continue
        tp_key_hit += 1
        vintages = registry[tp_key]
        if vintages:
            any_vintage_available += 1
        valid = [v for v in vintages
                 if v['ref_month'] <= row['Reference_Month'] and v['fc_value'] > 0]
        if valid:
            any_valid += 1
            total_vint += len(valid)
            n_with_valid += 1

    avg_vint = total_vint / n_with_valid if n_with_valid > 0 else 0
    print(f"{snap.strftime('%Y-%m-%d'):<12} {len(test):>6,} {test_has_fc:>10,} "
          f"{tp_key_hit:>9,} {any_vintage_available:>10,} {any_valid:>9,} {avg_vint:>8.2f}")

print()
print("Columns:")
print("  Test         = test rows (TPS in [snap, snap+3mo), ref_m < snap)")
print("  TestHasFC    = test rows where the row ITSELF has Has_Forecast=1")
print("  TPKeyHit     = test rows whose (site,TPS,TPE) exists in registry")
print("  VintAvail    = TPKeyHit AND at least one vintage is stored")
print("  AnyValid     = passes temporal (ref_m <= test.ref_m) + fc_value>0")
print("  AvgVint      = average valid vintages per row with AnyValid")
