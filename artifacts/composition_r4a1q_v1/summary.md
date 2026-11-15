# R4A-1Q Raw Pointer Metric Reconciliation

Outcome: `R4A1Q_METRICS_RECONCILED_SCOPE_MISMATCH`

The same raw pointer tensor is used, but the historical summaries reduce different masks:

- R4A-1P raw oracle-coordinate mass is foreground-only.
- R4A-0R pointer entropy and max probability are all-cell means over all 49 target rows, including background rows.

|seed|fg argmax|fg mass|fg max|fg entropy|all-cell max|all-cell entropy|same-color wrong|
|---|---:|---:|---:|---:|---:|---:|---:|
|20260628|1.000000|1.000000|1.000000|0.000000|0.260308|2.938722|0.000000|
|20260629|1.000000|1.000000|1.000000|0.000000|0.260308|2.938722|0.000000|
|20260630|1.000000|1.000000|1.000000|0.000000|0.260308|2.938722|0.000000|
