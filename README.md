# Reproducibility bundle for paper "How Many Variables Interact?"

Anonymized supplementary material for the order-sweep interaction-order
selection paper. Two independent levels of reproduction:

- **Step 1 — Verify:** recompute every number in the paper's tables from
  the shipped per-replicate artifacts, without running any experiment.
- **Step 2 — Rerun:** regenerate all artifacts from scratch in a hosted
  Python environment (e.g. Colab), then re-verify.

## Layout

```
order_sweep_repro/
  README.md
  verify_tables.py           Step 1 (Python 3, standard library only)
  results/                   per-replicate artifacts, one dir per experiment
    ordersweep_validation/   Tables: main suite + appendix full table
    ordersweep_censoring/    Table: censoring exhibit
    baselines_validation/    Table: baselines  
    discriminator_power/     Table + figure: order sweep vs CV-1SE
    vdem_ordersweep/         Table: panel application
    extensions/              Table: baselines on panels; d=5 suite
  notebooks/
    01_method_validation.ipynb   main suite + censoring exhibit
    02_baselines.ipynb           NID / hierNet / CV-1SE, paired
    03_discriminator.ipynb       noise-band separation
    04_panel_application.ipynb   blocked-split validation + panels
    05_extensions.ipynb          d=5, baselines-on-panels, split policy
  data/
    HDL_merged_notdev_selected.csv   country-year panel
```

Each experiment directory carries `per_seed*.csv` (every row stamped with
an `experiment` identifier), `metadata*.json` (configuration and a
provenance hash over function source plus configuration constants), and,
where applicable, `check*.txt` (the pre-registered acceptance-check
transcript).


## Step 1 — Verify

```
python verify_tables.py
```

No dependencies beyond the standard library. The script recomputes each
table cell from the artifact files using the paper's aggregation, compares
at the precision displayed in the paper (half-unit-in-last-place
tolerance), and recomputes closed-form reference columns from the
formulas. Expected output with all artifacts present ends:

```
... table cells checked, 0 failures.
ALL PRESENT TABLES VERIFIED
```

## Step 2 — Rerun from scratch

1. Copy this directory to Drive as `MyDrive/ORDER_SWEEP` (the notebooks
   resolve paths from that base; `data/` is included). Notebook 02
   additionally reads the frozen method artifact written by notebook 01;
   notebook 05 reads notebook 02's stored NID calibration.
2. Preserve the shipped artifacts (`mv results results_shipped`) — reruns
   write into `results/`.
3. Run each notebook top to bottom. Notebook 02 requires the R `hierNet`
   package through `rpy2`; if unavailable it falls back to an all-pairs
   1SE lasso and records which implementation ran in its metadata (the
   paper reports the implementation that produced its numbers).
4. Each notebook ends with a verification cell that re-reads its own
   artifacts and reruns the pre-registered checks.
5. Re-verify the tables: `python verify_tables.py --results-dir results`.

Determinism: all data generation, splits, and subsamples are seeded by a
CRC-based scheme; float-exact reproduction of per-replicate statistics was
observed across independent environments under numpy 2.0.x. Minor
floating-point drift under other versions is absorbed by the acceptance
tolerances and the displayed-precision comparison of Step 1.

## Notes

- Data provenance: `HDL_merged_notdev_selected.csv` is a country-year
  panel of political-institutional indicators derived from the V-Dem
  dataset; consult the original source for terms of use.
- The theory the paper relies on (the closed-form order-degradation law
  used as ground truth, the identifiability limits) is developed in the
  anonymized companion paper provided as separate supplementary material.
- This bundle contains no author-identifying information.
