# Rebuttal materials (order-sweep paper)

Additional experiments run during the review period, each in response to a specific
reviewer request. Nothing here changes the submitted paper or its artifacts; the
original results directories are untouched. All material is anonymized.

## Layout
```
rebuttal/
  README.md
  verify_rebuttal.py          recomputes every number quoted in the author responses
  cells/                      self-contained Colab cells (each mounts Drive and runs alone)
    01_aic_bic_baseline.py    AIC/BIC order selection on the paired validation suite + noise band
    02_runtime.py             full-sweep cost vs number of variables, paper implementation and a
                              direct constructor; the latter reaches d=20 pairwise
    07_runtime_d50.py         timed sweep at 50 variables and 10,000 observations
    05_pvalue_vs_coverage.py  does the accepted p-value predict certificate coverage?
    06_beijing_application.py second real-data application (air-quality panel)
  artifacts/
    aicbic_main.csv, aicbic_disc.csv, aicbic_summary.txt
    runtime.txt               (transcribed from the original run printout; rerun cell 02 to regenerate
                              it as a cell-written file with both implementations and the d=20 point)
    runtime_d50.txt           degree-two pairwise sweep at 50 variables, 10,000 rows: 35.3 s
    per_seed_coverage.csv     copy of the paper's coverage artifact, needed by the p-value check
    pvalue_coverage.txt
    beijing_app.csv, beijing_app.txt
    beijing_panel.npz         the air-quality panel used by cell 06 (12 sites, 11 columns,
                              per-column standardized; derived from the public Beijing
                              multi-site air-quality dataset)
```

## What each experiment answers
- **AIC/BIC baseline** (request for a comparison with information criteria). Same 640 datasets
  and seeds as the main suite, plus the noise-band grid. Pairing is asserted in the cell: every
  regenerated noise-band dataset must match the SHA-256 stored in the frozen artifact, and sampled
  main-suite datasets (whose frozen file predates the hash column) must reproduce the frozen
  selector's recorded statistic float-exactly. At noise 0.5, AIC overselects 15 of 160
  null cells (6 of 80 order-one, 9 of 80 order-two), BIC none, the sweep one. Noiseless cells
  degenerate for likelihood criteria and are reported but not used for the comparison.
- **Runtime** (request for cost at more variables). Full-sweep seconds at d = 3, 5, 8 (K = 3)
  and d = 10 (K = 2) as quoted in the responses were measured with the paper's implementation, which
  enumerates all exponent tuples (5^d of them) and so dominates the cost at d >= 8 and is infeasible
  near d = 20; cell 02 also times a direct constructor that builds the same monomials without
  enumeration and reaches d = 20 pairwise. At 50 variables and 10,000 rows the degree-four pairwise
  class (7,551 features) exceeds the 7,500 training rows and is not identified; the degree-two pairwise
  class (1,326 features, 106 MB) sweeps in 35.3 s (cell 07).
- **p-value vs coverage** (request for guidance on when the bound is trustworthy). Among the
  137 stop-at-2 replicates of the coverage study, acceptances with p in (0.05, 0.10] cover
  21 of 23 while acceptances with p > 0.5 cover 16 of 22: the accepted p-value does not flag
  unreliable bounds, which motivates the sample-splitting variant promised for the revision.
- **Second real-data application** (request for a domain beyond the panel in the paper).
  Standardized PM2.5 on three meteorological triples over 418,297 hourly rows with
  site-by-30-day blocks: additivity rejected on all three (p < 0.001), third order selected
  on two, pairwise with a 0.0001 bound on the third; blocked-fold CV-1SE selects additive on
  two and pairwise on one; the row-split sweep selects three where blocked selected two.

## Verify
```
cd rebuttal && python verify_rebuttal.py
```
Expected: `25 checks, 0 failures` and `ALL REBUTTAL NUMBERS VERIFIED`.

## Rerun
Copy the repository to Drive as `MyDrive/ORDER_SWEEP` (cells resolve paths from that base;
cell 01 and cell 05 read the paper's `results/ordersweep_validation`, `results/discriminator_power`
and `results/coverage_indep` artifacts; cell 06 reads `data/beijing_panel.npz`). Each cell writes
to `results/rebuttal/`.
