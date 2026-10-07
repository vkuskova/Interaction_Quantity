# Self-contained cell: timed sweep at a setting of 50 variables and 10,000 observations.
# Degree-two pairwise class (1,326 features): one full sweep, S=10 splits, K=2, plus the feature count and
# training-row count for the degree-four pairwise class, which is not identified at this n.
from google.colab import drive
drive.mount('/content/drive')
import os, time
import numpy as np
from itertools import product as iproduct
from math import comb
OUT = '/content/drive/MyDrive/ORDER_SWEEP/results/rebuttal'; os.makedirs(OUT, exist_ok=True)
def monomial_exps(d, D, k): return [c for c in iproduct(range(D+1), repeat=d) if sum(c) <= D and sum(1 for x in c if x > 0) <= k]
def pairwise_design(X, D):
    """Pairwise-capped design built directly (enumeration over product(range(D+1), repeat=50) is infeasible)."""
    n, d = X.shape; cols = [np.ones(n)]
    for j in range(d):
        for p in range(1, D+1): cols.append(X[:, j]**p)
    for i in range(d):
        for j in range(i+1, d):
            for a in range(1, D+1):
                for b in range(1, D+1):
                    if a + b <= D: cols.append((X[:, i]**a)*(X[:, j]**b))
    return np.column_stack(cols)
d, n, S = 50, 10_000, 10
rng = np.random.default_rng(0); X = rng.standard_normal((n, d)); h = X[:, 0]*X[:, 1] + rng.standard_normal(n)
n_tr = int(0.75*n)
p4 = 1 + d*4 + comb(d, 2)*6; p2 = 1 + d*2 + comb(d, 2)*1
lines = [f"d={d} n={n:,} train rows {n_tr:,}: degree-4 pairwise class {p4:,} features (> training rows, not identified); "
         f"degree-2 pairwise class {p2:,} features"]
print(lines[0], flush=True)
t0 = time.time(); P0 = pairwise_design(X, 2); assert P0.shape[1] == p2
lines.append(f"design build {time.time()-t0:.1f}s, {P0.nbytes/1e6:.0f} MB"); print(lines[-1], flush=True)
t1 = time.time()
for s in range(S):
    idx = np.random.default_rng(s).permutation(n); tr = idx[:n_tr]
    for k_cols in (1 + d*2, p2):   # C_1 (univariate, 101 features) and C_2 (full pairwise) on the same split
        P = P0[:, :k_cols]; mu = P[tr].mean(0); sd = P[tr].std(0); sd[sd == 0] = 1; Q = (P - mu)/sd; Q[:, 0] = 1
        np.linalg.lstsq(Q[tr], h[tr] - h[tr].mean(), rcond=None)
dt = time.time() - t1
lines.append(f"full degree-2 pairwise sweep (S={S}, K=2, 20 fits): {dt:.1f}s"); print(lines[-1], flush=True)
open(os.path.join(OUT, 'runtime_d50.txt'), 'w').write("\n".join(lines) + "\n"); print("wrote runtime_d50.txt")
