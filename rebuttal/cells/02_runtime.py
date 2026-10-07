# Self-contained cell: runtime of one full sweep vs number of variables (n=20,000, S=10, D=4).
# Two implementations are timed, because they answer different questions:
#   (a) the paper's implementation, which enumerates every exponent tuple in product(range(D+1), repeat=d);
#       this reproduces the transcript figures at d = 3, 5, 8, 10 but is infeasible beyond d ~ 12
#       (5^d tuples), which is why the original run never finished at d = 20;
#   (b) a direct constructor that builds the same monomial set without enumeration, timed at the same d
#       and additionally at d = 20 pairwise (1,221 features), the "minutes" claim in the responses.
# Both produce identical design matrices (asserted below at d <= 10).
from google.colab import drive
drive.mount('/content/drive')
import os, time
import numpy as np
from itertools import product as iproduct, combinations
OUT = '/content/drive/MyDrive/ORDER_SWEEP/results/rebuttal'; os.makedirs(OUT, exist_ok=True)

def poly_design_enum(X, D, k):                       # the paper's implementation
    exps = [c for c in iproduct(range(D+1), repeat=X.shape[1]) if sum(c) <= D and sum(1 for x in c if x > 0) <= k]
    cols = []
    for e in exps:
        col = np.ones(X.shape[0])
        for j, p in enumerate(e):
            if p > 0: col = col * X[:, j]**p
        cols.append(col)
    return np.column_stack(cols)
def poly_design_direct(X, D, k):                     # same monomials, no enumeration over inactive variables
    n, d = X.shape; cols = [np.ones(n)]
    for m in range(1, k+1):
        for S in combinations(range(d), m):
            for e in iproduct(range(1, D+1), repeat=m):
                if sum(e) <= D:
                    col = np.ones(n)
                    for j, p in zip(S, e): col = col * X[:, j]**p
                    cols.append(col)
    return np.column_stack(cols)
def sweep_seconds(X, h, K, builder, S=10, D=4, train_frac=0.75):
    n = X.shape[0]; n_tr = int(train_frac*n); t0 = time.time()
    designs = {k: builder(X, D, k) for k in range(1, K+1)}
    for s in range(S):
        idx = np.random.default_rng(s).permutation(n); tr = idx[:n_tr]
        for k in range(1, K+1):
            P0 = designs[k]; mu = P0[tr].mean(0); sd = P0[tr].std(0); sd[sd == 0] = 1; P = (P0 - mu)/sd; P[:, 0] = 1
            np.linalg.lstsq(P[tr], h[tr] - h[tr].mean(), rcond=None)
    return time.time() - t0, designs[K].shape[1]

lines = ["n=20,000, S=10, D=4; one full sweep (design construction + 10*K fits)"]
for d, K in [(3, 3), (5, 3), (8, 3), (10, 2)]:
    rng = np.random.default_rng(0); X = rng.standard_normal((20000, d)); h = X[:, 0]*X[:, 1] + rng.standard_normal(20000)
    # identical design sets from both builders (same columns up to order)
    A = poly_design_enum(X[:2000], 4, K); B = poly_design_direct(X[:2000], 4, K)
    assert A.shape == B.shape and np.allclose(np.sort(np.abs(A).sum(0)), np.sort(np.abs(B).sum(0)))
    te, p = sweep_seconds(X, h, K, poly_design_enum); td, _ = sweep_seconds(X, h, K, poly_design_direct)
    lines.append(f"d={d:2d} K={K} p_K={p:5d} seconds={te:7.1f} (paper implementation)   seconds={td:7.1f} (direct constructor)")
    print(lines[-1], flush=True)
rng = np.random.default_rng(0); X = rng.standard_normal((20000, 20)); h = X[:, 0]*X[:, 1] + rng.standard_normal(20000)
td, p = sweep_seconds(X, h, 2, poly_design_direct)
lines.append(f"d=20 K=2 p_K={p:5d} seconds={td:7.1f} (direct constructor; enumeration infeasible at this d)")
print(lines[-1], flush=True)
open(os.path.join(OUT, 'runtime.txt'), 'w').write("\n".join(lines) + "\n"); print("wrote runtime.txt")
