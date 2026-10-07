# Self-contained cell: second real-data application.
# Beijing multi-site air quality: PM2.5 regressed on meteorological triples, blocked sweep
# (site x 720-hour blocks) vs blocked-fold CV-1SE, with a row-split sweep as the leakage check.
# Outcome units are detected from the stored column: if it is already standardized (mean ~0, sd ~1,
# negative values present) it is used as stored; only a raw positive series is log-transformed.
from google.colab import drive
drive.mount('/content/drive')
import os, json, csv, time, hashlib
import numpy as np
from itertools import product as iproduct
from scipy import stats

OUT = '/content/drive/MyDrive/ORDER_SWEEP/results/rebuttal'; os.makedirs(OUT, exist_ok=True)
DATA = '/content/drive/MyDrive/ORDER_SWEEP/data/beijing_panel.npz'
assert os.path.exists(DATA), f"beijing_panel.npz not found at {DATA}"
print("data:", DATA)

COLNAMES = ["PM2.5","PM10","SO2","NO2","CO","O3","TEMP","PRES","DEWP","RAIN","WSPM"]
TRIPLES = {"temp_dewp_wspm": ["TEMP","DEWP","WSPM"],
           "temp_pres_wspm": ["TEMP","PRES","WSPM"],
           "dewp_pres_wspm": ["DEWP","PRES","WSPM"]}
OUTCOME = "PM2.5"; BLOCK_HOURS = 720

z = np.load(DATA, allow_pickle=True)
site_keys = sorted(k for k in z.keys() if k.startswith("X__"))
assert len(site_keys) == 12, f"expected 12 sites, found {len(site_keys)}"
parts, blocks_all = [], []
for si, k in enumerate(site_keys):
    arr = z[k].astype(float); parts.append(arr)
    blocks_all.append(si * 100000 + np.arange(len(arr)) // BLOCK_HOURS)   # site x 720-hour block ids
pooled = np.concatenate(parts, 0); blocks_all = np.concatenate(blocks_all)
assert pooled.shape[1] == 11
print(f"pooled rows {len(pooled):,}, blocks {len(np.unique(blocks_all)):,}")
yi0 = COLNAMES.index("PM2.5"); ycol = pooled[:, yi0]; yv = ycol[~np.isnan(ycol)]
print(f"PM2.5 as stored: n={len(yv):,} min={yv.min():.3f} mean={yv.mean():.3f} sd={yv.std():.3f} share<=0={np.mean(yv<=0):.3f}")
STANDARDIZED = (yv.min() < 0) and (abs(yv.mean()) < 0.2) and (0.5 < yv.std() < 2.0)
OUTCOME_LABEL = "PM2.5 (standardized, as stored)" if STANDARDIZED else "log PM2.5 (raw series)"
print("outcome treatment:", OUTCOME_LABEL)

def monomial_exps(d, D, k): return [c for c in iproduct(range(D+1), repeat=d) if sum(c) <= D and sum(1 for x in c if x > 0) <= k]
def poly_design(X, D, k):
    exps = monomial_exps(X.shape[1], D, k); cols = []
    for e in exps:
        col = np.ones(X.shape[0])
        for j, p in enumerate(e):
            if p > 0: col = col * X[:, j]**p
        cols.append(col)
    return np.column_stack(cols), exps.index(tuple([0]*X.shape[1]))
def split_indices(n, seed, train_frac, blocks):
    if blocks is None:
        idx = np.random.default_rng(seed).permutation(n); return idx[:int(train_frac*n)], idx[int(train_frac*n):]
    uniq = np.unique(blocks); perm = np.random.default_rng(seed).permutation(len(uniq))
    tr_b = set(uniq[perm[:int(train_frac*len(uniq))]])
    m = np.fromiter((b in tr_b for b in blocks), bool, count=n); return np.where(m)[0], np.where(~m)[0]
def select_order(X, h, K=3, S=10, alpha=0.05, D=4, train_frac=0.75, blocks=None):
    """Frozen fixed-sequence selector with the blocked-split policy of the panel application."""
    n = X.shape[0]; designs = {k: poly_design(X, D, k) for k in range(1, K+1)}
    r2, ntr, nte = {}, [], []
    for s in range(S):
        tr, te = split_indices(n, s, train_frac, blocks); out = {}
        for k in range(1, K+1):
            P0, ci = designs[k]; mu = P0[tr].mean(0); sd = P0[tr].std(0); sd[sd == 0] = 1.0
            P = (P0 - mu)/sd; P[:, ci] = 1.0; hm = h[tr].mean()
            beta, *_ = np.linalg.lstsq(P[tr], h[tr]-hm, rcond=None); res = (h[te]-hm) - P[te]@beta
            out[k] = 1.0 - float((res@res)/np.sum((h[te]-h[te].mean())**2))
        r2[s] = out; ntr.append(len(tr)); nte.append(len(te))
    if blocks is None: ratio = np.mean(nte)/np.mean(ntr)
    else:
        nb = len(np.unique(blocks)); ntb = int(train_frac*nb); ratio = (nb-ntb)/ntb
    corr = 1.0/S + ratio; pf = {k: designs[k][0].shape[1] for k in range(1, K+1)}
    om = float(np.mean([1.0 - r2[s][K] for s in range(S)])); n_tr_mean = float(np.mean(ntr)); stat = {}
    for k in range(1, K):
        g = np.array([r2[s][K] - r2[s][k] for s in range(S)]); m = g.mean(); v = g.var(ddof=1)*corr
        opt = (pf[K]-pf[k])*om/n_tr_mean
        if v > 0: t = m/np.sqrt(v); p = 1.0 - stats.t.cdf(t, df=S-1); ub = m + stats.t.ppf(1-alpha, df=S-1)*np.sqrt(v) + opt
        else: p = 0.0 if m > 0 else 1.0; ub = m + opt
        stat[k] = {"mean": float(m), "p": float(p), "ub": float(ub), "pi": float((g > 0).mean())}
    khat, ubc = K, None
    for k in range(1, K):
        if stat[k]["p"] > alpha: khat, ubc = k, stat[k]["ub"]; break
    return khat, stat, ubc
def cv1se_khat(X, h, K=3, D=4, folds=5, seed=0, blocks=None):
    n = X.shape[0]; rng = np.random.default_rng(seed)
    if blocks is None:
        idx = rng.permutation(n); fold = np.empty(n, dtype=int); fold[idx] = np.arange(n) % folds
    else:
        uniq = np.unique(blocks); bf = {b: f for b, f in zip(uniq[rng.permutation(len(uniq))], np.arange(len(uniq)) % folds)}
        fold = np.fromiter((bf[b] for b in blocks), int, count=n)
    errs = {k: [] for k in range(1, K+1)}
    for k in range(1, K+1):
        P0, ci = poly_design(X, D, k)
        for f in range(folds):
            tr, te = fold != f, fold == f
            mu = P0[tr].mean(0); sd = P0[tr].std(0); sd[sd == 0] = 1.0; P = (P0-mu)/sd; P[:, ci] = 1.0
            hm = h[tr].mean(); beta, *_ = np.linalg.lstsq(P[tr], h[tr]-hm, rcond=None)
            errs[k].append(float(np.mean(((h[te]-hm) - P[te]@beta)**2)))
    means = {k: float(np.mean(v)) for k, v in errs.items()}; ses = {k: float(np.std(v, ddof=1)/np.sqrt(folds)) for k, v in errs.items()}
    kb = min(means, key=means.get); return min(k for k in range(1, K+1) if means[k] <= means[kb] + ses[kb])

t0 = time.time(); rows = []; lines = []
yi = COLNAMES.index(OUTCOME)
for name, cols in TRIPLES.items():
    ci_ = [COLNAMES.index(c) for c in cols]
    sub = pooled[:, ci_ + [yi]]; ok = ~np.isnan(sub).any(1)
    if not STANDARDIZED: ok = ok & (sub[:, -1] > 0)
    X = sub[ok][:, :3]; blocks = blocks_all[ok]
    h = sub[ok][:, 3] if STANDARDIZED else np.log(sub[ok][:, 3])
    dh = hashlib.sha256(X.tobytes() + h.tobytes()).hexdigest()
    kb, sb, ub_b = select_order(X, h, blocks=blocks)
    kr, sr, ub_r = select_order(X, h, blocks=None)
    kcv = cv1se_khat(X, h, blocks=blocks)
    rows.append({"triple": name, "outcome": OUTCOME_LABEL, "n": len(h), "blocks": int(len(np.unique(blocks))),
                 "data_hash": dh, "khat_blocked": kb, "khat_row": kr, "khat_cv1se_blocked": kcv,
                 "rem1_mean": sb[1]["mean"], "rem1_p": sb[1]["p"], "rem2_mean": sb[2]["mean"], "rem2_p": sb[2]["p"],
                 "pi1": sb[1]["pi"], "pi2": sb[2]["pi"], "ub_cert": "" if ub_b is None else ub_b})
    line = (f"{name}: n={len(h):,} blocks={len(np.unique(blocks))} | blocked khat={kb} row khat={kr} CV1SE(blocked)={kcv} | "
            f"rem1={sb[1]['mean']:.4f} (p={sb[1]['p']:.3f}) rem2={sb[2]['mean']:.4f} (p={sb[2]['p']:.3f}) | "
            f"cert={'NA' if ub_b is None else f'{ub_b:.4f}'} ({time.time()-t0:.0f}s)")
    lines.append(line); print(line, flush=True)
with open(os.path.join(OUT, 'beijing_app.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
open(os.path.join(OUT, 'beijing_app.txt'), 'w').write("\n".join(lines) + "\n")
json.dump({"experiment": "beijing_app", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "data": DATA,
           "outcome": OUTCOME_LABEL, "triples": TRIPLES, "block": f"site x {BLOCK_HOURS}-hour chunks",
           "K": 3, "D": 4, "S": 10, "alpha": 0.05, "numpy": np.__version__},
          open(os.path.join(OUT, 'beijing_app_meta.json'), 'w'), indent=2)
print("wrote beijing_app.csv, beijing_app.txt, beijing_app_meta.json")
