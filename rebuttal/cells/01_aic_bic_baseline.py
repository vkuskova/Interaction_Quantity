# Self-contained cell: B1 AIC/BIC order selection on the paired validation suite + noise-band grid
# . Same seeds as the frozen artifacts; pairs with the stored sweep selections.
from google.colab import drive
drive.mount('/content/drive')
import os, csv, time, zlib
import numpy as np
from itertools import product as iproduct
import hashlib
from scipy import stats
BASE = '/content/drive/MyDrive/ORDER_SWEEP/results'
OUT = os.path.join(BASE, 'rebuttal'); os.makedirs(OUT, exist_ok=True)
def monomial_exps(d, D, k): return [c for c in iproduct(range(D+1), repeat=d) if sum(c) <= D and sum(1 for x in c if x > 0) <= k]
def poly_design(X, D, k):
    exps = monomial_exps(X.shape[1], D, k); cols = []
    for e in exps:
        col = np.ones(X.shape[0])
        for j, p in enumerate(e):
            if p > 0: col = col * X[:, j]**p
        cols.append(col)
    return np.column_stack(cols)
def ic_select(X, h, K=3, D=4):
    n = len(h); hc = h - h.mean(); aic = {}; bic = {}
    for k in range(1, K+1):
        P = poly_design(X, D, k); mu = P.mean(0); sd = P.std(0); sd[sd == 0] = 1; P = (P - mu)/sd; P[:, 0] = 1
        beta, *_ = np.linalg.lstsq(P, hc, rcond=None); rss = float(np.sum((hc - P@beta)**2)); p = P.shape[1]
        ll = n*np.log(max(rss, 1e-300)/n); aic[k] = ll + 2*p; bic[k] = ll + p*np.log(n)
    return min(aic, key=aic.get), min(bic, key=bic.get)
def make_X(n, dep, rng):
    if dep == "indep": return rng.standard_normal((n, 3))
    if dep.startswith("pair"):
        rho = float(dep[4:]); x1, x2, z = rng.standard_normal((3, n)); return np.column_stack([x1, x2, rho*x1 + np.sqrt(1-rho**2)*z])
    rho = float(dep[4:]); g = rng.standard_normal(n); z = rng.standard_normal((3, n)); return (np.sqrt(rho)*g + np.sqrt(1-rho)*z).T
def make_h(X, dgp, s, rng):
    x1, x2, x3 = X.T
    f = {"o1": x1 + np.tanh(x2) - 0.5*x3, "o2": x1*x2 + np.tanh(x3), "o3": x1*x2*x3, "o3mix": x1*x2*x3 + x1 + x2*x3}[dgp]
    return f + s*rng.standard_normal(len(f))
TRUE = {"o1": 1, "o2": 2, "o3": 3, "o3mix": 3}
frozen = list(csv.DictReader(open(os.path.join(BASE, 'ordersweep_validation', 'per_seed.csv')))); assert len(frozen) == 640
# ---- pairing check for the main suite: the frozen file predates the data_hash column, so the strongest
# available evidence is float-exact reproduction of the frozen selector's recorded statistic on sampled rows.
def select_rem1(X, h, K=3, S=10, D=4, train_frac=0.75):
    n = X.shape[0]; n_tr = int(train_frac*n); designs = {k: poly_design(X, D, k) for k in (1, K)}; g = []
    for s_ in range(S):
        idx = np.random.default_rng(s_).permutation(n); tr, te = idx[:n_tr], idx[n_tr:]; r2 = {}
        for k in (1, K):
            P0 = designs[k]; mu = P0[tr].mean(0); sd = P0[tr].std(0); sd[sd == 0] = 1; P = (P0 - mu)/sd; P[:, 0] = 1
            hm = h[tr].mean(); beta, *_ = np.linalg.lstsq(P[tr], h[tr]-hm, rcond=None); res = (h[te]-hm) - P[te]@beta
            r2[k] = 1.0 - float((res@res)/np.sum((h[te]-h[te].mean())**2))
        g.append(r2[K] - r2[1])
    return float(np.mean(g))
for i in (0, 213, 639):   # three rows spanning the suite
    r = frozen[i]; dgp, dep, s, rep = r["dgp"], r["dependence"], float(r["sigma"]), int(r["rep"])
    rng = np.random.default_rng(10_000 + zlib.crc32(f"{dgp}|{dep}|{s}".encode()) % 1000 + rep*977)
    X = make_X(20000, dep, rng); h = make_h(X, dgp, s, rng)
    assert abs(select_rem1(X, h) - float(r["rem1_mean"])) < 1e-10, f"main-suite pairing failed at row {i}"
print("main-suite pairing: frozen rem1_mean reproduced float-exactly on sampled rows")
t0 = time.time(); rows = []
for r in frozen:
    dgp, dep, s, rep = r["dgp"], r["dependence"], float(r["sigma"]), int(r["rep"])
    rng = np.random.default_rng(10_000 + zlib.crc32(f"{dgp}|{dep}|{s}".encode()) % 1000 + rep*977)
    X = make_X(20000, dep, rng); h = make_h(X, dgp, s, rng); ka, kb = ic_select(X, h)
    rows.append({"dgp": dgp, "dependence": dep, "sigma": s, "rep": rep, "true_order": TRUE[dgp],
                 "khat_sweep": int(r["khat"]), "khat_aic": ka, "khat_bic": kb})
with open(os.path.join(OUT, 'aicbic_main.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
def rate(flt, pred): xs = [x for x in rows if flt(x)]; return sum(pred(x) for x in xs), len(xs)
story = []
for nm, c in [("sweep", "khat_sweep"), ("AIC", "khat_aic"), ("BIC", "khat_bic")]:
    o5 = rate(lambda x: x["dgp"] in ("o1", "o2") and x["sigma"] == 0.5, lambda x, c=c: x[c] > x["true_order"])
    o0 = rate(lambda x: x["dgp"] in ("o1", "o2") and x["sigma"] == 0.0, lambda x, c=c: x[c] > x["true_order"])
    pw = rate(lambda x: x["dgp"] in ("o3", "o3mix") and x["dependence"] != "pair0.9", lambda x, c=c: x[c] == 3)
    story.append(f"{nm}: null-over sigma=0.5 {o5[0]}/{o5[1]} | null-over sigma=0 {o0[0]}/{o0[1]} | power {pw[0]}/{pw[1]}"); print(story[-1])
for dgp in ("o1", "o2"):
    for c, nm in [("khat_aic", "AIC"), ("khat_bic", "BIC")]:
        o = rate(lambda x, d=dgp: x["dgp"] == d and x["sigma"] == 0.5, lambda x, c=c: x[c] > x["true_order"])
        story.append(f"  {nm} overselection on {dgp} nulls at sigma=0.5: {o[0]}/{o[1]}"); print(story[-1])
disc = list(csv.DictReader(open(os.path.join(BASE, 'discriminator_power', 'per_seed.csv'))))
frozen_hash = {(float(r["sigma"]), int(r["rep"])): r["data_hash"] for r in disc}
drows = []
for sig in [1.0, 1.5, 2.0, 2.5, 3.0]:
    for rep in range(20):
        rng = np.random.default_rng((90_000 + zlib.crc32(f"disc|{sig:.17g}".encode()) + rep*977) % 2**32)
        x1, x2, z = rng.standard_normal((3, 20000)); x3 = 0.9*x1 + np.sqrt(0.19)*z
        X = np.column_stack([x1, x2, x3]); h = x1*x2*x3 + sig*rng.standard_normal(20000)
        dh = hashlib.sha256(X.astype(np.float64).tobytes() + h.astype(np.float64).tobytes()).hexdigest()
        assert dh == frozen_hash[(sig, rep)], f"noise-band pairing failed at sigma={sig} rep={rep}"
        ka, kb = ic_select(X, h); drows.append({"sigma": sig, "rep": rep, "khat_aic": ka, "khat_bic": kb})
print("noise-band pairing: all 100 regenerated datasets match the frozen SHA-256 hashes")
with open(os.path.join(OUT, 'aicbic_disc.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(drows[0].keys())); w.writeheader(); w.writerows(drows)
story.append("noise band, k=3 selections out of 20:  sweep | CV-1SE | AIC | BIC")
for sig in [1.0, 1.5, 2.0, 2.5, 3.0]:
    ds = [r for r in disc if float(r["sigma"]) == sig]
    line = (f"  sigma={sig}: {sum(int(r['khat_ordersweep'])==3 for r in ds)} | {sum(int(r['khat_cv1se'])==3 for r in ds)} | "
            f"{sum(d['khat_aic']==3 for d in drows if d['sigma']==sig)} | {sum(d['khat_bic']==3 for d in drows if d['sigma']==sig)}")
    story.append(line); print(line)
open(os.path.join(OUT, 'aicbic_summary.txt'), 'w').write("\n".join(story) + "\n")
print(f"done {time.time()-t0:.0f}s; wrote aicbic_main.csv, aicbic_disc.csv, aicbic_summary.txt")
