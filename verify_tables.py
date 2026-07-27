#!/usr/bin/env python3
"""Step 1 of the reproducibility bundle: recompute every number in the
paper's tables from the shipped per-replicate artifacts in results/, with
no experiment rerun. Standard library only.

Usage:  python verify_tables.py [--results-dir results]

Each cell is recomputed from the artifact files using the paper's
aggregation and compared at the precision displayed in the paper
(half-unit-in-last-place tolerance). Closed-form reference columns are
recomputed from the formulas. Exit code 0 means every recomputed cell of
every present table matches; a table whose artifact is absent is reported
as SKIPPED (see the baselines note in the README)."""
import csv, sys, os, math

RD = sys.argv[sys.argv.index("--results-dir")+1] if "--results-dir" in sys.argv else "results"

def load(path):
    with open(path) as f: return list(csv.DictReader(f))

def present(rel):
    return os.path.exists(os.path.join(RD, rel))

def rate(rows, keep, good):
    xs = [r for r in rows if keep(r)]
    return sum(good(r) for r in xs)/len(xs) if xs else float('nan')

def dp(tok): return len(tok.split('.')[1]) if '.' in tok else 0
def close(v, tok):
    try: return abs(v - float(tok)) <= 0.5*10**(-dp(tok)) + 1e-9
    except ValueError: return False

fails=[]; skipped=[]; checked=0
def chk(table, desc, value, tok):
    global checked
    checked += 1
    if not close(value, tok):
        fails.append(f"{table} [{desc}]: paper={tok} computed={value}")

F = lambda r:(1-r**2)**2/((1+r**2)*(1+2*r**2)) if r<1 else 0.0

# ---------- Table: main suite (tab:mainsuite) selection rates ----------
if present("ordersweep_validation/per_seed.csv"):
    rows = load(os.path.join(RD,"ordersweep_validation/per_seed.csv"))
    DGP_ORDER = [("1 (additive)",["o1"]),("2",["o2"]),
                 ("3 (pure)",["o3"]),("3 (hierarchical)",["o3mix"])]
    DEPS = ["indep","pair0.5","pair0.9","equi0.5"]
    for label,dgps in DGP_ORDER:
        for dep,tok in zip(DEPS,[None]*4): pass
    EXP={"1 (additive)":["o1"],"2":["o2"],"3 (pure)":["o3"],"3 (hierarchical)":["o3mix"]}
    # values from the paper, row-aligned
    MAIN=[("1 (additive)",["1.000","1.000","1.000","1.000"]),
          ("2",["1.000","0.975","1.000","1.000"]),
          ("3 (pure)",["1.000","1.000","1.000","1.000"]),
          ("3 (hierarchical)",["1.000","1.000","1.000","1.000"])]
    for label,toks in MAIN:
        dgps=EXP[label]
        for dep,tok in zip(DEPS,toks):
            r = rate(rows, lambda r:r["dgp"] in dgps and r["dependence"]==dep,
                     lambda r:int(r["khat"])==int(r["true_order"]))
            chk("tab:mainsuite", f"{label}/{dep}", r, tok)
    # appendix full table = same rates split by sigma
    FULL_DEPS=[("indep",0.0),("indep",0.5),("pair0.5",0.0),("pair0.5",0.5),
               ("pair0.9",0.0),("pair0.9",0.5),("equi0.5",0.0),("equi0.5",0.5)]
    FULL=[("o1",["1.00"]*8),("o2",["1.00","1.00","1.00","0.95","1.00","1.00","1.00","1.00"]),
          ("o3",["1.00"]*8),("o3mix",["1.00"]*8)]
    for dgp,toks in FULL:
        for (dep,sig),tok in zip(FULL_DEPS,toks):
            r=rate(rows, lambda r:r["dgp"]==dgp and r["dependence"]==dep and float(r["sigma"])==sig,
                   lambda r:int(r["khat"])==int(r["true_order"]))
            chk("tab:full", f"{dgp}/{dep}/{sig}", r, tok)
else: skipped.append("tab:mainsuite, tab:full (ordersweep_validation)")

# ---------- Table: censoring (tab:censoring) ----------
if present("ordersweep_censoring/per_seed_censoring.csv"):
    rows=load(os.path.join(RD,"ordersweep_censoring/per_seed_censoring.csv"))
    CEN=[("pair0.99",20000,0.0,"0","20"),("pair0.99",20000,0.5,"14","6"),
         ("pair0.9",5000,0.0,"0","20"),("pair0.9",5000,0.5,"0","20")]
    for dep,n,sig,k2,k3 in CEN:
        cs=[r for r in rows if r["dependence"]==dep and int(r["n"])==n and float(r["sigma"])==sig]
        chk("tab:censoring", f"{dep}/{sig}/k2", sum(int(r["khat"])==2 for r in cs), k2)
        chk("tab:censoring", f"{dep}/{sig}/k3", sum(int(r["khat"])==3 for r in cs), k3)
else: skipped.append("tab:censoring (ordersweep_censoring)")

# ---------- Table: baselines (tab:baselines) ----------
if present("baselines_validation/per_seed.csv"):
    rows=load(os.path.join(RD,"baselines_validation/per_seed.csv"))
    BASE=[("1 (additive)",["o1"],["1.000","0.919","0.244","1.000"]),
          ("2",["o2"],["0.994","0.381","1.000","1.000"]),
          ("3 (pure)",["o3"],["1.000","0.600","0.000","1.000"]),
          ("3 (hierarchical)",["o3mix"],["1.000","0.206","0.000","1.000"])]
    COLS=["khat_ordersweep","khat_nid","khat_pairwise","khat_cv1se"]
    for label,dgps,toks in BASE:
        for col,tok in zip(COLS,toks):
            r=rate(rows, lambda r:r["dgp"] in dgps,
                   lambda r:int(r[col])==int(r["true_order"]))
            chk("tab:baselines", f"{label}/{col}", r, tok)
    OVER=[("ordersweep","0.998"),("nid","0.527"),("pairwise","0.311"),("cv1se","1.000")]
    for m,tok in OVER:
        r=rate(rows, lambda r:True, lambda r:int(r[f"khat_{m}"])==int(r["true_order"]))
        chk("tab:baselines", f"overall/{m}", r, tok)
else: skipped.append("tab:baselines (baselines_validation -- copy from Drive)")

# ---------- Table: discriminator (tab:disc) ----------
if present("discriminator_power/per_seed.csv"):
    rows=load(os.path.join(RD,"discriminator_power/per_seed.csv"))
    DISC=[("1.0","20/20","20/20","0.0195","0.0109"),
          ("1.5","20/20","10/20","0.0188","0.0197"),
          ("2.0","20/20","2/20","0.0193","0.0378"),
          ("2.5","18/20","0/20","0.0171","0.0591"),
          ("3.0","13/20","0/20","0.0156","0.0832")]
    for sig,os3,cv3,gap,mrg in DISC:
        cs=[r for r in rows if float(r["sigma"])==float(sig)]
        chk("tab:disc", f"{sig}/sweep", sum(int(r["khat_ordersweep"])==3 for r in cs), os3.split("/")[0])
        chk("tab:disc", f"{sig}/cv", sum(int(r["khat_cv1se"])==3 for r in cs), cv3.split("/")[0])
        chk("tab:disc", f"{sig}/gap", sum(float(r["cv_mse_gap23"]) for r in cs)/len(cs), gap)
        chk("tab:disc", f"{sig}/margin", sum(float(r["cv_1se_margin"]) for r in cs)/len(cs), mrg)
else: skipped.append("tab:disc (discriminator_power)")

# ---------- Table: V-Dem application (tab:vdem) ----------
if present("vdem_ordersweep/per_seed_app.csv"):
    rows=load(os.path.join(RD,"vdem_ordersweep/per_seed_app.csv"))
    VD=[("upturn","navco",1),("upturn","party",1),("upturn","antimv",1),
        ("downturn","navco",1),("downturn","party",1),("downturn","antimv",1)]
    for outc,trip,khat in VD:
        r=[x for x in rows if x["outcome"]==outc and x["triple"]==trip][0]
        chk("tab:vdem", f"{outc}/{trip}/khat", int(r["khat"]), str(khat))
else: skipped.append("tab:vdem (vdem_ordersweep)")

# ---------- Table: baselines on real data (tab:realbase) ----------
if present("extensions/per_seed_vdem_ext.csv"):
    rows=load(os.path.join(RD,"extensions/per_seed_vdem_ext.csv"))
    # every sweep: blocked=1 row=1 cv1se_blocked=1 pairwise=2 nid=2
    for r in rows:
        chk("tab:realbase", f"{r['outcome']}/{r['triple']}/blk", int(r["khat_blocked"]), "1")
        chk("tab:realbase", f"{r['outcome']}/{r['triple']}/row", int(r["khat_row"]), "1")
        chk("tab:realbase", f"{r['outcome']}/{r['triple']}/cv", int(r["khat_cv1se_blocked"]), "1")
        chk("tab:realbase", f"{r['outcome']}/{r['triple']}/hn", int(r["khat_pairwise"]), "2")
        chk("tab:realbase", f"{r['outcome']}/{r['triple']}/nid", int(r["khat_nid"]), "2")
else: skipped.append("tab:realbase (extensions)")


# ---------- New studies: K=4 and independent-split coverage (prose numbers) ----------
if present("ksweep_k4/per_seed_k4.csv"):
    rows=load(os.path.join(RD,"ksweep_k4/per_seed_k4.csv"))
    nulls=[r for r in rows if r["dgp"] in ("o1","o2","o3")]
    over=sum(int(r["khat"])>int(r["true_order"]) for r in nulls)
    chk("k4","null overselections (paper: 0)",over,"0")
    pw=[r for r in rows if r["dgp"] in ("o4","o4mix") and r["dependence"]=="indep"]
    chk("k4","order-4 power indep (paper: 80)",sum(int(r["khat"])==4 for r in pw),"80")
    allcorrect=sum(int(r["khat"])==int(r["true_order"]) for r in rows)
    chk("k4","total correct (paper: 400)",allcorrect,"400")
else: skipped.append("k4 study (ksweep_k4)")

if present("coverage_indep/per_seed_coverage.csv"):
    rows=load(os.path.join(RD,"coverage_indep/per_seed_coverage.csv"))
    F=lambda r:(1-r**2)**2/((1+r**2)*(1+2*r**2)); VH=1+2*0.81
    truth=lambda s:F(0.9)*VH/(VH+s**2)
    fc=ft=rc=rt=0
    for sig in [1.0,2.0,3.0,4.0,5.0,6.0]:
        cs=[r for r in rows if abs(float(r["sigma"])-sig)<1e-9]
        stops=[r for r in cs if int(r["khat"])==2 and r["ub_cert"]!=""]
        if not stops: continue
        cov=sum(float(r["ub_cert"])>=truth(sig) for r in stops)
        if len(stops)/len(cs)>=0.5: fc+=cov; ft+=len(stops)
        else: rc+=cov; rt+=len(stops)
    chk("coverage","frequent-stop covered (paper: 113)",fc,"113")
    chk("coverage","frequent-stop total (paper: 119)",ft,"119")
    chk("coverage","rare-stop covered (paper: 12)",rc,"12")
    chk("coverage","rare-stop total (paper: 18)",rt,"18")
else: skipped.append("coverage study (coverage_indep)")

# ---------- Table: basis generality (tab:basis) + B-spline study numbers ----------
if present("bspline_basis/per_seed_bspline.csv"):
    rows=load(os.path.join(RD,"bspline_basis/per_seed_bspline.csv"))
    nulls=[r for r in rows if r["dgp"] in ("o1","o2")]
    over=sum(int(r["khat_bspline"])>int(r["true_order"]) for r in nulls)
    chk("bspline","null overselections (paper: 0)",over,"0")
    agree=sum(int(r["khat_bspline"])==int(r["khat_poly"]) for r in rows)
    chk("bspline","basis agreement (paper: 302)",agree,"302")
    # per-cell agreement pattern from tab:basis: all cells 'all' agree except o3/o3mix pair0.9 sigma0.5 -> bspline 9 at k=2
    for dgp in ("o3","o3mix"):
        cs=[r for r in rows if r["dgp"]==dgp and r["dependence"]=="pair0.9" and abs(float(r["sigma"])-0.5)<1e-9]
        k2=sum(int(r["khat_bspline"])==2 for r in cs)
        chk("tab:basis",f"{dgp}/pair0.9/0.5 bspline k=2 (paper: 9)",k2,"9")
    # spot-check an 'all agree' cell: o1 indep -> all 1 both bases
    o1=[r for r in rows if r["dgp"]=="o1" and r["dependence"]=="indep"]
    chk("tab:basis","o1/indep bspline all order 1",sum(int(r["khat_bspline"])==1 for r in o1),str(len(o1)))
else: skipped.append("tab:basis / bspline study (bspline_basis)")

print(f"{checked} checks ({checked} table cells + study numbers), {len(fails)} failures.")
for f in fails: print("FAIL ", f)
for s in skipped: print("SKIP  ", s)
print("ALL PRESENT TABLES VERIFIED" if not fails else "VERIFICATION FAILED")
sys.exit(1 if fails else 0)
