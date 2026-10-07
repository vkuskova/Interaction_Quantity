#!/usr/bin/env python3
"""Recompute every number quoted in the author responses from the shipped rebuttal
artifacts (standard library only). Run from the rebuttal/ directory:  python verify_rebuttal.py"""
import csv, os, sys
A = "artifacts"; fails = []; n = 0
def chk(label, cond):
    global n; n += 1
    if not cond: fails.append(label)
# ---- AIC/BIC baseline (response to the information-criteria request) ----
m = list(csv.DictReader(open(os.path.join(A, "aicbic_main.csv"))))
def cnt(flt, pred): xs = [r for r in m if flt(r)]; return sum(pred(r) for r in xs), len(xs)
for col, over5, over0 in [("khat_aic", 15, 131), ("khat_bic", 0, 51), ("khat_sweep", 1, 0)]:
    o5 = cnt(lambda r: r["dgp"] in ("o1", "o2") and float(r["sigma"]) == 0.5, lambda r: int(r[col]) > int(r["true_order"]))
    o0 = cnt(lambda r: r["dgp"] in ("o1", "o2") and float(r["sigma"]) == 0.0, lambda r: int(r[col]) > int(r["true_order"]))
    chk(f"{col} null overselections sigma=0.5 == {over5}/160", o5 == (over5, 160))
    chk(f"{col} null overselections sigma=0 == {over0}/160", o0 == (over0, 160))
o1 = cnt(lambda r: r["dgp"] == "o1" and float(r["sigma"]) == 0.5, lambda r: int(r["khat_aic"]) > 1)
o2 = cnt(lambda r: r["dgp"] == "o2" and float(r["sigma"]) == 0.5, lambda r: int(r["khat_aic"]) > 2)
chk("AIC order-1 nulls 6/80 and order-2 nulls 9/80 at sigma=0.5", o1 == (6, 80) and o2 == (9, 80))
d = list(csv.DictReader(open(os.path.join(A, "aicbic_disc.csv"))))
for sig, a, b in [(1.0, 20, 20), (1.5, 20, 20), (2.0, 20, 20), (2.5, 20, 17), (3.0, 20, 16)]:
    ds = [r for r in d if float(r["sigma"]) == sig]
    chk(f"noise band sigma={sig}: AIC {a}/20, BIC {b}/20",
        sum(int(r["khat_aic"]) == 3 for r in ds) == a and sum(int(r["khat_bic"]) == 3 for r in ds) == b)
# ---- p-value vs coverage (response to the bound-trustworthiness question) ----
cov = list(csv.DictReader(open(os.path.join(A, "per_seed_coverage.csv"))))
F = lambda r: (1-r**2)**2/((1+r**2)*(1+2*r**2)); VH = 1 + 2*0.81
truth = lambda s: F(0.9)*VH/(VH + s**2)
stops = [r for r in cov if int(r["khat"]) == 2 and r["ub_cert"] != ""]
chk("137 stop-at-2 replicates", len(stops) == 137)
bord = [r for r in stops if float(r["rem2_p"]) <= 0.10]; hi = [r for r in stops if float(r["rem2_p"]) > 0.5]
chk("borderline acceptances cover 21/23", (sum(float(r["ub_cert"]) >= truth(float(r["sigma"])) for r in bord), len(bord)) == (21, 23))
chk("acceptances with p>0.5 cover 16/22", (sum(float(r["ub_cert"]) >= truth(float(r["sigma"])) for r in hi), len(hi)) == (16, 22))
# ---- Beijing second application ----
bj = {r["triple"]: r for r in csv.DictReader(open(os.path.join(A, "beijing_app.csv")))}
chk("three triples, 418,297 rows each", len(bj) == 3 and all(int(r["n"]) == 418297 for r in bj.values()))
chk("additivity rejected on all three at p<0.001", all(float(r["rem1_p"]) < 0.001 for r in bj.values()))
chk("first-order gaps within 0.022-0.0322", all(0.022 <= float(r["rem1_mean"]) <= 0.0322 for r in bj.values()))
chk("blocked khat = 3,2,3 (temp_dewp, temp_pres, dewp_pres)",
    [bj[k]["khat_blocked"] for k in ("temp_dewp_wspm", "temp_pres_wspm", "dewp_pres_wspm")] == ["3", "2", "3"])
chk("order-3 gaps 0.0013 (p 0.003) and 0.0003 (p 0.028)",
    round(float(bj["temp_dewp_wspm"]["rem2_mean"]), 4) == 0.0013 and round(float(bj["temp_dewp_wspm"]["rem2_p"]), 3) == 0.003
    and round(float(bj["dewp_pres_wspm"]["rem2_mean"]), 4) == 0.0003 and round(float(bj["dewp_pres_wspm"]["rem2_p"]), 3) == 0.028)
chk("pairwise triple bound 0.0001", round(float(bj["temp_pres_wspm"]["ub_cert"]), 4) == 0.0001)
chk("CV-1SE blocked: additive on two, pairwise on one", sorted(r["khat_cv1se_blocked"] for r in bj.values()) == ["1", "1", "2"])
chk("row-split selects 3 where blocked selected 2", bj["temp_pres_wspm"]["khat_row"] == "3")
# ---- runtime (transcribed) ----
rt = open(os.path.join(A, "runtime.txt")).read()
chk("runtime lines 1.1 / 5.6 / 12.1 / 8.9 s", all(x in rt for x in ("1.1", "5.6", "12.1", "8.9")))
r50 = open(os.path.join(A, "runtime_d50.txt")).read()
chk("d=50 n=10,000: 7,551 features degree-4, 1,326 degree-2, 106 MB, 35.3 s", all(x in r50 for x in ("7,551", "1,326", "106 MB", "35.3s")))
print(f"{n} checks, {len(fails)} failures")
for f in fails: print("FAIL", f)
print("ALL REBUTTAL NUMBERS VERIFIED" if not fails else "VERIFICATION FAILED"); sys.exit(1 if fails else 0)
