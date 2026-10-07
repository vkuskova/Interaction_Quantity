# Self-contained cell: does the margin by which the accepted hypothesis clears alpha predict
# certificate coverage? Uses the recorded per-replicate p-values in results/coverage_indep/.
from google.colab import drive
drive.mount('/content/drive')
import os, csv
import numpy as np
BASE = '/content/drive/MyDrive/ORDER_SWEEP/results'
rows = list(csv.DictReader(open(os.path.join(BASE, 'coverage_indep', 'per_seed_coverage.csv'))))
F = lambda r: (1-r**2)**2/((1+r**2)*(1+2*r**2)); VH = 1 + 2*0.81
truth = lambda s: F(0.9)*VH/(VH + s**2)
# stops at order 2: the accepted hypothesis is H_2, its p-value is rem2_p
stops = [r for r in rows if int(r["khat"]) == 2 and r["ub_cert"] != ""]
print(f"{len(stops)} stop-at-2 replicates with recorded p-values")
# bin by the accepted p-value (margin above alpha = 0.05)
bins = [(0.05, 0.10, "borderline  0.05<p<=0.10"), (0.10, 0.25, "moderate    0.10<p<=0.25"),
        (0.25, 0.50, "clear       0.25<p<=0.50"), (0.50, 1.01, "systematic  p>0.50")]
lines = ["accepted p-value bin         stops   covered   coverage"]
for lo, hi, lab in bins:
    b = [r for r in stops if lo < float(r["rem2_p"]) <= hi]
    if not b: lines.append(f"{lab:28s} 0"); continue
    cov = sum(float(r["ub_cert"]) >= truth(float(r["sigma"])) for r in b)
    lines.append(f"{lab:28s} {len(b):5d}   {cov:7d}   {cov/len(b):.2f}")
for l in lines: print(l)
# also the two-way split the responses would quote
bord = [r for r in stops if float(r["rem2_p"]) <= 0.10]; rest = [r for r in stops if float(r["rem2_p"]) > 0.10]
cb = sum(float(r["ub_cert"]) >= truth(float(r["sigma"])) for r in bord); cr = sum(float(r["ub_cert"]) >= truth(float(r["sigma"])) for r in rest)
summary = (f"borderline acceptances (p<=0.10): {cb}/{len(bord)} covered; "
           f"clear acceptances (p>0.10): {cr}/{len(rest)} covered")
print(summary)
out = os.path.join(BASE, 'rebuttal'); os.makedirs(out, exist_ok=True)
open(os.path.join(out, 'pvalue_coverage.txt'), 'w').write("\n".join(lines) + "\n" + summary + "\n")
print("wrote pvalue_coverage.txt")
