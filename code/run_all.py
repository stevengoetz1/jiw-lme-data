"""Rebuild everything: python code/run_all.py

1. data/analysis/*.csv           (score, review flags, strictness index)
2. thesis results and tables     -> output/tables/thesis/, output/figures/
3. prospectus tables and figure  -> output/tables/prospectus/, output/figures/
Then copies the generated tables into paper/*/tables/ and saves all printed results to output/results_log.txt.
"""
import os, shutil, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for d in ["data/analysis", "output/tables/thesis", "output/tables/prospectus", "output/figures"]:
    os.makedirs(os.path.join(ROOT, d), exist_ok=True)
log = []
for step in ["01_build_score.py", "02_main_analysis.py", "03_prospectus_tables.py"]:
    print(">>", step)
    r = subprocess.run([sys.executable, step], cwd=HERE, capture_output=True, text=True)
    log.append(f"{'=' * 80}\n{step}\n{'=' * 80}\n{r.stdout}")
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-4000:]); sys.exit(f"{step} failed")
with open(os.path.join(ROOT, "output", "results_log.txt"), "w") as f: f.write("\n".join(log))
for paper in ["thesis", "prospectus"]:
    src = os.path.join(ROOT, "output", "tables", paper); dst = os.path.join(ROOT, "paper", paper, "tables")
    for name in sorted(os.listdir(src)):
        if name.endswith(".tex") and (paper == "thesis" or name != "h2.tex"):     # the prospectus keeps its H2 table inline in main.tex
            shutil.copy(os.path.join(src, name), os.path.join(dst, name))
print("done. Results: output/results_log.txt   Tables: output/tables/   Figures: output/figures/")
