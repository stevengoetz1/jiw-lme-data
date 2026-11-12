"""Step 3. Tables, figure and extra checks used in the prospectus.

Reads data/analysis/*.csv (step 1) and output/tables/thesis/sumstats.tex (step 2).
Writes output/tables/prospectus/{sumstats,reg_part1,robust,h2}.tex, output/figures/fig1_failure_by_cause.png
and output/tables/prospectus/fig1_data.csv (the numbers typed into the pgfplots figure in the paper).
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from common import *

sc, tm = load_scored()
# The prospectus's main measure is the two-label score (share of 70 shock-or-structural codings that say structural).
# The earlier score, from codings that allowed "mixed" (mixed = one half), is kept as score3 for robustness.
sc = sc.rename(columns={"score": "score3", "score_bin": "score3_bin", "struct_major": "struct_major3"}).rename(columns={"score2": "score", "score2_bin": "score_bin", "struct_major2": "struct_major"})
tm = tm.rename(columns={"score": "score3"}).rename(columns={"score2": "score"})
a = sc[sc.score.notna()].copy()
a["major"] = a.struct_major
pr = a[a.pro_rata == 1]
e2, e3, pool2 = pr[pr.elig_2y == 1], pr[pr.elig_3y == 1], a[a.elig_2y == 1]
star = lambda p: "$^{***}$" if p < .01 else "$^{**}$" if p < .05 else "$^{*}$" if p < .1 else ""
num = lambda x, d=3: f"{x:.{d}f}" if x >= 0 else f"$-{abs(x):.{d}f}$"
NOTE = r"$^{*}$p$<$0.1; $^{**}$p$<$0.05; $^{***}$p$<$0.01."
OUT = lambda f: P("output", "tables", "prospectus", f)
def write(name, lines):
    with open(OUT(name), "w") as f: f.write("\n".join(lines) + "\n")

# ------------------------------------------------------------------ Table 2: three versions of the measure
COLS = [("fail_2y", e2, ""), ("fail_3y", e3, ""), ("bk_2y", e2, ""), ("fail_2y", pool2, " + lme")]
PANELS = [("Panel A: structural score (main)", "score", [("score", "Structural score (0--1)")]),
          ("Panel B: binary, majority wins (score above one half)", "major", [("major", "Structural majority")]),
          ("Panel C: original three labels (shock omitted)", "mixed + structural", [("mixed", "Mixed"), ("structural", "Structural")])]
L = [r"\begin{table}[H]", r"\centering", r"\caption{Cause of distress and post-exchange failure, three versions of the measure}", r"\label{tab:reg_part1}", r"\small",
     r"\begin{tabular}{l c c c c}", r"\toprule", r" & Fail 2y & Fail 3y & Bankrupt 2y & Fail 2y (pooled) \\", r" & (1) & (2) & (3) & (4) \\", r"\midrule"]
print("TABLE 2")
for title, rhs, rows in PANELS:
    ms = [fit(f"{y} ~ {rhs}{ex}" + FE, d) for y, d, ex in COLS]
    L.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{title}}}}} \\\\")
    for v, lab in rows + [("lme", "LME")]:
        coef = [num(m.params[v]) + star(m.pvalues[v]) if v in m.params else "" for m in ms]
        se = [f"({m.bse[v]:.3f})" if v in m.params else "" for m in ms]
        L.append(f"\\quad {lab} & " + " & ".join(coef) + r" \\"); L.append(" & " + " & ".join(se) + r" \\")
        print(f"  {title[:7]} {lab:24s}", [f"{m.params[v]:+.3f} (p={m.pvalues[v]:.3f})" if v in m.params else "" for m in ms])
L += [r"\midrule", r"Sample & Pro rata & Pro rata & Pro rata & All \\", "Observations & " + " & ".join(str(len(d)) for _, d, _ in COLS) + r" \\", r"\bottomrule", r"\end{tabular}",
      r"\begin{minipage}{0.95\textwidth}\vspace{4pt}\footnotesize",
      r"\textit{Notes.} Linear probability models; each panel is a separate set of regressions, all with year and industry fixed effects. Standard errors clustered by issuer in parentheses. " + NOTE,
      r"\end{minipage}", r"\end{table}"]
write("reg_part1.tex", L)

# ------------------------------------------------------------------ Table 3: alternative constructions and samples
m_rt_d = e2[e2.rating_1y_notch.notna()]
ROB = [("Main: structural score", fit("fail_2y ~ score" + FE, e2), "score", True),
       ("Earlier score from codings that allowed ``mixed''", fit("fail_2y ~ score3" + FE, e2), "score3", True),
       ("First exchange per issuer only", fit("fail_2y ~ score" + FE, e2[e2.first_for_issuer == 1]), "score", False),
       ("Excluding open-market repurchases", fit("fail_2y ~ score" + FE, e2[e2.repurchase == 0]), "score", False),
       ("Controlling for rating one year before the deal", fit("fail_2y ~ score + rating_1y_notch" + FE, m_rt_d), "score", False)]
L = [r"\begin{table}[H]", r"\centering", r"\caption{The structural score under alternative constructions}", r"\label{tab:robust}", r"\small", r"\begin{adjustbox}{max width=\textwidth}",
     r"\begin{tabular}{l r r r r}", r"\toprule", r"Specification & Coefficient & Std.\ error & $p$ & $n$ \\", r"\midrule"]
print("\nTABLE 3")
for lab, m, v, gap in ROB:
    s = star(m.pvalues[v]) or "$^{}$"
    L.append(f"{lab} & {m.params[v]:.3f}{s} & {m.bse[v]:.3f} & {m.pvalues[v]:.3f} & {int(m.nobs)} \\\\")
    if gap: L.append(r"\addlinespace")
    print(f"  {lab:50s} {m.params[v]:+.3f} se {m.bse[v]:.3f} p {m.pvalues[v]:.3f} n {int(m.nobs)}")
L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\begin{minipage}{0.95\textwidth}\vspace{4pt}\footnotesize",
      r"\textit{Notes.} Dependent variable: failure within two years of a pro rata exchange. Each row is a separate linear probability model with year and industry fixed effects unless stated, standard errors clustered by issuer; the coefficient shown is on the cause measure named in the row. " + NOTE,
      r"\end{minipage}", r"\end{table}"]
write("robust.tex", L)

# ------------------------------------------------------------------ Table 4: H2, terms on the score
pt = tm[(tm.form == "Pro rata") & tm.score.notna()].copy()
H2 = [("Balance-sheet repair", [("haircut_any", "Haircut on principal", r"$\pm$"), ("equitized", "Debt converted to equity", r"$\pm$")]),
      ("Lender control", [("equity_control", "Controlling equity for lenders", "$+$"), ("governance", "Board or governance rights", "$+$"),
                          ("docs_tight", "Covenants tightened", "$+$"), ("nm_priming", "New money ranking ahead", "$+$")]),
      ("Runway", [("extension_any", "Maturity extended", "$-$"), ("pik", "PIK interest", "$-$"), ("runway_only", "Extension or PIK only, no debt cut", "$-$")])]
L = [r"\begin{table}[H]", r"\centering", r"\caption{Exchange terms and the structural score (H2)}", r"\label{tab:h2}", r"\small", r"\begin{tabular}{l c c c c c}", r"\toprule",
     r" & Share & Predicted & Coef.\ on score & $p$ & $n$ \\", r"\midrule"]
print("\nTABLE 4 (H2), negotiated pro rata exchanges:", len(pt))
for grp, rows in H2:
    L.append(f"\\textit{{{grp}}} & & & & & \\\\")
    for v, lab, sign in rows:
        d = pt.dropna(subset=[v]); m = fit(f"{v} ~ score" + FE, d)
        L.append(f"\\quad {lab} & {d[v].mean():.0%} & {sign} & {num(round(m.params.score, 2) + 0.0, 2)} & {m.pvalues.score:.2f} & {int(m.nobs)} \\\\".replace("%", r"\%"))
        print(f"  {lab:36s} share {d[v].mean():.2f} coef {m.params.score:+.3f} se {m.bse.score:.3f} p {m.pvalues.score:.2f} n {int(m.nobs)}")
d = pt.dropna(subset=["strict_klk"]); m_idx = fit("strict_klk ~ score" + FE, d)
L += [r"\addlinespace", f"Strictness index (pre-specified test) & & $+$ & {num(m_idx.params.score, 2)} & {m_idx.pvalues.score:.2f} & {int(m_idx.nobs)} \\\\", r"\bottomrule", r"\end{tabular}",
      r"\begin{minipage}{0.92\textwidth}\vspace{4pt}\footnotesize",
      f"\\textit{{Notes.}} Negotiated pro rata exchanges. Each row is a separate linear probability model of the term on the structural score with year and industry fixed effects and standard errors clustered by issuer. $n$ varies because releases do not state every term. The strictness index has a standard deviation of {d.strict_klk.std():.2f}.",
      r"\end{minipage}", r"\end{table}"]
write("h2.tex", L)
print(f"  strictness index: coef {m_idx.params.score:+.3f} p {m_idx.pvalues.score:.2f}; sd {d.strict_klk.std():.2f}; smallest detectable shift (80% power) {2.8 * m_idx.bse.score:.2f}")
rt = pt[pt.elig_2y == 1].dropna(subset=["fix"]); rt = rt[rt.relief_type.str.startswith(("balance", "runway"))]
print("  two-year failure by relief type:", rt.groupby("fix").fail_2y.mean().round(2).to_dict(), "(1 = debt cut)")

# ------------------------------------------------------------------ Table 1: trimmed summary statistics
drop = (r"\quad Median &", r"\quad Median deal year", r"\quad Deals observable at 1 year", r"\quad\quad Failed again within 1 year",
        r"\quad\quad Filed for bankruptcy within 1 year", r"\quad\quad Filed for bankruptcy within 3 years")
src = open(P("output", "tables", "thesis", "sumstats.tex")).read().rstrip("\n").split("\n")
src = [l.replace(r"\begin{table}[htbp]", r"\begin{table}[H]") for l in src if not l.startswith(drop)]
grp = [a[a.pro_rata == 1], a[a.lme == 1], a]
pc = lambda f: " & ".join(f"{f(g):.0%}".replace("%", r"\%") for g in grp)
extra = [r"\quad Share structural, binary (score above one half) & " + pc(lambda g: g.major.mean()) + r" \\",
         r"\addlinespace \textit{Original three labels} & & & \\"] + \
        [f"\\quad Share {lab} & " + pc(lambda g, c=c: (g.cause_group == c).mean()) + r" \\" for c, lab in [("A", "shock"), ("C", "mixed"), ("B", "structural")]]
k0 = [i for i, l in enumerate(src) if "Structural score (0 = shock" in l][0]
fm = lambda f, fmt: " & ".join(fmt(f(g)) for g in grp)
src[k0 + 1:k0 + 6] = [r"\quad Mean & " + fm(lambda g: g.score.mean(), "{:.2f}".format) + r" \\", r"\quad Standard deviation & " + fm(lambda g: g.score.std(), "{:.2f}".format) + r" \\"] + \
    [f"\\quad Share {lab} & " + pc(lambda g, b=b: (g.score_bin == b).mean()) + r" \\" for b, lab in zip(BINS, ["low (0--0.33)", "middle (0.33--0.67)", "high (0.67--1)"])]
k = [i for i, l in enumerate(src) if l.startswith(r"\quad Share high")][0]
write("sumstats.tex", src[:k + 1] + extra + src[k + 1:])

# ------------------------------------------------------------------ Figure 1: failure by cause, three versions
a["label3"] = a.cause_group.map({"A": "Shock", "C": "Mixed", "B": "Structural"})
a["major_lab"] = a.major.map({0.0: "Shock", 1.0: "Structural"})
a["bin_lab"] = a.score_bin.astype(str).str.split(" ").str[0]
PAN = [("(a) Structural score, thirds", "bin_lab", ["Low", "Middle", "High"]), ("(b) Binary, majority wins", "major_lab", ["Shock", "Structural"]),
       ("(c) Original three labels", "label3", ["Shock", "Mixed", "Structural"])]
ee = a[a.elig_2y == 1]; rows = []
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True, gridspec_kw={"width_ratios": [3, 2, 3]})
for ax, (title, col, order) in zip(axes, PAN):
    for j, (form, colr) in enumerate([("Pro rata", "#3b4a8c"), ("LME", "#b5545c")]):
        g = ee[ee.form == form].groupby(col).fail_2y.agg(["mean", "size"]).reindex(order)
        x = np.arange(len(order)) + (j - 0.5) * 0.36
        ax.bar(x, 100 * g["mean"], 0.34, color=colr, label="Pro rata exchanges" if form == "Pro rata" else "LMEs")
        for xi, (mu, n) in zip(x, g.values): ax.text(xi, 100 * mu + 2, f"{100 * mu:.0f}", ha="center", fontsize=8)
        rows += [{"panel": title, "group": o, "form": form, "fail_2y_pct": round(100 * mu), "n": int(n)} for o, (mu, n) in zip(order, g.values)]
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order); ax.set_xlabel(title); ax.set_ylim(0, 112); ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Failed within 2 years (%)"); axes[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2)
plt.savefig(P("output", "figures", "fig1_failure_by_cause.png"), dpi=200, bbox_inches="tight"); plt.close()
fd = pd.DataFrame(rows); fd.to_csv(OUT("fig1_data.csv"), index=False)
print("\nFIGURE 1 data"); print(fd.to_string(index=False))

# ------------------------------------------------------------------ Figure: every deal on the 0-1 scale, by deal date
a["t"] = a.deal_date.dt.year + (a.deal_date.dt.dayofyear - 1) / 365
LAB = [("J. Crew", 2017, 90, "J. Crew"), ("Chesapeake", 2019, 250, "Chesapeake"), ("Serta", 2020, 150, "Serta"), ("Steak n Shake", 2020, 200, "Steak n Shake"),
       ("Travelport", 2020, 290, "Travelport"), ("Bed Bath", 2022, 90, "Bed Bath \\& Beyond"), ("Shutterfly", 2023, 180, "Shutterfly"), ("WeWork", 2023, 270, "WeWork"),
       ("Lumen", 2024, 35, "Lumen"), ("AMC Entertainment", 2024, 270, "AMC")]        # (company, year, pin angle, text): up to ten well-known deals
co = lambda d: " ".join(f"({t:.3f},{y:.3f})" for t, y in zip(d.t, d.score))
ym = a.groupby("year").score.agg(["mean", "size"]); ym = ym[ym["size"] >= 15]
F = [r"\begin{figure}[H]", r"\centering", r"\begin{tikzpicture}",
     r"\begin{axis}[width=0.98\textwidth, height=6.2cm, xmin=2013.6, xmax=2026.2, ymin=-0.27, ymax=1.2,",
     r"  xtick={2014,2016,...,2026}, minor xtick={2015,2017,...,2025}, x tick label style={/pgf/number format/1000 sep=}, ytick={0,0.25,0.5,0.75,1},",
     r"  xlabel={Deal date}, ylabel={Structural score}, label style={font=\footnotesize}, ticklabel style={font=\footnotesize},",
     r"  ymajorgrids, grid style={gray!20}, axis lines*=left, clip=false,",
     r"  legend style={at={(0.5,1.02)}, anchor=south, draw=none, font=\footnotesize, legend columns=3, /tikz/every even column/.append style={column sep=8pt}}]",
     r"\addplot[only marks, mark=*, mark size=1.5pt, color=blue!55!black, opacity=0.45] coordinates {" + co(a[a.pro_rata == 1]) + "};",
     r"\addplot[only marks, mark=triangle*, mark size=2pt, color=red!65!black, opacity=0.6] coordinates {" + co(a[a.lme == 1]) + "};",
     r"\addplot[black, thick, mark=diamond*, mark size=2pt] coordinates {" + " ".join(f"({y + 0.5:.1f},{m:.3f})" for y, m in zip(ym.index, ym["mean"])) + "};",
     r"\legend{Pro rata exchanges, LMEs, Yearly average}"]
for name, yr, ang, text in LAB:
    r = a[a.company.str.contains(name, regex=False) & (a.year == yr)].iloc[0]
    F.append(f"\\node[pin={{[pin edge={{gray, thin}}, pin distance=11pt, font=\\scriptsize, inner sep=1pt, fill=white, fill opacity=0.85, text opacity=1]{ang}:{text}}}, inner sep=0pt] at (axis cs:{r.t:.3f},{r.score:.3f}) {{}};")
F += [r"\end{axis}", r"\end{tikzpicture}", r"\caption{Where each deal falls on the structural score, by deal date}", r"\label{fig:spectrum}",
      r"\begin{minipage}{0.95\textwidth}\vspace{2pt}\footnotesize",
      f"\\textit{{Notes.}} Each point is one of the {len(a)} scored deals. 0 means every one of the 70 codings says shock and 1 means every coding says structural. The yearly average is shown for years with at least 15 deals.",
      r"\end{minipage}", r"\end{figure}"]
write("fig_spectrum.tex", F)
print("\nSPECTRUM FIGURE: yearly mean score"); print(ym.round(2).T.to_string())

# ------------------------------------------------------------------ extra checks quoted in the prospectus text
print("\nEXTRA CHECKS (pro rata, two-year failure, year + industry fixed effects unless stated)")
def show(lab, m, v="score"): print(f"  {lab:52s} {m.params[v]:+.3f} se {m.bse[v]:.3f} p {m.pvalues[v]:.3f} n {int(m.nobs)}")
x = e2.assign(rex_2y=((e2.fail_2y == 1) & (e2.bk_2y == 0)).astype(int))
show("bankruptcy within 2y", fit("bk_2y ~ score" + FE, x)); show("repeat exchange, no bankruptcy, within 2y", fit("rex_2y ~ score" + FE, x))
show("excluding 2020", fit("fail_2y ~ score" + FE, e2[e2.year != 2020]))
for lab, ys in [("2018-19", [2018, 2019]), ("2020", [2020]), ("2021-22", [2021, 2022]), ("2023", [2023])]:
    show(f"vintage {lab} (no fixed effects)", fit("fail_2y ~ score", e2[e2.year.isin(ys)]))
show("three-label coding, gpt-4o-mini with names", fit("fail_2y ~ score_gpt4omini" + FE, e2), "score_gpt4omini")
show("three-label coding, gpt-4o-mini, names removed", fit("fail_2y ~ score_masked" + FE, e2), "score_masked")
show("binary majority of the main score", fit("fail_2y ~ major" + FE, e2), "major"); print("    raw rates by majority:", e2.groupby("major").fail_2y.agg(["mean", "size"]).round(3).to_dict("index"))
mb = fit("fail_2y ~ C(score_bin)" + FE, e2); print("    thirds vs low:", {k[-12:]: (round(v, 3), round(mb.pvalues[k], 2)) for k, v in mb.params.items() if "score_bin" in k})
mq = fit("fail_2y ~ score + I(score**2)" + FE, e2); print("    quadratic term p:", round(mq.pvalues["I(score ** 2)"], 2))
show("fail 3y", fit("fail_3y ~ score" + FE, e3)); mm = fit("fail_2y ~ score" + FE, e2); print("    main 95% CI:", mm.conf_int().loc["score"].round(3).tolist())
show("with rating control", fit("fail_2y ~ score + rating_1y_notch" + FE, m_rt_d))
from scipy import stats as _st
_r = pr.dropna(subset=["rating_1y_notch"]); print("    rating vs score Spearman:", [round(v, 3) for v in _st.spearmanr(_r.score, _r.rating_1y_notch)], "n", len(_r))
print("    pairwise correlations between the three models' scores:", a[[f"score2_{m}" for m in MODELS]].corr().round(2).values[np.triu_indices(3, 1)].tolist())
_X = a[[f"score2_{m}" for m in MODELS]].dropna().values; _n, _k = _X.shape; _gm = _X.mean()
_msr = _k * ((_X.mean(1) - _gm) ** 2).sum() / (_n - 1); _msc = _n * ((_X.mean(0) - _gm) ** 2).sum() / (_k - 1); _mse = ((_X - _X.mean(1, keepdims=True) - _X.mean(0, keepdims=True) + _gm) ** 2).sum() / ((_n - 1) * (_k - 1))
print("    ICC(2,1):", round((_msr - _mse) / (_msr + (_k - 1) * _mse + _k * (_msc - _mse) / _n), 2), " ICC(2,k):", round((_msr - _mse) / (_msr + (_msc - _mse) / _n), 2))
print("    main score vs earlier mixed-allowed score: r =", round(a[["score", "score3"]].corr().iloc[0, 1], 2), "| deals at exactly 0:", int((a.score == 0).sum()), "at exactly 1:", int((a.score == 1).sum()), "| each model unanimous:", [round(a[f"score2_{m}"].isin([0, 1]).mean(), 2) for m in MODELS])
print("    2023 raw rates by majority:", e2[e2.year == 2023].groupby("major").fail_2y.agg(["mean", "size"]).round(2).to_dict("index"), "| 2020:", e2[e2.year == 2020].groupby("major").fail_2y.agg(["mean", "size"]).round(2).to_dict("index"))
_ab = pt[pt.elig_2y == 1].dropna(subset=["strict_klk"]); show("H2 subsample: score alone", fit("fail_2y ~ score" + FE, _ab)); show("H2 subsample: score with strictness index", fit("fail_2y ~ score + strict_klk" + FE, _ab))
m = fit("fail_2y ~ score + lme" + FE, pool2); show("pooled: LME coefficient", m, "lme")
m = fit("fail_2y ~ score + lme" + FE, pool2[(pool2.lme == 0) | (pool2.sp_listed == 1)]); show("pooled, only LMEs in S&P tables: LME coefficient", m, "lme")
m = fit("fail_2y ~ score * lme" + FE, pool2); show("pooled: score x LME interaction", m, "score:lme")
r = fit("score ~ C(year) + C(industry)", e2); print(f"  share of score variation explained by year + industry fixed effects: {r.rsquared:.2f}")
m = fit("fail_2y ~ score" + FE, e2); print(f"  smallest detectable effect at 80% power, main model: {2.8 * m.bse.score:.2f}")
print(f"  score vs rating one year before (all pro rata deals with a rating): Spearman rho = {pr[['score', 'rating_1y_notch']].corr(method='spearman').iloc[0, 1]:.2f}")
print(f"  repurchases in the two-year pro rata sample: {int(e2.repurchase.sum())}")
