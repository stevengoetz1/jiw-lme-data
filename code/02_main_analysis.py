"""Step 2. Main analysis with the structural score (the thesis's Part 1 and Part 2 results).

Prints every result to the console (run_all.py saves it to output/results_log.txt), saves figures to
output/figures/, and writes the thesis tables to output/tables/thesis/.
This is the same code as "Part 4" of the JIW Database Colab notebook, reading local files.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd, numpy as np
import statsmodels.formula.api as smf
from scipy import stats
from common import *

deals, terms, cnt = load_raw()
sc = build_sample(deals, cnt)
scored = sc[sc.score.notna()].copy()
prs, lms = scored[scored.form == "Pro rata"], scored[scored.form == "LME"]
print(len(sc), "deals in the analysis sample;", len(scored), "with a score:", len(lms), "LME +", len(prs), "pro rata")
_fig = [0]
def savefig(name=None):
    _fig[0] += 1; plt.savefig(P("output", "figures", name or f"analysis_fig{_fig[0]}.png"), dpi=200, bbox_inches="tight"); plt.close()


# ======================================================================
# P4-2
# ======================================================================
# ---- What the score looks like ----
desc = scored.groupby("form").score.agg(N="count", mean="mean", sd="std", median="median").round(2)
shares = pd.crosstab(scored.form, scored.score_bin, normalize="index").round(2); shares.columns = shares.columns.astype(str)
print(desc.join(shares)); print()
print("exactly 0:", (scored.score == 0).sum(), "| exactly 1:", (scored.score == 1).sum(), "| strictly between:", ((scored.score > 0) & (scored.score < 1)).sum())
print("\nMean score by coder #1's original group (should rise A -> C -> B):")
print(scored.groupby(["form", "cause_group"]).score.agg(["mean", "median", "count"]).round(2))
print("\nWhere coder #1's groups land:"); print(pd.crosstab(scored.cause_group, scored.score_bin))

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for a, (lab, d) in zip(ax, (("Pro rata exchanges", prs), ("LMEs", lms))):
    a.hist(d.score, bins=np.linspace(0, 1, 11), color="#2a78d6", edgecolor="white")
    a.set_title(f"{lab} (n = {len(d)})"); a.set_xlabel("Structural score (0 = shock, 1 = structural)"); a.grid(axis="y", alpha=.3)
ax[0].set_ylabel("Number of deals"); plt.tight_layout(); savefig("score_distribution.png")

# ======================================================================
# P4-3
# ======================================================================
# ---- Reliability of a CONTINUOUS score: how closely do the three models' scores agree? ----
cols = [f"score_{m}" for m in MODELS]
R = scored[cols].rename(columns=lambda c: MODELS[c[6:]])
print("Pairwise Pearson correlations of the per-model scores:"); print(R.corr().round(2))
print("\nPairwise Spearman:"); print(R.corr(method="spearman").round(2))

def icc2(X):                                                   # ICC(2,1): two-way random effects, absolute agreement, single rater
    X = np.asarray(X, float); n, k = X.shape; gm = X.mean()
    msr = k * ((X.mean(1) - gm) ** 2).sum() / (n - 1); msc = n * ((X.mean(0) - gm) ** 2).sum() / (k - 1)
    mse = ((X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n), (msr - mse) / msr
i1, ik = icc2(R.values)
print(f"\nICC(2,1), single model: {i1:.2f}   |   ICC(2,k), average of the 3 models (what the score uses): {ik:.2f}")
print("  benchmark (Koo & Li 2016): < 0.5 poor, 0.5-0.75 moderate, 0.75-0.9 good, > 0.9 excellent")
print("\nScore vs coder #1 on the same 0-1 scale: Pearson r =", round(scored.score.corr(scored.coder1_score), 2),
      "| Spearman =", round(scored.score.corr(scored.coder1_score, method="spearman"), 2))
print("Name-leakage check (gpt-4o-mini, named vs name-scrubbed prompt): r =", round(scored.score_gpt4omini.corr(scored.score_masked), 2),
      "| mean abs. change =", round((scored.score_gpt4omini - scored.score_masked).abs().mean(), 2))
print("Main score vs the version that ignores 'mixed' runs: r =", round(scored.score.corr(scored.score_noC), 2), f"({scored.score_noC.notna().sum()} deals)")

# ======================================================================
# P4-4
# ======================================================================
# ---- PART 1 with the score: does failure rise with the structural score? ----
rows = []
for form, d in (("Pro rata", prs), ("LME", lms)):
    for b in scored.score_bin.cat.categories:
        r = {"Form": form, "Score bin": b}
        for h in (1, 2, 3):
            e = d[(d[f"elig_{h}y"] == 1) & (d.score_bin == b)]
            r[f"{h}y fail"] = round(e[f"fail_{h}y"].mean(), 2) if len(e) else np.nan; r[f"{h}y n"] = len(e)
        rows.append(r)
rates_bin = pd.DataFrame(rows).set_index(["Form", "Score bin"])
print("Raw failure rates by score bin:"); print(rates_bin); print()

e2, e3 = prs[prs.elig_2y == 1], prs[prs.elig_3y == 1]
m_s2   = fit("fail_2y ~ score + C(year) + C(industry)", e2)                       # main model
m_s3   = fit("fail_3y ~ score + C(year) + C(industry)", e3)
m_sbk  = fit("bk_2y ~ score + C(year) + C(industry)", e2)
pool2  = scored[scored.elig_2y == 1]
m_spool = fit("fail_2y ~ score + lme + C(year) + C(industry)", pool2)
m_sint  = fit("fail_2y ~ score * lme + C(year) + C(industry)", pool2)
m_slme  = fit("fail_2y ~ score", lms[lms.elig_2y == 1])                            # LMEs alone: too few deals for fixed effects
for name, m in (("Pro rata, fail 2y", m_s2), ("Pro rata, fail 3y", m_s3), ("Pro rata, bankrupt 2y", m_sbk),
                ("Pooled, fail 2y", m_spool), ("LMEs only, fail 2y (no FE)", m_slme)):
    ci = m.conf_int().loc["score"]
    print(f"{name:30s} score = {m.params.score:+.3f}  (se {m.bse.score:.3f}, p = {m.pvalues.score:.3f}, 95% CI {ci[0]:+.2f} to {ci[1]:+.2f}, n = {int(m.nobs)})")
print(f"{'Pooled':30s} LME   = {m_spool.params.lme:+.3f}  (p = {m_spool.pvalues.lme:.3f})")
print(f"{'Pooled, score x LME':30s}       = {m_sint.params['score:lme']:+.3f}  (p = {m_sint.pvalues['score:lme']:.3f})")

# ======================================================================
# P4-5
# ======================================================================
# ---- Chart: two-year failure by score bin, pro rata vs LME ----
g = scored[scored.elig_2y == 1].groupby(["form", "score_bin"]).fail_2y.agg(["mean", "count"]).reset_index()
fig, ax = plt.subplots(figsize=(7.5, 4))
x = np.arange(3); w = 0.38
for k, (form, col) in enumerate((("Pro rata", "#2a78d6"), ("LME", "#e34948"))):
    d = g[g.form == form]
    bars = ax.bar(x + (k - 0.5) * w, d["mean"] * 100, w, label=form, color=col, edgecolor="white")
    for bx, (mu, n) in zip(bars, zip(d["mean"], d["count"])):
        ax.text(bx.get_x() + bx.get_width() / 2, mu * 100 + 2, f"{mu:.0%}\n(n={n})", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(["Low score\n(mostly shock)", "Middle\n(coders split / mixed)", "High score\n(mostly structural)"])
ax.set_ylabel("Failed again within 2 years (%)"); ax.set_ylim(0, 115); ax.legend(frameon=False); ax.grid(axis="y", alpha=.3)
ax.set_title("Two-year failure by structural score"); plt.tight_layout(); savefig("failure_by_score_bin.png")

# ======================================================================
# P4-6
# ======================================================================
# ---- Is the relationship linear?  And does the result depend on how the score is built? ----
m_bins = fit("fail_2y ~ bin_mid + bin_high + C(year) + C(industry)", e2)
m_quad = fit("fail_2y ~ score + I(score ** 2) + C(year) + C(industry)", e2)
print(f"Three bins (vs low):  middle {m_bins.params.bin_mid:+.3f} (p = {m_bins.pvalues.bin_mid:.2f}),  high {m_bins.params.bin_high:+.3f} (p = {m_bins.pvalues.bin_high:.3f})")
print(f"Quadratic term: {m_quad.params['I(score ** 2)']:+.3f} (p = {m_quad.pvalues['I(score ** 2)']:.2f})\n")

def one(label, formula, d, var):
    m = fit(formula, d); return {"Specification": label, "coef": m.params[var], "se": m.bse[var], "p": m.pvalues[var], "n": int(m.nobs)}
FE = " + C(year) + C(industry)"
rob = [one("Main: continuous score", "fail_2y ~ score" + FE, e2, "score"),
       one("Binary, majority wins (score > 0.5)", "fail_2y ~ struct_major" + FE, e2, "struct_major"),
       one("High bin vs low bin (three-bin model)", "fail_2y ~ bin_mid + bin_high" + FE, e2, "bin_high"),
       one("Continuous, middle bin dropped", "fail_2y ~ score" + FE, e2[e2.bin_mid == 0], "score"),
       one("Score ignoring 'mixed' runs", "fail_2y ~ score_noC" + FE, e2[e2.score_noC.notna()], "score_noC"),
       one("Coder #1 on a 0 / 0.5 / 1 scale", "fail_2y ~ coder1_score" + FE, e2, "coder1_score"),
       one("Coder #1 dummies: structural vs shock", "fail_2y ~ structural + mixed" + FE, e2, "structural"),
       one("Score from gpt-4o-mini only", "fail_2y ~ score_gpt4omini" + FE, e2, "score_gpt4omini"),
       one("Score from gpt-4.1-mini only", "fail_2y ~ score_gpt41mini" + FE, e2, "score_gpt41mini"),
       one("Score from gpt-4o only", "fail_2y ~ score_gpt4o" + FE, e2, "score_gpt4o"),
       one("Name-scrubbed score (gpt-4o-mini)", "fail_2y ~ score_masked" + FE, e2, "score_masked"),
       one("First exchange per issuer only", "fail_2y ~ score" + FE, e2[e2.first_for_issuer == 1], "score"),
       one("Dropping coercion-flagged pro rata deals", "fail_2y ~ score" + FE, e2[e2.coercion_flag != 1], "score"),
       one("No fixed effects", "fail_2y ~ score", e2, "score")]
rob = pd.DataFrame(rob).set_index("Specification")
ml = smf.logit("fail_2y ~ score", data=e2).fit(disp=0).get_margeff().summary_frame().loc["score"]   # with year dummies the logit does not converge (sparse years)
rob.loc["Logit, average marginal effect (no fixed effects)"] = [ml["dy/dx"], ml["Std. Err."], ml["Pr(>|z|)"], len(e2)]
rob["n"] = rob["n"].astype(int)
print(rob.round(3).to_string())
tb = pd.crosstab(e2.struct_major, e2.fail_2y)
print("\nBinary version, raw rates:", e2.groupby("struct_major").fail_2y.agg(["mean", "count"]).round(3).to_dict("index"), "| Fisher exact p =", round(stats.fisher_exact(tb)[1], 3))

# ======================================================================
# P4-7
# ======================================================================
# ---- Two confounds my adviser raised: (1) do high-score firms just start out worse?  (2) is it a 2020 effect? ----
b = prs[prs.rating_1y_notch.notna()]
rho, p_rho = stats.spearmanr(b.score, b.rating_1y_notch)
print(f"(1) Pre-deal rating vs score (pro rata, n = {len(b)}): Spearman rho = {rho:+.2f} (p = {p_rho:.2f}); notch 15 = B-, 16 = CCC+")
print(b.groupby("score_bin").rating_1y_notch.agg(["mean", "median", "std", "count"]).round(2))
m_rt = fit("fail_2y ~ score + rating_1y_notch + C(year) + C(industry)", e2[e2.rating_1y_notch.notna()])
print(f"    main model + rating control: score = {m_rt.params.score:+.3f} (p = {m_rt.pvalues.score:.3f}, n = {int(m_rt.nobs)})\n")

print("(2) Mean score by deal year (pro rata):"); print(prs.groupby("year").score.agg(["mean", "count"]).round(2).T); print()
vint = []
for lab, yrs in (("2018-19", [2018, 2019]), ("2020", [2020]), ("2021-22", [2021, 2022]), ("2023", [2023])):
    d = e2[e2.year.isin(yrs)]; m = fit("fail_2y ~ score", d)
    lo, hi = d[d.score_bin.str.startswith("Low")], d[d.score_bin.str.startswith("High")]
    vint.append({"Vintage": lab, "n": len(d), "mean score": d.score.mean(), "low-bin fail": lo.fail_2y.mean(), "n low": len(lo),
                 "high-bin fail": hi.fail_2y.mean(), "n high": len(hi), "score coef": m.params.score, "p": m.pvalues.score})
vint = pd.DataFrame(vint).set_index("Vintage"); print(vint.round(2).to_string())
m_no20 = fit("fail_2y ~ score + C(year) + C(industry)", e2[e2.year != 2020])
print(f"\n    main model dropping 2020 deals: score = {m_no20.params.score:+.3f} (p = {m_no20.pvalues.score:.3f}, n = {int(m_no20.nobs)})")

# ======================================================================
# P4-8
# ======================================================================
# ---- PART 2 with the score: do lenders write tougher terms for higher-score borrowers? ----
key = scored[["company", "deal_date", "deal_id", "score", "score_bin", "struct_major"]]
neg = terms[terms.negotiated == 1].merge(key, on=["company", "deal_date"], how="left")
neg["industry"] = neg["industry"].fillna("Unknown"); neg["year"] = neg.deal_date.dt.year
print(len(neg), "negotiated deals;", neg.score.notna().sum(), "with a score")
pt = neg[(neg.form == "Pro rata") & neg.score.notna()].copy()

# strictness index (Kling-Liebman-Katz), exactly as in Part 2b
pt["strict_klk"] = strictness_index(pt)

TERMS = {"haircut_any": "Haircut", "equity_control": "Control equity", "governance": "Governance rights", "docs_tight": "Docs tightened",
         "nm_priming": "Priming new money", "pik": "PIK coupon", "extension_any": "Maturity extended", "sponsor_support": "Sponsor support"}
out = []; m_terms = {}
for v, lab in list(TERMS.items()) + [("control_index", "Control index (0-5)"), ("runway_index", "Runway index (0-4)"), ("strict_klk", "Strictness index (s.d.)")]:
    d = pt.dropna(subset=[v]); m = fit(f"{v} ~ score + C(year) + C(industry)", d); m_terms[v] = m
    lo, hi = d[d.score_bin.str.startswith("Low")][v], d[d.score_bin.str.startswith("High")][v]
    out.append({"Term": lab, "low-score mean": lo.mean(), "high-score mean": hi.mean(), "score coef": m.params.score, "se": m.bse.score, "p": m.pvalues.score, "n": int(m.nobs)})
print(pd.DataFrame(out).set_index("Term").round(3).to_string())

# "is whatever screening exists enough?": do the terms absorb the score's effect on failure?
ea = pt[(pt.elig_2y == 1)].dropna(subset=["strict_klk"])
ef = ea[ea.relief_type != "other / not stated"].copy(); ef["fix"] = ef.relief_type.str.startswith("balance-sheet").astype(int)
m_a1 = fit("fail_2y ~ fix + C(year)", ef); m_a2 = fit("fail_2y ~ fix + score + C(year)", ef)
m_a3 = fit("fail_2y ~ score + C(year)", ea); m_a4 = fit("fail_2y ~ strict_klk + score + C(year)", ea)
print(f"\nScore effect on 2y failure, no terms:      {m_a3.params.score:+.3f} (p = {m_a3.pvalues.score:.3f}, n = {int(m_a3.nobs)})")
print(f"                      + strictness index:  {m_a4.params.score:+.3f} (p = {m_a4.pvalues.score:.3f});  strictness = {m_a4.params.strict_klk:+.3f} (p = {m_a4.pvalues.strict_klk:.3f})")
print(f"                      + balance-sheet fix: {m_a2.params.score:+.3f} (p = {m_a2.pvalues.score:.3f});  fix = {m_a2.params.fix:+.3f} (p = {m_a2.pvalues.fix:.3f}, n = {int(m_a2.nobs)})")
print("Raw: fail within 2y by relief type:", ef.groupby("fix").fail_2y.agg(["mean", "count"]).round(2).to_dict("index"))

# ======================================================================
# P4-9
# ======================================================================
# ---- Summary statistics for the prospectus (Table: sample characteristics by deal form) ----
def summ(d):
    r = {"Deals": len(d), "Issuers": d.company.nunique(),
         "Structural score, mean": d.score.mean(), "Structural score, s.d.": d.score.std(), "Structural score, median": d.score.median(),
         "Share low score (0-0.33)": d.score_bin.str.startswith("Low").mean(), "Share middle (0.33-0.67)": d.score_bin.str.startswith("Middle").mean(),
         "Share high score (0.67-1)": d.score_bin.str.startswith("High").mean(),
         "Pre-deal rating notch, mean": d.rating_1y_notch.mean(), "Pre-deal rating notch, s.d.": d.rating_1y_notch.std(), "Deals with a pre-deal rating": d.rating_1y_notch.notna().sum(),
         "Deal year, median": d.year.median(), "Share of deals in 2020": (d.year == 2020).mean()}
    for h in (1, 2, 3):
        e = d[d[f"elig_{h}y"] == 1]; r[f"Eligible at {h}y"] = len(e); r[f"Failed within {h}y"] = e[f"fail_{h}y"].mean(); r[f"Bankrupt within {h}y"] = e[f"bk_{h}y"].mean()
    return pd.Series(r)
sumtab = pd.DataFrame({"Pro rata": summ(prs), "LME": summ(lms), "All": summ(scored)})
print(sumtab.round(2).to_string())
print("\nIndustry mix (pro rata):"); print(prs.industry.value_counts(normalize=True).round(2).head(6))
print("\nSD of fail_2y (pro rata, eligible):", round(e2.fail_2y.std(), 2), "| SD of score in that sample:", round(e2.score.std(), 2))
se = m_s2.bse.score; print(f"Minimum detectable score effect at 80% power, 5% two-sided (2.8 x SE): {2.8 * se:.2f}")

# ======================================================================
# P4-10
# ======================================================================
# ---- Export the score-based tables to LaTeX (paste each block into the Overleaf file named in its header) ----
star = lambda p: "^{***}" if p < .01 else "^{**}" if p < .05 else "^{*}" if p < .1 else "^{}"
pct = lambda x: "--" if pd.isna(x) else f"{x:.0%}".replace("%", r"\%")
NOTE = r"$^{*}$p$<$0.1; $^{**}$p$<$0.05; $^{***}$p$<$0.01."

def regtable(models, heads, rows, extra, caption, label, notes, sep=5):
    k = len(models); L = [r"\begin{table}[!htbp] \centering", f"  \\caption{{{caption}}}", r"\begin{adjustbox}{max width=\textwidth}",
         f"\\begin{{tabular}}{{@{{\\extracolsep{{{sep}pt}}}}l{'c' * k}}}", r"\\[-1.8ex]\hline", r"\hline \\[-1.8ex]",
         r"\\[-1.8ex] & " + " & ".join(f"\\multicolumn{{1}}{{c}}{{{h}}}" for h in heads) + r"  \\",
         r"\\[-1.8ex] & " + " & ".join(f"({i + 1})" for i in range(k)) + r" \\", r"\hline \\[-1.8ex]"]
    for var, lab in rows:
        L.append(f" {lab} & " + " & ".join(f"{m.params[var]:.3f}${star(m.pvalues[var])}$" if var in m.params else "" for m in models) + r" \\")
        L.append("& " + " & ".join(f"({m.bse[var]:.3f})" if var in m.params else "" for m in models) + r" \\")
    for lab, vals in extra: L.append(f" {lab} & " + " & ".join(vals) + r" \\")
    L += [r"\hline \\[-1.8ex]", " Observations & " + " & ".join(str(int(m.nobs)) for m in models) + r" \\",
          " $R^2$ & " + " & ".join(f"{m.rsquared:.3f}" for m in models) + r" \\", r"\hline", r"\hline \\[-1.8ex]", r"\end{tabular}", r"\end{adjustbox}",
          f"\\label{{{label}}}", r"\begin{minipage}{0.95\textwidth}\vspace{4pt}\footnotesize", f"\\textit{{Notes.}} {notes} {NOTE}", r"\end{minipage}", r"\end{table}"]
    return "\n".join(L)

TEX = {}
TEX["tables/reg_part1.tex"] = regtable([m_s2, m_s3, m_sbk, m_spool], ["Fail 2y", "Fail 3y", "Bankrupt 2y", "Fail 2y (pooled)"],
    [("score", "Structural score (0--1)"), ("lme", "LME")], [("Sample", ["Pro rata", "Pro rata", "Pro rata", "All"]), ("Year FE", ["Yes"] * 4), ("Industry FE", ["Yes"] * 4)],
    "Structural score and post-exchange failure", "tab:reg_part1",
    "Linear probability models. The structural score is the share of 70 blind re-codings that code the deal structural, counting a mixed code as one half (0 = every coder says shock, 1 = every coder says structural). Standard errors clustered by issuer in parentheses.")

L = [r"\begin{table}[htbp]", r"\centering", r"\caption{Any failure within $h$ years, by deal form and structural score}", r"\label{tab:rates}", r"\begin{tabular}{llrrrrrr}", r"\toprule",
     r" & & \multicolumn{2}{c}{1 year} & \multicolumn{2}{c}{2 years} & \multicolumn{2}{c}{3 years} \\", r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
     r"Form & Structural score & Rate & $n$ & Rate & $n$ & Rate & $n$ \\", r"\midrule"]
for (form, b), r in rates_bin.iterrows():
    L.append(f"{form if b.startswith('Low') else ''} & {b.replace('-', '--')} & " + " & ".join(f"{pct(r[f'{h}y fail'])} & {int(r[f'{h}y n'])}" for h in (1, 2, 3)) + r" \\")
    if form == "Pro rata" and b.startswith("High"): L.append(r"\addlinespace")
L += [r"\bottomrule", r"\end{tabular}", r"\begin{minipage}{0.9\textwidth}\vspace{4pt}\footnotesize",
      r"\textit{Notes.} A deal enters the $h$-year column only if it closed at least $h$ years before the data cutoff. Failure is a Chapter 11 filing or a further distressed exchange.", r"\end{minipage}", r"\end{table}"]
TEX["tables/rates.tex"] = "\n".join(L)

L = [r"\begin{table}[htbp]", r"\centering", r"\caption{The structural score under alternative constructions}", r"\label{tab:robust}", r"\small", r"\begin{adjustbox}{max width=\textwidth}", r"\begin{tabular}{l r r r r}", r"\toprule",
     r"Specification & Coefficient & Std.\ error & $p$ & $n$ \\", r"\midrule"]
for i, (lab, r) in enumerate(rob.iterrows()):
    if i in (1, 5, 7, 11): L.append(r"\addlinespace")
    L.append(f"{lab.replace('#', chr(92) + '#').replace('>', '$>$')} & {r.coef:.3f}${star(r.p)}$ & {r.se:.3f} & {r.p:.3f} & {int(r.n)} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\begin{minipage}{0.95\textwidth}\vspace{4pt}\footnotesize",
      r"\textit{Notes.} Dependent variable: failure within two years of a pro rata exchange. Each row is a separate linear probability model with year and industry fixed effects unless stated, standard errors clustered by issuer; the coefficient shown is on the cause measure named in the row. " + NOTE, r"\end{minipage}", r"\end{table}"]
TEX["tables/robust.tex"] = "\n".join(L)

tv = ["haircut_any", "equity_control", "docs_tight", "nm_priming", "control_index", "strict_klk"]
TEX["tables/reg_terms.tex"] = regtable([m_terms[v] for v in tv], ["Haircut", "Control equity", "Docs tightened", "Priming new money", "Control index", "Strictness index"],
    [("score", "Structural score (0--1)")], [("Year FE", ["Yes"] * 6), ("Industry FE", ["Yes"] * 6)], "Do lenders write different terms by cause of distress?", "tab:reg_terms",
    r"Negotiated pro rata exchanges with a structural score. Cols.\ 1--4 LPMs, cols.\ 5--6 OLS on the index. Standard errors clustered by issuer in parentheses.", sep=2)

TEX["tables/reg_absorb.tex"] = regtable([m_a1, m_a2, m_a3, m_a4], ["Fail 2y"] * 4, [("fix", "Balance-sheet fix"), ("strict_klk", "Strictness index"), ("score", "Structural score (0--1)")],
    [("Year FE", ["Yes"] * 4)], "Do the terms absorb the cause effect? Two-year failure on terms and the structural score", "tab:absorb",
    r"Negotiated pro rata exchanges eligible at two years; cols.\ 1--2 drop deals with relief type not stated. LPMs; SEs clustered by issuer. Balance-sheet fix = haircut or equitisation; strictness in s.d.\ units.")

rowsT = [("haircut_any", "Haircut on exchanged claims"), ("equitized", "Debt converted to equity"), ("equity_control", "Majority equity to creditors"), ("governance", "Board / governance rights to creditors"),
         ("docs_tight", "Covenants or blockers tightened"), ("docs_loose", "Covenants loosened or waived"), ("nm_priming", "Priming new money"), ("new_money_any", "Any new money"),
         ("extension_any", "Maturity extended"), ("pik", "PIK / payment-in-kind coupon"), ("coupon_up", "Coupon increased"), ("sponsor_support", "Sponsor contribution"), ("non_pro_rata_terms", "Unequal treatment of holders")]
lmeT = neg[neg.form == "LME"]; grp = {k: pt[pt.score_bin.str.startswith(k)] for k in ("Low", "Middle", "High")}
cell = lambda d, v, f: "--" if d[v].notna().sum() == 0 else f"{f(d[v].mean())} ({d[v].notna().sum()})"
L = [r"\begin{table}[htbp]\centering", r"\caption{Terms of the exchange by structural score}", r"\label{tab:terms}", r"\small", r"\begin{adjustbox}{max width=\textwidth}", r"\begin{tabular}{l rrr c r}", r"\toprule",
     r" & \multicolumn{3}{c}{Pro rata exchanges, by structural score} & Slope $p$ & LMEs \\", r" \cmidrule(lr){2-4}", r"Term & Low & Middle & High &  &  \\", r"\midrule"]
for v, lab in rowsT:
    d = pt.dropna(subset=[v]); p = fit(f"{v} ~ score", d).pvalues.score
    L.append(f"{lab} & " + " & ".join(cell(grp[k], v, pct) for k in ("Low", "Middle", "High")) + f" & {p:.2f} & {cell(lmeT, v, pct)} \\\\")
L.append(r"\midrule")
for v, lab in (("control_index", "Control index (0--5)"), ("ownership_index", "Ownership index (0--3)"), ("runway_index", "Runway index (0--4)"), ("strict_klk", r"Strictness index (s.d.\ units)")):
    d = pt.dropna(subset=[v]); p = fit(f"{v} ~ score", d).pvalues.score; f2 = lambda x: f"{x:.2f}"
    L.append(f"{lab} & " + " & ".join(cell(grp[k], v, f2) for k in ("Low", "Middle", "High")) + f" & {p:.2f} & {cell(lmeT, v, f2) if v in lmeT else '--'} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\begin{minipage}{0.95\textwidth}\vspace{4pt}\footnotesize",
      f"\\textit{{Notes.}} Negotiated deals only (open-market repurchases excluded): {len(pt)} pro rata exchanges with a structural score ({len(grp['Low'])} low, 0--0.33; {len(grp['Middle'])} middle; {len(grp['High'])} high, 0.67--1) and {len(lmeT)} LMEs. "
      r"Each cell is the share of deals with the term among deals whose release states it; the number stated is in parentheses. Slope $p$ is the $p$-value on the structural score in a bivariate regression of the term on the score, issuer-clustered. The strictness index is the Kling--Liebman--Katz standardised index; higher is tougher on the borrower.", r"\end{minipage}", r"\end{table}"]
TEX["tables/terms.tex"] = "\n".join(L)

f2 = lambda x: f"{x:.2f}"; i0 = lambda x: f"{int(x)}"
spec = [("Deals", i0, "Deals"), (None, None, r"\addlinespace \textit{Structural score (0 = shock, 1 = structural)}"),
        ("Structural score, mean", f2, r"\quad Mean"), ("Structural score, s.d.", f2, r"\quad Standard deviation"), ("Structural score, median", f2, r"\quad Median"),
        ("Share low score (0-0.33)", pct, r"\quad Share low (0--0.33)"), ("Share middle (0.33-0.67)", pct, r"\quad Share middle (0.33--0.67)"), ("Share high score (0.67-1)", pct, r"\quad Share high (0.67--1)"),
        (None, None, r"\addlinespace \textit{S\&P rating one year before the deal (15 = B$-$, 16 = CCC+)}"), ("Pre-deal rating notch, mean", f2, r"\quad Mean notch"),
        ("Pre-deal rating notch, s.d.", f2, r"\quad Standard deviation"), ("Deals with a pre-deal rating", i0, r"\quad Deals with a rating"),
        (None, None, r"\addlinespace \textit{Timing}"), ("Deal year, median", i0, r"\quad Median deal year"), ("Share of deals in 2020", pct, r"\quad Share closed in 2020"),
        (None, None, r"\addlinespace \textit{Outcomes (among deals observed for the full horizon)}")]
for h in (1, 2, 3): spec += [(f"Eligible at {h}y", i0, f"\\quad Deals observable at {h} year{'s' if h > 1 else ''}"), (f"Failed within {h}y", pct, f"\\quad\\quad Failed again within {h} year{'s' if h > 1 else ''}"),
                             (f"Bankrupt within {h}y", pct, f"\\quad\\quad Filed for bankruptcy within {h} year{'s' if h > 1 else ''}")]
L = [r"\begin{table}[htbp]", r"\centering", r"\caption{Summary statistics}", r"\label{tab:sumstats}", r"\small", r"\begin{adjustbox}{max width=\textwidth}", r"\begin{tabular}{l r r r}", r"\toprule", r" & Pro rata exchanges & LMEs & All \\", r"\midrule"]
for key_, f, lab in spec:
    L.append(lab + r" & & & \\" if key_ is None else f"{lab} & " + " & ".join(f(sumtab.loc[key_, c]) for c in ("Pro rata", "LME", "All")) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\begin{minipage}{0.9\textwidth}\vspace{4pt}\footnotesize",
      r"\textit{Notes.} The 275 deals with a structural score (two deals with no pre-deal agency text are excluded). Failure is a Chapter 11 filing or a further distressed exchange. Ratings are available for S\&P-rated issuers only.", r"\end{minipage}", r"\end{table}"]
TEX["tables/sumstats.tex"] = "\n".join(L)

for name, body in TEX.items(): print(f"%%%%%%%%%% {name} %%%%%%%%%%\n{body}\n")

# ---- write the thesis tables ----
import os
for name, body in TEX.items():
    with open(P("output", "tables", "thesis", os.path.basename(name)), "w") as f: f.write(body + "\n")
print("\nwrote", len(TEX), "tables to output/tables/thesis/")
