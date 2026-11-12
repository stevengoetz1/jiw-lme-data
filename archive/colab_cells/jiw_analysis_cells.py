# ==== CELL P1-0 (markdown) ====
## Part 1 analysis (added by Claude, Nov 2026)

These cells reproduce the Part 1 tables from the Excel workbook (`Results` sheet) inside the notebook so they can be tweaked.

**Two things that differ from the preliminary regressions above — worth knowing when comparing numbers:**

1. **Sample.** The prelim cells use all 310 rows. The workbook uses `in_analysis == 1` (277 = 89 LME + 188 pro rata), which drops the 21 "review" rows (RR companies' *other* S&P exchanges) and the 12 excluded S&P rows (non-U.S., mis-tagged).
2. **Outcome window.** `ever_fail` mixes a 2015 deal (11 years to fail) with a 2025 deal (months). The workbook follows Roe & Rotaru and asks "did the deal fail within *h* years" **only among deals old enough to observe h years** (`elig_hy == 1`). That is what `fail_1y/2y/3y` and `elig_1y/2y/3y` are for. This replaces dropping 2024–25 by hand.

Also: the cells below cluster standard errors by **issuer** (`company`) rather than industry — with only 11 industries, cluster-robust SEs are unreliable (Cameron & Miller 2015 suggest ≥ ~40 clusters); same issuer appearing more than once is the correlation we actually need to handle.

# ==== CELL P1-1 (code) ====
# --------------------------------------------------------------------------
# P1-1  Load the data straight from GitHub (no re-upload needed) and set up
#        the analysis sample.  Nothing below touches `lme_deals` / `lme_terms`
#        from the cells above; everything uses `deals` / `samp` / `terms`.
# --------------------------------------------------------------------------
import pandas as pd, numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 40)

BASE = "https://raw.githubusercontent.com/stevengoetz1/jiw-lme-data/main/"
deals = pd.read_csv(BASE + "lme_deals_clean.csv", parse_dates=["deal_date", "bk_date", "fail_date"])
terms = pd.read_csv(BASE + "lme_terms_clean.csv", parse_dates=["deal_date"])

# Analysis sample = 277 deals (89 RR LMEs + 188 S&P pro rata exchanges).
samp = deals[deals.in_analysis == 1].copy()
samp["form"] = np.where(samp.lme == 1, "LME", "Pro rata")   # short label used everywhere below
samp["industry"] = samp["industry"].fillna("Unknown")        # avoid NaN problems in fixed effects

print("rows loaded:", len(deals), "| analysis sample:", len(samp))
print(samp.groupby("form").size().rename("n").to_frame().T)
print(pd.crosstab(samp.form, samp.cause_group, margins=True))

# ==== CELL P1-2 (code) ====
# --------------------------------------------------------------------------
# P1-2  Helper functions used by all the Part 1 tables.
#   rate_table(): share failing within h years, by group, among eligible deals
#   fisher():     two-sided Fisher exact test that two groups' failure rates differ
# --------------------------------------------------------------------------
def rate_table(df, by, h, outcome="fail"):
    """Within-h-year failure rate by `by`, counting only deals with elig_{h}y == 1."""
    e = df[df[f"elig_{h}y"] == 1]
    g = e.groupby(by)[f"{outcome}_{h}y"].agg(["sum", "count", "mean"])
    g.columns = [f"{outcome}_{h}y_n", f"elig_{h}y_n", f"{outcome}_{h}y_rate"]
    return g

def fisher(df, by, a, b, h, outcome="fail"):
    """Fisher exact p-value comparing group a vs b on outcome within h years."""
    e = df[df[f"elig_{h}y"] == 1]
    x = e[e[by] == a][f"{outcome}_{h}y"]; y = e[e[by] == b][f"{outcome}_{h}y"]
    table = [[x.sum(), len(x) - x.sum()], [y.sum(), len(y) - y.sum()]]
    return stats.fisher_exact(table)[1]

def summary_by(df, by, outcome="fail"):
    """One wide table: 1y / 2y / 3y rates side by side (what the Excel 'Results' sheet shows)."""
    out = pd.concat([rate_table(df, by, h, outcome) for h in (1, 2, 3)], axis=1)
    return out.round(3)

# ==== CELL P1-3 (code) ====
# --------------------------------------------------------------------------
# P1-3  TABLE 1 — Headline: LME vs pro rata, any failure (bankruptcy OR another
#        distressed exchange) within 1/2/3 years.  Should match the workbook:
#        LME ≈ 52% / 78% / 86%,  pro rata ≈ 26% / 40% / 51%.
# --------------------------------------------------------------------------
print("ANY FAILURE (bankruptcy or repeat exchange) within h years, by deal form")
display(summary_by(samp, "form", "fail"))
print("\nBANKRUPTCY ONLY within h years, by deal form")
display(summary_by(samp, "form", "bk"))
for h in (1, 2, 3):
    print(f"Fisher p (LME vs pro rata, any failure, {h}y): {fisher(samp, 'form', 'LME', 'Pro rata', h):.4f}")

# ==== CELL P1-4 (code) ====
# --------------------------------------------------------------------------
# P1-4  TABLE 2 — The core Part 1 question: does outcome depend on CAUSE?
#        Rates by cause group (A shock / B structural / C mixed), separately
#        for pro rata exchanges and LMEs, plus A-vs-B Fisher tests.
#        Workbook: pro rata A 28% vs B 52% at 2y (p≈0.02); 3y 42% vs 69% (p≈0.03).
#                  LME A vs B not significant (78% vs 86% at 2y).
# --------------------------------------------------------------------------
for form in ("Pro rata", "LME"):
    sub = samp[samp.form == form]
    print(f"\n===== {form} exchanges: any failure by cause group =====")
    display(summary_by(sub, "cause_group", "fail"))
    print(f"      {form}: bankruptcy only by cause group")
    display(summary_by(sub, "cause_group", "bk"))
    for h in (1, 2, 3):
        print(f"  A vs B Fisher p — any failure {h}y: {fisher(sub,'cause_group','A','B',h):.3f} | "
              f"bankruptcy {h}y: {fisher(sub,'cause_group','A','B',h,'bk'):.3f}")

# ==== CELL P1-5 (code) ====
# --------------------------------------------------------------------------
# P1-5  TABLE 3 — Same thing at the six-code level (A1 demand/operating shock,
#        A2 financial shock, B1 secular industry, B2 firm-specific, B3 legal/
#        regulatory, C mixed).  Small cells — read the n's before the rates.
# --------------------------------------------------------------------------
for form in ("Pro rata", "LME"):
    print(f"\n===== {form}: any failure by six-code cause =====")
    display(summary_by(samp[samp.form == form], "cause_code", "fail"))

# ==== CELL P1-6 (code) ====
# --------------------------------------------------------------------------
# P1-6  TABLE 4 — Does the LME "penalty" (LME minus pro rata failure rate)
#        differ by cause?  This is the interaction the thesis cares about:
#        if LMEs only buy time for shock firms, the gap should be smallest in A.
# --------------------------------------------------------------------------
rows = []
for h in (1, 2, 3):
    for g in ("A", "B", "C"):
        sub = samp[(samp.cause_group == g) & (samp[f"elig_{h}y"] == 1)]
        r = sub.groupby("form")[f"fail_{h}y"].agg(["mean", "count"])
        rows.append(dict(horizon=f"{h}y", cause=g,
                         pro_rata=r.loc["Pro rata", "mean"] if "Pro rata" in r.index else np.nan,
                         n_pro_rata=r.loc["Pro rata", "count"] if "Pro rata" in r.index else 0,
                         lme=r.loc["LME", "mean"] if "LME" in r.index else np.nan,
                         n_lme=r.loc["LME", "count"] if "LME" in r.index else 0,
                         fisher_p=fisher(sub, "form", "LME", "Pro rata", h)))
gap = pd.DataFrame(rows)
gap["lme_penalty_pp"] = (gap.lme - gap.pro_rata) * 100
display(gap.round(3))

# ==== CELL P1-7 (code) ====
# --------------------------------------------------------------------------
# P1-7  TABLE 5 — Balance check: are shock (A) and structural (B) firms the
#        same credit quality going in?  rating_1y_notch is the S&P rating one
#        year before the deal on a numeric scale (higher = worse; 15 = B-).
#        Only the S&P-rated (pro rata) rows have it.  Workbook: median B- in
#        both groups, Mann-Whitney p ≈ 0.65  → cause is not just proxying for
#        a worse starting rating.
# --------------------------------------------------------------------------
pr = samp[(samp.form == "Pro rata") & samp.rating_1y_notch.notna()]
display(pr.groupby("cause_group").rating_1y_notch.describe()[["count", "mean", "50%", "min", "max"]].round(2))
a = pr[pr.cause_group == "A"].rating_1y_notch; b = pr[pr.cause_group == "B"].rating_1y_notch
print("Mann-Whitney U, A vs B rating notch: p =", round(stats.mannwhitneyu(a, b).pvalue, 3))
print("Kruskal-Wallis across A/B/C: p =", round(stats.kruskal(*[g.rating_1y_notch for _, g in pr.groupby("cause_group")]).pvalue, 3))
# Share rated CCC+ or worse one year before (notch >= 17)
print("\nShare rated CCC+ or worse 1y prior, by cause group:")
print((pr.rating_1y_notch >= 17).groupby(pr.cause_group).mean().round(3))

# ==== CELL P1-8 (code) ====
# --------------------------------------------------------------------------
# P1-8  TABLE 6 — Industry split (2-year any-failure, pro rata only) and
#        cause mix by industry.  Energy is mostly A (commodity shock); retail
#        and health care are mostly B (secular).  This is why industry FE
#        matter in the regressions below.
# --------------------------------------------------------------------------
pr = samp[samp.form == "Pro rata"]
print("Cause mix by industry (pro rata):")
display(pd.crosstab(pr.industry, pr.cause_group, margins=True))
print("\n2-year any-failure rate by industry x cause group (pro rata, eligible deals):")
e = pr[pr.elig_2y == 1]
display(e.pivot_table(index="industry", columns="cause_group", values="fail_2y", aggfunc=["mean", "count"]).round(2))

# ==== CELL P1-9 (code) ====
# --------------------------------------------------------------------------
# P1-9  TABLE 7 — Robustness of the A-vs-B gap (pro rata, 2y & 3y any failure):
#   (a) first exchange per issuer only  (drops repeat deals by the same company)
#   (b) drop coercive-looking pro rata deals (coercion_flag == 1)
#   (c) sourced codes only (cause code backed by a cited document)
#   (d) high-confidence codes only
# --------------------------------------------------------------------------
pr = samp[samp.form == "Pro rata"]
checks = {
    "baseline":                 pr,
    "(a) first deal per issuer": pr[pr.first_for_issuer == 1],
    "(b) no coercion flag":     pr[pr.coercion_flag != 1],
    "(c) sourced codes only":   pr[pr.sourced == 1],
    "(d) high confidence only": pr[pr.code_confidence.astype(str).str.lower().str.startswith("h")],
}
rows = []
for name, sub in checks.items():
    for h in (2, 3):
        e = sub[sub[f"elig_{h}y"] == 1]
        rA, rB = e[e.cause_group == "A"][f"fail_{h}y"], e[e.cause_group == "B"][f"fail_{h}y"]
        rows.append(dict(check=name, horizon=f"{h}y", n_A=len(rA), rate_A=rA.mean(), n_B=len(rB), rate_B=rB.mean(),
                         gap_pp=(rB.mean() - rA.mean()) * 100, fisher_p=fisher(sub, "cause_group", "A", "B", h)))
display(pd.DataFrame(rows).round(3))

# ==== CELL P1-10 (code) ====
# --------------------------------------------------------------------------
# P1-10  REGRESSIONS — Linear probability model (OLS on a 0/1 outcome) with
#         year and industry fixed effects, SEs clustered by issuer.
#         Coefficients read as percentage-point differences vs. the omitted
#         group (A = shock).  Logit version is in the next cell.
#
#   Model 1: pro rata only, fail_2y ~ cause            (the Part 1 result)
#   Model 2: pooled,        fail_2y ~ cause + LME      (LME penalty, holding cause fixed)
#   Model 3: pooled,        fail_2y ~ cause * LME      (does the penalty differ by cause?)
#   Each is also run for the 3-year window and for bankruptcy-only.
# --------------------------------------------------------------------------
def lpm(formula, df, h):
    e = df[df[f"elig_{h}y"] == 1]
    return smf.ols(formula, data=e).fit(cov_type="cluster", cov_kwds={"groups": e["company"]})

def show(res, keep=("structural", "mixed", "lme", "structural:lme", "mixed:lme")):
    """Print only the coefficients we care about (FE dummies hidden)."""
    tab = pd.DataFrame({"coef": res.params, "se": res.bse, "p": res.pvalues})
    tab = tab[[any(k == i or i.startswith(k) for k in keep) for i in tab.index]]
    print(tab.round(3).to_string(), f"\n   n = {int(res.nobs)}, R² = {res.rsquared:.3f}")

FE = " + C(year) + C(industry)"
for h in (2, 3):
    for outc in ("fail", "bk"):
        y = f"{outc}_{h}y"
        print(f"\n################  outcome = {y}  ################")
        print("Model 1 (pro rata only): ", y, "~ structural + mixed" + FE)
        show(lpm(f"{y} ~ structural + mixed{FE}", samp[samp.form == "Pro rata"], h))
        print("\nModel 2 (pooled, additive): ", y, "~ structural + mixed + lme" + FE)
        show(lpm(f"{y} ~ structural + mixed + lme{FE}", samp, h))
        print("\nModel 3 (pooled, interaction): ", y, "~ (structural + mixed) * lme" + FE)
        show(lpm(f"{y} ~ (structural + mixed) * lme{FE}", samp, h))

# ==== CELL P1-11 (code) ====
# --------------------------------------------------------------------------
# P1-11  Robustness: LOGIT with average marginal effects (AMEs).
#         AMEs are in the same percentage-point units as the LPM coefficients,
#         so the two tables should tell the same story.  Industry FE are
#         dropped here (small cells cause separation); year FE kept.
# --------------------------------------------------------------------------
def logit_ame(formula, df, h):
    e = df[df[f"elig_{h}y"] == 1]
    res = smf.logit(formula, data=e).fit(disp=0, maxiter=200, cov_type="cluster", cov_kwds={"groups": e["company"]})
    ame = res.get_margeff(at="overall").summary_frame()
    return ame[~ame.index.str.startswith("C(")].round(3)

for h in (2, 3):
    print(f"\n--- Logit AMEs, pro rata only, fail_{h}y ~ structural + mixed + year FE ---")
    display(logit_ame(f"fail_{h}y ~ structural + mixed + C(year)", samp[samp.form == "Pro rata"], h))
    print(f"--- Logit AMEs, pooled, fail_{h}y ~ structural + mixed + lme + year FE ---")
    display(logit_ame(f"fail_{h}y ~ structural + mixed + lme + C(year)", samp, h))

# ==== CELL P1-12 (code) ====
# --------------------------------------------------------------------------
# P1-12  Time-to-failure view: Kaplan–Meier style survival by cause group
#         (pro rata).  Uses days_to_fail; deals that have not failed are
#         censored at the data cutoff (2026-01-14).  No extra packages needed.
# --------------------------------------------------------------------------
import matplotlib.pyplot as plt
CUTOFF = pd.Timestamp("2026-01-14")
pr = samp[samp.form == "Pro rata"].copy()
pr["t"] = np.where(pr.fail_date.notna(), (pr.fail_date - pr.deal_date).dt.days, (CUTOFF - pr.deal_date).dt.days)
pr["event"] = pr.fail_date.notna().astype(int)

def km(t, e):
    t, e = np.asarray(t), np.asarray(e); order = np.argsort(t); t, e = t[order], e[order]
    times, surv, s = [0], [1.0], 1.0
    for ti in np.unique(t[e == 1]):
        at_risk = (t >= ti).sum(); d = ((t == ti) & (e == 1)).sum()
        s *= 1 - d / at_risk; times.append(ti); surv.append(s)
    return np.array(times) / 365.25, np.array(surv)

plt.figure(figsize=(7, 4.5))
for g, lab in (("A", "A: shock"), ("B", "B: structural"), ("C", "C: mixed")):
    sub = pr[pr.cause_group == g]
    x, y = km(sub.t, sub.event)
    plt.step(x, y, where="post", label=f"{lab} (n={len(sub)})")
plt.xlim(0, 4); plt.ylim(0, 1); plt.xlabel("Years after exchange"); plt.ylabel("Share not yet failed")
plt.title("Pro rata exchanges: survival without a second default, by cause of distress"); plt.legend(); plt.grid(alpha=.3)
plt.show()
# Log-rank test (A vs B) via lifelines if available; otherwise skip.
try:
    from lifelines.statistics import logrank_test
    A, B = pr[pr.cause_group == "A"], pr[pr.cause_group == "B"]
    print("log-rank A vs B: p =", round(logrank_test(A.t, B.t, A.event, B.event).p_value, 3))
except ImportError:
    print("(pip install lifelines for a log-rank test)")
# ==== CELL P2-0 (markdown) ====
## Part 2 analysis (added by Claude, Nov 2026)

Question: **do lenders write a different deal depending on *why* the borrower is distressed?** If creditors screen on cause, structural (B) borrowers should get tighter terms — more control transfer, tighter docs, less sponsor-friendly runway — than shock (A) borrowers.

Reproduces the `Part2 Results` sheet of the workbook. Key design choices, so you can defend them:

- **Sample:** `negotiated == 1` drops the 40 open-market buybacks (no lender-imposed terms to study). Most tables are **pro rata only** — the 54 LMEs are shown separately because an LME is by construction a non-pro-rata deal with different mechanics.
- **Blanks = not stated**, so every share below is computed over deals where the release addressed that term (`n` is shown).
- **Indices:** `control_index` (0–5: haircut, control equity, governance, docs tightened, priming lender new-money), `ownership_index` (0–3), `runway_index` (0–4: extension, PIK, docs loosened, sponsor support). `relief_type` splits deals into *balance-sheet fix* (haircut/equitization) vs *runway only* (extension/PIK, no haircut).
- Terms are coded from the S&P/Moody's default releases, not the credit agreements themselves — a document-layer check (EDGAR / Octus) is the natural extension.

# ==== CELL P2-1 (code) ====
# --------------------------------------------------------------------------
# P2-1  Set up the Part 2 sample.  (Run P1-1 first so `terms` exists.)
# --------------------------------------------------------------------------
FLAGS = ["haircut_any", "equitized", "extension_any", "pik", "coupon_up", "new_money_any", "nm_sponsor",
         "nm_lenders", "nm_priming", "equity_any", "equity_control", "governance", "docs_tight", "docs_loose",
         "sponsor_support", "non_pro_rata_terms"]
INDICES = ["control_index", "ownership_index", "runway_index"]

neg = terms[terms.negotiated == 1].copy()          # drop open-market repurchases
neg["industry"] = neg["industry"].fillna("Unknown")
pr2 = neg[neg.form == "Pro rata"].copy()           # main Part 2 sample
lme2 = neg[neg.form == "LME"].copy()

print("terms rows:", len(terms), "| negotiated:", len(neg), "| pro rata negotiated:", len(pr2), "| LME:", len(lme2))
print(pd.crosstab(neg.form, neg.cause_group, margins=True))
print("\nTransaction types (negotiated):"); print(neg.transaction_type.value_counts())

# ==== CELL P2-2 (code) ====
# --------------------------------------------------------------------------
# P2-2  TABLE 11 — Share of deals with each term, by cause group (pro rata,
#        negotiated).  "n" = deals where the release stated that term.
#        Fisher p = A vs B.  Workbook finding: NO cause gradient in control
#        or docs (docs tightened ≈ 1/3 everywhere); C gets control ~2× as often.
# --------------------------------------------------------------------------
def share_table(df, flags, by="cause_group", groups=("A", "B", "C")):
    rows = []
    for f in flags:
        r = {"term": f}
        for g in groups:
            v = df[df[by] == g][f].dropna()
            r[f"{g}_share"] = v.mean() if len(v) else np.nan; r[f"{g}_n"] = len(v)
        a = df[df[by] == "A"][f].dropna(); b = df[df[by] == "B"][f].dropna()
        if len(a) and len(b):
            r["fisher_p_A_vs_B"] = stats.fisher_exact([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]])[1]
        rows.append(r)
    return pd.DataFrame(rows).set_index("term")

t11 = share_table(pr2, FLAGS)
display(t11.round(3))

# ==== CELL P2-3 (code) ====
# --------------------------------------------------------------------------
# P2-3  TABLE 12 — Composite indices by cause group (pro rata, negotiated):
#        mean control / ownership / runway index, Kruskal–Wallis across A/B/C,
#        Mann–Whitney A vs B.  Workbook: ownership 0.40 (A) vs 0.41 (B).
# --------------------------------------------------------------------------
rows = []
for idx in INDICES:
    d = pr2.dropna(subset=[idx])
    g = d.groupby("cause_group")[idx]
    r = {"index": idx}
    for k in ("A", "B", "C"):
        r[f"{k}_mean"] = g.mean().get(k, np.nan); r[f"{k}_n"] = g.size().get(k, 0)
    r["kruskal_p"] = stats.kruskal(*[v for _, v in g]).pvalue
    r["mw_p_A_vs_B"] = stats.mannwhitneyu(d[d.cause_group == "A"][idx], d[d.cause_group == "B"][idx]).pvalue
    rows.append(r)
display(pd.DataFrame(rows).set_index("index").round(3))

print("\nDistribution of control_index by cause group (counts):")
display(pd.crosstab(pr2.cause_group, pr2.control_index))

# ==== CELL P2-4 (code) ====
# --------------------------------------------------------------------------
# P2-4  TABLE 13 — Regressions: does cause predict the terms, controlling for
#        year + industry FE (SEs clustered by issuer)?  Pro rata, negotiated.
#        Outcomes: each index (OLS) and the key binary terms (LPM).
#        A (shock) is the omitted group, so `structural` = B minus A.
# --------------------------------------------------------------------------
def fit_terms(y, df):
    d = df.dropna(subset=[y]).copy()
    d["structural"] = (d.cause_group == "B").astype(int); d["mixed"] = (d.cause_group == "C").astype(int)
    res = smf.ols(f"{y} ~ structural + mixed + C(year) + C(industry)", data=d.assign(year=d.deal_date.dt.year)) \
             .fit(cov_type="cluster", cov_kwds={"groups": d["company"]})
    return dict(outcome=y, n=int(res.nobs), B_minus_A=res.params["structural"], se_B=res.bse["structural"], p_B=res.pvalues["structural"],
                C_minus_A=res.params["mixed"], se_C=res.bse["mixed"], p_C=res.pvalues["mixed"])

import warnings; warnings.filterwarnings("ignore")
t13 = pd.DataFrame([fit_terms(y, pr2) for y in INDICES + ["haircut_any", "equity_control", "docs_tight", "docs_loose", "extension_any", "pik", "nm_priming", "sponsor_support"]])
display(t13.set_index("outcome").round(3))

# ==== CELL P2-5 (code) ====
# --------------------------------------------------------------------------
# P2-5  TABLE 14 — Deal FORM effect: LME vs pro rata terms (negotiated deals),
#        and whether the form effect differs by cause.  Workbook: LMEs have
#        0/52 control transfers, 88% priming new money, 67% haircuts.
# --------------------------------------------------------------------------
print("Term shares by form (negotiated deals):")
byform = share_table(neg, FLAGS, by="form", groups=("LME", "Pro rata")).drop(columns=["fisher_p_A_vs_B"], errors="ignore")
display(byform.round(3))

print("\nIndices by form:")
display(neg.groupby("form")[INDICES].agg(["mean", "count"]).round(2))

print("\nLME terms by cause group:")
display(share_table(lme2, ["haircut_any", "nm_priming", "equity_control", "docs_tight", "extension_any", "pik", "non_pro_rata_terms"]).round(3))

# ==== CELL P2-6 (code) ====
# --------------------------------------------------------------------------
# P2-6  TABLE 15 — Vintage shift: terms in 2018–21 vs 2023–25 (pro rata,
#        negotiated).  Workbook: equity-to-creditors 49% → 25%, PIK 48% → 75%,
#        priming new money 12% → 48%.  (2022 is shown as its own column.)
# --------------------------------------------------------------------------
pr2["vintage"] = pd.cut(pr2.deal_date.dt.year, [2014, 2021, 2022, 2026], labels=["2018-21", "2022", "2023-25"])
rows = []
for f in ["haircut_any", "equitized", "equity_any", "equity_control", "pik", "extension_any", "nm_priming", "docs_tight", "docs_loose", "sponsor_support", "governance"]:
    r = {"term": f}
    for v in ("2018-21", "2022", "2023-25"):
        x = pr2[pr2.vintage == v][f].dropna(); r[v] = x.mean() if len(x) else np.nan; r[f"n_{v}"] = len(x)
    a = pr2[pr2.vintage == "2018-21"][f].dropna(); b = pr2[pr2.vintage == "2023-25"][f].dropna()
    r["fisher_p_early_vs_late"] = stats.fisher_exact([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]])[1]
    rows.append(r)
display(pd.DataFrame(rows).set_index("term").round(3))
print("\nCause mix by vintage (is the vintage shift just a cause shift?):")
display(pd.crosstab(pr2.vintage, pr2.cause_group, normalize="index").round(2))

# ==== CELL P2-7 (code) ====
# --------------------------------------------------------------------------
# P2-7  Terms → outcomes.  Does WHAT the deal did predict whether it held?
#        relief_type: balance-sheet fix (haircut/equitization) vs runway only.
#        Workbook: fix 29% fail within 2y vs runway-only 51%.
#        Then an LPM: fail_2y ~ fix + structural + mixed + year FE — does the
#        cause effect survive controlling for the type of relief?
# --------------------------------------------------------------------------
e = pr2[(pr2.elig_2y == 1) & (pr2.relief_type != "other / not stated")].copy()
print("2-year any-failure by relief type (pro rata, negotiated, eligible):")
display(e.groupby("relief_type").fail_2y.agg(["mean", "count"]).round(3))
print("\n... by relief type x cause group:")
display(e.pivot_table(index="relief_type", columns="cause_group", values="fail_2y", aggfunc=["mean", "count"]).round(2))

e["fix"] = (e.relief_type.str.startswith("balance-sheet")).astype(int)
e["structural"] = (e.cause_group == "B").astype(int); e["mixed"] = (e.cause_group == "C").astype(int)
e["year"] = e.deal_date.dt.year
for spec in ("fail_2y ~ fix", "fail_2y ~ fix + structural + mixed", "fail_2y ~ fix + structural + mixed + C(year)",
             "fail_2y ~ fix * structural + mixed + C(year)"):
    res = smf.ols(spec, data=e).fit(cov_type="cluster", cov_kwds={"groups": e["company"]})
    tab = pd.DataFrame({"coef": res.params, "se": res.bse, "p": res.pvalues})
    print(f"\n{spec}   (n={int(res.nobs)})"); print(tab[~tab.index.str.startswith("C(")].round(3).to_string())

# ==== CELL P2-8 (code) ====
# --------------------------------------------------------------------------
# P2-8  Which individual terms predict a second default?  One LPM per term,
#        fail_2y ~ term + structural + mixed + year FE (pro rata, negotiated,
#        eligible).  Exploratory — many tests, small n, so treat as descriptive.
# --------------------------------------------------------------------------
rows = []
base = pr2[pr2.elig_2y == 1].copy()
base["structural"] = (base.cause_group == "B").astype(int); base["mixed"] = (base.cause_group == "C").astype(int)
base["year"] = base.deal_date.dt.year
for f in ["haircut_any", "equitized", "equity_control", "extension_any", "pik", "coupon_up", "new_money_any", "nm_priming",
          "docs_tight", "docs_loose", "sponsor_support", "governance", "control_index", "runway_index", "ownership_index"]:
    d = base.dropna(subset=[f])
    if d[f].nunique() < 2: continue
    res = smf.ols(f"fail_2y ~ {f} + structural + mixed + C(year)", data=d).fit(cov_type="cluster", cov_kwds={"groups": d["company"]})
    rows.append(dict(term=f, n=int(res.nobs), raw_rate_with=d[d[f] > 0].fail_2y.mean(), raw_rate_without=d[d[f] == 0].fail_2y.mean(),
                     coef=res.params[f], se=res.bse[f], p=res.pvalues[f]))
display(pd.DataFrame(rows).set_index("term").round(3))

# ==== CELL P2-9 (code) ====
# --------------------------------------------------------------------------
# P2-9  One picture for the advisor meeting: term shares by cause group.
# --------------------------------------------------------------------------
import matplotlib.pyplot as plt
plot_terms = ["haircut_any", "equity_control", "governance", "docs_tight", "nm_priming", "extension_any", "pik", "sponsor_support"]
m = t11.loc[plot_terms, ["A_share", "B_share", "C_share"]]
ax = m.plot(kind="bar", figsize=(10, 4.5), width=0.8)
ax.set_ylabel("Share of pro rata deals (where term stated)"); ax.set_xlabel("")
ax.set_title("Exchange terms by cause of distress — pro rata, negotiated deals")
ax.legend(["A: shock", "B: structural", "C: mixed"]); ax.set_ylim(0, 1); plt.xticks(rotation=30, ha="right"); plt.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.show()
