# ==== CELL S1-0 (markdown) ====
## Part 1: Does the outcome of an exchange depend on the cause of distress?

Three choices that differ from the preliminary regressions above:
1. **Sample** is `in_analysis == 1` (277 deals = 89 LMEs + 188 pro rata exchanges). This drops the 21 "review" rows and the 12 excluded S&P rows.
2. **Outcome** is "failed within h years" (`fail_2y`, `fail_3y`) measured only on deals old enough to observe h years (`elig_2y`, `elig_3y`). This replaces `ever_fail` + dropping 2024–25 by hand.
3. **Standard errors** are clustered by company (same issuer can appear more than once), not by industry — 11 industries is too few clusters.

# ==== CELL S1-1 (code) ====
# Load both files straight from GitHub (no upload needed) and build the analysis sample
import pandas as pd, numpy as np
import statsmodels.formula.api as smf
from scipy import stats
import warnings; warnings.filterwarnings("ignore")

BASE = "https://raw.githubusercontent.com/stevengoetz1/jiw-lme-data/main/"
deals = pd.read_csv(BASE + "lme_deals_clean.csv", parse_dates=["deal_date", "bk_date", "fail_date"])
terms = pd.read_csv(BASE + "lme_terms_clean.csv", parse_dates=["deal_date"])

samp = deals[deals.in_analysis == 1].copy()                 # 277 deals
samp["form"] = np.where(samp.lme == 1, "LME", "Pro rata")   # readable label
samp["industry"] = samp["industry"].fillna("Unknown")

pr = samp[samp.form == "Pro rata"]                          # 188 S&P pro rata exchanges
lme = samp[samp.form == "LME"]                              # 89 RR LMEs
print(len(samp), "deals:", len(lme), "LME +", len(pr), "pro rata")

# ==== CELL S1-2 (code) ====
# Failure rate (bankruptcy OR another distressed exchange) within 1, 2 and 3 years, LME vs pro rata
for h in (1, 2, 3):
    e = samp[samp[f"elig_{h}y"] == 1]                        # only deals old enough to observe h years
    print(f"\n--- within {h} year(s) ---")
    print(e.groupby("form")[f"fail_{h}y"].agg(["mean", "count"]).round(3))

# ==== CELL S1-2 (markdown) ====
LMEs fail far more often than pro rata exchanges at every horizon (78% vs 40% within two years), which is the Roe & Rotaru result reproduced on the full sample.

# ==== CELL S1-3 (code) ====
# Pro rata exchanges: failure rate by cause of distress (A = shock, B = structural, C = mixed)
for h in (2, 3):
    e = pr[pr[f"elig_{h}y"] == 1]
    print(f"\n--- pro rata, within {h} years ---")
    print(e.groupby("cause_group")[f"fail_{h}y"].agg(["mean", "count"]).round(3))
    # Fisher exact test: is A's failure rate different from B's?
    ab = e[e.cause_group.isin(["A", "B"])]
    p = stats.fisher_exact(pd.crosstab(ab.cause_group, ab[f"fail_{h}y"]))[1]
    print(f"Fisher exact p-value, A vs B: {p:.3f}")

# ==== CELL S1-3 (markdown) ====
Among pro rata exchanges, structural (B) firms fail about twice as often as shock (A) firms — 52% vs 28% within two years (p = 0.02) and 69% vs 42% within three (p = 0.03) — so cause of distress predicts whether the exchange holds.

# ==== CELL S1-4 (code) ====
# Same thing for LMEs
e = lme[lme.elig_2y == 1]
print(e.groupby("cause_group").fail_2y.agg(["mean", "count"]).round(3))
ab = e[e.cause_group.isin(["A", "B"])]
print("Fisher exact p-value, A vs B:", round(stats.fisher_exact(pd.crosstab(ab.cause_group, ab.fail_2y))[1], 3))

# ==== CELL S1-4 (markdown) ====
For LMEs the gap is small and not significant (78% vs 86%), so an LME looks like it fails regardless of why the firm got into trouble — though only 45 LMEs are old enough to measure, so this is a weak test.

# ==== CELL S1-5 (code) ====
# Main regression: pro rata, fail within 2 years ~ cause, with year and industry fixed effects, SEs clustered by company
# A (shock) is the omitted group, so `structural` = B minus A and `mixed` = C minus A, in percentage points
e = pr[pr.elig_2y == 1]
m_pr2 = smf.ols("fail_2y ~ structural + mixed + C(year) + C(industry)", data=e) \
           .fit(cov_type="cluster", cov_kwds={"groups": e["company"]})
print(m_pr2.summary())

# ==== CELL S1-5 (markdown) ====
With year and industry fixed effects, a structural firm is 36 percentage points more likely than a shock firm to fail within two years of a pro rata exchange (p = 0.001); mixed firms are not significantly different from shock firms.

# ==== CELL S1-6 (code) ====
# Same regression at the 3-year horizon
e = pr[pr.elig_3y == 1]
m_pr3 = smf.ols("fail_3y ~ structural + mixed + C(year) + C(industry)", data=e) \
           .fit(cov_type="cluster", cov_kwds={"groups": e["company"]})
print(m_pr3.summary())

# ==== CELL S1-6 (markdown) ====
The structural penalty is just as large at three years (+39 points, p = 0.02), so it is not an artifact of the horizon chosen.

# ==== CELL S1-7 (code) ====
# Same regression with BANKRUPTCY only as the outcome (ignores repeat exchanges)
e = pr[pr.elig_2y == 1]
m_bk2 = smf.ols("bk_2y ~ structural + mixed + C(year) + C(industry)", data=e) \
           .fit(cov_type="cluster", cov_kwds={"groups": e["company"]})
print(m_bk2.summary())

# ==== CELL S1-7 (markdown) ====
Restricting the outcome to Chapter 11 filings, structural firms are still 21 points more likely to go bankrupt within two years (p = 0.01), though the raw-rate Fisher test on this smaller outcome is not significant, so the bankruptcy-only result is more fragile.

# ==== CELL S1-8 (code) ====
# Pooled regression: LMEs and pro rata together, adding an LME dummy
e = samp[samp.elig_2y == 1]
m_pool = smf.ols("fail_2y ~ structural + mixed + lme + C(year) + C(industry)", data=e) \
            .fit(cov_type="cluster", cov_kwds={"groups": e["company"]})
print(m_pool.summary())

# ==== CELL S1-8 (markdown) ====
Holding cause, year and industry fixed, doing an LME rather than a pro rata exchange raises the two-year failure probability by about 27 points (p = 0.03), and the structural penalty (+24 points) survives alongside it — so form and cause are separate predictors.

# ==== CELL S1-9 (code) ====
# Interaction: does the LME penalty differ by cause?  (structural:lme = extra LME effect for structural firms)
m_int = smf.ols("fail_2y ~ (structural + mixed) * lme + C(year) + C(industry)", data=e) \
           .fit(cov_type="cluster", cov_kwds={"groups": e["company"]})
print(m_int.summary())

# ==== CELL S1-9 (markdown) ====
The interaction terms are negative but not significant (structural×LME = −0.28, p = 0.18), meaning the LME penalty is, if anything, largest for shock firms — the opposite of "LMEs are harmless when the problem is temporary" — but the sample is too small to say so with confidence.

# ==== CELL S1-10 (code) ====
# Balance check: were shock and structural firms the same credit quality a year before the deal?
# rating_1y_notch: S&P rating one year prior on a numeric scale (15 = B-, 16 = CCC+; higher = worse). S&P rows only.
r = pr[pr.rating_1y_notch.notna()]
print(r.groupby("cause_group").rating_1y_notch.describe()[["count", "mean", "50%"]].round(2))
A, B = r[r.cause_group == "A"].rating_1y_notch, r[r.cause_group == "B"].rating_1y_notch
print("Mann-Whitney p-value, A vs B:", round(stats.mannwhitneyu(A, B).pvalue, 3))

# ==== CELL S1-10 (markdown) ====
Shock and structural firms had the same median rating (B-) a year before their exchange and the distributions are statistically indistinguishable (p = 0.78), so the cause effect is not just picking up a worse starting rating.

# ==== CELL S1-11 (code) ====
# Robustness: keep only each company's FIRST exchange (drops repeat deals by the same issuer)
e = pr[(pr.elig_2y == 1) & (pr.first_for_issuer == 1)]
print(e.groupby("cause_group").fail_2y.agg(["mean", "count"]).round(3))
m_first = smf.ols("fail_2y ~ structural + mixed + C(year) + C(industry)", data=e) \
             .fit(cov_type="cluster", cov_kwds={"groups": e["company"]})
print(m_first.summary().tables[1])

# ==== CELL S1-11 (markdown) ====
Using only each issuer's first exchange, the gap is actually wider (29% vs 57%), so repeat deals by the same companies are not driving the result.

# ==== CELL S1-12 (code) ====
# Robustness: logit instead of the linear probability model.  Average marginal effects are in the same
# percentage-point units as the OLS coefficients.  (Industry FE dropped: small cells cause separation.)
e = pr[pr.elig_2y == 1]
m_logit = smf.logit("fail_2y ~ structural + mixed + C(year)", data=e).fit(disp=0)
print(m_logit.get_margeff().summary())

# ==== CELL S1-12 (markdown) ====
The logit gives a structural effect of +24 points (p = 0.01), in line with the linear model, so the choice of LPM vs logit does not change the conclusion.

# ==== CELL S2-0 (markdown) ====
## Part 2: Do lenders write different terms depending on the cause of distress?

If creditors screen on cause, structural (B) borrowers should get tougher terms — more control transfer, tighter documents — than shock (A) borrowers. Sample: `negotiated == 1` (drops the 40 open-market buybacks, which have no lender-imposed terms). Blank flags mean the release did not mention that term, so shares are computed over deals where it was stated.

# ==== CELL S2-1 (code) ====
# Share of pro rata deals with each term, by cause group
neg = terms[terms.negotiated == 1].copy()
neg["industry"] = neg["industry"].fillna("Unknown")
neg["year"] = neg.deal_date.dt.year
neg["structural"] = (neg.cause_group == "B").astype(int)
neg["mixed"] = (neg.cause_group == "C").astype(int)
pr2 = neg[neg.form == "Pro rata"]
print(len(neg), "negotiated deals;", len(pr2), "pro rata")

TERMS = ["haircut_any", "equity_control", "governance", "docs_tight", "docs_loose", "nm_priming", "extension_any", "pik", "sponsor_support"]
print(pr2.groupby("cause_group")[TERMS].mean().T.round(2))

# ==== CELL S2-1 (markdown) ====
There is no cause gradient in the control terms: structural firms are no more likely than shock firms to take a haircut (44% vs 39%), hand over control (10% vs 10%) or get tighter documents (33% vs 32%) — lenders are not visibly screening on cause.

# ==== CELL S2-2 (code) ====
# Composite indices by cause group, with a Kruskal-Wallis test across A/B/C
for idx in ["control_index", "ownership_index", "runway_index"]:
    d = pr2.dropna(subset=[idx])
    print(f"\n{idx}:"); print(d.groupby("cause_group")[idx].agg(["mean", "count"]).round(2))
    print("Kruskal-Wallis p:", round(stats.kruskal(*[g[idx] for _, g in d.groupby("cause_group")]).pvalue, 3))

# ==== CELL S2-2 (markdown) ====
The control index (0–5) averages 0.93 for shock firms and 1.07 for structural firms, a difference that is not significant (p = 0.29); the only group that stands out is mixed (C), which gets slightly more control transfer.

# ==== CELL S2-3 (code) ====
# Regression version: control_index ~ cause, with year and industry fixed effects, SEs clustered by company
m_ctrl = smf.ols("control_index ~ structural + mixed + C(year) + C(industry)", data=pr2) \
            .fit(cov_type="cluster", cov_kwds={"groups": pr2["company"]})
print(m_ctrl.summary().tables[1])

# and for the single most direct "did lenders tighten the docs" flag
d = pr2.dropna(subset=["docs_tight"])
m_docs = smf.ols("docs_tight ~ structural + mixed + C(year) + C(industry)", data=d) \
            .fit(cov_type="cluster", cov_kwds={"groups": d["company"]})
print(m_docs.summary().tables[1])

# ==== CELL S2-3 (markdown) ====
With fixed effects the structural coefficient on the control index is +0.33 (p = 0.10) and on documents-tightened +0.10 (p = 0.41), so at best there is weak evidence that lenders demand a little more from structural borrowers.

# ==== CELL S2-4 (code) ====
# Deal form matters much more than cause: LME vs pro rata terms
print(neg.groupby("form")[TERMS + ["non_pro_rata_terms"]].mean().T.round(2))

# ==== CELL S2-4 (markdown) ====
LMEs look nothing like pro rata deals — 88% involve priming new money, 67% a haircut on non-participants, and 0% transfer equity control to creditors — so the form of the deal, not the cause of distress, is what determines the terms.

# ==== CELL S2-5 (code) ====
# Vintage: have pro rata terms shifted from 2018-21 to 2023-25?
pr2 = pr2.assign(vintage=np.where(pr2.year <= 2021, "2018-21", np.where(pr2.year == 2022, "2022", "2023-25")))
print(pr2.groupby("vintage")[["equity_any", "equitized", "pik", "nm_priming", "docs_tight"]].mean().T.round(2))

# ==== CELL S2-5 (markdown) ====
Terms have drifted toward LME-style structures over time: equity to creditors fell from 49% to 25% of pro rata deals while PIK (48% → 75%) and priming new money (12% → 48%) rose, so later-vintage "pro rata" deals increasingly resemble runway-only LMEs.

# ==== CELL S2-6 (code) ====
# Do the TERMS predict whether the deal held?  Balance-sheet fix (haircut / equitization) vs runway-only (extension / PIK)
e = pr2[(pr2.elig_2y == 1) & (pr2.relief_type != "other / not stated")].copy()
e["fix"] = e.relief_type.str.startswith("balance-sheet").astype(int)
print(e.groupby("relief_type").fail_2y.agg(["mean", "count"]).round(3))

m_fix = smf.ols("fail_2y ~ fix + structural + mixed + C(year)", data=e) \
           .fit(cov_type="cluster", cov_kwds={"groups": e["company"]})
print(m_fix.summary().tables[1])

# ==== CELL S2-6 (markdown) ====
Deals that actually cut debt fail half as often as deals that only buy time (29% vs 51% within two years), and in the regression both matter independently — a balance-sheet fix lowers failure by 25 points (p = 0.007) while being structural raises it by 22 points (p = 0.03).

# ==== CELL S2-7 (code) ====
# One chart for the advisor meeting: term shares by cause group (pro rata deals)
import matplotlib.pyplot as plt
ax = pr2.groupby("cause_group")[TERMS].mean().T.plot(kind="bar", figsize=(10, 4.5), width=0.8)
ax.set_title("Exchange terms by cause of distress — pro rata, negotiated deals")
ax.set_ylabel("Share of deals (where term stated)"); ax.set_ylim(0, 1)
ax.legend(["A: shock", "B: structural", "C: mixed"]); plt.xticks(rotation=30, ha="right"); plt.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.show()

# ==== CELL S2-7 (markdown) ====
The three bars are roughly level within every term, which is the visual version of the Part 2 finding: lenders' terms respond to the structure of the deal and its vintage, not to why the borrower is distressed.
