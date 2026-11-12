# ==== CELL S4-0 (markdown) ====
## Part 2b: A single "strictness" score (so one missing term doesn't throw a deal out)

`control_index` is a raw count of five flags, which has two problems: a deal whose release never mentions covenants is scored as if covenants were untouched, and deals with more terms *stated* mechanically score higher. The standard fix in applied economics is a **standardized summary index** (Kling, Liebman & Katz 2007; Anderson 2008): z-score each component, flip signs so that higher always means "tougher on the borrower," and average over whatever components are observed for that deal. Anderson's version additionally weights components by the inverse of their covariance so that near-duplicate terms (e.g. haircut and equitization) don't double count. Both are computed below; the regression uses all 148 negotiated pro rata deals.

# ==== CELL S4-1 (code) ====
# Build the strictness index.  +1 direction = lender-protective / tougher on borrower.
STRICT_POS = ["haircut_any", "equitized", "equity_control", "governance", "docs_tight", "nm_priming", "non_pro_rata_terms"]
STRICT_NEG = ["docs_loose", "sponsor_support", "extension_any", "pik"]        # these are borrower-friendly runway terms
items = STRICT_POS + STRICT_NEG

X = pr2[items].astype(float).copy()
X[STRICT_NEG] = -X[STRICT_NEG]                                        # flip so higher = stricter
Z = (X - X.mean()) / X.std(ddof=0)                                     # z-score each item over deals where it is stated

# (1) Kling-Liebman-Katz: simple mean of the available z-scores (needs at least 4 stated items)
pr2["strict_klk"] = Z.mean(axis=1, skipna=True).where(Z.notna().sum(axis=1) >= 4)

# (2) Anderson (2008): inverse-covariance weighted mean, computed on the available items for each deal
S = Z.cov()
def anderson(row):
    avail = row.dropna().index
    if len(avail) < 4: return np.nan
    Sinv = np.linalg.pinv(S.loc[avail, avail].values)
    w = Sinv.sum(axis=1); w = w / w.sum()
    return float(w @ row[avail].values)
pr2["strict_anderson"] = Z.apply(anderson, axis=1)

print("deals scored:", pr2.strict_klk.notna().sum(), "of", len(pr2))
print(pr2.groupby("cause_group")[["strict_klk", "strict_anderson"]].agg(["mean", "count"]).round(2))
print("Kruskal-Wallis p (KLK):", round(stats.kruskal(*[g.strict_klk.dropna() for _, g in pr2.groupby("cause_group")]).pvalue, 3))

# ==== CELL S4-2 (code) ====
# Regress the strictness scores on cause, year + industry FE, clustered by company
for y in ["strict_klk", "strict_anderson"]:
    d = pr2.dropna(subset=[y])
    m = smf.ols(f"{y} ~ structural + mixed + C(year) + C(industry)", data=d).fit(cov_type="cluster", cov_kwds={"groups": d["company"]})
    print(f"\n{y}  (n={int(m.nobs)}; units = standard deviations of strictness)")
    print(m.summary().tables[1])

# ==== CELL S4-2 (markdown) ====
On a single strictness scale that uses every deal, structural firms score only 0.14–0.17 standard deviations tougher than shock firms (p = 0.23 and 0.13), so pooling the terms into one score with full n does not rescue the screening result — the point estimate is small, not just noisy.

# ==== CELL S4-3 (code) ====
# The "is the discrimination enough?" test.  If lenders already priced cause into the terms, then
# conditional on the terms the cause of distress should no longer predict failure.
e = pr2[(pr2.elig_2y == 1)].dropna(subset=["strict_klk"])
m0 = smf.ols("fail_2y ~ structural + mixed + C(year)", data=e).fit(cov_type="cluster", cov_kwds={"groups": e["company"]})
m1 = smf.ols("fail_2y ~ strict_klk + structural + mixed + C(year)", data=e).fit(cov_type="cluster", cov_kwds={"groups": e["company"]})
print("Without terms:  structural =", round(m0.params["structural"], 3), " p =", round(m0.pvalues["structural"], 3), f"(n={int(m0.nobs)})")
print("With strictness: structural =", round(m1.params["structural"], 3), " p =", round(m1.pvalues["structural"], 3),
      "| strictness coef =", round(m1.params["strict_klk"], 3), " p =", round(m1.pvalues["strict_klk"], 3))

# ==== CELL S4-3 (markdown) ====
This is the test of whether whatever discrimination lenders do is "enough": terms matter a lot (one standard deviation of strictness cuts two-year failure by 27 points, p < 0.001), yet adding them leaves the structural coefficient unchanged (0.23 → 0.22), which means the tough terms are not being allocated to the firms whose cause of distress predicts failure — if lenders were screening on cause, the terms would have absorbed the cause effect.
