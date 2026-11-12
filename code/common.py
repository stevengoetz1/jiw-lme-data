"""Shared loading code. The structural score is defined here and nowhere else."""
import os, re, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a)
MODELS = {"gpt4omini": "gpt-4o-mini", "gpt41mini": "gpt-4.1-mini", "gpt4o": "gpt-4o"}
BINS = ["Low (0-0.33)", "Middle (0.33-0.67)", "High (0.67-1)"]
FE = " + C(year) + C(industry)"


def add_scores(cnt):
    """score = share of the 70 blind re-codings that say structural, a mixed coding counting one half."""
    cnt = cnt.copy()
    nA = sum(cnt[f"A_{m}"] for m in MODELS); nB = sum(cnt[f"B_{m}"] for m in MODELS); nC = sum(cnt[f"C_{m}"] for m in MODELS)
    cnt["score"] = (nB + 0.5 * nC) / (nA + nB + nC)            # MAIN measure: 0 = pure shock, 1 = pure structural
    cnt["score_noC"] = nB / (nA + nB)                           # ignores the mixed runs
    for m in MODELS:                                            # the same score from each model alone
        cnt[f"score_{m}"] = (cnt[f"B_{m}"] + 0.5 * cnt[f"C_{m}"]) / (cnt[f"A_{m}"] + cnt[f"B_{m}"] + cnt[f"C_{m}"])
    mk = cnt[["A_gpt4omini_masked", "B_gpt4omini_masked", "C_gpt4omini_masked"]]
    cnt["score_masked"] = (mk.iloc[:, 1] + 0.5 * mk.iloc[:, 2]) / mk.sum(axis=1)   # company names removed
    return cnt


def add_scores2(c2):
    """Two-label score (main measure from 12 Nov 2026): share of the 70 shock-or-structural codings that say structural."""
    c2 = c2.copy()
    c2["score2"] = sum(c2[f"B2_{m}"] for m in MODELS) / sum(c2[f"A2_{m}"] + c2[f"B2_{m}"] for m in MODELS)
    for m in MODELS:
        c2[f"score2_{m}"] = c2[f"B2_{m}"] / (c2[f"A2_{m}"] + c2[f"B2_{m}"])
    return c2


def load_raw():
    deals = pd.read_csv(P("data", "clean", "lme_deals_clean.csv"), parse_dates=["deal_date", "bk_date", "fail_date"])
    terms = pd.read_csv(P("data", "clean", "lme_terms_clean.csv"), parse_dates=["deal_date"])
    cnt = add_scores(pd.read_csv(P("data", "coded", "llm_recoding_counts.csv")))
    return deals, terms, cnt


def build_sample(deals, cnt):
    """The 277-deal analysis sample with every version of the cause measure attached."""
    sc = deals[deals.in_analysis == 1].copy()
    sc["form"] = np.where(sc.lme == 1, "LME", "Pro rata")
    sc["industry"] = sc["industry"].fillna("Unknown")
    sc = sc.merge(cnt[["deal_id", "score", "score_noC", "score_masked"] + [f"score_{m}" for m in MODELS]], on="deal_id", how="left")
    sc["coder1_score"] = sc.cause_group.map({"A": 0.0, "C": 0.5, "B": 1.0})
    sc["struct_major"] = np.where(sc.score.isna(), np.nan, (sc.score > 0.5).astype(float))
    sc["score_bin"] = pd.cut(sc.score, [-0.01, 1 / 3, 2 / 3, 1.0], labels=BINS)
    sc["bin_mid"] = (sc.score_bin == BINS[1]).astype(int); sc["bin_high"] = (sc.score_bin == BINS[2]).astype(int)
    c2 = add_scores2(pd.read_csv(P("data", "coded", "llm_recoding_counts_binary.csv")))
    sc = sc.merge(c2[["deal_id", "score2"] + [f"score2_{m}" for m in MODELS]], on="deal_id", how="left")
    sc["struct_major2"] = np.where(sc.score2.isna(), np.nan, (sc.score2 > 0.5).astype(float))
    sc["score2_bin"] = pd.cut(sc.score2, [-0.01, 1 / 3, 2 / 3, 1.0], labels=BINS)
    return sc


def review_flags(sc, terms, inputs):
    """Rows a human should re-check before the final paper. Returns needs_review (0/1) and review_reason."""
    reasons = pd.Series("", index=sc.index)
    conf = sc.code_confidence.fillna("").str.upper()
    add = lambda mask, txt: reasons.where(~mask, reasons + txt + "; ")
    reasons = add(conf.str.contains("GUESS"), "cause code marked GUESS")
    reasons = add(conf.str.startswith("LOW") & ~conf.str.contains("GUESS"), "cause code low confidence")
    reasons = add(sc.sourced == 0, "no quotable agency source")
    reasons = add(sc.score.isna(), "no agency text, so no score")
    def late(r):                                                # every year cited in the quotes is after the deal year
        ys = [int(y) for y in re.findall(r"\b(20[0-2]\d)\b", str(r.agency_quotes))]
        return bool(ys) and min(ys) > r.deal_date.year
    inp = inputs.copy(); inp["deal_date"] = pd.to_datetime(inp.deal_date)
    late_ids = set(inp[inp.apply(late, axis=1)].deal_id)
    reasons = add(sc.deal_id.isin(late_ids), "agency quotes all dated after the deal year")
    return (reasons != "").astype(int), reasons.str.rstrip("; ")


def strictness_index(pt):
    """Kling-Liebman-Katz index: mean of the standardised terms, signed so higher = tougher on the borrower."""
    POS = ["haircut_any", "equitized", "equity_control", "governance", "docs_tight", "nm_priming", "non_pro_rata_terms"]
    NEG = ["docs_loose", "sponsor_support", "extension_any", "pik"]
    X = pt[POS + NEG].astype(float).copy(); X[NEG] = -X[NEG]; Z = (X - X.mean()) / X.std(ddof=0)
    return Z.mean(axis=1, skipna=True).where(Z.notna().sum(axis=1) >= 4)


def fit(formula, d):
    """OLS / linear probability model with standard errors clustered by issuer."""
    return smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["company"]})


def load_scored():
    """Read the built analysis files (run 01_build_score.py first)."""
    sc = pd.read_csv(P("data", "analysis", "deals_scored.csv"), parse_dates=["deal_date", "bk_date", "fail_date"])
    sc["score_bin"] = pd.Categorical(sc.score_bin, categories=BINS, ordered=True)
    sc["score2_bin"] = pd.Categorical(sc.score2_bin, categories=BINS, ordered=True)
    tm = pd.read_csv(P("data", "analysis", "terms_scored.csv"), parse_dates=["deal_date"])
    return sc, tm
