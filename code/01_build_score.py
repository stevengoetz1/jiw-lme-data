"""Step 1. Build the analysis files from the clean deal/terms files and the 70 re-codings per deal.

Writes data/analysis/deals_scored.csv  (277 deals, one row each, with the score and review flags)
       data/analysis/terms_scored.csv  (200 negotiated exchanges with the score and the strictness index)
"""
import pandas as pd
from common import *

deals, terms, cnt = load_raw()
inputs = pd.read_csv(P("data", "coded", "coding_inputs.csv"))
sc = build_sample(deals, cnt)

rep = terms[terms.negotiated == 0][["company", "deal_date"]].assign(repurchase=1)      # open-market repurchases
sc = sc.merge(rep, on=["company", "deal_date"], how="left"); sc["repurchase"] = sc.repurchase.fillna(0).astype(int)
sc["sp_listed"] = sc.sp_name.notna().astype(int)                                       # deal appears in S&P's default tables
sc["needs_review"], sc["review_reason"] = review_flags(sc, terms, inputs)
sc.to_csv(P("data", "analysis", "deals_scored.csv"), index=False)

neg = terms[terms.negotiated == 1].merge(sc[["company", "deal_date", "deal_id", "score", "score_bin", "struct_major", "score2", "score2_bin", "struct_major2"]], on=["company", "deal_date"], how="left")
neg["industry"] = neg["industry"].fillna("Unknown"); neg["year"] = neg.deal_date.dt.year
pr = neg.form == "Pro rata"
neg.loc[pr & neg.score.notna(), "strict_klk"] = strictness_index(neg[pr & neg.score.notna()])   # index is defined on the pro rata H2 sample
neg["fix"] = ((neg.haircut_any == 1) | (neg.equitized == 1)).astype(float)
neg["runway_only"] = neg.relief_type.str.startswith("runway").astype(float).where(~neg.relief_type.str.startswith("other"))
neg.to_csv(P("data", "analysis", "terms_scored.csv"), index=False)

print(f"deals_scored.csv: {len(sc)} deals, {sc.score.notna().sum()} scored ({(sc.lme == 1).sum()} LME, {(sc.pro_rata == 1).sum()} pro rata)")
print(f"  open-market repurchases: {sc.repurchase.sum()};  LMEs also in S&P tables: {int(sc[sc.lme == 1].sp_listed.sum())}")
print(f"  rows flagged needs_review: {sc.needs_review.sum()}")
print(sc[sc.needs_review == 1].review_reason.str.split("; ").explode().value_counts().to_string())
print(f"terms_scored.csv: {len(neg)} negotiated exchanges, {int((pr & neg.score.notna()).sum())} pro rata with a score")
