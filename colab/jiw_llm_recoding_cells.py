# ==== CELL S5-0 (markdown) ====
## Part 3: Re-coding cause of distress with repeated LLM "coders"

The cause codes in the dataset were produced by one pass of an AI coder applying the codebook to each firm's pre-deal rating-agency text. The cells below treat that as coder #1 and add coders #2, #3, … by asking a language model to code every deal many times, then use the **share of runs that land on each group** as a probability that the deal is A / B / C.

**What each coder is given** (file `coding_inputs.csv` on GitHub, one row per deal):
- company name, deal date, S&P industry sector;
- `agency_quotes`: the verbatim S&P / Moody's rating-action language from the 12 months before the deal (with the agency and month);
- `ebitda_trend`: direction of EBITDA over the two years before the deal (up / flat / down), from the same reports;
- `secondary_factors`: other stressors the reports mention.

**What it is deliberately *not* given:** the existing code, the first coder's one-line rationale, whether the deal was an LME or a pro rata exchange, and anything about what happened afterwards (outcome, terms). Industry is included because the codebook needs it (B1 is a secular *industry* decline).

**One honest caveat.** The quotes were selected by coder #1, so agreement here is an upper bound on true inter-coder reliability. To make it a fully independent coding, replace `agency_quotes` with the complete rating-action text from Bloomberg / Capital IQ and rerun — the code below does not need to change.

**How many runs, and why.** Each run at temperature 1 is one draw from the model's distribution over labels ("self-consistency", Wang et al. 2023). The Monte-Carlo error of a share estimated from $K$ runs is $\sqrt{p(1-p)/K}$: about ±9 pp at $K=30$, ±5 pp at $K=100$, ±1.6 pp at $K=1000$. Beyond ~50 runs per model you are measuring the model's noise very precisely without learning anything new about the deal, so the budget is better spent on **more models than more runs**: different model families are closer to independent coders than repeated draws from one model. Default below: 30 runs × 2 models = 60 codings per deal. The published practice this follows: Gilardi, Alizadeh & Kubli (2023, *PNAS*) and Törnberg (2024) for LLMs as annotators measured against human coders with Krippendorff's α; Reiss (2023) on why single-run LLM labels are unstable and should be repeated; Ziems et al. (2024, *Computational Linguistics*) for prompt design; Egami et al. (2023, *NeurIPS*, "design-based supervised learning") for how to use LLM labels in a downstream regression without bias when a human-coded subsample exists. Reliability benchmarks are Krippendorff's: α ≥ 0.80 reliable, 0.67–0.80 tentative.

**Cost.** ~1,200 input tokens per call; with `n=30` completions per call and 277 deals, gpt-4o-mini costs well under $1 and gpt-4o a few dollars. Set `DRY_RUN = True` first (3 deals) to check the output format.

# ==== CELL S5-1 (code) ====
# ---- Setup: API key from Colab secrets (key icon in the left sidebar -> add OPENAI_API_KEY), models, run count ----
!pip -q install openai krippendorff
import json, time, os
from openai import OpenAI
from google.colab import userdata

client = OpenAI(api_key=userdata.get("OPENAI_API_KEY"))
MODELS = ["gpt-4o-mini", "gpt-4o"]      # each model = one additional "coder"; add/remove freely
K = 30                                   # runs per model per deal (see markdown above for why 30 not 1000)
DRY_RUN = True                           # True = code 3 deals only, to check everything works
OUT_PATH = "llm_codes.csv"               # results cached here so you never pay twice (download it when done)

inputs = pd.read_csv(BASE + "coding_inputs.csv")
print(len(inputs), "deals;", inputs.agency_quotes.notna().sum(), "with agency text")

# ==== CELL S5-2 (code) ====
# ---- The codebook the coder applies.  This is Appendix A of the methods note, condensed. ----
CODEBOOK = """You are coding the CAUSE OF FINANCIAL DISTRESS for a company that completed a distressed debt exchange.
Read only the evidence provided (rating-agency language from the 12 months before the deal, the EBITDA trend, and
secondary factors). Assign exactly one code:

A = SHOCK (temporary; the business model is intact and the problem is expected to pass or is purely financial)
  A1  Demand / operating shock: an identifiable external event hit demand, prices or operations (pandemic, commodity
      price collapse, hurricane, strike, a one-off customer loss) and the agency language treats it as cyclical or temporary.
  A2  Financial shock: the operating business is stable (EBITDA flat or up) but the capital structure is the problem --
      LBO leverage, a maturity wall, refinancing risk, floating-rate interest burden, a covenant trip with stable EBITDA.
B = STRUCTURAL (the business itself is impaired and the agency does not expect it to recover on its own)
  B1  Secular industry decline: the whole industry is shrinking (print media, department stores, coal, landline telecom,
      offshore drilling after 2014, etc.).
  B2  Firm-specific deterioration: competitive losses, failed strategy, botched integration, lost contracts, management
      failures -- the firm is declining while its industry is not.
  B3  Legal / regulatory: litigation, regulatory change or government action is the primary driver (opioid liability,
      reimbursement cuts, a license loss).
C = MIXED: an A-type shock landed on a firm that already had B-type problems, and neither clearly dominates.

Tie-break rules (apply in order):
1. Identify the PRIMARY driver the agency names. If one driver is clearly primary, code it; secondary factors do not change the letter.
2. EBITDA rule: if EBITDA fell more than ~10% over the two years before the deal AND the agency attributes it to the
   business rather than an external event, lean B. If EBITDA is flat or up, lean A2.
3. COVID rule: a 2020-21 deal where the agency names the pandemic as primary is A1 unless the firm was already on
   negative outlook / downgraded for operating reasons in the 24 months before March 2020 -- then C.
4. LBO rule: leverage from a buyout or dividend recap with a stable business is A2, not B2.
5. If an external shock and a pre-existing structural problem are given roughly equal weight, code C.

Respond with JSON only: {"code": "A1|A2|B1|B2|B3|C", "group": "A|B|C", "confidence": 0.0-1.0, "reason": "<one sentence>"}"""

def deal_prompt(r):
    return (f"Company: {r.company}\nDeal date: {r.deal_date}\nIndustry (S&P sector): {r.industry}\n\n"
            f"Rating-agency language in the 12 months before the deal:\n{r.agency_quotes}\n\n"
            f"EBITDA trend over the prior two years: {r.ebitda_trend}\n"
            f"Secondary factors mentioned: {r.secondary_factors}\n\nCode this deal.")

# ==== CELL S5-3 (code) ====
# ---- Run the coders.  One API call per deal per model returns K sampled codings (n=K). Results append to OUT_PATH. ----
done = set()
if os.path.exists(OUT_PATH):
    prev = pd.read_csv(OUT_PATH); done = set(zip(prev.deal_id, prev.model)); print("already coded:", len(done), "deal-model pairs")

todo = inputs.dropna(subset=["agency_quotes"])
if DRY_RUN: todo = todo.head(3)

rows = []
for _, r in todo.iterrows():
    for model in MODELS:
        if (r.deal_id, model) in done: continue
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model, temperature=1.0, n=K, max_tokens=120,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": CODEBOOK}, {"role": "user", "content": deal_prompt(r)}])
                break
            except Exception as ex:
                print("retry", model, r.company, ex); time.sleep(5 * (attempt + 1))
        for i, ch in enumerate(resp.choices):
            try:
                j = json.loads(ch.message.content)
                rows.append(dict(deal_id=r.deal_id, company=r.company, model=model, run=i, code=str(j.get("code", "")).upper()[:2],
                                 group=str(j.get("group", "")).upper()[:1], confidence=j.get("confidence"), reason=j.get("reason")))
            except Exception:
                rows.append(dict(deal_id=r.deal_id, company=r.company, model=model, run=i, code=None, group=None, confidence=None, reason="PARSE_ERROR"))
    if len(rows) and len(rows) % (K * len(MODELS) * 10) == 0: print("coded", len(rows) // (K * len(MODELS)), "deals")

new = pd.DataFrame(rows)
if len(new):
    new.to_csv(OUT_PATH, mode="a", header=not os.path.exists(OUT_PATH), index=False)
llm = pd.read_csv(OUT_PATH)
llm["group"] = llm.group.where(llm.group.isin(["A", "B", "C"]), llm.code.str[0])   # fall back to first letter of code
print(len(llm), "codings |", llm.deal_id.nunique(), "deals |", llm.model.unique())
llm.head()

# ==== CELL S5-4 (code) ====
# ---- Aggregate: per deal, the share of runs in A / B / C (overall and per model), modal group, agreement with coder #1 ----
probs = llm.pivot_table(index="deal_id", columns="group", values="run", aggfunc="count", fill_value=0)
probs = probs.div(probs.sum(axis=1), axis=0).rename(columns=lambda g: f"p_{g}")
for g in "ABC":
    if f"p_{g}" not in probs: probs[f"p_{g}"] = 0.0
probs["llm_modal"] = probs[["p_A", "p_B", "p_C"]].idxmax(axis=1).str[-1]
probs["llm_maxp"] = probs[["p_A", "p_B", "p_C"]].max(axis=1)
by_model = llm.groupby(["deal_id", "model"]).group.agg(lambda s: s.value_counts().idxmax()).unstack()   # each model's modal label

cmp = samp.set_index("deal_id")[["company", "cause_group", "cause_code", "code_confidence", "form"]].join(probs, how="inner").join(by_model)
print("deals with LLM codes:", len(cmp))
print("agreement of pooled LLM modal label with coder #1:", round((cmp.llm_modal == cmp.cause_group).mean(), 3))
for mdl in MODELS:
    if mdl in cmp: print(f"  {mdl} modal vs coder #1:", round((cmp[mdl] == cmp.cause_group).mean(), 3))
print("\nconfusion (rows = coder #1, cols = LLM modal):"); print(pd.crosstab(cmp.cause_group, cmp.llm_modal))
print("\nshare of deals where >= 80% of runs agree:", round((cmp.llm_maxp >= 0.8).mean(), 3))
cmp.sort_values("llm_maxp").head(10)[["company", "cause_group", "p_A", "p_B", "p_C", "llm_maxp"]]   # the hardest cases

# ==== CELL S5-5 (code) ====
# ---- Reliability: Krippendorff's alpha across coders (coder #1 + each model's modal label); Fleiss kappa too ----
import krippendorff
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
coders = ["cause_group"] + [m for m in MODELS if m in cmp]
mat = cmp[coders].apply(lambda col: col.map({"A": 0, "B": 1, "C": 2})).T.values.astype(float)
print("Krippendorff's alpha (nominal), coder #1 + model modal labels:", round(krippendorff.alpha(reliability_data=mat, level_of_measurement="nominal"), 3))
print("  benchmark: >= 0.80 reliable, 0.67-0.80 tentative (Krippendorff 2018)")
table, _ = aggregate_raters(mat.T.astype(int))
print("Fleiss' kappa:", round(fleiss_kappa(table), 3), " (Landis & Koch: 0.61-0.80 substantial, > 0.80 almost perfect)")
# alpha treating EVERY run as a coder (how stable is the model with itself?)
runs = llm.pivot_table(index=["model", "run"], columns="deal_id", values="group", aggfunc="first")
runs_m = runs.apply(lambda col: col.map({"A": 0, "B": 1, "C": 2})).values.astype(float)
print("Krippendorff's alpha across all individual runs (within- and between-model):", round(krippendorff.alpha(reliability_data=runs_m, level_of_measurement="nominal"), 3))

# ==== CELL S5-6 (code) ====
# ---- Use the probabilities in Part 1 three ways ----
e = samp[(samp.form == "Pro rata") & (samp.elig_2y == 1)].merge(probs, left_on="deal_id", right_index=True, how="inner")
print("pro rata, eligible 2y, with LLM codes: n =", len(e))

# (a) Original codes, same deals (benchmark)
m_a = smf.ols("fail_2y ~ structural + mixed + C(year) + C(industry)", data=e).fit(cov_type="cluster", cov_kwds={"groups": e["company"]})
# (b) High-agreement subset only: drop deals where the coders disagree (max share < 0.8)
hi = e[e.llm_maxp >= 0.8]
m_b = smf.ols("fail_2y ~ structural + mixed + C(year) + C(industry)", data=hi).fit(cov_type="cluster", cov_kwds={"groups": hi["company"]})
# (c) Probabilities as regressors: p_B and p_C instead of 0/1 dummies (measurement-error-aware version)
m_c = smf.ols("fail_2y ~ p_B + p_C + C(year) + C(industry)", data=e).fit(cov_type="cluster", cov_kwds={"groups": e["company"]})
# (d) LLM modal label instead of coder #1
e["struct_llm"] = (e.llm_modal == "B").astype(int); e["mixed_llm"] = (e.llm_modal == "C").astype(int)
m_d = smf.ols("fail_2y ~ struct_llm + mixed_llm + C(year) + C(industry)", data=e).fit(cov_type="cluster", cov_kwds={"groups": e["company"]})

out = pd.DataFrame({
    "(a) coder #1 codes":        [m_a.params["structural"], m_a.pvalues["structural"], int(m_a.nobs)],
    "(b) high-agreement only":   [m_b.params["structural"], m_b.pvalues["structural"], int(m_b.nobs)],
    "(c) P(structural) as regressor": [m_c.params["p_B"], m_c.pvalues["p_B"], int(m_c.nobs)],
    "(d) LLM modal label":       [m_d.params["struct_llm"], m_d.pvalues["struct_llm"], int(m_d.nobs)],
}, index=["structural effect on fail_2y", "p-value", "n"]).T
out.round(3)

# ==== CELL S5-6 (markdown) ====
How to read this: (a) is the Part 1 result on the same deals; (b) asks whether it survives when the ambiguous deals are thrown out; (c) treats the coding as uncertain and lets a deal that is 70% B count as 0.7 of a structural firm; (d) swaps coders entirely. If all four agree in sign and rough size, the Part 1 finding does not depend on which coder did the coding. If (b) is much larger than (a), the ambiguous deals were diluting the effect; if (d) is much smaller, coder #1 and the LLM disagree in a way that matters and the disagreements in cell S5-4 need a human look.
