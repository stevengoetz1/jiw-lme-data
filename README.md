# JIW: cause of distress and the outcomes of distressed exchanges

Data, code and paper sources for my junior independent work. Everything in the papers is rebuilt from this repo with one command:

```
python code/run_all.py
```

That takes about a minute and needs `pandas`, `statsmodels`, `scipy` and `matplotlib`. In Colab, open `notebooks/JIW_run_all.ipynb`, which clones the repo and runs the same command.

## Layout

| Folder | What is in it | Edited by |
|---|---|---|
| `data/clean/` | `lme_deals_clean.csv` (310 rows, 277 in the analysis sample: 89 Roe-Rotaru LMEs + 188 pro rata S&P distressed exchanges) and `lme_terms_clean.csv` (terms of the 242 S&P-recorded exchanges) | hand |
| `data/coded/` | `coding_inputs.csv` (the rating-agency text each coder read) and `llm_recoding_counts.csv` (70 blind re-codings per deal) | hand / coding runs |
| `data/analysis/` | `deals_scored.csv` and `terms_scored.csv`, the files every analysis reads | **script only** (`01_build_score.py`) |
| `code/` | `common.py` (the score is defined here and nowhere else), `01_build_score.py`, `02_main_analysis.py`, `03_prospectus_tables.py`, `run_all.py` | hand |
| `output/` | `results_log.txt` (every printed result), `tables/thesis/`, `tables/prospectus/`, `figures/` | **script only** |
| `paper/` | LaTeX sources of the thesis and the prospectus; `run_all.py` copies the generated tables into each `tables/` folder | hand (text), script (tables) |
| `docs/` | `codebook.txt`, column definitions for every data file | hand |
| `notebooks/` | `JIW_run_all.ipynb`, a thin Colab runner | hand |
| `archive/` | the old Colab cell exports from before the restructure, kept for reference | nobody |

## The main variable

`score2` in `data/analysis/deals_scored.csv` is the structural score used in the prospectus: the share of 70 blind codings that call the deal's distress structural, where each coding had to choose shock or structural (0 = every coding says shock, 1 = every coding says structural). The counts behind it are in `data/coded/llm_recoding_counts_binary.csv`.

The same file carries the alternatives: `struct_major2` (binary, score2 above one half), `score2_bin` (thirds), `cause_group` (the original three labels: A shock, B structural, C mixed), and `score` (the earlier score from codings that allowed a mixed answer, mixed counting one half), which the thesis draft still uses. `03_prospectus_tables.py` renames `score2` to `score` internally, so in that script `score` is the two-label measure and `score3` the earlier one.

## Rows to re-check

`needs_review == 1` marks 27 deals, with the reason in `review_reason`: cause codes marked as guesses or low confidence, deals with no quotable agency source, deals whose quoted agency text is all dated after the deal year, and the two deals with no agency text.

## Where each number in the papers comes from

| Paper item | Script | Output |
|---|---|---|
| Thesis tables (rates, regressions, terms, robustness) | `02_main_analysis.py` | `output/tables/thesis/` |
| Prospectus Table 1 (summary statistics) | `03_prospectus_tables.py` | `output/tables/prospectus/sumstats.tex` |
| Prospectus Table 2 (three versions of the measure) | `03_prospectus_tables.py` | `reg_part1.tex` |
| Prospectus Table 3 (robustness) | `03_prospectus_tables.py` | `robust.tex` |
| Prospectus Table 4 (H2, terms) | `03_prospectus_tables.py` | `h2.tex` (the paper keeps this table inline in `main.tex`) |
| Prospectus Figure 1 | `03_prospectus_tables.py` | `fig1_data.csv` (the paper draws it in pgfplots from these numbers) and `figures/fig1_failure_by_cause.png` |
| Numbers quoted only in the text (vintages, outcome split, power, names-removed baseline) | `03_prospectus_tables.py` | "EXTRA CHECKS" at the end of `output/results_log.txt` |

`paper/thesis/tables/sample.tex` and `vintage.tex` are older hand-built tables and are not regenerated.

## Not in this repo

The raw S&P default-study tables and the Roe-Rotaru deal list that the clean files were assembled from, and the per-run re-coding output with rationales (`llm_codes.csv`).
