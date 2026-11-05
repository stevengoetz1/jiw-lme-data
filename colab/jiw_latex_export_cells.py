# ==== CELL S3-0 (markdown) ====
## Exporting tables to LaTeX (the Python version of stargazer)

`stargazer` exists for Python (`pip install stargazer`) and takes fitted statsmodels results directly. `render_latex()` gives a table you paste into Overleaf; `render_html()` previews it in Colab. For plain descriptive tables, pandas' `.to_latex()` does the job.

# ==== CELL S3-1 (code) ====
# Regression table: the four main Part 1 models side by side (needs the cells above to have run)
!pip -q install stargazer
from stargazer.stargazer import Stargazer

sg = Stargazer([m_pr2, m_pr3, m_bk2, m_pool])
sg.title("Cause of distress and post-exchange failure")
sg.custom_columns(["Fail 2y", "Fail 3y", "Bankrupt 2y", "Fail 2y (pooled)"], [1, 1, 1, 1])
sg.covariate_order(["structural", "mixed", "lme"])                         # hide the FE dummies and intercept
sg.rename_covariates({"structural": "Structural (B)", "mixed": "Mixed (C)", "lme": "LME"})
sg.add_line("Sample", ["Pro rata", "Pro rata", "Pro rata", "All"])
sg.add_line("Year FE", ["Yes"] * 4)
sg.add_line("Industry FE", ["Yes"] * 4)
sg.show_degrees_of_freedom(False)
sg.show_f_statistic = False; sg.show_residual_std_err = False; sg.show_adj_r2 = False
sg.add_custom_notes(["Linear probability models. Omitted cause group is A (shock).",
                     "Standard errors clustered by issuer in parentheses."])
latex = sg.render_latex().replace("\\end{table}", "\\label{tab:reg_part1}\n\\end{table}")   # add a label so \ref works
print(latex)                  # paste this into Overleaf as tables/reg_part1.tex
sg                            # HTML preview in Colab

# ==== CELL S3-2 (code) ====
# Descriptive table: failure rates by form x cause group, as LaTeX
rows = []
for form, df in (("LME", lme), ("Pro rata", pr)):
    for g in ("A", "B", "C"):
        r = {"Form": form, "Cause": g}
        for h in (1, 2, 3):
            e = df[(df[f"elig_{h}y"] == 1) & (df.cause_group == g)]
            r[f"{h}y rate"] = f"{e[f'fail_{h}y'].mean():.0%}".replace("%", r"\%"); r[f"{h}y n"] = len(e)
        rows.append(r)
tab = pd.DataFrame(rows).set_index(["Form", "Cause"])
latex2 = tab.to_latex(caption="Any failure within h years, by deal form and cause of distress", label="tab:rates", multirow=False, position="htbp")
print(latex2.replace("\\begin{table}[htbp]", "\\begin{table}[htbp]\n\\centering"))   # paste into Overleaf as tables/rates.tex
tab
