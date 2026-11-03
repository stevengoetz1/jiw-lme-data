# JIW: LME and distressed-exchange datasets

Data for my junior independent work on cause of distress and the outcomes of liability management exercises (LMEs).

| File | Rows | What it is |
|---|---|---|
| `lme_deals_clean.csv` | 310 (277 with `in_analysis == 1`) | One row per deal: 89 Roe-Rotaru LMEs + 188 pro rata S&P distressed exchanges, with cause-of-distress codes and 1/2/3-year outcomes |
| `lme_terms_clean.csv` | 242 | Terms of each S&P-recorded exchange (haircut, extension, PIK, new money, control, covenants); join on `company` + `deal_date` |
| `lme_data_codebook.txt` | | Column definitions |

## Load in Colab

```python
import pandas as pd
BASE = "https://raw.githubusercontent.com/stevengoetz1/jiw-lme-data/main/"
deals = pd.read_csv(BASE + "lme_deals_clean.csv", parse_dates=["deal_date", "bk_date", "fail_date"])
terms = pd.read_csv(BASE + "lme_terms_clean.csv", parse_dates=["deal_date"])
sample = deals[deals.in_analysis == 1]
```
