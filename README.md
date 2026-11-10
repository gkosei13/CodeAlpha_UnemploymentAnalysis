# CodeAlpha_UnemploymentAnalysis

**CodeAlpha Data Science Internship - Task 2**

Analyses unemployment data in India, with a focus on the impact of the Covid-19 lockdown.

## What this project does
1. **Cleans** the data: standardises column names, trims whitespace, parses dates, removes duplicates and missing values
2. **Explores** summary statistics and correlations
3. **Visualises**:
   - National unemployment trend with the lockdown period highlighted
   - Pre-Covid vs lockdown vs post-lockdown comparison
   - Top 10 regions hit hardest
   - Region-by-month heatmap
   - Distribution by zone / urban-rural area
   - Month-of-year (seasonality) view
   - Labour participation vs unemployment
4. **Generates insights** and policy implications automatically (`outputs/insights.txt`)

## How to run
```bash
pip install -r requirements.txt
python unemployment_analysis.py --csv "Unemployment in India.csv"
```
Charts, the cleaned dataset and the insights file are saved to `outputs/`.

## Key findings
*(Fill in from your `outputs/insights.txt` after running on the real dataset: peak month, pre-Covid vs lockdown averages, hardest-hit regions.)*

## Limitations
- The data spans only about 1.5 years, so true seasonality cannot be separated from the Covid shock.
- The analysis shows correlation with the lockdown period, not proof of cause.

## Tech stack
Python, Pandas, Matplotlib, Seaborn
