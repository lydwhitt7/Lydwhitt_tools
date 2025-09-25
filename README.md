# Lydwhitt_tools

A collection of functions and tools I have developed and find useful during my volcanology PhD.

Whilst I have a lot of tools and functions on my computer that I use regularly, I haven't yet put them all in a place to help others. There are a lot of very simple tasks that waste time, which I've created tools to complete, and I will be adding them to this repository over time.

I started collating this in August 2025 — so bear with me whilst I get it going!

---

## Install
pip install lydwhitt-tools

---

## Available tools
`gechemical_filter(df, phase, total_perc=None, percentiles=None)`: This function is used to filter geochemical datasets using the Mahalanobis distance method in two passes. Please note the fuction filters out rows with totals <96 (default) prior to performing the mahalnobis test passes but re-adds them after processing so they can be plotted. dataframes will need filtering for this and both test results after function is used. 

`KDE(df, 'column')` : This function creates a plottable KDE line for a column of values in a dataframe using the imporved sheather jones method to establish bandwidth. This methodology uses an integrated r script rather thna the usual python computing as this is more preferable in geochemical studies. 

`MD(x, y, z)` : This function finds the value of the first peak found using the KDE function based on a minimum height threshold (may need adjusting per dataset). 

`iqr_one_peak(df, 'data', z)` : This function finds the MD peak as with the previous function but also gives you the Q1 and Q3 range of each dataset.

`recalc(df, phase, anhydrous=True, mol_values=True)` : This function calculates the apfu or cation fraction of major elment data for Plg, Cpx, Ol and Liq (WR/Glass/MI) data. anhydrous needs to be specified for Liq data and if you dont want the mol fractions in the final dataframe just add the mol_values=true. 
---

## Detailed Usage
The `geochemical_filter` function filters geochemical datasets using the Mahalanobis distance method in two passes.

**Function signature:**
lwt.geochemical_filter(df, phase, total_perc=None, percentiles=None)

**Parameters:**
- `phase` *(str)* – e.g., `"Cpx"`, `"Plg"`, `"Liq"`. Must match the suffix used after oxide wt% values in your dataset.
- `total_perc` *(float, optional)* – Minimum total oxide percentage allowed. Defaults to `96`.
- `percentiles` *(int, float, or tuple, optional)* – Percentile cutoffs for Pass 1 and Pass 2.  
  - Single value applies to both passes (e.g., `percentiles=98`).  
  - Tuple gives different cutoffs for each pass (e.g., `percentiles=(95, 99)`).

**Example:**
Use different percentiles for each pass
filtered_df = lwt.geochemical_filter(df, phase="Plg", total_perc=97, percentiles=(95, 99))

---

## Output
The function returns the filtered DataFrame, including flags for each pass:
- `Mahalanobis` – Mahalanobis distance for each row in the given pass.
- `P1_Outlier` – Boolean flag indicating if the row was an outlier in Pass 1.
- `P2_Outlier` – Boolean flag indicating if the row was an outlier in Pass 2.

---

## Features Coming Soon
This repository will grow to include:
-formula recalculations for different phases
-simple plotting frameworks
-data re-oragnisation tools for popular gothermobarometry packages



---

## License
This project is licensed under the [MIT License](LICENSE).