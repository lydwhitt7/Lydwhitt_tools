# Lydwhitt_tools

A collection of functions and tools I have developed and find useful during my volcanology PhD.

Whilst I have a lot of tools and functions on my computer that I use regularly, I haven't yet put them all in a place to help others. There are a lot of very simple tasks that waste time, which I've created tools to complete, and I will be adding them to this repository over time.

I started collating this in August 2025 — so bear with me whilst I get it going!

---

## Install
pip install lydwhitt-tools

---

## Available tools

`geoscore_filter(df, phase, smooth_fn=None, smooth_kwargs=None, return_membership=False)`: End-to-end geochemical filtering pipeline (Liq, Plg, Cpx). Recalculates phase chemistry, fits smooth compositional trends against a progress variable, scores each analysis by how well it follows the main trend (robust local z) and how well it sits within the main high-density KDE modes (HDR membership plus proximity). Returns a continuous `geo_score` (0 to 1) and a boolean `final_pass`, plus optional membership diagnostics for debugging and plotting.

`filter_fig(df, diagnostics=None, geo_min=None, combine_mode=None, t_col=None, figsize=(18, 15), label_fs=11, tick_fs=9)`: Diagnostic plot for `geoscore_filter` results. Builds a multi-panel figure showing the geo_score distribution, the decision space (trend_score vs cluster_score), trend deviation versus the progress variable, retention versus threshold, Harker-style scatter plots with fitted trends, and KDE curves with the kept high-density (HDR) bands.

`mahalanobis_filter(df, phase, total_perc=None, percentiles=None)` : Two-pass multivariate outlier filter using robust Mahalanobis distance. Filters analyses by oxide totals, computes distances using a Minimum Covariance Determinant estimator, and applies chi-square thresholds to flag outliers in two stages. Returns the original dataframe with pass-1 and pass-2 distances, p-values, and outlier flags merged back for plotting and comparison.

`KDE(df, 'column')` : This function creates a plottable KDE line for a column of values in a dataframe using the imporved sheather jones method to establish bandwidth. This methodology uses an integrated r script rather than the usual python computing as this is more preferable in geochemical studies. Requires an R installation and the rpy2 Python package. 

`MD(x, y, z=None)` : Returns the x-value at the highest point of a KDE curve (the dominant peak). Optionally set `z` as a minimum peak height to ignore low-amplitude peaks.

`iqr_one_peak(df, 'data', z)` : Finds the dominant KDE peak (above a minimum height threshold) and returns that peak location plus a peak-specific IQR (Q1 and Q3) computed only from original data points within the dominant peak window.

`recalc(df, phase, anhydrous=True, mol_values=True)` : This function calculates the apfu or cation fraction of major elment data for Plg, Cpx, Ol and Liq (WR/Glass/MI) data. anhydrous needs to be specified for Liq data and if you dont want the mol fractions in the final dataframe just add mol_values=true.

`recalc_Fe(df)` : Ensures a single total-iron column (`FeOt`) is present by recalculating it from any combination of `FeO`, `Fe2O3`, or `Fe2O3t` (converting Fe2O3 to FeO equivalents). Preserves existing `FeOt` values and drops raw iron oxide columns.

`iron_ratios(df, ratio)` : Splits `FeOt_Liq` into `FeO_Liq` and `Fe2O3_Liq` using a supplied Fe3+/FeT ratio. Adds `Fe_wt`, `Fe3_wt`, and `Fe2_wt` intermediate columns and returns the updated dataframe.

---
## Worked Examples
Worked examples for each function are provided as Jupyter notebooks on GitHub.  
These notebooks show typical inputs, outputs, and common usage patterns.

- `geoscore_filter`, `filter_fig`
  https://shorturl.at/n7TYf

- `mahalanobis_filter`  
  in progress.

- `KDE`, `MD`, `iqr_one_peak` 
  https://shorturl.at/eXg4y

- `recalc`, `recalc_Fe`, `iron_ratios` 
  https://shorturl.at/ZH8vM
---

## Features Coming Soon
This repository will grow to include:
-data re-oragnisation tools for popular gothermobarometry packages

---
please contact Lydia: whittakl@tcd.ie with any further questions
---

## License
This project is licensed under the [MIT License](LICENSE).