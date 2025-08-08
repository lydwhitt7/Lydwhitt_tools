# Lydwhitt_tools

A collection of functions and tools I have developed and find useful during my volcanology PhD.

Whilst I have a lot of tools and functions on my computer that I use regularly, I haven't yet put them all in a place to help others. There are a lot of very simple tasks that waste time, which I've created tools to complete, and I will be adding them to this repository over time.

I started collating this in August 2025 — so bear with me whilst I get it going!

---

## Install
pip install lydwhitt-tools

---

## Quickstart
Here’s the simplest way to run the geochemical filter tool:
import pandas as pd
import lydwhitt_tools as lwt
Load your geochemical dataset
df = pd.read_csv("my_data.csv")
Apply the filter (defaults: total_perc=96, percentiles=(98, 98))
filtered_df = lwt.geochemical_filter(df, phase="Cpx")
print(filtered_df.head())


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