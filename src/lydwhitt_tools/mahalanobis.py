import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis

def geochemical_filter(df, phase, total_perc=None, percentiles=None):
    #remove all rows where Total is <96 or below a designated threshold
    if total_perc is None:
        total = 96
    else: 
        total = total_perc

    df1 = df[df["Total"] >= total]

    #Identify columns for analysis, 'type' is anything such as Cpx, Plg or Liq
    suffix = f"{phase}"
    numeric_cols = [col for col in df.columns if col.endswith(suffix)]

    #Keep Sample_ID column
    id_col = "Sample_ID"

    #Print columns so user knows which have been used for testing
    print(f"Using columns: {numeric_cols}")

    #delete rows with NA values- these are unuseable in this statistical test
    df1 = df1[[id_col]+numeric_cols].dropna(subset=numeric_cols)

    df1_numeric = df1[numeric_cols]

    #mahalanobis function
    def mahalnobis_test(df):

        if df.shape[0] < 2:
            print("Not enough rows to compute covariance.")
            return df, None, None, [0] * len(df)
        
        elif df.shape[1] < 2:
            print("Not enough columns to compute covariance (need at least 2).")
            return df, None, None, [0] * len(df)
        
        else:
            #find the mean vector
            mean_vector = df.mean().values

            #compute the covariance matrix and how the oxides vary together
            cov_matrix = np.cov(df.T)

            #invert the covariance matrix, this step adjusts for scale and correlation
            inv_cov_matrix = np.linalg.inv(cov_matrix)

            #calculate the mahalanobis distance for each row of data. each disnace is a single number, bigger means more different. 
            distances = []
            for i in range(len(df)):
                row = df.iloc[i].values
                d = mahalanobis(row, mean_vector, inv_cov_matrix)
                distances.append(d)

            #find the mean and standard deviation of these distances to filter the data


            return distances
        
    def resolve_percentiles(percentiles):

            default = (98, 98)

            if percentiles is None:
                p1, p2 = default
            elif isinstance(percentiles, (int, float)):
                p1 = p2 = float(percentiles)
            elif isinstance(percentiles, (list, tuple)) and len(percentiles) == 2:
                p1, p2 = map(float, percentiles)
            else:
                raise ValueError("percentiles must be None, a number, or a pair like (95, 99).")

            for p in (p1, p2):
                if not (0 < p <= 100):
                    raise ValueError(f"Percentile {p} must be in (0, 100].")

            return p1, p2
    
    p1, p2 = resolve_percentiles(percentiles)
    
    #run initial mahalanobis test over dataset
    distances1 = mahalnobis_test(df1_numeric)

    #define a threshold for pass 1, larger numbers will be considered outliers. we use the 95th percentile
    threshold1 = np.percentile(distances1, p1)

    #flag outliers
    df1['Mahalanobis1'] = distances1
    df1['P1_Outlier'] = df1['Mahalanobis1'] > threshold1
    
    #second pass of mahalnobis test
    #define new filtered dataset
    df2 = df1[df1["P1_Outlier"] == False]
    df2_numeric = df2[numeric_cols] 
    print(f"Pass 1: {df1.shape[0]} rows input")
    print(f"Pass 1 output: {df2.shape[0]} rows retained")

    #test that there is enough data remaining to complete second pass
    if df2.shape[0] < 2:
        print("Second pass aborted: not enough rows remaining after first pass.")
        return df1  # return the filtered dataset from pass 1
    
    #complete second pass of test
    distances2 = mahalnobis_test(df2_numeric)

    #define a threshold for pass 2, we use the 99th percentile
    threshold2 = np.percentile(distances2, p2)

    #flag outliers from second pass
    df2['Mahalanobis2'] = distances2
    df2['P2_Outlier'] = df2['Mahalanobis2'] > threshold2

    df3 = df2[df2["P2_Outlier"] == False]

    print(f"Pass 2: {df3.shape[0]} rows retained")
    print(f"Total rows lost: {df1.shape[0]-df3.shape[0]}")

    # Get common columns between df and df1 dataframes 
    common_columns1 = set(df.columns).intersection(df1.columns)
    # Exclude 'Sample Name' column from common columns
    common_columns1.discard('Sample_ID')
    # Drop common columns from df1 and df2 dataframes dataframe
    df1_dropped = df1.drop(columns=common_columns1, errors='ignore').drop_duplicates('Sample_ID')
      # Get common columns between df and df2 dataframes 
    common_columns2 = set(df1.columns).intersection(df2.columns)
    # Exclude 'Sample Name' column from common columns
    common_columns2.discard('Sample_ID')
    # Drop common columns from df1 and df2 dataframes dataframe
    df2_dropped = df2.drop(columns=common_columns2, errors='ignore').drop_duplicates('Sample_ID')
    
    df = pd.merge(df, df1_dropped, on="Sample_ID", how="left", validate="many_to_one")

    # attach pass-2 results to everyone who entered pass 2
    df = pd.merge(df, df2_dropped, on="Sample_ID", how="left", validate="many_to_one")

    return df




