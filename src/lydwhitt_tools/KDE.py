import pandas as pd
import numpy as np
from scipy.signal import find_peaks, peak_widths
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter


#creating a function to create a panda df with the required information for input into the KDE 
def KDE_Format(df, existing_column_name, new_column_name='KDE_in'): 
    KDE = pd.DataFrame({new_column_name: df[existing_column_name]})
    return KDE

# input like this: KDE(df, 'column name')
def KDE(x, y):
    # give df the correct formatting for OPAM removing unnecessary columns
    x = KDE_Format(x, y)

    # Convert pandas DataFrame to R dataframe (modern method)
    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_dataframe = robjects.conversion.py2rpy(x)

    # Load the R script containing the 'perform_kde' function
    r_script = '''
    perform_kde <- function(r_dataframe) {
        data <- r_dataframe[, 1]
        mean_val <- mean(data)
        stdev_val <- sd(data)
        BW <- bw.SJ(data)
        Dp <- density(data, bw = BW)
        output <- cbind(Dp$x, Dp$y)
        centre <- Dp$x[Dp$y == max(Dp$y)]
        output_df <- data.frame(x = output[, 1], y = output[, 2])
        return(output_df)
    }
    '''

    # Execute the R script in the current R session
    robjects.r(r_script)

    # Call the 'perform_kde' function in R and pass the R dataframe as an argument
    kde_output = robjects.r['perform_kde'](r_dataframe)

    # Convert the R data.frame object to a pandas DataFrame (modern method)
    with localconverter(robjects.default_converter + pandas2ri.converter):
        KDE_df = robjects.conversion.rpy2py(kde_output)

    return KDE_df


# input like this: KDE(df, 'column name')
#this function finds the highest point of a KDE peak and spits out the value. X and Y are data inputs as they appear on the graph. Parameter Z sets the minimum amplitude of a peak for it to be considered. 
# This function returns the x-value at the highest point of the KDE curve (the dominant peak).
# Optionally, you can pass z to ignore peaks with height below a minimum threshold.
def MD(x, y, z=None):
    # Convert to numpy arrays for consistent indexing
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

        # Remove NaNs (keep paired values)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]

    if x_arr.size == 0:
        raise ValueError("MD(): x and y contain no finite values.")

    # If a minimum peak height is supplied, pick the highest peak above that threshold
    if z is not None:
        peaks, props = find_peaks(y_arr, height=z)
        if len(peaks) > 0:
            best_peak = int(peaks[np.argmax(props["peak_heights"])])
            return float(x_arr[best_peak])

    # Default behaviour: return x at the global maximum of the KDE curve
    best_idx = int(np.nanargmax(y_arr))
    return float(x_arr[best_idx])


# Calculate a peak-specific IQR by isolating the data around the dominant KDE peak.
# Returns a tuple: (peak_x, Q1_peak, Q3_peak)
# - df, col: original data
# - z: minimum KDE peak height threshold
# The peak window is estimated using the KDE full-width-at-half-maximum (FWHM).
def iqr_one_peak(df, col, z):
    # --- 1) Generate KDE for the column ---
    kde_df = KDE(df, col)
    x_kde = np.asarray(kde_df['x'])
    y_kde = np.asarray(kde_df['y'])

    # Remove NaNs from KDE curve (paired)
    mask = np.isfinite(x_kde) & np.isfinite(y_kde)
    x_kde = x_kde[mask]
    y_kde = y_kde[mask]

    if x_kde.size == 0:
        return "No KDE values found (empty after removing NaNs)."

    # --- 2) Select the dominant KDE peak using the existing MD() function ---
    # MD returns the x-location of the dominant peak (optionally above a minimum height threshold z)
    peak_x = MD(x_kde, y_kde, z=z)

    # Convert the peak_x location back to the nearest index for peak_widths()
    peak = int(np.nanargmin(np.abs(x_kde - peak_x)))

    # --- 3) Estimate a window around the dominant peak using FWHM ---
    # peak_widths returns left/right positions in index-space; convert to x using interpolation
    widths, _, left_ips, right_ips = peak_widths(y_kde, [peak], rel_height=0.5)
    left_i = float(left_ips[0])
    right_i = float(right_ips[0])

    idx = np.arange(len(x_kde), dtype=float)
    left_x = float(np.interp(left_i, idx, x_kde))
    right_x = float(np.interp(right_i, idx, x_kde))

    # Ensure bounds are ordered
    lo, hi = (left_x, right_x) if left_x <= right_x else (right_x, left_x)

    # --- 4) Subset the ORIGINAL data around that peak window and compute peak-specific IQR ---
    data = np.asarray(df[col])
    data = data[np.isfinite(data)]

    peak_data = data[(data >= lo) & (data <= hi)]
    if peak_data.size == 0:
        return "No original data points fell within the dominant peak window."

    Q1 = float(np.percentile(peak_data, 25))
    Q3 = float(np.percentile(peak_data, 75))

    return peak_x, Q1, Q3

# Example usage:
# peak_x, Q1, Q3 = iqr_one_peak(df, 'data', z=0.02)