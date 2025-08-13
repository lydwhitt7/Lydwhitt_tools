import pandas as pd
import numpy as np
from scipy.signal import find_peaks
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
def MD(x, y, z):
    MD_FP,_ = find_peaks(y, height = z)
    MD_MD = (x)[MD_FP[0]]
    return MD_MD


#    Calculate the IQR for the entire data based on the peak in the KDE. It returns a tuple containing the peak location (peak_x), Q1 and Q3 values for the entire dataset.
def iqr_one_peak(df, col, z):
    # Generate KDE using your custom function
    kde_df = KDE(df, col)

    # Extract x_kde and y_kde from the DataFrame
    x_kde = kde_df['x']
    y_kde = kde_df['y']

    # Find peaks in the KDE
    peaks, _ = find_peaks(y_kde, height = z)

    # Process the first peak if it exists
    if len(peaks) >= 1:
        peak = peaks[0]
        peak_x = x_kde.iloc[peak]

        # Calculate the Q1 and Q3 for the entire dataset
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)

        return peak_x, Q1, Q3
    else:
        return "No peaks found in the KDE."

# Example usage:
# peak_x, Q1, Q3 = iqr_one_peak(df, 'data')