import os
import sys
import numpy as np
import pandas as pd
import glob
# Following the example file


def get_example_data_file_path(filename, data_dir='data'):
    # __file__ is the location of the source file currently in use (so
    # in this case io.py). We can use it as base path to construct
    # other paths from that should end up correct on other machines or
    # when the package is installed
    start = os.path.abspath(__file__)
    start_dir = os.path.dirname(start)
    data_dir = os.path.join(start_dir, data_dir)
    return os.path.join(data_dir, filename)


def load_data_temp(data_file):
    """Import temperature data from location data_file.

    Args:
        data_file (str): Location of data to be imported.

    Returns:
        data: Pandas data frame.

    """
    data = pd.read_csv(data_file, sep='\s+', header=None)
    data.columns = ["year", "month", "monthly_anomaly", "monthly_anomaly_unc",
                    "annual_anomaly", "annual_anomaly_unc",
                    "five_year_anomaly", "five_year_anomaly_unc",
                    "ten_year_anomaly", "ten_year_anomaly_unc",
                    "twenty_year_anomaly", "twenty_year_anomaly_unc"]
    return data


def load_scm_temp(data_file):
    """Import temperature output by simple climate model from data_file.

    Args:
        data_file (str): Location of data to be imported.

    Returns:
        data: Pandas data frame.

    """
    data = pd.read_csv(data_file, sep='\s+', header=None)
    data.columns = ["year", "temp"]
    return data


def load_data_flare(data_file, verbose=False):
    """Import flare index data

    Args:
        data_file (str): Location of data to be imported.
        verbose (bool): Set to 'True' to see loaded files.

    Returns:
        data: Pandas data frame.
    """
    files = sorted(glob.glob(data_file))
    if verbose:
        print('Flare data computed from the following files:')
        for f in files:
            print(str(f))

    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    sz = len(files)
    years = np.zeros(sz)
    flares_index_mean = np.zeros(sz)
    flares_index_unc = np.zeros(sz)
    for i,f in enumerate(files):
        year = pd.read_csv(f, skiprows=3, nrows=1, header=None)[0][0]
        data = pd.read_csv(f, skiprows=7, nrows=31, sep = '\s+', header=None)
        data.columns=["Day"] + months

        years[i] = year
        flares_index_mean[i] = data[months].stack().mean()
        flares_index_unc[i] = data.stack().std()
        

    all_data = pd.DataFrame({'years':years,
                             'flares_index_mean':flares_index_mean,
                             'flares_index_unc':flares_index_unc})
    
    return all_data
            
            
    

