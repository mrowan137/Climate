import os
import pandas as pd
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
