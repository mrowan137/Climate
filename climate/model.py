import numpy as np
from io import *
import pySCM.SimpleClimateModel as SCM


"""
Traditional climate model
"""
param_file = load_data(
    get_example_data_file_path('SimpleClimateModelParameterFile.txt',
                               data_dir='pySCM'))
#M = SCM.SimpleClimateModel('/pySCM/')

