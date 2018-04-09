import numpy as np
from data_io import *
import pySCM.SimpleClimateModel as SCM


"""
Traditional climate model
"""

def trad_climate_model():
    fileload = get_example_data_file_path('SimpleClimateModelParameterFile.txt', data_dir='pySCM')
    model = SCM.SimpleClimateModel(fileload)
    model.runModel()
