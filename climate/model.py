import numpy as np
from climate.data_io import *
from climate.pySCM.SimpleClimateModel import *


"""
Traditional climate model
"""

def trad_climate_model():
    fileload = get_example_data_file_path('SimpleClimateModelParameterFile.txt', data_dir='pySCM')
    model = SimpleClimateModel(fileload)
    model.runModel()
