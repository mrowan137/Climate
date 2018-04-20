from climate.inference import model
from climate.data_io import *

import pandas as  pd
import numpy.random as random
import unittest
from unittest import TestCase


class TestModel(TestCase):
    def test_ModifiedSimpleClimateModel(self):
        data = load_data_temp(get_example_data_file_path(
            'global_surface_temp_seaice_air_infer.txt'))
        x, y, yerr = data['year'], data['monthly_anomaly'], data['monthly_anomaly_unc']

        # downsample data to match emissions files
        x = x.values[0::12]
        y = y.values[0::12]
        yerr = yerr.values[0::12]

        # Generate dataset from model of known parameters
        # Create instance of ModifiedSimpleClimateModel
        SCM_generate = model.ModifiedSimpleClimateModel(x, y, yerr)

        # Set priors
        prior_type = ['uniform' for i in range(5) ]
        prior_param1 = [-0.5, 0.7, 0.7, 0.7, 0.7]
        prior_param2 = [0.5, 1.3, 1.3, 1.3, 1.3]
        SCM_generate.set_priors(prior_type, prior_param1, prior_param2)
        
        # Generate dataset
        x_generated, y_generated = SCM_generate([ 0.0, 1.0, 1.0, 1.0, 1.0])
        wh_past = np.where(x_generated <= 2018)
        x_generated = x_generated[wh_past]
        y_generated = y_generated[wh_past]
        y_generated = y_generated + [ random.normal(scale=0.2) for i in range(len(y_generated)) ]
        yerr_generated = np.ones(len(x)) * 0.2
        

        # Create new model and fit it to the dataset        
        SCM_test = model.ModifiedSimpleClimateModel(x, y_generated, yerr_generated)
        
        # Set priors
        prior_type = ['uniform' for i in range(5) ]
        prior_param1 = [-0.5, 0.7, 0.7, 0.7, 0.7]
        prior_param2 = [0.5, 1.3, 1.3, 1.3, 1.3]
        SCM_test.set_priors(prior_type, prior_param1, prior_param2)

        # Run MCMC with initial guess far from true value
        SCM_test.run_MCMC( param_guess=[0.3, 1.2, 1.1, 0.71, 0.8], nwalkers=10, nsteps=400)
        SCM_test.show_results(burnin=100)
        
        print('Check that parameter result is consisitent with 0.0 shift and 1.0',
                ' normalization on all emissions curves')


    def test_BasicCloudSeedingModel(self):
        data = load_data_temp(get_example_data_file_path(
            'global_surface_temp_seaice_air_infer.txt'))
        x, y, yerr = data['year'], data['monthly_anomaly'], data['monthly_anomaly_unc']

        # downsample data to match emissions files
        x = x.values[0::12]
        y = y.values[0::12]
        yerr = yerr.values[0::12]
        
        # Get solar data
        fileload = get_example_data_file_path(
            'flare-index_total_*.txt', data_dir='data/flares')
        data_flares = load_data_flare(fileload, verbose=0)
        
        # Get np.arrays for data series
        years_flares = data_flares['years'].values
        flares = data_flares['flares_index_mean'].values
        flares_unc = data_flares['flares_index_unc'].values

        # Generate dataset from model of known parameters
        # Create instance of ModifiedSimpleClimateModel
        CSM_generate = model.BasicCloudSeedingModel(x, y, yerr, years_flares, flares, flares_unc)

        # Set priors
        prior_type = ['uniform' for i in range(2) ]
        prior_param1 = [-0.5, 0.0]
        prior_param2 = [ 0.5, 5.0]
        CSM_generate.set_priors(prior_type, prior_param1, prior_param2)

        # Generate dataset
        x_generated, y_generated = CSM_generate([ 0.35, 1.0])
        wh_past = np.where(x_generated <= 2018)
        x_generated = x_generated[wh_past]
        y_generated = y_generated[wh_past]
        y_generated = y_generated + [ random.normal(scale=0.2) for i in range(len(y_generated)) ]
        yerr_generated = np.ones(len(x_generated)) * 0.2

        
        # Create new model and fit it to the dataset        
        CSM_test = model.BasicCloudSeedingModel(x_generated, y_generated, yerr_generated, years_flares, flares, flares_unc)

        # Set priors
        prior_type = ['uniform' for i in range(2) ]
        prior_param1 = [-0.5, 0.0]
        prior_param2 = [ 0.5, 2.5]
        CSM_test.set_priors(prior_type, prior_param1, prior_param2)
        
        
        # Run MCMC with initial guess far from true value
        CSM_test.run_MCMC( param_guess=[0.05, 2.2], nwalkers=10, nsteps=400)
        CSM_test.show_results(burnin=100)

        print('Check that parameter result is consisitent with 0.35, 1.0') 
    

if __name__ == '__main__':
    unittest.main()
