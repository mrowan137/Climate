from climate.inference import model
from climate.data_io import *
import pandas as  pd
import numpy.random as random
import unittest
from unittest import TestCase


class TestModel(TestCase):
    """
    TestModel class with TestCase as base class. Used for testing models in model.py
    """

    def test_ModifiedSimpleClimateModel(self):
        """
        Test function for the ModifiedSimpleClimateModel class
        """
    
        # Load emissions data
        fileload = get_example_data_file_path('EmissionsForSCM.dat', data_dir='pySCM')
        ems = load_data_emissions(fileload)

        # Load temperature
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
        prior_type = ['uniform' for i in range(SCM_generate.ndim) ]
        prior_param1 = [-0.5]*SCM_generate.ndim
        prior_param2 = [0.5]*SCM_generate.ndim
        SCM_generate.set_priors(prior_type, prior_param1, prior_param2)
        
        # Generate dataset
        x_generated, y_generated = SCM_generate([0.0]
                                                +[el for el in ems['CO2']]
                                                +[el for el in ems['N2O']]
                                                +[el for el in ems['CH4']]
                                                +[el for el in ems['SOx']])
        wh_past = np.where((x_generated <= 2018))
        x_generated = x_generated[wh_past]
        y_generated = y_generated[wh_past]
        y_generated = y_generated + [ random.normal(scale=0.2) for i in range(len(y_generated)) ]
        yerr_generated = np.ones(len(x)) * 0.2
        

        # Create new model and fit it to the dataset        
        SCM_test = model.ModifiedSimpleClimateModel(x, y_generated, yerr_generated)
        
        # Set priors
        prior_type = ['uniform' for i in range(SCM_test.ndim) ]
        prior_param1 = [-0.5]*SCM_test.ndim
        prior_param2 = [0.5]*SCM_test.ndim
        SCM_test.set_priors(prior_type, prior_param1, prior_param2)

        # Run MCMC with initial guess far from true value
        param_guess = ([0.3]
                       +[el for el in ems['CO2']]
                       +[el for el in ems['N2O']]
                       +[el for el in ems['CH4']]
                       +[el for el in ems['SOx']])
        SCM_test.run_MCMC( param_guess=[0.3],
                           nwalkers=2*SCM_test.ndim, nsteps=10)
        SCM_test.show_results(burnin=1,params_to_plot=[0])
        
        print('Check that parameter result is consisitent with 0.0 shift')


    def test_BasicCloudSeedingModel(self):
        """
        Test function for the BasicCloudSeedingModel class
        """

        data = load_data_temp(get_example_data_file_path(
            'global_surface_temp_seaice_air_infer.txt'))
        x, y, yerr = data['year'], data['monthly_anomaly'], data['monthly_anomaly_unc']

        # downsample data to match emissions files
        x = x.values[0::12]
        y = y.values[0::12]
        yerr = yerr.values[0::12]
        
        # Get solar data
        fileload = get_example_data_file_path(
            'SN_y_tot_V2.0.txt', data_dir='data/sunspots')
        data_sunspots = load_data_y_sunspot(fileload)
        
        # Get np.arrays for data series
        years_sunspots = np.floor(data_sunspots['year'].values[118::])
        sunspots = data_sunspots['yearly_sunspot_number'].values[118::]
        sunspots_unc = data_sunspots['stdev'].values[118::]
        
        # Generate dataset from model of known parameters
        # Create instance of ModifiedSimpleClimateModel with no interpolation
        CSM_generate = model.BasicCloudSeedingModel(x, y, yerr, years_sunspots, sunspots, sunspots_unc,1)

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
        CSM_test = model.BasicCloudSeedingModel(x_generated, y_generated, yerr_generated, years_sunspots, sunspots, sunspots_unc, 1)

        # Set priors
        prior_type = ['uniform' for i in range(2) ]
        prior_param1 = [-0.5, 0.0]
        prior_param2 = [ 0.5, 2.5]
        CSM_test.set_priors(prior_type, prior_param1, prior_param2)
        
        
        # Run MCMC with initial guess far from true value
        CSM_test.run_MCMC( param_guess=[0.05, 2.2], nwalkers=10, nsteps=400)
        CSM_test.show_results(burnin=100,params_to_plot=[0,1])

        print('Check that parameter result is consisitent with 0.35, 1.0') 
    

    def test_GPRInterpolator(self):
        """
        Test function for the GPRInterpolator class
        """

        # Get solar data
        fileload = get_example_data_file_path(
            'SN_y_tot_V2.0.txt', data_dir='data/sunspots')
        data_sunspots = load_data_y_sunspot(fileload)

        # Get np.arrays for data series
        years_sunspots = np.floor(data_sunspots['year'].values[118::])
        sunspots = data_sunspots['yearly_sunspot_number'].values[118::]
        sunspots_unc = data_sunspots['stdev'].values[118::]

        # Generate dataset from model of known parameters
        # Create instance of ModifiedSimpleClimateModel
        GPR_generate = model.GPRInterpolator(years_sunspots, sunspots, sunspots_unc, 5)

        # Set priors
        prior_type = ['uniform' for i in range(12) ]
        prior_param1 = [-100 for i in range(12) ]
        prior_param2 = [ 100 for i in range(12) ]
        GPR_generate.set_priors(prior_type, prior_param1, prior_param2)

        # Generate dataset
        GPR_generate.run_MCMC(nwalkers=24, nsteps=400)
        GPR_generate.show_results(100)

        x_generated, y_generated, yerr_generated = GPR_generate(years_sunspots)

        # Create new model and fit it to the dataset        
        GPR_test = model.GPRInterpolator(x_generated, y_generated, yerr_generated, 1)

        # Set priors
        params = GPR_generate.get_parameters()
        prior_type = ['uniform' for i in range(12) ]
        prior_param1 = params*.5
        prior_param2 = params*1.5
        GPR_test.set_priors(prior_type, prior_param1, prior_param2)


        # Run MCMC with initial guess far from true value
        GPR_test.run_MCMC(nwalkers=24, nsteps=400)
        GPR_test.show_results(burnin=100)

        print('Check that parameter result is consisitent with', params)


if __name__ == '__main__':
    unittest.main()
