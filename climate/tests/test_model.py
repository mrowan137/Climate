from climate.inference import model
from climate.data_io import *

import pandas as  pd

import unittest
from unittest import TestCase


class TestModel(TestCase):
    def test_model(self):
        data = load_data_temp(get_example_data_file_path(
            'global_surface_temp_seaice_air_infer.txt'))
        x, y, yerr = data['year'], data['monthly_anomaly'], data['monthly_anomaly_unc']

        # downsample data to match emissions files
        x = x.values[0::12]
        y = y.values[0::12]
        yerr = yerr.values[0::12]

        # here we expect res to be -inf because the overall temperature shift
        # of 1e9 is outside our prior range
        SCM = model.ModifiedSimpleClimateModel(5, x, y, yerr)
        prior_type = ['uniform' for i in range(5) ]
        prior_param1 = [-0.5, 0.7, 0.7, 0.7, 0.7]
        prior_param2 = [0.5, 1.3, 1.3, 1.3, 1.3]
        SCM.set_priors(prior_type, prior_param1, prior_param2)
        res = SCM.log_post([1e9, 0., 0., 0., 0.])
        print(res)
        assert res == -np.Inf

        
if __name__ == '__main__':
    unittest.main()
