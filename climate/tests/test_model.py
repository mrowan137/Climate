from climate.model import *

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
        res = log_post_tcm([1e9, 0., 0., 0., 0.], x, y, yerr)
        assert res == -np.Inf

        
if __name__ == '__main__':
    unittest.main()
