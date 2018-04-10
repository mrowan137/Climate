from climate.data_io import *#get_example_data_file_path, load_data

import pandas as  pd

import unittest
from unittest import TestCase


class TestIo(TestCase):
    def test_data_io(self):
        data = load_data_temp(get_example_data_file_path(
            'global_surface_temp_seaice_air_infer.txt'))
        assert data.year[0] == 1850

        data_tcm = load_tcm_temp(get_example_data_file_path(
            'TempChange.dat', data_dir='trad_climate_model_output'))
        print(data_tcm.year[0])
        assert data_tcm.year[0] == 1850
        
if __name__ == '__main__':
    unittest.main()
