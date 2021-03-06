from climate.inference import prior#.prior import LogUniformPrior, LogJefferysPrior, LogNormalPrior



import numpy as np



import unittest

from unittest import TestCase





class TestPriors(TestCase):

    def test_uniform(self):

        assert np.allclose(np.exp(prior.LogUniformPrior(3, 5)(4)), .5)



    def test_jefferys(self):

        assert np.allclose(np.exp(prior.LogJefferysPrior(10, 1000)(100)),

                           0.0021714724095162588)

        

    def test_normal(self):
    
        assert np.allclose(prior.LogGaussianPrior(0,1)(1), -1.418, atol=1e-3)



if __name__ == '__main__':

    unittest.main()
