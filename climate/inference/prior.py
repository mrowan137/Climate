import numpy as np
from numpy import log, inf

class Prior:
    """
    Prior Base class. This can store the shape parameters in the object instance 
    then be used as a function 
    """

    def __init__(self, *shapeparams):
        self.params = shapeparams


class LogUniformPrior(Prior):
    """
    Returns log of the value of the uniform prior at position x
    """

    def __call__(self, x):
        lower, upper = self.params
        if (lower <= x <= upper):
            return log(1 / (upper-lower))
        else:
            return -inf


class LogJefferysPrior(Prior):
    """
    Returns log of the value of the Jefferys prior at position x
    """

    def __call__(self, x):
        lower, upper = self.params
        if (lower <= x <= upper):
            return -log(x) - log( log(upper/lower) )
        else:
            return -inf


class LogGaussianPrior(Prior):
    """
    Returns log of the value of the Gaussian prior at position x
    """

    def __call__(self, x):
        mu, sig = self.params
        return -log(2*np.pi*sig**2)/2 - (x-mu)**2/(2*sig**2)
