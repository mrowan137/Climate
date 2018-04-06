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

    Returns the value of the uniform prior at position x for range xmin to xmax

    """



    def __call__(self, x):

        lower, upper = self.params

        return np.where(

            np.logical_and(x <= upper, x >= lower), -log(upper - lower), -inf)





class LogJefferysPrior(Prior):

    """

    Returns the value of the Jefferys prior at position x for range xmin to xmax

    """



    def __call__(self, x):

        lower, upper = self.params

        return np.where(

            np.logical_and(x <= upper, x >= lower),

            -log(x) - log(log(upper / lower)), -inf)

        

class LogNormalPrior(Prior):

    """

    Returns the value of the Jefferys prior at position x for range xmin to xmax

    """



    def __call__(self, x):

        mu, sig = self.params

        return -(x-mu)**2/(2*sig**2)