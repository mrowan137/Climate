import sys
import numpy as np
import emcee
import seaborn as sns
from scipy.interpolate import interp1d
from climate.data_io import *
from climate.pySCM.SimpleClimateModel import *
from climate.inference import prior


class Model:
    """
    Base class for generic model

    Instantiator Args:
        ndim: number of parameters for model
        x (array): Independent variable data array
        y (array): Dependent variable data array
        yerr (array): Uncertainty on y data series
    """

    def __init__(self, ndim, x, y, yerr):
        """
        Initialize global variables of generic model
    
        Args:
            x (array): Independent variable
            y (array): Dependent variable
            yerr (array): Uncertainty on y
        """

        # Set global variables
        self.ndim = ndim
        self.x = x
        self.y = y
        self.yerr = yerr
        # Create an array of ndim Priors
        self.priors = [ prior.Prior(0,1) for i in range(self.ndim) ]


    def set_priors(self, prior_type, param1, param2):
        """
        Setter for the priors on all model parameter

        Args:
            prior_type (array): type of prior to set (uniform, gaussian, jefferys)
            param1 (array): 1st parameter of prior (either lower bound or mean)
            param2 (array): 2nd parameter of prior (either upper bound or std dev)
        """

        # Assign Prior object depending on requested type 
        for i in range(self.ndim):
            if (prior_type[i] == 'uniform'):
                self.priors[i] = prior.LogUniformPrior(param1[i], param2[i])
            elif (prior_type[i] == 'gaussian'):
                self.priors[i] = prior.LogGaussianPrior(param1[i], param2[i])
            elif (prior_type[i] == 'jefferys'):
                self.priors[i] = prior.LogJefferysPrior(param1[i], param2[i])
            else:
                print("Invalid prior option. Modify inputs and try again.")


    def log_prior(self, params):
        """
        Computes log of prior probability distribution

        Args:
            params (array): Model parameters

        Returns:
            log of prior of all parameters
        """

        priors_eval = [ self.priors[i](params[i]) for i in range(self.ndim) ]
        return np.sum(priors_eval)    

    
    def log_post(self, params):
        """
        Returns log of posterior probability distribution for traditional climate model

        Args:
            params (array): Parameters for the traditional (simple) climate model
       
        Returns:
            Log of posterior distribution
        """

        return self.log_prior(params) + self.log_lh(params)


    def run_MCMC(self, param_guess, nwalkers, nsteps):
        """
        Samples the posterior distribution via the affine-invariant ensemble 
        sampling algorithm; plots are output to diagnose burn-in time; best-fit
        parameters are printed; best-fit line is overplotted on data, with errors.
    
        Args:
            param_guess (array): Initial guess for parameters. Can have length < 5
            nwalkers (int): Number of walkers for affine-invariant ensemble sampling;
                            must be an even number
            nsteps (int): Number of timesteps for which to run the algorithm


        Returns:
            Samples (array): Trajectories of the walkers through parameter spaces.
                             This array has dimension (nwalkers) x (nsteps) x (ndim)
        """
        
        # Randomize starting positions of walkers around initial guess
        starting_positions = [
            param_guess + 1e-4 * np.random.randn(self.ndim) for i in range(nwalkers)
        ]

        # Set up the sampler object
        sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self.log_post, args=())

        # Progress bar
        width = 80
        for i, result in enumerate(sampler.sample(starting_positions, iterations=nsteps)):
            n = int((width+1) * float(i) / nsteps)
            if (i == 0):
                print('Progress: ')
                
            print("\r[{0}{1}]".format('#' * n, ' ' * (width - n)), end='')

        print(os.linesep)
        # Run the sampler
        #sampler.run_mcmc(starting_positions, nsteps)

        # return the samples for later output
        self.samples = sampler.flatchain
        return self.samples


    def show_results(self, burnin):
        """
        Displays results from self.sample
    
        Args:
            burnin (int): Burn in time to trim the samples
        """  
   
        # Plot and check for burn in time
        fig, ax = plt.subplots(self.ndim, figsize=(10,10))
        plt.subplots_adjust(hspace=0.5)
        for i in range(self.ndim):
            ax[i].set(ylabel="Parameter %d"%i)
    
        for i in range(self.ndim):
            sns.distplot(self.samples[:,i], ax=ax[i])
        plt.show()

        # Store the samples in a dataframe
        index = [i for i in range(len(self.samples[:,0]))]
        columns = ['p'+str(i) for i in range(self.ndim)]
        samples_df = pd.DataFrame(self.samples, index=index, columns=columns)

        # Compute and print the MAP values
        q = samples_df.quantile([0.16, 0.50, 0.84], axis=0)
        for i in range(self.ndim):
            print("Param {:.0f} = {:.6f} + {:.6f} - {:.6f}".format( i, 
            q['p'+str(i)][0.50], q['p'+str(i)][0.84] - q['p'+str(i)][0.50], q['p'+str(i)][0.50] - q['p'+str(i)][0.16]))

        # Best-fit params
        self.bestfit_params = [ q['p'+str(i)][0.50] for i in range(self.ndim) ]
        
        # Evaluate the model with best fit params
        x_model, y_model = self.__call__(self.bestfit_params)

        # Select years from data and model to compare
        wh_model = np.where((x_model >= np.min(self.x)) & (x_model <= np.max(self.x)))
        x_model = x_model[wh_model]
        y_model = y_model[wh_model]
        
        # Plot the best-fit line, and data
        plt.figure(figsize=(10,8))
        plt.errorbar(self.x,self.y, self.yerr,  linestyle='none')
        plt.scatter(self.x, self.y, c='k',zorder=5,s=20, label='data')
        plt.plot(x_model, y_model, label='best fit')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim([np.max([self.x[0], x_model[0]]),
                  np.min([self.x[-1], x_model[-1]])])
        plt.title('Model Fit to Data');
        plt.legend()


    def get_samples(self):
        """
        Getter that returns the samples
        """

        return self.samples



class ModifiedSimpleClimateModel(Model):
    """
    Modified Simple Climate Model Class
    """
   
    def __init__(self, ndim, x, y, yerr):
        """
        Calls constructor for Model base class
        """        
        super().__init__(ndim, x, y, yerr)


    def __call__(self, *params):
        """
        Evaluate the model for input parameters

        Returns x and y series for model prediction
        """

        # Use global parameters (assume set by show_results) if none input
        #if (len(params) == 0):
        #    params = self.params

        if isinstance(params, tuple):
            params = params[0]

        # Run simple climate model
        fileload = get_example_data_file_path(
            'SimpleClimateModelParameterFile.txt', data_dir='pySCM')
        model_best = SimpleClimateModel(
            fileload, [params[i] for i in range(len(params))])
        model_best.runModel()
    
        # Read in temperature change output (from simple climate model)
        # .dat file format
        #fileload = get_example_data_file_path(
        #    'TempChange.dat', data_dir='trad_climate_model_output')

        # .json file format
        fileload = get_example_data_file_path(
            'TempChange.json', data_dir='trad_climate_model_output')

        #data_scm_best = load_scm_temp(fileload)
        data_scm_best = loadj_scm_temp(fileload)
        
        # Add the shift
        x_model, y_model = data_scm_best.year, data_scm_best.temp + params[0]

        return x_model.iloc[:].values, y_model.iloc[:].values


    def log_lh(self, params):
        """
        Computes log of Gaussian likelihood function

        Args:
            params (array): Parameters for the simple climate model,
	    contain subset (in order) of the following parameters:
                -shift: Overall shift of the temperature curve output by SCM
                -CO2_norm: Normalization for CO2 emissions
                -CH4_norm:       "        "  CH4     "
                -N2O_norm:       "        "  N2O     " 
                -SOx_norm:       "        "  SOx     "

        Returns:
            chisq: Sum of ((y_data - y_model)/y_err)**2 
        """
        shift, CO2_norm, CH4_norm, N2O_norm, SOx_norm = params
    
        # Run simple climate model and save output
        fileload = get_example_data_file_path(
            'SimpleClimateModelParameterFile.txt', data_dir='pySCM')
        model = SimpleClimateModel(fileload, [shift, CO2_norm, CH4_norm, N2O_norm, SOx_norm])
        model.runModel()

        # Read in temperature change output (from simple climate model)
        # .dat format
        #fileload = get_example_data_file_path(
        #    'TempChange.dat', data_dir='trad_climate_model_output')

        # .json format
        fileload = get_example_data_file_path(
            'TempChange.json', data_dir='trad_climate_model_output')
        data_scm = loadj_scm_temp(fileload)
        x_scm, y_scm = data_scm.year, data_scm.temp

        # Select years from data and scm to compare
        wh_scm = np.where((x_scm >= np.min(self.x)) & (x_scm <= np.max(self.x)))
        x_scm = x_scm.iloc[:].values[wh_scm]
        y_scm = y_scm.iloc[:].values[wh_scm] + shift
    
        # Compute chisq and return
        chisq = np.sum(((self.y - y_scm)/self.yerr)**2)
        constant = np.sum(np.log(1/np.sqrt(2.0*np.pi*self.yerr**2)))
        return constant - 0.5*chisq


class BasicCloudSeedingModel(Model):
    """
    Basic model for cloud seeding. DT = p0 * a_{sun}(t-Dt)
    """

    def __init__(self, ndim, x, y, yerr, solar_x, solar_y):
        """
        Initialize global variables of basic cloud seeding model
    
        Args:
            x (array): Independent variable
            y (array): Dependent variable
            yerr (array): Uncertainty on y
            solar_x (array): Data on solar activity (x)
            solar_y (array): Data on solar activity (y)
        """
        
        # Set global variables
        self.ndim = ndim
        self.x = x
        self.y = y
        self.yerr = yerr
        self.solar_x = solar_x
        self.solar_y = solar_y
        self.solar_f = interp1d(solar_x, solar_y, kind='cubic')
        self.priors = [ prior.Prior(0,1) for i in range(self.ndim) ]


    def __call__(self, *params):
        """
        Evaluate the model for given params

        Returns x and y series for model prediction
        """

        # Use global parameters (assume set by run_mcmc) if none input
        #if (len(params) == 0):
        #    params = self.params
        if isinstance(params, tuple):
            alpha, dt = params[0]
        else:
            alpha, dt = params

        # Select years from data and seeding model to compare
        wh_sm = np.where((self.x >= np.min(self.solar_x) + dt)
                         & (self.x <= np.max(self.solar_x) + dt))
        x_model = self.x[wh_sm]
        y_model = alpha * self.solar_f( x_model-dt )
        
        return x_model, y_model

    def log_lh(self, params):
        """
        Computes log of Gaussian likelihood function

        Args:
            params (array): Parameters for the simple climate model,
            contain subset (in order) of the following parameters:
                -alpha: proportionality constant
                -dt: time delay for cooling to take effect

        Returns:
            chisq: Sum of ((y_data - y_model)/y_err)**2 
        """

        # Evaluate model at given point
        x_model, y_model = self.__call__(params)
       
        # Get range over which to compare data and model 
        x_min = np.max([np.min(x_model), np.min(self.x)])
        x_max = np.min([np.max(x_model), np.max(self.x)])
        
        # Select the model and data values
        wh_data = np.where((self.x <= x_max) & (self.x >= x_min))
        wh_model = np.where((x_model <= x_max) & (x_model >= x_min))

        y_data = self.y[wh_data]
        yerr_data = self.yerr[wh_data]
        y_model = y_model[wh_model]
        
        # Compute chisq and return
        chisq = np.sum(((y_data - y_model)/yerr_data)**2)
        constant = np.sum(np.log(1/np.sqrt(2.0*np.pi*yerr_data**2)))
        return constant - 0.5*chisq


class CombinedModel(Model):
    """
    Combined model class; for parent models with n1, n2 model parameters, the 
    combined model has a total of n1 + n2 parameters

    Instantiator Args:
        Model1 (Model): 1st model 
        Model2 (Model): 2nd model, to be combined with first
    """

    def __init__(self, Model1, Model2):
        """
        Initialize global variables of combined model

        Args:
            Model1 (Model): 1st model
            Model2 (Model): 2nd model, to be combined with first
        """
        
        # Combined model has total number of parameters from parent Models
        self.ndim = Model1.ndim + Model2.ndim

        # Create an array of ndim Priors
        self.priors = [ prior.Prior(0,1) for i in range(self.ndim) ]

        # Check that parent Models are compatible (check that raw data are the same)
        if ((Model1.x != Model2.x).any()
            or (Model1.y != Model2.y).any()
            or (Model1.yerr != Model2.yerr)).any():
            raise ValueError('Input data for Models do not agree! Try with same data')
        else:
            self.x = Model1.x
            self.y = Model1.y
            self.yerr = Model1.yerr
            self.Model1 = Model1
            self.Model2 = Model2


    def __call__(self, *params):
        """
        Evaluate the model for input parameters

        Returns x and y series of combined model for model prediction
        """
        
        params1 = params[0][0:self.Model1.ndim]
        params2 = params[0][self.Model1.ndim:self.ndim]
        x_model1, y_model1 = self.Model1(params1)
        x_model2, y_model2 = self.Model2(params2)
        x_combined, wh1, wh2 = self.inter(x_model1, x_model2)

        # Sum the output of model1, model2
        y_combined = y_model1[wh1] + y_model2[wh2]

        return x_combined, y_combined
        

    def inter(self, arr1, arr2):
        """
        Get intersection of arrays arr1, arr2

        Args:
            arr1 (array): 1st array
            arr2 (array): 2nd array

        Returns: 
            inter (array): intersection of arr1, arr2
            ind1 (array): indices of arr1 which give elements in intersection
            ind2 (array): indices of arr2 which give elements in intersection
        """
        ind1 = np.in1d(arr1, arr2).nonzero()
        ind2 = np.in1d(arr2, arr1).nonzero()
        inter = arr1[ind1]
        return inter, ind1, ind2

    
    def log_lh(self, params):
        """
        Computes log of Gaussian likelihood function

        Args:
            params (array): parameters of the two Parents, Model1 and Model2

        Returns:
            chisq: Sum of ((y_data - y_model)/y_err)**2 
        """

        # Evaluate model at given point
        x_model, y_model = self.__call__(params)
       
        # Get range over which to compare data and model 
        x_min = np.max([np.min(x_model), np.min(self.x)])
        x_max = np.min([np.max(x_model), np.max(self.x)])
        
        # Select the model and data values
        wh_data = np.where((self.x <= x_max) & (self.x >= x_min))
        wh_model = np.where((x_model <= x_max) & (x_model >= x_min))

        y_data = self.y[wh_data]
        yerr_data = self.yerr[wh_data]
        y_model = y_model[wh_model]
        
        # Compute chisq and return
        chisq = np.sum(((y_data - y_model)/yerr_data)**2)
        constant = np.sum(np.log(1/np.sqrt(2.0*np.pi*yerr_data**2)))
        return constant - 0.5*chisq
        
    