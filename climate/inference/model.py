import sys
import numpy as np
import emcee
import seaborn as sns
from scipy.interpolate import interp1d
from climate.data_io import *
from climate.pySCM.SimpleClimateModel_opt import *
from climate.inference import prior
import george
from george import kernels
import scipy.optimize as op

"""
This module contains several classes that are used to define different climate 
models (listed below):
    -Model(): Base model containing initialization and functions that are common
    to other climate models
    -ModifiedSimpleClimateModel(): A traditional climate model, based on pySCM.
    -BasicCloudSeedingModel(): A model which assumes clouds are responsible for 
    climate trends
    -CombinedModel(): Class which allows for the combination of two previously 
    instantiated models.  In practice, it can be used to `merge` modified pySCM
    model with the cloud seeding model.

Sample usage is shown in tutorial.ipynb
"""

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
            elif (prior_type[i] == 'poisson'):
                self.priors[i] = prior.LogPoissonPrior(param1[i])
            elif (prior_type[i] == 'exponentialdecay'):
                self.priors[i] = prior.LogExponentialDecayPrior(param1[i])
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
            np.array(param_guess) + 1e-2 * np.random.randn(self.ndim) for i in range(nwalkers)
        ]

        # Set up the sampler object
        sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self.log_post, args=())

        # Progress bar
        width = 100
        for i, result in enumerate(sampler.sample(starting_positions, iterations=nsteps)):
            n = int((width+1) * float(i) / nsteps)
            if (i == 0):
                print('Progress: ')
                
            print("\r[{0}{1}]".format('#' * n, ' ' * (width - n)), end='')

        print(os.linesep)

        # return the samples for later output
        self.samples = sampler.flatchain
        self.sampler = sampler
        return self.samples


    def show_results(self, burnin, params_to_plot):
        """
        Displays results from self.sample
    
        Args:
            burnin (int): Burn in time to trim the samples
        """  
        
        # Modify self.samples for burn-in
        self.samples = self.sampler.chain[:, burnin:, :].reshape((-1, self.ndim))

        # Get number walkers and number of iterations
        nwalkers = self.sampler.chain.shape[0]
        nit = self.sampler.chain.shape[1]

        # Plot the traces and marginalized distributions for desired parameters
        fig, ax = plt.subplots(2*len(params_to_plot),
                               figsize=(10,len(params_to_plot)*3.))
        plt.subplots_adjust(hspace=0.5)

        for i,k in enumerate(params_to_plot):
            ax[2*i].set(ylabel="Parameter %d"%k)
            ax[2*i+1].set(ylabel="Parameter %d"%k)
            sns.distplot(self.samples[:,k], ax=ax[2*i])

            for j in range(nwalkers):
                ax[2*i+1].plot(np.linspace(1+burnin,nit,nit-burnin), self.sampler.chain[j,burnin:,i], "b", alpha=0.1)
        plt.show()

        # Store the samples in a dataframe
        index = [i for i in range(len(self.samples[:,0]))]
        columns = ['p'+str(i) for i in range(self.ndim)]
        samples_df = pd.DataFrame(self.samples, index=index, columns=columns)

        # Compute and print the MAP values
        q = samples_df.quantile([0.16, 0.50, 0.84], axis=0)
        for i in params_to_plot:
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
        plt.figure(figsize=(14,8))
        plt.errorbar(self.x,self.y, self.yerr,  linestyle='none')
        plt.scatter(self.x, self.y, c='k',zorder=5,s=20, label='data')
        plt.plot(x_model, y_model, label='best fit')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim([np.max([self.x[0], x_model[0]]),
                  np.min([self.x[-1], x_model[-1]])])
        plt.title('Model Fit to Data');
        plt.legend()
        plt.show()


    def get_samples(self):
        """
        Getter that returns the samples
        """

        return self.samples

        
    def set_parameters(self, params):
        """
        Setter for model parameters

        Args:
            params (array): parameters to set
        """

        self.bestfit_params = params



class ModifiedSimpleClimateModel(Model):
    """
    Modified Simple Climate Model Class
    """
   
    def __init__(self, x, y, yerr):
        """
        Calls constructor for Model base class
        """
        self.fileload_scm = get_example_data_file_path(
            'SimpleClimateModelParameterFile.txt', data_dir='pySCM')
        self.model = SimpleClimateModel(self.fileload_scm)
        
        # number of dimensions for this model is 1 + 4*(number of emissions pts)
        super().__init__(1+4*len(self.model.emissions['CO2']), x, y, yerr)

        # set selection indices for the different gases
        self.sel_CO2 = [1 + i for i in range(len(self.model.emissions['CO2']))]
        self.sel_N2O = [1 + i + self.sel_CO2[-1] for i in range(len(self.model.emissions['N2O']))]
        self.sel_CH4 = [1 + i + self.sel_N2O[-1] for i in range(len(self.model.emissions['CH4']))]
        self.sel_SOx = [1 + i + self.sel_CH4[-1] for i in range(len(self.model.emissions['SOx']))]


    def __call__(self, params):
        """
        Evaluate the model for input parameters

        Returns x and y series for model prediction
        """        
        
        if isinstance(params, tuple):
            params = params[0]

        # Set the gas emissions
        ems_CO2 = np.array(params)[self.sel_CO2]
        ems_N2O = np.array(params)[self.sel_N2O]
        ems_CH4 = np.array(params)[self.sel_CH4]
        ems_SOx = np.array(params)[self.sel_SOx]
        
        # Run simple climate model
        x_model, y_model = self.model.runModel(ems_CO2, ems_N2O, ems_CH4, ems_SOx)

        # Add the temperature shift
        y_model = y_model + params[0]

        return x_model, y_model


    def log_lh(self, params):
        """
        Computes log of Gaussian likelihood function

        Args:
            params (array): Parameters for the simple climate model,
	    contain subset (in order) of the following parameters:
                -shift: Overall shift of the temperature curve output by SCM

        Returns:
            chisq: Sum of ((y_data - y_model)/y_err)**2 
        """

        # Run simple climate model
        x_scm, y_scm = self.__call__(params)

        # Select years from data and scm to compare
        wh_scm = np.where((x_scm >= np.min(self.x)) & (x_scm <= np.max(self.x)))
        x_scm = x_scm[wh_scm]
        y_scm = y_scm[wh_scm]
    
        # Compute chisq and return
        chisq = np.sum(((self.y - y_scm)/self.yerr)**2)
        constant = np.sum(np.log(1/np.sqrt(2.0*np.pi*self.yerr**2)))
        return constant - 0.5*chisq

    

class BasicCloudSeedingModel(Model):
    """
    Basic model for cloud seeding. DT = p0 * a_{sun}(t-Dt)
    """

    def __init__(self, x, y, yerr, solar_x, solar_y, solar_yerr, tlagprec):
        """
        Initialize global variables of basic cloud seeding model
    
        Args:
            x (array): Independent variable
            y (array): Dependent variable
            yerr (array): Uncertainty on y
            solar_x (array): Data on solar activity (x)
            solar_y (array): Data on solar activity (y)
            solar_yerr (array): Uncertainty on solar activity data
            tlagprec (int): factor increase in resolution on t_lag (from 1 year)
        """

        # Set global variables
        self.ndim = 2
        self.x = x
        self.y = y
        self.yerr = yerr
        self.solar_x = solar_x
        self.solar_y = solar_y
        self.solar_yerr = solar_yerr
        self.subdivisions = tlagprec
        self.priors = [ prior.Prior(0,1) for i in range(self.ndim) ]

        # Value of parameters after optimization
        optpar = np.array([ -3.59865672e+03, 7.80363922e+01, 7.47410298e+00, -6.25627006e-01,
                            -1.28588659e+02, 3.63549497e+00, 3.00857323e+00,  5.38667302e+02,
                            -6.95411444e+02, 7.98969761e+00, 1.23748637e+00])

        # Set priors for GPR based off of optimized values       
        prior_type = [ 'uniform' for i in range(11) ]
        prior_param1 = optpar*0.5
        prior_param2 = optpar*1.5
        prior_type[5] = 'gaussian'
        prior_param1[5] = np.log(11.0)
        prior_param2[5] = 1.2 / 11.0
    
        # Initialize GPR, run MCMC, and show results
        self.GPR = GPRInterpolator(solar_x, solar_y, solar_yerr, self.subdivisions)
        self.GPR.set_priors(prior_type, prior_param1, prior_param2)
        print("Running MCMC on GPR of Solar Data")
        self.GPR.run_MCMC(22,400)
        print("Generating Point Estimate of Gaussian Process Regression")
        self.GPR.show_results(100)

    
    def get_model_prediction(self, params):
        """
        Evaluate the model for given params
        
        Args:
            params: can be *args from user or tuple from mcmc
        
        Returns:
             x, y, and yerr series for model prediction
        """

        if isinstance(params, tuple):
            alpha, dt = params[0]
        else:
            alpha, dt = params

        # Use explicit casting to avoid rounding errors
        # Make parameter dt from emcee rounded to the nearest 1/subdivisions
        dt = int(dt*self.subdivisions)
        dt = np.array(dt/self.subdivisions).astype(np.float64)

        self.x = self.x.astype(np.float64)
        self.solar_x = self.solar_x.astype(np.float64)

        # Select years from data and seeding model to compare
        wh_data = np.where((self.x >= np.min((self.solar_x+dt).astype(np.float32)))
                         & (self.x <= np.max((self.solar_x+dt).astype(np.float32))))

        x_model = self.x[wh_data] - dt
        
        # Get GPR prediction
        x_model, y_model, yerr_model = self.GPR(x_model)
        y_model = alpha * y_model

        return x_model + dt, y_model, yerr_model


    def __call__(self, params):
        """
        Evaluate the model for given params

        Args:
            params: can be *args from user or tuple from mcmc
        
        Returns:
             x and y  series for model prediction
        """

        x_model, y_model, yerr_model = self.get_model_prediction(params)

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

        if isinstance(params, tuple):
            alpha, dt = params[0]
        else:
            alpha, dt = params

        # Evaluate model at given point
        x_model, y_model, yerr_model = self.get_model_prediction(params)
        x_model = x_model.astype(np.float64)

        # Select years from data and seeding model to compare
        wh_data = np.where((self.x.astype(np.float32) >= np.min(x_model.astype(np.float32)))
                         & (self.x.astype(np.float32) <= np.max(x_model.astype(np.float32))))

        y_data = self.y[wh_data]
        yerr_data = self.yerr[wh_data]

        # Compute chisq and return
        chisq = np.sum( (y_data - y_model)**2 / (yerr_data**2 + alpha**2*yerr_model**2) )
        constant = np.sum(np.log(1 / np.sqrt(2.0 * np.pi * (yerr_data**2 + alpha**2*yerr_model**2)) ))
        return constant - 0.5*chisq



class GPRInterpolator(Model):
    """
    Gaussian Process Regression for interpolating a data set with variable factor increase
    in domain resolution
    """

    def __init__(self, x, y, yerr, subdivisions):
        """
        Initialize global variables of Gaussian Process Regression Interpolator
    
        Args:
            x (array): Independent variable
            y (array): Dependent variable
            yerr (array): Uncertainty on y
            subdivisions: The number of subdivisions between data points
        """

        # Define kernels
        kernel_expsq = 38**2 * kernels.ExpSquaredKernel(metric=10**2)
        kernel_periodic = 150**2 * kernels.ExpSquaredKernel(2**2) * kernels.ExpSine2Kernel(gamma=0.05, log_period=np.log(11))
        kernel_poly = 5**2 * kernels.RationalQuadraticKernel(log_alpha=np.log(.78), metric=1.2**2)
        kernel_extra = 5**2 * kernels.ExpSquaredKernel(1.6**2)
        kernel = kernel_expsq + kernel_periodic + kernel_poly + kernel_extra

        # Create GP object
        self.gp = george.GP(kernel, mean=np.mean(y), fit_mean=False)
        self.gp.compute(x, yerr)

        # Set global variables
        self.ndim = len(self.gp)
        self.x = x
        self.y = y
        self.yerr = yerr
        self.subdivisions = subdivisions
        self.priors = [ prior.Prior(0,1) for i in range(self.ndim) ]
        self.x_predict = np.linspace(min(self.x), max(self.x), subdivisions*(len(self.x)-1) + 1 )


    def run_MCMC(self, nwalkers, nsteps):
        """
        Samples the posterior distribution via the affine-invariant ensemble 
        sampling algorithm; plots are output to diagnose burn-in time; best-fit
        parameters are printed; best-fit line is overplotted on data, with errors.
    
        Args:
            nwalkers (int): Number of walkers for affine-invariant ensemble sampling;
                            must be an even number
            nsteps (int): Number of timesteps for which to run the algorithm


        Returns:
            Samples (array): Trajectories of the walkers through parameter spaces.
                             This array has dimension (nwalkers) x (nsteps) x (ndim)
        """

        # Define objective function to optimize (log-likelihood)
        def nll(p):
            self.gp.set_parameter_vector(p)
            ll = self.gp.lnlikelihood(self.y, quiet=True)
            return -ll if np.isfinite(ll) else 1e25

        # Define gradient of objective function
        def grad_nll(p):
            self.gp.set_parameter_vector(p)
            return -self.gp.grad_lnlikelihood(self.y, quiet=True)

        # Run optimization routine
        p0 = self.gp.get_parameter_vector()
        results = op.minimize(nll,p0, jac=grad_nll, method="BFGS")

        # Update kernel
        self.gp.set_parameter_vector(results.x)

        # Call Model.run_MCMC
        super().run_MCMC(self.gp.get_parameter_vector(), nwalkers, nsteps)


    def __call__(self, x_in):
        """
        Returns x, y, and yerr from GPR evaluated at x_in  
        """

        self.x_predict = self.x_predict.astype(np.float64)
        x_in = x_in.astype(np.float64)

        # Explicitly declare min/max to avoid binary conversion rounding error 
        x_in_min = np.min(self.x_predict.astype(np.float32))
        x_in_max = np.max(self.x_predict.astype(np.float32))
        wh_x_in = np.where((x_in.astype(np.float32) >= x_in_min) & (x_in.astype(np.float32) <= x_in_max))
        x_in_sub = x_in[wh_x_in]

        x_out_min = np.min(x_in_sub.astype(np.float32))
        x_out_max = np.max(x_in_sub.astype(np.float32))
        wh_x_out = np.where((self.x_predict.astype(np.float32) >= x_out_min) & (self.x_predict.astype(np.float32) <= x_out_max))
        x_out = self.x_predict[wh_x_out]

        # Downsample
        x_out = x_out[::self.subdivisions]
        y_out = self.y_predict[wh_x_out][::self.subdivisions]
        yerr_out = self.yerr_predict[wh_x_out][::self.subdivisions]

        return x_out, y_out, yerr_out


    def log_lh(self, params):
        """
        Computes log likelihood for GPR using GP.lnlikelihood
        Args:
            params (array): Parameters on which to compute log likelihood
        """

        # Update the kernel parameters and compute log-likelihood
        self.gp.set_parameter_vector(params)
        return self.gp.lnlikelihood(self.y, quiet=True)


    def show_results(self, burnin):
        """
        Displays results from interpolative prediction
    
        Args:
            burnin (int): Burn in time to trim the samples
        """
        
        # Modify self.samples for burn-in
        self.samples = self.sampler.chain[:, burnin:, :].reshape((-1, self.ndim))

        # Compute sample size 
        n_samples = np.minimum(200, self.samples.shape[0] - burnin*self.ndim)

        # Create array to save prediction values
        y_predict = np.zeros((n_samples*2, len(self.x_predict)))

        # Plot data
        plt.figure(figsize=(14,8))
        plt.errorbar(self.x, self.y, self.yerr,  linestyle='none')
        plt.scatter(self.x, self.y, c='k',zorder=5,s=20, label='Data')
        
        # Progress bar
        print('Progress: ')
        width = 100

        # Plot samples from GPR
        for i in range(n_samples):
            # Get random parameter sample
            r = np.random.randint(self.samples.shape[0])
            self.gp.set_parameter_vector(self.samples[r])

            # Get 2 predictions per sample: upper and lower bound
            y_predict[i,:], var = self.gp.predict(self.y, self.x_predict, return_var=True)
            y_predict[i+n_samples,:] = y_predict[i,:] + np.sqrt(np.abs(var))
            y_predict[i,:] = y_predict[i,:] -  np.sqrt(np.abs(var))
            
            # Update progress bar   
            n = int((width+1) * float(i) / n_samples)         
            print("\r[{0}{1}]".format('#' * n, ' ' * (width - n)), end='')

            # Plot 10 samples
            if i < 10:
                plt.plot(self.x_predict, y_predict[i,:], "b", alpha=0.1)
                plt.plot(self.x_predict, y_predict[i+n_samples,:], "b", alpha=0.1)

        print(os.linesep)

        plt.xlabel('X', fontsize=20)
        plt.ylabel('Y', fontsize=20)
        plt.xlim([np.max([self.x[0], self.x_predict[0]]),
                  np.min([self.x[-1], self.x_predict[-1]])])
        plt.title('Samples of GPR MCMC Fit to Data', fontsize=20);
        plt.show()

        # Compute point estimate for prediction and error
        self.y_predict = np.average(y_predict, axis=0)
        self.yerr_predict = np.sqrt(np.var(y_predict, axis=0))


    def get_parameters(self):
        """
        Getter for the GPR parameters.
        """

        return self.gp.get_parameter_vector()

    

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
