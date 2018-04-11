import numpy as np
import emcee
import seaborn as sns
from climate.data_io import *
from climate.pySCM.SimpleClimateModel import *
import sys


"""
Modified Simple Climate Model Class
"""

class SCM:
    def __init__(self, t, T, dT)
	"""
	Initialize global variables of Simple Climate Model class

	Args:
	    t (array): Independent variable, years
	    T (array): Dependent variable, temperature anamoly
	    dT (array): Uncertainty on temperature anamoly
	"""
	self.t = t
	self.T = T
	self.dT = dT
	self.prior_bounds = np.zeros([5,2])
	self.samples = np.zeros(1)
    
    def set_prior_bounds(self, i_param, low, high)
	"""
	Setter for the bounds on a particular model parameter

	Args:
	    i_param (integer): index of parameter to set
	    low (double): lower bound of flat prior for parameter
	    high (double): upper bound of flat prior for parameter
	"""
	# If high < low, assume input error. Allow low == high
	if (high < low):
	    low, high = high, low
	self.prior_bounds[i_param] = [low, high] 

    def get_full_theta_array(self, param):
	"""
	Helper function that returns length-5 array of all parameters given a
	length-n array of parameters corresponding to initialized PriorBounds
	
	Args:
	    param (array): array of parameters of length <=5

	Returns:
	    theta (array): length 5 of all parameters filled using entries
	    of prior_bounds that have low == high
	"""

	theta = self.prior_bounds.T[0]
	# Should check to make sure that len(param) = len(where())
	where_diff = np.where(self.prior_bounds.T[1] - prior_bounds.T[0] > 0)
	theta[where_diff] = param
	return theta

    def log_lh_scm(self, theta):
        """
        Computes log of Gaussian likelihood function

        Args:
            theta (array): Parameters for the simple climate model,
	    contain subset (in order) of the following parameters:
                -shift: Overall shift of the temperature curve output by SCM
                -CO2_norm: Normalization for CO2 emissions
                -CH4_norm:       "        "  CH4     "
                -N2O_norm:       "        "  N2O     " 
                -SOx_norm:       "        "  SOx     "

        Returns:
            chisq: Sum of ((y_data - y_model)/y_err)**2 
        """
        shift, CO2_norm, CH4_norm, N2O_norm, SOx_norm = self.get_full_theta_array(theta)
    
        # Run simple climate model and save output
        fileload = get_example_data_file_path(
            'SimpleClimateModelParameterFile.txt', data_dir='pySCM')
        model = SimpleClimateModel(fileload, [shift, CO2_norm, CH4_norm, N2O_norm, SOx_norm])
        model.runModel()

        # Read in temperature change output (from simple climate model)
        fileload = get_example_data_file_path(
            'TempChange.dat', data_dir='trad_climate_model_output')
        data_scm = load_scm_temp(fileload)
        x_scm, y_scm = data_scm.year, data_scm.temp

        # Select years from data and scm to compare
        wh_scm = np.where((x_scm >= np.min(t)) & (x_scm <= np.max(t)))
        x_scm = x_scm.iloc[:].values[wh_scm]
        y_scm = y_scm.iloc[:].values[wh_scm] + shift
    
        # Compute chisq and return
        chisq = np.sum(((T - y_scm)/dT)**2)
        constant = np.sum(np.log(1/np.sqrt(2.0*np.pi*dT**2)))
        return constant - 0.5*chisq
    

    def log_prior_scm(self, theta):
    	"""
        Computes log of prior probability distribution

    	Args:
            theta (array): Model parameters

        Returns:
            log of normalized prior value if within prior range, -inf if not
        """
        
 	# Unpack the model parameters
        shift, CO2_norm, CH4_norm, N2O_norm, SOx_norm = self.get_full_theta_array(theta)
    
        # Find where prior bounds form actual intervals
        where_diff = np.where(self.prior_bounds.T[1] - prior_bounds.T[0] > 0)
	bounds = self.prior_bounds
        
	if (bounds[0][0] <= shift <= bounds[1][0] and bounds[0][1] <= CO2_norm <= bounds[1][1] and bounds[0][2] < CH4_norm < bounds[1][2]
                         and bounds[0][3] < N2O_norm < bounds[1][3] and bounds[0][4] < SOx_norm < bounds[1][4]):
            return np.log(np.product((bounds[1]-bounds[0])[where_diff]))
        return -np.inf


    def log_post_scm(self, theta):
        """
	Returns log of posterior probability distribution for traditional climate model

        Args:
            theta (array): Parameters for the traditional (simple) climate model
            x (array): Independent variable, years
            y (array): Dependent variable, temperature anomaly
            yerr (array): Uncertainty on temperature
	    bounds (2Darray): Bounds on flat prior
       
        Returns:
            Log of posterior distribution
        """
        return self.log_prior_scm(theta) + self.log_lh_scm(theta)


    def run_MCMC(self, theta_guess, nwalkers, nsteps):
        """
	Samples the posterior distribution via the affine-invariant ensemble 
        sampling algorithm; plots are output to diagnose burn-in time; best-fit
        parameters are printed; best-fit line is overplotted on data, with errors.
    
        Args:
            theta_guess (array): Initial guess for parameters. Can have length < 5
            nwalkers (int): Number of walkers for affine-invariant ensemble sampling;
                            must be an even number
            nsteps (int): Number of timesteps for which to run the algorithm


        Returns:
            Samples (array): Trajectories of the walkers through parameter spaces.
                             This array has dimension (nwalkers) x (nsteps) x (ndim)
        """
        # Check that number of parameter guesses matches number of prior ranges set
        L1 = len(theta_guess)
        L2 = len(np.where(self.prior_bounds.T[1] - prior_bounds.T[0] > 0))
        if(L1 != L2):
	    print("Initial guess contains ", L1, " parameters, but prior range set for ", L2, " parameters")
	    sys.exit(1)
    
        starting_positions = [
            theta_guess + 1e-4 * np.random.randn(L1) for i in range(nwalkers)
        ]
    
        # Set up the sampler object
        sampler = emcee.EnsembleSampler(nwalkers, L1, self.log_post, args=())
    
        # Run the sampler
        sampler.run_mcmc(starting_positions, nsteps)
    
        # return the samples for later output
        self.samples = sampler.flatchain
        return self.samples

    def get_samples(self):
        """
        Getter that returns the samples
        """
        if(len(samples) = 1):
	    print("MCMC algorithm has not yet been run")

        return self.samples


"""
To do: 
Make function that returns the MAP value with error
"""

    def show_results(self, burnin):
        """
        Displays results from sample()
    
        Args:
            burnin (int): Burn in time to trim the samples; plots are output to 
                         aid in diagnosing this
        """  
   
        # Plot and check for burn in time
        fig, (ax_shift, ax_CO2, ax_CH4, ax_N2O, ax_SOx) = plt.subplots(5, figsize=(10,10))
        plt.subplots_adjust(hspace=0.5)
        ax_shift.set(ylabel='$T_{\\rm shift}$')
        ax_CO2.set(ylabel='CO2 norm')
        ax_CH4.set(ylabel='CH4 norm')
        ax_N2O.set(ylabel='N2O norm')
        ax_SOx.set(ylabel='SOx norm')
    
        sns.distplot(self.samples[:, 0], ax=ax_shift)
        sns.distplot(self.samples[:, 1], ax=ax_CO2)
        sns.distplot(self.samples[:, 2], ax=ax_CH4)
        sns.distplot(self.samples[:, 3], ax=ax_N2O)
        sns.distplot(self.samples[:, 4], ax=ax_SOx)

        # Store the samples in a dataframe
        parameter_samples = pd.DataFrame({'shift': self.samples[:,0],
                                          'CO2_norm': self.samples[:,1],
                                          'CH4_norm': self.samples[:,2],
                                          'N2O_norm': self.samples[:,3],
                                          'SOx_norm': self.samples[:,4]})

        # Compute and print the MAP values
        q = parameter_samples.quantile([0.16, 0.50, 0.84], axis=0)
        print("shift = {:.6f} + {:.6f} - {:.6f}".format(
        q['shift'][0.50], q['shift'][0.84] - q['shift'][0.50], q['shift'][0.50] - q['shift'][0.16]))
        print("CO2_norm = {:.6f} + {:.6f} - {:.6f}".format(
        q['CO2_norm'][0.50], q['CO2_norm'][0.84] - q['CO2_norm'][0.50], q['CO2_norm'][0.50] - q['CO2_norm'][0.16]))
        print("CH4_norm = {:.6f} + {:.6f} - {:.6f}".format(
        q['CH4_norm'][0.50], q['CH4_norm'][0.84] - q['CH4_norm'][0.50], q['CH4_norm'][0.50] - q['CH4_norm'][0.16]))
        print("N2O_norm = {:.6f} + {:.6f} - {:.6f}".format(
        q['N2O_norm'][0.50], q['N2O_norm'][0.84] - q['N2O_norm'][0.50], q['N2O_norm'][0.50] - q['N2O_norm'][0.16]))
        print("SOx_norm = {:.6f} + {:.6f} - {:.6f}".format(
        q['SOx_norm'][0.50], q['SOx_norm'][0.84] - q['SOx_norm'][0.50], q['SOx_norm'][0.50] - q['SOx_norm'][0.16]))

        # Best-fit params
        shift_best = q['shift'][0.50]
        CO2_norm_best = q['CO2_norm'][0.50]
        CH4_norm_best = q['CH4_norm'][0.50]
        N2O_norm_best = q['N2O_norm'][0.50]
        SOx_norm_best = q['SOx_norm'][0.50]

        # Run simple climate model with best fit params
        fileload = get_example_data_file_path(
            'SimpleClimateModelParameterFile.txt', data_dir='pySCM')
        model_best = SimpleClimateModel(
            fileload, [shift_best, CO2_norm_best, CH4_norm_best, N2O_norm_best, SOx_norm_best])
        model_best.runModel()
    
        # Read in temperature change output (from simple climate model)
        fileload = get_example_data_file_path(
            'TempChange.dat', data_dir='trad_climate_model_output')
        data_scm_best = load_scm_temp(fileload)
        x_scm_best, y_scm_best = data_scm_best.year, data_scm_best.temp

        # Select years from data and tcm to compare
        wh_scm_best = np.where((x_scm_best >= np.min(self.t)) & (x_scm_best <= np.max(self.t)))
        x_scm_best = x_scm_best.iloc[:].values[wh_scm_best]
        y_scm_best = y_scm_best.iloc[:].values[wh_scm_best] + shift_best
    
        # Plot the best-fit line, and data
        plt.figure(figsize=(10,8))
        plt.errorbar(t, T, dT,  linestyle='none')
        plt.scatter(t, T, c='k',zorder=5,s=20, label='data')
        plt.plot(x_scm_best, y_scm_best, label='best fit')
        plt.xlabel('Year')
        plt.ylabel('$\Delta T$ ($^{\circ}$C)')
        plt.title('Global Surface Temperature Anomaly');
        plt.legend()
