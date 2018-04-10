import numpy as np
import emcee
import seaborn as sns
from climate.data_io import *
from climate.pySCM.SimpleClimateModel import *


"""
Traditional climate model
"""

def log_lh_scm(theta, x, y, yerr):
    """Returns log of likelihood function

    Args:
        x (array): Independent variable, years
        y (array): Dependent variable, temperature anomaly
        yerr (array): Uncertainty on temperature
        theta (array): Parameters for the traditional (simple) climate model
            -shift: Overall shift of the temperature curve output by SCM
            -CO2_norm: Normalization for CO2 emissions
            -CH4_norm:       "        "  CH4     "
            -N2O_norm:       "        "  N2O     " 
            -SOx_norm:       "        "  SOx     "

    Returns:
        chisq: Sum of ((y_data - y_model)/y_err)**2 
    """
    shift, CO2_norm, CH4_norm, N2O_norm, SOx_norm = theta
    
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
    wh_scm = np.where((x_scm >= np.min(x)) & (x_scm <= np.max(x)))
    x_scm = x_scm.iloc[:].values[wh_scm]
    y_scm = y_scm.iloc[:].values[wh_scm] + shift
    
    # Compute chisq and return
    chisq = np.sum(((y - y_scm)/yerr)**2)
    constant = np.sum(np.log(1/np.sqrt(2.0*np.pi*yerr**2)))
    return constant - 0.5*chisq
    

def log_prior_scm(theta):
    """Returns log of prior probability distribution

    Args:
        theta (array): Model parameters

    Returns:
        0. if within prior range, -inf if not
    """
    # Unpack the model parameters
    shift, CO2_norm, CH4_norm, N2O_norm, SOx_norm = theta
    
    if (-5. < shift < 5. and 0.7 < CO2_norm < 1.3 and 0.7 < CH4_norm < 1.3
                         and 0.7 < N2O_norm < 1.3 and 0.7 < SOx_norm < 1.3):
        return 0.0
    return -np.inf


def log_post_scm(theta, x, y, yerr):
    """Returns log of posterior probability distribution for traditional climate model

    Args:
        theta (array): Parameters for the traditional (simple) climate model
        x (array): Independent variable, years
        y (array): Dependent variable, temperature anomaly
        yerr (array): Uncertainty on temperature
       
    Returns:
        Log of posterior distribution
    """
    return log_prior_scm(theta) + log_lh_scm(theta, x, y, yerr)


def sample(log_post, x, y, yerr, theta_guess, ndim, nwalkers, nsteps):
    """Samples the posterior distribution via the affine-invariant ensemble 
       sampling algorithm; plots are output to diagnose burn-in time; best-fit
       parameters are printed; best-fit line is overplotted on data, with errors.
    
       Args:
           log_post (function): Log of posterior distribution
           x (array): Independent variable, years
           y (array): Dependent variable, temperature anomaly
           yerr (array): Uncertainty on temperature
           theta_guess (array): Initial guess for parameters
           ndim (int): Dimension of the parameter space space to sample
           nwalkers (int): Number of walkers for affine-invariant ensemble sampling;
                           must be an even number
           nsteps (int): Number of timesteps for which to run the algorithm


       Returns:
           Samples (array): Trajectories of the walkers through parameter spaces.
                            This array has dimension (nwalkers) x (nsteps) x (ndim)
    """    
    starting_positions = [
        theta_guess + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)
    ]
    
    # Set up the sampler object
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_post, args=(x, y, yerr))
    
    # Run the sampler
    sampler.run_mcmc(starting_positions, nsteps)
    
    # return the samples and sampler.flatchain for later output
    samples = sampler.chain[:, :, :]
    sampler_flatchain = sampler.flatchain
    return samples, sampler_flatchain


def show_results(samples, sampler_flatchain, burnin, x, y, yerr):
    """Displays results from sample()
    
       Args:
           samples (array): Collection of walker trajectories, and is
                            a result of sample()
           sampler_flatchain (array): Flattened array of size (nwalkers*nsteps) x (ndim);
                                      array is flattened along the walker axis
           burnin (int): Burn in time to trim the samples; plots are output to 
                         aid in diagnosing this
           x (array): Independent variable, years
           y (array): Dependent variable, temperature anomaly
           yerr (array): Uncertainty on temperature
       
       Returns:
           Samples (array): Trajectories of the walkers through parameter spaces.
                            This array has dimension (nwalkers) x (nsteps) x (ndim)
    """  
   
    # Plot and check for burn in time
    fig, (ax_shift, ax_CO2, ax_CH4, ax_N2O, ax_SOx) = plt.subplots(5, figsize=(10,10))
    plt.subplots_adjust(hspace=0.5)
    ax_shift.set(ylabel='$T_{\\rm shift}$')
    ax_CO2.set(ylabel='CO2 norm')
    ax_CH4.set(ylabel='CH4 norm')
    ax_N2O.set(ylabel='N2O norm')
    ax_SOx.set(ylabel='SOx norm')
    
    sns.distplot(sampler_flatchain[:, 0], ax=ax_shift)
    sns.distplot(sampler_flatchain[:, 1], ax=ax_CO2)
    sns.distplot(sampler_flatchain[:, 2], ax=ax_CH4)
    sns.distplot(sampler_flatchain[:, 3], ax=ax_N2O)
    sns.distplot(sampler_flatchain[:, 4], ax=ax_SOx)
    traces = samples.reshape(-1, samples.shape[-1]).T

    # Store the samples in a dataframe
    parameter_samples = pd.DataFrame({'shift': traces[0],
                                      'CO2_norm': traces[1],
                                      'CH4_norm': traces[2],
                                      'N2O_norm': traces[3],
                                      'SOx_norm': traces[4]})

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
    wh_scm_best = np.where((x_scm_best >= np.min(x)) & (x_scm_best <= np.max(x)))
    x_scm_best = x_scm_best.iloc[:].values[wh_scm_best]
    y_scm_best = y_scm_best.iloc[:].values[wh_scm_best] + shift_best
    
    # Plot the best-fit line, and data
    plt.figure(figsize=(10,8))
    plt.errorbar(x, y, yerr,  linestyle='none')
    plt.scatter(x, y, c='k',zorder=5,s=20, label='data')
    plt.plot(x_scm_best, y_scm_best, label='best fit')
    plt.xlabel('Year')
    plt.ylabel('$\Delta T$ ($^{\circ}$C)')
    plt.title('Global Surface Temperature Anomaly');
    plt.legend()
