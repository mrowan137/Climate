import numpy as np
import emcee
import seaborn as sns
from climate.data_io import *
from climate.pySCM.SimpleClimateModel import *


"""
Traditional climate model
"""

def log_lh_tcm(theta, x, y, yerr):
    """Returns log of likelihood function

    Args:
        x (array): independent variable, years
        y (array): dependent variable, temperature anomaly
        yerr (array): uncertainty on temperature
        theta (array): parameters for the traditional (simple) climate model
            -shift: overall shift of the temperature curve output by SCM
            -CO2_norm: normalization for CO2 emissions
            -CH4_norm:       "        "  CH4     "
            -N2O_norm:       "        "  N2O     " 
            -SOx_norm:       "        "  SOx     "

    Returns:
        chisq: sum of ((y_data - y_model)/y_err)**2 
    """
    shift, CO2_norm, CH4_norm, N2O_norm, SOx_norm = theta
    
    # run simple climate model and save output
    fileload = get_example_data_file_path(
        'SimpleClimateModelParameterFile.txt', data_dir='pySCM')
    model = SimpleClimateModel(fileload, [shift, CO2_norm, CH4_norm, N2O_norm, SOx_norm])
    model.runModel()

    # read in temperature change output (from simple climate model)
    fileload = get_example_data_file_path(
        'TempChange.dat', data_dir='trad_climate_model_output')
    data_tcm = load_tcm_temp(fileload)
    x_tcm, y_tcm = data_tcm.year, data_tcm.temp

    # select years from data and tcm to compare
    wh_tcm = np.where((x_tcm >= np.min(x)) & (x_tcm <= np.max(x)))
    x_tcm = x_tcm.iloc[:].values[wh_tcm]
    y_tcm = y_tcm.iloc[:].values[wh_tcm] + shift
    
    # compute chisq and return
    chisq = np.sum(((y - y_tcm)/yerr)**2)
    constant = np.sum(np.log(1/np.sqrt(2.0*np.pi*yerr**2)))
    return constant - 0.5*chisq
    

def log_prior_tcm(theta):
    """Returns log of prior probability distribution

    Args:
        theta (array): model parameters

    Returns:
        0. if within prior range, -inf if not
    """
    # unpack the model parameters
    shift, CO2_norm, CH4_norm, N2O_norm, SOx_norm = theta
    
    if (-5. < shift < 5. and 0.7 < CO2_norm < 1.3 and 0.7 < CH4_norm < 1.3
                         and 0.7 < N2O_norm < 1.3 and 0.7 < SOx_norm < 1.3):
        return 0.0
    return -np.inf


def log_post_tcm(theta, x, y, yerr):
    """Returns log of posterior probability distribution for traditional climate model

    Args:
        theta (array): parameters for the traditional (simple) climate model
        x (array): independent variable, years
        y (array): dependent variable, temperature anomaly
        yerr (array): uncertainty on temperature
       
    Returns:
        Log of posterior distribution
    """
    return log_prior_tcm(theta) + log_lh_tcm(theta, x, y, yerr)


def sample(log_post, x, y, yerr, theta_guess, ndim, nwalkers, nsteps, burnin):
    """Samples the posterior distribution via the affine-invariant ensemble 
       sampling algorithm; plots are output to diagnose burn-in time; best-fit
       parameters are printed; best-fit line is overplotted on data, with errors.
    
       Args:
           log_post (function): log of posterior distribution
           x (array): independent variable, years
           y (array): dependent variable, temperature anomaly
           yerr (array): uncertainty on temperature
           theta_guess (array): initial guess for parameters
           ndim (int): dimension of the parameter space space to sample
           nwalkers (int): number of walkers for affine-invariant ensemble sampling;
                           must be an even number
           nsteps (int): number of timesteps for which to run the algorithm
           burnin: burn in time to trim the samples; plots are output to aid in
                   diagnosing this

       Returns:
           Nothing
    """    
    starting_positions = [
        theta_guess + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)
    ]
    
    # set up the sampler object
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_post, args=(x, y, yerr))
    
    # run the sampler
    sampler.run_mcmc(starting_positions, nsteps)
    
    # plot and check for burn in time
    fig, (ax_shift, ax_CO2, ax_CH4, ax_N2O, ax_SOx) = plt.subplots(5)
    ax_shift.set(ylabel='$T_{\\rm shift}$')
    ax_CO2.set(ylabel='CO2 norm')
    ax_CH4.set(ylabel='CH4 norm')
    ax_N2O.set(ylabel='N2O norm')
    ax_SOx.set(ylabel='SOx norm')
    for i in range(10):
        sns.tsplot(sampler.chain[i, :, 0], ax=ax_shift)
        sns.tsplot(sampler.chain[i, :, 1], ax=ax_CO2)
        sns.tsplot(sampler.chain[i, :, 2], ax=ax_CH4)
        sns.tsplot(sampler.chain[i, :, 3], ax=ax_N2O)
        sns.tsplot(sampler.chain[i, :, 4], ax=ax_SOx)

    # trime the samples and reshape
    samples = sampler.chain[:, burnin:, :]
    traces = samples.reshape(-1, ndim).T

    # store the samples in a dataframe
    parameter_samples = pd.DataFrame({'shift': traces[0],
                                      'CO2_norm': traces[1],
                                      'CH4_norm': traces[2],
                                      'N2O_norm': traces[3],
                                      'SOx_norm': traces[4]})

    # compute and print the MAP values
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

    # best-fit params
    shift_best = q['shift'][0.50]
    CO2_norm_best = q['CO2_norm'][0.50]
    CH4_norm_best = q['CH4_norm'][0.50]
    N2O_norm_best = q['N2O_norm'][0.50]
    SOx_norm_best = q['SOx_norm'][0.50]

    # run simple climate model with best fit params
    fileload = get_example_data_file_path(
        'SimpleClimateModelParameterFile.txt', data_dir='pySCM')
    model_best = SimpleClimateModel(
        fileload, [shift_best, CO2_norm_best, CH4_norm_best, N2O_norm_best, SOx_norm_best])
    model_best.runModel()
    
    # read in temperature change output (from simple climate model)
    fileload = get_example_data_file_path(
        'TempChange.dat', data_dir='trad_climate_model_output')
    data_tcm_best = load_tcm_temp(fileload)
    x_tcm_best, y_tcm_best = data_tcm_best.year, data_tcm_best.temp

    # select years from data and tcm to compare
    wh_tcm_best = np.where((x_tcm_best >= np.min(x)) & (x_tcm_best <= np.max(x)))
    x_tcm_best = x_tcm_best.iloc[:].values[wh_tcm_best]
    y_tcm_best = y_tcm_best.iloc[:].values[wh_tcm_best] + shift_best
    
    # plot the best-fit line, and data
    plt.errorbar(x, y, yerr, label='data')
    plt.plot(x_tcm_best, y_tcm_best, label='best fit')
    plt.xlabel('year')
    plt.ylabel('$\Delta T$ ($^{\circ}$C)')
    plt.title('Global surface temperature anomaly; data vs. fit');
    plt.legend()
