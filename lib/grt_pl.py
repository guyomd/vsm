import numpy as np
from scipy.stats import norm, multivariate_normal
from copy import deepcopy
from struct import pack, unpack, Struct, calcsize  # To read and write binary files
import matplotlib.pyplot as plt
from matplotlib import colorbar
import pygmt

from lib.plotutils import ProgressBar


def penalized_likelihood(nums, areas, lons, lats, m_bins, m_periods, mmax_pmf, m_weights,
                         m0=None, max_trials=100000, n_init=None, collection_interval=10,
                         prior=None,
                         proposal={'fun_type': 'uniform-local', 'sigma': 0.1, 'strategy': 'incremental'},
                         output_file='mcmc_samples.out', only_sample_prior=False, seed=None,
                         apply_smoothing=True):
    """
    Return cell-wise evaluation of Truncated Gutenberg-Richter parameters (a, b) using the penalized likelihood
    approach described in the EPRI-USNRC (2012) report entitled "Central and Eastern United States Seismic Source
    Characterization for Nuclear Facilities Project", NUREG-2115, US Nuclear Regulatory Commission, Washington DC,
    section 5.3.2.
    Estimation of parameters is done using a Monte-Carlo Metropolis-Hastings (MCMC) algorithm.

    :param nums: M-by-Q numpy.ndarray, earthquake counts per bin (can be non-integer) for all M cells
        and for every Q magnitude bin (per unit time and per unit area)
    :param areas: numpy.ndarray, cells areas for all M cells
    :param lons: M-by-1-shaped numpy.ndarray, mesh of central longitudes for all M cells. Any cell ordering is allowed,
        provided that it is consistent with the order of the 'lats' argument.
    :param lats: M-by-1-shaped numpy.ndarray, mesh of central latitudes for all M cells
    :param m_bins: Q-by-2 numpy.ndarray, upper and lower bounds for all Q magnitude bins, where each line contains
        2 values: [magmin, magmax], in respective order
    :param m_periods: M-by-Q numpy.ndarray, duration of completeness periods for all M cells and all Q magnitude bins
    :param mmax_pmf: M-by-K-by-2 numpy.ndarray, probability mass function (PMF) the maximum magnitude for all M cells.
        The PMF is composed of K elements, each mmax_pmf[j,i,:] being itself an with 2 values: [mmax, weight]
        in respective order
    :param m_weights: (Q,)-shaped numpy.ndarray, weights for all Q magnitude bins
    :param m0: float, minimum magnitude for the truncated-exponential distribution. If None: m0 set equal to m_bins[:,0].min()
    :param max_trials: int, total number of iterations reaized by the MCMC algorithm
    :param n_init: int, MCMC "burn-in" phase, in nb. of iterations. Typically, 1% of max_trials
    :param collection_interval: int, recurrent parameter set collecting interval, in number of iterations
    :param prior: dict, properties of the prior on parameters.
        Expected keys:
            'b_mean', 'b_std': for the gaussian prior on the b-value
            'mins', 'maxs': to define bounds on all parameters. Note that bounds in must be organized in the following
                            order and format: np.array([log10(rates)_j=1,M  beta_j=1,M  sm_beta, sm_rates]),
                            with rates expressed per unit time and per unit area.
    :param proposal: dict, properties of proposal function. Expected keys: 'fun_type', 'strategy', ['sigma'].
    :param output_file: str, path to output file for parameter samples + likelihood values. Default: "mcmc_samples.csv".
    :param only_sample_prior: bool, Specific mode for sampling the prior distribution only. Likelihood is held constant for
        every parameter set
    :param seed: int, seed for the Random Number Generator. If None, it is automatically drawn accordnig to Numpy's default.
    :param apply_smoothing: bool, specify whether (True) or not (False) to account for the spatial smoothing parameter
        in the likelihood function. Default: True. Note that smoothing is automatically disabled when M = 1.

    :return: None
    """
    def _icell2iparam(icell, m, n):
        if icell < m:
            i = [icell, m + icell]
        elif icell == m:
            i = n - 2
        elif icell == m + 1:
            i = n - 1
        return i


    def _pmf_m(mmax_pmf, beta, m0, m1, m2):
        """
        Returns the truncated-exponential bin probability for a single magnitude bin [m1, m2], see Eq. 5.3.2-11

        :param mmax_pmf: K-by-2 numpy.ndarray, probability mass function for the maximum magnitude. Values are stored in
                     mmax_pmf[:,0] and weights in mmax_pmf[:,1]
        :param beta, float, Guternberg-Richter parameter beta = b * ln(10)
        :param m0: float, magnitude of completeness
        :param m1: float, lower bound for the magnitude bin
        :param m2: float, upper bound for the magnitude bin
        :return: p: float, maximum-magnitude penalty term
        """
        mmax = mmax_pmf[:, 0]
        w_mmax = mmax_pmf[:, 1]
        p = np.zeros_like(mmax)  # implicitely accounts for the case "mmax < m1"
        p[mmax >= m2] = np.exp(-beta * (m1 - m0)) - np.exp((-beta * (m2 - m0)))
        im = np.logical_and(mmax >= m1, mmax < m2)
        p[im] = np.exp(-beta * (m1 - m0)) - np.exp((-beta * (mmax[im] - m0)))
        return (p * w_mmax).sum() / w_mmax.sum()


    def _update_p_m(beta, mmax_pmf, m_bins, m0):
        m = beta.shape[0]
        q = m_bins.shape[0]
        p_m = np.zeros((m, q))
        for j in range(m):  # Loop on cells
            for k in range(q):  # Loop on magnitude bins
                p_m[j, k] = _pmf_m(mmax_pmf[j, :, :], beta[j], m0, m_bins[k, 0], m_bins[k, 1])
        return p_m


    def _joint_likelihood(beta, rates, areas, nums, t_m, w_m, m_bins, m0, mmax_pmf):
        """
        Returns the joint likelihood function for the values of rate and beta in all M cells, such that
        M = len(beta) = len(rate), see Eq. 5.3.2-13

        :param beta: (M,)-shaped np.ndarray, values of beta for all cells
        :param rates: (M,)-shaped np.ndarray, earthquake rates for all cells (rate is expressed per unit time per unit
            area for all events with magnitude greater than or equal to m0)
        :param areas: (M,)-shaped np.ndarray, cell areas
        :param nums: M-by-Q np.ndarray, earthquake counts (can be non-integer) for all M cells and all Q magnitude bins
        :param t_m: M-by-Q numpy.ndarray, periods of completeness for all Q magnitude bins and all M cells
        :param w_m: (Q,)-shaped numpy.ndarray, weights for all Q magnitude bins
        :param m_bins: Q-by-2 element numpy.ndarray, parameter for all Q magnitude bins, where each bin is characterized
            by 2 values: [minmag, maxmag] in respective order
        :param m0: float, magnitude of completeness
        :param mmax_pmf: M-by-K-by-2 numpy.ndarray, probability mass function (PMF) the maximum magnitude for all M cells.
            The PMF is composed of K elements, each mmax_pmf[j,i,:] being itself an with 2 values: [mmax, weight]
            in respective order
        :return:
        """
        m, q = t_m.shape  # Q: number of magnitude bins, M: number of cells
        # Update p_m (M-by-Q numpy.ndarray) the truncated-exponential bin probabilities for all Q magnitude bins and all M cells:
        p_m = _update_p_m(beta, mmax_pmf, m_bins, m0)  # Returns a M-by-Q elements array
        rates_mq = np.broadcast_to(rates, (q, m)).T
        areas_mq = np.broadcast_to(areas, (q, m)).T
        w_m_mq = np.broadcast_to(w_m, (m, q))
        ll_j = np.sum(np.sum(nums * w_m_mq * np.log(rates_mq * areas_mq * t_m * p_m), axis=1) - rates * areas * np.sum(
            t_m * p_m * w_m_mq, axis=1))
        """
        ll_j = 0
        for j in range(m):
            tmp1 = 0
            q2 = p_m[j,:].nonzero()[0].max()  # Highest bin with nonzero p_m[j,:] value
            for k in range(q2):
                tmp1 += nums[j,k] * w_m[k] * rates[j] * t_m[j,k] * p_m[j,k]
            ll_j -= tmp1 * (rates[j] * np.sum(t_m[j,:] * p_m[j,:] * w_m[:]))
        """
        return ll_j  # Log-likelihood


    def _joint_likelihood_increment(beta, rate, area, nums, t_m, w_m, m_bins, m0, mmax_pmf):
        """
        Returns the joint likelihood increment for the rate and beta in the current cell, such that
        M = len(beta) = len(rate), see Eq. 5.3.2-13

        :param beta: (1,)-shaped np.ndarray, values of beta for the current cell
        :param rate: (1,)-shaped np.ndarray, earthquake rates for the current cell (rate is expressed per unit time per
            unit area for all events with magnitude greater than or equal to m0)
        :param area: (1,)-shaped np.ndarray, cell area
        :param nums: 1-by-Q np.ndarray, earthquake counts (can be non-integer) for the current cell and all Q magnitude bins
        :param t_m: 1-by-Q numpy.ndarray, periods of completeness for all Q magnitude bins and the current cell
        :param w_m: (Q,)-shaped numpy.ndarray, weights for all Q magnitude bins
        :param m_bins: Q-by-2 element numpy.ndarray, parameter for all Q magnitude bins, where each bin is characterized
            by 2 values: [minmag, maxmag] in respective order
        :param m0: float, magnitude of completeness
        :param mmax_pmf: 1-by-K-by-2 numpy.ndarray, probability mass function (PMF) the maximum magnitude for the current cell.
            The PMF is composed of K elements, each mmax_pmf[j,i,:] being itself an with 2 values: [mmax, weight]
            in respective order
        :return:
        """
        q = t_m.shape[0]  # Q: number of magnitude bins, M: number of cells
        mmax_pmf = np.reshape(mmax_pmf, (1,mmax_pmf.shape[0], mmax_pmf.shape[1]))
        beta = np.array([beta])
        # Update p_m (M-by-Q numpy.ndarray) the truncated-exponential bin probabilities for all Q magnitude bins:
        p_m = _update_p_m(beta, mmax_pmf, m_bins, m0)  # Returns a 1-by-Q elements array
        rates_mq = np.broadcast_to(rate, (q, 1)).T
        area_mq = np.broadcast_to(area, (q, 1)).T
        w_m_mq = np.broadcast_to(w_m, (1, q))
        ll_inc = np.sum(nums * w_m_mq * np.log(rates_mq * area_mq * t_m * p_m), axis=1) - rate * area * np.sum(
            t_m * p_m * w_m_mq, axis=1)
        return ll_inc  # Log-likelihood increment


    def _spatial_penalty_function(array, std_prm, laplacian):
        """
        Returns the penalty value associated with an array of parameters ARRAY. This function is implemented according to
        eqns. 5.3.2-14, 5.3.2-15 and 5.3.2-16.

        :param array: (M,)-element numpy.ndarray, parameter values for all M cells (with M: total number of cells)
        :param std_prm: float, smoothing parameters for the population of parameters
        :param laplacian: (M1, M)-element numpy.ndarray, laplacian operator for all M1 cells considered in the penalty term
        :return: penalty value associated with the spatial smoothness of the parameter distribution
        """
        deltas = laplacian @ array
        return np.sum(-np.log(np.sqrt(2 * np.pi) * std_prm) - 0.5 * np.power(deltas / std_prm, 2))  # Log-penalty
        # return np.prod( 1 / (np.sqrt(2 * np.pi) * std_prm) * np.exp(-0.5 * np.power(deltas / std_prm, 2)) )  # Penalty function


    def _gaussian_prior_density(prms, prior_mean, prior_std):
        """
        Returns a prior probability value on parameter, where density is modelled as a
        gaussian law, with mean prior_mean and standard deviation prior_std

        :param prms:
        :param prior_mean:
        :param prior_std:
        :return: prior function value
        """
        return np.sum(-np.log(np.sqrt(2 * np.pi) * prior_std) - 0.5 * np.power((prms - prior_mean) / prior_std,
                                                                               2))  # Log-probability
        # return np.prod( 1 / (np.sqrt(2 * np.pi) * prior_std) * np.exp(0.5 * np.power((prms - prior_mean) / prior_std, 2)) )  # Probability


    def _uniform_prior_density(prms, prior_mins, prior_maxs):
        """
        Returns a prior probability value on parameter, where density is modelled as a
        uniform distribution, bounded by prior_mins and prior_maxs

        :param prms:
        :param prior_mins:
        :param prior_maxs:
        :return: prior function value
        """
        norms = prior_maxs - prior_mins
        priors = np.zeros_like(prms)
        in_bounds = np.logical_and(prms >= prior_mins, prms <= prior_maxs)
        priors[in_bounds] = 1 / norms[in_bounds]
        priors[np.logical_not(in_bounds)] = 0
        # return np.prod(priors)  # Prior probability
        return np.sum(np.log(priors))  # Log


    def _likelihood(prms, areas, nums, m_bins, m_periods, m_w, m0, mmax_pmf, laplacian,
                    smoothing=True):
        """
        Returns the log-likelihood for all M cells, see Eq. 5.3.2-18
        """
        n = len(prms)
        m = int((n - 2) / 2)  # number of cells
        if m == 1:
            smoothing = False
        logrates = prms[:m]
        rates = np.power(10, logrates)
        beta = prms[m:-2]
        sm_beta = prms[-2]
        sm_rates = prms[-1]
        lj = _joint_likelihood(beta, rates, areas, nums, m_periods, m_w, m_bins, m0, mmax_pmf)
        if smoothing:
            fb = _spatial_penalty_function(beta, sm_beta, laplacian)  # penalty function on beta
            fr = _spatial_penalty_function(logrates, sm_rates, laplacian)  # penalty on log(rates)
        else:
            fb = 0
            fr = 0
        p = lj + fb + fr  # Log-likelihood
        return p


    def _likelihood_increment(prms, icell, prms0, areas, nums, m_bins, m_periods, m_w, m0, mmax_pmf, laplacian,
                              smoothing=True):
        """
        Returns the log-likelihood for all M cells, see Eq. 5.3.2-18
        """
        n = len(prms)
        m = int((n - 2) / 2)  # total number of cells
        if m == 1:
            smoothing = False
        all_logrates = prms[0: m]
        all_logrates0 = prms0[0: m]
        all_betas = prms[m: 2 * m]
        all_betas0 = prms0[m: 2 * m]
        ipar = _icell2iparam(icell, m, n)
        if isinstance(ipar, list) and (len(ipar) == 2):
            # Current iteration for a (log10(rate), beta) pair:
            rate = np.power(10, prms[ipar[0]])
            rate0 = np.power(10, prms0[ipar[0]])
            beta = prms[ipar[1]]
            beta0 = prms0[ipar[1]]
            sm_beta = prms[n - 2]
            sm_beta0 = prms0[n - 2]
            sm_rates = prms[n - 1]
            sm_rates0 = prms[n - 1]
            lj = _joint_likelihood_increment(beta, rate, areas[icell], nums[icell, :], m_periods[icell, :], m_w, m_bins, m0,
                    mmax_pmf[icell,:,:])
            lj0 = _joint_likelihood_increment(beta0, rate0, areas[icell], nums[icell, :], m_periods[icell, :], m_w, m_bins, m0,
                    mmax_pmf[icell,:,:])
            if smoothing:
                fb = _spatial_penalty_function(all_betas, sm_beta, laplacian[icell,:])  # local penalty function on beta
                fb0 = _spatial_penalty_function(all_betas0, sm_beta0, laplacian[icell,:])
                fr = _spatial_penalty_function(all_logrates, sm_rates, laplacian[icell,:])  # local penalty on log(rates)
                fr0 = _spatial_penalty_function(all_logrates0, sm_rates0, laplacian[icell,:])
                inc = lj - lj0 + fb - fb0 + fr - fr0
            else:
                inc = lj - lj0

        elif ipar == n - 2:
            # Current iteration for sm_beta parameter:
            if smoothing:
                sm_beta = prms[ipar]
                sm_beta0 = prms0[ipar]
                fb = _spatial_penalty_function(all_betas, sm_beta, laplacian)  # penalty term on beta for all cells
                fb0 = _spatial_penalty_function(all_betas0, sm_beta0, laplacian)
                inc = fb - fb0
            else:
                inc = 0

        elif ipar == n - 1:
            # Current iteration for sm_lograte parameter:
            if smoothing:
                sm_rates = prms[ipar]
                sm_rates0 = prms0[ipar]
                fr = _spatial_penalty_function(all_logrates, sm_rates, laplacian)  # penalty term on log(rates) for all cells
                fr0 = _spatial_penalty_function(all_logrates0, sm_rates0, laplacian)
                inc = fr - fr0
            else:
                inc = 0

        return np.float64(inc)   # Log-likelihood increment


    def _prior(prms, b_prior_mean, b_prior_std, prms_mins, prms_maxs):
        """
        Returns the log-prior probability for all parameters
        """
        n = len(prms)
        m = int((n - 2) / 2)
        logrates = prms[:m]
        logrates_mins = prms_mins[:m]
        logrates_maxs = prms_maxs[:m]
        beta = prms[m:-2]
        beta_mins = prms_mins[m:-2]
        beta_maxs = prms_maxs[m:-2]
        sm_beta = prms[-2]
        sm_beta_min = prms_mins[-2]
        sm_beta_max = prms_maxs[-2]
        sm_rates = prms[-1]
        sm_rates_min = prms_mins[-1]
        sm_rates_max = prms_maxs[-1]
        fp1 = _gaussian_prior_density(beta, b_prior_mean * np.log(10), b_prior_std * np.log(10))  # Log
        fp2 = _uniform_prior_density(beta, beta_mins, beta_maxs)  # Log
        fp3 = _uniform_prior_density(logrates, logrates_mins, logrates_maxs)  # Log
        fp4 = _uniform_prior_density(sm_beta, sm_beta_min, sm_beta_max)  # Log
        fp5 = _uniform_prior_density(sm_rates, sm_rates_min, sm_rates_max)  # Log
        #print(f'DEBUG: fp1 = {fp1}  fp2 = {fp2}  fp3 = {fp3}  fp4 = {fp4}  fp5 = {fp5} ')
        return fp1 + fp2 + fp3 + fp4 + fp5  # Log prior probability


    def _prior_increment(prms, icell, prms0, b_prior_mean, b_prior_std, prms_mins, prms_maxs):
        """
        Returns the log-prior probability increment associated with a change of parameters for cell ICELL in the new
        parameter set PRMS (other parameters remain equal to those in PRMS0)
        """
        n = len(prms)
        m = int((n - 2) / 2)
        ipar = _icell2iparam(icell, m, n)
        pars = prms[ipar]
        pars0 = prms0[ipar]
        pars_min = prms_mins[ipar]
        pars_max = prms_maxs[ipar]

        f = _uniform_prior_density(pars, pars_min, pars_max)  # Log
        if isinstance(ipar, list) and (len(pars) == 2):
            f += _gaussian_prior_density(pars[1], b_prior_mean * np.log(10), b_prior_std * np.log(10))  # Log

        f0 = _uniform_prior_density(pars0, pars_min, pars_max)  # Log
        if isinstance(ipar, list) and (len(pars0) == 2):
            f0 += _gaussian_prior_density(pars0[1], b_prior_mean * np.log(10), b_prior_std * np.log(10))  # Log

        return f - f0  # Log prior probability increment


    def _check_array_dims(var, required_ndim, required_shape, namespace):
        varname = [name for name in namespace.keys() if namespace[name] is var][0]
        if var.ndim != required_ndim:
            raise ValueError(f'input array {varname} must have {required_ndim} dimensions!')
        if np.sum(np.array(var.shape) - np.array(required_shape)) != 0:
            raise ValueError(f'input array {varname} must have shape {tuple(required_shape)} (actual shape is {var.shape})')


    def _proposal_all_prms(rng, x0, fun_type='uniform', xmin=None, xmax=None, sigma=None):
        """
        Propose a candidate parameter set for all parameters simultaneously
        """
        n = len(x0)
        if fun_type == "uniform":
            x = xmin + (xmax - xmin) * rng.uniform(size=n)
            ratio = 1
        elif fun_type == "uniform-local":
            dx = xmax - xmin
            x = x0 + dx * sigma * (1 - 2 * rng.uniform(size=n))
            ratio = 1
        elif fun_type == "normal":
            x = rng.normal(x0, sigma, size=n)
            ratio = 1
        return x, ratio


    def _proposal_local(rng, icell, x0, fun_type='uniform', xmin=None, xmax=None, sigma=None):
        """
        Propose a candidate parameter set by changing one single parameter in the current set x0
        """
        x = deepcopy(x0)
        n = len(x0)
        m = int((n - 2) / 2)
        i = _icell2iparam(icell, m, n)
        if isinstance(i, int):
            np = 1
        elif isinstance(i, list):
            np = len(i)
        if fun_type == "uniform":
            x[i] = xmin[i] + (xmax[i] - xmin[i]) * rng.uniform(size=np)
            ratio = 1
        elif fun_type == "uniform-local":
            dx = xmax[i] - xmin[i]
            x[i] = x0[i] + dx * sigma * (1 - 2 * rng.uniform(size=np))
            ratio = 1
        elif fun_type == "normal":
            x[i] = rng.normal(x0[i], sigma)
            ratio = 1
        return x, ratio


    # Defensive programming: Check shape and type of all input variables
    m = len(areas)  # number of cells
    q = len(m_weights)
    k = mmax_pmf.shape[1]
    _check_array_dims(nums, 2, (m, q), locals())
    _check_array_dims(areas, 1, (m,), locals())
    _check_array_dims(lons, 1, (m,), locals())
    _check_array_dims(lats, 1, (m,), locals())
    _check_array_dims(m_bins, 2, (q, 2), locals())
    _check_array_dims(m_periods, 2, (m, q), locals())
    _check_array_dims(mmax_pmf, 3, (m, k, 2), locals())
    if np.any(m_weights > 1.0):
        raise ValueError(f'input weights for magnitude bins must be <= 1.0')
    if np.any(m_periods == 0.0):
        raise ValueError(f'Completeness periods must all be greater than 0.0')
    if np.any(areas == 0.0):
        raise ValueError(f'Cell areas must all be greater than 0.0')
    if np.any(np.isinf(prior['mins'])):
        raise ValueError(f'At least one infinite value in minimum bounds for prior')
    if np.any(np.isinf(prior['maxs'])):
        raise ValueError(f'At least one infinite value in maximum bounds for prior')

    # Initialization of variables:
    ulons = np.unique(lons)
    ulats = np.unique(lats)
    dlon = ulons[1] - ulons[0]
    dlat = ulats[1] - ulats[0]
    laplacian = np.zeros((m, m))  # Laplacian operator
    for i in range(m):
        # Find indices of cells surrounding each cell:
        jw = np.where(np.logical_and(lons == lons[i] - dlon, lats == lats[i]))  # Index of cell on the west
        je = np.where(np.logical_and(lons == lons[i] + dlon, lats == lats[i]))  # Index of cell on the east
        js = np.where(np.logical_and(lons == lons[i], lats == lats[i] - dlat))  # Index of cell at south
        jn = np.where(np.logical_and(lons == lons[i], lats == lats[i] + dlat))  # Index of cell at north
        laplacian[i, i] = 1 + 1 / (np.cos(lats[i] * np.pi / 180.0)) ** 2
        if (len(jw) == 0) and (len(je) > 0):
            laplacian[i, je] = -1.0
        elif (len(jw) > 0) and (len(je) == 0):
            laplacian[i, jw] = -1.0
        else:
            laplacian[i, jw] = -0.5
            laplacian[i, je] = -0.5
        if (len(jn) == 0) and (len(js) > 0):
            laplacian[i, js] = -1.0
        elif (len(jn) > 0) and (len(js) == 0):
            laplacian[i, jn] = -1.0
        else:
            laplacian[i, jn] = -0.5
            laplacian[i, js] = -0.5
        laplacian[i, :] = -0.5 * laplacian[i, :]
    rng = np.random.default_rng(seed)
    # Earthquake rates per cell per unit time per unit area for events with mag >= m0
    rates = np.sum(nums / m_periods, axis=1) / areas
    # weights for magnitude bins #TODO: improve, see Petersen (2008)
    m_w = np.array(m_weights)
    if m0 is None:
        m0 = m_bins[:, 0].min()
        print(f"INFO: Magnitude of completeness automatically set to {m0}")
    sampling_strategy = proposal.pop('strategy')
    proposal.update({'xmin': prior['mins'], 'xmax': prior['maxs']})
    npar = 2 * m + 2  # Number of unknowns parameters

    # Set-up initial conditions for MCMC:
    prms0 = prior['mins'] + 0.5 * (prior['maxs'] - prior['mins'])
    post0 = _prior(prms0, prior['b_mean'], prior['b_std'], prior['mins'], prior['maxs'])
    if not only_sample_prior:
        post0 += _likelihood(prms0, areas, nums, m_bins, m_periods, m_w, m0, mmax_pmf, laplacian,
                             smoothing=apply_smoothing)

    n_accepted = 0
    post_max = -np.inf
    n_trial = 0
    if n_init is None:
        n_init = int(max_trials / 4)
        print(f"INFO: Burn-in phase automatically set to {n_init} iterations")

    fp = open(output_file, 'wb')  # Binary file, Little-endian
    # Header format:
    # Nb. cells; Centroid longitudes; Centroid latitudes; Min. magnitude
    fp.write(pack(f'<i{m}d{m}df', m, *(lons.astype('d').tolist()), *(lats.astype('d').tolist()), m0))
    struct_formatter = Struct(f'<ldf{2 * m + 2}d')

    if sampling_strategy == 'incremental':
        collection_interval = npar
        print(f'>> NOTE: Incremental sampling strategy --> modified collection interval to {npar}')
    pbar = ProgressBar(imax=max_trials, title='MCMC initialization')

    # Start MCMC:
    icell = -1
    while n_trial < max_trials:

        # a- Generate new sample and compute posterior (penalized likelihood) for new trial:
        n_trial += 1
        coef = 0.1
        if sampling_strategy in ["simultaneous", "all"]:
            prms, ratio = _proposal_all_prms(rng, prms0, **proposal)
            post = _prior(prms, prior['b_mean'], prior['b_std'], prior['mins'], prior['maxs'])
            if not only_sample_prior:
                post += _likelihood(prms, areas, nums, m_bins, m_periods, m_w, m0, mmax_pmf, laplacian,
                                    smoothing=apply_smoothing)
        else:
            if icell == m + 1:
                # Re-initialize loop on parameters:
                icell = 0
            else:
                # increment cell index for the proposal
                icell += 1
            prms, ratio = _proposal_local(rng, icell, prms0, **proposal)
            post = post0 \
                   + _prior_increment(prms, icell, prms0, prior['b_mean'], prior['b_std'], prior['mins'], prior['maxs'])
            if not only_sample_prior:
                post += _likelihood_increment(prms, icell, prms0, areas, nums, m_bins, m_periods, m_w, m0, mmax_pmf,
                                              laplacian, smoothing=apply_smoothing)

        # b- Check acceptance:
        if (post == -np.inf) and (post0 == -np.inf):
            alpha = 0  # Force rejection when prior or likelihood is 0 and log(prior or likelihood) --> -inf
        else:
            alpha = np.min(np.array([1, np.exp(post - post0) * ratio]))
        if rng.uniform() < alpha:
            # Accept current parameter set:
            post0 = post
            prms0 = prms
            n_accepted += 1
            if post > post_max:
                post_max = post

        # c- Collect parameter set:
        acceptance_rate = n_accepted / n_trial
        if (n_trial > n_init) and (np.mod(n_trial, collection_interval) == 0):
            # Output format:
            # N_trial; Posterior LL; Acceptance rate; Current parameters values (2 * m + 2 ):
            fp.write(struct_formatter.pack(n_trial, post0, acceptance_rate, *prms0))

        # d- Periodically update progress bar:
        if np.mod(n_trial, np.maximum(1, max_trials / 1000)) == 0:
            if n_trial <= n_init:
                pbar.update(i=n_trial, title='MCMC initialization')
            else:
                pbar.update(i=n_trial, title=f'MCMC exploration ({n_trial} iter.)')

    print(f'>> INFO: Total number of iterations: {n_trial}')
    print(f'>> INFO: Max. log-posterior: {post_max}')
    print(f'>> INFO: Average acceptance rate: {100 * n_accepted / n_trial:.2f}%')
    fp.close()
    return None


def _load_only_cell_coordinates(file='mcmc_samples.out'):
    with open(file, 'rb') as fp:
        print(f'Reading header...')
        nc = unpack('<i', fp.read(4))[0]  # First read number of cells
        fmt = f'<{nc}d{nc}df'
        fmt_len = calcsize(fmt)
        unpacked_vars = unpack(fmt, fp.read(fmt_len))
    lons = np.array(unpacked_vars[:nc])
    lats = np.array(unpacked_vars[nc:(2 * nc)])
    m0 = unpacked_vars[2 * nc]
    print(f'>> Longitude range: [{lons.min()}; {lons.max()}]')
    print(f'>> Latitude range: [{lats.min()}; {lats.max()}]')
    return lons, lats, nc, m0


def load_mcmc_results(file="mcmc_samples.out", subsampling_step=1, grid_reshape=None,
                      additive_scaling_for_a=0.0):
    """
    Reads parameters for all MCMC iterations and returned them arranged as
    (Ntrials-by-Ncells) numpy.ndarrays
    """
    lons, lats, nc, m0 = _load_only_cell_coordinates(file)
    print(f'Loading samples...', end='')
    npar = 2 * nc + 2  # Nb of parameters
    index = []
    post = []
    accept = []
    log10rates = []
    beta = []
    sm_beta = []
    sm_lograte = []
    cnt = 0
    with open(file, 'rb') as fp:
        fp.read(calcsize(f'<i{nc}d{nc}df'))  # Skip header
        while True:
            fmt_iter = f'<ldf{npar}d'
            fmt_len = calcsize(fmt_iter)
            buffer = fp.read(fmt_len)
            if buffer == b'':  # when EOF reached...
                print('Done.')
                break
            else:
                unpacked_vars = unpack(fmt_iter, buffer)
                index.append(unpacked_vars[0])
                post.append(unpacked_vars[1])
                accept.append(unpacked_vars[2])
                log10rates.append(unpacked_vars[3:(3 + nc)])
                beta.append(unpacked_vars[(3 + nc):(3 + 2 * nc)])
                sm_beta.append(unpacked_vars[3 + 2 * nc])
                sm_lograte.append(unpacked_vars[3 + 2 * nc + 1])
                cnt += 1
    print(f'>> Number of samples read: {cnt}')

    index = np.array(index)   # iteration indices
    post = np.array(post)     # log-posterior values
    accept = np.array(accept) # acceptance rate (between 0 and 1)
    log10rates = np.array(log10rates)  # log10(rate)
    beta = np.array(beta)     # beta values
    sm_beta = np.array(sm_beta)  # smoothing factor for beta
    sm_lograte = np.array(sm_lograte)  # smoothing factor for log10(rate)

    if subsampling_step > 1:
        # If required, sub-sample values:
        sub = [i for i in range(0, cnt, subsampling_step)]
        index = index[sub]
        post = post[sub]
        accept = accept[sub]
        log10rates = log10rates[sub, :]
        beta = beta[sub, :]
        sm_beta = sm_beta[sub]
        sm_lograte = sm_lograte[sub]

    # Compute Gutenberg-Richter (a, b) values:
    b = beta / np.log(10)
    a = log10rates + b * m0  # Gutenberg-Richter a coefficient: Log10( N(M>=0) ), per unit time and unit area
    a = a + additive_scaling_for_a

    if grid_reshape is not None:
        nlat, nlon = grid_reshape  # in this case, grid_shape = (nlat, nlon)
        assert nlat * nlon == nc
        a = np.reshape(a, (a.shape[0], nlat, nlon), order='C')
        b = np.reshape(b, (b.shape[0], nlat, nlon), order='C')
    return lons, lats, index, post, accept, a, b, sm_beta, sm_lograte


def plot_detailed_results(file="mcmc_samples.csv", subsampling_step=1,
                          additive_scaling_for_a=0.0, verbose=False):
    """
    Display results of the MCMC exploration algorithm implemented in the penalized_likelihood() method.

    :param file:
    :return:
    """
    lons, lats, index, post, accept, a, b, sm_beta, sm_lograte = \
        load_mcmc_results(file=file, subsampling_step=subsampling_step, additive_scaling_for_a=additive_scaling_for_a)
    nc = a.shape[1]

    # Convergence plots:
    fig = plt.figure()
    for i in range(nc):
        plt.plot(index, b[:,i])
    plt.xlabel('index')
    plt.ylabel('b')

    fig = plt.figure()
    for i in range(nc):
        plt.plot(index, a[:,i])
    plt.xlabel('index')
    plt.ylabel('a')

    fig = plt.figure()
    plt.plot(index, sm_beta, label='for $\\beta$')
    plt.plot(index, sm_lograte, label='for a')
    plt.xlabel('index')
    plt.ylabel('smoothing parameters')
    plt.legend()

    fig = plt.figure()
    plt.plot(index, post)
    plt.xlabel('index')
    plt.ylabel('log-posterior')

    fig = plt.figure()
    plt.plot(index, accept)
    plt.xlabel('index')
    plt.ylabel('acceptance rate')

    # Grouped distribution in 2D-histograms plots for beta and log10(rates)
    ip = np.meshgrid(index, np.arange(nc), indexing='ij')[1]  # parameter index
    ip = ip + 1  # Set indexing from 1 to nc
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
    fig.set_size_inches(8, 12)

    if nc > 1:
        xbins = np.linspace(0.5, ip.max() + 0.5, ip.max())
        ybins = np.linspace(a.min(), a.max(), 50)
        outs = axs[0].hist2d(ip.flatten(), a.flatten(), bins=[ip.max(), 50])
        h2d = outs[3]  # QuadMesh collection, with cmap and norm properties
        axs[0].set_xlabel('Cell index')
        axs[0].set_ylabel('a')
        cax, kw = colorbar.make_axes(axs[0], location='right')
        cbar = colorbar.ColorbarBase(cax, cmap=h2d.cmap, norm=h2d.norm)
        cbar.set_label('Counts')

        ybins = np.linspace(b.min(), b.max(), 50)
        outs = axs[1].hist2d(ip.flatten(), b.flatten(), bins=[ip.max(), 50])
        h2d = outs[3]  # QuadMesh collection, with cmap and norm properties
        axs[1].set_xlabel('Cell index')
        axs[1].set_ylabel('b')
        cax, kw = colorbar.make_axes(axs[1], location='right')
        cbar = colorbar.ColorbarBase(cax, cmap=h2d.cmap, norm=h2d.norm)
        cbar.set_label('Counts')


    # Distributions:
    fig, axs = plt.subplots(nrows=1, ncols=4)
    fig.set_size_inches(14, 5)
    for j, (var, var_title) in enumerate(
            zip([a, b, sm_beta, sm_lograte],
                ['a', 'b', 'sm_beta', 'sm_log10(nu)'])):
        if verbose:
            if len(var.shape) > 1:
                for k in range(var.shape[1]):
                    opt_value = var[post.argmax(), k]
                    print(f'{var_title}[{k}]: opt = {opt_value}  mean = {var[:,k].mean(axis=0)}  std = {var[:,k].std(axis=0)}')
            else:
                opt_value = var[post.argmax()]
                print(f'{var_title}: opt = {opt_value}  mean = {var.mean()}  std = {var.std()}')
        axs[j].hist(var, bins=50, histtype='bar')
        axs[j].set_xlabel(var_title)
    plt.show()

