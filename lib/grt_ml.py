"""
Estimation of (a,b) parameters for a truncated
Gutenberg-Richter model of the magnitude-frequency
distribution, based on the formulas published in:

Dutfoy, A., 2020, Estimation of the Gutenberg-Richter
Earthquake Recurrence Parameters for Unequal Observation
Periods and Imprecise Magnitudes, Pure App. Geophysics, 177,
10, 4597-4606, doi:10.1007/s00024-020-02551-8.
"""
import numpy as np
from scipy.optimize import minimize_scalar


def cumulative_fmd(mags, a, b, dm, m0=0.0, mmax=np.inf):
    """
    Return the cumulative number of earthquakes expected at magnitude MAGS, given a
    set of parameters for the truncated Gutenberg-Richter model

    :param mags: central values for each magnitude bin
    :param a, b: parameters of the Gutenberg-Richter law, with a = log10(lambda(M >= 0))
    :param dm: magnitude bin-width
    :param m0: optional, minimum magnitude (completeness). Default: 0.0.
    :param mmax: optional, maximum magnitude (truncation). Default: No truncation
    """
    l0 = (10 ** a) * np.exp(-b * np.log(10) * m0)
    ibc = np.where(mags - dm / 2 < m0)  # Lower bin edge below completeness
    num = 1 - 10 ** (-b * (mags - dm / 2 - m0))
    num[ibc] = 1
    denum = 1 - 10 ** (-b * (mmax - m0))
    cdf = num / denum
    cdf[cdf > 1.0] = 1.0
    return  l0 * (1 - cdf)


class Dutfoy2020_Estimator():

    def __init__(self, central_mags: np.ndarray, durations: np.ndarray, counts: np.ndarray, m0, mmax, dm):
        """Class for the computation of the truncated Gutenberg-Richter model log-likelihood.

        Args:
            central_mags (np.ndarray): Central magnitude for each bin
            durations (np.ndarray): Duration of completeness interval for each bin
            counts (np.ndarray): Number of observed earthquakes in each bin over the period of completeness
            m0 (_type_): Minimum magnitude of the truncated Gutenberg-Richter model
            mmax (_type_): Maximum magnitude of the truncated Gutenberg-Richter model
            dm (_type_): (constant) magnitude bin width. TODO: enable variable bin widths in all bins.
        """
        self.nb = len(central_mags)
        self.mags = central_mags.reshape((1, self.nb))
        assert len(durations) == self.nb
        self.durs = durations.reshape((1, self.nb))
        assert len(counts) == self.nb
        self.cnts = counts.reshape((1, self.nb))
        self.N = counts.sum()
        self.m0 = m0
        self.mmax = mmax
        self.dm = dm
        self.delta = self.dm / 2

    def __call__(self, ab: np.ndarray):
        return self.LL(ab)

    def ab2mubeta(self, ab: np.ndarray):
        """
        See eqn (7)
        """
        a = ab[:, 0]
        b = ab[:, 1]
        k0 = np.power(10, -b * self.m0)
        kmax = np.power(10, -b * self.mmax)
        mu = np.power(10, a + np.log10((k0 - kmax) / (1 - kmax)))
        return mu, b * np.log(10)

    def mubeta2ab(self, mubeta: np.ndarray):
        """
        See eqn (7)
        """
        mu = mubeta[:, 0]
        beta = mubeta[:, 1]
        b = beta / np.log(10)
        k0 = np.exp(-beta * self.m0)
        kmax = np.exp(-beta * self.mmax)
        a = np.log10(mu) - np.log10((k0 - kmax) / (1 - kmax))
        return a, b

    def alpha(self, beta):
        """
        See eqn (8)
        """
        return np.exp(-beta * self.m0) - np.exp(-beta * self.mmax)

    def gamma(self, beta):
        """
        See below eqn (13)
        """
        return self.mmax * np.exp(-beta * self.mmax) - self.m0 * np.exp(-beta * self.m0)

    def TiUi(self, beta: np.ndarray):
        n = len(beta)
        beta = np.tile(beta.reshape((n, 1)), (1, self.nb))
        Di = np.tile(self.durs, (n, 1))
        ci = np.tile(self.mags, (n, 1))
        ni = np.tile(self.cnts, (n, 1))
        T0 = np.sum(Di * np.exp(-beta * ci), axis=1)
        T1 = np.sum(ci * Di * np.exp(-beta * ci), axis=1)
        T2 = np.sum(ci * ci * Di * np.exp(-beta * ci), axis=1)
        U0 = np.sum(ni * np.log(Di), axis=1)
        U1 = np.sum(ni * ci, axis=1)
        U2 = np.sum(ni * ci * ci, axis=1)
        return T0, T1, T2, U0, U1, U2

    def LL(self, ab: np.ndarray):
        """
        see eqn (12)

        """
        mu, beta = self.ab2mubeta(ab)
        T0, T1, T2, U0, U1, U2 = self.TiUi(beta)
        r = np.sinh(beta * self.delta) / self.alpha(beta)
        return -2 * mu * r * T0 + self.N * np.log(mu) + U0 + self.N * np.log(2 * r) - beta * U1

    def LL_with_prior(self, ab: np.ndarray, prior_fun):
        """
        see eqn (12), added of the log of a prior probability on b.

        NB: PRIOR_FUN must be a callable returning log(pdf(b)) and must be
        parameterized as a function of b, not beta !
        """
        mu, beta = self.ab2mubeta(ab)
        T0, T1, T2, U0, U1, U2 = self.TiUi(beta)
        r = np.sinh(beta * self.delta) / self.alpha(beta)
        return -2 * mu * r * T0 + self.N * np.log(mu) + U0 + self.N * np.log(2 * r) - beta * U1 + prior_fun(
            beta / np.log(10))

    def LL_with_normal_prior(self, ab: np.ndarray, mean_b, std_b):
        """
        see eqn (12), added of the log of normal prior probability on b,
        with mean MEAN_B and standard deviation STD_B.
        """
        mu, beta = self.ab2mubeta(ab)
        T0, T1, T2, U0, U1, U2 = self.TiUi(beta)
        r = np.sinh(beta * self.delta) / self.alpha(beta)
        mean_beta = mean_b * np.log(10)
        std_beta = std_b * np.log(10)
        log_prior_term = np.log(1 / (std_beta * np.sqrt(2))) - 0.5 * np.power((beta - mean_beta) / std_beta, 2)
        return -2 * mu * r * T0 + self.N * np.log(mu) + U0 + self.N * np.log(2 * r) - beta * U1 + log_prior_term

    def _beta_root_function(self, beta):
        """
        See eqn (18)
        """
        T0, T1, T2, U0, U1, U2 = self.TiUi(np.array([beta]))
        return np.abs((T1 / T0) - (U1 / self.N))

    def _beta_root_function_with_normal_prior(self, beta, mean_beta, std_beta):
        """
        See eqn (18), modified to include a term to account for a normal prior
        probability on b (with mean=mean_b and std. dev.=std.b).
        """
        T0, T1, T2, U0, U1, U2 = self.TiUi(np.array([beta]))
        return np.abs((T1 / T0) - (U1 / self.N) - (beta - mean_beta) / (self.N * std_beta**2))

    def _mu_opt(self, beta_opt):
        """
        See eqn (17)
        """
        T0 = self.TiUi(beta_opt)[0]
        return (self.N * self.alpha(beta_opt)) / (2 * np.sinh(beta_opt * self.delta) * T0)

    def find_optimal_ab_no_prior(self):
        """
        Search for optimal (a,b) parameters, see eqns (17) and (18)
        """
        beta0 = np.log(10)
        res = minimize_scalar(self._beta_root_function,
                              method='bounded',
                              bounds=[0.5 * np.log(10), 3.0 * np.log(10)]
                              )
        N = np.sum(self.cnts * self.durs)
        beta = res.x
        mu = self._mu_opt(beta)
        a, b = self.mubeta2ab(np.array([[mu, beta]]))
        rho, cov = self.correlation_coef(np.array((a,b)))
        return a[0, 0], b[0, 0], rho, cov

    def find_optimal_ab_no_prior_b_truncated(self, bounds_b):
        """
        Search for optimal (a,b) parameters, see eqns (17) and (18)
        B-values are truncated within the [bounds_b[0], bounds_b[1]] interval.
        NB: When truncation is requested, covariance and Fisher information analyses
            can be incorrect.
        """
        beta0 = np.log(10)
        res = minimize_scalar(self._beta_root_function,
                              method='bounded',
                              bounds=[bounds_b[0] * np.log(10), bounds_b[1] * np.log(10)]
                              )
        N = np.sum(self.cnts * self.durs)
        beta = res.x
        mu = self._mu_opt(beta)
        a, b = self.mubeta2ab(np.array([[mu, beta]]))
        rho, cov = self.correlation_coef(np.array((a,b)))
        return a[0, 0], b[0, 0], rho, cov

    def find_optimal_ab_with_normal_prior(self, mean_b, std_b):
        """
        Search for optimal (a,b) parameters, by minimizing the product of the
        log-likelihood function with a normal prior probability function on b (i.e.
        wth mean: mean_b, and std. dev.: std_b).
        """
        mean_beta = mean_b * np.log(10)
        std_beta = std_b * np.log(10)
        prior_args = (mean_beta, std_beta)
        res = minimize_scalar(self._beta_root_function_with_normal_prior,
                              method='bounded',
                              bounds=[0.5 * np.log(10), 3.0 * np.log(10)],
                              args=prior_args
                              )
        N = np.sum(self.cnts * self.durs)
        beta = res.x
        mu = self._mu_opt(beta)  # Note: similar to the case without prior
        a, b = self.mubeta2ab(np.array([[mu, beta]]))
        rho, cov = self.correlation_coef(np.array((a,b)), std_b=std_b)
        return a[0, 0], b[0, 0], rho, cov

    def find_optimal_ab_with_truncated_normal_prior(self, mean_b, std_b, bounds_b):
        """
        Search for optimal (a,b) parameters, by minimizing the product of the
        log-likelihood function with a normal prior probability function on b (i.e.
        wth mean: mean_b, and std. dev.: std_b).
        B-values are truncated within the [bounds_b[0], bounds_b[1]] interval.
        NB: When truncation is requested, covariance and Fisher information analyses
            can be incorrect.
        """
        mean_beta = mean_b * np.log(10)
        std_beta = std_b * np.log(10)
        prior_args = (mean_beta, std_beta)
        res = minimize_scalar(self._beta_root_function_with_normal_prior,
                              method='bounded',
                              bounds=[bounds_b[0] * np.log(10), bounds_b[1] * np.log(10)],
                              args=prior_args
                              )
        N = np.sum(self.cnts * self.durs)
        beta = res.x
        mu = self._mu_opt(beta)  # Note: similar to the case without prior
        a, b = self.mubeta2ab(np.array([[mu, beta]]))
        rho, cov = self.correlation_coef(np.array((a,b)), std_b=std_b)
        return a[0, 0], b[0, 0], rho, cov

    def _fisher_information_matrix(self, mubeta: np.ndarray, std_beta=np.inf):
        """
        Approximate the Fisher Information Matrix in the neighbourhood
        of the solution parameter.
        The influence of a normal prior on b (or beta) can be accounted for by specifying
        its standard deviation STD_BETA.
        See eqns (21), (22), (23) and (24).
        """
        mu = mubeta[0]
        beta = mubeta[1]
        T0, T1, T2, U0, U1, U2 = self.TiUi(np.array([beta]))
        d = self.delta
        cotanh = 1 / np.tanh(beta * d)
        gamma = self.gamma(beta)
        alpha = self.alpha(beta)

        # Gamma(beta,beta):
        G11 = 2 * U1[0] * (gamma / alpha - d * cotanh) \
              + self.N * ((gamma ** 2) / (alpha ** 2) + (T2[0] / T0[0]) + (d ** 2) * (cotanh ** 2)
                          - 2 * d * gamma * cotanh / alpha) + 1 / (std_beta**2)
        # Gamma(beta,mu):
        G12 = 2 * (T0[0] / alpha) * (d * np.cosh(beta * d) - (gamma / alpha) * np.sinh(beta * d)) \
              - 2 * np.sinh(beta * d) * T1[0] / alpha
        # Gamma(mu,mu):
        G22 = self.N / (mu ** 2)
        G = np.array([[G11, G12], [G12, G22]])
        return G

    def covariance_matrix(self, ab: np.ndarray, std_b=np.inf):
        """
        Compute covariance matrix around solution parameters (a, b)
        for the truncated Gutenberg-Richter model.
        See eqns (28), (29), (30) and (31)
        Covariance matrix ordered as follows: cov = [[var(b), cov(b,a)], [cov(a,b), var(a)]]
        """
        std_beta = std_b * np.log(10)
        mu, beta = self.ab2mubeta(ab.reshape((1, 2)))
        h = -(self.gamma(beta[0]) / self.alpha(beta[0]) -
              (self.mmax * np.exp(-beta[0] * self.mmax)) / (1 - np.exp(-beta[0] * self.mmax))
              ) / np.log(10)
        dgdx = np.array([[1 / np.log(10), 0.0], [h, 1 / (mu[0] * np.log(10))]])
        G = self._fisher_information_matrix(np.array([mu[0], beta[0]]), std_beta=std_beta)
        cov = dgdx @ np.linalg.inv(G) @ dgdx.T
        return cov

    def correlation_coef(self, ab: np.ndarray, std_b=np.inf):
        """
        Returns the Pearson correlation coefficient for parameters (a, b), optionally accounting
        for a prior normal distribution on b, with standard deviation STD_B.
        """
        cov = self.covariance_matrix(ab, std_b=std_b)
        rho = cov[0, 1] / (np.sqrt(cov[0, 0]) * np.sqrt(cov[1, 1]))
        return rho, cov
