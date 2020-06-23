import numpy as np
import pandas as pd
from scipy.special import gamma
from scipy.stats import chi2, f
from numba import njit, prange
import statsmodels.api as sm
import matplotlib.pylab as plt


def R2(null_dev, dev):
    return (null_dev - dev)/null_dev


def BIC(ll, n, p):
    return -2*ll + np.log(n)*p


def AIC(ll, p):
    return -2*(ll - p)


def gamma_predict(tau, X, link='identity'):

    if link == 'identity':
        mu = X@tau
        mu = np.clip(mu, a_min=1e-6, a_max=np.inf)
    elif link == 'log':
        mu = np.exp(X@tau)
        mu = np.clip(mu, a_min=1e-6, a_max=np.inf)
    elif link == 'inverse':
        mu = -1./(X@tau)
    else:
        raise NotImplementedError

    return mu


def gamma_pearson_residuals(tau, X, y, u=None, link='identity', aggregate='msr'):

    if u is None:
        u = np.ones(len(X))

    mu = gamma_predict(tau, X, link=link)

    e = (1./u)*(y - mu)/mu

    if aggregate == 'mean':
        return np.mean(e)
    elif aggregate == 'msr':
        return np.mean(e**2)
    elif aggregate is None:
        return e
    else:
        raise NotImplementedError


def gamma_deviance(tau, X, y, u=None, link='identity', aggregate='sum'):

    if u is None:
        u = np.ones(len(X))

    mu = gamma_predict(tau, X, link=link)

    dev = (1./u)*2.*((y - mu)/mu + np.log(mu/y))

    if aggregate == 'sum':
        return np.sum(dev)
    elif aggregate == 'mean':
        return np.mean(dev*np.sign(y - mu))
    elif aggregate == 'msr':
        return np.mean(dev**2)
    elif aggregate == 'sumr':
        return np.sum(np.sqrt(dev)*np.sign(y - mu))
    elif aggregate is None:
        return dev
    else:
        raise NotImplementedError


def gamma_loglik(tau, X, y, u=None, link='identity', nu=0.5):

    mu = gamma_predict(tau, X, link=link)

    snp_ll = -nu*(y/mu + np.log(mu)) + nu*np.log(nu*y) - np.log(gamma(nu)*y)

    if u is None:
        return np.sum(snp_ll)
    else:
        return np.dot(1./u, snp_ll)


def approx_gamma_loglik(tau, X, Y, u=None, link='identity'):
    """
    The Speed et al. loglikelihood
    """
    if u is None:
        u = np.ones(len(X))

    mu = gamma_predict(tau, X, link=link)

    snp_ll = Y / (2. * mu) + np.log(Y) + np.log(2 * mu) + np.log(np.pi)

    return -.5 * np.dot(1. / u, snp_ll)


def score(tau, X, Y, u=None, link='identity'):

    mu = gamma_predict(tau, X, link=link)

    if link == 'identity':
        g_prime = np.ones(len(X))
    elif link == 'log':
        g_prime = 1. / mu
    elif link == 'inverse':
        g_prime = 1. / (mu ** 2)
    else:
        raise NotImplementedError

    if u is not None:
        g_prime *= u

    D = 1. / g_prime
    V = 1. / (mu ** 2)

    score = X.T @ (D*V*(Y - mu))

    return score


def hessian(tau, X, Y, u=None, use_fisher=True, link='identity'):

    mu = gamma_predict(tau, X, link=link)

    if link == 'identity':
        if use_fisher:
            Y = mu
        W = (mu - 2*Y) / (mu ** 3)
    elif link == 'log':
        if use_fisher:
            Y = mu
        W = -Y/mu
    elif link == 'inverse':
        W = -mu**2
    else:
        raise NotImplementedError

    if u is not None:
        W *= 1./u

    H = np.zeros((X.shape[1], X.shape[1]))
    for i in prange(X.shape[1]):
        for j in prange(i, X.shape[1]):
            H[i, j] = H[j, i] = np.dot(X[:, i].T, W*X[:, j])

    return H


def fit_gamma_model(X, y, u=None, nu=0.5, link='identity',
                    max_iter=30, tol=1e-8, tol_criterion='deviance', max_dec=10,
                    use_fisher=True, individual_update=False, update_method='solve'):

    t = 0
    n, p = X.shape

    history = {
        'deviance': [],
        'params': []
    }

    # Initialize with the null model (all parameters are zero except the intercept):
    tau_t = np.zeros(p)
    tau_t[0] = 1.
    converged = False
    count_decrease = 0

    deviance_t = gamma_deviance(tau_t, X, y, u=u, link=link)

    history['deviance'].append(deviance_t)
    history['params'].append(tau_t)

    while t < max_iter and not converged and count_decrease < max_dec:

        print(t)
        mu_t = score(tau_t, X, y, u=u, link=link)
        H_t = hessian(tau_t, X, y, u=u, link=link, use_fisher=use_fisher)

        if update_method == 'solve':
            tau_new = np.linalg.solve(H_t, H_t @ tau_t - mu_t)
        elif update_method == 'invert':
            iH_t = np.linalg.inv(H_t)
            tau_new = tau_t - iH_t @ mu_t
        elif update_method == 'lstsq':
            tau_new, _, _, _ = np.linalg.lstsq(H_t, H_t@tau_t - mu_t)
        else:
            raise NotImplementedError

        deviance_t = gamma_deviance(tau_new, X, y, u=u, link=link)

        if nu is None:
            nu_t = (n - p)/history['deviance'][-1]
            nu_new = (n - p)/deviance_t
        else:
            nu_t = nu
            nu_new = nu

        if (gamma_loglik(tau_new, X, y, u=u, link=link, nu=nu_new) -
                gamma_loglik(tau_t, X, y, u=u, link=link, nu=nu_t) < 0.):
            print(f"Warning: Decreasing loglikelihood...")
            count_decrease += 1
            if individual_update:
                for i in prange(p):
                    tau_new[i] = tau_t[i] - (1./H_t[i, i])*mu_t[i]

                deviance_t = gamma_deviance(tau_new, X, y, u=u, link=link)

        history['params'].append(tau_new)
        history['deviance'].append(deviance_t)

        converged = np.allclose(history[tol_criterion][-1], history[tol_criterion][-2], atol=tol)

        tau_t = tau_new
        t += 1

    if nu is None:
        nu_t = (n - p) / deviance_t
    else:
        nu_t = nu

    return {
        "params": history['params'][-1],
        "llf": gamma_loglik(tau_t, X, y, u=u, link=link, nu=nu_t),
        "dispersion": nu_t,
        "deviance": deviance_t,
        "deviance_mr": gamma_deviance(tau_t, X, y, u=u, link=link, aggregate='mean'),
        "deviance_msr": gamma_deviance(tau_t, X, y, u=u, link=link, aggregate='msr'),
        "pearson_mr": gamma_pearson_residuals(tau_t, X, y, u=u, link=link, aggregate='mean'),
        "pearson_msr": gamma_pearson_residuals(tau_t, X, y, u=u, link=link),
        "converged": converged,
        "history": history
    }


def fit_ldscore_model(ld_df, ld_col_names, w_col_name=None,
                      chisq_col='CHISQ',
                      null_fit_intercept=True, **kwargs):

    X = ld_df.loc[:, ['N'] + ld_col_names].values
    X[:, 0] = 1.

    y = np.fmax(1e-6, ld_df[chisq_col])

    if w_col_name is None:
        u = None
    else:
        u = np.maximum(ld_df[w_col_name].values, 1.)

    n, p = X.shape

    # -----------------------------------------------
    # Model fit:
    full_model = fit_gamma_model(X, y, u=u, **kwargs)

    # Null fit (intercept only):
    if null_fit_intercept:
        null_model = fit_gamma_model(X[:, :1], y, u=u, **kwargs)
    else:
        null_model = fit_gamma_model(X[:, :1], y, u=u, max_iter=0, **kwargs)

    # -----------------------------------------------

    if 'nu' not in kwargs:
        F_stat = (null_model['deviance'] - full_model['deviance']) / ((p - 1)*full_model['dispersion'])
        pval = 1. - f(p-1, n-p).cdf(F_stat)
    else:
        chi_stat = (null_model['deviance'] - full_model['deviance']) / kwargs['nu']
        pval = 1. - chi2(p-1).cdf(chi_stat)

    r2 = R2(null_model['deviance'], full_model['deviance'])

    params = full_model['params']
    params[1:] /= ld_df['N'].values[0]

    return {
        "LL": full_model['llf'],
        "null_LL": full_model['llf'],
        "AIC": AIC(full_model['llf'], p),
        "BIC": BIC(full_model['llf'], n, p),
        "R2": r2,
        "adjR2": 1. - (1. - r2)*(n - 1)/(n - p - 1),
        "LRT": -2.*(null_model['llf'] - full_model['llf']),
        "Pearson_MSR": full_model['pearson_msr'],
        "Deviance_MSR": full_model['deviance_msr'],
        "p-value": pval,
        "coef": params[1:],
        "intercept": params[0],
        "dispersion": full_model['dispersion'],
        "converged": full_model["converged"]
    }


def get_model_lrt(coef, est_intercept,
                  ld_df, ld_col_names, w_col_name,
                  chisq_col='CHISQ',
                  null_fit_intercept=False):

    X = ld_df.loc[:, ['N'] + ld_col_names].values
    X[:, 0] = 1.

    y = np.fmax(1e-6, ld_df[chisq_col])

    u = np.maximum(ld_df[w_col_name].values, 1.)

    fitted_tau = np.append([est_intercept], coef)
    fitted_ll = gamma_loglik(fitted_tau, X, y, u=u)

    if null_fit_intercept:
        null_model = fit_gamma_model(X[:, :1], y, u=u)
        null_ll = null_model['llf']
    else:
        null_tau = np.append([1], np.zeros_like(coef))
        null_ll = gamma_loglik(null_tau, X, y, u=u)

    return -2.*(null_ll - fitted_ll)

