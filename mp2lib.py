import numpy as np
from scipy.special import erfc

def inv_cdf_exponential(u, lam):
    return -np.log1p(-np.asarray(u)) / lam

def inv_cdf_rayleigh(u, sigma):
    return sigma * np.sqrt(-2 * np.log1p(-np.asarray(u)))

def sample_by_inversion(inv_cdf, n, rng, **kw):
    u = rng.random(n)
    return inv_cdf(u, **kw)

def ks_statistic(samples, cdf):
    samples = np.sort(np.asarray(samples))
    n = len(samples)

    theoretical = np.asarray(cdf(samples))
    empirical_upper = np.arange(1, n + 1) / n
    empirical_lower = np.arange(n) / n

    return float(max(
        np.max(empirical_upper - theoretical),
        np.max(theoretical - empirical_lower)
    ))

def q_function(x):
    return 0.5 * erfc(np.asarray(x) / np.sqrt(2))