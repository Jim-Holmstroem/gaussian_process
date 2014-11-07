from __future__ import print_function

from itertools import *
from functools import *

import numpy as np

import matplotlib.pyplot as plt


def sample_path(domain, autocorrelation):
    """
    domain :: ndarray(time)
        s, t \in domain, could possibly be irregular.

    autocorrelation :: *(t :: float, s :: float) -> float
        E[X(s)X(t)]
    """
#    observations = np.empty_like(domain)
#    observations[0] = 0.0
#    last_observed = 0


def autocorrelation(
    autocorr,
    domain,
    other_domain=None,
):
    other_domain = domain if other_domain is None\
        else other_domain

    autocorr_value = np.matrix(np.frompyfunc(autocorr, 2, 1).outer(
        domain,
        other_domain,
    ).astype(np.float64))

    return autocorr_value


def inference((X_domain, X_value), autocorr):
    """
    Y | X = X_value
    """
    def _inference(Y_domain):
        YgX_domain = Y_domain
        mu_X = 0
        mu_Y = 0
        _autocorrelation = partial(autocorrelation, autocorr)
        Sigma_XX = _autocorrelation(X_domain)
        Sigma_XY = _autocorrelation(X_domain, Y_domain)
        Sigma_YX = Sigma_XY.T
        Sigma_YY = _autocorrelation(Y_domain)

        SigmaYXinvXX = Sigma_YX * np.linalg.inv(Sigma_XX)
        mu_YgX = mu_Y + SigmaYXinvXX * (X_value - mu_X)
        Sigma_YgX = Sigma_YY - SigmaYXinvXX * Sigma_XY

        return YgX_domain, mu_YgX, Sigma_YgX

    return _inference

cov = lambda s, t: np.exp(-np.abs(t-s))
#cov = lambda s, t: (1 + (t-s)**2/(2*2*1**2)) ** 2  # rational quadratic
#cov = lambda s, t: np.minimum(s, t)

N = 64

X_domain = np.matrix(np.random.uniform(0.0, 1.0, (N, 1)))

X_value = np.matrix(np.random.multivariate_normal(
    np.zeros((N,)),  # TODO mu_X
    autocorrelation(cov, X_domain)
)).T

M = 1024
Y_domain = np.matrix(np.arange(0.0, 1.0, 1.0 / M)).T
posterior = inference(
    (
        X_domain,
        X_value,
    ),
    cov
)

domain, mu, Sigma = posterior(Y_domain)

plt.hold(True)
plt.scatter(X_domain, X_value)
plt.plot(
    domain,
    mu,
    domain,
    mu + np.outer(
        [-2, -1, 1, 2],
        np.sqrt(Sigma.diagonal())
    ).T,
)

plt.show()
