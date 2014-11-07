from __future__ import print_function

import numpy as np


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


def inference((X_domain, X_value), autocorrelation):
    """
    Y | X = X_value
    """
    def _inference(Y_domain):
        YgX_domain = Y_domain
        mu_X = 0
        mu_Y = 0

        def _autocorrelation(
            domain,
            other_domain=None,
            autocorrelation=autocorrelation
        ):
            other_domain = domain if other_domain is None\
                else other_domain

            autocorr = np.matrix(np.frompyfunc(autocorrelation, 2, 1).outer(
                domain,
                other_domain,
            ).astype(np.float64))

            return autocorr

        Sigma_XX = _autocorrelation(X_domain)
        Sigma_XY = _autocorrelation(X_domain, Y_domain)
        Sigma_YX = Sigma_XY.T
        Sigma_YY = _autocorrelation(Y_domain)

        SigmaYXinvXX = Sigma_YX * np.linalg.inv(Sigma_XX)
        mu_YgX = mu_Y + SigmaYXinvXX * (X_value - mu_X)
        Sigma_YgX = Sigma_YY - SigmaYXinvXX * Sigma_XY

        return YgX_domain, mu_YgX, Sigma_YgX

    return _inference
