import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as stats
from scipy.special import binom, lambertw
from helper import add_intercept


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if device == torch.device("cuda"):
  import cupy as cp
  cp.random.seed()
else:
  np.random.seed()



# estimators: classical fitted beta; classical optimal beta; small data no repeats
# estimator: classical fitted beta
# single neuron
def r2_largedata_classic_fitbeta_norepeats(X, Y):
    """classic estimator of r2 for linear regression for the fit beta.
    No trial to trial noise
    No high-dim case (d>m)
    No DOF

    Parameters
    ----------
    X : numpy.ndarray
        n_observations by n_features regressor matrix
    Y : numpy.ndarray
        (m observations * 1) regressand matrix

    Returns
    -------
    est_r2 :  (k_neurons,) an estimate of the r2 between (over)fit linear
                combination from regressor matrix and the regressand.
    """

    # X_int = add_intercept(X)
    X_int = X.copy() ######## todo: to change back
    n_observations, n_features = X_int.shape

    if n_observations < n_features:
      est_r2 = 1. # does not work for high d case; return R2=1 instead
      return est_r2
    else:
      if device == torch.device("cpu"):
        beta_hat, _, _, _ = np.linalg.lstsq(X_int, Y, rcond = None)
        res = np.sum((X_int @ beta_hat - Y) ** 2 , 0)
        est_r2 = (1 - (res/n_observations)/np.var(Y, ddof=1, axis=0)).squeeze()
      else:

        beta_hat, _, _, _ = cp.linalg.lstsq(X_int, Y, rcond = None)
        res = cp.sum((X_int @ beta_hat - Y) ** 2 , 0)
        est_r2 = (1 - (res/n_observations)/cp.var(Y, ddof=1, axis=0)).squeeze().get()

      try:
        if len(est_r2.shape) == 0:
          est_r2 = float(est_r2)
      except: pass

    return est_r2


# estimators: classical optimal beta;
# single neuron
def r2_largedata_classic_optimalbeta_norepeats(X, Y, use_rss=True):
    """classic estimator of r2 for linear regression with optimal beta by fitting linear model
    No trial to trial noise
    No high-dim case (d>m)
    YES DOF

    Parameters
    ----------
    X : numpy.ndarray
        n_observations by n_features regressor matrix
    Y : numpy.ndarray
        (m observations * 1) regressand matrix

    Returns
    -------
    est_r2 :  (k_neurons,) an estimate of the r2 between optimal linear
                combination drawn from regressor matrix and the regressand.

    """
    X_int = add_intercept(X)
    n_observations, n_features = X_int.shape

    if n_observations < n_features:
      est_r2 = 1. # does not work for high d case; return R2=1 instead
    else:
      if device == torch.device("cpu"):
        beta_hat, _, _, _ = np.linalg.lstsq(X_int, Y, rcond = None)
        hat_y = X_int @ beta_hat
        rss = np.sum((hat_y - Y) ** 2.,0)
        tv = np.var(Y, ddof=1, axis = 0).squeeze()

        if use_rss:
          est_r2 = 1-(rss/(n_observations - n_features))/tv
        else:
          uev = rss/(n_observations - n_features)
          ev = np.sum((hat_y)**2.,0)
          hat_ev = (ev - n_features*uev)/n_observations
          est_r2 = (hat_ev/tv).squeeze()
      else:
        beta_hat, _, _, _ = cp.linalg.lstsq(X_int, Y, rcond = None)
        hat_y = X_int @ beta_hat
        rss = cp.sum((hat_y - Y) ** 2.,0)
        tv = cp.var(Y, ddof=1, axis = 0).squeeze()

        if use_rss:
          est_r2 = 1-(rss/(n_observations - n_features))/tv
        else:
          uev = rss/(n_observations - n_features)
          ev = cp.sum((hat_y)**2.,0)
          hat_ev = (ev - n_features*uev)/n_observations
          est_r2 = (hat_ev/tv).squeeze()


      try:
        if len(est_r2.shape) == 0:
          est_r2 = float(est_r2)
      except: pass
    return est_r2

# estimator: small data no repeats Kong and Valiant 2018; Dicker. Biometrika, 2014
def hid_ev_est(X, Y):
    """estimator of explained variance for linear regression with optimal beta for the
    high-dimensional case, i.e. works when the number of observations is lower
    than the number of features (d>m) in the regressor matrix (from
    Dicker. Biometrika, 2014).
    Parameters
    ----------
    X : numpy.ndarray
        n_observations by n_features regressor matrix
    Y : numpy.ndarray
        (m observations * 1) regressand matrix
    Returns
    -------
    ev_est :  (k_neurons,) an unbiased estimate of the explained variance
    --------
    """

    n_observations, n_features = X.shape

    if device == torch.device("cpu"):
      if len(Y.shape)<2:
          Y = Y[..., np.newaxis]
      _, k = Y.shape

      XXT = X @ X.T
      G = np.triu(XXT, k=1)
      averaging_constant = binom(n_observations, 2) # calculate the binomial coefficient
      ev_est = np.einsum('iq,ij,jq->q', Y, G, Y)/averaging_constant # need to check if the dim match
    else:
      if len(Y.shape)<2:
          Y = Y[..., cp.newaxis]
      _, k = Y.shape

      XXT = X @ X.T
      G = cp.triu(XXT, k=1)
      averaging_constant = binom(n_observations, 2) # calculate the binomial coefficient
      ev_est = cp.einsum('iq,ij,jq->q', Y, G, Y)/averaging_constant # need to check if the dim match

    if len(ev_est)==1:
        ev_est = float(ev_est)
    return ev_est




def r2_smalldata_norepeats(X, Y, return_num_denom=False):
    """estimator of r2 for linear regression with optimal beta [Dicker. Biometrika, 2014]
    No trial to trial noise
    YES high-dim case (d>m)

    Parameters
    ----------
    X : numpy.ndarray
        n_observations by n_features regressor matrix
    Y : numpy.ndarray
        (m observations * 1) regressand matrix

    Returns
    -------
    est_r2 :  (k_neurons,) an estimate of the r2 between (over)fit linear
                combination from regressor matrix and the regressand.
    """

    # X_int = add_intercept(X)
    if device == torch.device("cpu"):
      X_int = np.copy(X)
      ev_est = hid_ev_est(X_int, Y)
      denom = np.var(Y, ddof=1, axis=0)
      est_r2 = (ev_est/denom).squeeze()

    else:
      X_int = cp.copy(X)
      ev_est = hid_ev_est(X_int, Y)
      denom = cp.var(Y, ddof=1, axis=0)
      est_r2 = (ev_est/denom).squeeze()
    if len(est_r2.shape)==0:
      return float(est_r2)
    else: return est_r2


