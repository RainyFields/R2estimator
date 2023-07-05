
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as stats
from scipy.special import binom, lambertw


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if device == torch.device("cuda"):
  import cupy as cp
  cp.random.seed()
else:
  np.random.seed()

def add_intercept(X):
    """
    Add 1 additional feature dimension to regressor matrix
    Input:
      X: n_observations * n_features
    Output:
      X_prime: n_observations * (n_features + 1)
    """
    if device == torch.device("cuda"):
      X_prime = cp.concatenate([X, cp.ones((X.shape[0], 1))], 1) # ?xlei: why intercept column constant 1?
    else:
      X_prime = np.concatenate([X, np.ones((X.shape[0], 1))], 1) # ?xlei: why intercept column constant 1?
    return X_prime

def beta_generator(n_features, n_sims, is_fixed_beta):
    if device == torch.device("cuda"):
      if is_fixed_beta:
          beta = cp.random.normal(size=(n_features, 1))
          betas = cp.repeat(cp.expand_dims(beta, axis=0), repeats=n_sims, axis=0)
      else:
          betas = cp.random.normal(size=(n_sims, n_features, 1))
    else:
      if is_fixed_beta:
          beta = np.random.standard_normal(size=(n_features, 1))
          # beta = np.random.normal(size=(n_features, 1))
          betas = np.repeat(np.expand_dims(beta, axis=0), repeats=n_sims, axis=0)
      else:
          betas = np.random.standard_normal(size=(n_sims, n_features, 1))
          # betas = np.random.normal(size=(n_sims, n_features, 1))
    print("here?")
    return betas


def confidence_interval(data, confidence=0.95, sigma=None):
    if device == torch.device("cpu"):
      n = len(data)  # Sample size
      mean = np.mean(data)  # Sample mean
      if sigma:  # Population standard deviation is known
          z = stats.norm.ppf(1 - (1 - confidence) / 2)  # Z-score
          margin_of_error = z * (sigma / np.sqrt(n))
      else:  # Population standard deviation is unknown
          t = stats.t.ppf(1 - (1 - confidence) / 2, n - 1)  # T-score
          sample_std = np.std(data, ddof=1)  # Sample standard deviation
          margin_of_error = t * (sample_std / np.sqrt(n))
    else:
      n = len(data)  # Sample size

      mean = cp.mean(data)  # Sample mean

      if sigma:  # Population standard deviation is known
          z = stats.norm.ppf(1 - (1 - confidence) / 2)  # Z-score
          margin_of_error = z * (sigma / cp.sqrt(n))
      else:  # Population standard deviation is unknown
          t = stats.t.ppf(1 - (1 - confidence) / 2, n - 1)  # T-score
          sample_std = cp.std(data, ddof=1)  # Sample standard deviation
          margin_of_error = t * (sample_std / cp.sqrt(n))

    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    return (lower_bound, upper_bound)

# calculate confidence intervel
def batched_confidence_interval(batched_data):
  upper_bounds = []
  lower_bounds = []
  for i in range(batched_data.shape[0]):
    data = batched_data[i]
    lower_bound, upper_bound = confidence_interval(data, confidence=0.95, sigma=None)
    try:
      upper_bounds.append(upper_bound.get())
      lower_bounds.append(lower_bound.get())
    except: 
      upper_bounds.append(upper_bound)
      lower_bounds.append(lower_bound)
  return lower_bounds, upper_bounds

def gt_R2(beta, noise_var):
    return 1-(noise_var/(noise_var + beta.T@beta))


