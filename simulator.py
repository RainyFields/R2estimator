import os
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

def gaussian_simulator_old(r2, n_observations, n_features, noise_var,n_sims, is_fixed_beta = True, betas = None):
    """
    Generate multidimensional gaussian distributed regressor X, linear regression fit beta and corresponding regressand Y with additional white noise
    input:
        n_observations
        n_features
        noise_var
        n_sims
        is_fixed_beta: True by default
    Returns:
        Xs: n_sims * n_observations * n_features
        Ys: n_sims * n_obervations * 1
        betas: n_sims * n_features * 1
        R2s: n_sims * 1
    """
    if device == torch.device("cpu"):
      X_cov = np.eye(n_features)
      X_mean = np.zeros(n_features)
    else:
      X_cov = cp.eye(n_features)
      X_mean = cp.zeros(n_features)
    desired_sig_var = r2 * noise_var / (1 - r2)
    Y_mean = 0

    Xs = []
    Ys = []
    R2s = []
    if not is_fixed_beta:
      if device == torch.device("cpu"):
        betas = np.zeros((n_sims, n_features))
        for i in range(n_sims):
            beta = np.random.normal(size=(n_features, 1))  # beta
            sig_var = beta.T @ X_cov @ beta
            beta = ((desired_sig_var / sig_var) ** 0.5) * beta
            cur_Y_mean = X_mean @ beta
            Y_mean = Y_mean - cur_Y_mean
            X = np.random.multivariate_normal(mean=X_mean, cov=X_cov,
                                          size=(n_observations,))#features
            eps = np.random.normal(scale=noise_var**0.5, size=(n_observations,))# generate noise
            sig = (X @ beta).squeeze()
            Y = Y_mean + sig + eps# generate observations with noise added
            Xs.append(X)
            Ys.append(Y)
            betas[i,:] = beta.T
            R2s.append(r2)
        return np.asarray(Xs), np.asarray(Ys), betas, np.asarray(R2s)
      else:
        betas = cp.zeros((n_sims, n_features))
        for i in range(n_sims):
            beta = cp.random.normal(size=(n_features, 1))  # beta
            sig_var = beta.T @ X_cov @ beta
            beta = ((desired_sig_var / sig_var) ** 0.5) * beta
            cur_Y_mean = X_mean @ beta
            Y_mean = Y_mean - cur_Y_mean
            X = cp.random.multivariate_normal(mean=X_mean, cov=X_cov,
                                          size=(n_observations,))#features
            eps = cp.random.normal(scale=noise_var**0.5, size=(n_observations,))# generate noise
            sig = (X @ beta).squeeze()

            Y = Y_mean + sig + eps# generate observations with noise added
            Xs.append(X)

            Ys.append(Y)
            betas[i,:] = beta.T
            R2s.append(r2)
        return cp.asarray(Xs), cp.asarray(Ys), betas, cp.asarray(R2s)
    else:
        if (betas == None).any():
            if device == torch.device("cpu"):
                betas = np.zeros((n_sims, n_features))
                beta = np.random.normal(size=(n_features, 1))
                for i in range(n_sims):
                    betas[i,:] = beta.T
                    sig_var = beta.T @ X_cov @ beta
                    beta = ((desired_sig_var / sig_var) ** 0.5) * beta
                    cur_Y_mean = X_mean @ beta
                    Y_mean = Y_mean - cur_Y_mean
                    X = np.random.multivariate_normal(mean=X_mean, cov=X_cov,
                                                  size=(n_observations,))#features
                    eps = np.random.normal(scale=noise_var**0.5, size=(n_observations,))# generate noise
                    sig = (X @ beta).squeeze()
                    Y = Y_mean + sig + eps# generate observations with noise added
                    Xs.append(X)
                    Ys.append(Y)
                    betas[i,:] = beta.T
                    R2s.append(r2)
                return np.asarray(Xs), np.asarray(Ys), betas, np.asarray(R2s)
            else:
                betas = cp.zeros((n_sims, n_features))
                beta = cp.random.normal(size=(n_features, 1))
                for i in range(n_sims):
                    betas[i,:] = beta.T
                    sig_var = beta.T @ X_cov @ beta
                    beta = ((desired_sig_var / sig_var) ** 0.5) * beta
                    cur_Y_mean = X_mean @ beta
                    Y_mean = Y_mean - cur_Y_mean
                    X = cp.random.multivariate_normal(mean=X_mean, cov=X_cov,
                                                  size=(n_observations,))#features
                    eps = cp.random.normal(scale=noise_var**0.5, size=(n_observations,))# generate noise
                    sig = (X @ beta).squeeze()

                    Y = Y_mean + sig + eps# generate observations with noise added
                    Xs.append(X)

                    Ys.append(Y)
                    betas[i,:] = beta.T
                    R2s.append(r2)
        else: 
            if device == torch.device("cpu"):
            
            
                for i in range(n_sims):
                    beta = betas[i,:]
                    
                    sig_var = beta.T @ X_cov @ beta
                    beta = ((desired_sig_var / sig_var) ** 0.5) * beta
                    cur_Y_mean = X_mean @ beta
                    Y_mean = Y_mean - cur_Y_mean
                    X = np.random.multivariate_normal(mean=X_mean, cov=X_cov,
                                                  size=(n_observations,))#features
                    eps = np.random.normal(scale=noise_var**0.5, size=(n_observations,))# generate noise
                    sig = (X @ beta).squeeze()
                    Y = Y_mean + sig + eps# generate observations with noise added
                    Xs.append(X)
                    Ys.append(Y)
                    betas[i,:] = beta
                    R2s.append(r2)
                return np.asarray(Xs), np.asarray(Ys), betas, np.asarray(R2s)
            else:
            
                for i in range(n_sims):
                    beta = betas[i,:].T
                    sig_var = beta.T @ X_cov @ beta
                    beta = ((desired_sig_var / sig_var) ** 0.5) * beta
                    cur_Y_mean = X_mean @ beta
                    Y_mean = Y_mean - cur_Y_mean
                    X = cp.random.multivariate_normal(mean=X_mean, cov=X_cov,
                                                  size=(n_observations,))#features
                    eps = cp.random.normal(scale=noise_var**0.5, size=(n_observations,))# generate noise
                    sig = (X @ beta).squeeze()

                    Y = Y_mean + sig + eps# generate observations with noise added
                    Xs.append(X)

                    Ys.append(Y)
                    betas[i,:] = beta.T
                    R2s.append(r2)
            
        return cp.asarray(Xs), cp.asarray(Ys), betas, cp.asarray(R2s)



def subset_feature_gaussian_simulator_old(r2, n_observations, n_features, n_subset_features, noise_var,n_sims, is_fixed_beta = True, betas = None):
    """
    Generate multidimensional gaussian distributed regressor X, linear regression fit beta and corresponding regressand Y with additional white noise
    input:
        n_observations
        n_features
        noise_var
        n_sims
        is_fixed_beta: True by default
    Returns:
        Xs: n_sims * n_observations * n_features
        Ys: n_sims * n_obervations * 1
        betas: n_sims * n_features * 1
        R2s: n_sims * 1
    """
    if device == torch.device("cpu"):
      X_cov = np.eye(n_features)
      X_mean = np.zeros(n_features)
    else:
      X_cov = cp.eye(n_features)
      X_mean = cp.zeros(n_features)
    desired_sig_var = r2 * noise_var / (1 - r2)
    Y_mean = 0

    Xs = []
    Ys = []
    R2s = []
    if not is_fixed_beta:
      if device == torch.device("cpu"):
        betas = np.zeros((n_sims, n_subset_features))
        mg_cord_x, mg_cord_y = np.meshgrid(np.arange(n_subset_features), np.arange(n_subset_features))
        for i in range(n_sims):
            beta = np.random.normal(size=(n_subset_features, 1))  # beta
            sig_var = beta.T @ X_cov[mg_cord_x,mg_cord_y ] @ beta
            beta = ((desired_sig_var / sig_var) ** 0.5) * beta
            cur_Y_mean = X_mean[:n_subset_features] @ beta
            Y_mean = Y_mean - cur_Y_mean
            
            X = np.random.standard_normal(size=(n_observations, n_features))
#             X = np.random.multivariate_normal(mean=X_mean, cov=X_cov,
#                                           size=(n_observations,))#features
            eps = np.random.normal(scale=noise_var**0.5, size=(n_observations,))# generate noise
            sig = (X[:,:n_subset_features] @ beta).squeeze()
            Y = Y_mean + sig + eps# generate observations with noise added
            Xs.append(X)
            Ys.append(Y)
            betas[i,:] = beta.T
            R2s.append(r2)
        return np.asarray(Xs), np.asarray(Ys), betas, np.asarray(R2s)
      else: ### todo: to be updated
        betas = cp.zeros((n_sims, n_features))
        for i in range(n_sims):
            beta = cp.random.normal(size=(n_features, 1))  # beta
            sig_var = beta.T @ X_cov @ beta
            beta = ((desired_sig_var / sig_var) ** 0.5) * beta
            cur_Y_mean = X_mean @ beta
            Y_mean = Y_mean - cur_Y_mean
            X = cp.random.multivariate_normal(mean=X_mean, cov=X_cov,
                                          size=(n_observations,))#features
            eps = cp.random.normal(scale=noise_var**0.5, size=(n_observations,))# generate noise
            sig = (X @ beta).squeeze()

            Y = Y_mean + sig + eps# generate observations with noise added
            Xs.append(X)

            Ys.append(Y)
            betas[i,:] = beta.T
            R2s.append(r2)
        return cp.asarray(Xs), cp.asarray(Ys), betas, cp.asarray(R2s)
    else:
        if (betas == None).all():
            if device == torch.device("cpu"):
                print("this branch")
                betas = np.zeros((n_sims, n_subset_features))
                beta = np.random.normal(size=(n_subset_features, 1))
                mg_cord_x, mg_cord_y = np.meshgrid(np.arange(n_subset_features), np.arange(n_subset_features))
                for i in range(n_sims):
                    betas[i,:] = beta.T
                    sig_var = beta.T @ X_cov[mg_cord_x, mg_cord_y] @ beta
#                     print("c1")
                    beta = ((desired_sig_var / sig_var) ** 0.5) * beta
                    cur_Y_mean = X_mean[:n_subset_features] @ beta
                    
                    Y_mean = Y_mean - cur_Y_mean
#                     print("c2")
#                     start_time = time.time()
                    X = np.random.standard_normal(size=(n_observations, n_features))
#                     X = np.random.multivariate_normal(mean=X_mean, cov=X_cov,
#                                                   size=(n_observations,))#features
#                     end_time = time.time()
#                     elapsed_time = end_time - start_time

                    # Print the elapsed time
#                     print("Elapsed time: {:.4f} seconds".format(elapsed_time))
#                     print("c3")
                    eps = np.random.normal(scale=noise_var**0.5, size=(n_observations,))# generate noise
                    sig = (X[:,:n_subset_features] @ beta).squeeze()
#                     print("c4")
                    Y = Y_mean + sig + eps# generate observations with noise added
                    Xs.append(X)
                    Ys.append(Y)
                    betas[i,:] = beta.T
                    R2s.append(r2)
                return np.asarray(Xs), np.asarray(Ys), betas, np.asarray(R2s)
            else:
                betas = cp.zeros((n_sims, n_features))
                beta = cp.random.normal(size=(n_features, 1))
                for i in range(n_sims):
                    betas[i,:] = beta.T
                    sig_var = beta.T @ X_cov @ beta
                    beta = ((desired_sig_var / sig_var) ** 0.5) * beta
                    cur_Y_mean = X_mean @ beta
                    Y_mean = Y_mean - cur_Y_mean
                    X = cp.random.multivariate_normal(mean=X_mean, cov=X_cov,
                                                  size=(n_observations,))#features
                    eps = cp.random.normal(scale=noise_var**0.5, size=(n_observations,))# generate noise
                    sig = (X @ beta).squeeze()

                    Y = Y_mean + sig + eps# generate observations with noise added
                    Xs.append(X)

                    Ys.append(Y)
                    betas[i,:] = beta.T
                    R2s.append(r2)
        else: 
            if device == torch.device("cpu"):
            
            
                for i in range(n_sims):
                    beta = betas[i,:]
                    
                    sig_var = beta.T @ X_cov @ beta
                    beta = ((desired_sig_var / sig_var) ** 0.5) * beta
                    cur_Y_mean = X_mean @ beta
                    Y_mean = Y_mean - cur_Y_mean
                    X = np.random.multivariate_normal(mean=X_mean, cov=X_cov,
                                                  size=(n_observations,))#features
                    eps = np.random.normal(scale=noise_var**0.5, size=(n_observations,))# generate noise
                    sig = (X @ beta).squeeze()
                    Y = Y_mean + sig + eps# generate observations with noise added
                    Xs.append(X)
                    Ys.append(Y)
                    betas[i,:] = beta
                    R2s.append(r2)
                return np.asarray(Xs), np.asarray(Ys), betas, np.asarray(R2s)
            else:
            
                for i in range(n_sims):
                    beta = betas[i,:].T
                    sig_var = beta.T @ X_cov @ beta
                    beta = ((desired_sig_var / sig_var) ** 0.5) * beta
                    cur_Y_mean = X_mean @ beta
                    Y_mean = Y_mean - cur_Y_mean
                    X = cp.random.multivariate_normal(mean=X_mean, cov=X_cov,
                                                  size=(n_observations,))#features
                    eps = cp.random.normal(scale=noise_var**0.5, size=(n_observations,))# generate noise
                    sig = (X @ beta).squeeze()

                    Y = Y_mean + sig + eps# generate observations with noise added
                    Xs.append(X)

                    Ys.append(Y)
                    betas[i,:] = beta.T
                    R2s.append(r2)
            
        return cp.asarray(Xs), cp.asarray(Ys), betas, cp.asarray(R2s)