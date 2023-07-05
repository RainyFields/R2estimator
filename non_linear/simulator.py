import sys
import os
import numpy as np 
import matplotlib.pyplot as plt
sys.path.append('/grid/cowley/home/xilei/R2estimator/')
from estimators import r2_smalldata_norepeats

def non_linear_simulator(noise_var, n_sims, n_observations):
    """
    generate y = x**2 + noise, x sampled from std gaussian, return estimated r2
    # how to calculate ground truth R2? 

    """
    Xs = np.random.standard_normal(size=(n_sims, n_observations))
    noise = np.random.normal(scale=noise_var**0.5, size=(n_sims, n_observations,))
    Ys = np.square(Xs) + noise

    return Xs, Ys,


noise_var_list = [0.0001, 0.001, 0.1,2,4,8,16,32,64]
n_sims = 1000
n_observations = 1000
est_r2s = np.zeros((len(noise_var_list), n_sims))
for j, noise_var in enumerate(noise_var_list):    
    Xs, Ys = non_linear_simulator(noise_var, n_sims, n_observations)
    Xs = np.expand_dims(Xs, axis = -1)
    
    for i in range(n_sims):
        est_r2 = r2_smalldata_norepeats(Xs[i,:, :], Ys[i,:])
        est_r2s[j, i] = est_r2

    

plt.figure()
plt.boxplot(est_r2s.T)
plt.xticks(np.arange(1,len(noise_var_list)+1), noise_var_list)
plt.savefig(os.getcwd() + "/figures/est_r2s.jpeg")
