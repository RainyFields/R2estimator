import os
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as stats
from scipy.special import binom, lambertw

curr_dir = os.getcwd()
fig_save_path = os.path.join(curr_dir, "figures")
if not os.path.isdir(fig_save_path):
	os.mkdir(fig_save_path)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if device == torch.device("cuda"):
  import cupy as cp
  cp.random.seed()
else:
  np.random.seed()
print("device:", device)

colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange']

from helper import beta_generator, confidence_interval, batched_confidence_interval, gt_R2
from simulator import subset_feature_gaussian_simulator_old
from estimators import r2_largedata_classic_fitbeta_norepeats, r2_largedata_classic_optimalbeta_norepeats, r2_smalldata_norepeats


n_sims = 1
n_bootstraps = 10

n_observations = 1500
n_subset_observations = 3000

is_random_feature = False
n_subset_features_gen = 1000
n_features = 10000
n_subset_features_list = [100,200,400,800,900,1000,2000,4000,6000,8000,10000]

r2 = 0.6
is_fixed_beta = False
noise_var = 2

# generate fixed_beta
betas = beta_generator(n_features, n_sims, is_fixed_beta = is_fixed_beta)

Xs, Ys, betas, R2s_gt = subset_feature_gaussian_simulator_old(r2, n_observations, n_features, 
                                                                  n_subset_features_gen, noise_var,
                                                                  n_sims, is_fixed_beta = is_fixed_beta, 
                                                                  betas = betas)
data_list = n_subset_features_list

if device == torch.device("cpu"):
    cf_est_r2s = np.zeros((len(data_list), n_bootstraps,))
    co_est_r2s_use_rss = np.zeros((len(data_list), n_bootstraps))
    co_est_r2s_no_rss = np.zeros((len(data_list), n_bootstraps))
    cs_est_r2s = np.zeros((len(data_list), n_bootstraps))
    R2s_gts = np.zeros((len(data_list), n_bootstraps,))
    sub_R2s_gts = np.zeros((len(data_list), n_bootstraps,))
else:
    cf_est_r2s = cp.zeros((len(data_list), n_bootstraps,))
    co_est_r2s_use_rss = cp.zeros((len(data_list), n_bootstraps))
    co_est_r2s_no_rss = cp.zeros((len(data_list), n_bootstraps))
    cs_est_r2s = cp.zeros((len(data_list), n_bootstraps))
    R2s_gts = cp.zeros((len(data_list), n_bootstraps,))
    sub_R2s_gts = cp.zeros((len(data_list), n_bootstraps,))

Xs = np.squeeze(Xs)
for i, n_subset_features in enumerate(data_list):
    print(i)
    
    R2s_gts[i,:] = R2s_gt[:]


    # bootstrap: with replacement
    # jackknife: without replacement
    for j in range(n_bootstraps):
        if is_random_feature:
            # with replacement [should not do it for features]
           # selected_indices = np.sort(np.random.randint(0,n_features , size=n_subset_features))
           # without replacement
           selected_indices = np.arange(n_features)
           np.random.shuffle(selected_indices)
           selected_indices = selected_indices[:n_subset_features]
        else:
          selected_indices = np.arange(n_subset_features)
    	

        # with replacement - bootstrap
        selected_indices_observations = np.sort(np.random.randint(0,n_observations , size=n_subset_observations))
        # print("len of selected observations:", len(selected_indices_observations))
        # print("unique:", len(set(selected_indices_observations)))
        # without replacement - jackknife
        # selected_indices_observations = np.arange(n_observations)
        # np.random.shuffle(selected_indices_observations)
        # selected_indices_observations = selected_indices_observations[:n_subset_observations]
       
        X = Xs[:,selected_indices]
        X = X[selected_indices_observations,:]

        Y = Ys[0, selected_indices_observations]
    
        if n_subset_features >= n_subset_features_gen:
            beta = betas[0]
        else:
            beta = betas[0][selected_indices]
         
        R2_gt = R2s_gt[0]
            
        sub_R2s_gts[i,j] = gt_R2(beta, noise_var)
        cf_est_r2s[i,j] = r2_largedata_classic_fitbeta_norepeats(X, Y)
        co_est_r2s_use_rss[i,j] = r2_largedata_classic_optimalbeta_norepeats(X, Y, use_rss=True)
        co_est_r2s_no_rss[i,j] = r2_largedata_classic_optimalbeta_norepeats(X, Y, use_rss=False)
        cs_est_r2s[i,j] = r2_smalldata_norepeats(X, Y, return_num_denom=False)



lower_bounds_cf_est, upper_bounds_cf_est = batched_confidence_interval(cf_est_r2s)
lower_bounds_gt, upper_bounds_gt = batched_confidence_interval(R2s_gts)
lower_bounds_sub_gt, upper_bounds_sub_gt = batched_confidence_interval(sub_R2s_gts)
lower_bounds_corss_est, upper_bounds_corss_est = batched_confidence_interval(co_est_r2s_use_rss)
lower_bounds_conorss_est, upper_bounds_conorss_est = batched_confidence_interval(co_est_r2s_no_rss)
lower_bounds_cs_est, upper_bounds_cs_est = batched_confidence_interval(cs_est_r2s)



# plot R2 versus variance
if device == torch.device("cpu"):
    plt.plot(data_list, np.mean(sub_R2s_gts, axis = 1), label = "subset_gt", color = colors[5])
    plt.plot(data_list,np.mean(cf_est_r2s, axis = 1), label = "classical_fit", color = colors[0])
    plt.plot(data_list,np.mean(R2s_gts, axis = 1), label = "ground_truth",color = colors[1])
    plt.plot(data_list,np.mean(co_est_r2s_use_rss, axis = 1), label = "classical_optimal_userss",color = colors[2])
    plt.plot(data_list,np.mean(co_est_r2s_no_rss, axis = 1), label = "classical_optimal_norss",color = colors[3])
    plt.plot(data_list,np.mean(cs_est_r2s, axis = 1), label = "Kong_estimator", color = colors[4])

    # # # Plot the confidence intervals and Add the shaded area
    plt.errorbar(data_list, np.mean(sub_R2s_gts, axis = 1), yerr=[np.mean(sub_R2s_gts, axis = 1) - lower_bounds_sub_gt, upper_bounds_sub_gt - np.mean(sub_R2s_gts, axis = 1)],
                fmt='none', ecolor=colors[5], capsize=4, )
    plt.fill_between(data_list, lower_bounds_sub_gt, upper_bounds_sub_gt, color='gray', alpha=0.3)

    
    plt.errorbar(data_list, np.mean(cf_est_r2s, axis = 1), yerr=[np.mean(cf_est_r2s, axis = 1) - lower_bounds_cf_est, upper_bounds_cf_est - np.mean(cf_est_r2s, axis = 1)],
                fmt='none', ecolor=colors[0], capsize=4, )
    plt.fill_between(data_list, lower_bounds_cf_est, upper_bounds_cf_est, color='gray', alpha=0.3)


    plt.errorbar(data_list, np.mean(R2s_gts, axis = 1), yerr=[np.mean(R2s_gts, axis = 1) - lower_bounds_gt, upper_bounds_gt - np.mean(R2s_gts, axis = 1)],
                fmt='none', ecolor=colors[1], capsize=4, )
    plt.fill_between(data_list, lower_bounds_gt, upper_bounds_gt, color='gray', alpha=0.3)

    plt.errorbar(data_list, np.mean(co_est_r2s_use_rss, axis = 1), yerr=[np.mean(co_est_r2s_use_rss, axis = 1) - lower_bounds_corss_est, upper_bounds_corss_est - np.mean(co_est_r2s_use_rss, axis = 1)],
                fmt='none', ecolor=colors[2], capsize=4, )
    plt.fill_between(data_list, lower_bounds_corss_est, upper_bounds_corss_est, color='gray', alpha=0.3)

    plt.errorbar(data_list, np.mean(co_est_r2s_no_rss, axis = 1), yerr=[np.mean(co_est_r2s_no_rss, axis = 1) - lower_bounds_conorss_est, upper_bounds_conorss_est - np.mean(co_est_r2s_no_rss, axis = 1)],
                fmt='none', ecolor=colors[2], capsize=4, )
    plt.fill_between(data_list, lower_bounds_conorss_est, upper_bounds_conorss_est, color='gray', alpha=0.3)


    plt.errorbar(data_list, np.mean(cs_est_r2s, axis = 1), yerr=[np.mean(cs_est_r2s, axis = 1) - lower_bounds_cs_est, upper_bounds_cs_est - np.mean(cs_est_r2s, axis = 1)],
                fmt='none', ecolor=colors[0], capsize=4, )
    plt.fill_between(data_list, lower_bounds_cs_est, upper_bounds_cs_est, color='gray', alpha=0.3)

else:
    plt.plot(data_list,cp.mean(cf_est_r2s, axis = 1).get(), label = "classical_fit", color = colors[0])
    plt.plot(data_list,cp.mean(R2s_gts, axis = 1).get(), label = "ground_truth",color = colors[1])
    plt.plot(data_list,cp.mean(co_est_r2s_use_rss, axis = 1).get(), label = "classical_optimal_userss",color = colors[2])
    plt.plot(data_list,cp.mean(co_est_r2s_no_rss, axis = 1).get(), label = "classical_optimal_norss",color = colors[3])
    plt.plot(data_list,cp.mean(cs_est_r2s, axis = 1).get(), label = "Kong_estimator", color = colors[4])

    # # # Plot the confidence intervals and Add the shaded area
    plt.errorbar(data_list, cp.mean(cf_est_r2s, axis = 1).get(), yerr=[cp.mean(cf_est_r2s, axis = 1).get() - lower_bounds_cf_est, upper_bounds_cf_est - cp.mean(cf_est_r2s, axis = 1).get()],
                fmt='none', ecolor=colors[0], capsize=4, )
    plt.fill_between(data_list, lower_bounds_cf_est, upper_bounds_cf_est, color='gray', alpha=0.3)


    plt.errorbar(data_list, cp.mean(R2s_gts, axis = 1).get(), yerr=[cp.mean(R2s_gts, axis = 1).get() - lower_bounds_gt, upper_bounds_gt - cp.mean(R2s_gts, axis = 1).get()],
                fmt='none', ecolor=colors[1], capsize=4, )
    plt.fill_between(data_list, lower_bounds_gt, upper_bounds_gt, color='gray', alpha=0.3)

    plt.errorbar(data_list, cp.mean(co_est_r2s_use_rss, axis = 1).get(), yerr=[cp.mean(co_est_r2s_use_rss, axis = 1).get() - lower_bounds_corss_est, upper_bounds_corss_est - cp.mean(co_est_r2s_use_rss, axis = 1).get()],
                fmt='none', ecolor=colors[2], capsize=4, )
    plt.fill_between(data_list, lower_bounds_corss_est, upper_bounds_corss_est, color='gray', alpha=0.3)

    plt.errorbar(data_list, cp.mean(co_est_r2s_no_rss, axis = 1).get(), yerr=[cp.mean(co_est_r2s_no_rss, axis = 1).get() - lower_bounds_conorss_est, upper_bounds_conorss_est - cp.mean(co_est_r2s_no_rss, axis = 1).get()],
                fmt='none', ecolor=colors[2], capsize=4, )
    plt.fill_between(data_list, lower_bounds_conorss_est, upper_bounds_conorss_est, color='gray', alpha=0.3)


    plt.errorbar(data_list, cp.mean(cs_est_r2s, axis = 1).get(), yerr=[cp.mean(cs_est_r2s, axis = 1).get() - lower_bounds_cs_est, upper_bounds_cs_est - cp.mean(cs_est_r2s, axis = 1).get()],
                fmt='none', ecolor=colors[0], capsize=4, )
    plt.fill_between(data_list, lower_bounds_cs_est, upper_bounds_cs_est, color='gray', alpha=0.3)


plt.title("total_features %d subset_features %d n_observations %d n_subset_observations %d noise_variance %.2f " % (n_features, n_subset_features_gen, n_observations, n_subset_observations,noise_var))
plt.legend()
plt.xlabel("size of subset of features")
plt.ylabel("R2")
if is_random_feature:
	plt.savefig(os.path.join(fig_save_path, "bootstrap_subset_feature_randomorder_n_obv_%d_n_subobv_%d_n_feat_%d_n_subfeat_%d_noisevar_%.2f.png" % (n_observations, n_subset_observations, n_features, n_subset_features_gen, noise_var)))	
else:
	plt.savefig(os.path.join(fig_save_path, "bootstrap_subset_feature_ordered_n_obv_%d_n_subobv_%d_n_feat_%d_n_subfeat_%d_noisevar_%.2f.png" % (n_observations, n_subset_observations, n_features, n_subset_features_gen, noise_var)))
# plt.xscale("log")
# plt.hist(est_r2s)