import time
import numpy as np

from sampling_methods.base import t_tensor
from distributions.CMultivariateNormal import CMultivariateNormal
from distributions.CGaussianMixtureModel import generateRandomGMM
from sampling_methods.metropolis_hastings import CMetropolisHastings
from sampling_methods.evaluation import evaluate_method

if __name__ == "__main__":
    ndims_list = [1, 2, 3, 4, 5] # Number of dimensions of the space to test
    space_size = 1          # Size of the domain for each dimension [0, n)
    num_gaussians_gmm = 5   # Number of mixture components in the GMM model
    gmm_sigma_min = 0.05    # Miminum sigma value for the Normal family models
    gmm_sigma_max = 0.01    # Maximum sigma value for the Normal family models
    max_samples_list = [10, 100, 1000, 10000]       # Number of desired samples to obtain from the algorithm
    sampling_eval_samples = 1000  # Number fo samples from the true distribution used for comparison

    print("dims | #samples |  kl_kde  |  bhat_kde |  kl_nn  | bhat_nn  |  time  | method ")
    for ndims in ndims_list:
        space_min = t_tensor([-space_size] * ndims)
        space_max = t_tensor([space_size] * ndims)
        origin = (space_min + space_max) / 2.0

        # Target distribution. A.k.a ground truth
        target_dist = generateRandomGMM(space_min, space_max, num_gaussians_gmm, sigma_min=[gmm_sigma_min] * ndims, sigma_max=[gmm_sigma_max] * ndims)

        # Configure sampling method
        MCMC_proposal_dist = CMultivariateNormal(origin, np.diag(np.ones_like(space_max)) * 0.1)
        sampling_method = CMetropolisHastings(space_min, space_max, MCMC_proposal_dist)
        sampling_method.name = "MCMC-MH"

        for max_samples in max_samples_list:
            t_ini = time.time()
            kl_div_kde, kl_div_nn, bhattacharyya_dist_kde, bhattacharyya_dist_nn, ev_mse = evaluate_method(ndims, space_size, target_dist, sampling_method, max_samples, sampling_eval_samples)
            t_elapsed = time.time() - t_ini
            print("  %02d | %08d |  %7.4f |  %.5f | %.5f | %.5f | %6.3f | %s" % (ndims, max_samples, kl_div_kde, bhattacharyya_dist_kde, kl_div_nn, bhattacharyya_dist_nn, t_elapsed, sampling_method.name))
