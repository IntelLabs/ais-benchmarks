import numpy as np
import matplotlib.pyplot as plt
import time

from ais_benchmarks.distributions.derived.CABCDistribution import ABCDistribution
from ais_benchmarks.distributions.parametric.CMultivariateUniform import CMultivariateUniform
from ais_benchmarks.distributions.parametric.CMultivariateNormal import CMultivariateNormal
from ais_benchmarks.sampling_methods.metropolis_hastings import CMetropolisHastings
from ais_benchmarks.sampling_methods.tree_pyramid import CTreePyramidSampling
from ais_benchmarks.distributions.derived.CGenericNoisyFunction import GenericNoisyFunction


def log_likelihood_f(o, o_hat, slack=0.1):
    dims = len(o)

    if len(o_hat.shape) == 1:
        x = o_hat.reshape(1, dims, 1)
    elif len(o_hat.shape) == 2:
        x = o_hat.reshape(len(o_hat), dims, 1)
    else:
        raise ValueError("Shape of samples does not match self.dims")

    mu = o.reshape(1, dims, 1)
    cov = np.diag(np.ones(dims)) * slack
    inv_cov = np.linalg.inv(cov)
    log_det = np.log(np.linalg.det(cov))

    term1 = - 0.5 * dims * np.log(np.pi * 2)
    term2 = - 0.5 * log_det

    diff = x - mu
    term3 = - 0.5 * ((np.transpose(diff, axes=(0, 2, 1))) @ inv_cov @ diff)
    res = (term1 + term2 + term3).reshape(len(o_hat))
    return res


def generative_f(x, theta):
    return (theta[0] * np.sin(x * 2 * np.pi * theta[1])) * .5


def sensor_f(x, theta):
    return x


if __name__ == '__main__':
    space_min = -np.pi
    space_max = np.pi

    param_space_min = np.array([0, 0])
    param_space_max = np.array([100, 100])

    n_inference_samples = 100

    # Define true generative function parameters
    theta_gt = np.array([2.873, 1.342])

    # Prepare plot and draw true generating function
    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.show(block=False)
    x = np.linspace(-1, 1, 100)
    plt.plot(x, generative_f(x, theta_gt))

    # Draw true generating function parameters
    plt.subplot(2, 1, 2)
    plt.scatter(theta_gt[0], theta_gt[1], marker="*", s=150, label="Ground truth params")

    # Prepare parameters for inference algorithm
    inference_params = dict()
    inference_params["space_min"] = param_space_min
    inference_params["space_max"] = param_space_max
    inference_params["dims"] = 2
    inference_params["n_steps"] = 1
    inference_params["n_burnin"] = 10
    inference_params["proposal_sigma"] = 10
    inference_params["proposal_d"] = None
    inference_params["kde_bw"] = 0.01
    inference_params["n_samples_kde"] = 20
    algo_mcmc = CMetropolisHastings(inference_params)
    inference_params["method"] = "simple"
    inference_params["resampling"] = "leaf"
    inference_params["kernel"] = "haar"
    inference_params["ess_target"] = 0.95
    inference_params["n_min"] = 5
    inference_params["parallel_samples"] = 1
    algo_tpais = CTreePyramidSampling(inference_params)

    # Configure the generative model
    gen_params = dict()
    gen_params["noise_model"] = CMultivariateNormal({"mean": np.array([0]), "sigma": np.diag([0.001])})
    gen_params["function"] = generative_f
    gen_params["support"] = np.array([space_min, space_max])
    gen_params["dims"] = 1
    gen_model = GenericNoisyFunction(gen_params)

    # Configure sensor model
    sensor_params = dict()
    sensor_params["noise_model"] = CMultivariateNormal({"mean": np.array([0]), "sigma": np.diag([0.001])})
    sensor_params["function"] = sensor_f
    sensor_params["support"] = np.array([space_min, space_max])
    sensor_params["dims"] = 1
    sensor = GenericNoisyFunction(sensor_params)

    # Configure prior on the latent parameters (center and radius)
    prior_center = (param_space_min + param_space_max) / 2.0
    prior_radius = (param_space_max - param_space_min) / 2.0
    prior_d = CMultivariateUniform({"center": prior_center, "radius": prior_radius})

    # Configure the ABC distribution with the prior, the sensor model, the generative process (in this case is equal
    # to the sensor model) and the likelihood function to be used.
    params = dict()
    params["prior_d"] = prior_d
    params["sensor_d"] = sensor
    params["gen_d"] = gen_model
    params["likelihood_f"] = None
    params["loglikelihood_f"] = log_likelihood_f
    params["slack"] = 0.1
    params["dims"] = 1
    params["support"] = np.array([space_min, space_max])
    target_d = ABCDistribution(params)

    # Draw the ground truth generative process
    plt.subplot(2, 1, 1)
    plt.plot(x, generative_f(x, theta_gt), label="ground truth process", c="m")

    # Generate the sensor observation of the true state and draw it
    gen_model.set_params(theta_gt)
    gen_model.condition(x)
    fx = gen_model.sample()
    sensor.condition(fx)
    obs = sensor.sample()
    plt.scatter(x, obs, marker=".", c='g', label="observation")
    plt.pause(0.001)

    # Condition the target distribution with the measurement
    target_d.condition(obs)

    # Do posterior inference with ABC and the MCMC method
    tic = time.time()
    posterior_samples, weights = algo_mcmc.importance_sample(target_d, n_inference_samples)
    print("MCMC inference. %d samples. %5.3f  seconds." % (len(posterior_samples), time.time() - tic))

    # Plot the MCMC results
    plt.subplot(2, 1, 2)
    algo_mcmc.draw(plt.gca())
    plt.pause(0.001)

    # Do posterior inference with TP-AIS method
    tic = time.time()
    posterior_samples, weights = algo_tpais.importance_sample(target_d, n_inference_samples)
    print("TP-AIS inference. %d samples. %5.3f seconds." % (len(posterior_samples), time.time() - tic))

    # Plot the TPAIS results
    plt.subplot(2, 1, 2)
    algo_tpais.draw(plt.gca())
    # Draw samples from the TPAIS proposal after posterior inference.
    posterior_samples = algo_tpais.sample(len(posterior_samples))
    plt.pause(0.001)
    plt.gca().scatter(list(posterior_samples), [-0.4] * len(posterior_samples), marker="x", c='g', label="TP-AIS: Samples")

    plt.subplot(2, 1, 2)
    plt.gca().set_xlim(space_min, space_max)
    plt.gca().set_ylim(-theta_gt[0] * 1.2, theta_gt[0] * 1.2)
    plt.legend(scatterpoints=1)
    plt.pause(0.001)
    plt.show(block=True)
