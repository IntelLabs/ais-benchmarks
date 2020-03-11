import numpy as np
import matplotlib.pyplot as plt

from distributions.CABCDistribution import ABCDistribution
from distributions.CMultivariateUniform import CMultivariateUniform
from distributions.CMultivariateNormal import CMultivariateNormal
from sampling_methods.metropolis_hastings import CMetropolisHastings
from sampling_methods.tree_pyramid import CTreePyramidSampling
from distributions.CGenericNoisyFunction import GenericNoisyFunction


def likelihood_f(o, o_hat, slack=0.1):
    return np.exp(log_likelihood_f(o, o_hat, slack))


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
    return (term1 + term2 + term3).reshape(len(o_hat))


def gt_gen_f(z):
    """
    Ground truth generative function. This is the unknown underlying process that generates data. Given a hidden state
    value z it returns the output of the generative process. Which should be in the observable space x.
    :param z:
    :return:
    """
    A = 3.0
    f = 0.5
    phi = 0
    return (A*np.sin(z*2*np.pi*f + phi) + A) / 2


def approx_gen_f(z):
    """
    Approximated generative function. This is our model of the underlying generative function. Given a hidden state
    value z it returns the output of the generative process. Which should be in the observable space x.
    :param z:
    :return:
    """
    A = 3.0
    f = 0.55
    phi = 0.2
    return (A*np.sin(z*2*np.pi*f + phi) + A) / 2


def sensor_f(x):
    """
    Sensor transfer function. This is our model of the sensor transfer function that converts an observable state x
    into a noiseless observation value.
    :param x:
    :return:
    """
    return x


if __name__ == '__main__':
    space_min = -1.0
    space_max = 1.0

    # Prepare plot and draw true generating function
    fig = plt.figure()
    plt.show(False)
    x = np.linspace(-1, 1, 100)
    plt.plot(x, gt_gen_f(x))

    # Prepare parameters for inference algorithm
    inference_params = dict()
    inference_params["space_min"] = np.array([space_min])
    inference_params["space_max"] = np.array([space_max])
    inference_params["dims"] = 1
    inference_params["n_steps"] = 1
    inference_params["n_burnin"] = 10
    inference_params["proposal_sigma"] = 10
    inference_params["proposal_d"] = CMultivariateNormal(np.zeros(inference_params["dims"]), np.diag(np.ones(inference_params["dims"]) * inference_params["proposal_sigma"]))
    inference_params["kde_bw"] = 0.01
    algo_mcmc = CMetropolisHastings(inference_params)
    inference_params["method"] = "simple"
    inference_params["resampling"] = "leaf"
    inference_params["kernel"] = "haar"
    algo_tpais = CTreePyramidSampling(inference_params)

    # Configure the generative model
    gen_params = dict()
    gen_params["noise"] = 1e-6
    gen_params["model"] = approx_gen_f
    gen_params["support"] = np.array([space_min, space_max])
    gen_model = GenericNoisyFunction(gen_params)

    # Configure sensor model
    sensor_params = dict()
    sensor_params["noise"] = 0.1
    sensor_params["model"] = sensor_f
    sensor_params["support"] = np.array([space_min, space_max])
    sensor = GenericNoisyFunction(sensor_params)

    # Configure prior on the latent parameters (center and radius)
    prior_d = CMultivariateUniform(np.array([(space_min + space_max)/2]), np.array([(space_max-space_min)/2.0]))
    prior_d.draw(plt.gca(), label="prior", color="b")

    # Configure the ABC distribution with the prior, the sensor model, the generative process (in this case is equal
    # to the sensor model) and the likelihood function to be used.
    params = dict()
    params["prior_d"] = prior_d
    params["sensor_d"] = sensor
    params["gen_d"] = gen_model
    params["likelihood_f"] = likelihood_f
    params["slack"] = 0.2
    params["support"] = np.array([space_min, space_max])
    target_d = ABCDistribution(params)

    # Randomly sample the ground truth point from the support
    gt_z = np.array([np.random.uniform(space_min, space_max)])

    # Generate and display the true state with the generative model
    gt_x = gt_gen_f(gt_z)

    # Draw the ground truth generative process
    plt.plot(x, gt_gen_f(x), label="$\pi(x)$", c="r")

    # Draw the approximated generative process
    plt.plot(x, approx_gen_f(x), label="$q(x)$", c="g")

    # Draw the ground truth point
    plt.gca().scatter(gt_z, gt_x, marker="*", s=150, c='b', label="gt")

    # Generate the sensor observation of the true state and draw it
    sensor.condition(gt_x)
    obs = sensor.sample()
    plt.gca().scatter(gt_z, obs, marker="*", s=150, c='g', label="obs")

    # Condition the target distribution with the measurement
    target_d.condition(obs)

    # Do posterior inference with ABC and the MCMC method
    posterior_samples = algo_mcmc.importance_sample(target_d, 100)

    # Plot the MCMC results
    # algo_mcmc.draw(plt.gca())

    # Do posterior inference with TP-AIS method
    posterior_samples = algo_tpais.importance_sample(target_d, 100)

    # Plot the TPAIS results
    # algo_tpais.draw(plt.gca())

    plt.gca().set_xlim(space_min, space_max)
    plt.gca().set_ylim(0, 4)
    plt.legend(scatterpoints=1)
    plt.show(True)

