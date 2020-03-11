import numpy as np
import matplotlib.pyplot as plt

from distributions.CABCDistribution import ABCDistribution
from distributions.CMultivariateUniform import CMultivariateUniform
from distributions.CMultivariateNormal import CMultivariateNormal
from sampling_methods.metropolis_hastings import CMetropolisHastings
from sampling_methods.tree_pyramid import CTreePyramidSampling
from distributions.CGenericNoisyFunction import GenericNoisyFunction

'''
Sources of uncertainty
1- There is uncertainty in the model selection: Polynomial? Harmonic? Exponential? Log?
2- There is uncertainty in the model parameters: What are the parameters given the dataset? p(w|D)
3- Uncertainty in model parameters induces uncertainty in model predictions: p(x|w)
4- There is uncertainty in the perception: p(o|x)
5- There is uncertainty in the effect of actions: p(x'|a,x)

Visualization
'''


def LikelihoodF(o, o_hat, slack=0.1):
    return np.exp(LogLikelihoodF(o, o_hat, slack))


def LogLikelihoodF(o, o_hat, slack=0.1):
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


def gt_gen_f(x):
    A = 3.0
    f = 0.5
    phi = 0
    return (A*np.sin(x*2*np.pi*f + phi) + A) / 2


def approx_gen_f(x):
    A = 3.0
    f = 0.55
    phi = 0.2
    return (A*np.sin(x*2*np.pi*f + phi) + A) / 2


def sensor_f(x):
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
    algo = CMetropolisHastings(inference_params)
    inference_params["method"] = "simple"
    inference_params["resampling"] = "leaf"
    inference_params["kernel"] = "haar"
    algo = CTreePyramidSampling(inference_params)

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
    prior_d.draw(plt.gca())

    # Configure the ABC distribution with the prior, the sensor model, the generative process (in this case is equal
    # to the sensor model) and the likelihood function to be used.
    params = dict()
    params["prior_d"] = prior_d
    params["sensor_d"] = sensor
    params["gen_d"] = gen_model
    params["likelihood_f"] = LikelihoodF
    params["slack"] = 0.2
    params["support"] = np.array([space_min, space_max])
    target_d = ABCDistribution(params)

    # Obtain the ground truth from the prior
    gt_z = np.array([space_min])

    while True:
        # Generate and display the true state with the generative model
        gt_x = gt_gen_f(gt_z)

        # Draw the ground truth generative process
        plt.plot(x, gt_gen_f(x))

        # Draw the approximated generative process
        plt.plot(x, approx_gen_f(x))

        # Draw the ground truth point
        plt.gca().scatter(gt_z, gt_x, marker="*", s=150, c='b', label="gt")

        # Generate the sensor observation of the true state and draw it
        sensor.condition(gt_x)
        obs = sensor.sample()
        plt.gca().scatter(gt_z, obs, marker="*", s=150, c='g', label="obs")

        # Compute the predictive posterior conditioned by the measurement
        # target_d.draw(plt.gca())
        target_d.condition(obs)
        posterior_samples = algo.importance_sample(target_d, 100)
        algo.draw(plt.gca())
        algo.reset()
        plt.gca().set_xlim(space_min, space_max)
        plt.gca().set_ylim(0, 4)
        plt.legend()
        plt.show(False)
        plt.pause(0.001)
        plt.clf()

        # Step the ground truth to sweep the space
        gt_z += 0.05
        if gt_z > space_max:
            gt_z = np.array([space_min])

