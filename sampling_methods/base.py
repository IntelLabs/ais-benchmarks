import numpy as np
from numpy import array as t_tensor
from abc import ABCMeta, abstractmethod
from utils.plot_utils import plot_pdf
from utils.plot_utils import plot_pdf2d
import matplotlib.cm as cm
from distributions.CMultivariateNormal import CMultivariateNormal
from distributions.CMixtureModel import CMixtureModel


class CSamplingMethod(metaclass=ABCMeta):
    def __init__(self, params):
        self.space_min = params["space_min"]
        self.space_max = params["space_max"]
        self.ndims = params["dims"]

        assert self.space_max.shape == self.space_min.shape
        assert self.ndims == len(self.space_max)

        self._num_pi_evals = 0
        self._num_q_evals = 0
        self._num_q_samples = 0
        self.samples = t_tensor([])
        self.weights = t_tensor([])
        self.variogram = t_tensor([])

    def reset(self):
        self._num_pi_evals = 0
        self._num_q_evals = 0
        self._num_q_samples = 0
        self.samples = t_tensor([])
        self.weights = t_tensor([])
        self.variogram = t_tensor([])

    def get_acceptance_rate(self):
        assert self.samples.size, "Invalid number of samples to compute acceptance rate"
        return len(self.samples) / self._num_q_samples

    def get_NESS(self):
        # https://jwalton.info/Efficient-effective-sample-size-python/
        x = self.samples
        n = len(x)

        variogram = lambda t: t_tensor([((x[t:] - x[:(n - t)]) ** 2).sum() / (n - t)])

        mean = x.mean()
        dist = x - mean

        W = (dist ** 2).sum() / (n - 1)

        # vt = len(self.variogram)
        # while len(self.variogram) < len(self.samples):
        #     self.variogram = np.concatenate((self.variogram, variogram(vt))) if self.variogram.size else variogram(vt)

        t = 1
        rho = np.ones(n)
        negative_autocorr = False
        while not negative_autocorr and (t < n):
            rho[t] = 1. - variogram(t) / (2. * W)

            if not t % 2:
                negative_autocorr = sum(rho[t - 1:t + 1]) < 0

            t += 1

        ess = n / (1 + 2 * rho[1:t].sum())

        return ess / len(self.samples)

    def get_approx_NESS(self):
        normweights = self.weights / np.sum(self.weights)
        ESS = 1 / np.sum(normweights*normweights)
        return ESS / len(self.samples)


    @property
    def num_proposal_samples(self):
        return self._num_q_samples

    @property
    def num_target_evals(self):
        return self._num_pi_evals

    @property
    def num_proposal_evals(self):
        return self._num_q_evals

    @abstractmethod
    def sample(self, n_samples):
        raise NotImplementedError

    @abstractmethod
    def importance_sample(self, target_d, n_samples, timeout):
        raise NotImplementedError

    @abstractmethod
    def prob(self, samples):
        raise NotImplementedError

    @abstractmethod
    def logprob(self, samples):
        raise NotImplementedError

    @abstractmethod
    def draw(self, ax):
        pass


class CMixtureSamplingMethod(CSamplingMethod):
    def __init__(self, params):
        super(CMixtureSamplingMethod, self).__init__(params)
        self.model = None

    def reset(self):
        super(CMixtureSamplingMethod, self).reset()
        self.model = None

    def sample(self, n_samples):
        if self.model is None:
            self._update_model()
        self._num_q_samples += n_samples
        return self.model.sample(n_samples)

    def prob(self, s):
        if self.model is None:
            self._update_model()
        return self.model.prob(s)

    def logprob(self, s):
        return np.log(self.prob(s))

    def _update_model(self):
        models = []
        for x in self.samples:
            cov = np.ones(len(self.space_max)) * self.bw
            if x.shape:
                models.append(CMultivariateNormal(x, np.diag(cov)))
            else:
                models.append(CMultivariateNormal(np.array([x]), np.diag(cov)))
        self.model = CMixtureModel(models, np.exp(self.weights))

    def draw(self, ax):
        if len(self.space_max) == 1:
            return self.draw1d(ax)
        elif len(self.space_max) == 2:
            return self.draw2d(ax)
        return []

    def draw1d(self, ax):
        res = []
        for q, w in zip(self.model.models, self.model.weights):
            res.extend(ax.plot(q.mean.flatten(), 0, "gx", markersize=20))
            res.extend(plot_pdf(ax, q, self.space_min, self.space_max, alpha=1.0,
                                options="r--", resolution=0.01, scale=w))

        # res.extend(ax.plot(q.mean.flatten(), 0, "gx", markersize=20, label="$\mu_n$"))
        res.extend(plot_pdf(ax, q, self.space_min, self.space_max, label="$q_n(x)$",
                            alpha=1.0, options="r--", resolution=0.01, scale=w))

        for s, w in zip(self.samples, self.weights):
            res.append(ax.vlines(s, 0, w, "g", alpha=0.1))

        res.append(ax.vlines(s, 0, w, "g", alpha=0.1, label="$w_k = \pi(x_k) / \\frac{1}{N}\sum_{n=0}^N q_n(x_k)$"))

        res.extend(plot_pdf(ax, self, self.space_min, self.space_max, alpha=1.0, options="r-", resolution=0.01, label="$q(x)$"))

        return res

    def draw2d(self, ax):
        res = []
        for q in self.proposals:
            res.append(ax.scatter(q.mean[0], q.mean[1], c="g", marker="o"))
            res.extend(plot_pdf2d(ax, q, self.space_min, self.space_max, alpha=0.3, resolution=0.02, colormap=cm.viridis, linestyles='dashed', scale=1/len(self.proposals)))
        res.append(ax.scatter(q.mean[0], q.mean[1], c="g", marker="o", label="$q_n(x)$"))

        # for s, w in zip(self.samples, self.weights):
        #     res.append(ax.scatter(s[0], s[1], w, c="g", marker="o", alpha=0.2))
        # res.extend(ax.plot(q.mean.flatten(), 0, "gx", markersize=20, label="$\mu_n$"))
        # res.append(plot_pdf2d(ax, q, self.space_min, self.space_max, alpha=0.5, resolution=0.02, colormap=cm.viridis, label="$q_n(x)$", scale=1/len(self.proposals)))


        res.extend(plot_pdf2d(ax, self, self.space_min, self.space_max, alpha=0.8, resolution=0.02, colormap=cm.viridis, label="$q(x)$"))

        return res


class CMixtureISSamplingMethod(CMixtureSamplingMethod):
    def __init__(self, params):
        super(CMixtureISSamplingMethod, self).__init__(params)

    def get_NESS(self):
        return self.get_approx_NESS()


def make_grid(space_min, space_max, resolution):
    dim_range = space_max - space_min
    num_samples = (dim_range / resolution).tolist()
    for i in range(len(num_samples)):
        num_samples[i] = int(num_samples[i])
        if num_samples[i] < 1:
            num_samples[i] = 1

    dimensions = []
    shape = t_tensor([0] * len(num_samples))
    for i in range(len(num_samples)):
        dimensions.append(np.linspace(space_min[i], space_max[i], num_samples[i]))
        shape[i] = len(dimensions[i])

    samples = np.array(np.meshgrid(*dimensions)).T.reshape(-1, len(space_min))
    return samples, dimensions, shape


def grid_sample_distribution(dist, space_min, space_max, resolution):
    grid, dims, shape= make_grid(space_min, space_max, resolution)
    log_prob = dist.logprob(t_tensor(grid))
    return grid, log_prob, dims, shape


def uniform_sample_distribution(dist, space_min, space_max, nsamples):
    samples = np.random.uniform(space_min, space_max, size=(nsamples,len(space_max)))
    log_prob = dist.logprob(samples.reshape(nsamples,len(space_max)))
    return samples.reshape(nsamples, len(space_max)), log_prob

