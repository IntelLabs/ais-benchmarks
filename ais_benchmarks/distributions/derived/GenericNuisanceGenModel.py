import numpy as np
from ais_benchmarks.distributions import CDistribution


class GenericNuisanceGenModel(CDistribution):
    def __init__(self, params):
        self._check_param(params, "gen_function")
        self._check_param(params, "params_mask")
        self._check_param(params, "noise_sigma")
        self._check_param(params, "nuisance_dist", CDistribution)
        params["family"] = "generative_models"
        params["type"] = "nuisance_gen_models"
        params["likelihood_f"] = lambda x: np.ones(len(x))  # This is a deterministic model.
        params["loglikelihood_f"] = lambda x: np.zeros(len(x))  # Probabilities of z generating x are always 1

        super().__init__(params)

        self.z_mask = params["params_mask"]
        self.n_mask = np.logical_not(self.z_mask)  # Parameters that are not latent are considered nuisance
        self.z = None  # Generative model relevant parameters
        self.n = None  # Generative model nuisance parameters
        self.n_dist = params["nuisance_dist"]  # Distribution used to sample nuisance values for each generation

        assert callable(params["gen_function"]), "Generative function must be a callable object"
        self.gen_f = params["gen_function"]

    def sample(self, nsamples=1):
        if self.z is None:
            raise ValueError("GenericNuisanceGenModel cannot generate data w/o conditioning it first with. Make sure "
                             "to call self.condition(z) first.")
        z = self._check_shape(self.z)
        n = self.n_dist.sample(len(z) * nsamples)
        x_hat = self.gen_f(z, n)
        # TODO: add noise to x_hat
        return x_hat.detach()

    def condition(self, z):
        z = self._check_shape(z)
        self.z = z

    def marginal(self, dim):
        raise NotImplementedError
