import numpy as np
from scipy.special import gamma
from sampling_methods.base import CMixtureSamplingMethod
from distributions.CMultivariateNormal import CMultivariateNormal
from utils.plot_utils import plot_pdf
from utils.plot_utils import plot_pdf2d
from sklearn.cluster import KMeans
import time
import matplotlib.cm as cm
import sys
import distributions


class CEllipsoid:
    def __init__(self, loc, scale, indices=[]):

        if len(loc) > 1:
            det = np.linalg.det(scale)
            if det <= 0:
                raise ValueError
        else:
            if scale <= 0:
                raise ValueError
            scale = scale.reshape(1,1)

        self.loc = loc
        self.scale = scale
        self.indices = indices

        self.sampler = CMultivariateNormal(mean=loc, cov=scale)

        d = len(loc)
        sphere_vol = 2/d * (np.pi ** (d/2) / gamma(d/2))

        self.volume = sphere_vol * np.linalg.det(scale)

    def __repr__(self):
        return "loc: " + str(self.loc) + "n_points:" + str(np.count_nonzero(self.indices)) + "vol: %f" % self.volume

    def sample(self):
        return self.sampler.sample()

    @staticmethod
    def fit(points, inflate=1.0):
        loc = np.mean(points, axis=0)

        assert len(points) >= len(loc)

        if len(points[0]) > 1:
            scale = np.cov(points, rowvar=False)
            if np.linalg.det(scale) <= 0:
                print("CEllipsoid: ERROR! Ellipsoid fit failed. Negative determinant. Identity covariance.", file=sys.stderr)
                ell = CEllipsoid(loc, np.diag(np.ones(len(loc)) * inflate))
                return ell
        else:
            scale = np.std(points)
            if scale <= 0:
                print("Ellipsoid fit failed. Negative or zero STD. Using 1.", file=sys.stderr)
                ell = CEllipsoid(loc, np.diag(np.ones(len(loc)) * inflate))
                return ell

        ell = CEllipsoid(loc, scale*inflate)
        return ell


class CMultiNestedSampling(CMixtureSamplingMethod):
    def __init__(self, params):
        """
        Implementation of multi nested sampling algorithm.
        F. Feroz, M.P. Hobson, E. Cameron, and Pettitt A.N.  Importance Nested Sampling and the MULTINEST Algorithm.
        Arxiv astro physics, 2014

        :param params:
            - space_min: Space lower boundary
            - space_max: Space upper boundary
            - dims: Number of dimensions
        """
        super(self.__class__, self).__init__(params)
        self.range = self.space_max - self.space_min
        self.proposal_dist = eval(params["proposal"])
        self.N = params["N"]
        self.bw = np.array([params["kde_bw"]])

        self.ellipsoids = []

        # Obtain initial samples from a uniform prior distribution
        self.live_points = np.random.uniform(0, 1, size=(self.N, len(self.space_max))) * self.range + self.space_min
        self.Z = None
        self.L = None

        self.reset()

    def reset(self):
        self.live_points = np.random.uniform(0, 1, size=(self.N, len(self.space_max))) * self.range + self.space_min
        self.L = np.array([])
        self.Z = 0

    def get_NESS(self):
        ESS = np.exp(-np.sum(self.weights * np.log(self.weights)))
        return ESS / len(self.samples)

    def resample(self, value, pdf, ellipsoid, timeout=60, ellipsoid_converged_radius=1e-3):
        new_sample = np.clip(ellipsoid.sample(), self.space_min, self.space_max)

        self._num_q_samples += 1

        new_value = pdf.prob(new_sample)
        self._num_pi_evals += 1
        elapsed_time = 0
        t_ini = time.time()
        # print("Ellipsoid volume: %f. Converged: " % ellipsoid.volume, not ellipsoid.volume > ellipsoid_converged_radius**self.ndims)
        while value > new_value and elapsed_time < timeout and ellipsoid.volume > ellipsoid_converged_radius**self.ndims:
            new_sample = ellipsoid.sample()
            self._num_q_samples += 1
            new_value = pdf.prob(new_sample)
            self._num_pi_evals += 1
            elapsed_time = time.time() - t_ini

        return new_sample, new_value

    # TODO: Include covariance in ellipsoid to ellipsoid distance
    @staticmethod
    def ellipsoid_distance(e1, e2):
        return np.sqrt(np.sum((e1.loc - e2.loc) * (e1.loc - e2.loc)))

    def recursive_clustering(self, points, ellipsoids=None, cluster_volume_ratio=0.8, cluster_min_distance=0.2, inflate_factor=2.0, debug=False):
        # Compute live points ellipsoid for the first iteration
        if ellipsoids is None:
            ellipsoids = [CEllipsoid.fit(points, inflate_factor)]
            ellipsoids[0].indices = np.array([True] * len(points))
            ellipsoid_volume = ellipsoids[0].volume
        else:
            ellipsoid_volume = 0
            for e in ellipsoids:
                ellipsoid_volume = ellipsoid_volume + e.volume

        # Compute cluster candidates
        estimator = KMeans(n_clusters=len(ellipsoids)+1)
        estimator.fit(points)

        # Accept cluster subdivision?
        # Condition 0: All clusters must have more than dimensions + 1 points
        for i in range(estimator.n_clusters):
            c_npoints = np.count_nonzero(estimator.labels_ == i)
            if c_npoints < len(self.space_min) + 1:
                if debug:
                    print("Rejected subdivision. Number of points: %d" % c_npoints)
                return points, ellipsoids

        # Condition 1: Volume of cluster ellipsoids must be below unclustered ellipsoid
        c_ellipsoid = []
        c_volume = 0
        for i in range(estimator.n_clusters):
            fit_ellipsoid = CEllipsoid.fit(points[estimator.labels_ == i], inflate_factor)
            if fit_ellipsoid is not None:
                c_ellipsoid.append(fit_ellipsoid)
                c_ellipsoid[-1].indices = estimator.labels_ == i
                c_volume = c_volume + c_ellipsoid[-1].volume
            else:
                if debug:
                    print("Rejected subdivision. Reason: unable to fit ellipsoid.")
                return points, ellipsoids

        if c_volume/ellipsoid_volume > cluster_volume_ratio:
            if debug:
                print("Rejected subdivision. Reason: volume ratio %f" % (c_volume/ellipsoid_volume))
            return points, ellipsoids

        # Condition 2: Clusters must be separated w/ little overlap
        distance = 0
        num_distances = 0
        for i in range(len(c_ellipsoid)-1):
            for j in range(i+1,len(c_ellipsoid)):
                distance = distance + self.ellipsoid_distance(c_ellipsoid[i], c_ellipsoid[j])
                num_distances = num_distances + 1
        if distance / num_distances < cluster_min_distance:
            if debug:
                print("Rejected subdivision. Reason: distance %f" % (distance / num_distances))
            return points, ellipsoids

        # Recursively subdivide into more clusters
        if debug:
            print("Accepted subdivision into %d clusters" % len(c_ellipsoid))
        return self.recursive_clustering(points, c_ellipsoid, cluster_volume_ratio, cluster_min_distance)

    def importance_sample(self, target_d, n_samples, timeout=60):
        points = self.live_points
        values = target_d.prob(points)
        self._num_pi_evals += len(points)

        # Perform recursive clustering on the sample points
        clusters, c_ellipsoids = self.recursive_clustering(points)

        # Perform nested sampling on each cluster
        n_samples_ell = int((n_samples-len(self.samples)) / len(c_ellipsoids))
        n_samples_ell = max(n_samples_ell, 1)
        for idx, c_ellipsoid in enumerate(c_ellipsoids):
            for i in range(n_samples_ell):
                Lmin = np.min(values[c_ellipsoid.indices])

                # Add the point with lowest likelihood to the resulting sample set
                g_indices = np.argwhere(c_ellipsoid.indices == True).flatten()
                assert len(g_indices) > 0

                min_idx = g_indices[np.argmin(values[c_ellipsoid.indices])]
                self.samples = np.vstack((self.samples, points[min_idx])) if self.samples.size else points[min_idx].reshape(1,-1)

                # Keep the likelihood value associated with the sample
                self.L = np.concatenate((self.L, np.array([Lmin]))) if self.L.size else np.array([Lmin])

                # Replace the point with lowest likelihood with a new sample from the proposal distribution
                points[min_idx], values[min_idx] = self.resample(Lmin, target_d, c_ellipsoid, timeout)

        # Update Z (evidence)
        X = np.array([np.exp(-i / self.N) for i in range(len(self.samples))])
        W = np.array([X[i - 1] - X[i] for i in range(1, len(self.samples))])
        W = np.concatenate((np.array([X[0]]), W))
        Z = np.sum(self.L * W)

        self.weights = (self.L * W) / Z

        self.live_points = points
        self.ellipsoids = c_ellipsoids
        self._update_model()
        return self.samples, self.weights

    def draw(self, ax):
        if len(self.space_max) == 1:
            return self.draw1d(ax)

        elif len(self.space_max) == 2:
            return self.draw2d(ax)
        return []

    def draw1d(self, ax):
        res = []
        for point in self.live_points:
            res.extend(ax.plot(point, 0.1, "go"))
        res.extend(ax.plot(point, 0.1, "go", label="live points"))
        res.extend(plot_pdf(ax, self, self.space_min, self.space_max, alpha=1.0, options="r-", resolution=0.01,label="$q(x)$"))
        return res

    def draw2d(self, ax):
        res = []
        for sample in self.live_points:
            res.extend(ax.plot([sample[0]], [sample[1]], c="g", marker="o", alpha=0.2))
        res.extend(ax.plot([sample[0]], [sample[1]], c="g", marker="o", alpha=0.2, label="live points"))

        res.extend(plot_pdf2d(ax, self, self.space_min, self.space_max, alpha=0.5, resolution=0.02, colormap=cm.viridis, label="$q(x)$"))
        return res

    # def draw2d(self, ax):
    #     res = []
    #     for sample in self.live_points:
    #         res.extend(ax.plot([sample[0]], [sample[1]], 0, c="g", marker="o", alpha=0.2))
    #     res.extend(ax.plot([sample[0]], [sample[1]], 0, c="g", marker="o", alpha=0.2, label="live points"))
    #
    #     res.extend(plot_pdf2d(ax, self, self.space_min, self.space_max, alpha=0.5, resolution=0.02, colormap=cm.viridis, label="$q(x)$"))
    #     return res