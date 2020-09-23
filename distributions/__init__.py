from distributions.derived.CABCDistribution import ABCDistribution
from distributions.derived.CGenericNoisyFunction import GenericNoisyFunction
from distributions.mixture.CMixtureModel import CMixtureModel
from distributions.mixture.CGaussianMixtureModel import CGaussianMixtureModel
from distributions.parametric.CMultivariateNormal import CMultivariateNormal
from distributions.parametric.CMultivariateUniform import CMultivariateUniform
from distributions.nonparametric.CNearestNeighbor import CNearestNeighbor
from distributions.nonparametric.CKernelDensity import CKernelDensity
import distributions.benchden

"""
TODO LIST

Make all the numbers of base type CDistribution

Add CDiracDelta to the available parametric distributions
    - Prob is always zero
    - sample always returns the parameter value

Change the base type to CDistribution, parameters or random variable values can be conditioned

Missing processing to implement the CDistribution interface:
CNearestNeighbor

"""

