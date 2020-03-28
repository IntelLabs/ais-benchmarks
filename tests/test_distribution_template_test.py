import unittest
import numpy as np
import distributions as d


class CDistributionsTests(unittest.TestCase):
    """
    Directions to write tests for probability distributions

    - Test batched and unbatched
        - Sample method (just check dimensions are correct)
        - prob and log_prob computations
        - condition (check that samples and probs are computed correctly after conditioning)

    - Test with multiple dimensions

    - Test other methods
        - marginal
        - integral
    """
