import numpy as np


class AbstractArm(object):
    def __init__(self, mean, random_state):
        """
        Args:
            mean: mean vector of the arm
            random_state (int): seed to make experiments reproducible
        """
        self.mean = mean
        self.D = len(mean)
        self.local_random = np.random.RandomState(random_state)

    def sample(self):
        pass


class ArmMultinomial(AbstractArm):
    def __init__(self, mean, random_state=0):
        """
        Multinomial arm
        Args:
             mean (array): mean vector
             random_state (int): seed to make experiments reproducible
        """
        super(ArmMultinomial, self).__init__(mean = np.array(mean),random_state = random_state)

    def sample(self):
        return np.array([ int(np.random.rand() < self.mean[i]) for i in range(self.D)])

class ArmExp(AbstractArm):
    def __init__(self, L, B=1., random_state=0):
        """
        Args:
            L (array): parameters of the exponential distributions
            B (float): upper bound of the distribution (lower is 0)
            random_state (int): seed to make experiments reproducible
        """
        assert B > 0.
        self.L = L
        self.B = B
        self.D = len(L)
        v_m = (1. - np.exp(-B*L)*(1. + B*L)) / L
        super(ArmExp, self).__init__(mean=v_m / (1. - np.exp(-L * B)),random_state=random_state)

    def cdf(self, x):
        cdf = lambda y: 1. - np.exp(-self.L*y)
        truncated_cdf = (cdf(x) - cdf(0)) / (cdf(self.B) - cdf(0))
        return truncated_cdf

    def inv_cdf(self, q):
        assert 0<= q.all() <= 1.
        v = - np.log(1. - (1. - np.exp(- self.L * self.B)) * q) / self.L
        return v

    def sample(self):
        # Inverse transform sampling
        # https://en.wikipedia.org/wiki/Inverse_transform_sampling
        q = np.random.rand(self.D)
        x = self.inv_cdf(q=q)
        return x
