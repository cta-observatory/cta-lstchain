import numba
import numpy as np
from numba.experimental import jitclass
from scipy.stats import norm, truncnorm


@jitclass(dict(
    n=numba.uint64,
    counts=numba.uint64[:],
    _mean=numba.float64[:],
    _m2=numba.float64[:],
))
class OnlineStats:
    '''
    A class implementing Welford's algorithm for online mean/variance.

    The class is able to track n statistics simultaneously.
    '''

    def __init__(self, n):
        self.n = n
        self.counts = np.zeros(n, dtype=np.uint64)
        self._mean = np.zeros(n, dtype=np.float64)
        self._m2 = np.zeros(n, dtype=np.float64)

    def add_value(self, idx, value):
        '''Add a new value at idx to the statistics'''
        self.counts[idx] += 1
        delta = value - self._mean[idx]
        self._mean[idx] += delta / self.counts[idx]
        delta2 = value - self._mean[idx]
        self._m2[idx] += delta * delta2

    def add_values(self, values):
        '''Add a new value in each of the tracked statistics'''
        if values.ndim != 1 or len(values) != self.n:
            raise ValueError('Expected a 1d array of length OnlineStats.n')

        for i in range(self.n):
            self.add_value(i, values[i])

    def add_values_at_indices(self, indices, values):
        '''Add a new value in each of the tracked statistics'''
        if values.ndim != 1 or len(values) != len(indices):
            raise ValueError('Expected two 1d arrays of matching length')

        for i in range(len(indices)):
            self.add_value(indices[i], values[i])

    @property
    def mean(self):
        '''Get the current mean values of all tracked statistics'''
        mean = np.full_like(self._mean, np.nan)
        valid = self.counts > 0
        mean[valid] = self._mean[valid]
        return mean

    @property
    def var(self):
        '''Get the current sample variance of all tracked statistics'''
        var = np.full_like(self._mean, np.nan)
        valid = self.counts > 1
        var[valid] = self._m2[valid] / (self.counts[valid] - 1)
        return var

    @property
    def std(self):
        '''Get the current sample std. dev. of all tracked statistics'''
        return np.sqrt(self.var)



def sigma_clipped_mean_std(values, axis=0, max_sigma=4, n_iterations=5):
    '''
    Compute robust estimates of mean and std via sigma clipping

    Values outside max_sigma * std are removed from the sample and then
    mean and std are computed again.
    '''


    # support masked array
    if hasattr(values, 'mask'):
        values = values.copy()
    else:
        mask = np.zeros(values.shape, dtype=bool)
        values = np.ma.array(values, mask=mask, copy=True)

    original_mask = values.mask.copy()
    mean = values.mean(axis=axis)
    std = values.std(axis=axis)

    for _ in range(n_iterations):
        values.mask = original_mask | (np.abs(values - mean) >= (max_sigma * std))
        mean = values.mean(axis=axis)
        std = values.std(axis=axis)

    # correct std for bias introduced by clipping
    std /= expected_std(max_sigma, n_iterations)

    return mean, std, values.mask


def expected_std(max_sigma, n_iterations):
    '''Expected std of std normal data after applying sigma clipping'''
    std = 1
    truncdist = truncnorm(-max_sigma, max_sigma)

    for _ in range(n_iterations):
        truncdist = truncnorm(-max_sigma * std, max_sigma * std)
        std = truncdist.std()

    return std


def expected_ignored(max_sigma, n_iterations):
    '''Calculate the expected percentage of discarded samples for
    a normal distribution without outliers
    '''
    std = expected_std(max_sigma, n_iterations - 1)
    stdnorm = norm(0, 1)
    return 1.0 - (stdnorm.cdf(max_sigma * std) - stdnorm.cdf(-max_sigma * std))
