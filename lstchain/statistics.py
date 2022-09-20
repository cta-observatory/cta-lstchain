import numba
import numpy as np
from numba.experimental import jitclass


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
