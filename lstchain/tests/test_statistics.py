import numpy as np


def test_online_statistics():
    from lstchain.statistics import OnlineStats


    rng = np.random.default_rng()
    data = rng.normal(size=(100, 10))

    stats = OnlineStats(n=10)
    assert np.isnan(stats.mean).all()
    assert np.isnan(stats.var).all()
    assert np.isnan(stats.std).all()
    assert (stats.counts == 0).all()

    for row, sample in enumerate(data):
        for i, value in enumerate(sample):
            stats.add_value(i, value)

        assert np.allclose(stats.mean, np.mean(data[:row + 1], axis=0))
        assert np.all(stats.counts == row + 1)

        if row >= 1:
            assert np.allclose(stats.var, np.var(data[:row + 1], axis=0, ddof=1))
            assert np.allclose(stats.std, np.std(data[:row + 1], axis=0, ddof=1))
        else:
            assert np.isnan(stats.var).all()
            assert np.isnan(stats.std).all()


def test_online_statistics_at_indices():
    from lstchain.statistics import OnlineStats


    rng = np.random.default_rng()
    N = 10
    data = rng.normal(size=10000)
    indices = rng.integers(0, N, len(data), endpoint=False)

    stats = OnlineStats(n=N)
    stats.add_values_at_indices(indices, data)


    assert np.all(stats.counts == np.bincount(indices))
    mean = stats.mean
    std = stats.std

    for i in range(N):
        assert np.isclose(mean[i], np.mean(data[indices == i]))
        assert np.isclose(std[i], np.std(data[indices == i], ddof=1))
