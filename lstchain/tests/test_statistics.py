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
