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



def test_sigma_clipping():
    from lstchain.statistics import sigma_clipped_mean_std

    rng = np.random.default_rng()

    n_pixels = 10
    n_events = 10000
    true_mean = np.linspace(45, 55, n_pixels)
    true_std = np.linspace(5, 10, n_pixels)
    values = rng.normal(true_mean, true_std, (n_events, n_pixels))

    outliers = rng.binomial(2, 0.01, values.shape).astype(bool)
    values[outliers] = rng.normal(5, 1, np.count_nonzero(outliers))

    assert not np.allclose(values.mean(axis=0), true_mean, rtol=0.01)
    assert not np.allclose(values.std(axis=0), true_std, rtol=0.01)

    with np.printoptions(precision=3):
        mean, std, mask = sigma_clipped_mean_std(values, axis=0, max_sigma=3)


    assert np.allclose(mean, true_mean, rtol=0.01)
    assert np.allclose(std, true_std, rtol=0.1)
    assert (mask[outliers] == False).all()


def test_sigma_clipping_no_outliers():
    from lstchain.statistics import sigma_clipped_mean_std, expected_ignored

    rng = np.random.default_rng()

    n_events = 1_000_000
    true_mean = 20
    true_std = 2
    values = rng.normal(true_mean, true_std, n_events)

    for max_sigma in (2, 3, 4):
        for n_iterations in range(1, 6):
            mean, std, mask = sigma_clipped_mean_std(
                values, max_sigma=max_sigma, n_iterations=n_iterations,
            )
            ignored = np.count_nonzero(~mask) / mask.size
            expected = expected_ignored(max_sigma, n_iterations)

            assert np.allclose(mean, true_mean, rtol=0.01), f'{max_sigma}, {n_iterations}'
            assert np.allclose(std, true_std, rtol=0.01), f'{max_sigma}, {n_iterations}'
            assert np.isclose(ignored, expected, rtol=0.1), f'{max_sigma}, {n_iterations}'


def test_sigma_clipping_masked():
    from lstchain.statistics import sigma_clipped_mean_std

    rng = np.random.default_rng()

    n_pixels = 10
    n_events = 10000
    true_mean = np.linspace(45, 55, n_pixels)
    true_std = np.linspace(5, 10, n_pixels)
    values = rng.normal(true_mean, true_std, (n_events, n_pixels))

    outliers = rng.binomial(2, 0.01, values.shape).astype(bool)
    values[outliers] = rng.normal(5, 1, np.count_nonzero(outliers))

    broken = np.zeros((n_events, n_pixels), dtype=bool)
    broken[:, 5] = True
    values[broken] = 0

    values = np.ma.array(values, mask=broken)

    with np.printoptions(precision=3):
        mean, std, mask = sigma_clipped_mean_std(values, axis=0, max_sigma=3)


    assert np.allclose(mean, true_mean, rtol=0.01)
    assert np.allclose(std, true_std, rtol=0.1)
    assert (mask[outliers] == False).all()
