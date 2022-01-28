import numpy as np


def test_log_gaussian():
    from lstchain.image.pdf import log_gaussian
    mean = 1.0
    sigma = 2.0
    x = np.asarray([0.0, 1.0, 2.0])
    log_gauss = log_gaussian(x, mean, sigma)
    assert log_gauss[0] == log_gauss[2]
    assert np.argmax(log_gauss) == 1
    assert np.isclose(log_gauss[1], -1.612086, atol=1e-5)


def test_log_asygaussian2d():
    from lstchain.image.pdf import log_asygaussian2d
    x_cm, y_cm, width, length, psi, rl = 1, 1, 0.5, 2, 0.0, 1.0
    size1, size2 = 5, 10
    x = np.asarray([-0.5, 0, 0.5, 1.5, 1.5])
    y = np.asarray([0.0, 0.0, 0.0, 0.0, 1.0])
    out1 = log_asygaussian2d(size1, x, y, x_cm, y_cm, width, length, psi, rl)
    out2 = log_asygaussian2d(size2, x, y, x_cm, y_cm, width, length, psi, rl)
    assert np.isclose(2*np.exp(out1), np.exp(out2), rtol=1e-3).all()
    assert out1[3] != out1[4]
    assert out2[3] < out2[4]
