import numpy as np
import numexpr as ne


def log_gaussian(x, mean, sigma):
    """
    Evaluate the log of a normal law

    Parameters
    ----------
    x: float or array-like
        Value at which the log gaussian is evaluated
    mean: float
        Central value of the normal distribution
    sigma: float
        Width of the normal distribution

    Returns
    -------
    log_pdf: float or array-like
        Log of the evaluation of the normal law at x

    """
    pi = np.pi # NOQA
    return ne.evaluate("-(x - mean) ** 2 / (2 * sigma ** 2) - log((sqrt(2 * pi) * sigma))")


def logAsy_gaussian2d(size, x, y, x_cm, y_cm, width, length, psi, rl):
    """
    Evaluate the log of a bi-dimensional gaussian law with asymmetry along the
    main axis

    Parameters
    ----------
    size: float
        Integral of the 2D Gaussian
    x, y: float or array-like
        Position at which the log gaussian is evaluated
    x_cm, y_cm: float
        Center of the 2D Gaussian
    width, length: float
        Standard deviations of the 2 dimensions of the 2D Gaussian law
    psi: float
        Orientation of the 2D Gaussian
    rl: float
        asymmetry factor between the two lengths

    Returns
    -------
    log_pdf: float or array-like
        Log of the evaluation of the 2D gaussian law at (x,y)

    """
    le = (x - x_cm) * np.cos(psi) + (y - y_cm) * np.sin(psi)
    wi = -(x - x_cm) * np.sin(psi) + (y - y_cm) * np.cos(psi)
    pos = (le < 0.0)
    rl_array = rl * pos + 1 * ~pos
    norm = 1/((rl + 1.0) * np.pi * width * length)
    a = 2 * (rl_array*length)**2
    b = 2 * width**2
    log_pdf = le**2/a + wi**2/b
    return np.log(norm) + np.log(size) - log_pdf
