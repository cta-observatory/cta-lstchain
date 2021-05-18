import astropy.units as u
import numpy as np

__all__ = [
    'crab_hegra',
    'crab_magic',
]


def crab_magic(E):
    """ From http://adsabs.harvard.edu/abs/2015JHEAp...5...30A
    For each energy point, return the Crab Nebula flux

    Parameters
    -----------
    E: `numpy.ndarray` of astropy.units.quantity.Quantity (energy units)

    Returns
    -------
    dFdE: `numpy.ndarray` differential energy spectrum.
          astropy.units.quantity.Quantity units: 1/u.TeV / u.cm**2 / u.s
    par: `dict` with spectral parameters
    """

    f0 = 3.23e-11 / u.TeV / u.cm ** 2 / u.s
    alpha = -2.47
    beta = -0.24
    e0 = 1. * u.TeV

    par_var = [f0, alpha, beta, e0]
    par_dic = ['f0', 'alpha', 'beta', 'e0']
    par = dict(zip(par_dic, par_var))

    dFdE = f0 * np.power(E / e0, alpha + beta * np.log10(E / e0))

    return dFdE.to(1 / u.TeV / u.cm ** 2 / u.s), par


def crab_hegra(E):
    """ From http://adsabs.harvard.edu/abs/2004ApJ...614..897A
    For each energy point, return the Crab Nebula flux

    Parameters
    -----------
    E: `numpy.ndarray` of astropy.units.quantity.Quantity (energy units)

    Returns
    -------
    dFdE: `numpy.ndarray` differential energy spectrum.
          astropy.units.quantity.Quantity units: 1/u.TeV / u.cm**2 / u.s
    par: `dict` with spectral parameters
    """

    f0 = 2.83e-11 / u.TeV / u.cm ** 2 / u.s
    alpha = -2.62
    e0 = 1. * u.TeV

    par_var = [f0, alpha, e0]
    par_dic = ['f0', 'alpha', 'e0']
    par = dict(zip(par_dic, par_var))

    dFdE = f0 * np.power(E / e0, alpha)

    return dFdE.to(1 / u.TeV / u.cm ** 2 / u.s), par
