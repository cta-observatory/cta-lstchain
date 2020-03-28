import astropy.units as u
import numpy as np


__all__ = ['proton_bess']


def proton_bess(E):
    """ From http://adsabs.harvard.edu/abs/2000ApJ...545.1135S
    For each energy point, return the Proton flux

    Parameters
    -----------
    E: `numpy.ndarray` of astropy.units.quantity.Quantity (energy units)

    Returns
    -------
    dFdEdO: `numpy.ndarray` differential energy spectrum.
          astropy.units.quantity.Quantity units: 1/u.TeV / u.cm**2 / u.s / u.sr
    par: `dict` with spectral parameters
    """

    f0 = 9.6e-6 / u.TeV / u.cm**2 / u.s / u.sr
    alpha = -2.70
    e0 = 1. * u.TeV

    par_var = [f0, alpha, e0]
    par_dic = ['f0', 'alpha', 'e0']
    par = dict(zip(par_dic, par_var))

    dFdEdO = f0 * np.power(E/e0, alpha)

    return dFdEdO.to(1 / u.TeV / u.cm**2 / u.s / u.sr), par
