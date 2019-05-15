import astropy.units as u
import numpy as np


__all__ = ['electron_HESS']

def electron_HESS(E):
    """From https://indico.in2p3.fr/event/15018/
    Broken power-law from HESS:
    For each energy point, return the Proton flux

    Parameters
    -----------
    E: `numpy.ndarray` of astropy.units.quantity.Quantity (energy units) 

    Returns
    -------
    dFdEdO: `numpy.ndarray` differential energy spectrum. 
          astropy.units.quantity.Quantity units: 1/u.TeV / u.cm**2 / u.s / u.sr
    """

    f0 = 104.9e-13 / u.GeV / u.cm**2 / u.s / u.sr
    Gamma1 = -3.04
    Gamma2 = -3.78
    Eb = 0.94 * u.TeV
    alpha = 0.12
    E0 = 1 * u.TeV

    par_var = [f0, Gamma1, Gamma2, Eb, alpha, E0]
    par_dic = ['f0', 'Gamma1', 'Gamma2', 'Eb', 'alpha', 'E0']
    par = dict(zip(par_dic, par_var))

    dFdEdO = f0 * np.power(E / E0, Gamma1) * \
        np.power(1 + np.power(E / Eb, 1 / alpha), (Gamma2-Gamma1) * alpha)

    return dFdEdO.to(1 / u.TeV / u.cm**2 / u.s / u.sr), par 
