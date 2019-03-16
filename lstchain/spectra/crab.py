import astropy.units as u
import numpy as np

def crab_MAGIC(E):
    """ From http://adsabs.harvard.edu/abs/2015JHEAp...5...30A
    For each energy point, return the Crab Nebula flux

    Parameters
    -----------
    E: `numpy.ndarray` of floats with astropy units of energy

    Returns
    -------
    dFdE: differential energy spectrum. 
          astropy units: 1/u.TeV / u.cm**2 / u.s
    """

    f0    = 3.23e-11 / u.TeV / u.cm**2 / u.s
    alpha = -2.47
    beta  = -0.24
    E0    = 1 * u.TeV
    dFdE  = f0 * np.power(E / E0, alpha + beta * np.log10(E/E0))

    return dFdE.to(1/u.TeV / u.cm**2 / u.s) 

def crab_HEGRA(E):
    f0    = 2.83e-11 / u.TeV / u.cm**2 / u.s
    alpha = -2.62
    E0    = 1 * u.TeV
    dFdE  = f0 * np.power(E / E0, alpha)

    return dFdE.to(1/u.TeV / u.cm**2 / u.s)
