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
    dFdE: `numpy.ndarray` differential energy spectrum. 
          astropy units: 1/u.TeV / u.cm**2 / u.s
    par: `dict` with spectral parameters
    """

    f0    = 3.23e-11 / u.TeV / u.cm**2 / u.s
    alpha = -2.47
    beta  = -0.24
    E0    = 1 * u.TeV

    par_var = [f0, alpha, beta, E0]
    par_dic = ['f0', 'alpha', 'beta', 'E0']
    par = dict(zip(par_dic, par_var))

    dFdE  = f0 * np.power(E / E0, alpha + beta * np.log10(E/E0))

    return dFdE.to(1/u.TeV / u.cm**2 / u.s), par

def crab_HEGRA(E):
    """ From http://adsabs.harvard.edu/abs/2004ApJ...614..897A
    For each energy point, return the Crab Nebula flux

    Parameters
    -----------
    E: `numpy.ndarray` of floats with astropy units of energy

    Returns
    -------
    dFdE: `numpy.ndarray` differential energy spectrum. 
          astropy units: 1/u.TeV / u.cm**2 / u.s
    par: `dict` with spectral parameters
    """

    f0    = 2.83e-11 / u.TeV / u.cm**2 / u.s
    alpha = -2.62
    E0    = 1 * u.TeV

    par_var = [f0, alpha, E0]
    par_dic = ['f0', 'alpha', 'E0']
    par = dict(zip(par_dic, par_var))

    dFdE  = f0 * np.power(E / E0, alpha)

    return dFdE.to(1/u.TeV / u.cm**2 / u.s), par
