import astropy.units as u
import numpy as np

def Proton_BESS(E):
    """ From http://adsabs.harvard.edu/abs/2000ApJ...545.1135S
    For each energy point, return the Proton flux

    Parameters
    -----------
    E: `numpy.ndarray` of floats with astropy units of energy

    Returns
    -------
    dFdEdO: `numpy.ndarray` differential energy spectrum. 
          astropy units: 1 / u.TeV / u.cm**2 / u.s / u.sr
    par: `dict` with spectral parameters
    """

    f0 = 9.6e-9 / u.GeV / u.cm**2 / u.s / u.sr
    alpha = -2.70
    E0 = 1 * u.TeV

    par_var = [f0, alpha, E0]
    par_dic = ['f0', 'alpha', 'E0']
    par = dict(zip(par_dic, par_var))

    dFdEdO = f0 * np.power(E/E0, alpha)

    return dFdEdO.to(1 / u.TeV / u.cm**2 / u.s / u.sr), par 
