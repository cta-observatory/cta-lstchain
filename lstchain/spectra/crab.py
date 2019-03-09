import astropy.units as u
import numpy as np

def Crab_MAGIC(E):
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
    MAGIC_par=[3.23e-11 / u.TeV / u.cm**2 / u.s, -2.47, -0.24]
    E0 = 1 * u.TeV
    dFdE = MAGIC_par[0]*np.power(E/E0,MAGIC_par[1]+MAGIC_par[2]*np.log10(E/E0))

    return dFdE.to(1/u.TeV / u.cm**2 / u.s) 
