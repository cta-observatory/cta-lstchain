import astropy.units as u
import numpy as np

def Electron_HESS_Fermi(E):
    """
    For each energy point, return the Proton flux

    Parameters
    -----------
    E: `numpy.ndarray` of floats with astropy units of energy

    Returns
    -------
    dFdEdO: differential energy spectrum. 
          astropy units: 1 / u.TeV / u.cm**2 / u.s / u.sr
    """

    f0 = 104.9e-13 / u.GeV / u.cm**2 / u.s / u.sr
    Gamma1 = 3.04
    Gamma2 = 3.78
    Eb = 0.94 * u.TeV
    alpha = 0.12
    E0 = 1 * u.TeV
    dFdEdO = f0 * np.power(E / E0, Gamma1) * \
        np.power(1 + np.power(E / Eb, 1 / alpha), (Gamma2-Gamma1) * alpha)

    return dFdEdO.to(1 / u.TeV / u.cm**2 / u.s / u.sr) 
