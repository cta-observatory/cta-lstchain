import numpy as np
import astropy.units as u
import sys
from gammapy.modeling.models import LogParabolaSpectralModel

__all__ = [
    'power_law_integrated_distribution',
    'int_diff_sp',
    'rate',
    'weight'
]

def power_law_integrated_distribution(emin, emax, tot_num_events, spectral_index, bin_number=30):
    """
    For each bin, return the expected number of events for a power-law distribution.
    bins: `numpy.ndarray`, e.g. `np.logspace(np.log10(emin), np.logspace(emax))`

    Parameters
    ----------
    emin:   `float` minimum energy
    emax:   `float` maximum energy
    tot_num_events:    `int`   total number of events simulated
    spectral_index: `float` spectral index of the power-law distribution

    Returns
    -------
    (bins, y):
    bins: `np.logspace(np.log10(emin), np.log10(emax), bin_number)`
    tuple of `numpy.ndarray`, len(y) = len(bins) - 1

    TODO: Introduce any spectral form
    """

    bins = np.logspace(np.log10(emin), np.log10(emax), bin_number)

    if spectral_index == -1:
        y0 = tot_num_events / np.log(emax / emin)
        y = y0 * np.log(bins[1:] / bins[:-1])
    else:

        y0 = tot_num_events / (emax**(spectral_index + 1) - emin**(spectral_index + 1)) * (spectral_index + 1)

        y = y0 * (bins[1:]**(spectral_index + 1) - bins[:-1]**(spectral_index + 1)) / (spectral_index + 1)
    return bins, y

def int_diff_sp(emin, emax, sp_idx, e0):
    """

    Parameters
    --------
    emin:  `float`  minimum energy
    emax:  `float`  maximum energy
    sp_idx:`float`  spectral index of the power-law distribution
    e0:    `float`  normalization energy

    Returns
    --------
    integral_e: `float` energy integral

    TODO: Introduce any spectral form
    """

    if sp_idx == -1:
        integral_e = np.log(emax / emin) / e0**sp_idx
    else:
        integral_e = (emax**(sp_idx + 1) - emin**(sp_idx + 1)) \
            / (sp_idx + 1) / e0**sp_idx

    return integral_e

def rate(shape, emin, emax, param, cone, area):
    """
    Calculates the rate of events for a power-law distribution,
    in a given energy range, collection area and solid angle

    Parameters
    ----------
    shape: `string` weighted spectrum shape
    emin:  `float`  minimum energy
    emax:  `float`  maximum energy
    param: `dict` with weighted spectral parameters
    cone:  `float`  angle [deg] for the solid angle cone
    area:  `float`  collection area [cm**2]

    if shape is 'PowerLaw':
    param should include 'f0','e0','alpha'
    dFdE  = f0 * np.power(E / e0, alpha)

    if shape is 'LogParabola':
    param should include 'f0','e0','alpha','beta'
    dFdE  = f0 * np.power(E / e0, alpha + beta * np.log10(E/e0))

    Returns
    ----------
    rate: `float` rate of events

    """

    if(cone == 0):
        omega = 1
    else:
        omega = 2 * np.pi * (1-np.cos(cone)) * u.sr

    if(shape == "PowerLaw"):
        if(len(param) != 3):
            print("param included {} parameters, not 3".format(len(param)))
            print("param should include 'f0', 'e0', 'alpha'")
            sys.exit(1)

        for key in ['f0','e0','alpha']:
            if(key not in param.keys()):
                print("{} is missing in param".format(key))
                print("param should include 'f0', 'e0', 'alpha'")
                sys.exit(1)
            
        integral = param['f0'] * int_diff_sp(emin, emax, param['alpha'], param['e0'])
    
    elif(shape == "LogParabola"):
        if(len(param) != 4):
            print("param included {} parameters, not 4".format(len(param)))
            print("param should include 'f0', 'e0', 'alpha', 'beta'")
            sys.exit(1)

        for key in ['f0','e0','alpha', 'beta']:
            if(key not in param.keys()):
                print("{} is missing in param".format(key))
                print("param should include 'f0', 'e0', 'alpha', 'beta'")
                sys.exit(1)

        log_parabola =  LogParabolaSpectralModel.from_log10(amplitude=param['f0'], reference=param['e0'], alpha=-1*param['alpha'], beta=-1*param['beta'])
        integral = log_parabola.integral(emin, emax)

    rate = area * omega * integral

    return rate


def weight(shape, emin, emax, sim_sp_idx, rate, nev, w_param):
    """
    Calculates the weight of events to transform a power-law distribution
    with spectral index sim_sp_idx to a power-law distribution with 
    spectral index w_sp_idx

    Parameters
    ----------
    shape:      `string` estimated spectrum shape
    emin:       `float` minimum energy
    emax:       `float` maximum energy
    sim_sp_idx: `float` simulated spectral index of the power-law distribution
    rate:       `float` rate of simulated events
    nev:        `int`   number of simulated events 
    w_param:    `dict` with weighted spectral parameters

    if shape is 'PowerLaw':
    w_param should include 'f0','e0','alpha'
    dFdE  = f0 * np.power(E / e0, alpha)

    if shape is 'LogParabola':
    w_param should include 'f0','e0','alpha','beta'
    dFdE  = f0 * np.power(E / e0, alpha + beta * np.log10(E/e0))

    Returns
    ----------
    weight: `float` rate of events
    """

    sim_integral = nev / int_diff_sp(emin, emax, sim_sp_idx, w_param['e0'])

    if(shape == "PowerLaw"):
        if(len(w_param) != 3):
            print("param included {} parameters, not 3".format(len(w_param)))
            print("param should include 'f0', 'e0', 'alpha'")
            sys.exit(1)
        
        for key in ['f0','e0','alpha']:
            if(key not in w_param.keys()):
                print("{} is missing in param".format(key))
                print("param should include 'f0', 'e0', 'alpha'")
                sys.exit(1)

        norm_sim = sim_integral * int_diff_sp(emin, emax, w_param['alpha'], w_param['e0'])
    
    elif(shape == "LogParabola"):
        if(len(w_param) != 4):
            print("param included {} parameters, not 4".format(len(w_param)))
            print("param should include 'f0', 'e0', 'alpha', 'beta'")
            sys.exit(1)

        for key in ['f0','e0','alpha', 'beta']:
            if(key not in w_param.keys()):
                print("{} is missing in param".format(key))
                print("param should include 'f0', 'e0', 'alpha', 'beta'")
                sys.exit(1)

        log_parabola =  LogParabolaSpectralModel.from_log10(amplitude=sim_integral/(u.s * u.cm * u.cm), reference=w_param['e0'], alpha=-1*w_param['alpha'], beta=-1*w_param['beta'])

        norm_sim = log_parabola.integral(emin, emax) * (u.s * u.cm * u.cm)

    weight = rate / norm_sim

    return weight
