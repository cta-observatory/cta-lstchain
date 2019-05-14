import numpy as np


def power_law_integrated_distribution(emin, emax, nev, sp_idx, bin_number=30):
    """
    For each bin, return the expected number of events for a power-law distribution.
    bins: `numpy.ndarray`, e.g. `np.logspace(np.log10(emin), np.logspace(emax))`
    Parameters
    ----------
    emin:   `float` minimum energy
    emax:   `float` maximum energy
    nev:    `int`   total number of events simulated
    sp_idx: `float` spectral index of the power-law distribution

    Returns
    -------
    (bins, y):
    bins: `np.logspace(np.log10(emin), np.log10(emax), bin_number)`
    tuple of `numpy.ndarray`, len(y) = len(bins) - 1

    TODO: Introduce any spectral form
    """
    bins = np.logspace(np.log10(emin), np.log10(emax), bin_number)

    if sp_idx == -1:
        y0 = nev / np.log(emax / emin)
        y = y0 * np.log(bins[1:] / bins[:-1])
    else:
        y0 = nev / (emax**(sp_idx + 1) - emin**(sp_idx + 1)) / (sp_idx + 1)
        y = y0 * (bins[1:]**(sp_idx + 1) - bins[:-1]**(sp_idx + 1)) / (sp_idx + 1)
    return bins, y

def int_diff_sp(emin, emax, sp_idx, e0):
    """
    
    TODO: Introduce any spectral form
    """

    if sp_idx == -1:
        integral_E = np.log(emax / emin) / e0**sp_idx
    else:
        integral_E = (emax**(sp_idx + 1) - emin**(sp_idx + 1)) \
            / (sp_idx + 1) / e0**sp_idx

    return integral_E

def rate(emin, emax, sp_idx, cone, area, norm, e0):
    """
    Calculates the rate of events for a power-law distribution, 
    in a given energy range, collection area and solid angle

    Parameters      
    ----------
    emin:  `float`  minimum energy
    emax:  `float`  maximum energy
    sp_idx:`float`  spectral index of the power-law distribution
    cone:  `float`  angle [deg] for the solid angle cone
    area:  `float`  collection area [cm**2]
    norm:  `float`  normalization of the differential energy spectrum at e0 
    e0:    `float`  normalization energy 

    Returns
    ----------
    rate: `float` rate of events

    TODO: Introduce any spectral form
    """

    if(cone == 0):
        omega = 1
    else:
        omega = 2 * np.pi * (1-np.cos(cone))

    integral = int_diff_sp(emin, emax, sp_idx, e0)

    rate = norm * area * omega * integral 

    return rate


def weight(emin, emax, sim_sp_idx, w_sp_idx, rate, nev, e0):
    """




    """
    sim_integral = nev / int_diff_sp(emin, emax, sim_sp_idx, e0)
    norm_sim = sim_integral * int_diff_sp(emin, emax, w_sp_idx, e0)

    weight = rate / norm_sim

    return weight
