import numpy as np
import astropy.units as u

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

    TODO: Introduce any spectral form
    """

    if sp_idx == -1:
        integral_e = np.log(emax / emin) / e0**sp_idx
    else:
        integral_e = (emax**(sp_idx + 1) - emin**(sp_idx + 1)) \
            / (sp_idx + 1) / e0**sp_idx

    return integral_e

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
        omega = 2 * np.pi * (1-np.cos(cone)) * u.sr

    integral = int_diff_sp(emin, emax, sp_idx, e0)

    rate = norm * area * omega * integral

    return rate


def weight(emin, emax, sim_sp_idx, w_sp_idx, rate, nev, e0):
    """
    Calculates the weight of events to transform a power-law distribution
    with spectral index sim_sp_idx to a power-law distribution with 
    spectral index w_sp_idx

    Parameters
    ----------
    emin:       `float` minimum energy
    emax:       `float` maximum energy
    sim_sp_idx: `float` simulated spectral index of the power-law distribution
    w_sp_idx:   `float` weighted spectral index of the power-law distribution
    rate:       `float` rate of simulated events
    nev:        `int`   number of simulated events 
    e0:         `float` normalization energy

    Returns
    ----------
    weight: `float` rate of events
    """
    sim_integral = nev / int_diff_sp(emin, emax, sim_sp_idx, e0)
    norm_sim = sim_integral * int_diff_sp(emin, emax, w_sp_idx, e0)

    weight = rate / norm_sim

    return weight
