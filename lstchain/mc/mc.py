import numpy as np


def power_law_integrated_distribution(emin, emax, tot_num_events, spectral_index, bin_number=30):
    """
    For each bin, return the expected number of events for a power-law distribution.
    bins: `numpy.ndarray`, e.g. `np.logspace(np.log10(emin), np.logspace(emax))`
    Parameters
    ----------
    emin: `float`
    emax: `float`
    tot_num_events: `int`
    spectral_index: `float`

    Returns
    -------
    (bins, y):
    bins: `np.logspace(np.log10(emin), np.log10(emax), bin_number)`
    tuple of `numpy.ndarray`, len(y) = len(bins) - 1
    """
    bins = np.logspace(np.log10(emin), np.log10(emax), bin_number)

    if spectral_index == -1:
        y0 = tot_num_events / np.log(emax / emin)
        y = y0 * np.log(bins[1:] / bins[:-1])
    else:
        y0 = tot_num_events / (emax ** (spectral_index + 1) - emin ** (spectral_index + 1)) / (spectral_index + 1)
        y = y0 * (bins[1:] ** (spectral_index + 1) - bins[:-1] ** (spectral_index + 1)) / (spectral_index + 1)
    return bins, y