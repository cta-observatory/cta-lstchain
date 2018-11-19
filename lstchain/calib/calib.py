"""
Calibration functions
"""

import numpy as np
from ctapipe.image.charge_extractors import LocalPeakIntegrator


def lst_calibration(event, telescope_id):
    """
    Custom lst calibration.
    Update event.dl1.tel[telescope_id] with calibrated image and peakpos

    Parameters
    ----------
    event: ctapipe event container
    telescope_id: int
    channel: int
    """

    data = event.r0.tel[telescope_id].waveform

    ped = event.mc.tel[telescope_id].pedestal  # the pedestal is the
    # average (for pedestal events) of the *sum* of all samples,
    # from sim_telarray


    nsamples = data.shape[2]  # total number of samples

    # Subtract pedestal baseline. atleast_3d converts 2D to 3D matrix

    pedcorrectedsamples = data - np.atleast_3d(ped) / nsamples

    integrator = LocalPeakIntegrator(None, None)
    integration, peakpos, window = integrator.extract_charge(
        pedcorrectedsamples)  # these are 2D matrices num_gains * num_pixels

    signals = integration.astype(float)

    dc2pe = event.mc.tel[telescope_id].dc_to_pe  # numgains * numpixels
    signals *= dc2pe

    event.dl1.tel[telescope_id].image = signals
    event.dl1.tel[telescope_id].peakpos = peakpos