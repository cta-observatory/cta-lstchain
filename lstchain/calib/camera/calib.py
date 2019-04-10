"""
Calibration functions
"""

import numpy as np
from ctapipe.image.charge_extractors import LocalPeakIntegrator
from ctapipe.calib.camera.gainselection import ThresholdGainSelector


def lst_calibration(event, telescope_id):
    """
    Custom lst calibration.
    Update event.dl1.tel[telescope_id] with calibrated image and peakpos

    Parameters
    ----------
    event: ctapipe event container
    telescope_id: int
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


def gain_selection(waveform, image, peakpos, cam_id, threshold):

    """
    Custom lst calibration.
    Update event.dl1.tel[telescope_id] with calibrated image and peakpos

    Parameters
    ----------
    waveform: array of waveforms of the events
    image: array of calibrated pixel charges
    peakpos: array of pixel peak positions
    cam_id: str
    threshold: int threshold to change form high gain to low gain
    """
    assert image.shape[0] == 2

    gainsel = ThresholdGainSelector(select_by_sample=True)
    gainsel.thresholds[cam_id] = threshold

    waveform, gain_mask = gainsel.select_gains(cam_id, waveform)
    signal_mask = gain_mask.max(axis=1)

    combined_image = image[0].copy()
    combined_image[signal_mask] = image[1][signal_mask].copy()
    combined_peakpos = peakpos[0].copy()
    combined_peakpos[signal_mask] = peakpos[1][signal_mask].copy()

    return combined_image, combined_peakpos



def combine_channels(event, tel_id, threshold):
    """
    Combine the channels for the image and peakpos arrays in the event.dl1 containers
    The `event.dl1.tel[tel_id].image` and `event.dl1.tel[tel_id].peakpos` are replaced by their combined versions

    Parameters
    ----------
    event: `ctapipe.io.containers.DataContainer`
    """

    cam_id = event.inst.subarray.tel[tel_id].camera.cam_id

    waveform = event.r0.tel[tel_id].waveform
    signals = event.dl1.tel[tel_id].image
    peakpos = event.dl1.tel[tel_id].peakpos

    combined_image, combined_peakpos = gain_selection(waveform, signals, peakpos, cam_id, threshold)
    event.dl1.tel[tel_id].image = combined_image
    event.dl1.tel[tel_id].peakpos = combined_peakpos
