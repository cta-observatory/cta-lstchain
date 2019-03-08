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


def gain_selection(waveform, signals, peakpos, cam_id, threshold):

    """
    Custom lst calibration.
    Update event.dl1.tel[telescope_id] with calibrated image and peakpos

    Parameters
    ----------
    waveform: array of waveforms of the events
    signals: array of calibrated pixel charges
    peakpos: array of pixel peak positions
    cam_id: str
    threshold: int threshold to change form high gain to low gain
    """
    '''
    combined = signals[0].copy() 
    peaks = peakpos[0].copy()
    for pixel in range(0, combined.size):
            if np.any(waveform[0][pixel] >= threshold):
                combined[pixel] = signals[1][pixel]
                peaks[pixel] = peakpos[1][pixel]
    '''
    ###Gain Selection using ctapipe GainSelector###
    gainsel = ThresholdGainSelector(select_by_sample=True)
    gainsel.thresholds[cam_id] = threshold
    
    waveform, gainmask = gainsel.select_gains(cam_id,
                                              waveform)
    signalmask = np.zeros(waveform.shape[0],
                          dtype=bool)

    for i in range(signalmask.size):
        signalmask[i] = gainmask[i].any()==True

    combined = signals[0].copy()
    combined[signalmask] = signals[1][signalmask]
    peaks = peakpos[0].copy()
    peaks[signalmask] = peakpos[1][signalmask]
                
    return combined, peaks
