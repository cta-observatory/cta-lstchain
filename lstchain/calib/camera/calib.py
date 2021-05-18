"""
Calibration functions
"""

__all__ = [
    'lst_calibration',
    'load_calibrator_from_config',
    'load_image_extractor_from_config',
    'load_gain_selector_from_config'
]


import numpy as np
from ctapipe.image import ImageExtractor, extractor
from ctapipe.calib.camera import GainSelector
from ctapipe.calib.camera import gainselection
from ctapipe.calib import CameraCalibrator
from astropy.utils import deprecated
from traitlets.config import Config
from ...io.config import get_standard_config, replace_config


def load_gain_selector_from_config(custom_config):
    """
    Return a gain selector from a custom config.
    The passed custom_config superseeds the standard config.
    Parameters
    ----------
    custom_config: dictionnary. Should contain:
        - gain_selector
        - gain_selector_config

    Returns
    -------

    """

    config = replace_config(get_standard_config(), custom_config)
    conf = Config({config['gain_selector']: config['gain_selector_config']})
    return GainSelector.from_name(config['gain_selector'], config=conf)


def load_image_extractor_from_config(custom_config, subarray):
    """
    Return an image extractor from a custom config.
    The passed custom_config superseeds the standard config.
    Parameters
    ----------
    config: dictionnary. Should contains:
        - image_extractor
        - image_extractor_config

    Returns
    -------

    """
    config = replace_config(get_standard_config(), custom_config)
    conf = Config(config)
    return ImageExtractor.from_name(config['image_extractor'], subarray=subarray, config=conf)


def load_calibrator_from_config(custom_config, subarray):
    """
    Return a CameraCalibrator class corresponding to the given configuration.
    The passed custom_config superseeds the standard config.

    Parameters
    ----------
    custom_config: dictionnary. Should contain:
        - image_extractor
        - image_extractor_config
        - gain_selector
        - gain_selector_config

    Returns
    -------

    """

    config = replace_config(get_standard_config(), custom_config)

    image_extractor = load_image_extractor_from_config(config, subarray)

    cal = CameraCalibrator(subarray=subarray,
                           image_extractor=image_extractor,
                           )
    return cal

@deprecated('08/07/2020', message='this function will be deleted in the next release')
def lst_calibration(event, telescope_id, high_gain_threshold):
    """
    Custom lst calibration.
    Update event.dl1.tel[telescope_id] with calibrated image and peakpos

    Parameters
    ----------
    event: ctapipe event container
    telescope_id: int
    high_gain_threshold: ADC threshold to select low-gain channel
    """

    data = event.r0.tel[telescope_id].waveform

    ped = event.mc.tel[telescope_id].pedestal  # the pedestal is the
    # average (for pedestal events) of the *sum* of all samples,
    # from sim_telarray


    nsamples = data.shape[2]  # total number of samples

    # Subtract pedestal baseline. atleast_3d converts 2D to 3D matrix

    pedcorrectedsamples = data - np.atleast_3d(ped) / nsamples

    integrator = extractor.LocalPeakWindowSum()
    integration, peak_time = integrator(pedcorrectedsamples)  # these are 2D matrices num_gains * num_pixels

    signals = integration.astype(float)

    dc2pe = event.mc.tel[telescope_id].dc_to_pe  # numgains * numpixels
    signals *= dc2pe

    # threshold = 4094
    image, peak_time = gain_selection(data, signals, peak_time, high_gain_threshold)

    event.dl1.tel[telescope_id].image = image
    event.dl1.tel[telescope_id].peak_time = peak_time


@deprecated('28/06/2019', message='gain selection is now performed at <= R1 calibration level')
def gain_selection(waveform, charges, peak_time, threshold):

    """
    Custom lst calibration.
    Update event.dl1.tel[telescope_id] with calibrated image and peakpos

    Parameters
    ----------
    waveform: array of waveforms of the events
    charges: array of calibrated pixel charges
    peak_time: array of pixel peak positions
    cam_id: str
    threshold: int threshold to change form high gain to low gain
    """
    assert charges.shape[0] == 2

    gain_selector = gainselection.ThresholdGainSelector(threshold=threshold)

    gain_mask = gain_selector(waveform)

    combined_image = charges[gain_mask, np.arange(charges.shape[1])]
    combined_peak_time = peak_time[gain_mask, np.arange(peak_time.shape[1])]

    return combined_image, combined_peak_time


@deprecated('28/06/2019', message='channel selection is now performed at <= R1 calibration level')
def combine_channels(event, tel_id, threshold):
    """
    Combine the channels for the image and peakpos arrays in the event.dl1 containers
    The `event.dl1.tel[tel_id].image` and `event.dl1.tel[tel_id].peakpos` are replaced by their combined versions

    Parameters
    ----------
    event: `ctapipe.containers.ArrayEventContainer`
    tel_id: int
        id of the telescope
    threshold: float
        threshold value to consider a pixel as saturated in the waveform
    """

    waveform = event.r0.tel[tel_id].waveform
    charges = event.dl1.tel[tel_id].image
    peak_time = event.dl1.tel[tel_id].peak_time

    combined_image, combined_peak_time = gain_selection(waveform, charges, peak_time, threshold)
    event.dl1.tel[tel_id].image = combined_image
    event.dl1.tel[tel_id].peak_time = combined_peak_time
