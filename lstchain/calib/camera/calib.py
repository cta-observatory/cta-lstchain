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
from ctapipe.image import extractor
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
    gain_selector = getattr(gainselection, config['gain_selector'])
    return gain_selector(conf)


def load_image_extractor_from_config(custom_config):
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
    conf = Config({config['image_extractor']: config['image_extractor_config']})
    image_extractor = getattr(extractor, config['image_extractor'])
    return image_extractor(conf)


def load_calibrator_from_config(custom_config):
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

    gain_selector = load_gain_selector_from_config(config)
    image_extractor = load_image_extractor_from_config(config)

    cal = CameraCalibrator(image_extractor=image_extractor,
                           gain_selector=gain_selector,
                           )

    return cal


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
    integration, pulse_time = integrator(pedcorrectedsamples)  # these are 2D matrices num_gains * num_pixels

    signals = integration.astype(float)

    dc2pe = event.mc.tel[telescope_id].dc_to_pe  # numgains * numpixels
    signals *= dc2pe

    # threshold = 4094
    image, pulse_time = gain_selection(data, signals, pulse_time, high_gain_threshold)

    event.dl1.tel[telescope_id].image = image
    event.dl1.tel[telescope_id].pulse_time = pulse_time


@deprecated('28/06/2019', message='gain selection is now performed at <= R1 calibration level')
def gain_selection(waveform, charges, pulse_time, threshold):

    """
    Custom lst calibration.
    Update event.dl1.tel[telescope_id] with calibrated image and peakpos

    Parameters
    ----------
    waveform: array of waveforms of the events
    charges: array of calibrated pixel charges
    pulse_time: array of pixel peak positions
    cam_id: str
    threshold: int threshold to change form high gain to low gain
    """
    assert charges.shape[0] == 2

    gain_selector = gainselection.ThresholdGainSelector(threshold=threshold)

    waveform, gain_mask = gain_selector(waveform)

    combined_image = charges[gain_mask, np.arange(charges.shape[1])]
    combined_pulse_time = pulse_time[gain_mask, np.arange(pulse_time.shape[1])]

    return combined_image, combined_pulse_time


@deprecated('28/06/2019', message='channel selection is now performed at <= R1 calibration level')
def combine_channels(event, tel_id, threshold):
    """
    Combine the channels for the image and peakpos arrays in the event.dl1 containers
    The `event.dl1.tel[tel_id].image` and `event.dl1.tel[tel_id].peakpos` are replaced by their combined versions

    Parameters
    ----------
    event: `ctapipe.io.containers.DataContainer`
    tel_id: int
        id of the telescope
    threshold: float
        threshold value to consider a pixel as saturated in the waveform
    """

    waveform = event.r0.tel[tel_id].waveform
    charges = event.dl1.tel[tel_id].image
    pulse_time = event.dl1.tel[tel_id].pulse_time

    combined_image, combined_pulse_time = gain_selection(waveform, charges, pulse_time, threshold)
    event.dl1.tel[tel_id].image = combined_image
    event.dl1.tel[tel_id].pulse_time = combined_pulse_time
