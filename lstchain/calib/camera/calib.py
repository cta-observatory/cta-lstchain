"""
Calibration functions
"""

__all__ = [
    'load_calibrator_from_config',
    'load_image_extractor_from_config',
    'load_gain_selector_from_config'
]


from ctapipe.image import ImageExtractor
from ctapipe.calib.camera import GainSelector
from ctapipe.calib import CameraCalibrator
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
    gain_selector: ctapipe.calib.camera.GainSelector
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
    config: dictionnary. Should contain:
     - image_extractor
     - image_extractor_config

    Returns
    -------
    image_extractor: ctapipe.image.ImageExtractor
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
    calibrator: ctapipe.calib.CameraCalibrator
    """

    config = replace_config(get_standard_config(), custom_config)

    image_extractor = load_image_extractor_from_config(config, subarray)

    cal = CameraCalibrator(subarray=subarray,
                           image_extractor=image_extractor,
                           )
    return cal
