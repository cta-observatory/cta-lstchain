from lstchain.calib import load_calibrator_from_config, load_gain_selector_from_config, load_image_extractor_from_config
from lstchain.io.config import get_standard_config


def test_load_calibrator_from_config(lst1_subarray):
    from lstchain.io.config import get_standard_config
    from ctapipe.calib import CameraCalibrator
    config = get_standard_config()
    cal = load_calibrator_from_config(config, lst1_subarray)
    assert isinstance(cal, CameraCalibrator)


def test_load_calibrator_from_config_LocalPeakWindowSum(lst1_subarray):
    config = {"image_extractor": "LocalPeakWindowSum"}
    cal = load_calibrator_from_config(config, lst1_subarray)
    assert cal.image_extractor_type.tel[1] == 'LocalPeakWindowSum'


def test_load_calibrator_from_config_GlobalPeakWindowSum(lst1_subarray):
    config = {"image_extractor": "GlobalPeakWindowSum"}
    cal = load_calibrator_from_config(config, lst1_subarray)
    assert cal.image_extractor_type.tel[1] == 'GlobalPeakWindowSum'


def test_load_image_extractor_from_config(lst1_subarray):
    from ctapipe.image import LocalPeakWindowSum

    config = get_standard_config()
    image_extractor = load_image_extractor_from_config(config, lst1_subarray)

    assert isinstance(image_extractor, LocalPeakWindowSum)
    assert image_extractor.window_shift[0][2] == 4
    assert image_extractor.window_width[0][2] == 8

    config = {'image_extractor': 'LocalPeakWindowSum',
              'LocalPeakWindowSum': {
                  "window_shift": 1,
                  "window_width": 10
              }
    }

    image_extractor = load_image_extractor_from_config(config, lst1_subarray)

    assert isinstance(image_extractor, LocalPeakWindowSum)
    assert image_extractor.window_shift[0][2] == 1
    assert image_extractor.window_width[0][2] == 10


def test_load_gain_selector_from_config_ManualGainSelector():
    from ctapipe.calib.camera.gainselection import ManualGainSelector

    for chan in ["HIGH", "LOW"]:
        config = {"gain_selector": "ManualGainSelector",
                  "gain_selector_config": {"channel": chan}
        }

        gain_selector = load_gain_selector_from_config(config)

        assert isinstance(gain_selector, ManualGainSelector)
        assert gain_selector.channel == chan
