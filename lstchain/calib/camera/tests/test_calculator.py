import os
from pathlib import Path
from traitlets.config import Config
from ctapipe.image.extractor import NeighborPeakWindowSum, FixedWindowSum

test_data = Path(os.getenv('LSTCHAIN_TEST_DATA', 'test_data'))

test_systematics_path = test_data / 'real/monitoring/PixelCalibration/LevelA/ffactor_systematics/20200725/v0.8.2.post2.dev48+gb1343281/ffactor_systematics_20200725.h5'




def test_calculator_config(lst1_subarray):
    from lstchain.calib.camera.calibration_calculator import CalibrationCalculator
    from lstchain.calib.camera.calibration_calculator import LSTCalibrationCalculator
    from lstchain.calib.camera.flatfield import FlasherFlatFieldCalculator
    from lstchain.calib.camera.pedestals import PedestalIntegrator

    # WARNING: this config contains nonsense value to test if the
    # nodefault values are taken over. DO NOT USE.
    config = Config({
        'LSTCalibrationCalculator': {
            'systematic_correction_path': test_systematics_path,
            'squared_excess_noise_factor': 10,
            'flatfield_product': 'FlasherFlatFieldCalculator',
            'FlasherFlatFieldCalculator': {
                'sample_size': 10,
                'sample_duration': 5,
                'tel_id': 7,
                'charge_median_cut_outliers': [-0.2, 0.2],
                'charge_std_cut_outliers': [-5, 5],
                'time_cut_outliers': [5, 35],
                'charge_product': 'NeighborPeakWindowSum',
                'NeighborPeakWindowSum': {
                    'window_shift': 7,
                    'window_width': 10,
                    'apply_integration_correction': False
                },

            },
            'pedestal_product': 'PedestalIntegrator',
            'PedestalIntegrator': {
                'sample_size': 2000,
                'sample_duration': 100,
                'tel_id': 7,
                'charge_median_cut_outliers': [-5, 5],
                'charge_std_cut_outliers': [-7, 7],
                'charge_product': 'FixedWindowSum',
                'FixedWindowSum': {
                    'window_shift': 10,
                    'window_width': 20,
                    'peak_index': 15,
                    'apply_integration_correction': False
                },
            },
        },
    })

    calibration_calculator = CalibrationCalculator.from_name(
        "LSTCalibrationCalculator",
        config=config,
        subarray=lst1_subarray,
    )

    assert isinstance(calibration_calculator, LSTCalibrationCalculator)
    assert calibration_calculator.systematic_correction_path.resolve().absolute() == test_systematics_path.resolve().absolute()
    assert calibration_calculator.squared_excess_noise_factor == 10

    ff = calibration_calculator.flatfield
    assert isinstance(ff, FlasherFlatFieldCalculator)
    assert isinstance(ff.extractor, NeighborPeakWindowSum)
    assert ff.extractor.window_shift.tel[1] == 7
    assert ff.extractor.window_width.tel[1] == 10
    assert ff.extractor.apply_integration_correction.tel[1] is False

    ped = calibration_calculator.pedestal
    assert isinstance(ped, PedestalIntegrator)
    assert isinstance(ped.extractor, FixedWindowSum)
    assert ped.extractor.window_shift.tel[1] == 10
    assert ped.extractor.window_width.tel[1] == 20
    assert ped.extractor.peak_index.tel[1] == 15
    assert ped.extractor.apply_integration_correction.tel[1] is False
