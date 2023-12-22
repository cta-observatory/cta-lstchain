import numpy as np

from ctapipe.core import run_tool
from ctapipe_io_lst.constants import N_GAINS, N_PIXELS
from ctapipe.io import read_table
from lstchain.onsite import DEFAULT_CONFIG

from lstchain.tests.test_lstchain import (
    test_data,
    test_drs4_pedestal_path,
    test_time_calib_path,
    test_run_summary_path,
    test_systematics_path
)


def test_create_calibration_file(tmp_path):
    '''Test the lstchain_create_calibration_file tool'''
    from lstchain.tools.lstchain_create_calibration_file import CalibrationHDF5Writer

    input_path = test_data / "real/R0/20200218/LST-1.1.Run02006.0000_first50.fits.fz"
    output_path = tmp_path / "calibration_02006.h5"
    stat_events = 90
    
    ret =  run_tool(
        CalibrationHDF5Writer(),
        argv=[
            f"--input_file={input_path}",
            f"--output_file={output_path}",
            f"--LSTCalibrationCalculator.systematic_correction_path={test_systematics_path}",
            f"--LSTEventSource.EventTimeCalculator.run_summary_path={test_run_summary_path}",
            f"--LSTEventSource.LSTR0Corrections.drs4_time_calibration_path={test_time_calib_path}",
            f"--LSTEventSource.LSTR0Corrections.drs4_pedestal_path={test_drs4_pedestal_path}",
            "--LSTEventSource.default_trigger_type=tib",
            f"--FlasherFlatFieldCalculator.sample_size={stat_events}",
            f"--PedestalIntegrator.sample_size={stat_events}",
            "--events_to_skip=0",
            f"--config={DEFAULT_CONFIG}",
            "--overwrite",
        ],
        cwd=tmp_path,
        )
    
    assert ret == 0, 'Running CalibrationHDF5Writer tool failed'
    assert output_path.is_file(), 'Output file not written'

    cal_data = read_table(output_path, '/tel_1/calibration')[0]

    n_pe = cal_data['n_pe']
    unusable_pixels = cal_data['unusable_pixels']
    dc_to_pe = cal_data["dc_to_pe"]
    
    assert n_pe.shape == (N_GAINS, N_PIXELS)
    assert np.sum(unusable_pixels) == 16
    assert np.isclose(np.median(n_pe[~unusable_pixels]), 86.45, rtol=0.1)
    assert np.isclose(np.median(dc_to_pe[~unusable_pixels], axis=0), 0.0135, rtol=0.01)

