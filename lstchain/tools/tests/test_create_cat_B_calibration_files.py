import numpy as np

from ctapipe.core import run_tool
from ctapipe_io_lst.constants import N_GAINS, N_PIXELS
from ctapipe.io import read_table
from lstchain.onsite import DEFAULT_CONFIG_CAT_B_CALIB 

from lstchain.tests.test_lstchain import (
    test_systematics_path,
    test_calib_path
)

def test_create_catB_calibration_file(tmp_path,interleaved_r1_file):
    '''Test the lstchain_create_cat_B_calibration_file tool'''
    from lstchain.tools.lstchain_create_cat_B_calibration_file import CatBCalibrationHDF5Writer

    input_path = interleaved_r1_file.parent
    output_path = tmp_path / "calibration_cat_B_02006.h5"
    stat_events = 90
    
    ret =  run_tool(
        CatBCalibrationHDF5Writer(),
        argv=[
            f"--input_path={input_path}",
            f"--output_file={output_path}",
            f"--cat_A_calibration_file={test_calib_path}",
            f"--LSTCalibrationCalculator.systematic_correction_path={test_systematics_path}",
            f"--FlasherFlatFieldCalculator.sample_size={stat_events}",
            f"--PedestalIntegrator.sample_size={stat_events}",
            f"--config={DEFAULT_CONFIG_CAT_B_CALIB}",
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

    assert np.sum(unusable_pixels) == 4
    assert np.isclose(np.median(n_pe[~unusable_pixels]), 86.34, rtol=0.1)
    assert np.isclose(np.median(dc_to_pe[~unusable_pixels], axis=0), 1.07, rtol=0.01)

