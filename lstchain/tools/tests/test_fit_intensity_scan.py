import os
from pathlib import Path
import math
import tables

from ctapipe.core import run_tool
from lstchain.tools.lstchain_fit_intensity_scan import FitIntensityScan
from lstchain.io.io import get_resource_path


test_data = Path(os.getenv('LSTCHAIN_TEST_DATA', 'test_data')).absolute()


def test_fit_intensity_scan(tmp_path):
    '''Test the lstchain_fit_intensity_scan tool'''
    
    input_dir = test_data / "real/monitoring/PixelCalibration/Cat-A/calibration/20221001/ctapipe-v0.17/"

    config_file = get_resource_path("data/lstchain_fit_intensity_scan_config_example.json")
 
    ret = run_tool(
        FitIntensityScan(),
        argv=[
            f"--config={config_file}",
            f"--input_dir={input_dir}"
        ],
        cwd=tmp_path,
    )
    
    assert ret == 0, 'Running tool FitIntensityScan failed'

    # test fit output
    fit_data = tables.open_file(f"{tmp_path}/filter_scan_fit.h5")
    gain = fit_data.root.gain
    pixel = 0
    assert math.isclose(gain[0,pixel], 75.751, abs_tol = 0.001) and math.isclose(gain[1,pixel], 4.216, abs_tol = 0.001)
