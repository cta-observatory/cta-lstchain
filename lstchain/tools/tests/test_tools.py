import pytest
from pathlib import Path

from ctapipe.core import run_tool
from lstchain.tools.lstchain_create_calibration_file import CalibrationHDF5Writer
from lstchain.tests.test_lstchain import test_r0_path, test_drs4_pedestal_path


@pytest.mark.private_data
def test_create_drs4_pedestal_file(temp_dir_observed_files):

    config = Path("./lstchain/data/onsite_camera_calibration_param.json").absolute()
    output = temp_dir_observed_files / "calibration.hdf5"

    assert (
        run_tool(
            CalibrationHDF5Writer(),
            argv=[
                f"--config={config}",
                f"--i={test_r0_path}",
                f"--o={output}",
                f"--drs4_pedestal_path={test_drs4_pedestal_path}",
            ]
        )
        == 0
    )
