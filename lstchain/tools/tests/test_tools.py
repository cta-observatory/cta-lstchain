import pytest
from pathlib import Path

from ctapipe.core import run_tool
from lstchain.tools.lstchain_create_drs4_pedestal_file import PedestalFITSWriter
from lstchain.tests.test_lstchain import test_r0_path


@pytest.mark.private_data
def test_create_drs4_pedestal_file(temp_dir_observed_files):

    output_file = temp_dir_observed_files / "drs4_pedestal.fits"
    config = Path("./lstchain/data/onsite_camera_calibration_param.json").absolute()

    assert (
        run_tool(
            PedestalFITSWriter(),
            argv=[
                f"--config={config}",
                f"--input={test_r0_path}",
                f"--output={output_file}"
            ]
        )
        == 0
    )
