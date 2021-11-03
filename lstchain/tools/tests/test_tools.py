import os
import pytest

from ctapipe.core import run_tool


@pytest.mark.private_data
def test_create_drs4_pedestal_file(temp_dir_observed_files):
    """
    Create a drs4 pedestal fits file
    """
    from lstchain.tools.lstchain_create_drs4_pedestal_file import PedestalFITSWriter
    from lstchain.tests.test_lstchain import test_r0_path

    output_file = temp_dir_observed_files / "drs4_pedestal.fits"

    assert (
        run_tool(
            PedestalFITSWriter(),
            argv=[
                f"--input={test_r0_path}",
                f"--output={output_file}"
            ]
        )
        == 0
    )


def test_create_irf(temp_dir_observed_files, simulated_dl2_file):
    """
    Generating an IRF file from a test DL2 files
    """
    from lstchain.tools.lstchain_create_irf_files import IRFFITSWriter

    irf_file = temp_dir_observed_files / "irf.fits.gz"
    config_file = os.path.join(os.getcwd(), "./docs/examples/irf_tool_config.json")

    assert (
        run_tool(
            IRFFITSWriter(),
            argv=[
                f"--input-gamma-dl2={simulated_dl2_file}",
                f"--input-proton-dl2={simulated_dl2_file}",
                f"--input-electron-dl2={simulated_dl2_file}",
                f"--output-irf-file={irf_file}",
                "--overwrite",
            ],
            cwd=temp_dir_observed_files,
        )
        == 0
    )

    assert (
        run_tool(
            IRFFITSWriter(),
            argv=[
                f"--input-gamma-dl2={simulated_dl2_file}",
                f"--input-proton-dl2={simulated_dl2_file}",
                f"--input-electron-dl2={simulated_dl2_file}",
                f"--output-irf-file={irf_file}",
                "--point-like",
                "--overwrite",
            ],
            cwd=temp_dir_observed_files,
        )
        == 0
    )
    assert (
        run_tool(
            IRFFITSWriter(),
            argv=[
                f"--input-gamma-dl2={simulated_dl2_file}",
                f"--input-proton-dl2={simulated_dl2_file}",
                f"--input-electron-dl2={simulated_dl2_file}",
                f"--output-irf-file={irf_file}",
                f"--config={config_file}",
                "--overwrite",
            ],
            cwd=temp_dir_observed_files,
        )
        == 0
    )


@pytest.mark.private_data
@pytest.mark.run(after="test_create_irf")
def test_create_dl3(temp_dir_observed_files, observed_dl2_file):
    """
    Generating an DL3 file from a test DL2 files and test IRF file
    """
    from lstchain.tools.lstchain_create_dl3_file import DataReductionFITSWriter

    irf_file = temp_dir_observed_files / "irf.fits.gz"
    config_file = os.path.join(os.getcwd(), "docs/examples/dl3_tool_config.json")

    assert (
        run_tool(
            DataReductionFITSWriter(),
            argv=[
                f"--input-dl2={observed_dl2_file}",
                f"--output-dl3-path={temp_dir_observed_files}",
                f"--input-irf={irf_file}",
                "--source-name=Crab",
                "--source-ra=83.633deg",
                "--source-dec=22.01deg",
                "--overwrite",
            ],
            cwd=temp_dir_observed_files,
        )
        == 0
    )

    assert (
        run_tool(
            DataReductionFITSWriter(),
            argv=[
                f"--input-dl2={observed_dl2_file}",
                f"--output-dl3-path={temp_dir_observed_files}",
                f"--input-irf={irf_file}",
                "--source-name=Crab",
                "--source-ra=83.633deg",
                "--source-dec=22.01deg",
                f"--config={config_file}",
                "--overwrite",
            ],
            cwd=temp_dir_observed_files,
        )
        == 0
    )


@pytest.mark.private_data
@pytest.mark.run(after="test_create_dl3")
def test_index_dl3_files(temp_dir_observed_files):
    """
    Generating Index files from a given path and glob pattern for DL3 files
    """
    from lstchain.tools.lstchain_create_dl3_index_files import FITSIndexWriter

    assert (
        run_tool(
            FITSIndexWriter(),
            argv=[
                f"--input-dl3-dir={temp_dir_observed_files}",
                "--overwrite",
            ],
            cwd=temp_dir_observed_files,
        )
        == 0
    )
