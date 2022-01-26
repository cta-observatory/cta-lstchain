import pytest
from ctapipe.core import run_tool
import os
from astropy.io import fits


def test_create_irf_full_enclosure(temp_dir_observed_files, simulated_dl2_file):
    """
    Generating full enclosure IRF file from a test DL2 files
    """
    from lstchain.tools.lstchain_create_irf_files import IRFFITSWriter

    irf_file = temp_dir_observed_files / "irf.fits.gz"

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

def test_create_irf_point_like(temp_dir_observed_files, simulated_dl2_file):
    """
    Generating point-like IRF file from a test DL2 files
    """
    from lstchain.tools.lstchain_create_irf_files import IRFFITSWriter

    irf_file = temp_dir_observed_files / "irf.fits.gz"

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

    with fits.open(irf_file) as hdul:
        for hdu in hdul[1:]:
            assert 'RAD_MAX' in hdu.header
            assert isinstance(hdu.header['RAD_MAX'], float)


def test_create_irf_full_enclosure_with_config(temp_dir_observed_files, simulated_dl2_file):
    """
    Generating full enclosure IRF file from a test DL2 files, using
    a config file
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
                f"--config={config_file}",
                "--overwrite",
            ],
            cwd=temp_dir_observed_files,
        )
        == 0
    )


def test_create_irf_point_like_optimized_cuts(temp_dir_observed_files, simulated_dl2_file):
    """
    Generating point-like IRF file from a test DL2 files, using
    energy-dependent optimized cuts
    """
    from lstchain.tools.lstchain_create_irf_files import IRFFITSWriter

    irf_file = temp_dir_observed_files / "irf.fits.gz"

    assert (
        run_tool(
            IRFFITSWriter(),
            argv=[
                f"--input-gamma-dl2={simulated_dl2_file}",
                f"--input-proton-dl2={simulated_dl2_file}",
                f"--input-electron-dl2={simulated_dl2_file}",
                f"--output-irf-file={irf_file}",
                "--overwrite",
                "--optimize-gh",
                "--point-like",
                "--optimize-th"
            ],
            cwd=temp_dir_observed_files,
        )
        == 0
    )


@pytest.mark.private_data
def test_create_dl3_optimized_cuts(temp_dir_observed_files, observed_dl2_file):
    """
    Generating an DL3 file from a test DL2 files and test IRF file, using
    optimized cuts
    """
    from lstchain.tools.lstchain_create_dl3_file import DataReductionFITSWriter

    irf_file = temp_dir_observed_files / "irf.fits.gz"

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
                "--optimize-gh",
            ],
            cwd=temp_dir_observed_files,
        )
        == 0
    )


@pytest.mark.private_data
def test_create_dl3(temp_dir_observed_files, observed_dl2_file, simulated_irf_file):
    """
    Generating an DL3 file from a test DL2 files and test IRF file
    """
    from lstchain.tools.lstchain_create_dl3_file import DataReductionFITSWriter

    assert (
        run_tool(
            DataReductionFITSWriter(),
            argv=[
                f"--input-dl2={observed_dl2_file}",
                f"--output-dl3-path={temp_dir_observed_files}",
                f"--input-irf={simulated_irf_file}",
                "--source-name=Crab",
                "--source-ra=83.633deg",
                "--source-dec=22.01deg",
                "--overwrite",
            ],
            cwd=temp_dir_observed_files,
        )
        == 0
    )

@pytest.mark.private_data
def test_create_dl3_with_config(temp_dir_observed_files, observed_dl2_file):
    """
    Generating an DL3 file from a test DL2 files and test IRF file, using
    a config file
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
                f"--config={config_file}",
                "--overwrite",
            ],
            cwd=temp_dir_observed_files,
        )
        == 0
    )


@pytest.mark.private_data
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
