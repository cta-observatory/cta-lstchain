import pytest
from ctapipe.core import run_tool


def test_create_irf(temp_dir_observed_files, simulated_dl2_file):
    """
    Generating an IRF file from a test DL2 files
    """
    from lstchain.tools.lstchain_create_irf_files import IRFFITSWriter
    import os
    import json

    irf_file = temp_dir_observed_files / "irf.fits.gz"
    sel_cuts_file = temp_dir_observed_files / "sel_cuts.json"

    if os.path.exists(sel_cuts_file):
        open(sel_cuts_file, "r")
    else:
        data = json.load(
            open(os.path.join('./lstchain/data/data_selection_cuts.json'))
        )
        data["DataSelection"]["fixed_gh_cut"] = 0.3
        data["DataSelection"]["intensity"] = [0, 10000]
        json.dump(data, open(sel_cuts_file, "x"), indent=3)

    assert (
        run_tool(
            IRFFITSWriter(),
            argv=[
                f"--config={sel_cuts_file}",
                f"--input_gamma_dl2={simulated_dl2_file}",
                f"--input_proton_dl2={simulated_dl2_file}",
                f"--input_electron_dl2={simulated_dl2_file}",
                f"--output_irf_file={irf_file}",
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
    sel_cuts_file = temp_dir_observed_files / "sel_cuts.json"

    assert (
        run_tool(
            DataReductionFITSWriter(),
            argv=[
                f"--config={sel_cuts_file}",
                f"--input_dl2={observed_dl2_file}",
                f"--output_dl3_path={temp_dir_observed_files}",
                f"--input_irf={irf_file}",
                "--source_name=Crab",
                "--source_ra=83.633deg",
                "--source_dec=22.01deg",
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
                f"--input_dl3_dir={temp_dir_observed_files}",
                "--file_pattern=dl3*fits.gz",
                "--overwrite",
            ],
            cwd=temp_dir_observed_files,
        )
        == 0
    )
