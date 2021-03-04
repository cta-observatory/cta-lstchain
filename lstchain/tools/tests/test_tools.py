import os
import pytest
from ctapipe.core import run_tool


@pytest.mark.parametrize("point_like_IRF", [True, False])
def test_create_irf(temp_dir_observed_files, simulated_dl2_file, point_like_IRF):
    """
    Generating an IRF file from a test DL2 files
    """
    from lstchain.tools.lstchain_create_irf_files import IRFFITSWriter
    import json

    irf_file = temp_dir_observed_files / "irf.fits.gz"
    cuts = temp_dir_observed_files / "cuts.json"

    if os.path.exists(cuts):
        open(cuts, "r")
    else:
        data = json.load(open(os.path.join("lstchain/data/data_selection_cuts.json")))
        data["fixed_cuts"]["gh_score"][0] = 0.3
        data["events_filters"]["intensity"][0] = 0
        json.dump(data, open(cuts, "x"))

    assert (
        run_tool(
            IRFFITSWriter(),
            argv=[
                f"--input_gamma_dl2={simulated_dl2_file}",
                f"--input_proton_dl2={simulated_dl2_file}",
                f"--input_electron_dl2={simulated_dl2_file}",
                f"--output_irf_file={irf_file}",
                f"--point_like={point_like_IRF}",
                f"--config={cuts}",
            ],
            cwd=temp_dir_observed_files,
        )
        == 0
    )


@pytest.mark.private_data
@pytest.mark.run(after="test_create_irf")
def test_create_dl3(temp_dir_observed_files):
    """
    Generating an DL3 file from a test DL2 files and test IRF file
    """
    from lstchain.tools.lstchain_create_dl3_file import DataReductionFITSWriter
    from lstchain.tests.test_lstchain import test_r0_path

    real_data_dl2_file = temp_dir_observed_files / (
        "dl2_" + test_r0_path.with_suffix('').stem + ".h5")
    irf_file = temp_dir_observed_files / "irf.fits.gz"
    cuts = temp_dir_observed_files / "cuts.json"

    assert (
        run_tool(
            DataReductionFITSWriter(),
            argv=[
                f"--input_dl2={real_data_dl2_file}",
                f"--output_dl3_path={temp_dir_observed_files}",
                f"--input_irf={irf_file}",
                f"--config={cuts}",
                "--source_name=Crab",
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
            ],
            cwd=temp_dir_observed_files,
        )
        == 0
    )
