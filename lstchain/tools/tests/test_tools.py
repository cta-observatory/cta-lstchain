import pytest
from ctapipe.core import run_tool

@pytest.mark.parametrize("point_like_IRF", [True, False])
def test_create_irf(temp_dir_observed_files, simulated_dl2_file, point_like_IRF):
    """
    Generating an IRF file from a test DL2 files
    """
    from lstchain.tools.lstchain_create_irf_files import IRFFITSWriter
    import os
    import json

    irf_file = temp_dir_observed_files / "irf.fits.gz"
    sel_cuts_file = temp_dir_observed_files / "sel_cuts.json"

    if os.path.exists(sel_cuts_file):
        open(sel_cuts_file, 'r')
    else:
        data = json.load(
            open(os.path.join('./lstchain/data/data_selection_cuts.json'))
        )
        data["DataSelection"]["fixed_gh_cut"] = 0.3
        data["DataSelection"]["intensity"] = [0, 10000]
        json.dump(data, open(sel_cuts_file, 'x'), indent=3)

    assert (
        run_tool(
            IRFFITSWriter(),
            argv=[
                f"--config={sel_cuts_file}",
                f"--input_gamma_dl2={simulated_dl2_file}",
                f"--input_proton_dl2={simulated_dl2_file}",
                f"--input_electron_dl2={simulated_dl2_file}",
                f"--output_irf_file={irf_file}",
                f"--point_like={point_like_IRF}"
            ],
            cwd=temp_dir_observed_files,
        )
        == 0
    )


@pytest.mark.private_data
@pytest.mark.run(after="test_create_irf")
def test_create_dl3(temp_dir_observed_files, simulated_dl2_file):
    """
    Generating an DL3 file from a test DL2 files and test IRF file
    """
    from lstchain.tools.lstchain_create_dl3_file import DataReductionFITSWriter
    from lstchain.reco.utils import add_delta_t_key
    import numpy as np
    # Temporary use of simulated data
    dl2_file_new = temp_dir_observed_files / "dl2_LST-1.0001.h5"
    )

    dl2 = pd.read_hdf(simulated_dl2_file, key=dl2_params_lstcam_key)

    # Adding some necessary columns for reading it as real data file
    # Simulated data file is being used as this is run before the test_lstchain_scripts
    dl2["tel_id"] = dl2["tel_id"].min()
    dl2["dragon_time"] = dl2["tel_id"] + np.arange(0, len(dl2["tel_id"]) * 1e-3, 1e-3)
    dl2 = add_delta_t_key(dl2)
    dl2["alt_tel"] = dl2["mc_alt_tel"]
    dl2["az_tel"] = dl2["mc_az_tel"]
    dl2.to_hdf(dl2_file_new, key=dl2_params_lstcam_key)

    """real_data_dl2_file = temp_dir_observed_files / (
        observed_dl1_files["dl1_file1"].name.replace("dl1", "dl2")
    )"""
    irf_file = temp_dir_observed_files / "irf.fits.gz"
    sel_cuts_file = temp_dir_observed_files / "sel_cuts.json"

    assert (
        run_tool(
            DataReductionFITSWriter(),
            argv=[
                f"--config={sel_cuts_file}",
                f"--input_dl2={dl2_file_new}",
                f"--output_dl3_path={temp_dir_observed_files}",
                f"--input_irf={irf_file}",
                "--source_name=Crab"
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
