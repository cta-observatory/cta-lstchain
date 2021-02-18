import os
import shutil
import subprocess as sp
from pathlib import Path

import numpy as np
import pandas as pd
import pkg_resources
import pytest
from astropy import units as u
from astropy.time import Time

from lstchain.io.io import (
    dl1_params_lstcam_key,
    dl2_params_lstcam_key,
    get_dataset_keys,
    dl1_params_src_dep_lstcam_key
)
from lstchain.tests.test_lstchain import (
    test_dir,
    mc_gamma_testfile,
    produce_fake_dl1_proton_file,
    fake_dl1_proton_file,
    test_drive_report,
    test_drs4_pedestal_path,
    test_calib_path,
    test_time_calib_path,
    test_r0_path,
    test_r0_path2
)


output_dir = Path(test_dir, "scripts")
output_dir_realdata = output_dir / "real_data"
dl1_file = output_dir / "dl1_gamma_test_large.h5"
merged_dl1_file = output_dir / "script_merged_dl1.h5"
dl2_file = output_dir / "dl2_gamma_test_large.h5"
file_model_energy = output_dir / "reg_energy.sav"
file_model_disp = output_dir / "reg_disp_vector.sav"
file_model_gh_sep = output_dir / "cls_gh.sav"


def find_entry_points(package_name):
    """from: https://stackoverflow.com/a/47383763/3838691"""
    entrypoints = [
        ep.name
        for ep in pkg_resources.iter_entry_points("console_scripts")
        if ep.module_name.startswith(package_name)
    ]
    return entrypoints


ALL_SCRIPTS = find_entry_points("lstchain")


def run_program(*args):
    result = sp.run(args, stdout=sp.PIPE, stderr=sp.STDOUT, encoding="utf-8")

    if result.returncode != 0:
        raise ValueError(
            f"Running {args[0]} failed with return code {result.returncode}"
            f", output: \n {result.stdout}"
        )


@pytest.mark.parametrize("script", ALL_SCRIPTS)
def test_all_help(script):
    """Test for all scripts if at least the help works"""
    run_program(script, "--help")


def test_lstchain_mc_r0_to_dl1():
    input_file = mc_gamma_testfile
    run_program("lstchain_mc_r0_to_dl1", "-f", input_file, "-o", output_dir)
    assert os.path.exists(dl1_file)


@pytest.fixture
def observed_dl1_files(tmp_path):
    """
    Produce dl1, datacheck and muons files from real observed data.
    The initial timestamps and counters used here are extracted
    from the night summary file. In this case these values
    correspond to the third event.
    """
    # FIXME: naming criteria (suffixes, no stream) of dl1, dl2,
    #  muons and datacheck files should be coherent
    dl1_output_path1 = tmp_path / ("dl1_" + test_r0_path.with_suffix('').stem + ".h5")
    muons_file1 = tmp_path / "muons_LST-1.Run02008.0000_first50.fits"
    datacheck_file1 = tmp_path / "datacheck_dl1_LST-1.Run02008.0000.h5"
    dl1_output_path2 = tmp_path / ("dl1_" + test_r0_path2.with_suffix('').stem + ".h5")
    muons_file2 = tmp_path / "muons_LST-1.Run02008.0100_first50.fits"
    datacheck_file2 = tmp_path / "datacheck_dl1_LST-1.Run02008.0100.h5"

    run_program(
        "lstchain_data_r0_to_dl1",
        "-f",
        test_r0_path,
        "-o",
        tmp_path,
        "--pedestal-file",
        test_drs4_pedestal_path,
        "--calibration-file",
        test_calib_path,
        "--time-calibration-file",
        test_time_calib_path,
        "--pointing-file",
        test_drive_report,
        "--ucts-t0-dragon",
        "1582059789516351903",
        "--dragon-counter0",
        "2516351600",
        "--ucts-t0-tib",
        "1582059789516351903",
        "--tib-counter0",
        "2516351200",
    )

    run_program(
        "lstchain_data_r0_to_dl1",
        "-f",
        test_r0_path2,
        "-o",
        tmp_path,
        "--pedestal-file",
        test_drs4_pedestal_path,
        "--calibration-file",
        test_calib_path,
        "--time-calibration-file",
        test_time_calib_path,
        "--pointing-file",
        test_drive_report
    )

    return {
        'dl1_file1': dl1_output_path1,
        'muons1': muons_file1,
        'datacheck1': datacheck_file1,
        'dl1_file2': dl1_output_path2,
        'muons2': muons_file2,
        'datacheck2': datacheck_file2,
        'path': tmp_path
    }


@pytest.mark.private_data
def test_lstchain_data_r0_to_dl1(observed_dl1_files):
    assert observed_dl1_files["dl1_file1"].is_file()
    assert observed_dl1_files["muons1"].is_file()
    assert observed_dl1_files["datacheck1"].is_file()
    assert observed_dl1_files["dl1_file2"].is_file()
    assert observed_dl1_files["muons2"].is_file()
    assert observed_dl1_files["datacheck2"].is_file()


@pytest.mark.private_data
def test_dl1_realdata_validity(observed_dl1_files):
    dl1_df = pd.read_hdf(observed_dl1_files["dl1_file1"], key=dl1_params_lstcam_key)
    # The first valid timestamp in the test run corresponds
    # to its third event (see night summary)
    first_timestamp_nightsummary = 1582059789516351903  # ns
    first_event_timestamp = dl1_df["dragon_time"].iloc[2]  # third event

    assert 'dl1/event/telescope/monitoring/calibration' in get_dataset_keys(observed_dl1_files["dl1_file1"])
    assert 'dl1/event/telescope/monitoring/flatfield' in get_dataset_keys(observed_dl1_files["dl1_file1"])
    assert 'dl1/event/telescope/monitoring/pedestal' in get_dataset_keys(observed_dl1_files["dl1_file1"])
    assert 'dl1/event/telescope/image/LST_LSTCam' in get_dataset_keys(observed_dl1_files["dl1_file1"])

    assert "alt_tel" in dl1_df.columns
    assert "az_tel" in dl1_df.columns
    assert "trigger_type" in dl1_df.columns
    assert "ucts_trigger_type" in dl1_df.columns
    assert "trigger_time" in dl1_df.columns
    assert "dragon_time" in dl1_df.columns
    assert "tib_time" in dl1_df.columns
    assert "ucts_time" in dl1_df.columns
    assert np.isclose(
        (Time(first_event_timestamp, format='unix') -
         Time(first_timestamp_nightsummary / 1e9, format='unix_tai')
         ).to_value(u.s), 0)
    np.testing.assert_allclose(dl1_df["dragon_time"], dl1_df["trigger_time"])


@pytest.mark.run(after="test_lstchain_mc_r0_to_dl1")
def test_add_source_dependent_parameters():
    run_program("lstchain_add_source_dependent_parameters", "-f", dl1_file)
    dl1_params_src_dep = pd.read_hdf(dl1_file, key=dl1_params_src_dep_lstcam_key)
    assert "alpha" in dl1_params_src_dep.columns


@pytest.mark.run(after="test_lstchain_mc_r0_to_dl1")
def test_lstchain_mc_trainpipe():
    gamma_file = dl1_file
    proton_file = dl1_file

    run_program(
        "lstchain_mc_trainpipe", "--fg", gamma_file, "--fp", proton_file, "-o", output_dir
    )

    assert os.path.exists(file_model_gh_sep)
    assert os.path.exists(file_model_disp)
    assert os.path.exists(file_model_energy)


@pytest.mark.run(after="test_lstchain_mc_r0_to_dl1")
def test_lstchain_mc_rfperformance():
    gamma_file = dl1_file
    produce_fake_dl1_proton_file(dl1_file)
    proton_file = fake_dl1_proton_file

    run_program(
        "lstchain_mc_rfperformance",
        "--g-train",
        gamma_file,
        "--g-test",
        gamma_file,
        "--p-train",
        proton_file,
        "--p-test",
        proton_file,
        "-o",
        output_dir,
    )

    assert os.path.exists(file_model_gh_sep)
    assert os.path.exists(file_model_disp)
    assert os.path.exists(file_model_energy)


@pytest.mark.run(after="test_lstchain_mc_r0_to_dl1")
def test_lstchain_merge_dl1_hdf5_files():
    shutil.copy(dl1_file, output_dir / "dl1_copy.h5")
    run_program(
        "lstchain_merge_hdf5_files",
        "-d", output_dir,
        "-o", merged_dl1_file,
        "--no-image", "True",
    )
    assert os.path.exists(merged_dl1_file)


@pytest.mark.private_data
def test_lstchain_merge_dl1_hdf5_observed_files(tmp_path, observed_dl1_files):
    merged_dl1_observed_file = tmp_path / "dl1_LST-1.Run02008_merged.h5"
    run_program(
        "lstchain_merge_hdf5_files",
        "-d", observed_dl1_files["path"],
        "-o", merged_dl1_observed_file,
        "--no-image", "False",
        "--smart", "False",
        "--run-number", "2008",
        "--pattern", "dl1_*.h5"
    )
    dl1a_df = pd.read_hdf(observed_dl1_files["dl1_file1"], key=dl1_params_lstcam_key)
    dl1b_df = pd.read_hdf(observed_dl1_files["dl1_file1"], key=dl1_params_lstcam_key)
    merged_dl1_df = pd.read_hdf(merged_dl1_observed_file, key=dl1_params_lstcam_key)
    assert os.path.exists(merged_dl1_observed_file)
    assert len(dl1a_df) + len(dl1b_df) == len(merged_dl1_df)
    assert 'dl1/event/telescope/image/LST_LSTCam' in get_dataset_keys(merged_dl1_observed_file)
    assert 'dl1/event/telescope/parameters/LST_LSTCam' in get_dataset_keys(merged_dl1_observed_file)


@pytest.mark.private_data
def test_merge_datacheck_files(tmp_path, observed_dl1_files):
    run_program(
        "lstchain_check_dl1",
        "--input-file", observed_dl1_files["path"] / "datacheck_dl1_LST-1.Run02008.*.h5",
        "--output-dir", tmp_path
    )
    assert (tmp_path / "datacheck_dl1_LST-1.Run02008.h5").is_file()
    assert (tmp_path / "datacheck_dl1_LST-1.Run02008.pdf").is_file()


@pytest.mark.run(after="test_lstchain_merge_dl1_hdf5_files")
def test_lstchain_merged_dl1_to_dl2():
    output_file = merged_dl1_file.with_name(merged_dl1_file.name.replace('dl1', 'dl2'))
    run_program(
        "lstchain_dl1_to_dl2",
        "-f",
        merged_dl1_file,
        "-p",
        output_dir,
        "-o",
        output_dir,
    )
    assert os.path.exists(output_file)


@pytest.mark.run(after="test_lstchain_trainpipe")
def test_lstchain_dl1_to_dl2():
    run_program(
        "lstchain_dl1_to_dl2",
        "-f",
        dl1_file,
        "-p",
        output_dir,
        "-o",
        output_dir,
    )
    assert os.path.exists(dl2_file)


@pytest.mark.run(after="test_lstchain_mc_trainpipe")
@pytest.mark.private_data
def test_lstchain_realdata_dl1_to_dl2(tmp_path, observed_dl1_files):
    real_data_dl2_file = tmp_path / ("dl2_" + test_r0_path.with_suffix('').stem + ".h5")
    run_program(
        "lstchain_dl1_to_dl2",
        "--input-file",
        observed_dl1_files["dl1_file1"],
        "--path-models",
        output_dir,
        "--output-dir",
        tmp_path,
    )
    assert os.path.exists(real_data_dl2_file)
    dl2_df = pd.read_hdf(real_data_dl2_file, key=dl2_params_lstcam_key)
    assert "gammaness" in dl2_df.columns
    assert "reco_type" in dl2_df.columns
    assert "reco_energy" in dl2_df.columns
    assert "reco_alt" in dl2_df.columns
    assert "reco_az" in dl2_df.columns
    assert "reco_src_x" in dl2_df.columns
    assert "reco_src_y" in dl2_df.columns
    assert "reco_disp_dx" in dl2_df.columns
    assert "reco_disp_dy" in dl2_df.columns


@pytest.mark.run(after="test_lstchain_mc_r0_to_dl1")
def test_dl1ab():
    output_file = output_dir / "dl1ab.h5"
    run_program(
        "lstchain_dl1ab",
        "-f",
        dl1_file,
        "-o",
        output_file,
    )
    assert os.path.exists(output_file)


@pytest.mark.private_data
def test_dl1ab_realdata(tmp_path, observed_dl1_files):
    output_dl1ab = tmp_path / "dl1ab.h5"
    run_program("lstchain_dl1ab", "-f", observed_dl1_files["dl1_file1"], "-o", output_dl1ab)
    assert os.path.exists(output_dl1ab)
    dl1ab = pd.read_hdf(output_dl1ab, key=dl1_params_lstcam_key)
    dl1 = pd.read_hdf(observed_dl1_files["dl1_file1"], key=dl1_params_lstcam_key)
    np.testing.assert_allclose(dl1, dl1ab, rtol=1e-4, equal_nan=True)


@pytest.mark.run(after="test_dl1ab")
def test_dl1ab_validity():
    dl1 = pd.read_hdf(dl1_file, key=dl1_params_lstcam_key)
    dl1ab = pd.read_hdf(os.path.join(output_dir, "dl1ab.h5"), key=dl1_params_lstcam_key)
    np.testing.assert_allclose(dl1, dl1ab, rtol=1e-4, equal_nan=True)


@pytest.mark.run(after="test_lstchain_dl1_to_dl2")
def test_mc_r0_to_dl2():
    os.remove(dl1_file)
    os.remove(dl2_file)

    run_program(
        "lstchain_mc_r0_to_dl2",
        "-f",
        mc_gamma_testfile,
        "-p",
        output_dir,
        "-s1",
        "False",
        "-o",
        output_dir,
    )
    assert os.path.exists(dl2_file)


@pytest.mark.run(after="test_mc_r0_to_dl2")
def test_read_dl2_to_pyirf():
    from lstchain.io.io import read_dl2_to_pyirf
    import astropy.units as u

    events, sim_info = read_dl2_to_pyirf(dl2_file)
    assert "true_energy" in events.colnames
    assert sim_info.energy_max == 330 * u.TeV
