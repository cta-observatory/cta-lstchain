import shutil
import subprocess as sp

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
    test_drive_report,
    test_drs4_pedestal_path,
    test_calib_path,
    test_time_calib_path,
    test_r0_path,
    test_r0_path2
)


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
    """Test for all scripts if at least the help works."""
    run_program(script, "--help")


@pytest.fixture(scope="session")
def simulated_dl1ab(temp_dir_simulated_files, simulated_dl1_file):
    """Produce a new simulated dl1 file using the dl1ab script."""
    output_file = temp_dir_simulated_files / "dl1ab.h5"
    run_program(
        "lstchain_dl1ab",
        "-f",
        simulated_dl1_file,
        "-o",
        output_file
    )
    return output_file

def test_add_source_dependent_parameters(simulated_dl1_file):
    run_program('lstchain_add_source_dependent_parameters', '-f', simulated_dl1_file)
    dl1_params_src_dep = pd.read_hdf(simulated_dl1_file, key=dl1_params_src_dep_lstcam_key)
    dl1_params_src_dep.columns = pd.MultiIndex.from_tuples([tuple(col[1:-1].replace('\'', '').replace(' ', '').split(",")) for col in dl1_params_src_dep.columns])
    assert 'alpha' in dl1_params_src_dep['on'].columns


@pytest.fixture(scope="session")
def merged_simulated_dl1_file(simulated_dl1_file, temp_dir_simulated_files):
    """Produce a merged file from two identical dl1 hdf5 files."""
    shutil.copy(simulated_dl1_file, temp_dir_simulated_files / "dl1_copy.h5")
    merged_dl1_file = temp_dir_simulated_files / "script_merged_dl1.h5"
    run_program(
        "lstchain_merge_hdf5_files",
        "-d", temp_dir_simulated_files,
        "-o", merged_dl1_file,
        "--no-image", "True"
    )
    return merged_dl1_file


@pytest.mark.private_data
@pytest.fixture(scope="session")
def observed_dl1_files(temp_dir_observed_files):
    """
    Produce dl1, datacheck and muons files from real observed data.
    The initial timestamps and counters used for the first set of files
    here are extracted from the night summary. In this case these values
    correspond to the third event. A second set of files are produced
    without using the first valid timestamps.
    """
    # FIXME: naming criteria (suffixes, no stream) of dl1, dl2,
    #  muons and datacheck files should be coherent

    # First set of files to be produced
    dl1_output_path1 = temp_dir_observed_files / ("dl1_" + test_r0_path.with_suffix('').stem + ".h5")
    muons_file1 = temp_dir_observed_files / "muons_LST-1.Run02008.0000_first50.fits"
    datacheck_file1 = temp_dir_observed_files / "datacheck_dl1_LST-1.Run02008.0000.h5"

    # Second set of files
    dl1_output_path2 = temp_dir_observed_files / ("dl1_" + test_r0_path2.with_suffix('').stem + ".h5")
    muons_file2 = temp_dir_observed_files / "muons_LST-1.Run02008.0100_first50.fits"
    datacheck_file2 = temp_dir_observed_files / "datacheck_dl1_LST-1.Run02008.0100.h5"

    run_program(
        "lstchain_data_r0_to_dl1",
        "-f",
        test_r0_path,
        "-o",
        temp_dir_observed_files,
        "--pedestal-file",
        test_drs4_pedestal_path,
        "--calibration-file",
        test_calib_path,
        "--time-calibration-file",
        test_time_calib_path,
        "--pointing-file",
        test_drive_report,
        "--dragon-reference-time",
        "1582059789516351903",
        "--dragon-reference-counter",
        "2516351600",
    )

    run_program(
        "lstchain_data_r0_to_dl1",
        "-f",
        test_r0_path2,
        "-o",
        temp_dir_observed_files,
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
        'datacheck2': datacheck_file2
    }


def test_lstchain_mc_r0_to_dl1(simulated_dl1_file):
    assert simulated_dl1_file.is_file()


@pytest.mark.private_data
def test_lstchain_data_r0_to_dl1(observed_dl1_files):
    assert observed_dl1_files["dl1_file1"].is_file()
    assert observed_dl1_files["muons1"].is_file()
    assert observed_dl1_files["datacheck1"].is_file()
    assert observed_dl1_files["dl1_file2"].is_file()
    assert observed_dl1_files["muons2"].is_file()
    assert observed_dl1_files["datacheck2"].is_file()


@pytest.mark.private_data
def test_observed_dl1_validity(observed_dl1_files):
    dl1_df = pd.read_hdf(observed_dl1_files["dl1_file1"], key=dl1_params_lstcam_key)
    # The first valid timestamp in the test run corresponds
    # to its third event (see night summary)
    first_timestamp_nightsummary = 1582059789516351903  # ns
    first_event_timestamp = dl1_df["dragon_time"].iloc[2]  # third event

    dl1_tables = get_dataset_keys(observed_dl1_files["dl1_file1"])

    assert 'dl1/event/telescope/monitoring/calibration' in dl1_tables
    assert 'dl1/event/telescope/monitoring/flatfield' in dl1_tables
    assert 'dl1/event/telescope/monitoring/pedestal' in dl1_tables
    assert 'dl1/event/telescope/image/LST_LSTCam' in dl1_tables

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


def test_lstchain_mc_trainpipe(rf_models):
    assert rf_models["energy"].is_file()
    assert rf_models["disp"].is_file()
    assert rf_models["gh_sep"].is_file()


def test_lstchain_mc_rfperformance(tmp_path, simulated_dl1_file, fake_dl1_proton_file):
    gamma_file = simulated_dl1_file
    proton_file = fake_dl1_proton_file
    output_dir = tmp_path
    file_model_energy = output_dir / "reg_energy.sav"
    file_model_disp = output_dir / "reg_disp_vector.sav"
    file_model_gh_sep = output_dir / "cls_gh.sav"

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

    assert file_model_gh_sep.is_file()
    assert file_model_disp.is_file()
    assert file_model_energy.is_file()


def test_lstchain_merge_dl1_hdf5_files(merged_simulated_dl1_file):
    assert merged_simulated_dl1_file.is_file()


@pytest.mark.private_data
def test_lstchain_merge_dl1_hdf5_observed_files(temp_dir_observed_files, observed_dl1_files):
    merged_dl1_observed_file = temp_dir_observed_files / "dl1_LST-1.Run02008_merged.h5"
    run_program(
        "lstchain_merge_hdf5_files",
        "-d", temp_dir_observed_files,
        "-o", merged_dl1_observed_file,
        "--no-image", "False",
        "--smart", "False",
        "--run-number", "2008",
        "--pattern", "dl1_*.h5"
    )
    dl1a_df = pd.read_hdf(observed_dl1_files["dl1_file1"], key=dl1_params_lstcam_key)
    dl1b_df = pd.read_hdf(observed_dl1_files["dl1_file1"], key=dl1_params_lstcam_key)
    merged_dl1_df = pd.read_hdf(merged_dl1_observed_file, key=dl1_params_lstcam_key)
    assert merged_dl1_observed_file.is_file()
    assert len(dl1a_df) + len(dl1b_df) == len(merged_dl1_df)
    assert 'dl1/event/telescope/image/LST_LSTCam' in get_dataset_keys(merged_dl1_observed_file)
    assert 'dl1/event/telescope/parameters/LST_LSTCam' in get_dataset_keys(merged_dl1_observed_file)


@pytest.mark.private_data
def test_merge_datacheck_files(temp_dir_observed_files):
    run_program(
        "lstchain_check_dl1",
        "--batch",
        "--input-file", temp_dir_observed_files / "datacheck_dl1_LST-1.Run02008.*.h5",
        "--output-dir", temp_dir_observed_files
    )
    assert (temp_dir_observed_files / "datacheck_dl1_LST-1.Run02008.h5").is_file()
    assert (temp_dir_observed_files / "datacheck_dl1_LST-1.Run02008.pdf").is_file()


def test_lstchain_merged_dl1_to_dl2(temp_dir_simulated_files, merged_simulated_dl1_file, rf_models):
    output_file = merged_simulated_dl1_file.with_name(
        merged_simulated_dl1_file.name.replace('dl1', 'dl2')
    )
    run_program(
        "lstchain_dl1_to_dl2",
        "-f",
        merged_simulated_dl1_file,
        "-p",
        rf_models["path"],
        "--output-dir",
        temp_dir_simulated_files
    )
    assert output_file.is_file()


def test_lstchain_dl1_to_dl2(simulated_dl2_file):
    assert simulated_dl2_file.is_file()
    dl2_df = pd.read_hdf(simulated_dl2_file, key=dl2_params_lstcam_key)
    assert "gammaness" in dl2_df.columns
    assert "reco_type" in dl2_df.columns
    assert "reco_energy" in dl2_df.columns
    assert "reco_disp_dx" in dl2_df.columns
    assert "reco_disp_dy" in dl2_df.columns
    assert "reco_src_x" in dl2_df.columns
    assert "reco_src_y" in dl2_df.columns


@pytest.mark.private_data
def test_lstchain_observed_dl1_to_dl2(temp_dir_observed_files, observed_dl1_files,  rf_models):
    real_data_dl2_file = temp_dir_observed_files / ("dl2_" + test_r0_path.with_suffix('').stem + ".h5")
    run_program(
        "lstchain_dl1_to_dl2",
        "--input-file",
        observed_dl1_files["dl1_file1"],
        "--path-models",
        rf_models["path"],
        "--output-dir",
        temp_dir_observed_files
    )
    assert real_data_dl2_file.is_file()
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


def test_dl1ab(simulated_dl1ab):
    assert simulated_dl1ab.is_file()


@pytest.mark.private_data
def test_observed_dl1ab(tmp_path, observed_dl1_files):
    output_dl1ab = tmp_path / "dl1ab.h5"
    run_program("lstchain_dl1ab", "-f", observed_dl1_files["dl1_file1"], "-o", output_dl1ab)
    assert output_dl1ab.is_file()
    dl1ab = pd.read_hdf(output_dl1ab, key=dl1_params_lstcam_key)
    dl1 = pd.read_hdf(observed_dl1_files["dl1_file1"], key=dl1_params_lstcam_key)
    np.testing.assert_allclose(dl1, dl1ab, rtol=1e-4, equal_nan=True)


def test_simulated_dl1ab_validity(simulated_dl1_file, simulated_dl1ab):
    assert simulated_dl1ab.is_file()
    dl1_df = pd.read_hdf(simulated_dl1_file, key=dl1_params_lstcam_key)
    dl1ab_df = pd.read_hdf(simulated_dl1ab, key=dl1_params_lstcam_key)
    np.testing.assert_allclose(dl1_df, dl1ab_df, rtol=1e-4, equal_nan=True)


def test_mc_r0_to_dl2(tmp_path, rf_models, mc_gamma_testfile):
    dl2_file = tmp_path / "dl2_gamma_test_large.h5"
    run_program(
        "lstchain_mc_r0_to_dl2",
        "--input-file",
        mc_gamma_testfile,
        "--path-models",
        rf_models["path"],
        "--store-dl1",
        "False",
        "--output-dir",
        tmp_path
    )
    assert dl2_file.is_file()


def test_read_dl2_to_pyirf(simulated_dl2_file):
    from lstchain.io.io import read_dl2_to_pyirf

    events, sim_info = read_dl2_to_pyirf(simulated_dl2_file)
    assert "true_energy" in events.colnames
    assert sim_info.energy_max == 330 * u.TeV
