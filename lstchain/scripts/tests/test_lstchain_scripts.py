import os
import shutil
import subprocess as sp
from pathlib import Path

import numpy as np
import pandas as pd
import pkg_resources
import pytest

from lstchain.io.io import dl1_params_lstcam_key, dl2_params_lstcam_key
from lstchain.io.io import dl1_params_src_dep_lstcam_key
from lstchain.tests.test_lstchain import (
    test_dir,
    mc_gamma_testfile,
    produce_fake_dl1_proton_file,
    fake_dl1_proton_file,
    test_drive_report,
    test_drs4_pedestal_path,
    test_calib_path,
    test_time_calib_path,
    test_r0_path
)

output_dir = Path(test_dir, "scripts")
dl1_file = output_dir / "dl1_gamma_test_large.h5"
merged_dl1_file = output_dir / "script_merged_dl1.h5"
dl2_file = output_dir / "dl2_gamma_test_large.h5"
file_model_energy = output_dir / "reg_energy.sav"
file_model_disp = output_dir / "reg_disp_vector.sav"
file_model_gh_sep = output_dir / "cls_gh.sav"
# Real data files to be produced in the tests
real_data_dl1_file = output_dir / ('dl1_' + test_r0_path.with_suffix('').stem + '.h5')
real_data_dl2_file = output_dir / ('dl2_' + test_r0_path.with_suffix('').stem + '.h5')
# FIXME: naming criteria of dl1, dl2, muons and datacheck files should be coherent
# muons_file = output_dir / ('muons_' + test_r0_path.stem)  # This does not work since the stream is stripped out
# datacheck_file = output_dir / ('datacheck_dl1_' + test_r0_path.with_suffix('').stem + '.h5')
muons_file = output_dir / 'muons_LST-1.Run02008.0000_first50.fits'
datacheck_file = output_dir / 'datacheck_dl1_LST-1.Run02008.0000.h5'


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


@pytest.mark.private_data
def test_lstchain_data_r0_to_dl1():
    run_program(
        "lstchain_data_r0_to_dl1",
        "-f",
        test_r0_path,
        "-o",
        output_dir,
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
    assert real_data_dl1_file.is_file()
    assert muons_file.is_file()
    assert datacheck_file.is_file()


@pytest.mark.run(after="test_lstchain_data_r0_to_dl1")
@pytest.mark.private_data
def test_dl1_realdata_validity():
    dl1_df = pd.read_hdf(real_data_dl1_file, key=dl1_params_lstcam_key)
    assert "alt_tel" in dl1_df.columns
    assert "az_tel" in dl1_df.columns
    assert "trigger_type" in dl1_df.columns
    assert "ucts_trigger_type" in dl1_df.columns
    assert "trigger_time" in dl1_df.columns
    assert "dragon_time" in dl1_df.columns
    assert "tib_time" in dl1_df.columns
    assert "ucts_time" in dl1_df.columns


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
    shutil.copy(dl1_file, os.path.join(output_dir, "dl1_copy.h5"))
    run_program(
        "lstchain_merge_hdf5_files",
        "-d",
        output_dir,
        "-o",
        merged_dl1_file,
        "--no-image",
        "True",
    )
    # FIXME: test it with smart False option?
    assert os.path.exists(merged_dl1_file)


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


@pytest.mark.run(after="test_lstchain_data_r0_to_dl1")
@pytest.mark.run(after="test_lstchain_mc_trainpipe")
@pytest.mark.private_data
def test_lstchain_realdata_dl1_to_dl2():
    run_program(
        "lstchain_dl1_to_dl2",
        "--input-file",
        real_data_dl1_file,
        "--path-models",
        output_dir,
        "--output-dir",
        output_dir,
    )
    assert os.path.exists(real_data_dl2_file)


@pytest.mark.run(after="test_lstchain_realdata_dl1_to_dl2")
@pytest.mark.private_data
def test_dl2_realdata_validity():
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


@pytest.mark.run(after="test_lstchain_data_r0_to_dl1")
def test_dl1ab_realdata():
    output_real_data_dl1ab = output_dir / "dl1ab_realdata.h5"
    run_program("lstchain_dl1ab", "-f", real_data_dl1_file, "-o", output_real_data_dl1ab)
    assert os.path.exists(output_real_data_dl1ab)


@pytest.mark.run(after="test_dl1ab")
def test_dl1ab_validity():
    dl1 = pd.read_hdf(dl1_file, key=dl1_params_lstcam_key)
    dl1ab = pd.read_hdf(os.path.join(output_dir, "dl1ab.h5"), key=dl1_params_lstcam_key)
    np.testing.assert_allclose(dl1, dl1ab, rtol=1e-4, equal_nan=True)


@pytest.mark.run(after="test_dl1ab_realdata")
def test_dl1ab_realdata_validity():
    output_real_data_dl1ab = output_dir / "dl1ab_realdata.h5"
    dl1 = pd.read_hdf(real_data_dl1_file, key=dl1_params_lstcam_key)
    dl1ab = pd.read_hdf(output_real_data_dl1ab, key=dl1_params_lstcam_key)
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
