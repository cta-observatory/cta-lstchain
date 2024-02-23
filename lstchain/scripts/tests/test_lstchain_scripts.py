import json
import os
import shutil
import subprocess as sp
from pathlib import Path

import numpy as np
import pandas as pd
import pkg_resources
import pytest
import tables
from importlib.resources import files
from astropy import units as u
from astropy.table import Table
from astropy.time import Time
from ctapipe.instrument import SubarrayDescription
from ctapipe.io import read_table
from ctapipe.io import EventSource
from ctapipe.containers import EventType


from lstchain.io.config import get_srcdep_config, get_standard_config
from lstchain.io.io import (dl1_images_lstcam_key, dl1_params_lstcam_key,
                            dl1_params_tel_mon_cal_key,
                            dl1_params_tel_mon_flat_key,
                            dl1_params_tel_mon_ped_key, dl2_params_lstcam_key,
                            get_dataset_keys, get_srcdep_params)


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
    result = sp.run(args, stdout=sp.PIPE, stderr=sp.STDOUT, encoding='utf-8')

    if result.returncode != 0:
        raise ValueError(
            f"Running {args[0]} failed with return code {result.returncode}"
            f", output: \n {result.stdout}"
        )
    else:
        return result


@pytest.mark.parametrize("script", ALL_SCRIPTS)
def test_all_help(script):
    """Test for all scripts if at least the help works."""
    run_program(script, "--help")


@pytest.fixture(scope="session")
def simulated_dl1ab(temp_dir_simulated_files, simulated_dl1_file):
    """Produce a new simulated dl1 file using the dl1ab script."""
    output_file = temp_dir_simulated_files / "dl1ab.h5"
    run_program("lstchain_dl1ab", "-f", simulated_dl1_file, "-o", output_file)
    return output_file


@pytest.fixture(scope="session")
def merged_simulated_dl1_file(simulated_dl1_file, temp_dir_simulated_files):
    """Produce a merged file from two identical dl1 hdf5 files."""
    shutil.copy(simulated_dl1_file, temp_dir_simulated_files / "dl1_copy.h5")
    merged_dl1_file = temp_dir_simulated_files / "script_merged_dl1.h5"
    run_program(
        "lstchain_merge_hdf5_files",
        "-d",
        temp_dir_simulated_files,
        "-o",
        merged_dl1_file,
        "--no-image",
        "--pattern=dl1_*.h5"
    )
    return merged_dl1_file


def test_lstchain_mc_r0_to_dl1(simulated_dl1_file):
    assert simulated_dl1_file.is_file()

@pytest.mark.private_data
def test_lstchain_r0_to_r0g(tmp_path, temp_dir_observed_files):
    test_data = Path(os.getenv('LSTCHAIN_TEST_DATA', 'test_data'))
    input_file = test_data / "real/R0/20231214/LST-1.1.Run16102.0000_first50" \
                             ".fits.fz"
    output_dir = temp_dir_observed_files / "R0G"
    output_dir.mkdir()
    run_program("lstchain_r0_to_r0g", "-f", input_file, "-o", output_dir)
    output_file = output_dir / input_file.name
    assert output_file.is_file()

    src = EventSource(input_url=output_file)
    src.pointing_information = False
    src.trigger_information = False
    src.apply_drs4_corrections = False
    # Check number of gains for first event of each type:
    for evtype, ngains in zip([EventType.FLATFIELD, EventType.SKY_PEDESTAL, 
                               EventType.SUBARRAY], [2, 2, 1]):
        for event in src:
            if event.trigger.event_type == evtype:
                break
        assert(event.r0.tel[1].waveform.shape[0] == ngains)  

@pytest.mark.private_data
def test_lstchain_r0g_to_r0v(tmp_path, temp_dir_observed_files):
    test_data = Path(os.getenv('LSTCHAIN_TEST_DATA', 'test_data'))
    input_file = temp_dir_observed_files / "R0G/LST-1.1.Run16102.0000_first50" \
                                           ".fits.fz"
    pixel_selection_file = test_data / "real/R0DVR/Pixel_selection_LST-1.Run16102.0000.h5"
    output_dir = temp_dir_observed_files / "R0V"
    output_dir.mkdir()
    run_program("lstchain_r0g_to_r0v", "-f", input_file, "-o", output_dir,
                "--pixselection-file", pixel_selection_file)
    output_file = output_dir / input_file.name
    assert output_file.is_file()

@pytest.mark.private_data
def test_lstchain_data_r0_to_dl1(observed_dl1_files):
    assert observed_dl1_files["dl1_file1"].is_file()
    assert observed_dl1_files["muons1"].is_file()
    assert observed_dl1_files["datacheck1"].is_file()
    assert observed_dl1_files["interleaved_file1"].is_file()
    assert observed_dl1_files["dl1_file2"].is_file()
    assert observed_dl1_files["muons2"].is_file()
    assert observed_dl1_files["datacheck2"].is_file()
    assert observed_dl1_files["interleaved_file2"].is_file()


@pytest.mark.private_data
def test_observed_dl1_validity(observed_dl1_files):
    dl1_df = pd.read_hdf(observed_dl1_files["dl1_file1"], key=dl1_params_lstcam_key)
    # The first valid timestamp in the test run corresponds
    # to its third event (see night summary)
    first_timestamp_nightsummary = 1582059789516351903  # ns
    first_event_timestamp = dl1_df["dragon_time"].iloc[2]  # third event

    dl1_tables = get_dataset_keys(observed_dl1_files["dl1_file1"])

    assert dl1_params_lstcam_key in dl1_tables
    assert dl1_images_lstcam_key in dl1_tables
    assert dl1_params_tel_mon_cal_key in dl1_tables
    assert dl1_params_tel_mon_ped_key in dl1_tables
    assert dl1_params_tel_mon_flat_key in dl1_tables

    subarray = SubarrayDescription.from_hdf(observed_dl1_files['dl1_file1'])
    assert 1 in subarray.tel
    assert subarray.tel[1].name == "LST"

    assert "alt_tel" in dl1_df.columns
    assert "az_tel" in dl1_df.columns
    assert "trigger_type" in dl1_df.columns
    assert "ucts_trigger_type" in dl1_df.columns
    assert "trigger_time" in dl1_df.columns
    assert "dragon_time" in dl1_df.columns
    assert "tib_time" in dl1_df.columns
    assert "ucts_time" in dl1_df.columns
    assert np.isclose(
        (
            Time(first_event_timestamp, format="unix")
            - Time(first_timestamp_nightsummary / 1e9, format="unix_tai")
        ).to_value(u.s),
        0,
    )
    np.testing.assert_allclose(dl1_df["dragon_time"], dl1_df["trigger_time"])

@pytest.mark.private_data
def test_dvr_file_validity(observed_dl1_files):
    dvr_file = pd.read_hdf(observed_dl1_files["dvr_file1"], key="run_summary")
    assert "mean_pixel_survival_fraction" in dvr_file.columns
    assert dvr_file['number_of_events'].iloc[0] == 200
    assert dvr_file['mean_pixel_survival_fraction'].iloc[0] < 0.1

@pytest.mark.private_data
def test_pixmasks_file_validity(observed_dl1_files):
    pixmasks_file = tables.open_file(observed_dl1_files["pixmasks_file1"])
    pixmasks = pixmasks_file.root.selected_pixels_masks.col('pixmask')
    assert pixmasks.sum() < 0.1 * len(pixmasks.flatten())

@pytest.mark.private_data
@pytest.fixture(scope="session")
def tune_nsb(mc_gamma_testfile, observed_dl1_files):
    return run_program(
        "lstchain_tune_nsb",
        "--config",
        "lstchain/data/lstchain_standard_config.json",
        "--input-mc",
        mc_gamma_testfile,
        "--input-data",
        observed_dl1_files["dl1_file1"],
    )


def test_validity_tune_nsb(tune_nsb):
    output_lines = tune_nsb.stdout.splitlines()
    for line in output_lines:
        if "increase_nsb" in line:
            assert line == '  "increase_nsb": true,'
        if "extra_noise_in_dim_pixels" in line:
            assert line == '  "extra_noise_in_dim_pixels": 1.962,'
        if "extra_bias_in_dim_pixels" in line:
            assert line == '  "extra_bias_in_dim_pixels": 0,'
        if "transition_charge" in line:
            assert line == '  "transition_charge": 8,'
        if "extra_noise_in_bright_pixels" in line:
            assert line == '  "extra_noise_in_bright_pixels": 0.0'


def test_lstchain_mc_trainpipe(rf_models):
    assert rf_models["energy"].is_file()
    assert rf_models["disp_norm"].is_file()
    assert rf_models["disp_sign"].is_file()
    assert rf_models["gh_sep"].is_file()


def test_lstchain_mc_trainpipe_srcdep(rf_models_srcdep):
    assert rf_models_srcdep["energy"].is_file()
    assert rf_models_srcdep["disp_norm"].is_file()
    assert rf_models_srcdep["disp_sign"].is_file()
    assert rf_models_srcdep["gh_sep"].is_file()


def test_lstchain_mc_rfperformance(tmp_path, simulated_dl1_file, fake_dl1_proton_file):
    gamma_file = simulated_dl1_file
    proton_file = fake_dl1_proton_file
    output_dir = tmp_path
    file_model_energy = output_dir / "reg_energy.sav"
    file_model_disp_norm = output_dir / "reg_disp_norm.sav"
    file_model_disp_sign = output_dir / "cls_disp_sign.sav"
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
    # assert file_model_disp.is_file()
    assert file_model_disp_norm.is_file()
    assert file_model_disp_sign.is_file()
    assert file_model_energy.is_file()


def test_lstchain_merge_dl1_hdf5_files(merged_simulated_dl1_file):
    assert merged_simulated_dl1_file.is_file()
    hdf5_file = tables.open_file(merged_simulated_dl1_file)
    assert len(hdf5_file.root.source_filenames.filenames) == 2


@pytest.mark.private_data
def test_lstchain_merge_dl1_hdf5_observed_files(
    temp_dir_observed_files, observed_dl1_files
):
    merged_dl1_observed_file = temp_dir_observed_files / "dl1_LST-1.Run02008_merged.h5"
    run_program(
        "lstchain_merge_hdf5_files",
        "-d",
        temp_dir_observed_files,
        "-o",
        merged_dl1_observed_file,
        "--run-number",
        "2008",
        "--pattern",
        "dl1_*.h5",
    )
    dl1a_df = pd.read_hdf(observed_dl1_files["dl1_file1"], key=dl1_params_lstcam_key)
    dl1b_df = pd.read_hdf(observed_dl1_files["dl1_file1"], key=dl1_params_lstcam_key)
    merged_dl1_df = pd.read_hdf(merged_dl1_observed_file, key=dl1_params_lstcam_key)
    assert merged_dl1_observed_file.is_file()
    assert len(dl1a_df) + len(dl1b_df) == len(merged_dl1_df)
    assert dl1_images_lstcam_key in get_dataset_keys(
        merged_dl1_observed_file
    )
    assert dl1_params_lstcam_key in get_dataset_keys(
        merged_dl1_observed_file
    )


@pytest.mark.private_data
def test_merge_datacheck_files(temp_dir_observed_files):
    run_program(
        "lstchain_check_dl1",
        "--batch",
        "--input-file",
        temp_dir_observed_files / "datacheck_dl1_LST-1.Run02008.*.h5",
        "--output-dir",
        temp_dir_observed_files,
    )
    assert (temp_dir_observed_files / "datacheck_dl1_LST-1.Run02008.h5").is_file()
    assert (temp_dir_observed_files / "datacheck_dl1_LST-1.Run02008.pdf").is_file()


def test_lstchain_merged_dl1_to_dl2(
    temp_dir_simulated_files, simulated_dl1_file, merged_simulated_dl1_file, rf_models
):
    simulated_dl1_file_ = simulated_dl1_file.parent.joinpath('another_dl1.h5')
    simulated_dl1_file_.symlink_to(simulated_dl1_file)
    output_file_1 = simulated_dl1_file_.with_name(
        simulated_dl1_file_.name.replace("dl1", "dl2")
    )
    output_file_2 = merged_simulated_dl1_file.with_name(
        merged_simulated_dl1_file.name.replace("dl1", "dl2")
    )
    run_program(
        "lstchain_dl1_to_dl2",
        "-f",
        simulated_dl1_file_,
        merged_simulated_dl1_file,
        "-p",
        rf_models["path"],
        "--output-dir",
        temp_dir_simulated_files,
    )
    assert output_file_1.is_file()
    assert output_file_2.is_file()


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


def test_lstchain_dl1_to_dl2_srcdep(simulated_srcdep_dl2_file):
    assert simulated_srcdep_dl2_file.is_file()
    dl2_srcdep_df = get_srcdep_params(simulated_srcdep_dl2_file)
    assert "expected_src_x" in dl2_srcdep_df['on'].columns
    assert "expected_src_y" in dl2_srcdep_df['on'].columns
    assert "dist" in dl2_srcdep_df['on'].columns
    assert "alpha" in dl2_srcdep_df['on'].columns
    assert "time_gradient_from_source" in dl2_srcdep_df['on'].columns
    assert "skewness_from_source" in dl2_srcdep_df['on'].columns
    assert "gammaness" in dl2_srcdep_df['on'].columns
    assert "reco_type" in dl2_srcdep_df['on'].columns
    assert "reco_energy" in dl2_srcdep_df['on'].columns
    assert "reco_disp_dx" in dl2_srcdep_df['on'].columns
    assert "reco_disp_dy" in dl2_srcdep_df['on'].columns
    assert "reco_src_x" in dl2_srcdep_df['on'].columns
    assert "reco_src_y" in dl2_srcdep_df['on'].columns
    assert "reco_disp_norm_diff" in dl2_srcdep_df['on'].columns
    assert "reco_disp_sign_correctness" in dl2_srcdep_df['on'].columns

@pytest.mark.private_data
def test_lstchain_find_pedestals(temp_dir_observed_files, observed_dl1_files):
    run_program(
        "lstchain_find_pedestals",
        "--input-dir",
        temp_dir_observed_files,
        temp_dir_observed_files,
    )
    for subrun, expected_length in zip((0, 100), (0, 1)):
        path = temp_dir_observed_files / f"pedestal_ids_Run02008.{subrun:04d}.h5"
        assert path.is_file()
        t = read_table(path, "/interleaved_pedestal_ids")
        assert len(t) == expected_length

@pytest.mark.private_data
def test_lstchain_observed_dl1_to_dl2(observed_dl2_file):
    assert observed_dl2_file.is_file()
    dl2_df = pd.read_hdf(observed_dl2_file, key=dl2_params_lstcam_key)
    assert "gammaness" in dl2_df.columns
    assert "reco_type" in dl2_df.columns
    assert "reco_energy" in dl2_df.columns
    assert "reco_alt" in dl2_df.columns
    assert "reco_az" in dl2_df.columns
    assert "reco_src_x" in dl2_df.columns
    assert "reco_src_y" in dl2_df.columns
    assert "reco_disp_dx" in dl2_df.columns
    assert "reco_disp_dy" in dl2_df.columns


@pytest.mark.private_data
def test_lstchain_observed_dl1_to_dl2_srcdep(observed_srcdep_dl2_file):
    assert observed_srcdep_dl2_file.is_file()
    dl2_srcdep_df = get_srcdep_params(observed_srcdep_dl2_file)
    srcdep_config = get_srcdep_config()
    srcdep_assumed_positions = ['on']
    n_off_wobble=srcdep_config.get('n_off_wobble')
    for ioff in range(n_off_wobble):
        off_angle = 2 * np.pi / (n_off_wobble + 1) * (ioff + 1)
        srcdep_assumed_positions.append('off_{:03}'.format(round(np.rad2deg(off_angle))))

    srcdep_dl2_params = [
        'expected_src_x',
        'expected_src_y',
        'dist',
        'alpha',
        'time_gradient_from_source',
        'skewness_from_source',
        'gammaness',
        'reco_type',
        'reco_energy',
        'reco_disp_dx',
        'reco_disp_dy',
        'reco_src_x',
        'reco_src_y',
        'reco_disp_norm_diff',
        'reco_disp_sign_correctness',
    ]

    for srcdep_assumed_position in srcdep_assumed_positions:
        for srcdep_dl2_param in srcdep_dl2_params:
            assert srcdep_dl2_param in dl2_srcdep_df[srcdep_assumed_position].columns


def test_dl1ab(simulated_dl1ab):
    assert simulated_dl1ab.is_file()
    with tables.open_file(simulated_dl1ab) as output:
        assert dl1_images_lstcam_key in output.root
        assert '/source_filenames' in output.root
        assert len(output.root.source_filenames.filenames[:]) == 1


def test_dl1ab_no_images(simulated_dl1_file, tmp_path):
    """Produce a new simulated dl1 file using the dl1ab script."""
    output_file = tmp_path / "dl1ab_no_images.h5"

    config_path = tmp_path / 'config.json'
    with config_path.open('w') as f:
        config = get_standard_config()
        config['tailcut']["picture_thresh"] = 10
        config['tailcut']["boundary_thresh"] = 5
        json.dump(config, f)

    run_program(
        "lstchain_dl1ab",
        "-f", simulated_dl1_file,
        "-o", output_file,
        "-c", config_path,
        '--no-image',
    )

    with tables.open_file(output_file, 'r') as output:
        assert dl1_images_lstcam_key not in output.root
        assert dl1_params_lstcam_key in output.root

        new_parameters = output.root[dl1_params_lstcam_key][:]

        with tables.open_file(simulated_dl1_file, 'r') as input_file:
            old_parameters = input_file.root[dl1_params_lstcam_key][:]

            # new cleaning should result in less pixels
            assert (new_parameters['n_pixels'] <= old_parameters['n_pixels']).all()
            assert (new_parameters['n_pixels'] < old_parameters['n_pixels']).any()
            assert (new_parameters['length'] != old_parameters['length']).any()


def test_dl1ab_on_modified_images(simulated_dl1ab, tmp_path):
    config_path = tmp_path / 'config_image_modifier.json'
    output_file = tmp_path / 'dl1ab_on_modified_images.h5'
    reprocess_output_file = tmp_path / 'dl1ab_on_modified_images_reprocess.h5'
    config_path = files("lstchain.data").joinpath("lstchain_dl1ab_tune_MC_to_Crab_config.json")

    run_program(
        'lstchain_dl1ab',
        '-f', simulated_dl1ab,
        '-o', output_file,
        '-c', config_path,
    )

    # second process should fail as we are trying to re-modify already modified images
    with pytest.raises(ValueError):
        run_program(
            'lstchain_dl1ab',
            '-f', output_file,
            '-o', reprocess_output_file,
            '-c', config_path,
        )


@pytest.mark.private_data
def test_observed_dl1ab(tmp_path, observed_dl1_files):
    output_dl1ab = tmp_path / "dl1ab.h5"
    run_program(
        "lstchain_dl1ab",
        "-f", observed_dl1_files["dl1_file1"],
        "-o", output_dl1ab,
        "--no-pedestal-cleaning"
    )
    assert output_dl1ab.is_file()

    dl1 = pd.read_hdf(observed_dl1_files["dl1_file1"], key=dl1_params_lstcam_key)
    dl1ab = pd.read_hdf(output_dl1ab, key=dl1_params_lstcam_key)
    pd.testing.assert_frame_equal(dl1, dl1ab)


def test_simulated_dl1ab_validity(simulated_dl1_file, simulated_dl1ab):
    assert simulated_dl1ab.is_file()
    dl1_df = pd.read_hdf(simulated_dl1_file, key=dl1_params_lstcam_key)
    dl1ab_df = pd.read_hdf(simulated_dl1ab, key=dl1_params_lstcam_key)
    pd.testing.assert_frame_equal(dl1_df, dl1ab_df)


def test_mc_r0_to_dl2(tmp_path, rf_models, mc_gamma_testfile):
    dl2_file = tmp_path / "dl2_simtel_theta_20_az_180_gdiffuse_10evts.h5"
    run_program(
        "lstchain_mc_r0_to_dl2",
        "--input-file",
        mc_gamma_testfile,
        "--path-models",
        rf_models["path"],
        "--no-dl1",
        "--output-dir",
        tmp_path,
    )
    assert dl2_file.is_file()


def test_read_mc_dl2_to_QTable(simulated_dl2_file):
    import astropy.units as u

    from lstchain.io.io import read_mc_dl2_to_QTable

    events, sim_info, simu_geomag = read_mc_dl2_to_QTable(simulated_dl2_file)
    assert "true_energy" in events.colnames
    assert sim_info.energy_max == 5 * u.TeV


@pytest.mark.private_data
def test_read_data_dl2_to_QTable(temp_dir_observed_files, observed_dl1_files):
    from lstchain.io.io import read_data_dl2_to_QTable

    real_data_dl2_file = temp_dir_observed_files / (
        observed_dl1_files["dl1_file1"].name.replace("dl1", "dl2")
    )
    events, data_pars = read_data_dl2_to_QTable(real_data_dl2_file)

    assert "gh_score" in events.colnames
    assert "B_DELTA" in data_pars


@pytest.mark.private_data
def test_run_summary(run_summary_path):
    from datetime import datetime

    from astropy.table import Table

    date = "20200218"

    assert run_summary_path.is_file()

    run_summary_table = Table.read(run_summary_path)

    assert run_summary_table.meta["date"] == datetime.strptime(date, "%Y%m%d").date().isoformat()
    assert "lstchain_version" in run_summary_table.meta
    assert "run_id" in run_summary_table.columns
    assert "n_subruns" in run_summary_table.columns
    assert "run_type" in run_summary_table.columns
    assert "ucts_timestamp" in run_summary_table.columns
    assert "run_start" in run_summary_table.columns
    assert "dragon_reference_time" in run_summary_table.columns
    assert "dragon_reference_module_id" in run_summary_table.columns
    assert "dragon_reference_module_index" in run_summary_table.columns
    assert "dragon_reference_counter" in run_summary_table.columns
    assert "dragon_reference_source" in run_summary_table.columns

    assert (run_summary_table["run_type"] == ["DATA", "PEDCALIB", "DATA"]).all()


@pytest.mark.private_data
def test_merge_run_summaries(tmp_path):
    test_data = Path(os.getenv('LSTCHAIN_TEST_DATA', 'test_data'))
    monitoring_path = test_data / "real/monitoring"
    merged_file = tmp_path / "merged_summaries.ecsv"
    run_program(
        "lstchain_merge_run_summaries", "-m", monitoring_path, merged_file
    )
    assert merged_file.exists()
    table = Table.read(merged_file)
    # Since no drive log is present in the test sample, only the runs
    # of the first summary appear
    assert 2005 in table["run_id"]
