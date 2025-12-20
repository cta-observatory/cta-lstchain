from ctapipe_io_lst import LSTEventSource
import pandas as pd
import pytest
from ctapipe.instrument import SubarrayDescription
import os

from lstchain.io.io import dl1_params_lstcam_key
from lstchain.scripts.tests.test_lstchain_scripts import run_program
from lstchain.tests.test_lstchain import (
    test_calib_path,
    test_data,
    test_drive_report,
    test_drs4_pedestal_path,
    test_r0_path,
    test_r0_path2,
    test_time_calib_path,
)


# add a marker for the tests that need private data and don't run them
# by default
def pytest_configure(config):
    if "private_data" not in config.option.markexpr:
        if config.option.markexpr:
            config.option.markexpr += " and "
        config.option.markexpr += "not private_data"


@pytest.fixture(scope="session")
def lst1_subarray():
    return LSTEventSource.create_subarray(tel_id=1)


@pytest.fixture(scope="session")
def temp_dir_simulated_files(tmp_path_factory):
    """Temporal common directory for processing simulated data."""
    return tmp_path_factory.mktemp("simulated_files")

@pytest.fixture(scope="session")
def temp_dir_simulated_srcdep_files(tmp_path_factory):
    """Temporal common directory for processing simulated data for source-dependent analysis."""
    return tmp_path_factory.mktemp("simulated_srcdep_files")


@pytest.fixture(scope="session")
def temp_dir_observed_files(tmp_path_factory):
    """Temporal common directory for processing observed data."""
    return tmp_path_factory.mktemp("observed_files")


@pytest.fixture(scope="session")
def temp_dir_observed_srcdep_files(tmp_path_factory):
    """Temporal common directory for processing observed data."""
    return tmp_path_factory.mktemp("observed_srcdep_files")


@pytest.fixture(scope="session")
def mc_gamma_testfile():
    """Get a simulated test file."""
    return test_data / "mc/simtel_theta_20_az_180_gdiffuse_10evts.simtel.gz"


@pytest.fixture(scope="session")
def simulated_dl1_file(temp_dir_simulated_files, mc_gamma_testfile):
    """Produce a dl1 file from simulated data."""
    output_dl1_path = temp_dir_simulated_files / "dl1_simtel_theta_20_az_180_gdiffuse_10evts.h5"
    run_program(
        "lstchain_mc_r0_to_dl1", "-f", mc_gamma_testfile, "-o", temp_dir_simulated_files
    )
    return output_dl1_path


@pytest.fixture(scope='session')
def run_summary_path(temp_dir_observed_files):
    date = "20200218"
    r0_path = test_data / "real/R0"
    run_summary_path = temp_dir_observed_files / f"RunSummary_{date}.ecsv"
    run_program(
        "lstchain_create_run_summary",
        "--date", date,
        "--r0-path", r0_path,
        "--output-dir", temp_dir_observed_files
    )

    return run_summary_path


@pytest.fixture(scope="session")
def observed_dl1_files(temp_dir_observed_files, run_summary_path):
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
    dl1_output_path1 = temp_dir_observed_files / "dl1_LST-1.Run02008.0000.h5"
    muons_file1 = temp_dir_observed_files / "muons_LST-1.Run02008.0000.fits"
    datacheck_file1 = temp_dir_observed_files / "datacheck_dl1_LST-1.Run02008.0000.h5"
    dvr_file1 = temp_dir_observed_files / "DVR_settings_LST-1.Run02008.h5"
    pixmasks_file1 = temp_dir_observed_files / "Pixel_selection_LST-1.Run02008.0000.h5"
    interleaved_file1 = temp_dir_observed_files / "interleaved/interleaved_LST-1.Run02008.0000.h5"
 
    # Second set of files
    dl1_output_path2 = temp_dir_observed_files / "dl1_LST-1.Run02008.0100.h5"
    muons_file2 = temp_dir_observed_files / "muons_LST-1.Run02008.0100.fits"
    datacheck_file2 = temp_dir_observed_files / "datacheck_dl1_LST-1.Run02008.0100.h5"
    interleaved_file2 = temp_dir_observed_files / "interleaved/interleaved_LST-1.Run02008.0100.h5"

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
        "--dragon-module-id",
        "132",
        "--default-trigger-type=tib",
    )

    run_program(
        "lstchain_check_dl1",
        "-b",
        "--omit-pdf",
        "--output-dir",
        temp_dir_observed_files,
        "--input-file",
        dl1_output_path1
    )

    run_program(
        "lstchain_dvr_pixselector",
        "--dl1-files",
        dl1_output_path1,
        "--output-dir",
        temp_dir_observed_files
    )

    run_program(
        "lstchain_dvr_pixselector",
        "--dl1-files",
        dl1_output_path1,
        "--output-dir",
        temp_dir_observed_files,
        "--action",
        "create_pixel_masks"
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
        test_drive_report,
        '--run-summary-path',
        run_summary_path,
        "--default-trigger-type=tib",
    )

    run_program(
        "lstchain_check_dl1",
        "-b",
        "--omit-pdf",
        "--output-dir",
        temp_dir_observed_files,
        "--input-file",
        dl1_output_path2
    )

    return {
        'dl1_file1': dl1_output_path1,
        'muons1': muons_file1,
        'datacheck1': datacheck_file1,
        'dvr_file1': dvr_file1,
        'pixmasks_file1': pixmasks_file1,
        'interleaved_file1': interleaved_file1,
        'dl1_file2': dl1_output_path2,
        'muons2': muons_file2,
        'datacheck2': datacheck_file2,
        'interleaved_file2': interleaved_file2
    }



@pytest.fixture(scope="session")
def interleaved_r1_file(temp_dir_observed_files, run_summary_path):
    test_pedcal_run = test_data / 'real/R0/20200218/LST-1.1.Run02006.0000_first50.fits.fz'

    run_program(
        "lstchain_data_r0_to_dl1",
        "-f",
        test_pedcal_run,
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
        '--run-summary-path',
        run_summary_path,
        "--default-trigger-type=tib"
    )

    return temp_dir_observed_files / "interleaved/interleaved_LST-1.Run02006.0000.h5"


@pytest.fixture(scope="session")
def simulated_dl2_file(temp_dir_simulated_files, simulated_dl1_file, rf_models):
    """
    Produce the test dl2 file from the simulated dl1 test file
    using the random forest test models.
    """
    dl2_file = temp_dir_simulated_files / "dl2_simtel_theta_20_az_180_gdiffuse_10evts.h5"
    run_program(
        "lstchain_dl1_to_dl2",
        "--input-file",
        simulated_dl1_file,
        "--path-models",
        rf_models["path"],
        "--output-dir",
        temp_dir_simulated_files,
    )
    return dl2_file


@pytest.fixture(scope="session")
def simulated_srcdep_dl2_file(temp_dir_simulated_srcdep_files, simulated_dl1_file, rf_models_srcdep):
    """
    Produce the test source-dependent dl2 file from the simulated dl1 test file
    using the random forest test models.
    """
    srcdep_dl2_file = temp_dir_simulated_srcdep_files / "dl2_simtel_theta_20_az_180_gdiffuse_10evts.h5"
    srcdep_config_file = os.path.join(os.getcwd(), "./lstchain/data/lstchain_src_dep_config.json")
    run_program(
        "lstchain_dl1_to_dl2",
        "--input-file",
        simulated_dl1_file,
        "--path-models",
        rf_models_srcdep["path"],
        "--output-dir",
        temp_dir_simulated_srcdep_files,
        "--config",
        srcdep_config_file,
    )
    return srcdep_dl2_file

@pytest.fixture(scope="session")
def fake_dl1_proton_file(temp_dir_simulated_files, simulated_dl1_file):
    """
    Produce a fake dl1 proton file by copying the dl1 gamma test file
    and changing mc_type.
    """
    dl1_proton_file = temp_dir_simulated_files / "dl1_fake_proton.simtel.h5"
    events = pd.read_hdf(simulated_dl1_file, key=dl1_params_lstcam_key)
    events.mc_type = 101
    events.to_hdf(dl1_proton_file, key=dl1_params_lstcam_key)
    return dl1_proton_file

@pytest.fixture(scope="session")
def simulated_dl1_srcdep_file(temp_dir_simulated_files, simulated_dl1_file):
    """
    Produce a fake dl1 gamma file by copying the dl1 gamma test file
    and changing src_x & src_y to zeros to keep events after src_r cuts.
    """
    dl1_gamma_srcdep_file = temp_dir_simulated_files / "dl1_fake_gamma_srcdep.simtel.h5"
    events = pd.read_hdf(simulated_dl1_file, key=dl1_params_lstcam_key)
    subarray_info = SubarrayDescription.from_hdf(simulated_dl1_file)
    events.src_x = 0
    events.src_y = 0
    events.to_hdf(dl1_gamma_srcdep_file, key=dl1_params_lstcam_key)
    subarray_info.to_hdf(dl1_gamma_srcdep_file)
    return dl1_gamma_srcdep_file


@pytest.fixture(scope="session")
def rf_models(temp_dir_simulated_files, simulated_dl1_file):
    """Produce test random forest models."""
    gamma_file = simulated_dl1_file
    proton_file = simulated_dl1_file
    models_path = temp_dir_simulated_files
    file_model_energy = models_path / "reg_energy.sav"
    file_model_gh_sep = models_path / "cls_gh.sav"
    file_model_disp_norm = models_path / "reg_disp_norm.sav"
    file_model_disp_sign = models_path / "cls_disp_sign.sav"

    run_program(
        "lstchain_mc_trainpipe",
        "--fg",
        gamma_file,
        "--fp",
        proton_file,
        "-o",
        models_path,
    )
    return {
        "energy": file_model_energy,
        "gh_sep": file_model_gh_sep,
        "path": models_path,
        "disp_norm": file_model_disp_norm,
        "disp_sign": file_model_disp_sign,
    }

@pytest.fixture(scope="session")
def rf_models_srcdep(temp_dir_simulated_srcdep_files, simulated_dl1_file, simulated_dl1_srcdep_file):
    """Produce test random forest models for source-dependent analysis."""
    gamma_file = simulated_dl1_srcdep_file
    proton_file = simulated_dl1_file
    models_srcdep_path = temp_dir_simulated_srcdep_files
    file_model_energy_srcdep = models_srcdep_path / "reg_energy.sav"
    file_model_gh_sep_srcdep = models_srcdep_path / "cls_gh.sav"
    file_model_disp_norm_srcdep = models_srcdep_path / "reg_disp_norm.sav"
    file_model_disp_sign_srcdep = models_srcdep_path / "cls_disp_sign.sav"
    srcdep_config_file = os.path.join(os.getcwd(), "./lstchain/data/lstchain_src_dep_config.json")

    run_program(
        "lstchain_mc_trainpipe",
        "--fg",
        gamma_file,
        "--fp",
        proton_file,
        "-o",
        models_srcdep_path,
        "-c",
        srcdep_config_file,
    )
    return {
        "energy": file_model_energy_srcdep,
        "gh_sep": file_model_gh_sep_srcdep,
        "path": models_srcdep_path,
        "disp_norm": file_model_disp_norm_srcdep,
        "disp_sign": file_model_disp_sign_srcdep,
    }

@pytest.fixture(scope="session")
def observed_dl2_file(temp_dir_observed_files, observed_dl1_files, rf_models):
    """Produce a dl2 file from an observed dl1 file."""
    real_data_dl2_file = temp_dir_observed_files / (observed_dl1_files["dl1_file1"].name.replace("dl1", "dl2"))
    run_program(
        "lstchain_dl1_to_dl2",
        "--input-file",
        observed_dl1_files["dl1_file1"],
        "--path-models",
        rf_models["path"],
        "--output-dir",
        temp_dir_observed_files
    )
    return real_data_dl2_file


@pytest.fixture(scope="session")
def observed_srcdep_dl2_file(temp_dir_observed_srcdep_files, observed_dl1_files, rf_models_srcdep):
    """Produce a source-dependent dl2 file from an observed dl1 file."""
    real_data_srcdep_dl2_file = temp_dir_observed_srcdep_files / (observed_dl1_files["dl1_file1"].name.replace("dl1", "dl2"))
    srcdep_config_file = os.path.join(os.getcwd(), "./lstchain/data/lstchain_src_dep_config.json")
    run_program(
        "lstchain_dl1_to_dl2",
        "--input-file",
        observed_dl1_files["dl1_file1"],
        "--path-models",
        rf_models_srcdep["path"],
        "--output-dir",
        temp_dir_observed_srcdep_files,
        "--config",
        srcdep_config_file,
    )
    return real_data_srcdep_dl2_file


@pytest.fixture(scope="session")
def simulated_irf_file(simulated_dl2_file):
    """
    Produce test irf file from the simulated dl2 test file.
    Using the same test file for gamma, proton and electron inputs
    """

    irf_file = simulated_dl2_file.parent / "irf.fits.gz"
    run_program(
        "lstchain_create_irf_files",
        "--input-gamma-dl2",
        simulated_dl2_file,
        "--input-proton-dl2",
        simulated_dl2_file,
        "--input-electron-dl2",
        simulated_dl2_file,
        "--output-irf-file",
        irf_file
    )
    return irf_file


@pytest.fixture(scope="session")
def simulated_srcdep_irf_file(simulated_srcdep_dl2_file):
    """
    Produce test source-dependent irf file from the simulated dl2 test file.
    """

    srcdep_irf_file = simulated_srcdep_dl2_file.parent / "srcdep_irf.fits.gz"
    run_program(
        "lstchain_create_irf_files",
        "--input-gamma-dl2",
        simulated_srcdep_dl2_file,
        "--output-irf-file",
        srcdep_irf_file,
        "--point-like",
        "--source-dep"
    )
    return srcdep_irf_file
