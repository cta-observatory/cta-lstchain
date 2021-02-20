import tempfile
from pathlib import Path

import pandas as pd
import pytest

from lstchain.io.io import dl1_params_lstcam_key
from lstchain.scripts.tests.test_lstchain_scripts import run_program
from ctapipe.utils import get_dataset_path


# add a marker for the tests that need private data and don't run them
# by default
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "private_data: mark tests that needs the private test data"
    )

    if 'private_data' not in config.option.markexpr:
        if config.option.markexpr:
            config.option.markexpr += ' and '
        else:
            config.option.markexpr += 'not private_data'


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory(prefix='test_lstchain') as d:
        yield Path(d)


@pytest.fixture
def mc_gamma_testfile():
    """Get a simulated test file."""
    return get_dataset_path('gamma_test_large.simtel.gz')


@pytest.fixture
def simulated_dl1(tmp_path, mc_gamma_testfile):
    """Produce a dl1 file from simulated data."""
    output_dl1_path = tmp_path / "dl1_gamma_test_large.h5"
    run_program("lstchain_mc_r0_to_dl1", "-f", mc_gamma_testfile, "-o", tmp_path)
    return {
        'file': output_dl1_path,
        'path': tmp_path
    }


@pytest.fixture
def simulated_dl2(tmp_path, simulated_dl1, rf_models):
    """
    Produce the test dl2 file from the simulated dl1 test file
    using the random forest test models.
    """
    dl2_file = tmp_path / "dl2_gamma_test_large.h5"
    run_program(
        "lstchain_dl1_to_dl2",
        "--input-file",
        simulated_dl1["file"],
        "--path-models",
        rf_models["path"],
        "--output-dir",
        tmp_path
    )
    return dl2_file


@pytest.fixture
def fake_dl1_proton_file(tmp_path, simulated_dl1):
    """
    Produce a fake dl1 proton file by copying the dl2 gamma test file
    and changing mc_type.
    """
    dl1_proton_file = tmp_path / 'dl1_fake_proton.simtel.h5'
    events = pd.read_hdf(simulated_dl1["file"], key=dl1_params_lstcam_key)
    events.mc_type = 101
    events.to_hdf(dl1_proton_file, key=dl1_params_lstcam_key)
    return dl1_proton_file


@pytest.fixture
def rf_models(tmp_path, simulated_dl1):
    """Produce test random forest models."""
    gamma_file = simulated_dl1["file"]
    proton_file = simulated_dl1["file"]
    output_dir = tmp_path
    file_model_energy = output_dir / "reg_energy.sav"
    file_model_disp = output_dir / "reg_disp_vector.sav"
    file_model_gh_sep = output_dir / "cls_gh.sav"

    run_program(
        "lstchain_mc_trainpipe", "--fg", gamma_file, "--fp", proton_file, "-o", output_dir
    )
    return {
        'energy': file_model_energy,
        'disp': file_model_disp,
        'gh_sep': file_model_gh_sep,
        'path': output_dir
    }
