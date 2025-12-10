import pytest
from ctapipe.core import run_tool
import os
from astropy.io import fits
import numpy as np
import astropy.units as u
from gammapy.irf import EffectiveAreaTable2D, EnergyDispersion2D
from lstchain.tools.lstchain_create_irf_files import IRFFITSWriter


def test_create_irf_full_enclosure(temp_dir_observed_files, simulated_dl2_file):
    """
    Generating full enclosure IRF file from a test DL2 files
    """
    irf_file = temp_dir_observed_files / "fe_irf.fits.gz"

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
    irf_file = temp_dir_observed_files / "pnt_irf.fits.gz"

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
            assert "RAD_MAX" in hdu.header
            assert isinstance(hdu.header["RAD_MAX"], float)


def test_create_irf_full_enclosure_with_config(
    temp_dir_observed_files, simulated_dl2_file
):
    """
    Generating full enclosure IRF file from a test DL2 files, using
    a config file
    """

    irf_file = temp_dir_observed_files / "fe_irf.fits.gz"
    config_file = os.path.join(os.getcwd(), "./docs/examples/irf_dl3_tool_config.json")

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


def test_create_irf_point_like_srcdep(
    temp_dir_observed_srcdep_files, simulated_srcdep_dl2_file
):
    """
    Generating point-like source-dependent IRF file from a test DL2 files
    """

    irf_file = temp_dir_observed_srcdep_files / "irf.fits.gz"

    assert (
        run_tool(
            IRFFITSWriter(),
            argv=[
                f"--input-gamma-dl2={simulated_srcdep_dl2_file}",
                f"--output-irf-file={irf_file}",
                "--point-like",
                "--source-dep",
                "--overwrite",
            ],
            cwd=temp_dir_observed_srcdep_files,
        )
        == 0
    )

    with fits.open(irf_file) as hdul:
        for hdu in hdul[1:]:
            assert "AL_CUT" in hdu.header
            assert isinstance(hdu.header["AL_CUT"], float)


def test_create_irf_point_like_energy_dependent_cuts(
    temp_dir_observed_files, simulated_dl2_file
):
    """
    Generating point-like IRF file from a test DL2 files, using
    energy-dependent cuts
    """
    from gammapy.irf import RadMax2D

    irf_file = temp_dir_observed_files / "pnt_irf.fits.gz"

    assert (
        run_tool(
            IRFFITSWriter(),
            argv=[
                f"--input-gamma-dl2={simulated_dl2_file}",
                f"--input-proton-dl2={simulated_dl2_file}",
                f"--input-electron-dl2={simulated_dl2_file}",
                f"--output-irf-file={irf_file}",
                "--overwrite",
                "--energy-dependent-gh",
                "--point-like",
                "--energy-dependent-theta",
                "--DL3Cuts.max_theta_cut=1",
                "--DL3Cuts.fill_theta_cut=1",
                "--DL3Cuts.min_event_p_en_bin=2",
            ],
            cwd=temp_dir_observed_files,
        )
        == 0
    )

    assert RadMax2D.read(irf_file, hdu="RAD_MAX")


def test_create_irf_point_like_srcdep_energy_dependent_cuts(
    temp_dir_observed_srcdep_files, simulated_srcdep_dl2_file
):
    """
    Generating point-like source-dependent IRF file from a test DL2 files,
    using energy-dependent cuts
    """
    from astropy.table import QTable

    irf_file = temp_dir_observed_srcdep_files / "irf_edep.fits.gz"

    assert (
        run_tool(
            IRFFITSWriter(),
            argv=[
                f"--input-gamma-dl2={simulated_srcdep_dl2_file}",
                f"--output-irf-file={irf_file}",
                "--point-like",
                "--source-dep",
                "--energy-dependent-gh",
                "--energy-dependent-alpha",
                "--DL3Cuts.min_event_p_en_bin=2",
                "--overwrite",
            ],
            cwd=temp_dir_observed_srcdep_files,
        )
        == 0
    )

    gh_cuts = QTable.read(irf_file, hdu="GH_CUTS")
    assert isinstance(gh_cuts.meta["GH_EFF"], float)

    al_cuts = QTable.read(irf_file, hdu="AL_CUTS")
    assert isinstance(al_cuts.meta["AL_CONT"], float)


@pytest.mark.private_data
def test_create_dl3_energy_dependent_cuts(temp_dir_observed_files, observed_dl2_file):
    """
    Generating an DL3 file from a test DL2 files and test IRF file, using
    energy dependent cuts. Here the previously created IRF is used.
    """
    from lstchain.tools.lstchain_create_dl3_file import DataReductionFITSWriter
    from gammapy.data import Observation

    irf_file = temp_dir_observed_files / "pnt_irf.fits.gz"

    dl2_name = observed_dl2_file.name
    observed_dl3_file = temp_dir_observed_files / dl2_name.replace("dl2", "dl3")
    observed_dl3_file = observed_dl3_file.with_suffix(".fits")

    assert (
        run_tool(
            DataReductionFITSWriter(),
            argv=[
                f"--input-dl2={observed_dl2_file}",
                f"--output-dl3-path={temp_dir_observed_files}",
                f"--input-irf-path={temp_dir_observed_files}",
                "--irf-file-pattern=pnt_irf.fits.gz",
                "--source-name=Crab",
                "--source-ra=83.633deg",
                "--source-dec=22.01deg",
                "--overwrite",
            ],
            cwd=temp_dir_observed_files,
        )
        == 0
    )

    assert (
        Observation.read(event_file=observed_dl3_file, irf_file=irf_file).obs_id == 2008
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
                f"--input-irf-path={simulated_irf_file.parent}",
                f"--irf-file-pattern={simulated_irf_file.name}",
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
def test_dl2_converter(temp_dir_observed_files, observed_dl2_file):
    """
    Creating a ctapipe-compatible DL2 file from an observed DL2 file
    """
    from lstchain.tools.lstchain_dl2_converter import DL2Converter
    from ctapipe.io import TableLoader

    ctapipe_dl2_file = temp_dir_observed_files / "observed_dl2_file_ctapipe.dl2.h5"
    assert (
        run_tool(
            DL2Converter(),
            argv=[
                f"--input_url={observed_dl2_file}",
                f"--output={ctapipe_dl2_file}",
                "--overwrite",
            ],
            cwd=temp_dir_observed_files,
        )
        == 0
    )
    with TableLoader(ctapipe_dl2_file, pointing=True) as loader:
        events = loader.read_telescope_events([1])
        assert len(events) > 0
        assert "telescope_pointing_azimuth" in events.colnames
        assert "telescope_pointing_altitude" in events.colnames

@pytest.mark.private_data
def test_create_dl3_with_config(temp_dir_observed_files, observed_dl2_file):
    """
    Generating an DL3 file from a test DL2 files and test IRF file, using
    a config file
    """
    from lstchain.tools.lstchain_create_dl3_file import DataReductionFITSWriter

    config_file = os.path.join(os.getcwd(), "docs/examples/irf_dl3_tool_config.json")

    assert (
        run_tool(
            DataReductionFITSWriter(),
            argv=[
                f"--input-dl2={observed_dl2_file}",
                f"--output-dl3-path={temp_dir_observed_files}",
                f"--input-irf-path={temp_dir_observed_files}",
                "--irf-file-pattern=fe_irf.fits.gz",
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
def test_create_srcdep_dl3(
    temp_dir_observed_srcdep_files, observed_srcdep_dl2_file, simulated_srcdep_irf_file
):
    """
    Generating a source-dependent DL3 file from a test DL2 files and test IRF file
    """
    from lstchain.tools.lstchain_create_dl3_file import DataReductionFITSWriter
    from lstchain.paths import dl2_to_dl3_filename

    assert (
        run_tool(
            DataReductionFITSWriter(),
            argv=[
                f"--input-dl2={observed_srcdep_dl2_file}",
                f"--output-dl3-path={temp_dir_observed_srcdep_files}",
                f"--input-irf-path={simulated_srcdep_irf_file.parent}",
                f"--irf-file-pattern={simulated_srcdep_irf_file.name}",
                "--source-name=Crab",
                "--source-ra=83.633deg",
                "--source-dec=22.01deg",
                "--source-dep",
                "--overwrite",
            ],
            cwd=temp_dir_observed_srcdep_files,
        )
        == 0
    )

    hdulist = fits.open(
        temp_dir_observed_srcdep_files / dl2_to_dl3_filename(observed_srcdep_dl2_file)
    )
    ra = hdulist[1].data["RA"]
    dec = hdulist[1].data["DEC"]

    np.testing.assert_allclose(ra, 83.63, atol=1e-2)
    np.testing.assert_allclose(dec, 22.01, atol=1e-2)


@pytest.mark.private_data
def test_create_srcdep_dl3_energy_dependent_cuts(
    temp_dir_observed_srcdep_files, observed_srcdep_dl2_file
):
    """
    Generating a source-dependent DL3 file from a test DL2 files and test IRF file,
    using energy-dependent cuts
    """
    from lstchain.tools.lstchain_create_dl3_file import DataReductionFITSWriter

    irf_file = temp_dir_observed_srcdep_files / "irf_edep.fits.gz"

    assert (
        run_tool(
            DataReductionFITSWriter(),
            argv=[
                f"--input-dl2={observed_srcdep_dl2_file}",
                f"--output-dl3-path={temp_dir_observed_srcdep_files}",
                f"--input-irf-path={irf_file.parent}",
                f"--irf-file-pattern={irf_file.name}",
                "--source-name=Crab",
                "--source-ra=83.633deg",
                "--source-dec=22.01deg",
                "--source-dep",
                "--overwrite",
            ],
            cwd=temp_dir_observed_srcdep_files,
        )
        == 0
    )


@pytest.mark.private_data
def test_index_dl3_files(temp_dir_observed_files):
    """
    Generating Index files from a given path and glob pattern for DL3 files
    """
    from lstchain.tools.lstchain_create_dl3_index_files import FITSIndexWriter
    from gammapy.data import DataStore

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
    data = DataStore.from_dir(temp_dir_observed_files)

    assert 2008 in data.obs_table["OBS_ID"]

    for hdu_name in [
        "EVENTS",
        "GTI",
        "POINTING",
        "EFFECTIVE AREA",
        "ENERGY DISPERSION",
        "BACKGROUND",
        "PSF",
    ]:
        assert hdu_name in data.hdu_table["HDU_NAME"]


@pytest.mark.private_data
def test_index_srcdep_dl3_files(temp_dir_observed_srcdep_files):
    """
    Generating Index files from a given path and glob pattern for srcdep DL3 files
    """
    from lstchain.tools.lstchain_create_dl3_index_files import FITSIndexWriter
    from gammapy.data import DataStore

    assert (
        run_tool(
            FITSIndexWriter(),
            argv=[
                f"--input-dl3-dir={temp_dir_observed_srcdep_files}",
                "--overwrite",
            ],
            cwd=temp_dir_observed_srcdep_files,
        )
        == 0
    )
    data = DataStore.from_dir(temp_dir_observed_srcdep_files)

    assert 2008 in data.obs_table["OBS_ID"]

    for hdu_name in [
        "EVENTS",
        "GTI",
        "POINTING",
        "EFFECTIVE AREA",
        "ENERGY DISPERSION",
    ]:
        assert hdu_name in data.hdu_table["HDU_NAME"]


@pytest.mark.private_data
@pytest.mark.parametrize(
    "e_true, scale_factor",
    [
        (0.2 * u.TeV, 1.2),
        (2 * u.TeV, 1.2),
        (20 * u.TeV, 1.2),
        (0.2 * u.TeV, 0.8),
        (2 * u.TeV, 0.8),
        (20 * u.TeV, 0.8)
    ]
)
def test_scale_true_energy_irf(temp_dir_observed_files, simulated_dl2_file, e_true, scale_factor):
    """Check the validity of modified IRFs after scaling the True Energy by a factor."""
    irf_file = temp_dir_observed_files / "fe_irf.fits.gz"
    irf_file_mod = temp_dir_observed_files / "mod_irf.fits.gz"
    config_file = os.path.join(os.getcwd(), "docs/examples/irf_dl3_tool_config.json")

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
                "--DataBinning.true_energy_n_bins=2",
                "--DataBinning.reco_energy_n_bins=2",
                "--DataBinning.true_energy_min: 0.2",
                "--DataBinning.true_energy_max: 0.3",
                "--DL3Cuts.min_event_p_en_bin=2",
            ],
            cwd=temp_dir_observed_files,
        )
        == 0
    )
    assert (
        run_tool(
            IRFFITSWriter(),
            argv=[
                f"--input-gamma-dl2={simulated_dl2_file}",
                f"--input-proton-dl2={simulated_dl2_file}",
                f"--input-electron-dl2={simulated_dl2_file}",
                f"--output-irf-file={irf_file_mod}",
                f"--config={config_file}",
                "--overwrite",
                "--DataBinning.true_energy_n_bins=2",
                "--DataBinning.reco_energy_n_bins=2",
                "--DataBinning.true_energy_min: 0.2",
                "--DataBinning.true_energy_max: 0.3",
                "--DL3Cuts.min_event_p_en_bin=2",
                f"--DataBinning.scale_true_energy={scale_factor}",
            ],
            cwd=temp_dir_observed_files,
        )
        == 0
    )

    aeff_hdu = EffectiveAreaTable2D.read(irf_file, hdu="EFFECTIVE AREA")
    aeff_mod_hdu = EffectiveAreaTable2D.read(irf_file_mod, hdu="EFFECTIVE AREA")

    edisp_hdu = EnergyDispersion2D.read(irf_file, hdu="ENERGY DISPERSION")
    edisp_mod_hdu = EnergyDispersion2D.read(irf_file_mod, hdu="ENERGY DISPERSION")

    assert aeff_mod_hdu.data.shape == aeff_hdu.data.shape
    assert edisp_mod_hdu.data.shape == edisp_hdu.data.shape

    edisp = EnergyDispersion2D.read(irf_file)
    edisp_mod = EnergyDispersion2D.read(irf_file_mod)

    e_migra = edisp.axes["migra"].center
    e_migra_mod = edisp_mod.axes["migra"].center

    e_migra_prob = edisp.evaluate(
        offset=0.4 * u.deg,
        energy_true=e_true,
        migra=e_migra
    )

    e_migra_prob_mod = edisp_mod.evaluate(
        offset=0.4 * u.deg,
        energy_true=e_true,
        migra=e_migra_mod,
    )

    # Check that the maximum of the probability distribution has shifted between the two EDISP
    order_max = np.argmax(e_migra_prob)
    order_max_mod = np.argmax(e_migra_prob_mod)
    assert order_max != order_max_mod, f"Index of maximum did not shift for energy {e_true}"
