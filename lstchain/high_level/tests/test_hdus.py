import pytest
from astropy.table import Table


@pytest.fixture(scope='session')
def tmp_dl3_path(tmp_path_factory):
    return tmp_path_factory.mktemp('dl3')


@pytest.fixture
def dl3_file(tmp_dl3_path, observed_dl2_file, simulated_irf_file):
    from lstchain.high_level.hdu_table import create_event_list, add_icrs_position_params
    from lstchain.io.io import read_data_dl2_to_QTable
    from lstchain.reco.utils import get_effective_time
    from astropy.coordinates import SkyCoord
    from astropy.io import fits

    events = read_data_dl2_to_QTable(observed_dl2_file)
    t_eff, t_tot = get_effective_time(events)
    events = events[events["intensity"] > 200]
    source_pos = SkyCoord(ra=83.633, dec=22.01, unit="deg")
    events = add_icrs_position_params(events, source_pos)

    evts, gti, pnt = create_event_list(
        events,
        run_number=2008,
        source_name="Crab",
        source_pos=source_pos,
        effective_time=t_eff.value,
        elapsed_time=t_tot.value,
    )

    name = observed_dl2_file.name
    observed_dl3_file = tmp_dl3_path / name.replace('dl2', 'dl3')
    observed_dl3_file = observed_dl3_file.with_suffix(".fits")

    # create a temp dl3 file to test indexing function

    temp_hdulist = fits.HDUList([fits.PrimaryHDU(), evts, gti, pnt])
    for f in fits.open(simulated_irf_file)[1:]:
        temp_hdulist.append(f)

    temp_hdulist.writeto(observed_dl3_file, overwrite=True)

    return observed_dl3_file


@pytest.mark.private_data
def test_create_event_list(dl3_file):
    assert dl3_file.is_file()


@pytest.mark.private_data
def test_create_obs_hdu_index(tmp_path, dl3_file):
    from lstchain.high_level.hdu_table import create_hdu_index_hdu, create_obs_index_hdu

    hdu_index = tmp_path / "hdu-index.fits.gz"
    obs_index = tmp_path / "obs-index.fits.gz"

    create_hdu_index_hdu(
        [dl3_file],
        tmp_path,
        hdu_index,
        overwrite=True,
    )
    create_obs_index_hdu(
        [dl3_file],
        tmp_path,
        obs_index,
        overwrite=True,
    )

    assert "HDU_CLASS" in Table.read(hdu_index).columns
    assert "OBJECT" in Table.read(obs_index).columns
