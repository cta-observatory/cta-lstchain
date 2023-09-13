import pytest
import astropy.units as u
from astropy.table import Table
from astropy.time import Time
import numpy as np


@pytest.fixture(scope='session')
def tmp_dl3_path(tmp_path_factory):
    return tmp_path_factory.mktemp('dl3')


@pytest.fixture
def dl3_file(tmp_dl3_path, observed_dl2_file, simulated_irf_file):
    from lstchain.high_level.hdu_table import create_event_list
    from lstchain.io.io import read_data_dl2_to_QTable
    from lstchain.reco.utils import get_effective_time
    from astropy.coordinates import SkyCoord
    from astropy.io import fits

    events, data_pars = read_data_dl2_to_QTable(observed_dl2_file)
    t_eff, t_tot = get_effective_time(events)
    events = events[events["intensity"] > 200]
    source_pos = SkyCoord(ra=83.633, dec=22.01, unit="deg")

    evts, gti, pnt = create_event_list(
        events,
        run_number=2008,
        source_name="Crab",
        source_pos=source_pos,
        effective_time=t_eff.value,
        elapsed_time=t_tot.value,
        data_pars=data_pars
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
        hdu_index,
        overwrite=True,
    )
    create_obs_index_hdu(
        [dl3_file],
        obs_index,
        overwrite=True,
    )

    assert "HDU_CLASS" in Table.read(hdu_index).columns
    assert "OBJECT" in Table.read(obs_index).columns


def test_get_timing_params():
    from lstchain.high_level.hdu_table import get_timing_params, LST_EPOCH
    t = Time(['2020-01-01T00:00:00', '2020-01-01T00:00:01', '2020-01-01T00:00:10'])

    data = Table({
        'event_id': np.arange(len(t)),
        'dragon_time': t.unix,
    })

    params, time_utc = get_timing_params(data)

    # Converting to a single mjd number loses precision but should be better than 0.1 us
    epoch = Time(params["MJDREFI"], params["MJDREFF"], format="mjd")
    assert (epoch + params['t_start']).isclose(t[0], atol=0.1 * u.us)
    assert (epoch + params['t_stop']).isclose(t[-1], atol=0.1 * u.us)

    assert params["date_obs"] == "2020-01-01"
    assert params["date_end"] == "2020-01-01"
    assert params["time_obs"] == "00:00:00.000"
    assert params["time_end"] == "00:00:10.000"
    assert epoch == LST_EPOCH
