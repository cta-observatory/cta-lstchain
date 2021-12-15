import pytest
from astropy.table import Table


@pytest.mark.run(after="test_write_dl2_dataframe")
@pytest.mark.private_data
def test_create_event_list(observed_dl2_file, simulated_irf_file):
    from lstchain.irf.hdu_table import create_event_list, add_icrs_position_params
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

    assert "TIME" in Table.read(evts).columns
    assert "START" in Table.read(gti).columns
    assert "RA_PNT" in Table.read(pnt).columns

    observed_dl3_file = observed_dl2_file.name.replace("dl2", "dl3")
    observed_dl3_file = (
        observed_dl2_file.parent / observed_dl3_file.replace(".h5", ".fits")
    )
    # create a temp dl3 file to test indexing function

    temp_hdulist = fits.HDUList([fits.PrimaryHDU(), evts, gti, pnt])
    for f in fits.open(simulated_irf_file)[1:]:
        temp_hdulist.append(f)

    temp_hdulist.writeto(observed_dl3_file, overwrite=True)

    assert observed_dl3_file.is_file()


@pytest.mark.run(after="test_create_event_list")
@pytest.mark.private_data
def test_create_obs_hdu_index(observed_dl2_file):
    from lstchain.irf.hdu_table import create_hdu_index_hdu, create_obs_index_hdu

    dl3_file = observed_dl2_file.name.replace("dl2", "dl3")
    dl3_file = dl3_file.replace(".h5", ".fits")

    hdu_index = observed_dl2_file.parent / "hdu-index.fits.gz"
    obs_index = observed_dl2_file.parent / "obs-index.fits.gz"

    create_hdu_index_hdu(
        [dl3_file],
        observed_dl2_file.parent,
        hdu_index,
        overwrite=True,
    )
    create_obs_index_hdu(
        [dl3_file],
        observed_dl2_file.parent,
        obs_index,
        overwrite=True,
    )

    assert "HDU_CLASS" in Table.read(hdu_index).columns
    assert "OBJECT" in Table.read(obs_index).columns
