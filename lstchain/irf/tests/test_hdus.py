import pytest
from astropy.table import Table


@pytest.mark.run(after="test_write_dl2_dataframe")
def test_create_event_list(temp_observed_dl2_file, simulated_irf_file):
    from lstchain.irf.hdu_table import create_event_list
    from lstchain.io.io import dl2_params_lstcam_key, read_data_dl2_to_QTable
    from lstchain.reco.utils import add_delta_t_key, get_effective_time
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    import pandas as pd

    dl2 = pd.read_hdf(temp_observed_dl2_file, key=dl2_params_lstcam_key)

    dl2 = add_delta_t_key(dl2)
    dl2.to_hdf(temp_observed_dl2_file, key=dl2_params_lstcam_key)

    events = read_data_dl2_to_QTable(temp_observed_dl2_file)
    t_eff, t_tot = get_effective_time(events)

    evts, gti, pnt = create_event_list(
        events,
        run_number=2008,
        source_name="Crab",
        source_pos=SkyCoord(ra=83.633, dec=22.01, unit="deg"),
        effective_time=t_eff.value,
        elapsed_time=t_tot.value,
    )

    assert "TIME" in Table.read(evts).columns
    assert "START" in Table.read(gti).columns
    assert "RA_PNT" in Table.read(pnt).meta

    temp_observed_dl3_file = temp_observed_dl2_file.name.replace("dl2", "dl3")
    temp_observed_dl3_file = (
        temp_observed_dl2_file.parent / temp_observed_dl3_file.replace(".h5", ".fits")
    )
    # create a temp dl3 file to test indexing function

    temp_hdulist = fits.HDUList([fits.PrimaryHDU(), evts, gti, pnt])
    for f in fits.open(simulated_irf_file)[1:]:
        temp_hdulist.append(f)

    temp_hdulist.writeto(temp_observed_dl3_file, overwrite=True)

    assert temp_observed_dl3_file.is_file()


@pytest.mark.run(after="test_create_event_list")
def test_create_obs_hdu_index(temp_observed_dl2_file):
    from lstchain.irf.hdu_table import create_obs_hdu_index

    dl3_file = "dl3_LST-1.1.Run02008.fits"
    hdu_list, obs_list = create_obs_hdu_index(
        [dl3_file],
        temp_observed_dl2_file.parent,
        "hdu-index.fits.gz",
        "obs-index.fits.gz",
    )

    assert "HDU_CLASS" in Table.read(hdu_list).columns
    assert "OBJECT" in Table.read(obs_list).columns
