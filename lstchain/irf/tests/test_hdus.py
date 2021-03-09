import pytest
import pandas as pd
from astropy.io import fits
from astropy.table import Table

from lstchain.io.io import dl2_params_lstcam_key, read_data_dl2_to_QTable


@pytest.mark.run(after="test_write_dl2_dataframe")
def test_create_event_list(simulated_dl2_file, simulated_irf_file):
    from lstchain.irf.hdu_table import create_event_list
    from lstchain.reco.utils import add_delta_t_key, get_effective_time
    import numpy as np
    from astropy.coordinates import SkyCoord

    dl2_file_new = simulated_dl2_file.parent / simulated_dl2_file.name.replace(
        ".h5", "_new.h5"
    )

    dl2 = pd.read_hdf(simulated_dl2_file, key=dl2_params_lstcam_key)

    # Adding some necessary columns for reading it as real data file
    # Simulated data file is being used as this is run before the test_lstchain_scripts
    dl2["tel_id"] = dl2["tel_id"].min()
    dl2["dragon_time"] = dl2["tel_id"] + np.arange(0, len(dl2["tel_id"]) * 1e-3, 1e-3)
    dl2 = add_delta_t_key(dl2)
    dl2["alt_tel"] = dl2["mc_alt_tel"]
    dl2["az_tel"] = dl2["mc_az_tel"]
    dl2.to_hdf(dl2_file_new, key=dl2_params_lstcam_key)

    events = read_data_dl2_to_QTable(dl2_file_new)
    t_eff, t_tot = get_effective_time(events)

    evts, gti, pnt = create_event_list(
        events,
        run_number=0,
        source_name="Crab",
        source_pos=SkyCoord(ra=83.633, dec=22.01, unit="deg"),
        effective_time=t_eff.value,
        elapsed_time=t_tot.value,
    )

    assert "TIME" in Table.read(evts).columns
    assert "START" in Table.read(gti).columns
    assert "RA_PNT" in Table.read(pnt).meta

    dl3_file = dl2_file_new.name.replace("dl2", "dl3")
    dl3_file = simulated_dl2_file.parent / dl3_file.replace(".h5", ".fits")
    # create a temp dl3 file to test indexing function

    temp_hdulist = fits.HDUList(
        [fits.PrimaryHDU(), evts, gti, pnt]
    )
    for f in fits.open(simulated_irf_file)[1:]:
        temp_hdulist.append(f)

    temp_hdulist.writeto(dl3_file, overwrite=True)

    assert dl3_file.is_file()


@pytest.mark.run(after="test_create_event_list")
def test_create_obs_hdu_index(simulated_dl2_file):
    from lstchain.irf.hdu_table import create_obs_hdu_index

    dl3_file = "dl3_gamma_test_large_new.fits"
    hdu_list, obs_list = create_obs_hdu_index(
        [dl3_file], simulated_dl2_file.parent, "hdu-index.fits.gz", "obs-index.fits.gz"
    )

    assert "HDU_CLASS" in Table.read(hdu_list).columns
    assert "OBJECT" in Table.read(obs_list).columns
