import os
import pytest
import pandas as pd

from astropy.io import fits
from astropy.table import Table

from lstchain.tests.test_lstchain import test_dir
from lstchain.io.io import dl2_params_lstcam_key, read_data_dl2_to_QTable

dl2_test_file = os.path.join(test_dir, 'dl2_test.h5')
dl2_file_new = os.path.join(test_dir, 'dl2_test_new.h5')
dl3_file = os.path.join(test_dir, 'dl3_test_new.fits')
dl3_hdu_index = os.path.join(test_dir, 'hdu-index.fits.gz')
dl3_obs_index = os.path.join(test_dir, 'obs-index.fits.gz')

@pytest.mark.run(after='test_write_dl2_dataframe')
def test_create_event_list():
    from lstchain.irf.hdu_table import create_event_list
    from lstchain.io.tests.test_io import dl2_test_file
    import numpy as np

    dl2 = pd.read_hdf(dl2_test_file, key = dl2_params_lstcam_key)
    # Adding some necessary columns for reading it as real data file
    dl2['tel_id'] = dl2['tel_id'].min()
    dl2['dragon_time'] = dl2['tel_id']+np.arange(0,len(dl2['tel_id'])*1e-5, 1e-5)
    dl2['alt_tel'] = dl2["mc_alt_tel"]
    dl2['az_tel'] = dl2["mc_az_tel"]
    dl2.to_hdf(dl2_file_new, key=dl2_params_lstcam_key)

    events = read_data_dl2_to_QTable(dl2_file_new)
    evts, gti, pnt = create_event_list(events, run_number=0, source_name='test')

    assert 'TIME' in Table.read(evts).columns
    assert 'START' in Table.read(gti).columns
    assert 'RA_PNT' in Table.read(pnt).meta

    # create a temp dl3 file to test indexing function
    temp_hdulist = fits.HDUList([fits.PrimaryHDU(), evts, gti, pnt])
    temp_hdulist.writeto(dl3_file, overwrite=True)
    assert os.path.exists(dl3_file)

@pytest.mark.run(after='test_create_event_list')
def test_create_obs_hdu_index():
    from lstchain.irf.hdu_table import create_obs_hdu_index
    from pathlib import Path

    hdu_list, obs_list = create_obs_hdu_index(
                                            ['dl3_test_new.fits'],
                                            Path(test_dir),
                                            'hdu-index.fits.gz',
                                            'obs-index.fits.gz'
                                            )

    assert 'HDU_CLASS' in Table.read(hdu_list).columns
    assert 'OBJECT' in Table.read(obs_list).columns
