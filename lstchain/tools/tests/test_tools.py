import os
import pytest

import pandas as pd

from ctapipe.core import run_tool

from lstchain.tests.test_lstchain import test_dir
from lstchain.scripts.tests.test_lstchain_scripts import dl2_file

# Create a separate temp folder to store these files?
dl2_file_pnt_like = os.path.join(test_dir, 'dl2_gamma_test_point_like.h5')
dl2_file_new = os.path.join(test_dir, 'dl2_gamma_test_large_new.h5')
merged_dl2_file = os.path.join(test_dir, 'script_merged_dl2.h5')
cuts = os.path.join(test_dir, 'cuts.json')
irf_file = os.path.join(test_dir, 'irf.fits.gz')
dl3_file = os.path.join(test_dir, 'dl3_gamma_test_large_new.fits')
dl3_hdu_index = os.path.join(test_dir, 'hdu-index.fits.gz')
dl3_obs_index = os.path.join(test_dir, 'obs-index.fits.gz')

@pytest.mark.run(after='test_lstchain_dl1_to_dl2')
@pytest.mark.parametrize("point_like_IRF", [True,False])
def test_create_irf(point_like_IRF):
    """
    Generating an IRF file from a test DL2 files
    """
    from lstchain.tools.lstchain_create_irf_files import IRFFITSWriter
    import json

    # Selection cuts have to be changed for tests
    if os.path.exists(cuts):
        open(cuts,'r')
    else:
        data = json.load(open(os.path.join('lstchain/data/data_selection_cuts.json')))
        data["fixed_cuts"]["gh_score"][0] = 0.3
        data["events_filters"]["intensity"][0]=0
        json.dump(data, open(cuts,'x'))

    assert (
          run_tool(
              IRFFITSWriter(),
              argv=[
                  f"--input_gamma_dl2={dl2_file}",
                  f"--output_irf_file={irf_file}",
                  f"--point_like={point_like_IRF}",
                  f"--config_file={cuts}",
              ]
          )
          == 0
      )

    assert (
        run_tool(
            IRFFITSWriter(),
            argv=["--help-all"]
        )
        == 0
    )

@pytest.mark.run(after='test_create_irf')
@pytest.mark.parametrize("add_IRF", [True,False])
def test_create_dl3(add_IRF):
    """
    Generating an DL3 file from a test DL2 files and test IRF file
    """
    from lstchain.tools.lstchain_create_dl3_file import DataReductionFITSWriter
    from lstchain.io.io import dl2_params_lstcam_key
    import numpy as np

    dl2 = pd.read_hdf(dl2_file, key = dl2_params_lstcam_key)
    # Adding some necessary columns for reading it as real data file
    dl2['tel_id'] = dl2['tel_id'].min()
    dl2['dragon_time'] = dl2["obs_id"]+np.arange(0,len(dl2['tel_id'])*1e-5, 1e-5)
    dl2['alt_tel'] = dl2["mc_alt_tel"]
    dl2['az_tel'] = dl2["mc_az_tel"]
    dl2.to_hdf(dl2_file_new, key=dl2_params_lstcam_key)

    assert (
          run_tool(
              DataReductionFITSWriter(),
              argv=[
                  f"--input_dl2={dl2_file_new}",
                  f"--output_dl3_path={test_dir}",
                  f"--add_irf={add_IRF}",
                  f"--input_irf={irf_file}",
                  f"--config_file={cuts}",
                  "--source_name=Crab"
              ]
          )
          == 0
      )

    assert (
        run_tool(
            DataReductionFITSWriter(),
            argv=["--help-all"]
        )
        == 0
    )

@pytest.mark.run(after='test_create_dl3')
def test_index_dl3_files():
    """
    Generating Index files from a given path and glob pattern for DL3 files
    """
    from lstchain.tools.lstchain_create_dl3_index_files import FITSIndexWriter

    assert (
        run_tool(
            FITSIndexWriter(),
            argv=[
            f"--input_dl3_dir={test_dir}",
            f"--file_pattern=dl3*fits"
            ]
            == 0
        )

    )

    assert (
        run_tool(
            FITSIndexWriter(),
            argv=["--help-all"]
        )
        == 0
    )
