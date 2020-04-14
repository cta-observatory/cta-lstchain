from lstchain.io import io
import numpy as np
import pandas as pd
import tables
import tempfile
import pytest
import os

from lstchain.tests.test_lstchain import dl1_file, test_dir

merged_dl1_file = os.path.join(test_dir, 'dl1_merged.h5')

def test_write_dataframe():
    a = np.ones(3)
    df = pd.DataFrame(a, columns=['a'])
    with tempfile.NamedTemporaryFile() as f:
        io.write_dataframe(df, f.name, 'data/awesome_table')
        with tables.open_file(f.name) as file:
            np.testing.assert_array_equal(file.root.data.awesome_table[:]['a'], a)

@pytest.mark.run(after='test_apply_models')
def test_write_dl2_dataframe():
    from lstchain.tests.test_lstchain import dl2_file, test_dir
    from lstchain.io.io import dl2_params_lstcam_key
    dl2 = pd.read_hdf(dl2_file, key=dl2_params_lstcam_key)
    from lstchain.io import write_dl2_dataframe
    write_dl2_dataframe(dl2, os.path.join(test_dir, 'dl2_test.h5'))

@pytest.mark.run(after='test_r0_to_dl1')
def test_merging_check():
    # the same file should be mergeable with itself
    [dl1_file, dl1_file] == io.merging_check([dl1_file, dl1_file])


@pytest.mark.run(after='test_r0_to_dl1')
def test_smart_merge_h5files():
    io.smart_merge_h5files([dl1_file, dl1_file], output_filename=merged_dl1_file)
    assert os.path.exists(merged_dl1_file)


@pytest.mark.run(after='test_r0_to_dl1')
def test_read_simu_info_hdf5():
    mcheader = io.read_simu_info_hdf5(dl1_file)
    assert mcheader.simtel_version == 1462392225  # simtel verion of the mc_gamma_testfile defined in test_lstchain
    assert mcheader.num_showers == 20000


@pytest.mark.run(after='test_smart_merge_h5files')
def test_read_simu_info_merged_hdf5():
    mcheader = io.read_simu_info_merged_hdf5(merged_dl1_file)
    assert mcheader.simtel_version == 1462392225  # simtel verion of the mc_gamma_testfile defined in test_lstchain
    assert mcheader.num_showers == 40000

@pytest.mark.run(after='test_r0_to_dl1')
def test_trigger_type_in_dl1_params():
    from lstchain.io.io import dl1_params_lstcam_key
    params = pd.read_hdf(dl1_file, key=dl1_params_lstcam_key)
    assert 'trigger_type' in params.columns

