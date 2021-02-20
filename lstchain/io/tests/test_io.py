import tempfile

import numpy as np
import pandas as pd
import pytest
import tables
from astropy.table import Table


@pytest.fixture
def merged_h5file(tmp_path, simulated_dl1_file):
    """Produce a smart merged h5 file from simulated dl1 files."""
    from lstchain.io.io import smart_merge_h5files
    merged_dl1_file = tmp_path / 'dl1_merged.h5'
    smart_merge_h5files(
        [simulated_dl1_file, simulated_dl1_file],
        output_filename=merged_dl1_file
    )
    return merged_dl1_file


def test_write_dataframe():
    from lstchain.io.io import write_dataframe
    df = pd.DataFrame({
        'x': np.random.normal(size=10),
        'N': np.random.poisson(5, size=10),
    })

    with tempfile.NamedTemporaryFile() as f:
        write_dataframe(df, f.name, 'data/awesome_table')

        with tables.open_file(f.name) as h5_file:
            # make sure nothing else in this group
            # (e.g. like pandas writes _i_ tables)
            assert h5_file.root.data._v_children.keys() == {'awesome_table'}

            table = h5_file.root.data.awesome_table[:]
            for col in df.columns:
                np.testing.assert_array_equal(table[col], df[col])

        # test it's also readable by pandas directly
        df_read = pd.read_hdf(f.name, 'data/awesome_table')
        assert df.equals(df_read)

        # and with astropy
        t = Table.read(f.name, 'data/awesome_table')
        for col in df.columns:
            np.testing.assert_array_equal(t[col], df[col])


def test_write_dataframe_index():
    """Test that also an index can be written."""
    from lstchain.io.io import write_dataframe
    df = pd.DataFrame({
        'x': np.random.normal(size=10),
        'N': np.random.poisson(5, size=10),
    })
    df.index.name = 'event_id'

    with tempfile.NamedTemporaryFile() as f:
        write_dataframe(df, f.name, 'data/awesome_table', index=True)

        with tables.open_file(f.name) as file:
            table = file.root.data.awesome_table[:]
            for col in df.columns:
                np.testing.assert_array_equal(table[col], df[col])

            np.testing.assert_array_equal(table['event_id'], df.index)


@pytest.mark.run(after='test_apply_models')
def test_write_dl2_dataframe(tmp_path, simulated_dl2_file):
    from lstchain.io.io import dl2_params_lstcam_key
    from lstchain.io import write_dl2_dataframe

    dl2 = pd.read_hdf(simulated_dl2_file, key=dl2_params_lstcam_key)
    write_dl2_dataframe(dl2, tmp_path / 'dl2_test.h5')


@pytest.mark.run(after='test_r0_to_dl1')
def test_merging_check(simulated_dl1_file):
    from lstchain.io.io import merging_check
    # the same file should be mergeable with itself
    dl1_file = simulated_dl1_file

    assert merging_check([dl1_file, dl1_file]) == [dl1_file, dl1_file]


@pytest.mark.run(after='test_r0_to_dl1')
def test_smart_merge_h5files(merged_h5file):
    assert merged_h5file.is_file()


@pytest.mark.run(after='test_r0_to_dl1')
def test_read_simu_info_hdf5(simulated_dl1_file):
    from lstchain.io.io import read_simu_info_hdf5
    mcheader = read_simu_info_hdf5(simulated_dl1_file)
    # simtel verion of the mc_gamma_testfile defined in test_lstchain
    assert mcheader.simtel_version == 1462392225
    assert mcheader.num_showers == 20000


@pytest.mark.run(after='test_smart_merge_h5files')
def test_read_simu_info_merged_hdf5(merged_h5file):
    from lstchain.io.io import read_simu_info_merged_hdf5
    mcheader = read_simu_info_merged_hdf5(merged_h5file)
    # simtel verion of the mc_gamma_testfile defined in test_lstchain
    assert mcheader.simtel_version == 1462392225
    assert mcheader.num_showers == 40000


@pytest.mark.run(after='test_r0_to_dl1')
def test_trigger_type_in_dl1_params(simulated_dl1_file):
    from lstchain.io.io import dl1_params_lstcam_key
    params = pd.read_hdf(simulated_dl1_file, key=dl1_params_lstcam_key)
    assert 'trigger_type' in params.columns


@pytest.mark.run(after='test_r0_to_dl1')
def test_read_subarray_description(mc_gamma_testfile, simulated_dl1_file):
    from lstchain.io.io import read_subarray_description
    from ctapipe.io import EventSource
    source = EventSource(mc_gamma_testfile)
    dl1_subarray = read_subarray_description(simulated_dl1_file)
    dl1_subarray.peek()
    dl1_subarray.info()
    assert len(dl1_subarray.tels) == len(source.subarray.tels)
    assert (dl1_subarray.to_table() == source.subarray.to_table()).all()
