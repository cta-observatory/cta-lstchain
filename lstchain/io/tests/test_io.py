import tempfile

import numpy as np
import pandas as pd
import pytest
import tables
from astropy.table import Table
from ctapipe.instrument import SubarrayDescription


@pytest.fixture
def merged_h5file(tmp_path, simulated_dl1_file):
    """Produce a merged h5 file from simulated dl1 files."""
    from lstchain.io.io import auto_merge_h5files

    subarray_before = SubarrayDescription.from_hdf(simulated_dl1_file)

    merged_dl1_file = tmp_path / "dl1_merged.h5"
    auto_merge_h5files(
        [simulated_dl1_file, simulated_dl1_file], output_filename=merged_dl1_file
    )

    subarray_merged = SubarrayDescription.from_hdf(merged_dl1_file)

    # check that subarray name is correctly retained
    assert subarray_before.name == subarray_merged.name
    return merged_dl1_file


def test_write_dataframe():
    from lstchain.io import config, global_metadata
    from lstchain.io.io import write_dataframe

    df = pd.DataFrame(
        {
            "x": np.random.normal(size=10),
            "N": np.random.poisson(5, size=10),
        }
    )
    config = config.get_standard_config()

    with tempfile.NamedTemporaryFile() as f:
        meta = global_metadata()
        write_dataframe(df, f.name, "data/awesome_table", config=config, meta=meta)

        with tables.open_file(f.name) as h5_file:
            # make sure nothing else in this group
            # (e.g. like pandas writes _i_ tables)
            assert h5_file.root.data._v_children.keys() == {"awesome_table"}

            table = h5_file.root.data.awesome_table[:]
            for col in df.columns:
                np.testing.assert_array_equal(table[col], df[col])

            # test global metadata and config are properly written
            for k in meta.keys():
                assert meta[k] == h5_file.root.data.awesome_table.attrs[k]
            assert config == h5_file.root.data.awesome_table.attrs["config"]

        # test it's also readable by pandas directly
        df_read = pd.read_hdf(f.name, "data/awesome_table")
        assert df.equals(df_read)

        # and with astropy
        t = Table.read(f.name, "data/awesome_table")
        for col in df.columns:
            np.testing.assert_array_equal(t[col], df[col])


def test_write_dataframe_index():
    """Test that also an index can be written."""
    from lstchain.io.io import write_dataframe

    df = pd.DataFrame(
        {
            "x": np.random.normal(size=10),
            "N": np.random.poisson(5, size=10),
        }
    )
    df.index.name = "event_id"

    with tempfile.NamedTemporaryFile() as f:
        write_dataframe(df, f.name, "data/awesome_table", index=True)

        with tables.open_file(f.name) as file:
            table = file.root.data.awesome_table[:]
            for col in df.columns:
                np.testing.assert_array_equal(table[col], df[col])

            np.testing.assert_array_equal(table["event_id"], df.index)


def test_write_dl2_dataframe(tmp_path, simulated_dl2_file):
    from lstchain.io.io import dl2_params_lstcam_key
    from lstchain.io import write_dl2_dataframe

    dl2 = pd.read_hdf(simulated_dl2_file, key=dl2_params_lstcam_key)
    write_dl2_dataframe(dl2, tmp_path / "dl2_test.h5")


def test_merging_check(simulated_dl1_file):
    from lstchain.io.io import merging_check

    # the same file should be mergeable with itself
    dl1_file = simulated_dl1_file

    assert merging_check([dl1_file, dl1_file]) == [dl1_file, dl1_file]


def test_merge_h5files(merged_h5file):
    assert merged_h5file.is_file()

    # check source filenames is properly written
    with tables.open_file(merged_h5file) as file:
        assert len(file.root.source_filenames.filenames) == 2


def test_read_simu_info_hdf5(simulated_dl1_file):
    from lstchain.io.io import read_simu_info_hdf5

    mcheader = read_simu_info_hdf5(simulated_dl1_file)
    # simtel verion of the mc_gamma_testfile defined in test_lstchain
    assert mcheader.simtel_version == 1462392225
    assert mcheader.num_showers == 20000


def test_read_simu_info_merged_hdf5(merged_h5file):
    from lstchain.io.io import read_simu_info_merged_hdf5

    mcheader = read_simu_info_merged_hdf5(merged_h5file)
    # simtel verion of the mc_gamma_testfile defined in test_lstchain
    assert mcheader.simtel_version == 1462392225
    assert mcheader.num_showers == 40000


def test_trigger_type_in_dl1_params(simulated_dl1_file):
    from lstchain.io.io import dl1_params_lstcam_key

    params = pd.read_hdf(simulated_dl1_file, key=dl1_params_lstcam_key)
    assert "trigger_type" in params.columns


def test_extract_simulation_nsb(mc_gamma_testfile):
    from lstchain.io.io import extract_simulation_nsb
    nsb = extract_simulation_nsb(mc_gamma_testfile)
    assert np.isclose(nsb[0], 0.317, rtol=0.1)
    assert np.isclose(nsb[1], 0.276, rtol=0.1)
