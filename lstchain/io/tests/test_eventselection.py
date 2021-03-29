import pytest
from lstchain.io import EventSelector, DataBinning


def test_event_selection():
    import pandas as pd
    import numpy as np
    from astropy.table import Table
    import astropy.units as u

    evt_fil = EventSelector()

    data_t = Table(
        {
            "a": u.Quantity([1, 2, 3], unit=u.kg),
            "b": u.Quantity([np.nan, 2.2, 3.2], unit=u.m),
            "c": u.Quantity([1, 3, np.inf], unit=u.s),
        }
    )

    evt_fil.filters = dict(a=[0, 2.5], b=[0, 3], c=[0, 4])
    evt_fil.finite_params = ["b"]

    data_t = evt_fil.filter_cut(data_t)
    data_t_df = evt_fil.filter_cut(data_t.to_pandas())

    np.testing.assert_array_equal(
        data_t_df, pd.DataFrame({"a": [2], "b": [2.2], "c": [3]})
    )

    np.testing.assert_array_equal(
        data_t,
        Table(
            {
                "a": u.Quantity([2], unit=u.kg),
                "b": u.Quantity([2.2], unit=u.m),
                "c": u.Quantity([3], unit=u.s),
            }
        ),
    )


def test_data_binning():
    tempbin = DataBinning()

    tempbin.true_energy_min = 0.01
    tempbin.true_energy_max = 100
    tempbin.true_energy_n_bins_per_decade = 5.5
    tempbin.reco_energy_min = 0.01
    tempbin.reco_energy_max = 100
    tempbin.reco_energy_n_bins_per_decade = 5.5
    tempbin.energy_migration_min = 0.2
    tempbin.energy_migration_max = 5
    tempbin.energy_migration_n_bins = 15
    tempbin.fov_offset_min = 0.1
    tempbin.fov_offset_max = 1.1
    tempbin.fov_offset_n_edges = 9
    tempbin.bkg_fov_offset_min = 0
    tempbin.bkg_fov_offset_max = 10
    tempbin.bkg_fov_offset_n_edges = 11
    tempbin.source_offset_min = 0
    tempbin.source_offset_max = 1.0001
    tempbin.source_offset_n_edges = 1001

    e_true = tempbin.true_energy_bins()
    e_reco = tempbin.reco_energy_bins()
    e_migra = tempbin.energy_migration_bins()
    fov_off = tempbin.fov_offset_bins()
    bkg_fov = tempbin.bkg_fov_offset_bins()
    src_off = tempbin.source_offset_bins()

    assert len(e_true) == 22
    assert len(e_reco) == 22
    assert len(e_migra) == 15
    assert len(fov_off) == 9
    assert len(bkg_fov) == 11
    assert len(src_off) == 1001
