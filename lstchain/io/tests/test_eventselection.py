import pytest
from lstchain.io import EventSelector, DL3FixedCuts, DataBinning
import numpy as np
import pandas as pd
from astropy.table import QTable
import astropy.units as u


def test_event_selection():
    evt_fil = EventSelector()

    data_t = QTable(
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
        QTable(
            {
                "a": u.Quantity([2], unit=u.kg),
                "b": u.Quantity([2.2], unit=u.m),
                "c": u.Quantity([3], unit=u.s),
            }
        ),
    )


def test_dl3_fixed_cuts():
    temp_cuts = DL3FixedCuts()

    temp_cuts.fixed_gh_cut = 0.7
    temp_cuts.fixed_theta_cut = 0.2
    temp_cuts.allowed_tels = [1, 2]

    temp_data = QTable({
        "gh_score": u.Quantity(np.arange(0.1, 1.1, 0.1)),
        "theta": u.Quantity(np.arange(0., 1., 0.1), unit=u.deg),
        "tel_id": u.Quantity([1, 1, 2, 2, 1, 2, 1, 3, 4, 5])
        })

    assert len(temp_cuts.gh_cut(temp_data)) == 4
    assert len(temp_cuts.theta_cut(temp_data)) == 2
    assert len(temp_cuts.allowed_tels_filter(temp_data)) == 7


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
