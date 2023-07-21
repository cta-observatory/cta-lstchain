from lstchain.io import EventSelector, DL3Cuts, DataBinning
import numpy as np
import pandas as pd
from astropy.table import QTable
import astropy.units as u


def test_event_selection():
    evt_fil = EventSelector()

    data_t = QTable(
        {
            "a": u.Quantity([1, 2, 3, 4], unit=u.kg),
            "b": u.Quantity([np.nan, 2.2, 3.2, 5], unit=u.m),
            "c": u.Quantity([1, 3, np.inf, 4], unit=u.s),
        }
    )

    evt_fil.filters = dict(a=[0, 2.5], b=[0, 3], c=[0, 3.5])
    evt_fil.finite_params = ["b"]

    data_t2 = evt_fil.filter_cut(data_t)
    data_t_df = evt_fil.filter_cut(data_t2.to_pandas())

    np.testing.assert_array_equal(
        data_t_df,
        pd.DataFrame(
            {
                "a": [2],
                "b": [2.2],
                "c": [3],
            }
        ),
    )

    np.testing.assert_array_equal(
        data_t2,
        QTable(
            {
                "a": u.Quantity([2], unit=u.kg),
                "b": u.Quantity([2.2], unit=u.m),
                "c": u.Quantity([3], unit=u.s),
            }
        ),
    )


def test_dl3_global_cuts():
    temp_cuts = DL3Cuts()

    temp_cuts.global_gh_cut = 0.7
    temp_cuts.global_theta_cut = 0.2
    temp_cuts.global_alpha_cut = 20
    temp_cuts.allowed_tels = [1, 2]

    temp_data = QTable(
        {
            "gh_score": u.Quantity(np.tile(np.arange(0.35, 0.85, 0.05), 3)),
            "reco_energy": np.geomspace(50 * u.GeV, 50 * u.TeV, 30),
            "theta": u.Quantity(np.tile(np.arange(0.05, 0.35, 0.03), 3), unit=u.deg),
            "alpha": u.Quantity(np.tile(np.arange(5, 85, 8), 3), unit=u.deg),
            "tel_id": u.Quantity(np.repeat([1, 2, 3], 10)),
        }
    )

    assert len(temp_cuts.apply_global_gh_cut(temp_data)) == 6
    assert len(temp_cuts.apply_global_theta_cut(temp_data)) == 15
    assert len(temp_cuts.apply_global_alpha_cut(temp_data)) == 6
    assert len(temp_cuts.allowed_tels_filter(temp_data)) == 20


def test_dl3_energy_dependent_cuts():
    temp_cuts = DL3Cuts()

    temp_cuts.gh_max_efficiency = 0.8
    temp_cuts.theta_containment = 0.68
    temp_cuts.alpha_containment = 0.68
    temp_cuts.max_alpha_cut = 20
    temp_cuts.fill_alpha_cut = 20
    temp_cuts.min_event_p_en_bin = 4

    temp_data = QTable(
        {
            "gh_score": u.Quantity(np.tile(np.arange(0.35, 0.85, 0.05), 4)),
            "reco_energy": np.geomspace(50 * u.GeV, 50 * u.TeV, 40),
            "theta": u.Quantity(np.tile(np.arange(0.05, 0.35, 0.03), 4), unit=u.deg),
            "alpha": u.Quantity(np.tile(np.arange(3, 25, 5), 8), unit=u.deg),
            "reco_disp_sign": u.Quantity(np.tile([1, 1, -1, 1, 1], 8)),
            "disp_sign": u.Quantity(np.tile([-1, 1, -1, 1, 1], 8)),
        }
    )
    en_range = u.Quantity([0.01, 0.1, 1, 10, 100], unit=u.TeV)

    theta_cut = temp_cuts.energy_dependent_theta_cuts(
        temp_data,
        en_range,
        use_same_disp_sign=True,
    )
    theta_cut_2 = temp_cuts.energy_dependent_theta_cuts(
        temp_data,
        en_range,
        use_same_disp_sign=False,
    )
    gh_cut = temp_cuts.energy_dependent_gh_cuts(temp_data, en_range)
    alpha_cut = temp_cuts.energy_dependent_alpha_cuts(temp_data, en_range)

    data_th = temp_cuts.apply_energy_dependent_theta_cuts(temp_data, theta_cut)
    data_th_2 = temp_cuts.apply_energy_dependent_theta_cuts(temp_data, theta_cut_2)
    data_gh = temp_cuts.apply_energy_dependent_gh_cuts(temp_data, gh_cut)
    data_al = temp_cuts.apply_energy_dependent_alpha_cuts(temp_data, alpha_cut)

    assert theta_cut["cut"][-1] == 0.2528 * u.deg
    assert theta_cut_2["cut"][-1] == 0.2336 * u.deg
    assert gh_cut["cut"][1] == 0.38
    assert alpha_cut["cut"][0] == 13.2 * u.deg
    assert len(data_th) == 30
    assert len(data_th_2) == 29
    assert len(data_gh) == 36
    assert len(data_al) == 31


def test_update_fill_cut():
    temp_cuts = DL3Cuts()

    temp_cuts.min_event_p_en_bin = 5

    temp_cut_table_1 = QTable(
        {
            "n_events": u.Quantity(np.array([3, 10, 15, 4])),
            "cut": u.Quantity(np.array([0.4, 0.07, 0.1, 0.4]) * u.m),
        }
    )
    temp_cut_table_2 = QTable(
        {
            "n_events": u.Quantity(np.array([13, 10, 15, 4])),
            "cut": u.Quantity(np.array([0.04, 0.07, 0.1, 0.4]) * u.s),
        }
    )
    temp_cut_table_3 = QTable(
        {
            "n_events": u.Quantity(np.array([3, 10, 15, 14])),
            "cut": u.Quantity(np.array([0.4, 0.07, 0.1, 0.04])),
        }
    )
    temp_cut_table_4 = QTable(
        {
            "n_events": u.Quantity(np.array([3, 4, 10, 3, 15, 14, 4, 2])),
            "cut": u.Quantity(np.array(
                [0.4, 0.4, 0.07, 0.4, 0.1, 0.04, 0.4, 0.4]
            )),
        }
    )


    cut_table_new_1 = temp_cuts.update_fill_cuts(temp_cut_table_1)
    cut_table_new_2 = temp_cuts.update_fill_cuts(temp_cut_table_2)
    cut_table_new_3 = temp_cuts.update_fill_cuts(temp_cut_table_3)
    cut_table_new_4 = temp_cuts.update_fill_cuts(temp_cut_table_4)

    assert cut_table_new_1["cut"][0] == 0.07 * u.m
    assert cut_table_new_1["cut"][-1] == 0.1 * u.m
    assert cut_table_new_2["cut"][-1] == 0.1 * u.s
    assert cut_table_new_3["cut"][0] == 0.07
    assert cut_table_new_4["cut"][3] == 0.085


def test_data_binning():
    tempbin = DataBinning()

    tempbin.true_energy_min = 0.01
    tempbin.true_energy_max = 100
    tempbin.true_energy_n_bins = 20
    tempbin.reco_energy_min = 0.01
    tempbin.reco_energy_max = 100
    tempbin.reco_energy_n_bins = 20
    tempbin.energy_migration_min = 0.2
    tempbin.energy_migration_max = 5
    tempbin.energy_migration_n_bins = 14
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

    assert len(e_true) == 21
    assert len(e_reco) == 21
    assert len(e_migra) == 15
    assert len(fov_off) == 9
    assert len(bkg_fov) == 11
    assert len(src_off) == 1001
