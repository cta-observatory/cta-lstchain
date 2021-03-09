import pytest
from lstchain.io import DataSelection, DataBinning


def test_data_selection(simulated_dl2_file):
    from lstchain.io import read_mc_dl2_to_pyirf

    tempsel = DataSelection()
    tempsel.intensity = [0, 1000]
    tempsel.length = [0, 1]
    tempsel.width = [0, 1]
    tempsel.r = [0, 1]
    tempsel.wl = [0, 1]
    tempsel.leakage_intensity_width_2 = [0, 1]
    tempsel.fixed_gh_cut = 0.5
    tempsel.fixed_theta_cut = 1
    tempsel.fixed_source_fov_offset_cut = 2
    tempsel.src_dep_alpha = 10
    tempsel.lst_tel_ids = [1]
    tempsel.magic_tel_ids = [1, 2]

    data, _ = read_mc_dl2_to_pyirf(simulated_dl2_file)

    data_filter = tempsel.filter_cut(data)
    data_gh = tempsel.gh_cut(data)
    data_tel = tempsel.tel_ids_filter(data)

    assert data_filter["intensity"].max() < 10000
    assert data_gh["gh_score"].max() > 0.5
    assert data_tel["tel_id"].mean() == 1


def test_data_binning():
    tempbin = DataBinning()
    tempbin.true_energy_bins = [0.1, 100, 5]
    tempbin.reco_energy_bins = [0.1, 100, 5]
    tempbin.energy_migra_bins = [0.1, 100, 5]
    tempbin.single_fov_offset_bins = [0.1, 0.2, 0.3]
    tempbin.multiple_fov_offset_bins = [0.1, 0.2, 0.3, 0.4]
    tempbin.bkg_fov_offset_bins = [1, 10]
    tempbin.source_offset_bins = [0.1, 1., 0.05]

    e_true = tempbin.true_energy()
    e_reco = tempbin.reco_energy()
    e_migra = tempbin.energy_migration()
    src_off = tempbin.source_offset()
    bkg_off = tempbin.background_offset()

    assert len(e_true) == 15
    assert len(e_reco) == 15
    assert len(e_migra) == 5
    assert len(src_off) == 18
    assert len(bkg_off) == 9
