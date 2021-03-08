import pytest
from lstchain.io import DataSelection, DataBinning


class temp_sel(DataSelection):
    DataSelection.intensity = [0, 10000]
    DataSelection.length = [0, 100]
    DataSelection.width = [0, 100]
    DataSelection.r = [0, 1]
    DataSelection.wl = [0, 1]
    DataSelection.leakage_intensity_width_2 = [0, 1]
    DataSelection.fixed_gh_cut = 0.5
    DataSelection.fixed_theta_cut = 1
    DataSelection.fixed_source_fov_offset_cut = 2
    DataSelection.src_dep_alpha = 10
    DataSelection.lst_tel_ids = [1]
    DataSelection.magic_tel_ids = [1, 2]


class temp_bin(DataBinning):
    DataBinning.true_energy_bins = [0.1, 100, 5]
    DataBinning.reco_energy_bins = [0.1, 100, 5]
    DataBinning.energy_migra_bins = [0.1, 100, 5]
    DataBinning.single_fov_offset_bins = [0.1, 0.2, 0.3]
    DataBinning.multiple_fov_offset_bins = [0.1, 0.2, 0.3, 0.4]
    DataBinning.bkg_fov_offset_bins = [1, 10]
    DataBinning.source_offset_bins = [0.1, 1., 0.05]


def test_data_selection():
    tempsel = temp_sel()

    evt = tempsel.event_filters()
    assert 'r' in evt
    

def test_data_binning():
    tempbin = temp_bin()

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
