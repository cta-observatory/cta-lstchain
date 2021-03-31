import pytest
from traitlets.config.loader import JSONFileConfigLoader


def test_event_selection(simulated_dl2_file):
    from lstchain.io import EventSelector
    from lstchain.io import read_mc_dl2_to_QTable
    import numpy as np

    config = JSONFileConfigLoader(
        'event_selection_config.json', 'docs/examples/'
    ).load_config()
    evt_fil = EventSelector(config=config)

    data_t, _ = read_mc_dl2_to_QTable(simulated_dl2_file)

    evt_fil.finite_params = ["intensity", "width", "length"]

    data_t = evt_fil.filter_cut(data_t)
    data_t_df = evt_fil.filter_cut(data_t.to_pandas())

    assert data_t["intensity"].min() > evt_fil.filters["intensity"][0]
    assert np.sum(~np.isfinite(data_t["intensity"])) == 0
    assert data_t_df["intensity"].min() > evt_fil.filters["intensity"][0]
    assert np.sum(~np.isfinite(data_t_df["intensity"])) == 0


def test_dl3_fixed_cuts(simulated_dl2_file):
    from lstchain.io import DL3FixedCuts
    from lstchain.io import read_mc_dl2_to_QTable
    import numpy as np

    config = JSONFileConfigLoader(
        'dl3_fixed_cuts_config.json', 'docs/examples/'
    ).load_config()
    temp_cuts = DL3FixedCuts(config=config)

    data_t, _ = read_mc_dl2_to_QTable(simulated_dl2_file)

    assert temp_cuts.gh_cut(data_t)["gh_score"].min() > temp_cuts.fixed_gh_cut
    assert np.unique(
        temp_cuts.allowed_tels_filter(data_t)["tel_id"]
    ) == temp_cuts.allowed_tels


def test_data_binning():
    from lstchain.io import DataBinning

    config = JSONFileConfigLoader(
        'data_binning_config.json', 'docs/examples/'
    ).load_config()
    tempbin = DataBinning(config=config)

    e_true = tempbin.true_energy_bins()
    e_reco = tempbin.reco_energy_bins()
    e_migra = tempbin.energy_migration_bins()
    fov_off = tempbin.fov_offset_bins()
    bkg_fov = tempbin.bkg_fov_offset_bins()
    src_off = tempbin.source_offset_bins()

    assert len(e_true) == 22
    assert len(e_reco) == 22
    assert len(e_migra) == 31
    assert len(fov_off) == 9
    assert len(bkg_fov) == 21
    assert len(src_off) == 1000
