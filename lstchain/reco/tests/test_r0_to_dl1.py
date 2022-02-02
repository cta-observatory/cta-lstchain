from ctapipe.containers import ArrayEventContainer
import numpy as np
from lstchain.reco.r0_to_dl1 import rescale_dl1_charge
from copy import copy


def test_rescale_dl1_charge():
    event = ArrayEventContainer()
    tel_ids = [1, 3]
    images = {}
    for tel_id in tel_ids:
        images[tel_id] = np.random.rand(1855)
        event.dl1.tel[tel_id].image = copy(images[tel_id])

    rescaling_factor = np.random.rand() * 10
    rescale_dl1_charge(event, rescaling_factor)

    for tel_id in tel_ids:
        np.testing.assert_allclose(event.dl1.tel[tel_id].image, images[tel_id]*rescaling_factor)


def test_r0_to_dl1_nsb_tuning(tmp_path, mc_gamma_testfile):
    from lstchain.reco.r0_to_dl1 import r0_to_dl1
    from lstchain.io import standard_config
    import os
    config = standard_config
    config['source_config']['EventSource']['allowed_tels'] = [1]
    config['waveform_nsb_tuning']['nsb_tuning'] = True
    config['waveform_nsb_tuning']['spe_location'] = os.path.join(os.path.dirname(__file__),
                                                                 '../../data/SinglePhE_ResponseInPhE_expo2Gaus.dat')
    r0_to_dl1(mc_gamma_testfile, custom_config=config, output_filename=tmp_path / "tmp.h5")


def test_r0_to_dl1_lhfit(tmp_path, mc_gamma_testfile):
    from lstchain.reco.r0_to_dl1 import r0_to_dl1
    from lstchain.io import standard_config
    import os
    config = standard_config
    config['source_config']['EventSource']['max_events'] = 5
    config['source_config']['EventSource']['allowed_tels'] = [1]
    config['lh_fit_config'] = {
                               "sigma_s": 0.3282,
                               "crosstalk": 0.0,
                               "ncall": 2000,
                               "sigma_space": 3,
                               "sigma_time": 4,
                               "time_before_shower": 0,
                               "time_after_shower": 20,
                               "n_peaks": 50,
                               "no_asymmetry": False,
                               "use_weight": False,
                               "verbose": 4
                              }
    os.makedirs('./event', exist_ok=True)
    r0_to_dl1(mc_gamma_testfile, custom_config=config, output_filename=tmp_path / "tmp.h5")
    assert len(os.listdir('./event')) > 1
    for path in os.listdir('./event'):
        os.remove('./event/'+path)
    os.rmdir('./event')
